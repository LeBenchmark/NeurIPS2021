# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union
from collections import OrderedDict
from fairseq.file_io import PathManager
from fairseq import options, utils, checkpoint_utils, init_functions
from fairseq.modules import LayerNorm
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.lstm import (
    LSTMEncoder, 
    LSTMDecoder,
)

from fairseq.models.TarcTransformerModules import TarcTransformerEncoder

from fairseq.modules import AdaptiveSoftmax

_DEBUG_ = False

DEFAULT_MAX_SOURCE_POSITIONS = 2048
DEFAULT_MAX_TARGET_POSITIONS = 2048

char_reduce = torch.sum

def load_pretrained_component_from_multilayermodel(
    component: Union[FairseqEncoder, FairseqDecoder], checkpoint: str
):
    """
    Like the 'load_pretrained_component_from_model' function in checkpoint_utils, but specialized for the TatcMultiTask encoder and decoder
    Load a pretrained FairseqEncoder or FairseqDecoder from checkpoint into the
    provided `component` object. If state_dict fails to load, there may be a
    mismatch in the architecture of the corresponding `component` found in the
    `checkpoint` file.
    """
    if not PathManager.exists(checkpoint):
        raise IOError("Model file not found: {}".format(checkpoint))
    state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint)
    if isinstance(component, FairseqEncoder):
        component_type = "encoder.encoders"
    elif isinstance(component, FairseqDecoder):
        component_type = "decoder.decoders"
    else:
        raise ValueError(
            "component to load must be either a FairseqEncoder or "
            "FairseqDecoder. Loading other component types are not supported."
        )
    scratch_state_dict = component.state_dict()
    component_state_dict = OrderedDict()
    for key in state["model"].keys():
        if key.startswith(component_type):
            # e.g. encoders.0.input_layers.0.0.weight --> input_layers.0.0.weight
            component_subkey = key[len(component_type) + 3 :]

            #print(' - Param {}: {}'.format(key, state["model"][key].size()))
            #print('   - vs. {}: {}'.format(component_subkey, scratch_state_dict[component_subkey].size()))
            #sys.stdout.flush()

            component_state_dict[component_subkey] = state["model"][key]
    component.load_state_dict(component_state_dict, strict=True)
    return component

def char2token_features_(src, dims, bds, tk_sort, offset):
    
    (T, B, C) = dims
    total_offset = 2*offset if offset == 1 else offset
    bounds_offset = 0 if offset == 1 or offset == 0 else 1

    res = torch.zeros(T-total_offset, B, C).to(src)
    for i in range(B):
        bounds = bds[i]+bounds_offset 
        res[:bounds.size(0),i,:] = torch.stack( [char_reduce(src[bounds[bi,0]:bounds[bi,1],i,:], 0) for bi in range(bounds.size(0))] , 0 )

    return res

def get_chars_from_tokens_(tokens, t2c_map, mdict):

    (B, T) = tokens.size()
    assert T == 1
    tmp = []
    for bi in range(B): 
        t = tokens[bi,0].item()
        if t in t2c_map:
            tmp.append( t2c_map[t].to(tokens) ) # = [t2c_map[t].to(tokens)]
        else:
            sys.stderr.write(' *** get_chars_from_tokens WARNING: predicted token {} ({}) is not defined in current map, backing off to <bos>\n'.format(t, mdict.string(tokens[bi,:]))) 
            tmp.append( t2c_map[mdict.bos()].to(tokens) ) #= [t2c_map[mdict.bos()].to(tokens)]
    max_len = max( [t.size(0) for t in tmp if len(t.size()) > 0] )
    if max_len == 0:
        max_len = 1
    res = torch.LongTensor(B, max_len).fill_(mdict.pad()).to(tokens)
    for bi in range(B):
        bound = tmp[bi].size(0) if len(tmp[bi].size()) > 0 else 1
        res[bi,:bound] = tmp[bi]
    return res

class TarcLSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(
        self, args, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=True,
        left_pad=False, pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
        token_map=None,
        granularity_flags=None
    ):
        super().__init__(dictionary)
        self.args = args
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.max_source_positions = max_source_positions
        self.token2components_map = token_map
        self.token_sequences = granularity_flags[0] if granularity_flags is not None else False
        self.char_sequences = granularity_flags[1] if granularity_flags is not None else False
        self.merge_flag = False
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()

        self.embedder = None
        if hasattr(args, 'use_transformer_layers') and args.use_transformer_layers:
            tmp = args.encoder_layers
            args.encoder_layers = args.transformer_layers
            trans_embed = init_functions.TransformerEmbedding(len(dictionary), embed_dim, self.padding_idx)
            self.embedder = TarcTransformerEncoder(args,
                    dictionary,
                    trans_embed,
                    token_map=token_map,
                    granularity_flags=granularity_flags
            )

            if args.load_transformer_layers:
                self.load_transformer_layers( args.load_transformer_layers )

            args.encoder_layers = tmp

        num_embeddings = len(dictionary) 
        if pretrained_embed is None:
            self.embed_tokens = init_functions.LSTMEmbedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lm = None
        if args.input_lm:
            from fairseq import tasks
            print('   *** TarcLSTMEncoder: loading language model from {}'.format(args.input_lm))
            sys.stdout.flush()

            lm_state = checkpoint_utils.load_checkpoint_to_cpu(args.input_lm)
            lm_args = lm_state["args"] 
            lm_task = tasks.setup_task(lm_args)
            lm_data = {}
            lm_data['dev'] = torch.load(args.lm_data + '.dev')
            lm_data['vocab'] = torch.load(args.lm_data + '.vocab')
            lm_data['token2components'] = torch.load(args.lm_data + '.token2components')
            lm_task.input_vocab = lm_data['vocab']
            lm_task.output_vocab = lm_data['vocab']
            lm_task.token2components_tsr = lm_data['token2components']
            lm_tensors, lm_lengths = lm_data['dev']

            assert lm_args.sub_task == 'base'
            _, gflags = tasks.TArCMultiTask.choose_column_processing(len(lm_tensors[0]), lm_args)
            assert gflags is not None
            lm_task.granularity_merging_flags = gflags[lm_args.sub_task]

            # build model for ensemble
            lm_model = lm_task.build_model(lm_args)
            lm_model.load_state_dict(lm_state["model"], strict=True, args=lm_args)
            self.lm = lm_model.eval()
            for param in self.lm.parameters():
                param.requires_grad = False

            print(' * Language model archtecture:')
            print(lm_model)
            print(' -----')
            sys.stdout.flush()

        self.lstm = init_functions.LSTM(
            input_size=2*embed_dim if self.token_sequences and self.char_sequences else embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def set_merge_flag(self, val): 
        self.merge_flag = val

    def load_transformer_layers(self, checkpoint):

        self.embedder = load_pretrained_component_from_multilayermodel(self.embedder, checkpoint)
        if self.args.freeze_transformer_layers:
            for param in self.embedder.parameters():
                param.requires_grad = False

    def _forward_lstm(self, src_tokens, src_lengths, src_tok_bounds, sort_order, lstm):

        toks_src_tokens, char_src_tokens = src_tokens
        toks_src_lengths, char_src_lengths = src_lengths
        toks_sort_order, char_sort_order = sort_order

        bsz, seqlen = toks_src_tokens.size()

        #print(' -----')
        #print(' * Encoder, first token sequence: {}'.format(self.dictionary.string(toks_src_tokens[0])))
        #print(' -----')
        #print(' * Encoder, first char sequence: {}'.format(self.dictionary.string(char_src_tokens[0])))
        #print(' -----')
        #sys.stdout.flush()

        # embed tokens
        if self.lm is not None:
            lm_out = self.lm.encoder(src_tokens, src_lengths, src_tok_bounds, sort_order)
            x = lm_out.encoder_out.detach()
            x = x.transpose(0, 1)

            #print('[DEBUG] TarcLSTMEncoder, input encoded with lm: {}, device: {}'.format(x.size(), x.device))
            #sys.stdout.flush()
            #sys.exit(0)
        else:
            if self.embedder is None:
                x = self.embed_tokens(toks_src_tokens)
            else:
                x = self.embedder(src_tokens, src_lengths, src_tok_bounds, sort_order)
                x = x.encoder_out.transpose(0, 1)

            #print('[DEBUG] TarcLSTMEncoder, input encoded with embeddings: {}, device: {}'.format(x.size(), x.device))
            #sys.stdout.flush()
            #sys.exit(0)

        x = F.dropout(x, p=self.dropout_in, training=self.training)

        #print('[DEBUG] x shape after embedding: {}'.format(x.size()))
        #sys.stdout.flush()

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        (T, B, C) = x.size() 

        if self.merge_flag and self.token_sequences and self.char_sequences:
            padding_flag = toks_src_tokens[0,0] == self.dictionary.bos()
            offset = 0
            if padding_flag:
                offset = 1

            y = self.embed_tokens(char_src_tokens)
            y = F.dropout(y, p=self.dropout_in, training=self.training)
            y = y.transpose(0, 1)

            toks_from_chars = char2token_features_(y, x.size(), src_tok_bounds, toks_sort_order, offset)
            xx = torch.zeros_like(x).copy_(x)
            xx[offset:T-1*offset,:,:] = toks_from_chars.flip(dims=[0]) if self.args.reverse_input else toks_from_chars
 
            x = torch.cat( [x, xx], -1 )

            #print('   * Encoder: concatenating char level representations')
            #sys.stdout.flush()
        elif self.token_sequences and self.char_sequences:
            x = torch.cat( [x, torch.zeros_like(x)], -1)

            #print('   * Encoder: concatenating zeros')
            #sys.stdout.flush()

        #print('[DEBUG] x shape before packing: {}'.format(x.size()))
        #print('[DEBUG] toks_src_lengths: {}'.format(toks_src_lengths.data.tolist()))
        #sys.stdout.flush()

        # pack embedded source tokens into a PackedSequence
        #packed_x = nn.utils.rnn.pack_padded_sequence(x, toks_src_lengths.data.tolist()) # TODO: solve the length missmatch problem and use packed sequences again !

        #print('[DEBUG] packed_x shape: {}'.format(packed_x))
        #sys.stdout.flush()

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        x, (final_hiddens, final_cells) = lstm(x, (h0, c0)) # TODO: see TODO above and use packed_x instead of input x and packed_outs instead of output x

        #print('[DEBUG] packed_out shape: {}'.format(packed_outs))
        #sys.stdout.flush()

        # unpack outputs and apply dropout
        #x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)

        #print('[DEBUG] final x shape: {}'.format(x.size()))
        #sys.stdout.flush()

        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units], 'Expected encoder output shape: {} x {} x {}; got {}'.format(seqlen, bsz, self.output_units, list(x.size()))

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = toks_src_tokens.eq(self.padding_idx).t()

        #print('[DEBUG] TarcLSTMEncoder _forward_lstm passed.')
        #sys.stdout.flush()

        return x, final_hiddens, final_cells, encoder_padding_mask

    def forward(self, src_tokens, src_lengths, src_tok_bounds, sort_order):

        toks_src_tokens, char_src_tokens = src_tokens
        toks_src_lengths, char_src_lengths = src_lengths
        toks_sort_order, char_sort_order = sort_order

        if self.left_pad:
            # nn.utils.rnn.pack_padded_sequence requires right-padding;
            # convert left-padding to right-padding
            toks_src_tokens = utils.convert_padding_direction(
                toks_src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

            char_src_tokens = utils.convert_padding_direction(
                char_src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        if self.token_sequences and self.char_sequences:
            #print(' ### ENCODER: using both sequences')
            #sys.stdout.flush()

            x, final_hiddens, final_cells, encoder_padding_mask = self._forward_lstm([toks_src_tokens, char_src_tokens], [toks_src_lengths, char_src_lengths], src_tok_bounds, sort_order, self.lstm)
        elif self.token_sequences:
            #print(' ### ENCODER: using token sequences')
            #sys.stdout.flush()

            x, final_hiddens, final_cells, encoder_padding_mask = self._forward_lstm([toks_src_tokens, toks_src_tokens], [toks_src_lengths, toks_src_lengths], src_tok_bounds, sort_order, self.lstm)
        elif self.char_sequences:
            #print(' ### ENCODER: using char sequences')
            #sys.stdout.flush()

            x, final_hiddens, final_cells, encoder_padding_mask = self._forward_lstm([char_src_tokens, char_src_tokens], [char_src_lengths, char_src_lengths], src_tok_bounds, sort_order, self.lstm) 

        '''print(' - TarcLSTMEncoder, output shapes:')
        print('   * x: {}'.format(x.size()))
        print('   * final_hiddens: {}'.format(final_hiddens.size()))
        print('   * final_cells: {}'.format(final_cells.size()))
        print('   * encoder_padding_mak: {}'.format(encoder_padding_mask.size()))
        print(' -----')
        sys.stdout.flush()
        sys.exit(0)''' 

        #print('[DEBUG] TarcLSTMEncoder forward passed.')
        #sys.stdout.flush()

        return {
            'encoder_out': (x, final_hiddens, final_cells),
            'sources': toks_src_tokens,
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out['encoder_out']
        )
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class TarcAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = init_functions.LSTMLinear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = init_functions.LSTMLinear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2) # srclen x bsz

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0) # bsz x source_embed_dim

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))  # bsz x output_embed_dim
        return x, attn_scores


class TarcLSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, args, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, num_attentions=1,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        token_map=None,
        granularity_flags=None,
        double_learning=False,
        input_dict=None
    ):
        super().__init__(dictionary)
        self.args = args
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.token2components_map = token_map
        self.token_sequences = granularity_flags[0] if granularity_flags is not None else False
        self.char_sequences = granularity_flags[1] if granularity_flags is not None else False
        self.g_id = 'char' if ((not self.token_sequences) and self.char_sequences) else 'token'
        self.merge_flag = False
        self.double_learning = double_learning
        self.input_dict = input_dict[0]
        self.output_dict = input_dict[1]
        self.punct_dict = input_dict[2]
        self.inverse_input_dict = input_dict[3]
        self.inverse_output_dict = input_dict[4]

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = init_functions.LSTMEmbedding(num_embeddings, embed_dim, padding_idx) 
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size * num_attentions
        total_embed_size = 2*embed_dim if self.token_sequences and self.char_sequences else embed_dim

        # TODO: new, we try to modify the decoder to perform a chain of operations similar to the one of transformer.
        '''self.lstm_input_map = init_functions.LSTMLinear(input_feed_size+total_embed_size, hidden_size, bias=False) if input_feed_size+total_embed_size != hidden_size else None
        self.lstm_input_norm = [init_functions.LSTMLayerNorm(hidden_size) for layer in range(num_layers)]
        self.attn_output_norm = init_functions.LSTMLayerNorm(hidden_size)
        self.final_norm = init_functions.LSTMLayerNorm(hidden_size)'''

        self.layers = nn.ModuleList([
            init_functions.LSTMCell(
                input_size=input_feed_size + total_embed_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])

        self.num_attentions = num_attentions
        self.attentions = nn.ModuleList()
        for i in range(num_attentions):
            # TODO make bias configurable
            query_size = hidden_size
            key_size = encoder_output_units if i == 0 else out_embed_dim
            value_size = hidden_size 
            self.attentions.append( TarcAttentionLayer(query_size, key_size, value_size, bias=False) ) 
 
        if self.double_learning or hidden_size != out_embed_dim:
            if hidden_size != out_embed_dim:
                self.tk_additional_fc = init_functions.LSTMLinear(hidden_size, out_embed_dim)
            if self.double_learning:
                self.char_rnn = init_functions.LSTMCell(input_size=hidden_size, hidden_size=hidden_size) #init_functions.LSTMLinear(embed_dim + hidden_size, hidden_size)    # input: char embed, $c_{i-1}$ (input-feed) => $h_i$
                self.char2tok_att = TarcAttentionLayer(hidden_size, hidden_size, hidden_size, bias=False)   # input: $h_i$, x => $c_i$
                if hidden_size != out_embed_dim:
                    self.char_out = init_functions.LSTMLinear(hidden_size, out_embed_dim)  # input: $c_i$ => $o_i$
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff, dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = init_functions.LSTMLinear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def set_merge_flag(self, val): 
        self.merge_flag = val

    def get_token_from_chars_(self, tokens, embed):

        (B, T) = tokens.size()
        assert T == 1, 'TarcLSTMDecoder.get_token_from_chars_ function must be used at inference time with incremental_state only'

        embs = []
        C = embed.embedding_dim
        for t in tokens:
            if t.item() in self.token2components_map:
                embs.append( F.dropout(embed(self.token2components_map[t.item()].to(embed.weight.device)), p=self.dropout_in, training=self.training) )
            else:
                sys.stderr.write( 'TarcLSTMDecoder.get_token_from_chars_, token {} ({}) is not defined in current map, backing off with bos\n'.format(t.item(), self.dictionary.string(t)) )
                embs.append( F.dropout( embed(torch.LongTensor([self.dictionary.bos()]).to(embed.weight.device)), p=self.dropout_in, training=self.training) )
        res = torch.zeros(T, B, C).to(embed.weight.device)

        for i, t in enumerate(embs):
            res[:,i,:] = char_reduce(embs[i], 0)
 
        return res

    def forward(self, prev_output_tokens, tgt_tok_bounds, sort_order, encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, tgt_tok_bounds, sort_order, encoder_out, incremental_state
        )
        hidden_states = x[0]    # NOTE: x[0] is the token-level representation on which following decoders will put attention...it could be interesting to try x[1], the character-level representation, or both...

        x_tk = self.output_layer(x[0])
        x_ch = self.output_layer(x[1]) if x[1] is not None else None

        if self.input_dict is not None:
            assert self.inverse_input_dict is not None
            assert self.inverse_output_dict is not None

            force_input_decoding_state = utils.get_incremental_state(self, incremental_state, 'cached_forced_input_decoding')
            if force_input_decoding_state is not None:
                in_idx, punct_parse = force_input_decoding_state
            else:
                in_idx = 0
                punct_parse = 0

            #print(' * Forcing input decoding (input dictionary size: {}, output dictionary size: {})'.format(len(self.input_dict), len(self.output_dict)))
            #sys.stdout.flush() 

            # TODO: add forced input decoding here
            #emask = encoder_out[0]['encoder_padding_mask'] #
            src_tokens = encoder_out[0]['sources']
            #iB, iT = src_tokens.size() #
            oB, oT, oC = x_tk.size()
            #ntokens = iB*iT if emask == None else iB*iT - torch.sum(emask).item() #

            mscores, preds = torch.max( x_tk, -1 ) 
            #s_max = 5.0 #torch.max(x_tk).item()
            #s_min = -5.0 #torch.min(x_tk).item()

            #corrections = 0 #
            #breaked = 0 #
            for b_idx in range(oB): 

                src_tokens_i = src_tokens[b_idx].flip(dims=[0]) if self.args.reverse_input else src_tokens[b_idx]
                nonpad_idx = (src_tokens_i != self.dictionary.pad()).nonzero()
                src_tokens_i = src_tokens_i[nonpad_idx]

                if incremental_state is None:
                    in_idx = 0
                    punct_parse = 0

                #print(' * Input sequence (len: {}): {}'.format(src_tokens_i.size(0), self.dictionary.string(src_tokens_i)))
                #sys.stdout.flush() 

                #show_flag = True #
                for o_idx in range(oT):
                    o = x_tk[b_idx,o_idx,:] 

                    #if in_idx < src_tokens_i.size(0):
                    #    print(' ##########')
                    #    p_tok = 'UNK?'
                    #    if preds[b_idx, o_idx].item() in self.inverse_input_dict:
                    #        p_tok = self.inverse_input_dict[preds[b_idx, o_idx].item()]
                    #    elif preds[b_idx, o_idx].item() in self.inverse_output_dict:
                    #        p_tok = self.inverse_output_dict[preds[b_idx, o_idx].item()]
                    #    s_tok = self.inverse_input_dict[src_tokens_i[in_idx].item()]
                    #    print(' ##### Predicted token: {} ({}), expected input token: {} ({})'.format(preds[b_idx, o_idx].item(), p_tok, src_tokens_i[in_idx].item(), s_tok))
                    #    sys.stdout.flush()
                    #elif show_flag:
                    #    show_flag = False
                    #    print(' ##### Input decoded, skipping corrections...')
                    #    sys.stdout.flush()

                    #if show_flag:
                    #    print('   * Correction conditions:')
                    #    print('     - {} in self.input_dict: {}'.format(preds[b_idx, o_idx].item(), preds[b_idx, o_idx].item() in self.input_dict))
                    #    print('     - <idem> not in self.output_dict: {}'.format(preds[b_idx, o_idx].item() not in self.output_dict))
                    #    if in_idx < src_tokens_i.size(0):
                    #        print('     - {} in self.output_dict: {}'.format(src_tokens_i[in_idx].item(), src_tokens_i[in_idx].item() in self.output_dict))
                    #    print(' -----')

                    if self.args.tree_format == 'GAFL':
                        correcting_condition = in_idx < src_tokens_i.size(0) and preds[b_idx, o_idx].item() in self.input_dict and ((preds[b_idx, o_idx].item() not in self.output_dict) or (src_tokens_i[in_idx].item() in self.output_dict))
                    else:
                        correcting_condition = in_idx < src_tokens_i.size(0) and preds[b_idx, o_idx].item() in self.input_dict and ((preds[b_idx, o_idx].item() not in self.output_dict) or (src_tokens_i[in_idx].item() in self.output_dict)) and not (preds[b_idx, o_idx].item() in self.output_dict and preds[b_idx, o_idx].item() not in self.punct_dict and src_tokens_i[in_idx].item() in self.punct_dict)

                    if correcting_condition:
                        if self.args.tree_format == 'GAFL' or (preds[b_idx, o_idx].item() not in self.punct_dict or punct_parse == 1):
                            #print('   * correcting input item score ({} or {})...'.format(preds[b_idx, o_idx].item() not in self.output_dict, src_tokens_i[in_idx].item() in self.output_dict))
                            #sys.stdout.flush()

                            # TODO: TEST swapping the scores of predicted token and expected source token instead of spotting the expected input token
                            #       THAT IS: comment the next line, and uncomment the following one.
                            o[:] = -1.0 * mscores[b_idx, o_idx]
                            #o[preds[b_idx, o_idx]] = o[src_tokens_i[in_idx]].detach()
                            o[src_tokens_i[in_idx]] = mscores[b_idx, o_idx]
                            x_tk[b_idx, o_idx] = o

                            #print('   *** set o[{}] to {}'.format(src_tokens_i[in_idx], o[src_tokens_i[in_idx]]))
                            #_, o_max = torch.max(o, 0)
                            #_, p_max = torch.max(x_tk[b_idx, o_idx], 0)
                            #print('   *** double check: {} vs. {} (scores: {} vs. {})'.format(o_max, p_max, x_tk[b_idx, o_idx, preds[b_idx, o_idx]], x_tk[b_idx, o_idx, src_tokens_i[in_idx]]))
                            #sys.stdout.flush()

                            in_idx += 1
                            punct_parse = 0

                        elif punct_parse == 0:
                            punct_parse = 1

                            #corrections += 1 #

                #print(' * Input sequence (len: {}): {}'.format(src_tokens_i.size(0), self.dictionary.string(src_tokens_i)))
                #_, new_preds = torch.max( x_tk[b_idx], -1 )
                #print(' * New predictions (indeces: {}): {}'.format(new_preds, self.dictionary.string(new_preds)))
                #sys.stdout.flush()

                utils.set_incremental_state(self, incremental_state, 'cached_forced_input_decoding', (in_idx, punct_parse))
             
            #print(' * Input forced deconding done ({:.2f} corrections, {} out of {}).'.format(corrections/ntokens *100.0, corrections, ntokens))
            #sys.stdout.flush()
            #sys.exit(0)

        attn_scores['hidden'] = hidden_states
        return (x_tk, x_ch), attn_scores

    def extract_features_layers(
        self, prev_output_tokens, tgt_tok_bounds, sort_order, encoder_out, layers, atts, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """ 

        toks_prev_output, char_prev_output = prev_output_tokens
        toks_sort_order, char_sort_order = sort_order
        padding_flag = toks_prev_output[0,0].item() == self.dictionary.eos()

        #print(' -----')
        #print(' * Decoder, first token sequence: {}'.format(self.dictionary.string(toks_prev_output[0])))
        #print(' -----')
        #print(' * Decoder, first char sequence: {}'.format(self.dictionary.string(char_prev_output[0])))
        #print(' -----')
        #sys.stdout.flush()

        if len( encoder_out ) != self.num_attentions:
            raise ValueError('The number of attention heads must match the number of inputs (got {} and {})'.format(len(encoder_out), self.num_attentions))

        if encoder_out[0] is not None:
            encoder_padding_mask = encoder_out[0]['encoder_padding_mask']
            encoder_outputs = encoder_out[0]['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_outputs = None 
 
        if incremental_state is not None:
            toks_prev_output = toks_prev_output[:, -1:]
            char_prev_output = get_chars_from_tokens_(toks_prev_output, self.token2components_map, self.dictionary) if (self.token_sequences and self.char_sequences) else char_prev_output[:,-1:] 

        double_signal_flag = self.double_learning and self.merge_flag and self.token_sequences and self.char_sequences 

        tk_bsz, tk_seqlen = toks_prev_output.size()
        ch_bsz, ch_seqlen = char_prev_output.size()

        bsz = tk_bsz
        seqlen = tk_seqlen #max(tk_seqlen, ch_seqlen) if double_signal_flag else tk_seqlen

        # get outputs from encoder
        if encoder_outputs is not None: 
            encoder_outs, encoder_hiddens, encoder_cells = encoder_outputs[:3]
            srclen = encoder_outs.size(0) 
        else:
            srclen = None 

        # embed tokens
        x = self.embed_tokens(toks_prev_output)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        T, B, C = x.size()

        #print('[DEBUG] TarcLSTMDecoder, predictions embedded.')
        #sys.stdout.flush()

        if self.merge_flag and self.token_sequences and self.char_sequences: 
            offset = 0
            if padding_flag and incremental_state is None:
                offset = 2
                assert toks_prev_output[0,0].item() == self.dictionary.eos()
                assert toks_prev_output[0,1].item() == self.dictionary.bos()
                assert char_prev_output[0,0].item() == self.dictionary.eos()
                assert char_prev_output[0,1].item() == self.dictionary.bos()

            if incremental_state is None:
                y = self.embed_tokens(char_prev_output)
                y = F.dropout(y, p=self.dropout_in, training=self.training)
                y = y.transpose(0, 1) 
                toks_from_chars = char2token_features_(y, [T, B, C], tgt_tok_bounds, toks_sort_order, offset)
            else:
                toks_from_chars = self.get_token_from_chars_(toks_prev_output, self.embed_tokens)
            xx = torch.zeros_like(x).copy_(x)
            xx[offset:T,:,:] = toks_from_chars.flip(dims=[0]) if self.args.reverse_input else toks_from_chars # NOTE: anything else than toks_from_chars should be padding, or bos or eos

            x = torch.cat( [x, xx], -1 )

            #print('   * Decoder: concatenating char level representations')
            #sys.stdout.flush()
        elif self.token_sequences and self.char_sequences:
            x = torch.cat( [x, torch.zeros_like(x)], -1)

            #print('   * Decoder: concatenating zeros')
            #sys.stdout.flush()

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state' + self.g_id)
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        elif encoder_outputs is not None:
            # setup recurrent cells
            num_layers = len(self.layers)

            #input_feed = x.new_zeros(bsz, self.hidden_size * self.num_attentions)
            #if len(encoder_out) > 1:
            #    prev_hiddens, prev_cells, prev_input_feed = encoder_out[-1]['encoder_out'][1]
            #    input_feed[:, :prev_input_feed.size(1)] = prev_input_feed
            #else:
            if _DEBUG_:
                print('[DEBUG] TarcLSTMDecoder, encoder_hiddens and encoder_cells shape: {}, {}'.format(encoder_hiddens[-1].size(), encoder_cells[-1].size()))
                sys.stdout.flush()

            prev_hiddens = [encoder_hiddens[-1] for i in range(num_layers)]
            prev_cells = [encoder_cells[-1] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size * self.num_attentions) 
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None

        assert srclen is not None, \
            "attention needs encoder outputs, got None"
        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        ch_outs = []
        for i in range(self.num_attentions):
            outs.append([])
            ch_outs.append([])
        tc_hiddens = []

        #print('[DEBUG] TarcLSTMDecoder starting decoding loop.')
        #sys.stdout.flush()

        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1) 
            else:
                input = x[j]

            #print('[DEBUG] TarcLSTMDecoder, lstm input computed')
            #sys.stdout.flush()

            for i, rnn in enumerate(layers):
                # recurrent cell

                if _DEBUG_:
                    print('[DEBUG] ----------')
                    print('[DEBUG] TarcLSTMDecoder, lstm input sizes:')
                    print('[DEBUG]   * input: {}'.format(input.size()))
                    print('[DEBUG]   * prev_hiddens[{}]: {}'.format(i, prev_hiddens[i].size()))
                    print('[DEBUG]   * prev_cells[{}]: {}'.format(i, prev_cells[i].size()))
                    print('[DEBUG] ***')
                    sys.stdout.flush()

                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training) 

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            #print('[DEBUG] TarcLSTMDecoder, lstm output computed.')
            #sys.stdout.flush()

            if double_signal_flag:
                tc_hiddens.append( hidden )
            # apply attention using the last layer's hidden state

            #print('[DEBUG] TarcLSTMDecoder, computing attentions')
            #sys.stdout.flush()

            attn_out = []
            for i in range(self.num_attentions):
                if i == 0:     
                    out, attn_scores[:, j, :] = atts[i](hidden, encoder_outs, encoder_padding_mask)
                else:
                    if _DEBUG_:
                        print('[DEBUG] TarcLSTMDecoder, attention tensor size: k: {}, v: {}'.format(hidden.size(), encoder_out[i]['encoder_out'][0].size()))
                        sys.stdout.flush()

                    decoder_padding_mask = None
                    #if incremental_state is None:
                    #    decoder_padding_mask = encoder_out[i]['encoder_padding_mask'].transpose(0, 1) 
                    #else:
                    #    decoder_padding_mask = encoder_out[i]['encoder_out'][0].sum(-1) == 0.0
                    out, _ = atts[i](hidden, encoder_out[i]['encoder_out'][0], decoder_padding_mask) 

                #print('[DEBUG] attention {} computed'.format(i))
                #sys.stdout.flush()

                out = F.dropout(out, p=self.dropout_out, training=self.training)
                attn_out.append( out )
 
                outs[i].append( out ) 

            #print('[DEBUG] TarcLSTMDecoder, computing input feed.')
            #sys.stdout.flush()

            # input feeding
            if input_feed is not None:
                #print('[DEBUG] -----')
                #sys.stdout.flush()
                #for a in attn_out:
                #    print('[DEBUG]     * attention output size: {}'.format(a.size()))
                #    sys.stdout.flush()
                #print('[DEBUG] -----')
                #sys.stdout.flush()

                input_feed = torch.cat(attn_out, -1) #attn_out[0]

            #print('[DEBUG] TarcLSTMDecoder, input feed computed.')
            #sys.stdout.flush()

        #print('[DEBUG] TarcLSTMDecoder decoding loop passed.')
        #sys.stdout.flush()

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state' + self.g_id,
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.zeros(tk_seqlen, bsz, self.hidden_size).to(x)
        #xx = torch.zeros(ch_seqlen, bsz, self.hidden_size).to(x) if double_signal_flag else None 
        for l in outs: 
            x = x + torch.cat(l, dim=0).view(tk_seqlen, bsz, self.hidden_size)

        #print('[DEBUG] TarcLSTMDecoder output computed.')
        #sys.stdout.flush()

        # TODO: add code here to predict char sequence
        xx = None
        if double_signal_flag:
            xx = []
            tc_hiddens = torch.stack(tc_hiddens, 0)
            ch_cached_state = utils.get_incremental_state(self, incremental_state, 'ch_cached_state')
            if ch_cached_state is not None:
                ch_feed, ch_hidden, ch_cell = ch_cached_state
            else:
                ch_feed = torch.zeros(bsz, self.hidden_size).to(x)
                ch_hidden = torch.zeros(bsz, self.hidden_size).to(x)
                ch_cell = torch.zeros(bsz, self.hidden_size).to(x)
            '''y = self.embed_tokens(char_prev_output)
            y = F.dropout(y, p=self.dropout_in, training=self.training)
            y = y.transpose(0, 1)'''
            for ch_i in range(ch_seqlen):
                # 1. TODO: test this solution (depending only on x, not on y), NEEDS TO CHANGE THE INPUT SIZE OF CHAR_RNN
                padding_mask = (toks_prev_output == self.dictionary.pad()).transpose(0, 1)
                ch_att, _ = self.char2tok_att(ch_hidden, tc_hiddens, padding_mask)
                ch_hidden, ch_cell = self.char_rnn(ch_att, (ch_hidden, ch_cell))
                ch_hidden = F.dropout(ch_hidden, p=self.dropout_out, training=self.training)
                if hasattr(self, 'char_out'):
                    ch_hidden = self.char_out(ch_hidden)
                xx.append(ch_hidden)

                # 2. Current colution
                '''input = torch.cat( [y[ch_i,:,:], ch_feed], -1)
                ch_hidden, ch_cell = self.char_rnn( input, (ch_hidden, ch_cell) ) 
                padding_mask = (toks_prev_output == self.dictionary.pad()).transpose(0, 1)
                ch_att, _ = self.char2tok_att(ch_hidden, x, padding_mask)
                ch_att = F.dropout(ch_att, p=self.dropout_out, training=self.training)
                ch_feed = ch_att
                #if hasattr(self, 'char_out'):
                ch_att = self.char_out(ch_att)
                xx.append( ch_att )'''
            utils.set_incremental_state(self, incremental_state, 'ch_cached_state', (ch_feed, ch_hidden, ch_cell))
            xx = torch.stack(xx, dim=0)

            #print(' ########## Computing character level decoding')
            #sys.stdout.flush()

        #if xx is not None:
        #    for l in ch_outs:
        #        xx = xx + torch.cat(l, dim=0).view(ch_seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        if xx is not None:
            xx = xx.transpose(1, 0)

        '''print(' - DoubleLearning debug, x shape: {}'.format(x.size()))
        if xx is not None:
            print(' - DoubleLearning debug, xx shape: {}'.format(xx.size()))
            sys.exit(0)'''

        #x = self.attn_output_norm(x)
        if hasattr(self, 'tk_additional_fc') and self.adaptive_softmax is None:
            x = self.tk_additional_fc(x)
            #x = self.final_norm(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
            #if xx is not None:
            #    xx = self.ch_additional_fc(xx)
            #    xx = F.dropout(xx, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        #return (x, xx), {'attn': attn_scores, 'final_state': (prev_hiddens, prev_cells, input_feed)}
        return (x, xx), {'attn': attn_scores}

    def extract_features(
        self, prev_output_tokens, tgt_tok_bounds, sort_order, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """ 

        toks_prev_output_tokens, char_prev_output_tokens = prev_output_tokens
        toks_sort_order, char_sort_order = sort_order

        if self.token_sequences and self.char_sequences:
            #print(' ### DECODER: using both sequences')
            #sys.stdout.flush()

            x, attn_scores = self.extract_features_layers([toks_prev_output_tokens, char_prev_output_tokens], tgt_tok_bounds, sort_order, encoder_out, self.layers, self.attentions, incremental_state=incremental_state) 
        elif self.token_sequences:
            #print(' ### DECODER: using token sequences')
            #sys.stdout.flush()

            x, attn_scores = self.extract_features_layers([toks_prev_output_tokens, toks_prev_output_tokens], tgt_tok_bounds, sort_order, encoder_out, self.layers, self.attentions, incremental_state=incremental_state)
        elif self.char_sequences:
            #print(' ### DECODER: using char sequences')
            #sys.stdout.flush()

            x, attn_scores = self.extract_features_layers([char_prev_output_tokens, char_prev_output_tokens], tgt_tok_bounds, sort_order, encoder_out, self.layers, self.attentions, incremental_state=incremental_state)

        return x, attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight) 
            else:
                x = self.fc_out(x) 
        return x

    def restore_incremental_state(self, incremental_state):

        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state' + self.g_id)
        if cached_state is None:
            return

        backup_state = utils.get_incremental_state(self, incremental_state, 'backup_state' + self.g_id)
        if backup_state is None:
            backup_state = ()
            for s in cached_state:
                backup_state.append( s.clone() )
            utils.set_incremental_state(self, incremental_state, 'backup_state' + self.g_id, backup_state)
        else:
            cached_state = ()
            for s in backup_state:
                cached_state.append( s.clone() )
            utils.set_incremental_state(self, incremental_state, 'cached_state' + self.g_id, cached_state)


    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order) 
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state' + self.g_id)
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list) or isinstance(state, tuple):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:

                if _DEBUG_:
                    print('[DEBUG] TarcLSTMDecoder, reordering state: order {}, state shape {}'.format(new_order, state.size()))
                    sys.stdout.flush()

                return state.index_select(0, new_order)
            else:
                return None 

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state' + self.g_id, new_state)

        ch_cached_state = utils.get_incremental_state(self, incremental_state, 'ch_cached_state')
        if ch_cached_state is None:
            return

        new_ch_state = tuple(map(reorder_state, ch_cached_state))
        utils.set_incremental_state(self, incremental_state, 'ch_cached_state', new_ch_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m


def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

