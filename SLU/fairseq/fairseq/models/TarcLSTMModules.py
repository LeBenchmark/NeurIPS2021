# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import LayerNorm
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)

from fairseq.models.lstm import (
    LSTMEncoder,
    AttentionLayer,
    LSTMDecoder,
)

from fairseq.modules import AdaptiveSoftmax

DEFAULT_MAX_SOURCE_POSITIONS = 2048
DEFAULT_MAX_TARGET_POSITIONS = 2048

char_reduce = torch.sum

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
    for bi in range(B): 
        t = tokens[bi,0].item()
        if t in t2c_map:
            tmp = [t2c_map[t].to(tokens)]
        else:
            sys.stderr.write(' *** get_chars_from_tokens WARNING: predicted token {} ({}) is not defined in current map, backing off to <bos>\n'.format(t, mdict.string(tokens[bi,:]))) 
            tmp = [t2c_map[mdict.bos()].to(tokens)]
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
        self, dictionary, embed_dim=512, hidden_size=512, num_layers=1,
        dropout_in=0.1, dropout_out=0.1, bidirectional=True,
        left_pad=False, pretrained_embed=None, padding_idx=None,
        max_source_positions=DEFAULT_MAX_SOURCE_POSITIONS,
        token_map=None,
        granularity_flags=None
    ):
        super().__init__(dictionary)
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

        num_embeddings = len(dictionary)
        self.padding_idx = padding_idx if padding_idx is not None else dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.lstm = LSTM(
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

    def _forward_lstm(self, src_tokens, src_lengths, src_tok_bounds, sort_order, lstm):

        toks_src_tokens, char_src_tokens = src_tokens
        toks_src_lengths, char_src_lengths = src_lengths
        toks_sort_order, char_sort_order = sort_order

        bsz, seqlen = toks_src_tokens.size()

        # embed tokens
        x = self.embed_tokens(toks_src_tokens)

        x = F.dropout(x, p=self.dropout_in, training=self.training)

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
            xx[offset:T-1*offset,:,:] = toks_from_chars
 
            x = torch.cat( [x, xx], -1 )
        elif self.token_sequences and self.char_sequences:
            x = torch.cat( [x, torch.zeros_like(x)], -1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, toks_src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.new_zeros(*state_size)
        c0 = x.new_zeros(*state_size)
        packed_outs, (final_hiddens, final_cells) = lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:

            def combine_bidir(outs):
                out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
                return out.view(self.num_layers, bsz, -1)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = toks_src_tokens.eq(self.padding_idx).t()

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
            x, final_hiddens, final_cells, encoder_padding_mask = self._forward_lstm([toks_src_tokens, char_src_tokens], [toks_src_lengths, char_src_lengths], src_tok_bounds, sort_order, self.lstm)
        elif self.token_sequences: 
            x, final_hiddens, final_cells, encoder_padding_mask = self._forward_lstm([toks_src_tokens, toks_src_tokens], [toks_src_lengths, toks_src_lengths], src_tok_bounds, sort_order, self.lstm)
        elif self.char_sequences:
            x, final_hiddens, final_cells, encoder_padding_mask = self._forward_lstm([char_src_tokens, char_src_tokens], [char_src_lengths, char_src_lengths], src_tok_bounds, sort_order, self.lstm) 

        '''print(' - TarcLSTMEncoder, output shapes:')
        print('   * x: {}'.format(x.size()))
        print('   * final_hiddens: {}'.format(final_hiddens.size()))
        print('   * final_cells: {}'.format(final_cells.size()))
        print('   * encoder_padding_mak: {}'.format(encoder_padding_mask.size()))
        print(' -----')
        sys.stdout.flush()
        sys.exit(0)'''

        return {
            'encoder_out': (x, final_hiddens, final_cells),
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

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

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
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, num_attentions=1,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=DEFAULT_MAX_TARGET_POSITIONS,
        token_map=None,
        granularity_flags=None,
        double_learning=False
    ):
        super().__init__(dictionary)
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

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx) 
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size * num_attentions
        total_embed_size = 2*embed_dim if self.token_sequences and self.char_sequences else embed_dim

        # TODO: new, we try to modify the decoder to perform a chain of operations similar to the one of transformer.
        '''self.lstm_input_map = Linear(input_feed_size+total_embed_size, hidden_size, bias=False) if input_feed_size+total_embed_size != hidden_size else None
        self.lstm_input_norm = [LayerNorm(hidden_size) for layer in range(num_layers)]
        self.attn_output_norm = LayerNorm(hidden_size)
        self.final_norm = LayerNorm(hidden_size)'''

        self.layers = nn.ModuleList([
            LSTMCell(
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
            self.attentions.append( AttentionLayer(query_size, key_size, value_size, bias=False) ) 
 
        if self.double_learning or hidden_size != out_embed_dim:
            if hidden_size != out_embed_dim:
                self.tk_additional_fc = Linear(hidden_size, out_embed_dim)
            if self.double_learning:
                self.char_rnn = LSTMCell(input_size=hidden_size, hidden_size=hidden_size) #Linear(embed_dim + hidden_size, hidden_size)    # input: char embed, $c_{i-1}$ (input-feed) => $h_i$
                self.char2tok_att = AttentionLayer(hidden_size, hidden_size, hidden_size, bias=False)   # input: $h_i$, x => $c_i$
                if hidden_size != out_embed_dim:
                    self.char_out = Linear(hidden_size, out_embed_dim)  # input: $c_i$ => $o_i$
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff, dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

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
        hidden_states = x[0]    # NOTE: x[0] is the token-level representation on which following decoders will put attention...it could be interesting to try x[1], the character-level representation

        x_tk = self.output_layer(x[0])
        x_ch = self.output_layer(x[1]) if x[1] is not None else None

        return (x_tk, x_ch), {'attn' : attn_scores, 'hidden' : hidden_states}

    def extract_features_layers(
        self, prev_output_tokens, tgt_tok_bounds, sort_order, encoder_out, layers, atts, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """ 

        toks_prev_output, char_prev_output = prev_output_tokens
        toks_sort_order, char_sort_order = sort_order
        padding_flag = toks_prev_output[0,0].item() == self.dictionary.eos()

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
            xx[offset:T,:,:] = toks_from_chars # NOTE: anything else than toks_from_chars should be padding, or bos or eos

            x = torch.cat( [x, xx], -1 )
        elif self.token_sequences and self.char_sequences:
            x = torch.cat( [x, torch.zeros_like(x)], -1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state' + self.g_id)
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        elif encoder_outputs is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
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
        for j in range(seqlen):
            #if double_signal_flag and j >= min(tk_seqlen, ch_seqlen):
            #    x = torch.cat( [y, torch.zeros_like(y)], -1)

            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((x[j, :, :], input_feed), dim=1) 
            else:
                input = x[j]

            #rnn_input = self.lstm_input_map(input) if self.lstm_input_map is not None else input
            for i, rnn in enumerate(layers):
                #residual = rnn_input
                #rnn_input = self.lstm_input_norm[i](rnn_input)

                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                #rnn_input = residual + rnn_input

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            if double_signal_flag:
                tc_hiddens.append( hidden )
            # apply attention using the last layer's hidden state
            attn_out = []
            for i in range(self.num_attentions):
                if i == 0:
                    '''print(' - Attention {} input shapes:'.format(i))
                    print('   * hidden: {}'.format(hidden.size()))
                    print('   * encoder_outs: {}'.format(encoder_outs.size()))
                    print(' -----')
                    sys.stdout.flush()'''
                    
                    out, attn_scores[:, j, :] = atts[i](hidden, encoder_outs, encoder_padding_mask)
                else:
                    '''print(' - Attention {} input shapes:'.format(i))
                    print('   * hidden: {}'.format(hidden.size()))
                    print('   * encoder_outs: {}'.format(encoder_out[i]['encoder_out'][0].size()))
                    print(' -----')
                    sys.stdout.flush()'''

                    out, _ = atts[i](hidden, encoder_out[i]['encoder_out'][0], None) 

                out = F.dropout(out, p=self.dropout_out, training=self.training)
                attn_out.append( out )

                #if j < tk_seqlen:
                outs[i].append( out ) 
                #if j < ch_seqlen and double_signal_flag:
                #    ch_outs[i].append( out.clone() )

            # input feeding
            if input_feed is not None:
                input_feed = torch.cat(attn_out, -1) #attn_out[0]

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
        return (x, xx), attn_scores

    def extract_features(
        self, prev_output_tokens, tgt_tok_bounds, sort_order, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """ 

        toks_prev_output_tokens, char_prev_output_tokens = prev_output_tokens
        toks_sort_order, char_sort_order = sort_order

        if self.token_sequences and self.char_sequences:
            x, attn_scores = self.extract_features_layers([toks_prev_output_tokens, char_prev_output_tokens], tgt_tok_bounds, sort_order, encoder_out, self.layers, self.attentions, incremental_state=incremental_state) 
        elif self.token_sequences:
            x, attn_scores = self.extract_features_layers([toks_prev_output_tokens, toks_prev_output_tokens], tgt_tok_bounds, sort_order, encoder_out, self.layers, self.attentions, incremental_state=incremental_state)
        elif self.char_sequences:
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

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order) 
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state' + self.g_id)
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:
                return state.index_select(0, new_order)
            else:
                return None

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state' + self.g_id, new_state)

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

