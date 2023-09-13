# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Dict, Optional, Tuple

from torch.nn import Parameter
from fairseq.incremental_decoding_utils import with_incremental_state

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import options, utils, tarc_utils, init_functions
from torch import Tensor

import sys
import math
from typing import Any, Dict, List, Optional, Tuple

from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm, 
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttention,
    TransformerEncoderLayer,
)

from fairseq.data.TarcMultiTaskDataset import collate_tokens_ex

_DEBUG_ = False

class TarcTransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = init_functions.TransformerLinear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = init_functions.TransformerLinear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape (T_tgt, T_src), where
            T_tgt is the length of query, while T_src is the length of key,
            though here both query and key is x here,
            attn_mask[t_tgt, t_src] = 1 means when calculating embedding
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # TODO: to formally solve this problem, we need to change fairseq's
        # MultiheadAttention. We will do this later on.
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        #print(' TarcTransformerLayer, returning x: {}'.format(x.size()))
        #sys.stdout.flush()

        return x


class TarcTransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens, token_map=None, granularity_flags=None, ssl_encoder=None):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.args = args
        self.token2components_map = token_map
        self.token_sequences = granularity_flags[0] if granularity_flags is not None else False
        self.char_sequences = granularity_flags[1] if granularity_flags is not None else False
        self.merge_flag = False
        self.left_pad = False

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        # NOTE: Encoder size is determined by args.encoder_embed_dim
        #if args.model_type == 'lstm' or (self.token_sequences and self.char_sequences):
        #    #args.encoder_embed_dim = 2*args.encoder_embed_dim
        #    embed_dim = args.encoder_embed_dim if embed_dim != args.encoder_embed_dim else embed_dim
        self.padding_idx = dictionary.pad() #embed_tokens.padding_idx
        if ssl_encoder is not None:
            if self.args.ssl_type in ['camembert-base', 'camembert-large']:
                self.padding_idx = ssl_encoder.task.source_dictionary.pad()
            elif self.args.ssl_type in ['flaubert1-base', 'flaubert1-large']:
                self.padding_idx = ssl_encoder['model'].embeddings.padding_idx
            elif self.args.ssl_type == 'flauberto1-base':
                self.padding_idx = ssl_encoder['model'].embeddings.padding_idx # NOTE: this becomes .transformer.embeddings.padding_idx for model tuned on MEDIA from vpelloin
            elif self.args.ssl_type == 'jargon-base' or self.args.ssl_type == 'jargon-large':
                self.padding_idx = ssl_encoder['model'].roberta.embeddings.padding_idx
        self.max_source_positions = args.max_source_positions

        self.ssl_encoder = ssl_encoder
        if ssl_encoder is None:
            self.embed_tokens = embed_tokens
            self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)
            self.embed_positions = (
                PositionalEmbedding(
                    args.max_source_positions,
                    embed_dim,
                    self.padding_idx,
                    learned=args.encoder_learned_pos,
                )
                if not args.no_token_positional_embeddings
                else None
            )
        else:
            self.embed_tokens = self.embed_positions = None
            self.embed_scale = 1.0

            ssl_embed_dim = 768 if 'base' in args.ssl_type else 1024
            if 'flaubert' in self.args.ssl_type or 'jargon' in self.args.ssl_type:
                self.ssl_model = self.ssl_encoder['model']
            if ssl_embed_dim != embed_dim:
                self.normalize_before = args.encoder_normalize_before
                self.embed_to_dmodel = init_functions.TransformerLinear(ssl_embed_dim, embed_dim, bias=False) 
                self.hid_to_dmodel = init_functions.TransformerLinear(ssl_embed_dim, embed_dim, bias=False)
                self.hid_to_dmodel_norm = LayerNorm(ssl_embed_dim) if args.encoder_normalize_before else LayerNorm(embed_dim)

        self.layer_wise_attention = getattr(args, "layer_wise_attention", False) 
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TarcTransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def set_merge_flag(self, val):
        self.merge_flag = val

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def encode_with_ssl(self, src_tokens):

        if self.args.ssl_type in ['camembert-base', 'camembert-large']:
            #src_tokens[src_tokens == self.dictionary.pad()] = self.ssl_encoder.task.source_dictionary.pad()
            #x = self.ssl_encoder.extract_features(src_tokens, return_all_hiddens=True)
            #embeddings = x[0]
            #x = torch.stack( x[-4:], dim=-1 )
            #x = torch.mean( x, dim=3 )

            samples = [self.ssl_encoder.encode( self.dictionary.string(t) ) for t in src_tokens]
            lengths = torch.LongTensor( [len(s) for s in samples] ).to(src_tokens)
            new_tokens = collate_tokens_ex(samples, self.ssl_encoder.task.source_dictionary.pad())
            #_, sort_order = lengths.sort(descending=True)
            #new_tokens = new_tokens.index_select(0, sort_order)
            new_tokens = new_tokens.to(src_tokens)
            x = self.ssl_encoder.extract_features(new_tokens, return_all_hiddens=True)
            embeddings = x[0]
            x = torch.stack( x[-4:], dim=-1 )
            x = torch.mean( x, dim=3 )

            return x, embeddings, new_tokens

        elif self.args.ssl_type in ['flaubert1-base','flaubert1-large']:
            #src_tokens[src_tokens == self.dictionary.pad()] = self.ssl_encoder['model'].embeddings.padding_idx 
            #outputs = self.ssl_encoder['model'](src_tokens, output_hidden_states=True)
            #x = torch.stack( outputs.hidden_states[-4:], dim=-1 )
            #x = torch.mean( x, dim=3 )

            samples = [torch.tensor( [self.ssl_encoder['tokenizer'].encode( self.dictionary.string(t) )] ).squeeze() for t in src_tokens]
            lengths = torch.LongTensor( [len(s) for s in samples] ).to(src_tokens)
            new_tokens = collate_tokens_ex( samples, self.ssl_encoder['model'].embeddings.padding_idx )
            new_tokens = new_tokens.to( src_tokens )
            outputs = self.ssl_encoder['model'](new_tokens, output_hidden_states=True)
            embeddings = outputs.hidden_states[0]
            x = torch.stack( outputs.hidden_states[-4:], dim=-1 )
            x = torch.mean(x, dim=3)

            return x, embeddings, new_tokens

        elif self.args.ssl_type == 'flauberto1-base':

            samples = [self.dictionary.string(t) for t in src_tokens]
            new_tokens = self.ssl_encoder['tokenizer']( samples, padding=True, return_tensors='pt') 
            new_tokens = new_tokens.to( src_tokens.device )
            outputs = self.ssl_encoder['model'](**new_tokens, output_hidden_states=True)
            embeddings = outputs.hidden_states[0]
            x = torch.stack( outputs.hidden_states[-4:], dim=-1 )
            x = torch.mean(x, dim=3)

            return x, embeddings, new_tokens.input_ids

        elif self.args.ssl_type == 'jargon-base':

            samples = [self.dictionary.string(t) for t in src_tokens]
            new_tokens = self.ssl_encoder['tokenizer']( samples, padding=True, return_tensors='pt') 
            new_tokens = new_tokens.to( src_tokens.device )
            outputs = self.ssl_encoder['model'](**new_tokens, output_hidden_states=True)
            embeddings = outputs.hidden_states[0]
            x = torch.stack( outputs.hidden_states[-4:], dim=-1 )
            x = torch.mean(x, dim=3)

            return x, embeddings, new_tokens.input_ids

        else:
            raise ValueError('Unrecognized ssl encoder type {}'.format(self.args.ssl_type))

    def _forward_transformer(
        self,
        src_tokens,
        src_lengths,
        src_tok_bounds,
        sort_order,
        cls_input: Optional[Tensor] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        toks_src_tokens, char_src_tokens = src_tokens
        toks_src_lengths, char_src_lengths = src_lengths
        toks_sort_order, char_sort_order = sort_order

        bsz, seqlen = toks_src_tokens.size()

        if self.ssl_encoder is None:
            x, encoder_embedding, toks_src_tokens = self.forward_embedding(toks_src_tokens)
            bsz, seqlen = toks_src_tokens.size()
        else:
            x, encoder_embedding, toks_src_tokens = self.encode_with_ssl(toks_src_tokens)

            #print('[DEBUG] x shape: {}'.format(x.size()))
            #print('[DEBUG] encoder_embedding shape: {}'.format(encoder_embedding.size()))
            #print('[DEBUG] toks_src_tokens shape: {}'.format(toks_src_tokens.size()))
            #sys.stdout.flush()
            #sys.exit(0)

            if hasattr(self, 'hid_to_dmodel'):
                #residual = x
                if self.normalize_before:
                    x = self.hid_to_dmodel_norm(x)
                x = self.hid_to_dmodel(x)
                encoder_embedding = self.embed_to_dmodel(encoder_embedding)
                x = F.dropout(x, p=self.dropout, training=self.training)
                #x = residual + x
                if not self.normalize_before:
                    x = self.hid_to_dmodel_norm(x)

        '''print(' - TarcTransformerEncoder:')
        print('   * x shape: {}'.format(x.size()))
        print('   * x min, max, mean, sum: {}, {}, {}, {}'.format(torch.min(x), torch.max(x), torch.mean(x), torch.sum(x)))
        sys.stdout.flush()
        sys.exit(0)'''

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        (T, B, C) = x.size()

        if self.merge_flag and self.token_sequences and self.char_sequences:
            padding_flag = toks_src_tokens[0,0] == self.dictionary.bos()
            offset = 0
            if padding_flag:
                offset = 1

            y = self.forward_embedding(char_src_tokens)
            y = F.dropout(y, p=self.dropout, training=self.training)
            y = y.transpose(0, 1)

            toks_from_chars = tarc_utils.char2token_features_(y, x.size(), src_tok_bounds, toks_sort_order, offset)
            xx = torch.zeros_like(x).copy_(x)
            xx[offset:T-1*offset,:,:] = toks_from_chars

            x = torch.cat( [x, xx], -1 )
        elif self.token_sequences and self.char_sequences:
            x = torch.cat( [x, torch.zeros_like(x)], -1)

        # compute padding mask
        encoder_padding_mask = toks_src_tokens.eq(self.padding_idx)
        if 'jargon' in self.args.ssl_type:
            encoder_padding_mask = encoder_padding_mask.transpose(0, 1)
        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    assert encoder_states is not None
                    encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        if 'jargon' in self.args.ssl_type:
            encoder_padding_mask = encoder_padding_mask.transpose(0, 1)

        #print('[DEBUG] $$$$$$$$$$$$$$$$$$$$$')
        #print('[DEBUG] Encoder output shape: {}'.format(x.size()))
        #print('[DEBUG] $$$$$$$$$$$$$$$$$$$$$')
        #sys.stdout.flush()

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

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
            enc_out = self._forward_transformer([toks_src_tokens, char_src_tokens], [toks_src_lengths, char_src_lengths], src_tok_bounds, sort_order)
        elif self.token_sequences:
            enc_out = self._forward_transformer([toks_src_tokens, toks_src_tokens], [toks_src_lengths, toks_src_lengths], src_tok_bounds, sort_order)
        elif self.char_sequences:
            enc_out = self._forward_transformer([char_src_tokens, char_src_tokens], [char_src_lengths, char_src_lengths], src_tok_bounds, sort_order)

        '''print(' TarcTransformerEncoder, returning EncoderOut structure:')
        print('  * encoder_out shape: {}'.format(enc_out.encoder_out.size()))
        print('  * encoder_padding_mask shape: {}'.format(enc_out.encoder_padding_mask.size()))
        print('  * encoder_embedding shape: {}'.format(enc_out.encoder_embedding.size()))
        if enc_out.encoder_states is not None:
            print('  * encoder_states length: {}'.format(len(enc_out.encoder_states)))
        print(' -----')
        sys.stdout.flush()'''

        return enc_out
        '''inv_idx = range(enc_out.encoder_out.size(0)-1, -1, -1)
        encoder_out = torch.cat( [enc_out.encoder_out, enc_out.encoder_out[inv_idx, :, :]], -1 )
        final_hiddens = torch.stack( [encoder_out[-1,:,:] for i in range(self.num_layers)], 0 )
        final_cells = torch.stack( [encoder_out[-1,:,:] for i in range(self.num_layers)], 0 )

        return {
            'encoder_out': (encoder_out, final_hiddens, final_cells),
            'encoder_padding_mask': enc_out.encoder_padding_mask.transpose(0, 1) if enc_out.encoder_padding_mask.any() else None
        }'''


    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out: Dict[str, Tensor] = {}

        new_encoder_out["encoder_out"] = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_out["encoder_padding_mask"] = (
            encoder_out.encoder_padding_mask
            if encoder_out.encoder_padding_mask is None
            else encoder_out.encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_out["encoder_embedding"] = (
            encoder_out.encoder_embedding
            if encoder_out.encoder_embedding is None
            else encoder_out.encoder_embedding.index_select(0, new_order)
        )

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out["encoder_out"],  # T x B x C
            encoder_padding_mask=new_encoder_out["encoder_padding_mask"],  # B x T
            encoder_embedding=new_encoder_out["encoder_embedding"],  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, "_future_mask")
            or self._future_mask is None
            or self._future_mask.device != tensor.device
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(tensor.new(dim, dim)), 1
            )
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(
                    utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1
                )
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict

class TarcTransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, num_cross_attentions=0, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.num_cross_attentions = num_cross_attentions
        self.embed_dim = args.decoder_embed_dim
        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not self.cross_self_attention,
        )
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        self.activation_dropout = getattr(args, "activation_dropout", 0)
        if self.activation_dropout == 0:
            # for backwards compatibility with models that use args.relu_dropout
            self.activation_dropout = getattr(args, "relu_dropout", 0)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(
                self.embed_dim,
                args.decoder_attention_heads,
                kdim=getattr(args, "encoder_embed_dim", None),
                vdim=getattr(args, "encoder_embed_dim", None),
                dropout=args.attention_dropout,
                encoder_decoder_attention=True,
            )
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        # This is my main modification: cross-attentions to attend the other decoder outputs, queries are the same as the encoder MHA 
        self.cross_attentions = nn.ModuleList()
        self.cross_attentions_norm = nn.ModuleList()
        if num_cross_attentions > 0:    # TODO: for new architecture (xattn concatenation), validation in progress...
            if self.normalize_before:
                self.xattn_norm = LayerNorm(self.embed_dim * (num_cross_attentions+1), export=export)
            self.xattn_fc1 = init_functions.TransformerLinear(self.embed_dim * (num_cross_attentions+1), args.decoder_ffn_embed_dim)
            self.xattn_fc2 = init_functions.TransformerLinear(args.decoder_ffn_embed_dim, self.embed_dim)
        for i in range( num_cross_attentions ):
            self.cross_attentions.append(
                MultiheadAttention(
                    self.embed_dim,
                    args.decoder_attention_heads,
                    kdim=self.embed_dim,
                    vdim=self.embed_dim,
                    dropout=args.attention_dropout,
                    encoder_decoder_attention=True,
                )
            )
            self.cross_attentions_norm.append(
                LayerNorm(self.embed_dim, export=export)
            )

        if num_cross_attentions == 0:   # TODO: for new architecture (xattn concatenation) ...
            self.fc1 = init_functions.TransformerLinear(self.embed_dim, args.decoder_ffn_embed_dim)
            self.fc2 = init_functions.TransformerLinear(args.decoder_ffn_embed_dim, self.embed_dim)

        if self.num_cross_attentions == 0 or not self.normalize_before: # TODO: for new architecture (xattn concatenation) ...
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(
        self,
        x,
        encoder_out: Optional[List[torch.Tensor]] = None,
        encoder_padding_mask: Optional[List[torch.Tensor]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        prev_cross_attn_state: Optional[List[List[torch.Tensor]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        assert len(self.cross_attentions)+1 == len(encoder_out)

        '''print(' TarcTransformerDecoderLayer, input x shape: {}'.format(x.size()))
        print('    * num. of value sets: {}'.format(len(encoder_out)))
        sys.stdout.flush()'''

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        #print(' *** [DEBUG] TarcTransformerDecoderLayer, prev_self_attn_state is None ? {}'.format(prev_self_attn_state is None))
        #sys.stdout.flush()

        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

        #print(' *** [DEBUG] TarcTransformerDecoderLayer, _self_attn_input_buffer is None ? {}'.format(_self_attn_input_buffer is None))
        #sys.stdout.flush()

        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):

            #print(' ***** [DEBUG] ***** TarcTransformerDecoderLayer, setting query for self-attention')
            #sys.stdout.flush()

            if self_attn_mask is not None:
                assert encoder_out[0] is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out[0].size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                encoder_zero_padding_mask = encoder_padding_mask[0]
                if encoder_padding_mask[0] is None:
                    assert encoder_out[0] is not None
                    encoder_zero_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out[0].size(1), encoder_out[0].size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_zero_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out[0] is not None
            y = torch.cat((encoder_out[0], x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        #print(' TarcTransformerDecoderLayer, self attention computed...')
        #sys.stdout.flush()

        cross_attn_x = x

        #print('[DEBUG] cross_attn_x shape: {}'.format(cross_attn_x.size()))
        #sys.stdout.flush()

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            #print(' *** [DEBUG] TarcTransformerDecoderLayer, prev_attn_state is None ? {}'.format(prev_attn_state is None))
            #sys.stdout.flush()

            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)


            #print('[DEBUG] -')
            #print('[DEBUG] encoder attn:')
            #print('[DEBUG] query shape: {}'.format(x.size()))
            #print('[DEBUG] key shape: {}'.format(encoder_out[0].size()))
            #print('[DEBUG] key_padding_mask shape: {}'.format(encoder_padding_mask[0].size() if encoder_padding_mask is not None else None))
            #print('[DEBUG] -')
            #sys.stdout.flush()

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out[0],
                value=encoder_out[0],
                key_padding_mask=encoder_padding_mask[0],
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            #print(' TarcTransformerDecoderLayer, cross attention (encoder) computed...')
            #sys.stdout.flush()

        if self.num_cross_attentions > 0:
            #residual = cross_attn_x
            all_att_output = [] #torch.zeros_like(cross_attn_x) # TODO: [] is for new architecture (xattn concatenation)...
            if self.normalize_before:
                cross_attn_x = self.cross_attentions_norm[0](cross_attn_x)
            for i in range( len(self.cross_attentions) ):

                #print(' *** [DEBUG] TarcTransformerDecoderLayer, prev_cross_attn_state is None ? {}'.format(prev_cross_attn_state is None))
                #sys.stdout.flush()

                if prev_cross_attn_state is not None:
                    prev_key, prev_value = prev_cross_attn_state[i][:2]
                    cross_saved_state: Dict[str, Optional[Tensor]] = {
                        "prev_key": prev_key,
                        "prev_value": prev_value,
                    }
                    if len(prev_cross_attn_state[i]) >= 3:
                        cross_saved_state["prev_key_padding_mask"] = prev_cross_attn_state[i][2]
                    assert incremental_state is not None
                    self.cross_attentions[i]._set_input_buffer(incremental_state, cross_saved_state)

                #print('[DEBUG] cross_attention {} input shape:'.format(i))
                #print('[DEBUG] * query: {}'.format(cross_attn_x.size()))
                #print('[DEBUG] * key: {}'.format(encoder_out[i+1].size()))
                #print('[DEBUG] * padding mask: {}'.format(encoder_padding_mask[i+1].size() if encoder_padding_mask[i+1] is not None else None))
                #print('[DEBUG] ----------')
                #sys.stdout.flush()

                att_output, attn = self.cross_attentions[i](
                    query=cross_attn_x,
                    key=encoder_out[i+1],
                    value=encoder_out[i+1],
                    key_padding_mask=encoder_padding_mask[i+1],
                    incremental_state=incremental_state,
                    static_kv=True,
                    need_weights=need_attn or (not self.training and self.need_attn),
                    need_head_weights=need_head_weights,
                )

                #print('[DEBUG] att_output shape: {}'.format(att_output.size()))
                #sys.stdout.flush()

                att_output = F.dropout(att_output, p=self.dropout, training=self.training)
                #att_output = cross_attn_x + att_output # Residual add also here ???
                if not self.normalize_before:
                    att_output = self.cross_attentions_norm[i](att_output)
                all_att_output.append(att_output) #= att_output + all_att_output    # TODO: .append(att_output) is for new architecture (xattn concatenation) ...

                #print(' TarcTransformerDecoderLayer, cross attention (decoder {}) computed'.format(i))
                #sys.stdout.flush()

            # TODO: for new architecture (xattn concatenation) ...
            all_att_output.append(x)
            all_att_output = torch.cat(all_att_output, -1)

            #print('[DEBUG] all_att_output shape: {}'.format(all_att_output.size()))

            if self.normalize_before:
                all_att_output = self.xattn_norm(all_att_output)
            all_att_output = self.activation_fn( self.xattn_fc1(all_att_output) )
            all_att_output = F.dropout(all_att_output, p=float(self.activation_dropout), training=self.training)
            all_att_output = self.xattn_fc2( all_att_output )
            all_att_output = F.dropout(all_att_output, p=float(self.activation_dropout), training=self.training)
            # END TODO: for new architecture

            #print('[DEBUG] final all_att_output shape: {}'.format(all_att_output.size()))
            #sys.stdout.flush()

            x = x + all_att_output
            if not self.normalize_before:
                x = self.final_layer_norm(x)
            #if not self.normalize_before:
            #    x = self.cross_attentions_norm[0](x)

        if self.num_cross_attentions == 0:  # TODO: for new architecture (xattn concatenation) ...
            residual = x
            if self.normalize_before:
                x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = F.dropout(x, p=float(self.activation_dropout), training=self.training)
            x = self.fc2(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            if not self.normalize_before:
                x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        #print(' TarcTransformerDecoderLayer passed, output shape: {}'.format(x.size()))
        #sys.stdout.flush()

        #print('[DEBUG] ##########')
        #sys.stdout.flush()

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Scriptable reorder incremental state in transformer layers."""
        self.self_attn.reorder_incremental_state(incremental_state, new_order)

        if self.encoder_attn is not None:
            self.encoder_attn.reorder_incremental_state(incremental_state, new_order)

        if self.num_cross_attentions > 0:
            [attn.reorder_incremental_state(incremental_state, new_order) for attn in self.cross_attentions]
            #for i in range(len(self.cross_attentions)):
            #    self.cross_attentions[i].reorder_incremental_state(incremental_state, new_order)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

class TarcTransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, num_cross_attentions, dictionary, embed_tokens, no_encoder_attn=False,
        token_map=None,
        granularity_flags=None,
        double_learning=False,
        input_dict=None
    ):
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)
        self.num_cross_attentions = num_cross_attentions

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.token2components_map = token_map
        self.token_sequences = granularity_flags[0] if granularity_flags is not None else False
        self.char_sequences = granularity_flags[1] if granularity_flags is not None else False
        self.g_id = 'char' if ((not self.token_sequences) and self.char_sequences) else 'token'
        self.merge_flag = False
        self.double_learning = double_learning

        self.embed_tokens = embed_tokens
        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            init_functions.TransformerLinear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                args.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )

        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        self.layer_wise_attention = getattr(args, "layer_wise_attention", False)

        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                TarcTransformerDecoderLayer(args, no_encoder_attn, self.num_cross_attentions)
                for _ in range(args.decoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        self.adaptive_softmax = None

        self.project_out_dim = (
            init_functions.TransformerLinear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        if double_learning:
            raise NotImplementedError

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(
                torch.Tensor(len(dictionary), self.output_embed_dim)
            )
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def set_merge_flag(self, val):
        self.merge_flag = val

    def get_token_from_chars_(self, token, embed):
        raise NotImplementedError

    def forward(
        self,
        prev_output_tokens, tgt_tok_bounds, sort_order,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens, tgt_tok_bounds, sort_order,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        hidden_states = x[0]
        x_tk = x[0]
        x_ch = x[1] if x[1] is not None else None

        if not features_only:
            x_tk = self.output_layer(x[0])
            if x[1] is not None:
                x_ch = self.output_layer(x[1])

        '''print(' TarcTransformerDecoder, forward passed:')
        print('   * output shape: {}'.format(x_tk.size()))
        print('   * hidden states shape: {}'.format(hidden_states.size()))
        sys.stdout.flush()'''

        return (x_tk, x_ch), {'extra': extra, 'hidden': hidden_states}

    def extract_features_layers(
        self,
        prev_output_tokens, tgt_tok_bounds, sort_order,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """ 

        #print(' TarcTransformerDecoder, extracting features from layers...')
        #sys.stdout.flush() 

        toks_prev_output, char_prev_output = prev_output_tokens
        toks_sort_order, char_sort_order = sort_order
        padding_flag = toks_prev_output[0,0].item() == self.dictionary.eos()

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        toks_positions = (
            self.embed_positions(
                toks_prev_output, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )
        char_positions = (
            self.embed_positions(
                char_prev_output, incremental_state=incremental_state
            )
            if self.embed_positions is not None and self.token_sequences and self.char_sequences
            else None
        )

        if incremental_state is not None:
            #prev_output_tokens = prev_output_tokens[:, -1:]
            toks_prev_output = toks_prev_output[:, -1:]
            if toks_positions is not None:
                toks_positions = toks_positions[:, -1:]
            char_prev_output = get_chars_from_tokens_(toks_prev_output, self.token2components_map, self.dictionary) if (self.token_sequences and self.char_sequences) else char_prev_output[:,-1:]
            if char_positions is not None:
                char_positions = char_positions[:, -1:]

        double_signal_flag = self.double_learning and self.merge_flag and self.token_sequences and self.char_sequences
        #tk_bsz, tk_seqlen = toks_prev_output.size()
        #ch_bsz, ch_seqlen = char_prev_output.size()

        #bsz = tk_bsz
        #seqlen = tk_seqlen
 
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(toks_prev_output)

        '''print(' TarcTransformerDecoder, input embedded:')
        print('   * input shape: {}'.format(toks_prev_output.size()))
        print('   * embedding shape: {}'.format(x.size()))
        sys.stdout.flush()'''

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if toks_positions is not None:
            x += toks_positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

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

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or toks_prev_output.eq(self.padding_idx).any():
            self_attn_padding_mask = toks_prev_output.eq(self.padding_idx)

        #print(' TarcTransformerDecoder, final input x computed: {}'.format(x.size()))
        #sys.stdout.flush() 

        outs = []
        ch_outs = []
        for i in range(self.num_cross_attentions):
            outs.append([])
            ch_outs.append([])
        tc_hiddens = []

        if _DEBUG_:
            print('[DEBUG] TarcTransformerDecoder, encoder_out length: {}'.format(len(encoder_out)))
            for e in encoder_out:
                print('[DEBUG]    * TarcTransformerDecoder, encoder_out element type: {}'.format(type(e)))
                if isinstance(e, list):
                    print('[DEBUG]    * TarcTransformerDecoder, detected list type:')
                    for ee in e:
                        if ee is not None:
                            print('[DEBUG]    * TracTransformerDecoder: {}'.format(ee.size()))
                        else:
                            print('[DEBUG]    * TarcTransformerDecoder: {}'.format(ee))
            sys.stdout.flush()

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state: Optional[Tensor] = None
            if encoder_out[0] is not None:
                if self.layer_wise_attention:
                    encoder_states = encoder_out[0].encoder_states
                    assert encoder_states is not None
                    encoder_state = encoder_states[idx]
                else:
                    encoder_state = encoder_out[0].encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None 

            '''print(' TarcTransformerDecoder, values computed: {}'.format(encoder_state.size()))
            if self_attn_mask is not None:
                print('    self attention future mask computed')
            sys.stdout.flush()'''

            trans_layer_input = [encoder_state]
            trans_layer_mask = [encoder_out[0].encoder_padding_mask]
            for i_idx in range(1,len(encoder_out)):
                trans_layer_input.append( encoder_out[i_idx].encoder_out )  # TODO: add modifications in order to be able to give encoder_states also here
                trans_layer_mask.append( encoder_out[i_idx].encoder_padding_mask )
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.empty(1).uniform_()

            '''print(' TarcTransformerDecoder, all values computed: {} value sets'.format(len(trans_layer_input)))
            if dropout_probability > self.decoder_layerdrop:
                print('   possibly dropping decoder layer ({}, {})'.format(dropout_probability, self.decoder_layerdrop))
            sys.stdout.flush()'''

            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn, _ = layer(
                    x,
                    trans_layer_input,
                    trans_layer_mask, #encoder_out[0].encoder_padding_mask if encoder_out[0] is not None else None,  # TODO: this may possibly be wrong, we are masking all previous decoder outputs with the encoder padding mask.
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=bool((idx == alignment_layer)),
                    need_head_weights=bool((idx == alignment_layer)),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float().to(x)

        #print(' TarcTransformerDecoder, layers computation passed')
        #sys.stdout.flush() 

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # TODO: RESTART WORKING FROM HERE!
        xx = None
        if double_signal_flag:
            raise NotImplementedError

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        if xx is not None:
            xx = xx.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)
            if xx is not None:
                xx = self.project_out_dim(xx)

        #print(' TarcTransformerDecoder, extrated features from layers, output shape: {}'.format(x.size()))
        #sys.stdout.flush() 

        return (x, xx), {"attn": [attn], "inner_states": inner_states}

    def extract_features(
        self,
        prev_output_tokens, tgt_tok_bounds, sort_order,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ): 
        """
        Similar to *forward* but only return features.
        """ 

        #print(' TarcTransformerDecoder, extracting features...')
        #sys.stdout.flush()

        toks_prev_output_tokens, char_prev_output_tokens = prev_output_tokens
        toks_sort_order, char_sort_order = sort_order

        if self.token_sequences and self.char_sequences:
            #print(' TarcTransformerDecoder, extracting features from tokens and characters...')
            #sys.stdout.flush()

            x, attn_scores = self.extract_features_layers([toks_prev_output_tokens, char_prev_output_tokens], tgt_tok_bounds, sort_order,
                    encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        elif self.token_sequences:
            #print(' TarcTransformerDecoder, extracting features from tokens...')
            #sys.stdout.flush()

            x, attn_scores = self.extract_features_layers([toks_prev_output_tokens, toks_prev_output_tokens], tgt_tok_bounds, sort_order,
                    encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        elif self.char_sequences:
            #print(' TarcTransformerDecoder, extracting features from characters...')
            #sys.stdout.flush()

            x, attn_scores = self.extract_features_layers([char_prev_output_tokens, char_prev_output_tokens], tgt_tok_bounds, sort_order,
                    encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads)
        else:
            raise ValueError('At least one between character and token sequences must be processed ({}, {})'.format(self.token_sequences, self.char_sequences))

        #print(' TarcTransformerDecoder, features extracted, shape: {}'.format(x[0].size()))
        #sys.stdout.flush()

        return x, attn_scores

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    # Overwirte the method to temporaily soppurt jit scriptable in Transformer
    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        """Scriptable reorder incremental state in the transformer."""
        for layer in self.layers:
            layer.reorder_incremental_state(incremental_state, new_order)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

