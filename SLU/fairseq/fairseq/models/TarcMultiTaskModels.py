# Code for adding the architecture for End2End SLU to Fairseq

import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from fairseq import utils
from fairseq.tarc_utils import *
from fairseq import checkpoint_utils, options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)

from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, base_architecture as trans_ba
from fairseq.models.lstm import LSTMModel, AttentionLayer, LSTMEncoder, LSTMDecoder, base_architecture as lstm_ba
from fairseq.models.TarcLSTMModules import TarcLSTMEncoder, TarcLSTMDecoder
from fairseq.models.TarcTransformerModules import TarcTransformerEncoder, TarcTransformerDecoder

# Importing for compatibilities with the Transformer model
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from torch import Tensor

DEFAULT_MAX_SOURCE_POSITIONS=2048
DEFAULT_MAX_TARGET_POSITIONS=2048

### ENCODER ###
class TarcMultiTaskEncoder(FairseqEncoder):
    
    def __init__(
                 self, args, dictionary, encoders
                 ):
        
        super().__init__(dictionary)
        self.args = args
        self.padding_idx = dictionary.pad_index
        self.sequence_separator = dictionary.add_symbol( args.sequence_separator )
        self.encoders = encoders
        self.granularity_merging_flags = () 

    def set_g_merging_flags_(self, g_flags: Tuple):
        assert len(g_flags) == len(self.encoders)
        self.granularity_merging_flags = g_flags
        for ei in range(len(self.encoders)):
            self.encoders[ei].set_merge_flag( g_flags[ei] )

    def forward(self, src_tokens, src_lengths, src_tok_bounds=None, sort_order=None, **kwargs):
        # The inputs to the ``forward()`` function are determined by the
        # Task, and in particular the ``'net_input'`` key in each
        # mini-batch.
        
        # *src_tokens* has shape `(batch, src_len, dim)`
        # *src_lengths* has shape `(batch)`.

        # Note that the source is typically padded on the left. This can be
        # configured by adding the `--left-pad-source "False"` command-line
        # argument, but here we'll make the Encoder handle either kind of
        # padding by converting everything to be right-padded.
        #if self.args.left_pad_source:
        #    # Convert left-padding to right-padding.
        #    src_tokens = utils.convert_padding_direction(
        #        src_tokens,
        #        padding_idx=self.dictionary.pad(),
        #        left_to_right=True
        #    ) 

        if len(self.encoders) > 1:
            toks_src_tokens = split_on_sep(src_tokens[0], self.sequence_separator)
            char_src_tokens = split_on_sep(src_tokens[1], self.sequence_separator)
            assert len(self.encoders) == len(toks_src_tokens) == len(char_src_tokens)
        else:
            toks_src_tokens = [src_tokens[0]]
            char_src_tokens = [src_tokens[1]] 

        outputs = []
        for i in range(len(self.encoders)):
            src_tokens_i = [toks_src_tokens[i], char_src_tokens[i]]
            src_lengths_i = [src_lengths[0][i], src_lengths[1][i]]
            curr_encoder_out = self.encoders[i](src_tokens_i, src_lengths_i, src_tok_bounds[i], sort_order)
            outputs.append( curr_encoder_out ) 

        if self.args.model_type == 'transformer':
            encoder_out = []
            encoder_embedding = []
            encoder_states = []
            encoder_padding_mask = outputs[0].encoder_padding_mask
            for o in outputs:
                if o.encoder_out is not None:
                    encoder_out.append( o.encoder_out )
                if o.encoder_embedding is not None:
                    encoder_embedding.append( o.encoder_embedding )
                if o.encoder_states is not None:
                    encoder_states.append( o.encoder_states )
            mt_encoder_states = None
            if len(encoder_states) > 0:
                n_layers = len(encoder_states[0])
                mt_encoder_states = [None] * n_layers
                for i in range(n_layers):
                    mt_encoder_states[i] = torch.cat( [encoder_layers[i] for encoder_layers in encoder_states], -1 )
            return EncoderOut(
                encoder_out=torch.cat( encoder_out, -1 ) if len(encoder_out) > 0 else None,
                encoder_padding_mask=encoder_padding_mask,
                encoder_embedding=torch.cat( encoder_embedding, -1 ) if len(encoder_embedding) > 0 else None,
                encoder_states=mt_encoder_states
            )
        else:
            mt_x = []
            mt_final_hiddens = []
            mt_final_cells = []
            encoder_padding_mask = outputs[0]['encoder_padding_mask'] 

            for o in outputs:
                mt_x.append( o['encoder_out'][0] )
                mt_final_hiddens.append( o['encoder_out'][1] )
                mt_final_cells.append( o['encoder_out'][2] )
            return {
                'encoder_out': (torch.cat(mt_x, -1), torch.cat(mt_final_hiddens, -1), torch.cat(mt_final_cells, -1)),
                'encoder_padding_mask': encoder_padding_mask
            }


    # Encoders are required to implement this method so that we can rearrange
    # the order of the batch elements during inference (e.g., beam search).
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

        return self.encoders[0].reorder_encoder_out(encoder_out, new_order)


### SIMPLE DECODER (for training the ICASSP2020 paper's Basic models in Fairseq) ###
from fairseq.models import FairseqDecoder

class TarcMultiTaskDecoder(FairseqIncrementalDecoder):

    def __init__(
        self, args, dictionary, decoders, embeddings=None,
    ):
        
        super().__init__(dictionary)

        self.args = args
        self.sequence_separator = dictionary.add_symbol(args.sequence_separator)
        self.decoders = decoders
        self.first_decoder = 0
        self.last_decoder = len(decoders)
        self.decoder_hidden_states = [[] for t_idx in range(len(decoders))]
        self.granularity_merging_flags = ()

    def set_g_merging_flags_(self, g_flags: Tuple):
        assert len(g_flags) == len(self.decoders)
        self.granularity_merging_flags = g_flags
        for di in range(len(self.decoders)):
            self.decoders[di].set_merge_flag( g_flags[di] )

    def set_active_tasks(self, start, end):
        self.first_decoder = start
        self.last_decoder = end

    def reset_hidden_states(self):
        self.decoder_hidden_states = [[] for t_idx in range(len(self.decoders))]

    def create_lstm_final_states_(self, hidden_state):
        if hidden_state is None:
            return (None, None)
        else:
            final_hiddens = torch.stack( [hidden_state[-1].clone() for i in range(self.args.decoder_layers)], 0 )
            final_cells = torch.stack( [hidden_state[-1].clone() for i in range(self.args.decoder_layers)], 0 )
            return (final_hiddens, final_cells)

    def forward(self, prev_output_tokens, shapes=None, tgt_tok_bounds=None, sort_order=None, encoder_out=None, src_lengths=None, incremental_state=None, **kwargs):
        if incremental_state is not None: 
            for i in range( len(self.decoders) ):
                es = utils.get_incremental_state(self, incremental_state, 'decoder-' + str(i+1))
                if es is None:
                    utils.set_incremental_state(self, incremental_state, 'decoder-' + str(i+1), {} )

        char_flag = (not self.args.token_sequences) and self.args.char_sequences
        if len(self.decoders) > 1:
            g_shapes = shapes if not char_flag else None
            toks_prev_output_tokens = split_on_sep(prev_output_tokens[0], self.sequence_separator, shapes=g_shapes)
            g_shapes = shapes if char_flag else None
            char_prev_output_tokens = split_on_sep(prev_output_tokens[1], self.sequence_separator, shapes=g_shapes)
 
            assert len(toks_prev_output_tokens) == len(self.decoders)
            assert len(char_prev_output_tokens) == len(self.decoders)
        else:
            toks_prev_output_tokens = [prev_output_tokens[0]]
            char_prev_output_tokens = [prev_output_tokens[1]]

        outputs = []
        decoder_input = [encoder_out]
        if self.training:
            self.decoder_hidden_states = [[] for i in range(len(self.decoders))]
        for i in range(self.first_decoder, self.last_decoder):
            incremental_state_i = utils.get_incremental_state(self, incremental_state, 'decoder-' + str(i+1))
            if not self.training and self.first_decoder > 0 and i == self.first_decoder:
                assert len(decoder_input) == 1
                for d_idx in range(0,i):
                    if self.args.model_type == 'lstm':
                        new_decoder_input = {
                            'encoder_out' : (torch.cat(self.decoder_hidden_states[d_idx], 0), None, None),
                            'encoder_padding_mask' : None
                        }
                        decoder_input.append( new_decoder_input )
                    else:
                        new_decoder_input = EncoderOut(
                            encoder_out=torch.cat(self.decoder_hidden_states[d_idx], 0),
                            encoder_padding_mask=None,
                            encoder_embedding=None,
                            encoder_states=None
                        )
                        decoder_input.append( new_decoder_input )

            feats_only = False
            prev_output_tokens_i = [toks_prev_output_tokens[i], char_prev_output_tokens[i]]
            src_lengths_i = [src_lengths[0][0], src_lengths[1][0]] if src_lengths is not None else [[], []]
            decoder_out = self.decoders[i](
                prev_output_tokens_i,
                tgt_tok_bounds[i],
                sort_order,
                encoder_out=decoder_input,
                features_only=feats_only,
                incremental_state=incremental_state_i,
                src_lengths=src_lengths_i if src_lengths is not None else None,
                return_all_hiddens = False,
            )
            outputs.append( decoder_out )

            hidden_state = decoder_out[1]['hidden'].transpose(0,1) 
            if not self.training:
                self.decoder_hidden_states[i].append( hidden_state ) 

            if self.args.model_type == 'transformer': 
                new_decoder_input = EncoderOut(
                    encoder_out = hidden_state, 
                    encoder_padding_mask = None,
                    encoder_embedding = None,
                    encoder_states = None,
                )
                decoder_input.append( new_decoder_input )
            else: 
                new_decoder_input = {
                    'encoder_out' : (hidden_state, None, None),
                    'encoder_padding_mask' : None
                }
                decoder_input.append( new_decoder_input ) 
 
        if not self.training and len(outputs) != len(self.decoders):
            assert len(outputs) == self.last_decoder - self.first_decoder
            output_clone = outputs[-1]
            for o_idx in range(0,len(self.decoders)-len(outputs)):
                outputs = [output_clone] + outputs

        x = []
        attn_scores = []
        for o in outputs:
            x.append( o[0] )
            attn_scores.append( o[1] ) 

        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):

        for i in range( len( self.decoders) ):
            self.decoders[i].reorder_incremental_state(utils.get_incremental_state(self, incremental_state, 'decoder-' + str(i+1)), new_order)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.args.max_target_positions


### ENCODER-DECODER MODEL ###
from fairseq.models import FairseqEncoderDecoderModel, register_model

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@register_model('tarc_multitask_model')
class TarcMultiTaskModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--model-type', type=str, default='lstm',
            help='Type of encoder and decoder to use: 1) lstm (default), 2) transformer',
        )
        parser.add_argument('--encoder-hidden-dim', type=int, default=256,
                            help="Size of encoder\'s hidden layer")
        parser.add_argument('--decoder-hidden-dim', type=int, default=256,
                            help="Size of decoder\'s hidden layer")
        parser.add_argument('--decoder-out-embed-dim', type=int, default=256,
                            help="Size of decoder\'s output embeddings")
        parser.add_argument('--num-of-inputs', type=int, default=1,
                            help='Number of different input item sequences')
        parser.add_argument(
            '--source-index', type=int, default=0,
            help='Index of the source among those provided as input (used for training a single task in a multi-task framework)',
        )
        parser.add_argument(
            '--target-index', type=int, default=1,
            help='Index of the target among those provided as input (used for training a single task in a multi-task framework)',
        )
        parser.add_argument(
            '--match-source-len', action='store_true', default=False,
            help='For scheduled-sampling decoding, same behavior as for fairseq-generate',
        )
        parser.add_argument(
            '--max-lan-a', type=float, default=0.4,
            help='For scheduled-sampling decoding, same behavior as for fairseq-generate',
        )
        parser.add_argument(
            '--max-len-b', type=int, default=1,
            help='For scheduled-sampling decoding, same behavior as for fairseq-generate',
        ) 
        TransformerModel.add_args(parser)
    
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, token2components_map):
        if args.model_type == 'transformer':
            return TarcTransformerEncoder(
                                        args,
                                        src_dict,
                                        embed_tokens,
                                        token_map=token2components_map,
                                        granularity_flags=(args.token_sequences, args.char_sequences)
            )
        elif args.model_type == 'lstm':
            return TarcLSTMEncoder(
                               dictionary=src_dict,
                               embed_dim=args.encoder_embed_dim,
                               hidden_size=args.encoder_hidden_dim,
                               num_layers=args.encoder_layers,
                               dropout_in=args.encoder_dropout_in,
                               dropout_out=args.encoder_dropout_out,
                               bidirectional=True,
                               pretrained_embed=embed_tokens,
                               max_source_positions=args.max_source_positions,
                               token_map=token2components_map,
                               granularity_flags=(args.token_sequences, args.char_sequences)
            )
        else:
            raise NotImplementedError

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens, idx, token2components_map):
        if args.model_type == 'transformer':
            return TarcTransformerDecoder(
                                      args,
                                      idx,
                                      tgt_dict,
                                      embed_tokens,
                                      no_encoder_attn=getattr(args, "no_cross_attention", False),
            )
        elif args.model_type == 'lstm':
            encoder_output_units = args.num_of_inputs * args.encoder_hidden_dim * 2
            hid_size = args.decoder_hidden_dim
            out_embed_dim = args.decoder_out_embed_dim

            return TarcLSTMDecoder(
                dictionary=tgt_dict,
                embed_dim=args.decoder_embed_dim,
                hidden_size=hid_size,
                out_embed_dim=out_embed_dim,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout_in,
                dropout_out=args.decoder_dropout_out,
                num_attentions=idx+1,
                encoder_output_units=encoder_output_units,
                pretrained_embed=embed_tokens,
                share_input_output_embed=args.share_decoder_input_output_embed,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                max_target_positions=args.max_target_positions,
                token_map=token2components_map,
                granularity_flags=(args.token_sequences, args.char_sequences),
                double_learning=args.double_learning
            )
        else:
            raise NotImplementedError

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called. 

        from fairseq import tasks 
 
        #if args.model_type == 'transformer':
        trans_ba(args)
        #elif args.model_type == 'lstm':
        lstm_ba(args)
        #else:
        #    raise NotImplementedError
        
        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        args.num_of_inputs = task.num_of_inputs 
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary 
        
        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = nn.Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )

            print(' TarcMultiTaskModel, sharing all embeddings')
            sys.stdout.flush()
 
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            ) 

            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
 
        if args.sub_task == 'tarc-full' and args.load_madar_data != 'None' and args.load_madar_model != 'None':
            print(' - TarcMultiTaskModel, pre-initializing common embeddings...')
            sys.stdout.flush()

            madar_splits = torch.load( args.load_madar_data)

            state = checkpoint_utils.load_checkpoint_to_cpu(args.load_madar_model)
            loaded_args = state['args']
            loaded_task = tasks.setup_task(loaded_args)

            loaded_task.input_vocab = madar_splits['vocab']
            loaded_task.output_vocab = madar_splits['vocab']
            loaded_task.token2components_tsr = madar_splits['token2components']
            loaded_task.splits = madar_splits
            loaded_task.set_granularity_merging_flags( task.get_granularity_merging_flags() )
            madar_model = loaded_task.build_model(loaded_args)

            madar_model.load_state_dict(state["model"], strict=True, args=loaded_args) 
            madar_dict = madar_splits['vocab']
            for split in ['train', 'dev', 'test']:
                madar_splits[split] = None

            pre_init_embs = 0
            for sym in madar_dict.indices.keys():
                if src_dict.__contains__(sym):
                    encoder_embed_tokens.weight.data[src_dict.index(sym),:] = madar_model.encoder.encoders[0].embed_tokens.weight.data[madar_dict.index(sym),:]
                    pre_init_embs += 1

            print(' - TarcMultiTaskModels: pre-initialized {} symbols out of {}'.format(pre_init_embs, len(src_dict)))
            sys.stdout.flush()

        encoders = nn.ModuleList()
        decoders = nn.ModuleList() 

        args.double_learning = task.double_learning
        num_encoders = task.num_of_inputs
        num_decoders = task.args.num_of_tasks
        enc_count = 0
        dec_count = 0

        for i in range( min(num_encoders, num_decoders) ):
            encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, task.token2components_tsr[enc_count])
            decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, dec_count, task.token2components_tsr[num_encoders+dec_count])
            encoders.append( encoder )
            decoders.append( decoder )
            enc_count += 1
            dec_count += 1
        while enc_count < num_encoders:
            encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, task.token2components_tsr[enc_count])
            encoders.append( encoder )
            enc_count += 1
        while dec_count < num_decoders:
            decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens, dec_count, task.token2components_tsr[num_encoders+dec_count])
            decoders.append( decoder )
            dec_count += 1
 
        if args.sub_task == 'tarc-full' and args.load_madar_data != 'None' and args.load_madar_model != 'None':
            print(' - Initializing decoder layers with pre-trained model...')
            sys.stdout.flush()

            for e_idx in range(len(encoders)):
                encoders[e_idx].lstm.load_state_dict( madar_model.encoder.encoders[e_idx].lstm.state_dict() )
            for d_idx in range(len(decoders)):
                # self.encoder_hidden_proj = self.encoder_cell_proj = None
                # self.layers
                # self.attentions
                decoders[d_idx].encoder_hidden_proj = madar_model.decoder.decoders[d_idx].encoder_hidden_proj if decoders[d_idx].encoder_hidden_proj is not None else None
                decoders[d_idx].encoder_cell_proj = madar_model.decoder.decoders[d_idx].encoder_cell_proj if decoders[d_idx].encoder_cell_proj is not None else None
                for l_idx in range( len(decoders[d_idx].layers) ):
                    decoders[d_idx].layers[l_idx].load_state_dict( madar_model.decoder.decoders[d_idx].layers[l_idx].state_dict() )
                for a_idx in range( len(decoders[d_idx].attentions) ):
                    decoders[d_idx].attentions[a_idx].load_state_dict( madar_model.decoder.decoders[d_idx].attentions[a_idx].state_dict() )


            '''# Decoder 3: POS tagging; Attentions: 1. encoder state, 2. classification, 3. transcription, 4. tokenization
            #     Loaded decoder 2: POS tagging; attentions: 1. encoder state (transcription), 2. classification, 3. tokenization
            decoders[3].layers.load_state_dict( madar_model.decoder.decoders[2].layers.state_dict() )
            decoders[3].attentions[0].load_state_dict( madar_model.decoder.decoders[2].attentions[0].state_dict() )
            decoders[3].attentions[1].load_state_dict( madar_model.decoder.decoders[2].attentions[1].state_dict() )
            # NOTE: We try to initialize the transcription attention with the attention to tokenization of the pre-trained model.
            #   They are not the same but it's better than no pre-initialization
            decoders[3].attentions[2].load_state_dict( madar_model.decoder.decoders[2].attentions[2].state_dict() )
            decoders[3].attentions[3].load_state_dict( madar_model.decoder.decoders[2].attentions[2].state_dict() )'''

            print(' - Decoders pre-initialized')
            sys.stdout.flush() 

        encoder = TarcMultiTaskEncoder(args, src_dict, encoders)
        decoder = TarcMultiTaskDecoder(args, tgt_dict, decoders)
        encoder.set_g_merging_flags_( task.get_granularity_merging_flags()[:len(encoders)] )
        decoder.set_g_merging_flags_( task.get_granularity_merging_flags()[len(encoders):] )

        return cls(args, encoder, decoder, tgt_dict)

    def __init__(self, args, encoder, decoder, tgt_dict):
        super().__init__(encoder, decoder)

        self.args = args
        self.dict = tgt_dict
        self.tmp_dict = {}
        self.num_of_tasks = len(decoder.decoders)
        self.sequence_separator = tgt_dict.add_symbol(args.sequence_separator)

        # Kept for possible future use
        self.teacher_forcing = True
        self.scheduled_sampling = False
    
    

    def set_scheduled_sampling(self, ss_val=False):
        self.scheduled_sampling = ss_val

    def switch_scheduled_sampling(self):
        self.scheduled_sampling = not self.scheduled_sampling
 
    # Kept for possible future use
    def _decode(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):

        return None, None

    # We could override the ``forward()`` if we wanted more control over how
    # the encoder and decoder interact, but it's not necessary for this
    # tutorial since we can inherit the default implementation provided by
    # the FairseqEncoderDecoderModel base class, which looks like:
    #
    def forward(self, src_tokens, src_lengths, prev_output_tokens, shapes=None, src_tok_bounds=None, sort_order=None, tgt_tok_bounds=None, incremental_state=None, **kwargs):
        """
            Returns:
                    - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                    - an extra dictionary with task specific content
        """

        encoder_out = self.encoder(src_tokens, src_lengths, src_tok_bounds=src_tok_bounds, sort_order=sort_order, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens,
            shapes=shapes,
            tgt_tok_bounds=tgt_tok_bounds,
            sort_order=sort_order,
            encoder_out=encoder_out,
            features_only=False,
            src_lengths=src_lengths,
            return_all_hiddens = False,
            incremental_state=incremental_state,
            **kwargs
        )

        return decoder_out

    def load_base_encoder(self, bsencoder): 

        base_encoder = torch.load( bsencoder )
        self.encoder.encoder.convolutions.load_state_dict( base_encoder.encoder.convolutions.state_dict() )
        for param in self.encoder.encoder.convolutions.parameters():
            param.requires_grad = False
        self.encoder.encoder.rnns.load_state_dict( base_encoder.encoder.rnns.state_dict() )
        for param in self.encoder.encoder.rnns.parameters():
            param.requires_grad = False

    def load_fairseq_encoder(self, fsencoder, task):
        
        models, _model_args = checkpoint_utils.load_model_ensemble(
            [fsencoder], 
            task=task,
        )
        fs_model = models[0]
        self.encoder.load_state_dict( fs_model.encoder.state_dict() )
        for param in self.encoder.parameters():
            param.requires_grad = False


### REGISTER THE MODEL FOR USING IN FAIRSEQ ###
from fairseq.models import register_model_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'simple_lstm'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('tarc_multitask_model', 'tarc_multitask_arch')
def tarc_multitask_arch(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
    args.load_encoder = getattr(args, 'load_encoder', 'None')
    args.encoder_transformer_layers = getattr(args, 'encoder_transformer_layers', False)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.double_learning = getattr(args, 'double_learning', False)

    args.model_type = getattr(args, 'model_type')
    if args.model_type == 'transformer':
        trans_ba(args)
    elif args.model_type == 'lstm':
        lstm_ba(args)
    else:
        raise NotImplementedError

