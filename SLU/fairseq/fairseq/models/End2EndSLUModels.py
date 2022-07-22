# Code for adding the architecture for End2End SLU to Fairseq

import sys
import numpy as np
import math
import itertools
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from fairseq.globals import *
from fairseq import utils, init_functions
from fairseq import checkpoint_utils, options, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture, 
)

#from fairseq.models.slu_models import BasicEncoder, BasicSpeechEncoder, BasicSpeechSeqEncoder, BasicSpeechBiseqEncoder, MLSpeechEncoder, MLSpeechSeqEncoder, SLUSimpleDecoder, SLUBiDecoder

from fairseq.models.transformer import TransformerModel, base_architecture as trans_ba
#from fairseq.models.ctc_transformer import CTCTransformerDecoder
from fairseq.models.lstm import LSTMModel, LSTMDecoder, base_architecture as lstm_ba

# Importing for compatibilities with the Transformer model
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from torch import Tensor

class Conv1dNormWrapper(nn.Module):
    '''
        class Conv1dNormWrapper
        
        Wrap a Conv1d class to be used in a nn.Sequential module, adding a layer normalization module.
        '''
    
    def __init__(self, input_size, output_size, kernel, stride_factor):
        
        super(Conv1dNormWrapper,self).__init__()

        #self.cNorm = nn.LayerNorm(input_size)
        self.conv = init_functions.LSTMConv1d(input_size, output_size, kernel, stride=stride_factor)
        self.cNorm = nn.LayerNorm( output_size )
    
    def forward(self, input):

        # L x B x C -> B x C x L for convolution; B x C x L -> L x B x C back to input shape
        #x = self.cNorm( input ).permute(1, 2, 0)
        x = self.conv( input.permute(1, 2, 0) ).permute(2, 0, 1)
        x = self.cNorm( x )
        return x

class LSTMWrapper(nn.Module):
    '''
        LSTMWrapper
        
        Wrap a LSTM layer to be used in a nn.Sequential module.
        '''
    
    def __init__(self, input_size, output_size, bidirFlag, drop_ratio):
        
        super(LSTMWrapper,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.drop_ratio = drop_ratio
        #self.lstm_norm = nn.LayerNorm(input_size)

        #print(' *** LSTMWrapper, adding LSTM of size: {} x {}'.format(input_size, output_size))
        #sys.stdout.flush()

        self.lstm = init_functions.SLULSTM(input_size, output_size, bidirectional=bidirFlag)
        norm_size = output_size * 2 if bidirFlag else output_size
        self.lstm_norm = nn.LayerNorm(norm_size)

        self.fc1 = None
        '''init_functions.LSTMLinear(input_size, 4*output_size) if input_size == output_size and bidirFlag else None
        self.fc2 = init_functions.LSTMLinear(4*output_size, 2*output_size) if input_size == output_size and bidirFlag else None'''

        self.fc3 = None
        '''init_functions.LSTMLinear(2*output_size, 4*output_size) if input_size // 2 == output_size and bidirFlag else None
        self.fc4 = init_functions.LSTMLinear(4*output_size, 2*output_size) if input_size // 2 == output_size and bidirFlag else None'''

        class BogusClass :
            name = 'BogusClass'
        args = BogusClass()

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "tanh")
        ) if self.fc1 is not None or self.fc3 is not None else None

    def forward(self, input):

        #output = self.lstm_norm( input )
        output, _ = self.lstm( input )  # 1. "Transform"
        #if self.fc3 is None:
        output = self.lstm_norm( output )
        #if self.input_size == self.output_size:
        #    return F.dropout(output, p=self.drop_ratio, training=self.training) + input # 2. & 3. Drop & Residual add
        #else:

        '''if self.fc1 is not None: 
            residual = self.activation_fn(self.fc1(input))
            #zero_tsr = torch.zeros_like(residual)
            #residual = torch.max(zero_tsr, residual)
            residual = F.dropout(residual, p=self.drop_ratio, training=self.training)
            residual = self.fc2(residual)
            residual = F.dropout(residual, p=self.drop_ratio, training=self.training)
            #output = F.dropout(output, p=self.drop_ratio, training=self.training) + residual
            output = output + residual'''
        #else:
        output = F.dropout(output, p=self.drop_ratio, training=self.training)

        '''if self.fc3 is not None:
            # 1. Store residual value
            residual = output
            # 2. Normalize
            output = self.lstm_norm(output)
            # 3. Transform
            output = self.activation_fn(self.fc3(output))
            #zero_tsr = torch.zeros_like(output)
            #output = torch.max(zero_tsr, output)
            output = F.dropout(output, p=self.drop_ratio, training=self.training)
            output = self.fc4(output)
            # 4. Drop
            output = F.dropout(output, p=self.drop_ratio, training=self.training)
            # 5. Residual add
            output = output + residual'''

        return output

class FFNN(nn.Module):

    def __init__(self, input_size, output_size, dropout_ratio=0.5):

        super(FFNN,self).__init__()

        self.dropout_ratio = dropout_ratio
        self.activation_fn = utils.get_activation_fn('relu')
        self.fc1 = init_functions.LSTMLinear(input_size, 2*output_size)
        self.fc2 = init_functions.LSTMLinear(2*output_size, output_size)

    def forward(self, x):

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = self.fc2(x)
        return x

class SLUAttentionLayer(nn.Module):
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
            #print(' * SLUAttentionLayer, filling attn_scores with -1e8 (instead of float(\'-inf\')) at mask')
            #sys.stdout.flush()
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                -1e8
            ).type_as(attn_scores)  # FP16 support: cast to float and back 
        
        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz 

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0) # bsz x source_embed_dim
        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))  # bsz x output_embed_dim 

        return x, attn_scores

class PyramidalRNNEncoder(nn.Module):

    def __init__(self, params, dictionary):

        super(PyramidalRNNEncoder,self).__init__()

        self.params = params
        bidirectional = True
        self.kheops_base_height = 1
        self.kheops_hidd_height = params.pyramid_hidden_layers
        self.kheops_ptop_height = 1
        self.min_output_length = 15

        self.layers = nn.ModuleList() 
        self.norms = nn.ModuleList()
        for i in range(self.params.num_lstm_layers):
            input_size = 2*self.params.speech_lstm_size
            nlayers = 1
            if i == 0:
                input_size = self.params.num_features   # TODO: replace num_features with speech_conv_size if the convolution is added to the encoder.
                nlayers = self.kheops_base_height
            elif i < self.params.num_lstm_layers-1:
                nlayers = self.kheops_hidd_height
            else:
                nlayers = self.kheops_ptop_height
 
            self.layers.append( init_functions.SLULSTM(input_size, self.params.speech_lstm_size, num_layers=nlayers, bidirectional=bidirectional) ) # TODO: Try adding in LSTM constructor dropout=self.params.drop_ratio
            self.norms.append( nn.LayerNorm( 2*self.params.speech_lstm_size ) )

        self.encoder_output_units = self.params.speech_lstm_size
        if bidirectional:
            self.encoder_output_units = 2 * self.encoder_output_units 

    def forward(self, x):
        # x is expected as Length x Batch-size x Num. of features, that is T x B x C

        T, B, C = x.size()
        '''if self.params.character_level_slu:
            idxs = torch.LongTensor( [(i,i) for i in range(T)] )
            x = x[idxs,:,:].view(-1,B,C)

            #print(' New input tensor size: {} (old {})'.format(x.size(), T))
            #sys.exit(0)'''
        if T % 2 != 0: 
            x = torch.cat( [x,torch.zeros(1,B,C).to(x)], 0 )
            T, B, C = x.size() 

        H = x
        for i in range(len(self.layers)):
            H, _ = self.layers[i](H)

            if i < len(self.layers)-1: 
                T, B, h = H.size()
                indices = torch.LongTensor( [(i-1,i) for i in range(1,T,2)] ).to(H.device) 
                H = torch.mean(H[indices], 1) 

            H = self.norms[i](H)
            H = F.dropout(H, p=self.params.drop_ratio, training=self.training)
            
        if H.size(0) < self.min_output_length:
            #min_len = self.min_output_length +2
            #ratio = H.size(0) / (self.min_output_length+2)
            idxs = [int(i * (H.size(0) / (self.min_output_length+2))) for i in range(1, self.min_output_length+1)]    # We exclude first and last frame which may encode just the speaker marker.
            H = H[idxs,:,:]

        return H

class TranscriptionEncoder(nn.Module):

    def __init__(self, args, dictionary):

        super(TranscriptionEncoder, self).__init__()

        self.args = args
        self.dictionary = dictionary
        num_embeddings = len(dictionary)
        embed_dim = self.args.decoder_embed_dim
        self.embed_tokens = init_functions.LSTMEmbedding(num_embeddings, embed_dim, self.dictionary.pad())
        recurrent_layers = []
        for i in range(self.args.num_lstm_layers):
            input_size = 2*self.args.speech_lstm_size
            if i == 0:
                input_size = embed_dim
            recurrent_layers.append( ('LSTM'+str(i+1), LSTMWrapper(input_size, self.args.speech_lstm_size, True, self.args.drop_ratio)) )
        self.rnns = nn.Sequential( OrderedDict(recurrent_layers) )

    def forward(self, x):

        x = self.embed_tokens(x)
        x = F.dropout(x, p=self.args.drop_ratio, training=self.training)
        x = x.transpose(0, 1)
        x = self.rnns(x)

        return x


class DeepSpeechLikeEncoder(nn.Module):
    
    def __init__(self, params):
        
        super(DeepSpeechLikeEncoder,self).__init__()
        #self.window_size = params.window_size
        
        # Parameter initialization
        self.params = params
        bidirectional = True
        
        # Layer initialization
        # 1. Convolutions
        conv_layers = []
        for i in range(self.params.speech_conv):
            conv_stride = 2
            if i == self.params.speech_conv-1:
                conv_stride = 2
            input_size = self.params.speech_conv_size
            if i == 0:
                input_size = self.params.num_features 
            conv_layers.append( ('Conv'+str(i+1), Conv1dNormWrapper(input_size, self.params.speech_conv_size, self.params.conv_kernel, conv_stride)) )
            conv_layers.append( ('ConvDropout'+str(i+1), nn.Dropout(p=params.drop_ratio)) )
        self.convolutions = nn.Sequential( OrderedDict(conv_layers) )

        lstm_size = 2*self.params.speech_lstm_size
        #self.conv2lstm = init_functions.LSTMLinear(self.params.speech_conv_size, lstm_size, bias=False)

        # 2. Recurrent layers
        recurrent_layers = [] 
        for i in range(self.params.num_lstm_layers):
            input_size = lstm_size
            if i == 0:
                input_size = self.params.speech_conv_size 
            recurrent_layers.append( ('LSTM'+str(i+1), LSTMWrapper(input_size, self.params.speech_lstm_size, bidirectional, self.params.drop_ratio)) )
            #recurrent_layers.append( ('ConvNorm'+str(i+1), nn.LayerNorm( 2*self.params.speech_lstm_size )) )
            #recurrent_layers.append( ('LSTMDropout'+str(i+1), nn.Dropout(p=self.params.drop_ratio)) )
        self.rnns = nn.Sequential( OrderedDict(recurrent_layers) )
            
        #Linear Layer
        #self.linear_layer = init_functions.LSTMLinear(2*self.params.speech_lstm_size, self.params.output_size)

        self.encoder_output_units = self.params.speech_lstm_size
        if bidirectional:
            self.encoder_output_units = 2 * self.encoder_output_units

    def forward(self, x):
        # Input has shape (sequence_length, batch_size, num. of channels), that is (L, N, C), convolution needs it to be (N, C, L)

        conv_state = self.convolutions( x )
        #x = self.conv2lstm(conv_state)
        hidden_state = self.rnns( conv_state )
        #output = self.linear_layer( hidden_state )

        return (conv_state, hidden_state)

def create_lstm_final_states_(decoder_layers, hidden_state):
    
    if hidden_state is None:
        return (None, None)
    else:
        final_hiddens = torch.stack( [hidden_state[-1].clone() for i in range(decoder_layers)], 0 )
        final_cells = torch.stack( [hidden_state[-1].clone() for i in range(decoder_layers)], 0 )
        return (final_hiddens, final_cells)

### ENCODER ###
class End2EndSLUEncoder(FairseqEncoder):
    
    def __init__(
                 self, args, dictionary,
                 ):
        
        super().__init__(dictionary)
        self.args = args
        self.padding_idx = dictionary.pad_index
        self.blank_idx = dictionary.blank() 

        if hasattr(self.args, 'use_transcription_as_input') and self.args.use_transcription_as_input:
            self.trs_encoder = TranscriptionEncoder(args, dictionary)

        if hasattr(args, 'speech_encoder') and args.speech_encoder == 'deep-speech':
            self.encoder = DeepSpeechLikeEncoder(args)
            self.se_type = 'ds'
        elif hasattr(args, 'speech_encoder') and args.speech_encoder == 'ziggurat':
            self.encoder = PyramidalRNNEncoder(args, dictionary)
            self.se_type = 'zg'
        elif not hasattr(args, 'speech_encoder'):
            self.encoder = PyramidalRNNEncoder(args, dictionary)
            self.se_type = 'zg'
        else:
            raise NotImplementedError('{} speech encoder not defined'.format(args.speech_encoder if hasattr(args, 'speech_encoder') else 'unknown'))

        self.encoder_output_units = self.encoder.encoder_output_units  # NEW ARCHITECTURE FOR COMBINED LOSS: the current output is the projection to the output dict, unless we are giving hidden states to the decoder instead of output projections
        self.return_all_hiddens = False
        self.speakers = None

        # NEW ARCHITECTURE FOR COMBINED LOSS
        #self.output_projection = init_functions.LSTMLinear(args.encoder_hidden_dim*2, len(dictionary))
        #self.output_norm = nn.LayerNorm(len(dictionary))

    def set_turn_speaker(self, val):
        self.speakers = val
        return self.speakers

    def conv_padding_mask(self, t, c, stride):

        assert stride == 2, 'padding convolution not implemented for stride other than 2'

        '''print(' * conv_padding_mask input:')
        print('   - t: {}'.format(t.size()))
        print('   - c: {}'.format(c.size()))
        print(' *****')'''

        (T1, B, C1) = t.size()
        (T2, B, C2) = c.size()
        t_padding = torch.zeros(T1, B) 
        for i in range(B):
            tmp = torch.sum(t[:,i,:], -1)
            t_padding[:,i] = (tmp == 0)
        #print(' * conv_padding_mask, t_padding: {}'.format(t_padding))
        c_padding = torch.zeros(T2, B)
        for j in range(B):
            c_idx = 0
            for i in range(0,T1,stride):
                if c_idx < T2:
                    c_padding[c_idx,j] = (t_padding[i-1,j] and t_padding[i,j])
                c_idx += 1
        return c_padding

    def forward(self, src_signals, src_tokens, src_lengths, src_tok_lengths):
        # B x T x C -> T x B x C 

        trs_x = None
        trs_mask = None
        if hasattr(self, 'trs_encoder'):
            #src_sig, src_trs = src_tokens
            trs_x = self.trs_encoder(src_tokens)
            x = src_signals.transpose(0, 1)
            trs_mask = src_tokens.eq(self.dictionary.pad()).transpose(0, 1)
        else:
            x = src_tokens.transpose(0,1) 

        if self.se_type == 'ds':
            (conv_states, hidden_states) = self.encoder( x )
        else:
            hidden_states = self.encoder( x ) 
        (src_len, bsz, dim) = hidden_states.size() #conv_states.size() 

        # TODO: uncomment this to debug dialog-level SLU, batch size should be the same as long as a whole dialog (or a batch of dialogs having the same length) is processed.
        #print('[DEBUG] End2EndSLUEncoder: current batch size {}'.format(bsz))
        #sys.stdout.flush()

        # T x B x C -> B x T x C
        # NEW ARCHITECTURE FOR COMBINED LOSS: these operations have been moved below (and correct!)
        encoder_embedding = hidden_states.transpose(0,1) #conv_states.transpose(0,1)
        # NOTE: input is speech signal (spectrogram frames, or wav2vec features), padding is zero, i.e. there's no padding from a NLP point of view...
        encoder_padding = torch.LongTensor(bsz, src_len).fill_(self.padding_idx+13).to(src_tokens.device)
        encoder_padding_mask = encoder_padding.eq(self.padding_idx)

        encoder_states = [] if self.return_all_hiddens else None

        x = hidden_states

        # NEW ARCHITECTURE FOR COMBINED LOSS
        #x = self.output_projection(x)
        #x = self.output_norm(x)
        #scores = F.softmax(x, -1)
        #_, preds = torch.max(scores, -1)
        #encoder_embedding = x.transpose(0, 1)
        #encoder_padding_mask = preds.eq(self.blank_idx)

        encoder_out = x

        # Return the Encoder's output. This can be any object and will be
        # passed directly to the Decoder.
        if self.args.decoder in ['basic', 'transformer']:
            return EncoderOut(
                encoder_out=encoder_out,   # T x B x C
                encoder_padding_mask=encoder_padding_mask,    # B x T   # NEW ARCHITECTURE FOR COMBINED LOSS: this mask must be transposed with the new architecture
                encoder_embedding=encoder_embedding,    # B x T x C
                encoder_states=encoder_states, # List[T x B x C]
            )
        else: 
            final_hiddens, final_cells = create_lstm_final_states_(self.args.decoder_layers, encoder_out)

            return {
                    'encoder_out': (encoder_out, final_hiddens, final_cells),
                    'encoder_padding_mask': None,    # NEW ARCHITECTURE FOR COMBINED LOSS: set this back to None to come back to previous architecture.
                    'trs_out': trs_x,
                    'trs_mask': trs_mask,
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
        
        new_encoder_out: Dict[str, Tensor] = {}

        if self.args.decoder in ['basic', 'transformer']:
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
        elif self.args.decoder in ['lstm', 'ctclstm', 'icassp_lstm', 'deliberation']:
            hidden_states = encoder_out['encoder_out'][0]
            new_hidden_states = hidden_states if hidden_states is None else hidden_states.index_select(1, new_order)
            new_final_h, new_final_c = create_lstm_final_states_(self.args.decoder_layers, new_hidden_states)
            trs_x = encoder_out['trs_out']
            if trs_x is not None:
                trs_x = trs_x.index_select(1, new_order)
            trs_mask = encoder_out['trs_mask']
            if trs_mask is not None:
                trs_mask = trs_mask.index_select(1, new_order)

            return {
                    'encoder_out': (new_hidden_states, new_final_h, new_final_c),
                    'encoder_padding_mask': None,
                    'trs_out': trs_x,
                    'trs_mask': trs_mask
            }
        elif self.args.decoder == 'end2end_slu':
            # TODO: add the branch for the transformer as above
            hidden_states = encoder_out['encoder_out'][0]
            new_hidden_states = hidden_states if hidden_states is None else hidden_states.index_select(1, new_order)
            new_final_h, new_final_c = create_lstm_final_states_(self.args.decoder_layers, new_hidden_states)

            return {
                    'encoder_out': (new_hidden_states, new_final_h, new_final_c),
                    'encoder_padding_mask': None
            }
        else:
            raise NotImplementedError

### SIMPLE DECODER (for training the ICASSP2020 paper's Basic models in Fairseq) ###
from fairseq.models import FairseqDecoder

class BasicDecoder(FairseqIncrementalDecoder):
    '''
        BasicDecoder: just a linear layer mapping hidden states to output dictionary.
            This is helpful to pre-train the encoder like in the ICASSP2020 paper's system

        NOTE:   Since the output is produced only using a linear layer, this decoder generates an output for every speech frame.
                It does not make sense thus using the cross-entropy loss like for other decoders, only the CTC loss makes sense.
                For these reasons also, the --match-source-len option must be used with fairseq-generate.
    '''
    
    def __init__(
            self, dictionary, hidden_dim=512,
            ):
        super().__init__(dictionary)

        self.speaker = None

        # Define the output projection.
        self.is_didx_key = 'decoding_idx'
        #self.input_norm = nn.LayerNorm(hidden_dim)

        # NEW ARCHITECTURE FOR COMBINED LOSS: these components have been moved in the encoder
        self.output_projection = init_functions.LSTMLinear(hidden_dim, len(dictionary))
        self.output_norm = nn.LayerNorm(len(dictionary))

    def set_turn_speaker(self, val):
        self.speaker = val
        return self.speaker

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **kwargs): 

        dec_idx = 0
        T, B, C = encoder_out.encoder_out.size()
        if incremental_state is not None:
            dec_state = utils.get_incremental_state(self, incremental_state, self.is_didx_key)
            if dec_state is not None:
                dec_idx = dec_state

        x = encoder_out.encoder_out 

        '''bsz, seqlen = prev_output_tokens.size()
        srclen = x.size(0)
        if srclen < seqlen:
            print(' - BasicDecoder: TARGET LEN MISS!')
            sys.stdout.flush()'''

        # Project the outputs to the size of the vocabulary. 
        # NEW ARCHITECTURE FOR COMBINED LOSS: these operations have been moved into the encoder
        x = self.output_projection(x)
        x = self.output_norm(x) 

        utils.set_incremental_state(self, incremental_state, self.is_didx_key, dec_idx+1)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        #print(' - forward, got input of shape: {}'.format(x.size()))
        if incremental_state is not None:
            #print(' - forward, decoding at position {}'.format(dec_idx))
            #sys.stdout.flush()
            if dec_idx >= x.size(1):
                print(' - forward, binding decoding position at {}'.format(x.size(1)-1))
                sys.stdout.flush()

                dec_idx = x.size(1)-1   # sequence_generator can decode for max_len+1 steps, it may thus exceed the source length
            x = x[:,dec_idx,:].view(B,1,-1)

        # Return the logits and ``None`` for the attention weights 
        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):

        return


class ICASSPDecoder(FairseqIncrementalDecoder):

    def __init__(
        self, dictionary, args, embeddings=None,
    ):
        
        super().__init__(dictionary)

        self.args = args
        self.share_input_output_embed = args.share_decoder_input_output_embed
        self.adaptive_softmax = None    # TODO: add this in the future, it may improve results.
        self.io_size_ratio = args.io_size_ratio
        self.need_attn = True
        self.max_target_positions = args.max_target_positions
        self.encoder_output_size = args.encoder_hidden_dim
        if args.bidirectional_encoder:
            self.encoder_output_size *= 2

        if embeddings is not None:
            self.dec_embeddings = embeddings
        else:
            self.dec_embeddings = init_functions.LSTMEmbedding(
                num_embeddings=len(dictionary),
                embedding_dim=args.decoder_embed_dim,
                padding_idx=dictionary.pad(),
            )

        #print(' - decoder embeddings: {} x {}'.format(args.decoder_embed_dim, len(dictionary)))

        mapping_size = args.decoder_embed_dim + args.decoder_hidden_dim
        if mapping_size != args.decoder_hidden_dim:
            self.input_map = init_functions.LSTMLinear(mapping_size, args.decoder_hidden_dim, bias=False) 
        else:
            self.input_map = None
        if self.encoder_output_size != args.decoder_hidden_dim:
            self.input_map_h = init_functions.LSTMLinear(self.encoder_output_size, args.decoder_hidden_dim, bias=False)
            self.input_map_c = init_functions.LSTMLinear(self.encoder_output_size, args.decoder_hidden_dim, bias=False)
        else:
            self.input_map_h = None
            self.input_map_c = None

        #print(' - input_map: {} x {}'.format(mapping_size, args.decoder_hidden_dim))

        self.dropout = nn.Dropout(p=args.dropout)

        self.layers = nn.ModuleList([
            init_functions.SLULSTMCell(
                input_size=args.decoder_hidden_dim,
                hidden_size=args.decoder_hidden_dim,
            )
            for layer in range(args.decoder_layers)
        ])

        #print(' - LSTM cells and Norm layers: {}'.format(args.decoder_hidden_dim))

        self.rnn_norm = nn.ModuleList([
            nn.LayerNorm(args.decoder_hidden_dim)
            for layer in range(args.decoder_layers)
        ])

        self.attention = SLUAttentionLayer(args.decoder_hidden_dim, self.encoder_output_size, args.decoder_hidden_dim, bias=False)
        self.outs_norm = nn.LayerNorm(args.decoder_hidden_dim)

        # Linear layers to gate attention vectors and hidden state (instead of just using attention vectors)
        self.att_gate_x = init_functions.LSTMLinear(args.decoder_hidden_dim, args.decoder_hidden_dim, bias=False)
        self.att_gate_y = init_functions.LSTMLinear(args.decoder_hidden_dim, args.decoder_hidden_dim, bias=False)

        if args.decoder_hidden_dim != args.decoder_embed_dim:
            self.additional_fc = init_functions.LSTMLinear(args.decoder_hidden_dim, args.decoder_embed_dim)
        if not self.share_input_output_embed:
            self.fc_out = init_functions.LSTMLinear(args.decoder_embed_dim, len(dictionary), dropout=args.dropout)

        self.speaker = None

    def set_turn_speaker(self, val):
        self.speaker = val
        return self.speaker

    def get_cached_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]):
        
        cached_state = self.get_incremental_state(incremental_state, 'cached_state')
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.args.decoder_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.args.decoder_layers)]
        input_feed = cached_state["input_feed"]
        assert input_feed is not None
        src_idx = cached_state["src_index"]
        return prev_hiddens, prev_cells, input_feed, src_idx

    def extract_features(
        self, prev_output_tokens, encoder_out: Dict[str, Tuple[Tensor, Tensor, Tensor]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None
    ):

        """
        Similar to *forward* but only return features.
        """

        # FOR DEBUGGING:
        #incremental_state = None

        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']
 
        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]        
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # Set some parameters for decoding
        src_idx = 0
        src_bin_size = int(float(srclen) / float(self.io_size_ratio)) + 1
        if self.training:
            src_bin_size = int(float(srclen) / float(seqlen)) + 1

        # embed tokens
        x = self.dec_embeddings(prev_output_tokens)
        #x = F.dropout(x, p=self.args.dropout, training=self.training)  # NOTE: decoder embeddings are dropped after the input_map liner layer.

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        #print(' - forward, embedding size: {}'.format(x.size()))

        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed, src_idx = self.get_cached_state(incremental_state)
        else: 
            # setup recurrent cells
            prev_hiddens = [encoder_hiddens[i] for i in range(self.args.decoder_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.args.decoder_layers)]
            if self.input_map_h is not None:
                prev_hiddens = [self.input_map_h(y) for y in prev_hiddens]
                prev_cells = [self.input_map_c(y) for y in prev_cells]
            input_feed = x.new_zeros(bsz, self.args.decoder_hidden_dim)

        assert srclen > 0, "encoder outputs are needed in LSTM decoder"

        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        inputs = []
        outs = []

        for j in range(srclen):
            # input feeding: concatenate context vector from previous time step 
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            if self.input_map is not None:
                input = self.input_map(input)
                input = F.dropout(input, p=self.args.dropout, training=self.training)

            for i, rnn in enumerate(self.layers):
                input = self.rnn_norm[i]( input )   # 1. Normalize

                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i])) # 2. "Transform"

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.args.dropout, training=self.training) + input  # 3. & 4. drop & residual add

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out, attn_scores[:, j, :] = self.attention(input, encoder_outs, encoder_padding_mask)   # The Fairseq version uses 'hidden' instead of 'input'
            out = F.dropout(out, p=self.args.dropout, training=self.training)

            # input feeding
            if input_feed is not None:
                input_feed = out

            # save final outputs
            inputs.append(input)
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        src_idx = src_idx + int(float(src_bin_size) / 2.0) + 1
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {"prev_hiddens": prev_hiddens_tensor, "prev_cells": prev_cells_tensor, "input_feed": input_feed, "src_index": src_idx})
        self.set_incremental_state(
            incremental_state, 'cached_state', cache_state
        )

        # collect outputs across time steps
        y = torch.cat(inputs, dim=0).view(seqlen, bsz, self.args.decoder_hidden_dim)
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.args.decoder_hidden_dim)
        # use gated attention vs. hidden state, instead of just attention, as final predictor representation
        alpha = F.softmax( self.att_gate_x(x) + self.att_gate_y(y), dim=2 )
        x = self.outs_norm( alpha * x + (1.0 - alpha) * y)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.args.dropout, training=self.training)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None
        return x, attn_scores


    def output_layer(self, x):
        """Project features to the vocabulary size."""

        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.dec_embeddings.weight)
            else:
                x = self.fc_out(x)
        return x

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs): 

        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def reorder_state(self, state: List[Tensor], new_order):
        return [state_i.index_select(0, new_order) for state_i in state]


    def reorder_incremental_state(self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]], new_order):
        if incremental_state is None or len(incremental_state) == 0:
            return

        prev_hiddens, prev_cells, input_feed, src_idx = self.get_cached_state(incremental_state)
        cached_state = (prev_hiddens, prev_cells, [input_feed])
        new_state = [self.reorder_state(state, new_order) for state in cached_state]
        prev_hiddens_tensor = torch.stack(new_state[0])
        prev_cells_tensor = torch.stack(new_state[1])
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {"prev_hiddens": prev_hiddens_tensor, "prev_cells": prev_cells_tensor, "input_feed": new_state[2][0], "src_index": src_idx})
        self.set_incremental_state(incremental_state, 'cached_state', cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class ICASSPDecoderEx(FairseqIncrementalDecoder):
    """ICASSPDecoderEx decoder. Similar to the decoder used in my ICASSP 2020 paper, but uses an attention on all previous predictions instead of only the previous prediction, and uses an attention on a bin of source element around the current decoding position, instead of on all source elements."""
    def __init__(
        self, args, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=1024
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        #print(' - ICASSPDecoderEx, dropout_in and dropout_out: {}, {}'.format(self.dropout_in, self.dropout_out))
        #sys.stdout.flush()

        self.hidden_size = hidden_size
        self.out_embed_dim = out_embed_dim
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.blank_idx = dictionary.blank()
        if args.scheduled_sampling:
            self.register_buffer('prev_output_tokens_h', torch.LongTensor(args.max_sentences, 4*max_target_positions).fill_(dictionary.pad()))
            self.prev_output_tokens_h.requires_grad = False
        self.start_ss = args.start_scheduled_sampling

        self.bin_ray = args.encoder_state_window
        self.declen = args.constrained_output_length if hasattr(args, 'constrained_output_length') else -1
        if self.declen != -1:
            print(' - ICASSPDecoderEx, decoding length constrained to {}'.format(self.declen))
            sys.stdout.flush()
        
        self.prev_pred_query = args.prev_prediction_query
        self.scheduled_sampling = False
        self.scheduled_sampling_updates = 0
        self.n_scheduled_sampling = 0
        self.scheduled_sampling_init = 0.90
        self.scheduled_sampling_p = 0.90

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = init_functions.LSTMEmbedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        self.pred_attention = SLUAttentionLayer(hidden_size, embed_dim, hidden_size, bias=False) # TODO: use this as input to the LSTM instead of x
        #self.pred_attention = SLUAttentionLayer(encoder_output_units, embed_dim, hidden_size, bias=False)

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.lstm_flag = True
        if self.lstm_flag:
            self.layers = nn.ModuleList([
                init_functions.SLULSTMCell(
                    input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                    hidden_size=hidden_size,
                )
                for layer in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
            nn.GRUCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        #for layer in range(num_layers):
        #    input_size = input_feed_size + embed_dim if layer == 0 else hidden_size 

        if attention:
            # TODO make bias configurable
            self.attention = SLUAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = init_functions.LSTMLinear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = init_functions.LSTMLinear(out_embed_dim, num_embeddings)

        self.speaker = None

    def set_turn_speaker(self, val):
        self.speaker = val
        return self.speaker

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def set_scheduled_sampling(self, val=False, epoch=1):
        self.scheduled_sampling = val
        if self.scheduled_sampling:
            self.scheduled_sampling_updates = epoch
            if self.scheduled_sampling_updates >= self.start_ss:
                self.scheduled_sampling_p = self.scheduled_sampling_init * (0.98**(epoch-self.start_ss))

        print(' - ICASSPDecoderEx: scheduled sampling set to {} (updates: {}; p: {})'.format(self.scheduled_sampling, self.scheduled_sampling_updates, self.scheduled_sampling_p))
        sys.stdout.flush()

    def n_scheduled_sampling_(self, val=0):
        old_val = self.n_scheduled_sampling
        self.n_scheduled_sampling = val
        return old_val

    def scheduled_sampling_output_(self, x):
        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training) 
        return x

    def scheduled_sampling_predict_(self, x):
        x = self.output_layer(x)
        scores = F.log_softmax(x, -1)

        #print(' - scheduled_sampling processing, x: {}; scores: {}'.format(x.size(), scores.size()))
        #sys.stdout.flush() 

        max_sc, pred = torch.max(scores, -1)
        return pred

    def filter_prev_output_tokens_(self, predictions, T, debug_flag=False):

        samples = []
        max_L = 0
        B, L = predictions.size()

        if debug_flag:
            print(' - predictions: {}'.format(predictions.size()))

        for i in range(B):
            if debug_flag:
                print(' - predictions[{}]: {}'.format(i, predictions[i,:]))

            t = torch.unique_consecutive(predictions[i,:T])

            if debug_flag:
                print(' - compacted predictions: {}'.format(t))

            t = t[t != self.blank_idx]
            t = t[t != self.padding_idx]

            if debug_flag:
                print(' - cleaned predictions: {}'.format(t))

            if max_L < t.size(0):
                max_L = t.size(0)
            samples.append( t )

        res = torch.LongTensor(B, max_L).fill_(self.padding_idx)
        for i in range(B):
            res[i,0:samples[i].size(0)] = samples[i]

        if debug_flag:
            print(' - result ({}): {}'.format(res.size(), res))
            sys.stdout.flush()

        res = res.to(predictions)
        x = self.embed_tokens(res)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = x.transpose(0, 1)
        padding_mask = res.eq(self.padding_idx).transpose(0, 1)

        return x, padding_mask

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            #prev_predictions = prev_output_tokens.clone()
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            srclen = encoder_outs.size(0)
        else:
            srclen = None

        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state 

        #bin_ray = 2
        enc_cached_state = utils.get_incremental_state(self, incremental_state, 'enc_cached_state')
        #print(' - blank token index: {}'.format(self.dictionary.blank()))
        if enc_cached_state is not None:
            enc_idx = enc_cached_state #, prev_predictions = enc_cached_state
            enc_idx = min(srclen-1,enc_idx) 
            low_bound = max(0,enc_idx-self.bin_ray)
            up_bound = min(enc_idx+self.bin_ray+1,srclen)
            att_encoder_outs = encoder_outs[low_bound:up_bound,:,:]
            att_encoder_mask = encoder_padding_mask[low_bound:up_bound,:,:] if encoder_padding_mask is not None else None
            encoder_outs = encoder_outs[enc_idx,:,:].view(1, bsz, -1)

            #print(' prev_predictions shape: {}'.format(prev_predictions.size()))
            #print(' prev_output_tokens shape: {}'.format(prev_output_tokens.size()))
            #sys.stdout.flush()

            prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 ) 
        elif incremental_state is not None:
            enc_idx = 0 
            low_bound = 0
            up_bound = min(2*self.bin_ray+1, srclen)
            att_encoder_outs = encoder_outs[low_bound:up_bound,:,:]
            att_encoder_mask = encoder_padding_mask[low_bound:up_bound,:,:] if encoder_padding_mask is not None else None
            encoder_outs = encoder_outs[enc_idx,:,:].view(1, bsz, -1) 
            prev_predictions = prev_output_tokens 
        else:
            enc_idx = -1
            prev_predictions = prev_output_tokens

        srclen = encoder_outs.size(0)
        prev_predictions_mask = prev_predictions.eq(self.padding_idx)
        prev_predictions_mask[prev_predictions == self.blank_idx] = True
        prev_predictions_mask = prev_predictions_mask.transpose(0, 1)
        utils.set_incremental_state(self, incremental_state, 'enc_cached_state', enc_idx+1) #, prev_predictions))

        # embed tokens
        x = self.embed_tokens(prev_predictions)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state
            prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 )
        elif encoder_out is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
            prev_predictions = prev_output_tokens
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None
            prev_predictions = prev_output_tokens

        assert srclen is not None or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, srclen, bsz) if self.attention is not None else None
        outs = []

        declen = srclen
        if incremental_state is None:
            if self.declen != -1:
                declen = self.declen

        #print(' *** bin_ray and declen: {}, {}'.format(self.bin_ray, declen))
        #print(' *** prev_predictions: {}'.format(prev_predictions))
        #sys.stdout.flush()
 
        ss_backup_val = self.scheduled_sampling
        if self.scheduled_sampling:
            p = random.random()
            if p < self.scheduled_sampling_p:
                self.scheduled_sampling = False
            else:
                self.n_scheduled_sampling += 1
                #print(' - ICASSPDecoderEx, using scheduled sampling ({})'.format(self.n_scheduled_sampling))
                #sys.stdout.flush()
        train_flag = incremental_state is None and self.scheduled_sampling
        if train_flag:
            scheduled_sampling_out = torch.zeros(declen, bsz, self.out_embed_dim).to(x)
            #scheduled_sampling_prev_lst = []
            #scheduled_sampling_prev = torch.LongTensor(bsz, 1).fill_(self.padding_idx).to(prev_predictions)
            #scheduled_sampling_prev[:,0] = self.dictionary.eos()
            #scheduled_sampling_prev_lst.append( scheduled_sampling_prev ) 
            self.prev_output_tokens_h.fill_(self.dictionary.pad())
            self.prev_output_tokens_h[:,0] = self.dictionary.eos()
        for j in range(declen):
            # input feeding: concatenate context vector from previous time step
            curr_pred = x
            curr_pred_mask = prev_predictions_mask
            if incremental_state is None:
                if self.scheduled_sampling:

                    #print(' - ICASSPDecoderEx: using scheduled sampling training...')
                    #sys.stdout.flush()

                    cacheB, cacheT = self.prev_output_tokens_h.size()
                    if cacheB != bsz:
                        print(' - ICASSPDecoderEx, reducing scheduled sampling predicted tokens cache: {} vs. {} (length: {} vs. {})'.format(cacheB, bsz, cacheT, seqlen))
                        sys.stdout.flush()

                    prev_predictions = self.prev_output_tokens_h[:bsz,:j+1].clone() #torch.cat( scheduled_sampling_prev_lst, 1 )
                    prev_predictions_mask = prev_predictions.eq(self.padding_idx)
                    prev_predictions_mask[prev_predictions == self.blank_idx] = True
                    prev_predictions_mask = prev_predictions_mask.transpose(0, 1)
                    x = self.embed_tokens(prev_predictions)
                    x = F.dropout(x, p=self.dropout_in, training=self.training)
                    x = x.transpose(0, 1)
                    tgt_idx = j+1
                else:
                    tgt_idx = int( j * (float(seqlen)/float(srclen)) ) + 1   # +1 because we use tgt_idx in a range (:tgt_idx), otherwise the element at position tgt_idx would not be used
                curr_pred = x[:tgt_idx,:,:]
                curr_pred_mask = prev_predictions_mask[:tgt_idx,:] 
           
            #print(' - ICASSPDecoderEx: input for prediction attention computed (shape: {}; {})'.format(curr_pred.size(), curr_pred_mask.size()))
            #sys.stdout.flush()

            if train_flag:
                cacheB, cacheT = self.prev_output_tokens_h.size()
                if cacheB != bsz:
                    print(' - ICASSPDecoderEx, attention input shapes:')
                    print('   - prev_hiddens[0]: {}'.format(prev_hiddens[0].size()))
                    print('   - curr_pred: {}'.format(curr_pred.size()))
                    print('   - curr_pred_mask: {}'.format(curr_pred_mask.size()))
                    sys.stdout.flush() 

            #curr_pred, curr_pred_mask = self.filter_prev_output_tokens_(prev_predictions, tgt_idx if incremental_state is None else j+1)
            query = prev_hiddens[self.prev_pred_query] #encoder_outs[j] if incremental_state is None else encoder_outs.view(bsz, -1)
            pred_attn, _ = self.pred_attention(query, curr_pred, curr_pred_mask)   # TODO: prev_hiddens[0] as query seems to work better.
            pred_attn = F.dropout(pred_attn, p=self.dropout_out, training=self.training)
            if input_feed is not None:
                input = torch.cat((pred_attn, input_feed), dim=1)
            else:
                input = pred_attn 

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                if self.lstm_flag:
                    hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                else:
                    hidden = rnn(input, prev_hiddens[i])
                    cell = hidden

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training) 

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                if incremental_state is None:
                    src_idx = j 
                    low_bound = max(src_idx-self.bin_ray,0)
                    up_bound = min(src_idx+self.bin_ray+1,srclen) if src_idx > 0 else min(2*src_idx+1, srclen)
                    att_encoder_mask = encoder_padding_mask[low_bound:up_bound,:,:] if encoder_padding_mask is not None else None
                    att_encoder_outs = encoder_outs[low_bound:up_bound,:,:]
                    '''if att_encoder_outs.size(0) < self.bin_ray*2:
                        att_encoder_outs = torch.cat( [att_encoder_outs, torch.zeros(self.bin_ray*2 - att_encoder_outs.size(0), att_encoder_outs.size(1), att_encoder_outs.size(2)).to(att_encoder_outs)] )
                        att_encoder_mask = torch.cat( [att_encoder_mask, torch.ones(self.bin_ray*2 - att_encoder_mask.size(0), att_encoder_mask.size(1), att_encoder_mask.size(2)).to(att_encoder_mask)] ) if encoder_padding_mask is not None else None
                    elif att_encoder_outs.size(0) != self.bin_ray*2:
                        raise ValueError'''

                out, _ = self.attention(hidden, att_encoder_outs, att_encoder_mask) 
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            if input_feed is not None:
                input_feed = out

            if train_flag:
                #print(' - ICASSPDecoderEx, computing predictions...')
                #sys.stdout.flush() 

                ss_x = self.scheduled_sampling_output_(out)
                ss_pred = self.scheduled_sampling_predict_(ss_x)

                #print(' - ICASSPDecoderEx, predictions computed: {}; {}'.format(ss_x.size(), ss_pred.size()))
                #sys.stdout.flush()

                scheduled_sampling_out[j,:,:] = ss_x
                #scheduled_sampling_prev_lst.append( ss_pred.view(bsz, -1) )
                self.prev_output_tokens_h[:,j+1] = ss_pred

                #print(' ====================')
                #sys.stdout.flush()
            else:
                # save final output
                outs.append(out)
 
        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed, prev_predictions),
        )

        if train_flag:
            x = scheduled_sampling_out
        else:
            # collect outputs across time steps
            x = torch.cat(outs, dim=0).view(declen, bsz, self.hidden_size)

        #print(' - ICASSPDecoderEx, final x shape: {}'.format(x.size()))
        #sys.stdout.flush()

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if not train_flag:
            if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        self.scheduled_sampling = ss_backup_val
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
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
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
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class End2EndSLUDecoder(FairseqIncrementalDecoder):
    "End2EndSLUDecoder: uses 2 ICASSPDecoderEx decoders, 1 for extracting concept boundaries, the other for extracting concepts"
    def __init__(
        self, dictionary, args, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=1024
    ):
        super().__init__(dictionary)

        self.embed_tokens = init_functions.LSTMEmbedding(len(dictionary), embed_dim, padding_idx=dictionary.pad())

        self.decoder_layers = num_layers
        self.boundary_decoder = ICASSPDecoderEx(
            args=args,
            dictionary=dictionary, 
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            out_embed_dim=out_embed_dim,
            num_layers=num_layers,
            dropout_in=dropout_in,
            dropout_out=dropout_out,
            attention=attention,
            encoder_output_units=encoder_output_units,
            pretrained_embed=self.embed_tokens,
            share_input_output_embed=share_input_output_embed,
            adaptive_softmax_cutoff=adaptive_softmax_cutoff, 
            max_target_positions=max_target_positions
        )

        self.blank_idx = dictionary.blank()
        self.start_concept = dictionary.index( slu_start_concept_mark )
        if self.start_concept == dictionary.unk():
            raise ValueError('Start concept symbol {} is expected to be defined in the dictinary'.format(slu_start_concept_mark))
        self.end_concept = dictionary.index( slu_end_concept_mark )
        if self.end_concept == dictionary.unk():
            raise ValueError('End concept symbol {} is expected to be defined in the dictinoary'.format(slu_end_concept_mark))
        self.hidden_size = 2*hidden_size
        self.output_dim = len(dictionary)
        self.boundary_attention = SLUAttentionLayer(embed_dim, self.output_dim, self.hidden_size, bias=False) 

        self.concept_decoder = ICASSPDecoderEx(
            args=args,
            dictionary=dictionary,
            embed_dim=embed_dim,
            hidden_size=hidden_size,
            out_embed_dim=out_embed_dim,
            num_layers=num_layers,
            dropout_in=dropout_in,
            dropout_out=dropout_out,
            attention=attention,
            encoder_output_units=encoder_output_units,
            pretrained_embed=self.embed_tokens,
            share_input_output_embed=share_input_output_embed,
            adaptive_softmax_cutoff=adaptive_softmax_cutoff, 
            max_target_positions=max_target_positions
        )

        self.speaker = None

    def set_turn_speaker(self, val):
        self.speaker = val
        return self.speaker

    def get_boundaries_(self, predictions, eoc):

        B, T = predictions.size()
        boundaries = [[] for i in range(B)]
        for i in range(B):
            #if not self.training:
            #print(' ### predictions[{}]: {}'.format(i, predictions[i]))
            #print('   === {}'.format(self.dictionary.string(predictions[i])))
            boundaries[i] = [j for j in range(T) if predictions[i,j] == eoc]
            
            #if not self.training:
            #print(' *** b[{}]: {}'.format(i, boundaries[i]))

        return boundaries

    def compute_concept_reps_(self, q, H, b):

        B, T, C = H.size()
        qB, qT = q.size()
        reps = [[] for i in range(B)]
        for i in range(B):
            #soc = 0
            #for j in range(len(b[i])):
            #for q_idx in range(qT):

            #if not self.training:
            #print('  *** creating chunk at boundaries: {} - {}'.format(soc, b[i][j]))

            #if b[i][j] > soc:

            #chunk = H[i, soc:b[i][j], :].unsqueeze(1) # chunk: T x B x C; B == 1 since we loop over every single chunk
            chunk = H[i,:,:].unsqueeze(1)

            #if not self.training:
            #print('  *** chunk shape: {}'.format(chunk.size()))
            #print('  *** query shape: {}'.format(chunk[0,:,:].size()))
            sys.stdout.flush()

            chunk_mask = None #chunk.eq(self.blank_idx)

            #print('  *** chunk_mask shape: {}'.format(chunk_mask.size()))
            #sys.stdout.flush()

            #print('  *** unfiltered querires: {}'.format(q[i]))

            queries = q[i, q[i,:] != self.dictionary.pad()]

            #print('  *** filtered queries: {}'.format(queries))

            #r = [self.boundary_attention(chunk[k,:,:], chunk, chunk_mask) for k in range(chunk.size(0))]

            #print('  *** one query shape: {}'.format( self.embed_tokens(queries[0]).size() ))
            #sys.stdout.flush()
            #sys.exit(0)

            r = [self.boundary_attention(self.embed_tokens(q_idx).view(1,-1), chunk, chunk_mask) for q_idx in queries]
            #r = torch.sum( torch.stack( [ri[0] for ri in r], 0 ), 0 )   # TODO: maybe there is a better way to compute the final representation of a chunk...
                
            #if not self.training:
            #print('  *** attention r shape: {}'.format(r.size()))
            #sys.stdout.flush()

            reps[i] = [ri[0].squeeze() for ri in r] #.append(r.squeeze())
            #soc = b[i][j]+1

            #print('  *** -----')
            #sys.stdout.flush()

            if len(reps[i]) == 0:   # NOTE: At test phase predictions can be messy and yield empty chunks
                reps[i] = [torch.zeros(self.hidden_size).to(H)]

            #print(' ==========')
            #sys.stdout.flush()

        '''print(' - Intermediate representations 1:')
        for i in range(B):
            print('    - {}'.format(len(reps[i])))
        sys.stdout.flush()'''

        for i in range(B):
            reps[i] = torch.stack(reps[i], 0)
            #print('  ***** stacked reps shape: {}'.format(reps[i].size()))
            #sys.stdout.flush()

        '''print(' - Intermediate representations 2:')
        for i in range(B):
            print('    - {}'.format(reps[i].size()))
        sys.stdout.flush()'''

        max_len = max([t.size(0) for t in reps])
        res = torch.zeros(B, max_len, self.hidden_size).to(H)

        #print('  *** res shape: {}'.format(res.size()))
        #sys.stdout.flush()

        for i in range(B):
            res[i,0:reps[i].size(0),:] = reps[i]

        #print(' - Final representations shape: {}'.format(res.size()))
        #sys.stdout.flush()

        return res

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        if incremental_state is not None:
            bds = utils.get_incremental_state(self, incremental_state, 'boundary-decoder-state')
            if bds is None:
                utils.set_incremental_state(self, incremental_state, 'boundary-decoder-state', {} )

            cds = utils.get_incremental_state(self, incremental_state, 'concept-decoder-state')
            if cds is None:
                utils.set_incremental_state(self, incremental_state, 'concept-decoder-state', {} )

        # TODO: manage 'prev_output_tokens' and concept_decoder (not needed) at generation phase

        #if not self.training:
        #print(' - prev_output_tokens shape: {}'.format(prev_output_tokens.size()))
        #sys.stdout.flush()

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bds = utils.get_incremental_state(self, incremental_state, 'boundary-decoder-state')
        b_outs, b_attn = self.boundary_decoder(prev_output_tokens, encoder_out=encoder_out, incremental_state=bds, **kwargs)
        if incremental_state is None:
            #print(' - getting boundaries from gold standard...')
            boundaries = self.get_boundaries_(prev_output_tokens, self.end_concept)
        else:
            b_scores = F.log_softmax(b_outs, -1)
            _, preds = torch.max(b_scores, -1)  # preds: B x T
            boundaries = self.get_boundaries_(preds, self.end_concept)

        if incremental_state is None:   # NOTE: we only need this at training phase to learn a good segmentation and tagging sequence
            concept_reps = self.compute_concept_reps_(prev_output_tokens, b_outs, boundaries)
            concept_reps = concept_reps.transpose(0,1)

            cr_T, cr_B, cr_C = concept_reps.size()
            if self.speaker is not None:
                assert len(self.speaker) == cr_B
                spk_shift = concept_reps.new_zeros(cr_T, cr_B, cr_C)
                for spk_idx in range(cr_B):
                    if self.speaker[spk_idx] == user_ID:
                        spk_val = self.spk_abs_val
                    elif self.speaker[spk_idx] == machine_ID:
                        spk_val = -1.0*self.spk_abs_val
                    else:
                        spk_val = -2.0*self.spk_abs_val
                    spk_shift[:,spk_idx,:] = spk_val
                concept_reps = spk_shift + concept_reps #torch.cat( [spk_pad, concept_reps, spk_pad], 0 )

            concepts, _ = extract_concepts(prev_output_tokens, self.dictionary.bos(), self.dictionary.eos(), self.dictionary.pad(), self.end_concept, move_trail=True) 

            final_hiddens, final_cells = create_lstm_final_states_(self.decoder_layers, concept_reps)
            encoder_out = {
                'encoder_out': (concept_reps, final_hiddens, final_cells),
                'encoder_padding_mask': None
            }

            cds = utils.get_incremental_state(self, incremental_state, 'concept-decoder-state')
            c_outs, c_attn = self.concept_decoder(concepts, encoder_out=encoder_out, incremental_state=cds, **kwargs)

            '''if self.training:
                assert prev_output_tokens.size(0) == b_outs.size(0), '{} vs. {}'.format(prev_output_tokens.size(0), b_outs.size(0))
                assert prev_output_tokens.size(1) == b_outs.size(1), '{} vs. {}'.format(prev_output_tokens.size(1), b_outs.size(1))
                assert concepts.size(0) == c_outs.size(0), '{} vs. {}'.format(concepts.size(0), c_outs.size(0))
                assert concepts.size(1) == c_outs.size(1), '{} vs. {}'.format(concepts.size(1), c_outs.size(1))'''
        else:
            c_attn = None
            c_outs = None

        return b_outs, {'attn': b_attn, 'c_attn': c_attn, 'c_outs': c_outs}

    def reorder_incremental_state(self, incremental_state, new_order):
        if incremental_state is not None:
            bds = utils.get_incremental_state(self, incremental_state, 'boundary-decoder-state')
            if bds is None:
                utils.set_incremental_state(self, incremental_state, 'boundary-decoder-state', {} )

            cds = utils.get_incremental_state(self, incremental_state, 'concept-decoder-state')
            if cds is None:
                utils.set_incremental_state(self, incremental_state, 'concept-decoder-state', {} )

        bds = utils.get_incremental_state(self, incremental_state, 'boundary-decoder-state')
        cds = utils.get_incremental_state(self, incremental_state, 'concept-decoder-state')
        self.boundary_decoder.reorder_incremental_state(bds, new_order)
        self.concept_decoder.reorder_incremental_state(cds, new_order)

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.boundary_decoder.make_generation_fast_(need_attn=need_attn, **kwargs)
        self.concept_decoder.make_generation_fast_(need_attn=need_attn, **kwargs)


class LSTMDecoderEx(FairseqIncrementalDecoder):
    """Like LSTM decoder, but it uses an attention on all previous predictions, instead of using only the previous prediction."""
    def __init__(
        self, dictionary, args, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=1024
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.out_embed_dim = out_embed_dim
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.args = args
        self.blank_idx = dictionary.blank()
        self.num_masked_tokens = 0

        self.use_dhistory = False
 
        self.scheduled_sampling = False
        self.scheduled_sampling_updates = 0
        self.n_scheduled_sampling = 0
        self.scheduled_sampling_init = 0.99
        self.scheduled_sampling_p = 0.99

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = init_functions.LSTMEmbedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        self.pred_attention = SLUAttentionLayer(hidden_size, embed_dim, hidden_size, bias=False)  # TODO: use this as input to the LSTM instead of x

        if (args.dialog_batches or args.dialog_level_slu) and args.use_dialog_history:
            self.dh_hidden_states = []
            self.dialog_history = [] 
            self.dialog_history_backup = [] # NOTE: this is needed at inference phase as the reorder_incremental_state function modifies the history in size along the batch dimension
            self.turn_completed = []
            self.finalized_sequences = []
            self.context_attention_weights = None
            self.context_attention = SLUAttentionLayer(hidden_size, out_embed_dim, hidden_size)
            self.hal = SLUAttentionLayer(hidden_size, hidden_size, hidden_size)

            if self.args.context_fusion == 'gating':
                self.context_gate_h = init_functions.LSTMLinear(hidden_size, hidden_size)
                self.context_gate_y = init_functions.LSTMLinear(hidden_size, hidden_size)
            if self.args.context_fusion not in ['sum', 'gating']:
                raise ValueError('SLUEnd2EndModel.LSTMDecoderEx: on of "sum" and "gating" is expected with --context-fusion option')

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList([
            init_functions.SLULSTMCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = SLUAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hasattr(self.args, 'use_transcription_as_input') and self.args.use_transcription_as_input:
            self.trs_attention = SLUAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
            #self.sig_gate = init_functions.LSTMLinear(hidden_size, hidden_size)
            #self.trs_gate = init_functions.LSTMLinear(hidden_size, hidden_size)
        if hidden_size != out_embed_dim:
            self.additional_fc = init_functions.LSTMLinear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = init_functions.LSTMLinear(out_embed_dim, num_embeddings)

        self.speaker = None

    def set_turn_speaker(self, val):
        self.speaker = val
        return self.speaker

    #def set_eos_flag(self, flag):
    #    self.eos_flag = flag 

    def use_dialog_history(self, value):
        self.use_dhistory = value and (self.args.dialog_batches or self.args.dialog_level_slu)

        if self.use_dhistory:
            print('[DEBUG] LSTMDecoderEx: instruction to use dialog history received!')
            sys.stdout.flush()

    def set_finalized_sequences(self, fh):
        self.finalized_sequences = fh

    def build_eos_idx(self):

        EOS_idx = torch.LongTensor(len(self.finalized_sequences), 2).fill_(0).to(self.embed_tokens.weight.device)
        #new_list = list(new_order.view(-1))
        idx = 0
        #found = False 
        for ii in self.finalized_sequences: 
            EOS_idx[idx,0] = ii
            idx += 1
        return EOS_idx

    def finalize_sequences(self, finalized_sequences): #, EOS_idx, curr_hidden_state):

        self.set_finalized_sequences( finalized_sequences )
        EOS_idx = self.build_eos_idx()
        curr_hidden_state = self.dh_hidden_states if hasattr(self, 'dh_hidden_states') else []

        skip_flag = False
        if len(curr_hidden_state) > 0:
            T, B, C = curr_hidden_state[0].size()
            scores = F.log_softmax( self.output_layer(curr_hidden_state[0]), -1 )
            _, preds = torch.max(scores, -1)
            skip_flag = len(curr_hidden_state) == 1 and torch.sum( preds == self.dictionary.eos() ).item() == B

        if len(curr_hidden_state) > 0 and not skip_flag:
            tmp_completed = []
            for i in range(EOS_idx.size(0)):
                idx = EOS_idx[i,0].item()
                orig_idx = idx
                #print('[DEBUG] current dimension index: {}'.format(idx))
                for tt in sorted(self.turn_completed, key=lambda tuple: tuple[0]):
                    #print('[DEBUG] reconstructing original index: {} vs. {}'.format(tt[0], orig_idx))
                    if tt[0] <= orig_idx:
                        orig_idx += 1
                #tmp_turn = torch.stack( [tsr[idx,:,:] for tsr in curr_hidden_state], 1 )
                tmp_completed.append( (orig_idx, torch.stack( [tsr[idx,:,:] for tsr in curr_hidden_state], 1 ).expand(self.args.model_beam, -1, -1)) ) # TODO: replace the 1 in expand with the beam size, as the --beam param doesn't seem to be passed to the model Namespace
                #self.turn_completed.append( (orig_idx, torch.stack( [tsr[idx,:,:] for tsr in curr_hidden_state], 1 ) ) )
                #print('[DEBUG] LSTMDecoderEx.finalize_sequences, stored completed turn {} @{} of shape: {}'.format(orig_idx, idx, tmp_turn.size()))
            self.turn_completed.extend( tmp_completed )
            if self.args.print_history_attn_weights and len(self.dialog_history) > 0:
                #print('[DEBUG]# Attention weights from turn shape: {}'.format(tmp_completed[0][1].size()))
                #sys.stdout.flush()
                assert len(tmp_completed) == 1
                assert len(self.context_attention_weights) <= len(self.dialog_history)
                t_scores = F.log_softmax( self.output_layer(tmp_completed[0][1].squeeze()), -1 )
                _, preds = torch.max(t_scores, -1)
                print('ATTN-SCORES:START')
                print('ATTN-SCORES:TARGET {}'.format(SOS_tag + ' ' + self.dictionary.string(preds) + ' ' + EOS_tag))
                for h_idx in range(len(self.context_attention_weights)):
                    s_scores = F.log_softmax( self.output_layer(self.dialog_history[h_idx].squeeze()), -1 )
                    _, preds = torch.max(s_scores, -1)
                    print('ATTN-SCORES:SOURCE {}'.format(SOS_tag + ' ' + self.dictionary.string(preds) + ' ' + EOS_tag))
                    self.context_attention_weights[h_idx] = torch.stack( self.context_attention_weights[h_idx], 1 )
                    #print('[DEBUG]# Attention weights @{} (shape: {}): {}'.format(h_idx, self.context_attention_weights[h_idx].size(), self.context_attention_weights[h_idx]))
                    print('ATTN-SCORES:WEIGHTS')
                    for j in range(self.context_attention_weights[h_idx].size(1)):
                        print( ' '.join([str(val) for val in self.context_attention_weights[h_idx][:, j].tolist()]) )
                    sys.stdout.flush()
                print('ATTN-WEIGHTS:END')
                sys.stdout.flush()
        self.finalized_sequences = []
        #print('[DEBUG] -----')

    def finalize_turn_batch(self, incremental_state):

        if len(self.finalized_sequences) > 0:

            #EOS_idx = self.build_eos_idx()
            curr_hidden_state = self.dh_hidden_states #utils.get_incremental_state(self, incremental_state, 'curr_hidden_state')
            max_len = max([tt[1].size(1) for tt in self.turn_completed]) if len(self.turn_completed) > 0 else 0 
            assert curr_hidden_state is not None and len(curr_hidden_state) >= max_len, 'curr_hidden_state is None ?: {}, or curr_hidden_state len: {}; max_len: {}'.format(curr_hidden_state is None, len(curr_hidden_state) if curr_hidden_state is not None else None, max_len)
            self.finalize_sequences( self.finalized_sequences ) #EOS_idx, curr_hidden_state)
            #self.finalized_sequences = []

        #curr_hidden_state = []
        if hasattr(self, 'turn_completed') and len(self.turn_completed) > 0:

            max_len = max([t[1].size(1) for t in self.turn_completed])
            max_bz = sum([t[1].size(0) for t in self.turn_completed])
            dialog_turn_batch = torch.zeros(max_len, max_bz, self.turn_completed[0][1].size(-1)).to(self.turn_completed[0][1])

            #print('[DEBUG] LSTMDecoderEx.finalize_turn_batch, initialized dialog_turn_batch of shape: {}'.format(dialog_turn_batch.size()))
            #sys.stdout.flush()

            for tt in self.turn_completed:
                #print('[DEBUG]   * LSTMDecoderEx.finalize_turn_batch, accessing dialog_turn_batch at: [:{}, {}, :]'.format(tt[1].size(1), tt[0]))
                #print('[DEBUG]     * Source tensor has shape: {}'.format(tt[1].size()))
                #sys.stdout.flush()

                dialog_turn_batch[:tt[1].size(1),tt[0]*self.args.model_beam:(tt[0]+1)*self.args.model_beam,:] = tt[1].expand(self.args.model_beam, -1, -1).transpose(0, 1) # TODO: same as for the 1 in the expand in the 'finalize_sequences' function. ALSO THE ONES MULTIPLYING THE tt[0] IN THE dialog_turn_batch TENSOR
            #self.dialog_history.append(dialog_turn_batch)
            self.dialog_history_backup.append( dialog_turn_batch )
            self.dialog_history = []
            for tb in self.dialog_history_backup:
                self.dialog_history.append( tb.clone() )
            self.turn_completed = []
            #self.last_order_state = None 
            #utils.set_incremental_state(self, incremental_state, 'curr_hidden_state', curr_hidden_state)
            self.dh_hidden_states = []
            self.context_attention_weights = None

        #print('[DEBUG] LSTMDecoderEx.finalize_turn_batch, end-of-turn @batch detected, hidden state reset')
        #print('[DEBUG] LSTMDecoderEx.finalize_turn_batch, dialog history size: {}'.format(len(self.dialog_history)))
        #sys.stdout.flush()
        #sys.exit(0)

        return [] #curr_hidden_state

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):

        # NOTE: this is for debuggin dialog-level SLU
        #print('[DEBUG] ******************************')
        #print('[DEBUG] LSTMDecoderEx batch@0: -----\n{}'.format(self.dictionary.string(prev_output_tokens[0,:])))
        B, T = prev_output_tokens.size()
        nEOD = torch.sum( prev_output_tokens == self.dictionary.index(EOD_tag)).item()
        if nEOD > 0 and nEOD < B and self.use_dhistory:
            nBogus = torch.sum( prev_output_tokens == self.dictionary.index(bogus_token) ).item()
            assert T == 3 and nEOD+nBogus == B, 'Dialog-level data organization has probably been corrupted'
            print('End2EndSLU FATAL ERROR: you probably recharged a previously run training and this corrupted dialog-level data organization (got {} end-of-dialog markers, expected {})'.format(nEOD, B))
            sys.exit(0)
            #print('[DEBUG] LSTMDecoderEx: detected {} EOD tags (batch size = {}; T = {})'.format(nEOD, B, T))
            #print('[DEBUG]    - complete batch: {}'.format(self.dictionary.string(prev_output_tokens)))
            #sys.stdout.flush()
        EOD_flag = nEOD == B and T == 3
        if EOD_flag and incremental_state is None:
            self.dialog_history = []
            #print('[DEBUG] LSTMDecoderEx: End-of-Dialog marker detected!')
        #print('[DEBUG] ==========')
        #print('[DEBUG] LSTMDecoderEx, current output turn length (batch size: {}): {}'.format(B, prev_output_tokens.size(1)))
        #print('[DEBUG] LSTMDecoderEx: dialog history size: {}'.format(len(self.dialog_history)))
        #sys.stdout.flush()
        #sys.exit(0)
        
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        if hasattr(self, 'dialog_history') and incremental_state is None:
            if not EOD_flag:
                self.dialog_history.append(x.clone().detach().transpose(0,1))
        elif hasattr(self, 'dialog_history'): 
            # 1. Recover current hidden state from the incremental state and add (to the list) x
            #curr_hidden_state = utils.get_incremental_state(self, incremental_state, 'curr_hidden_state')
            curr_hidden_state = self.dh_hidden_states
            if curr_hidden_state is None: 
                curr_hidden_state = []
            curr_hidden_state.append(x)
            # 2. Predict current tokens and...
            scores = F.log_softmax(self.output_layer(x), -1)
            _, preds = torch.max(scores, -1)
            B = preds.size(0)

            #print('[DEBUG] LSTMDecoderEx:')
            #print('[DEBUG]    * Internal predictions: {} (EOS index: {})'.format(preds, self.dictionary.eos()))
            #print('[DEBUG]    * Stored output state of shape: {}'.format(x.size()))
            #print('[DEBUG]    * curr_hidden_state size: {}'.format(len(curr_hidden_state)))
            #tmp_src_len = encoder_out['encoder_out'][0].size(0) if encoder_out is not None else -1
            #print('[DEBUG]    * source size: {}'.format(tmp_src_len))
            #print('[DEBUG]    * decoding size limit: {} x {} + {} = {}'.format(self.args.max_len_a, tmp_src_len, self.args.max_len_b, self.args.max_len_a * tmp_src_len + self.args.max_len_b))
            
            #print('[DEBUG]   * current batch size: {}'.format(B))
            #print('[DEBUG]   * Internal predictions (str): {}'.format(self.dictionary.string(preds)))
            #print('[DEBUG] ----------')
            #sys.stdout.flush()
            #sys.exit(0)
     
            self.dh_hidden_states = curr_hidden_state
            #utils.set_incremental_state(self, incremental_state, 'curr_hidden_state', curr_hidden_state)
            # 4. If EOD is predicted everywhere that's the end of a dialog: reset current hidden state and dialog history
            if torch.sum( preds == self.dictionary.index(EOD_tag) ).item() == B:
                self.dialog_history = []
                self.dialog_history_backup = []
                #utils.set_incremental_state(self, incremental_state, 'curr_hidden_state', [])
                self.dh_hidden_states = []

                #print('[DEBUG] LSTMDecoderEx: end-of-dialog mark detected!')
                #sys.stdout.flush()
        return self.output_layer(x), attn_scores

    def set_scheduled_sampling(self, val=False, epoch=1):
        self.scheduled_sampling = val
        if self.scheduled_sampling:
            self.scheduled_sampling_updates = epoch
            if self.scheduled_sampling_updates >= self.args.start_scheduled_sampling:
                self.scheduled_sampling_p = self.scheduled_sampling_init * (0.99**(epoch-self.args.start_scheduled_sampling))

        print(' - LSTMDecoderEx: scheduled sampling set to {} (updates: {}; p: {})'.format(self.scheduled_sampling, self.scheduled_sampling_updates, self.scheduled_sampling_p))
        sys.stdout.flush()

    def n_scheduled_sampling_(self, val=0):
        old_val = self.n_scheduled_sampling
        self.n_scheduled_sampling = val
        return old_val

    def scheduled_sampling_output_(self, x):
        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training) 
        return x

    def scheduled_sampling_predict_(self, x):
        x = self.output_layer(x)
        scores = F.log_softmax(x, -1)

        #print(' - scheduled_sampling processing, x: {}; scores: {}'.format(x.size(), scores.size()))
        #sys.stdout.flush() 

        max_sc, pred = torch.max(scores, -1)
        return pred

    def get_context_length(self):
        assert self.args.context_size >= self.args.context_first_turns
        context_limit = self.args.context_first_turns
        context_last = self.args.context_size - self.args.context_first_turns
        context_bound = 0
        if hasattr(self, 'dialog_history') and len(self.dialog_history) > 0:
            context_bound = min(len(self.dialog_history), context_limit)
            curr_last = 0
            while curr_last < context_last and context_bound+curr_last < len(self.dialog_history):
                curr_last += 1
            context_bound += curr_last 
            if self.context_attention_weights is None and self.args.print_history_attn_weights:
                self.context_attention_weights = [[] for i in range(context_bound)]
        return context_bound

    def get_dialog_context(self, context_bound):
        bounded_dialog_history = []
        if hasattr(self, 'dialog_history'):
            context_size = min(len(self.dialog_history), self.args.context_first_turns)
            bounded_dialog_history = self.dialog_history[:context_size]    # NOTE: first turns are always the most informative
            if context_bound > self.args.context_first_turns:
                bounded_dialog_history = bounded_dialog_history + self.dialog_history[-(context_bound-self.args.context_first_turns):]  # NOTE: plus some of the last turns
        return bounded_dialog_history

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        trs_x = encoder_out['trs_out']
        trs_mask = encoder_out['trs_mask']
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size() 

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            srclen = encoder_outs.size(0)
        else:
            srclen = None
        #if self.scheduled_sampling:
        ss_backup_val = self.scheduled_sampling
        if self.scheduled_sampling:
            p = random.random()
            if p < self.scheduled_sampling_p:
                self.scheduled_sampling = False
            else:
                self.n_scheduled_sampling += 1
                #old_val = seqlen
                seqlen = int(srclen * self.args.max_len_a + self.args.max_len_b) + 1

                #print(' [DEBUG] seqlen set to {} with scheduled sampling (instead of {})'.format(seqlen, old_val))
                #sys.stdout.flush()

        self.num_masked_tokens += torch.sum(encoder_padding_mask == True).item() if encoder_padding_mask != None else 0

        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state 

        att_cached_state = utils.get_incremental_state(self, incremental_state, 'for_att_cached_state') 
        if att_cached_state is not None:
            #dec_idx, tmp_prev_predictions = att_cached_state
            dec_idx = att_cached_state

            #print('prev_predictions shape: {}; prev_output_tokens shape: {}'.format(prev_predictions.size(), prev_output_tokens.size()))
            #sys.stdout.flush()

            prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 ) 
        elif incremental_state is not None:
            dec_idx = 0 
            prev_predictions = prev_output_tokens 
        else: 
            dec_idx = -1
            prev_predictions = prev_output_tokens
         
        prev_predictions_mask = prev_predictions.eq(self.padding_idx)
        prev_predictions_mask[prev_predictions == self.blank_idx] = True
        prev_predictions_mask = prev_predictions_mask.transpose(0, 1) 
        utils.set_incremental_state(self, incremental_state, 'for_att_cached_state', dec_idx+1)

        # embed tokens
        x = self.embed_tokens(prev_predictions)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) 

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state
            prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 ) 
        elif encoder_out is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
            prev_predictions = prev_output_tokens
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None
            prev_predictions = prev_output_tokens

        assert srclen is not None or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []

        #ss_backup_val = self.scheduled_sampling
        #if self.scheduled_sampling:
        #    p = random.random()
        #    if p < self.scheduled_sampling_p:
        #        self.scheduled_sampling = False
        #    else:
        #        self.n_scheduled_sampling += 1
        finalized_hypos = 0
        train_flag = incremental_state is None and self.scheduled_sampling
        if train_flag:
            scheduled_sampling_out = torch.zeros(seqlen, bsz, self.out_embed_dim).to(x)
            prev_output_tokens_h = torch.LongTensor(bsz, 4*self.max_target_positions).to(x.device)
            prev_output_tokens_h.fill_(self.dictionary.pad())
            prev_output_tokens_h[:,0] = self.dictionary.eos()

        '''context_limit = 6
        context_bound = 0 
        if hasattr(self, 'dialog_history') and len(self.dialog_history) > 0:
            context_bound = min(len(self.dialog_history), context_limit)
            if context_bound+2 <= len(self.dialog_history):
                context_bound += 2
            elif context_bound+1 <= len(self.dialog_history):
                context_bound += 1
            if self.context_attention_weights is None and self.args.print_history_attn_weights:
                self.context_attention_weights = [[] for i in range(context_bound)]'''
        context_bound = self.get_context_length()
        for j in range(seqlen):
            curr_pred = x
            curr_pred_mask = prev_predictions_mask
            if incremental_state is None:
                if self.scheduled_sampling:
                    #cacheB, cacheT = prev_output_tokens_h.size()
                    #if cacheB != bsz:
                    #    print(' - LSTMDecoderEx, reducing scheduled sampling predicted tokens cache: {} vs. {} (length: {} vs. {})'.format(cacheB, bsz, cacheT, seqlen))
                    #    sys.stdout.flush()

                    prev_predictions = prev_output_tokens_h[:,:j+1].clone() #torch.cat( scheduled_sampling_prev_lst, 1 )
                    prev_predictions_mask = prev_predictions.eq(self.padding_idx)
                    prev_predictions_mask[prev_predictions == self.blank_idx] = True
                    prev_predictions_mask = prev_predictions_mask.transpose(0, 1)
                    x = self.embed_tokens(prev_predictions)
                    x = F.dropout(x, p=self.dropout_in, training=self.training)
                    x = x.transpose(0, 1)
                    #tgt_idx = j+1
                #else:
                #    dec_idx = j
                curr_pred = x[:j+1,:,:]   # We mask the future 
                curr_pred_mask = prev_predictions_mask[:j+1,:]

            '''if train_flag:
                cacheB, cacheT = prev_output_tokens_h.size()
                if cacheB != bsz:
                    print(' - LSTMDecoderEx, attention input shapes:')
                    print('   - prev_hiddens[0]: {}'.format(prev_hiddens[0].size()))
                    print('   - curr_pred: {}'.format(curr_pred.size()))
                    print('   - curr_pred_mask: {}'.format(curr_pred_mask.size()))
                    sys.stdout.flush()'''

            pred_attn, _ = self.pred_attention(prev_hiddens[-1], curr_pred, curr_pred_mask)
            pred_attn = F.dropout(pred_attn, p=self.dropout_out, training=self.training)

            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((pred_attn, input_feed), dim=1)
            else:
                input = pred_attn

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None: 
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            if hasattr(self, 'trs_attention'):
                #trs_x = encoder_out['trs_out']
                #trs_mask = encoder_out['trs_mask']
                trs_att, _ = self.trs_attention(hidden, trs_x, trs_mask)
                # Solution 1.
                out = out + trs_att
                # Solution 2.
                #gate = F.softmax( self.sig_gate(out) + self.trs_gate(trs_att), dim=-1 )
                #out = gate * out + (1.0 - gate) * trs_att
            out = F.dropout(out, p=self.dropout_out, training=self.training)
            
            if hasattr(self, 'dialog_history') and len(self.dialog_history) > 0:
                #print('[DEBUG] LSTMDecoderEx.extract_features: dialog history length {}'.format(len(self.dialog_history)))
                #sys.stdout.flush()
 
                #bounded_dialog_history = self.dialog_history[:context_limit]    # NOTE: first turns are always the most informative
                #if context_bound > context_limit:
                #    bounded_dialog_history = bounded_dialog_history + self.dialog_history[-(context_bound-context_limit):]  # NOTE: plus the last 1 or 2
                bounded_dialog_history = self.get_dialog_context(context_bound)

                #print('[DEBUG] LSTMDecoderEx.extract_features, context_bound = {} ; history size = {}.'.format(context_bound, len(bounded_dialog_history))) 
                #print('[DEBUG]   =========================')
                #sys.stdout.flush()

                context_att = []
                for h_idx, h in enumerate(bounded_dialog_history):
                    #print('[DEBUG] LSTMDecoderEx.extract_features, computing attention on context size {}, turn shape {}'.format(len(bounded_dialog_history), h.size()))

                    #print('[DEBUG] LSTMDecoderEx.extract_features, context attention input shapes: out = {}, h = {}'.format(out.size(), h.size()))
                    #sys.stdout.flush()

                    #print('[DEBUG] LSTMDecoderEx.extract_features, computing attention on history turn {}: '.format(h_idx))
                    #scores = F.log_softmax(self.output_layer(h.transpose(0,1)), -1)
                    #_, tpreds = torch.max(scores, -1)
                    #print('[DEBUG] > {}'.format(self.dictionary.string(tpreds[0,:])))

                    h_att, h_att_scores = self.context_attention(out, h, None)  # TODO: consider segmenting h, based on h_att_scores, for keeping only meaningful chunks
                    if self.args.print_history_attn_weights:
                        self.context_attention_weights[h_idx].append(h_att_scores)

                    #print('[DEBUG] LSTMDecoderEx.extract_features, computed context vector of shape: {}'.format(h_att.size()))
                    #sys.stdout.flush()

                    context_att.append( h_att )
                #print('[DEBUG] ----------')
                #sys.stdout.flush()

                bounded_dialog_history = []
                context_att = torch.stack(context_att, 0)
                hal_att, _ = self.hal(out, context_att, None)   # TODO: consider using an LSTM instead of the hal
                #hal_att = F.dropout(hal_att, p=self.dropout_out, training=self.training)
                if self.args.context_fusion == 'sum':
                    out = out + self.args.context_discount * hal_att
                else:
                    #gate = F.log_softmax( self.context_gate(torch.cat([out, hal_att], dim=-1)), -1 )
                    gate = F.softmax( self.context_gate_h(out) + self.context_gate_y(hal_att), dim=-1 )
                    out = gate * out + (1.0 - gate) * hal_att 

            # input feeding
            if input_feed is not None:
                input_feed = out

            if train_flag:
                ss_x = self.scheduled_sampling_output_(out)
                ss_pred = self.scheduled_sampling_predict_(ss_x)

                scheduled_sampling_out[j,:,:] = ss_x              
                prev_output_tokens_h[:,j+1] = ss_pred

                finalized_hypos += torch.sum( ss_pred == self.dictionary.eos() ).item()
                if finalized_hypos >= bsz:
                    #print(' [DEBUG] found {} eos tokens, breaking decoding...'.format(finalized_hypos))
                    #sys.stdout.flush()
                    break
            else:
                # save final output
                outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed, prev_predictions),
        )
        '''print(' - after extracting features:')
        print('   * prev_hiddens shape: {}'.format(prev_hiddens[-1].size()))
        print('   * prev_cells shape: {}'.format(prev_cells[-1].size()))
        print('   * input_feed shape: {}'.format(input_feed.size()))
        print(' -----')
        sys.stdout.flush()'''

        if train_flag:
            x = scheduled_sampling_out
        else:
            # collect outputs across time steps
            x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if not train_flag:
            if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        self.scheduled_sampling = ss_backup_val
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
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return 

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:
                return state.index_select(0, new_order)
            else:
                return None

        #print('[DEBUG] LSTMDecoderEx.reorder_incremental_state, new_order: {}'.format(new_order))
        #sys.stdout.flush()

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

        curr_hidden_state = self.dh_hidden_states if hasattr(self, 'dh_hidden_states') else None #utils.get_incremental_state(self, incremental_state, 'curr_hidden_state')
        if curr_hidden_state is not None:
 
            if len(self.finalized_sequences) > 0:
                #print('[DEBUG] LSTMDecoderEx.reorder_incremental_state: detected missed end-of-sequence {} vs. {} (finalized sequences: {})'.format(self.last_order_state, new_order, self.finalized_sequences))
                #sys.stdout.flush()

                bz = curr_hidden_state[0].size(0)
                for h in curr_hidden_state:
                    assert h.size(0) == bz
                assert len(self.finalized_sequences) <= bz

            new_state = list(map(reorder_state, curr_hidden_state)) 
            self.dh_hidden_states = new_state

        if hasattr(self, 'dialog_history'):
            for i in range(len(self.dialog_history)):
                self.dialog_history[i] = self.dialog_history[i].index_select(1, new_order)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class DraftLSTMDecoder(FairseqIncrementalDecoder):
    """Like LSTM decoder, but it uses an attention on all previous predictions, instead of using only the previous prediction."""
    def __init__(
        self, dictionary, args, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=1024
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.out_embed_dim = out_embed_dim
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.args = args
        self.blank_idx = dictionary.blank()
        self.num_masked_tokens = 0

        #if args.scheduled_sampling:
        #    self.register_buffer('prev_output_tokens_h', torch.LongTensor(args.max_sentences, 4*max_target_positions).fill_(dictionary.pad()))
        #    self.prev_output_tokens_h.requires_grad = False
        self.scheduled_sampling = False
        self.scheduled_sampling_updates = 0
        self.n_scheduled_sampling = 0
        self.scheduled_sampling_init = 0.99
        self.scheduled_sampling_p = 0.99

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = init_functions.LSTMEmbedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        self.pred_attention = SLUAttentionLayer(hidden_size, embed_dim, hidden_size, bias=False)  # TODO: use this as input to the LSTM instead of x

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList([
            init_functions.SLULSTMCell(
                input_size=input_feed_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = SLUAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = init_functions.LSTMLinear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = init_functions.LSTMLinear(out_embed_dim, num_embeddings)

        self.speaker = None

    def set_turn_speaker(self, val):
        self.speaker = val
        return self.speaker

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), {'attn': attn_scores, 'feats': x}

    def set_scheduled_sampling(self, val=False, epoch=1):
        self.scheduled_sampling = val
        if self.scheduled_sampling:
            self.scheduled_sampling_updates = epoch
            if self.scheduled_sampling_updates >= self.args.start_scheduled_sampling:
                self.scheduled_sampling_p = self.scheduled_sampling_init * (0.99**(epoch-self.args.start_scheduled_sampling))

        print(' - DraftLSTMDecoder: scheduled sampling set to {} (updates: {}; p: {})'.format(self.scheduled_sampling, self.scheduled_sampling_updates, self.scheduled_sampling_p))
        sys.stdout.flush()

    def n_scheduled_sampling_(self, val=0):
        old_val = self.n_scheduled_sampling
        self.n_scheduled_sampling = val
        return old_val

    def scheduled_sampling_output_(self, x):
        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)
        return x

    def scheduled_sampling_predict_(self, x):
        x = self.output_layer(x)
        scores = F.log_softmax(x, -1)

        #print(' - scheduled_sampling processing, x: {}; scores: {}'.format(x.size(), scores.size()))
        #sys.stdout.flush()

        max_sc, pred = torch.max(scores, -1)
        return pred

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            encoder_outs = encoder_outs.clone().detach()    # We make encoder_outs a leaf in the draft decoder so that the encoder weights are not updated twice.
            srclen = encoder_outs.size(0)
        else:
            srclen = None
        #if self.scheduled_sampling:
        ss_backup_val = self.scheduled_sampling
        if self.scheduled_sampling:
            p = random.random()
            if p < self.scheduled_sampling_p:
                self.scheduled_sampling = False
            else:
                self.n_scheduled_sampling += 1
                #old_val = seqlen
                seqlen = int(srclen * self.args.max_len_a + self.args.max_len_b) + 1

                #print(' [DEBUG] seqlen set to {} with scheduled sampling (instead of {})'.format(seqlen, old_val))
                #sys.stdout.flush()
        if incremental_state is not None:
            seqlen = int(srclen * self.args.max_len_a + self.args.max_len_b) + 1

            #print('[DEBUG] DraftLSTMDecoder, seqlen set to {} (srclen: {})'.format(seqlen, srclen))
            #sys.stdout.flush()

        self.num_masked_tokens += torch.sum(encoder_padding_mask == True).item() if encoder_padding_mask != None else 0

        '''cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state'''

        '''att_cached_state = utils.get_incremental_state(self, incremental_state, 'for_att_cached_state')
        if att_cached_state is not None: 
            dec_idx = att_cached_state 
            prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 )
        elif incremental_state is not None:
            dec_idx = 0
            prev_predictions = prev_output_tokens
        else:
            dec_idx = -1
            prev_predictions = prev_output_tokens'''

        prev_predictions = prev_output_tokens
        prev_predictions_mask = prev_predictions.eq(self.padding_idx)
        prev_predictions_mask[prev_predictions == self.blank_idx] = True
        prev_predictions_mask = prev_predictions_mask.transpose(0, 1)
        #utils.set_incremental_state(self, incremental_state, 'for_att_cached_state', dec_idx+1)

        # embed tokens
        x = self.embed_tokens(prev_predictions)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state
            #prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 )
        elif encoder_out is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
            #prev_predictions = prev_output_tokens
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None
            #prev_predictions = prev_output_tokens

        assert srclen is not None or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []

        #ss_backup_val = self.scheduled_sampling
        #if self.scheduled_sampling:
        #    p = random.random()
        #    if p < self.scheduled_sampling_p:
        #        self.scheduled_sampling = False
        #    else:
        #        self.n_scheduled_sampling += 1
        finalized_hypos = 0
        train_flag = incremental_state is None and self.scheduled_sampling
        if train_flag:
            scheduled_sampling_out = torch.zeros(seqlen, bsz, self.out_embed_dim).to(x)
            prev_output_tokens_h = torch.LongTensor(bsz, 4*self.max_target_positions).to(x.device)
            prev_output_tokens_h.fill_(self.dictionary.pad())
            prev_output_tokens_h[:,0] = self.dictionary.eos()

        #print('[DEBUG] DraftLSTMDecoder, decoding a sequence of length {}'.format(seqlen))
        #sys.stdout.flush()
 
        inf_finalized_hypos_dim = [0] * bsz
        inf_finalized_hypos_idx = [0] * bsz
        for j in range(seqlen):
            curr_pred = x
            curr_pred_mask = prev_predictions_mask
            if incremental_state is None:
                if self.scheduled_sampling:
                    cacheB, cacheT = prev_output_tokens_h.size()
                    if cacheB != bsz:
                        print(' - DraftLSTMDecoder, reducing scheduled sampling predicted tokens cache: {} vs. {} (length: {} vs. {})'.format(cacheB, bsz, cacheT, seqlen))
                        sys.stdout.flush()

                    prev_predictions = prev_output_tokens_h[:,:j+1].clone() #torch.cat( scheduled_sampling_prev_lst, 1 )
                    prev_predictions_mask = prev_predictions.eq(self.padding_idx)
                    prev_predictions_mask[prev_predictions == self.blank_idx] = True
                    prev_predictions_mask = prev_predictions_mask.transpose(0, 1)
                    x = self.embed_tokens(prev_predictions)
                    x = F.dropout(x, p=self.dropout_in, training=self.training)
                    x = x.transpose(0, 1)
                    #tgt_idx = j+1
                #else:
                #    dec_idx = j
                curr_pred = x[:j+1,:,:]   # We mask the future
                curr_pred_mask = prev_predictions_mask[:j+1,:]

            if train_flag:
                cacheB, cacheT = prev_output_tokens_h.size()
                if cacheB != bsz:
                    print(' - DraftLSTMDecoder, attention input shapes:')
                    print('   - prev_hiddens[0]: {}'.format(prev_hiddens[0].size()))
                    print('   - curr_pred: {}'.format(curr_pred.size()))
                    print('   - curr_pred_mask: {}'.format(curr_pred_mask.size()))
                    sys.stdout.flush()

            pred_attn, _ = self.pred_attention(prev_hiddens[-1], curr_pred, curr_pred_mask)
            pred_attn = F.dropout(pred_attn, p=self.dropout_out, training=self.training)

            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((pred_attn, input_feed), dim=1)
            else:
                input = pred_attn

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            if input_feed is not None:
                input_feed = out

            if train_flag:
                ss_x = self.scheduled_sampling_output_(out)
                ss_pred = self.scheduled_sampling_predict_(ss_x)

                scheduled_sampling_out[j,:,:] = ss_x
                prev_output_tokens_h[:,j+1] = ss_pred

                finalized_hypos += torch.sum( ss_pred == self.dictionary.eos() ).item()
                if finalized_hypos >= bsz:
                    #print(' [DEBUG] found {} eos tokens, breaking decoding...'.format(finalized_hypos))
                    #sys.stdout.flush()
                    break
            else:
                # save final output
                outs.append(out)

            if incremental_state is not None:
                # NOTE: for generating predictions "all at once" in the DraftDecoder we use functions 'scheduled_sampling_output_' and 'scheduled_sampling_predict_' to compute predictions at inference phase as they do exactly what would be done in the last part of the 'extract_features' function.
                draft_x = self.scheduled_sampling_output_(out)
                draft_preds = self.scheduled_sampling_predict_(draft_x)
                prev_predictions = torch.cat([prev_predictions, draft_preds], 1)
                #prev_predictions = prev_output_tokens
 
                prev_predictions_mask = prev_predictions.eq(self.padding_idx)
                prev_predictions_mask[prev_predictions == self.blank_idx] = True
                prev_predictions_mask = prev_predictions_mask.transpose(0, 1)

                x = self.embed_tokens(prev_predictions)
                x = F.dropout(x, p=self.dropout_in, training=self.training)

                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

                for o_idx in range(bsz):
                    if draft_preds[o_idx] == self.dictionary.eos():
                        inf_finalized_hypos_dim[o_idx] = 1
                        inf_finalized_hypos_idx[o_idx] = j
                if sum(inf_finalized_hypos_dim) >= bsz:
                    break

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed, prev_predictions),
        )
        '''print(' - after extracting features:')
        print('   * prev_hiddens shape: {}'.format(prev_hiddens[-1].size()))
        print('   * prev_cells shape: {}'.format(prev_cells[-1].size()))
        print('   * input_feed shape: {}'.format(input_feed.size()))
        print(' -----')
        sys.stdout.flush()'''

        if train_flag:
            x = scheduled_sampling_out
        else:
            # collect outputs across time steps
            x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if not train_flag:
            if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        #print('[DEBUG] DraftLSTMDecoder, draft decoded: {}'.format(x.size()))
        #sys.stdout.flush()

        self.scheduled_sampling = ss_backup_val
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
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        '''print(' - reorder_incremental_state, shapes before reordering:')
        for ii in cached_state:
            if isinstance(ii, list):
                print(ii[-1].size())
            elif ii is not None:
                print(ii.size())
        print(' ---')'''

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:
                return state.index_select(0, new_order)
            else:
                return None

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

        '''print(' - reorder_incremental_state, shapes AFTER reordering:')
        for ii in cached_state:
            if isinstance(ii, list):
                print(ii[-1].size())
            elif ii is not None:
                print(ii.size())
        print(' ---')
        sys.stdout.flush()'''

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class DeliberationLSTMDecoder(FairseqIncrementalDecoder):
    """Like LSTM decoder, but it uses an attention on all previous predictions, instead of using only the previous prediction."""
    def __init__(
        self, dictionary, args, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        max_target_positions=1024
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.out_embed_dim = out_embed_dim
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.max_target_positions = max_target_positions
        self.args = args
        self.blank_idx = dictionary.blank()
        self.num_masked_tokens = 0

        #if args.scheduled_sampling:
        #    self.register_buffer('prev_output_tokens_h', torch.LongTensor(args.max_sentences, 4*max_target_positions).fill_(dictionary.pad()))
        #    self.prev_output_tokens_h.requires_grad = False
        self.scheduled_sampling = False
        self.scheduled_sampling_updates = 0
        self.n_scheduled_sampling = 0
        self.scheduled_sampling_init = 0.99
        self.scheduled_sampling_p = 0.99

        self.draft_decoder = DraftLSTMDecoder(dictionary, args,
            embed_dim=embed_dim, hidden_size=hidden_size, out_embed_dim=out_embed_dim,
            num_layers=num_layers,
            dropout_in=dropout_in, dropout_out=dropout_out,
            attention=attention,
            encoder_output_units=encoder_output_units,
            pretrained_embed=pretrained_embed,
            share_input_output_embed=share_input_output_embed,
            adaptive_softmax_cutoff=adaptive_softmax_cutoff,
            max_target_positions=max_target_positions
        )
        self.draft_attention = SLUAttentionLayer(hidden_size, out_embed_dim, hidden_size, bias=False)

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = init_functions.LSTMEmbedding(num_embeddings, embed_dim, self.padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size and encoder_output_units != 0:
            self.encoder_hidden_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = init_functions.LSTMLinear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None

        self.pred_attention = SLUAttentionLayer(hidden_size, embed_dim, hidden_size, bias=False)  # TODO: use this as input to the LSTM instead of x

        # disable input feeding if there is no encoder
        # input feeding is described in arxiv.org/abs/1508.04025
        input_feed_size = 0 if encoder_output_units == 0 else hidden_size
        self.layers = nn.ModuleList([
            init_functions.SLULSTMCell(
                input_size=input_feed_size + embed_dim + hidden_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = SLUAttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = init_functions.LSTMLinear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, hidden_size, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = init_functions.LSTMLinear(out_embed_dim, num_embeddings)

        self.speaker = None

    def set_turn_speaker(self, val):
        self.speaker = val
        return self.speaker

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, incremental_state
        )
        return self.output_layer(x), attn_scores

    def set_scheduled_sampling(self, val=False, epoch=1):
        self.scheduled_sampling = val
        if self.scheduled_sampling:
            self.scheduled_sampling_updates = epoch
            if self.scheduled_sampling_updates >= self.args.start_scheduled_sampling:
                self.scheduled_sampling_p = self.scheduled_sampling_init * (0.99**(epoch-self.args.start_scheduled_sampling))

        print(' - LSTMDecoderEx: scheduled sampling set to {} (updates: {}; p: {})'.format(self.scheduled_sampling, self.scheduled_sampling_updates, self.scheduled_sampling_p))
        sys.stdout.flush()

    def n_scheduled_sampling_(self, val=0):
        old_val = self.n_scheduled_sampling
        self.n_scheduled_sampling = val
        return old_val

    def scheduled_sampling_output_(self, x):
        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training) 
        return x

    def scheduled_sampling_predict_(self, x):
        x = self.output_layer(x)
        scores = F.log_softmax(x, -1)

        #print(' - scheduled_sampling processing, x: {}; scores: {}'.format(x.size(), scores.size()))
        #sys.stdout.flush() 

        max_sc, pred = torch.max(scores, -1)
        return pred

    def extract_features(
        self, prev_output_tokens, encoder_out, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        draft_encoder_out = encoder_out
        if encoder_out is not None:
            encoder_padding_mask = encoder_out['encoder_padding_mask']
            encoder_out = encoder_out['encoder_out']
        else:
            encoder_padding_mask = None
            encoder_out = None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size() 

        if incremental_state is not None:
            draft_states = utils.get_incremental_state(self, incremental_state, 'draft_state')
            if draft_states == None:
                draft_out = self.draft_decoder(
                    prev_output_tokens,
                    encoder_out=draft_encoder_out,
                    incremental_state=incremental_state,
                    features_only=False,
                    return_all_hiddens = False,
                )
                draft_state = draft_out[1]['feats'].transpose(0, 1)   # B x T x C -> T x B x C
                draft_x = draft_out[0] 
                utils.set_incremental_state(self, incremental_state, 'draft_state', (draft_state, draft_x))

                #print('[DEBUG] DeliberationLSTMDecoder, draft decoded: {}'.format(draft_state.size()))
                #sys.stdout.flush()
            else:
                draft_state, draft_x = draft_states
                draft_out = (draft_x, {'attn': None, 'feats': draft_state})
                #print('[DEBUG] DeliberationLSTMDecoder, draft retrieved from cache: {}'.format(draft_state.size()))
                #sys.stdout.flush()
        else:
            draft_out = self.draft_decoder(
                prev_output_tokens,
                encoder_out=draft_encoder_out,
                features_only=False, 
                return_all_hiddens = False,
            )
            draft_state = draft_out[1]['feats'].transpose(0, 1)   # B x T x C -> T x B x C
            draft_x = draft_out[0]

        # Precomputes draft decoder predictions to create a mask to padding and blanks
        T, B, C = draft_state.size() 

        draft_lprobs = F.log_softmax(draft_x, dim=-1)
        _, draft_preds = torch.max(draft_lprobs, -1)
        draft_mask = draft_preds.eq(self.padding_idx)   # B x T
        draft_mask[draft_mask == self.blank_idx] = True
        draft_mask = draft_mask.transpose(0,1)  # T x B 

        #print('[DEBUG] DeliberationLSTMDecoder, computed draft state: {}'.format(draft_state.size()))
        #sys.stdout.flush()

        # get outputs from encoder
        if encoder_out is not None:
            encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
            srclen = encoder_outs.size(0)
        else:
            srclen = None
        #if self.scheduled_sampling:
        ss_backup_val = self.scheduled_sampling
        if self.scheduled_sampling:
            p = random.random()
            if p < self.scheduled_sampling_p:
                self.scheduled_sampling = False
            else:
                self.n_scheduled_sampling += 1
                #old_val = seqlen
                seqlen = int(srclen * self.args.max_len_a + self.args.max_len_b) + 1

                #print(' [DEBUG] seqlen set to {} with scheduled sampling (instead of {})'.format(seqlen, old_val))
                #sys.stdout.flush()

        self.num_masked_tokens += torch.sum(encoder_padding_mask == True).item() if encoder_padding_mask != None else 0

        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state
            '''print(' - prev_hiddens shape: {}'.format(prev_hiddens[-1].size()))
            print(' - prev_cells shape: {}'.format(prev_cells[-1].size()))
            print(' - input_feed shape: {}'.format(input_feed.size()))
            sys.stdout.flush()'''

        att_cached_state = utils.get_incremental_state(self, incremental_state, 'for_att_cached_state') 
        if att_cached_state is not None:
            #dec_idx, tmp_prev_predictions = att_cached_state
            dec_idx = att_cached_state

            #print('prev_predictions shape: {}; prev_output_tokens shape: {}'.format(prev_predictions.size(), prev_output_tokens.size()))
            #sys.stdout.flush()

            prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 ) 
        elif incremental_state is not None:
            dec_idx = 0 
            prev_predictions = prev_output_tokens 
        else: 
            dec_idx = -1
            prev_predictions = prev_output_tokens
         
        prev_predictions_mask = prev_predictions.eq(self.padding_idx)
        prev_predictions_mask[prev_predictions == self.blank_idx] = True
        prev_predictions_mask = prev_predictions_mask.transpose(0, 1) 
        utils.set_incremental_state(self, incremental_state, 'for_att_cached_state', dec_idx+1)

        # embed tokens
        x = self.embed_tokens(prev_predictions)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1) 

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed, prev_predictions = cached_state
            prev_predictions = torch.cat( [prev_predictions, prev_output_tokens], 1 )
        elif encoder_out is not None:
            # setup recurrent cells
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)
            prev_predictions = prev_output_tokens
        else:
            # setup zero cells, since there is no encoder
            num_layers = len(self.layers)
            zero_state = x.new_zeros(bsz, self.hidden_size)
            prev_hiddens = [zero_state for i in range(num_layers)]
            prev_cells = [zero_state for i in range(num_layers)]
            input_feed = None
            prev_predictions = prev_output_tokens

        assert srclen is not None or self.attention is None, \
            "attention is not supported if there are no encoder outputs"
        attn_scores = x.new_zeros(srclen, seqlen, bsz) if self.attention is not None else None
        outs = []

        #ss_backup_val = self.scheduled_sampling
        #if self.scheduled_sampling:
        #    p = random.random()
        #    if p < self.scheduled_sampling_p:
        #        self.scheduled_sampling = False
        #    else:
        #        self.n_scheduled_sampling += 1
        finalized_hypos = 0
        train_flag = incremental_state is None and self.scheduled_sampling
        if train_flag:
            scheduled_sampling_out = torch.zeros(seqlen, bsz, self.out_embed_dim).to(x)
            prev_output_tokens_h = torch.LongTensor(bsz, 4*self.max_target_positions).to(x.device)
            prev_output_tokens_h.fill_(self.dictionary.pad())
            prev_output_tokens_h[:,0] = self.dictionary.eos()
        for j in range(seqlen):
            curr_pred = x
            curr_pred_mask = prev_predictions_mask
            if incremental_state is None:
                if self.scheduled_sampling:
                    cacheB, cacheT = prev_output_tokens_h.size()
                    if cacheB != bsz:
                        print(' - LSTMDecoderEx, reducing scheduled sampling predicted tokens cache: {} vs. {} (length: {} vs. {})'.format(cacheB, bsz, cacheT, seqlen))
                        sys.stdout.flush()

                    prev_predictions = prev_output_tokens_h[:,:j+1].clone() #torch.cat( scheduled_sampling_prev_lst, 1 )
                    prev_predictions_mask = prev_predictions.eq(self.padding_idx)
                    prev_predictions_mask[prev_predictions == self.blank_idx] = True
                    prev_predictions_mask = prev_predictions_mask.transpose(0, 1)
                    x = self.embed_tokens(prev_predictions)
                    x = F.dropout(x, p=self.dropout_in, training=self.training)
                    x = x.transpose(0, 1)
                    #tgt_idx = j+1
                #else:
                #    dec_idx = j
                curr_pred = x[:j+1,:,:]   # We mask the future 
                curr_pred_mask = prev_predictions_mask[:j+1,:]

            if train_flag:
                cacheB, cacheT = prev_output_tokens_h.size()
                if cacheB != bsz:
                    print(' - LSTMDecoderEx, attention input shapes:')
                    print('   - prev_hiddens[0]: {}'.format(prev_hiddens[0].size()))
                    print('   - curr_pred: {}'.format(curr_pred.size()))
                    print('   - curr_pred_mask: {}'.format(curr_pred_mask.size()))
                    sys.stdout.flush()

            pred_attn, _ = self.pred_attention(prev_hiddens[-1], curr_pred, curr_pred_mask)
            pred_attn = F.dropout(pred_attn, p=self.dropout_out, training=self.training)

            draft_attn, _ = self.draft_attention(prev_hiddens[-1], draft_state, draft_mask)
            draft_attn = F.dropout(draft_attn, p=self.dropout_out, training=self.training)

            #print('[DEBUG] DeliberationLSTMDecoder, draft attention computed: {}'.format(draft_attn.size()))
            #sys.stdout.flush() 

            # input feeding: concatenate context vector from previous time step
            if input_feed is not None:
                input = torch.cat((pred_attn, draft_attn, input_feed), dim=1)
            else:
                input = torch.cat((pred_attn, draft_attn), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            #print('[DEBUG] DeliberationLSTMDecoder, LSTM output computed')
            #sys.stdout.flush() 

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            if input_feed is not None:
                input_feed = out

            if train_flag:
                ss_x = self.scheduled_sampling_output_(out)
                ss_pred = self.scheduled_sampling_predict_(ss_x)

                scheduled_sampling_out[j,:,:] = ss_x              
                prev_output_tokens_h[:,j+1] = ss_pred

                finalized_hypos += torch.sum( ss_pred == self.dictionary.eos() ).item()
                if finalized_hypos >= bsz:
                    #print(' [DEBUG] found {} eos tokens, breaking decoding...'.format(finalized_hypos))
                    #sys.stdout.flush()
                    break
            else:
                # save final output
                outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed, prev_predictions),
        )
        '''print(' - after extracting features:')
        print('   * prev_hiddens shape: {}'.format(prev_hiddens[-1].size()))
        print('   * prev_cells shape: {}'.format(prev_cells[-1].size()))
        print('   * input_feed shape: {}'.format(input_feed.size()))
        print(' -----')
        sys.stdout.flush()'''

        if train_flag:
            x = scheduled_sampling_out
        else:
            # collect outputs across time steps
            x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if not train_flag:
            if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn and self.attention is not None:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None 

        #print('[DEBUG] DeliberationLSTMDecoder, output computed: {}'.format(x.size()))
        #sys.stdout.flush()

        self.scheduled_sampling = ss_backup_val
        return x, {'attn': attn_scores, 'draft_out': draft_out}

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
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        '''print(' - reorder_incremental_state, shapes before reordering:')
        for ii in cached_state:
            if isinstance(ii, list):
                print(ii[-1].size())
            elif ii is not None:
                print(ii.size())
        print(' ---')'''

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            elif state is not None:
                return state.index_select(0, new_order)
            else:
                return None

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

        draft_states = utils.get_incremental_state(self, incremental_state, 'draft_state')
        if draft_states is not None:
            draft_state, draft_x = draft_states
            draft_state = draft_state.index_select(1, new_order)
            draft_x = draft_x.index_select(0, new_order)
            utils.set_incremental_state(self, incremental_state, 'draft_state', (draft_state, draft_x))

        '''print(' - reorder_incremental_state, shapes AFTER reordering:')
        for ii in cached_state:
            if isinstance(ii, list):
                print(ii[-1].size())
            elif ii is not None:
                print(ii.size())
        print(' ---')
        sys.stdout.flush()'''

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

### ENCODER-DECODER MODEL ###
from fairseq.models import FairseqEncoderDecoderModel, register_model

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@register_model('end2end_slu')
class End2EndSLUModel(FairseqEncoderDecoderModel):

    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument( '--encoder-hidden-dim', type=int, metavar='N', help='dimensionality of the encoder hidden state' )
        parser.add_argument( '--encoder-dropout', type=float, default=0.5, help='encoder dropout probability' )
        parser.add_argument( '--decoder', type=str, default='transformer', help='which decoder to use in the model: basic, transformer (default), lstm, icassp_lstm' )
        parser.add_argument( '--decoder-hidden-dim', type=int, metavar='N', help='dimensionality of the decoder hidden state' )
        parser.add_argument( '--decoder-dropout', type=float, default=0.12, help='decoder dropout probability' )
        parser.add_argument( '--decoder-output-size', type=int, default=2650, help='A-priori output dictionary size (for model refinement...)' )
        parser.add_argument( '--speech-conv', type=int, default=1, help='Number of convolution layers in the encoder' )
        parser.add_argument( '--speech-conv-size', type=int, default=81, help='Size of convolution layers in the encoder' )
        parser.add_argument( '--num-features', type=int, default=81, help='Number of features as input to the first convolution layer in the encoder' )
        parser.add_argument( '--conv-kernel', type=int, default=3, help='Size of the convolution kernels in the encoder' )
        parser.add_argument( '--num-lstm-layers', type=int, default=2, help='Number of LSTM layers in the encoder' )
        parser.add_argument( '--speech-lstm-size', type=int, default=256, help='Size of the LSTM layers in the encoder' )
        parser.add_argument( '--drop-ratio', type=float, default=0.5, help='Dropout ratio in the encoder' )
        parser.add_argument( '--output-size', type=int, default=512, help='Size of the output-layer of the encoder' )
        #parser.add_argument( '--load-encoder', type=str, default='None', help='Load a pre-trained (basic) encoder' )
        #parser.add_argument( '--load-fairseq-encoder', type=str, default='None', help='Load the encoder from a fairseq checkpoint' )
        parser.add_argument( '--match-source-len', action='store_true', default=False, help='For scheduled-sampling decoding, same behavior as for fairseq-generate' )
        #parser.add_argument( '--max-len-a', type=float, default=1.0, help='For scheduled-sampling decoding, same behavior as for fairseq-generate' )
        #parser.add_argument( '--max-len-b', type=int, default=0, help='For scheduled-sampling decoding, same behavior as for fairseq-generate' )
        parser.add_argument( '--freeze-encoder', action='store_true', default=False, help='Freeze encoder parameters, only learns the decoder' )
        #parser.add_argument('--pyramid-hidden-layers', type=int, default=1, help='Number of layers in the hidden LSTM stages of the pyramidal encoder')

        TransformerModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        # Fairseq initializes models by calling the ``build_model()``
        # function. This provides more flexibility, since the returned model
        # instance can be of a different type than the one that was called.

        args.model_beam = task.args.model_beam
        args.print_history_attn_weights = task.args.print_history_attn_weights

        if task.num_features != args.num_features:
            sys.stderr.write(' - End2EndSLU build_model: detected and given number of features missmatch ({} VS. {}). Using detected...\n'.format(task.num_features, args.num_features))
            args.num_features = task.num_features

        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))
 
        #if task.args.corpus_name == 'fsc':
        #    task.args.encoder_state_window = 5
        #    task.args.constrained_output_length = 10

        # Initialize our Encoder and Decoder.
        encoder = End2EndSLUEncoder(
            args=args,
            dictionary=task.label_vocab,
        )

        decoder_embeddings = None
        if hasattr(args, 'load_embeddings') and args.load_embeddings:
            if not args.load_dictionary:
                print(' * End2EndSLUModel CRITICAL WARNING: loading pre-trained embeddings, but no pre-defined dictionary has been loaded, indexing may be corrupted')
                sys.stdout.flush()

            print(' * End2EndSLUModel: loading pre-trained embeddings...')
            sys.stdout.flush()

            decoder_embeddings = torch.load(args.load_embeddings, map_location=torch.device('cpu'))
            decoder_embeddings = decoder_embeddings.detach()
            if args.freeze_embeddings:
                print(' * End2EndSLUModel: freezing model embeddings')
                sys.stdout.flush()

                for param in decoder_embeddings.parameters():
                    param.requires_grad = False

        tgt_dict = task.label_vocab
        decoder = None
        if args.decoder == 'basic':
            decoder = BasicDecoder(tgt_dict, hidden_dim=2*args.decoder_hidden_dim)
        elif args.decoder == 'transformer':
            if decoder_embeddings is None:
                decoder_embeddings = cls.build_embedding(
                    args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
                )
            decoder = cls.build_decoder(args, tgt_dict, decoder_embeddings)
        elif args.decoder == 'ctclstm': 
            decoder = LSTMDecoderEx(
                dictionary=tgt_dict,
                args=args,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_dim,
                out_embed_dim=args.output_size,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout,
                dropout_out=args.decoder_dropout,
                attention=True,
                encoder_output_units=encoder.encoder_output_units,
                pretrained_embed=decoder_embeddings,
                share_input_output_embed=False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                max_target_positions=args.max_target_positions
            )
        elif args.decoder == 'deliberation':
            decoder = DeliberationLSTMDecoder(
                dictionary=tgt_dict,
                args=args,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_dim,
                out_embed_dim=args.output_size,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout,
                dropout_out=args.decoder_dropout,
                attention=True,
                encoder_output_units=encoder.encoder_output_units,
                pretrained_embed=decoder_embeddings,
                share_input_output_embed=False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                max_target_positions=args.max_target_positions
            )
        elif args.decoder == 'lstm':
            decoder = LSTMDecoder(
                dictionary=tgt_dict, 
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_dim,
                out_embed_dim=args.output_size,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout,
                dropout_out=args.decoder_dropout,
                attention=True,
                encoder_output_units=encoder.encoder_output_units,
                pretrained_embed=decoder_embeddings,
                share_input_output_embed=False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                max_target_positions=args.max_target_positions
            )
        elif args.decoder == 'icassp_lstm':
            '''decoder = ICASSPDecoder(
                dictionary=tgt_dict,
                args=args,
                embeddings=None)'''
            decoder = ICASSPDecoderEx(
                args=args,
                dictionary=tgt_dict,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_dim,
                out_embed_dim=args.output_size,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout,
                dropout_out=args.decoder_dropout,
                attention=True,
                encoder_output_units=encoder.encoder_output_units,
                pretrained_embed=decoder_embeddings,
                share_input_output_embed=False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                max_target_positions=args.max_target_positions
            )
        elif args.decoder == 'end2end_slu':
            decoder = End2EndSLUDecoder(
                dictionary=tgt_dict,
                args=args,
                embed_dim=args.decoder_embed_dim,
                hidden_size=args.decoder_hidden_dim,
                out_embed_dim=args.output_size,
                num_layers=args.decoder_layers,
                dropout_in=args.decoder_dropout,
                dropout_out=args.decoder_dropout,
                attention=True,
                encoder_output_units=encoder.encoder_output_units,
                pretrained_embed=decoder_embeddings,
                share_input_output_embed=False,
                adaptive_softmax_cutoff=(
                    options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                    if args.criterion == 'adaptive_loss' else None
                ),
                max_target_positions=args.max_target_positions
            )
        else:
            raise NotImplementedError
 
        model = End2EndSLUModel(args, encoder, decoder, tgt_dict)
        model.spk_abs_val = task.spk_abs_val

        # Print the model architecture.
        #print(model)

        if args.load_encoder: 
            print(' - Loading base encoder {}'.format(args.load_encoder))
            sys.stdout.flush()
            model.load_base_encoder( args.load_encoder )
        elif args.load_fairseq_encoder:
            if hasattr(task.args, 'decoder'):
                print(' - Loading base encoder from fairseq checkpoint {}'.format(args.load_fairseq_encoder))
                sys.stdout.flush()
                model.load_fairseq_encoder( args.load_fairseq_encoder, task )

        if hasattr(args, 'load_fairseq_decoder') and args.load_fairseq_decoder:
            print(' - Loading decoder from fairseq checkpoint {}'.format(args.load_fairseq_decoder))
            sys.stdout.flush()
            model.load_fairseq_decoder( args.load_fairseq_decoder, task )

        return model

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        
        emb = init_functions.LSTMEmbedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        '''return CTCTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )'''
        pass

    def __init__(self, args, encoder, decoder, tgt_dict):
        super().__init__(encoder, decoder)

        self.spk_abs_val = 1.0
        self.args = args
        self.dict = tgt_dict
        self.teacher_forcing = True
        self.scheduled_sampling = args.scheduled_sampling

        #self.user_pad = src_tokens.new_zeros(3,args.num_features).fill_(5.0)
        #self.mach_pad = src_tokens.new_zeros(3,args.num_features).fill_(-5.0)

    def num_masked_tokens(self, val=0):
        num_masked = self.decoder.num_masked_tokens
        self.decoder.num_masked_tokens = val
        return num_masked

    def set_scheduled_sampling(self, ss_val=False, epoch=1):
        self.scheduled_sampling = ss_val
        if isinstance(self.decoder, ICASSPDecoderEx) or isinstance(self.decoder, LSTMDecoderEx):
            self.decoder.set_scheduled_sampling(val=ss_val, epoch=epoch)

    def get_scheduled_sampling_updates(self):
        if isinstance(self.decoder, ICASSPDecoderEx) or isinstance(self.decoder, LSTMDecoderEx):
            return self.decoder.n_scheduled_sampling_()
        else:
            return 0

    def switch_scheduled_sampling(self):
        self.scheduled_sampling = not self.scheduled_sampling
 
    def _decode(self, prev_output_tokens, encoder_out=None, incremental_state=None, **kwargs):

        encoder_states = None
        if self.args.decoder in ['basic', 'transformer']:
            encoder_states = encoder_out.encoder_out
        else:
            encoder_states = encoder_out['encoder_out'][0]

        (src_len, src_bsz, dim) = encoder_states.size()
        (tgt_bsz, tgt_len) = prev_output_tokens.size()

        seq_len = 0
        if self.args.match_source_len:
            seq_len = src_len
        elif self.teacher_forcing:
            seq_len = tgt_len
        else:
            seq_len = int(self.args.max_len_a * float(src_len) + float(self.args.max_len_b))

        # Initialize predicted tokens and their corresponding model outputs
        #   - First predicted token is for free: bos marker. We initialize model output (lprobs) accordingly
        starter = self.dict.eos()
        tokens = torch.LongTensor(tgt_bsz, seq_len).fill_(self.dict.pad()).to(encoder_states.device)
        tokens[:,0] = starter

        lprobs = torch.FloatTensor(tgt_bsz, seq_len, len(self.dict)).fill_(-np.log( len(self.dict) )).to(encoder_states)
        lprobs[:,0,:starter] = -math.inf
        lprobs[:,0,starter] = 1.0
        lprobs[:,0,starter+1:] = -math.inf

        attn = None
        for step in range(0,seq_len):
            is_eos = tokens[:, step].eq(self.dict.eos())
            if step == seq_len-1 or (step > 0 and is_eos.sum() == is_eos.size(0)):
                # all predictions are finished (i.e., ended with eos)
                tokens = tokens[:, :step+1]
                if attn is not None and self.args.decoder != 'transformer':
                    attn = attn[:, :step+1, :]
                break 

            # Attention is:
            # None for basic decoder
            # {"attn": [attn], "inner_states": inner_states} for TransformerDecoder
            # tensor (bsz, tgt_len, src_len) for LSTMDecoder
            # tensor (bsz, tgt_len, src_len) for ICASSPDecoder
            log_probs, avg_attn_scores = self.forward_decoder(
                tokens[:, :step+1],
                encoder_outs,
                features_only=False,
                src_lengths=src_lengths,
                return_all_hiddens=False,
            )
            log_probs[log_probs != log_probs] = -math.inf
            # Forbid a few tokens:
            log_probs[:, self.pad] = -math.inf 
            log_probs[:, self.bos] = -math.inf
            log_probs[:, self.unk] = -math.inf
            #log_probs[:, self.blank] = -math.inf

            tokens[:, step+1] = log_probs.argmax(-1) 
            if step > 0:  # deal with finished predictions
                # make log_probs uniform if the previous output token is EOS
                # and add consecutive EOS to the end of prediction
                log_probs[is_eos, :] = -np.log(log_probs.size(1))
                tokens[is_eos, step + 1] = self.dict.eos()
            #if self.teacher_forcing and step < seq_len:
            #    lprobs[:, step, :] = log_probs
            lprobs[:, step+1, :] = log_probs

            # Record attention scores 
            if avg_attn_scores is not None:
                if self.args.decoder in ['lstm', 'icassp_lstm']:
                    if attn is None:
                        attn = avg_attn_scores.new(tgt_bsz, seq_len, src_len)
                    attn[:,step+1,:].copy_(avg_attn_scores)
                elif self.args.decoder == 'transformer':
                    if attn is None:
                        attn = []
                    attn.append( avg_attn_scores ) 

        if self.args.decoder == 'transformer':
            attention = torch.stack( [d['attn'][0] for d in attn], 1 )  # bsz, tgt_len, src_len I guess # TODO: check this !
            inner_states = torch.stack( [d['inner_states'] for d in attn], 1 )  # bsz, tgt_len, src_len I guess # TODO: check this !
            attn = {'attn' : [attention], 'inner_states' : inner_states}

        return lprobs, attn

    # We could override the ``forward()`` if we wanted more control over how
    # the encoder and decoder interact, but it's not necessary for this
    # tutorial since we can inherit the default implementation provided by
    # the FairseqEncoderDecoderModel base class, which looks like:
    #
    def forward(self, src_signals, src_tokens, src_lengths, src_tok_lengths, prev_output_tokens):
        """
            Returns:
                    - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                    - an extra dictionary with task specific content
        """
        '''src_sig, src_trs = None, None
        if isinstance(src_tokens, tuple):
            src_sig, src_trs = src_tokens
            #if str(src_sig.device) == 'cpu':
            #    src_sig = src_sig.to(prev_output_tokens.device)
            #    src_trs = src_trs.to(prev_output_tokens.device)
            src_tokens = src_sig'''
        tmp = src_tokens
        if src_signals is not None:
            src_tokens = src_signals

        B, T, C = src_tokens.size()
        user_pad = src_tokens.new_zeros(3,C).fill_(self.spk_abs_val)
        mach_pad = src_tokens.new_zeros(3,C).fill_(-self.spk_abs_val)
        bogus_pad = src_tokens.new_zeros(3, C).fill_(-2.0*self.spk_abs_val)
        #eod_pad = src_tokens.new_zeros(3, C).fill_(EOD_value)
 
        speakers = []
        for i in range(B):         
            if torch.sum(src_tokens[i,:3,:] == user_pad).item() == 3*C or torch.sum(src_tokens[i,-3:,:] == user_pad).item() == 3*C:
                speakers.append(user_ID) 
            elif torch.sum(src_tokens[i,:3,:] == mach_pad).item() == 3*C or torch.sum(src_tokens[i,-3:,:] == mach_pad).item() == 3*C:
                speakers.append(machine_ID)
            elif torch.sum(src_tokens[i,:3,:] == bogus_pad).item() == 3*C or torch.sum(src_tokens[i,-3:,:] == bogus_pad).item() == 3*C:
                speakers.append(bogus_ID)
            #elif torch.sum(src_tokens[i,:3,:] == eod_pad).item() == 3*C or torch.sum(src_tokens[i,-3:,:] == eod_pad).item() == 3*C:
            #    speakers.append(bogus_ID)
            else: 
                raise ValueError(' Speaker is expected to be "User", "Machine" or "_BOGUS_", no valid speaker detected ')
        self.encoder.set_turn_speaker( speakers )
        if not isinstance(self.decoder, LSTMDecoder):
            self.decoder.set_turn_speaker( speakers )

        if src_signals is not None:
            src_tokens = tmp

        #if self.args.freeze_encoder: 
        #    self.encoder.eval()
        encoder_out = self.encoder(src_signals, src_tokens, src_lengths, src_tok_lengths)

        if False: #self.scheduled_sampling:
            raise NotImplementedError
            #decoder_out = self._decode(
            #    prev_output_tokens,
            #    encoder_out=encoder_out,
            #    features_only=False,
            #    src_lengths=src_lengths,
            #    return_all_hiddens=False,
            #)
        else:
            decoder_out = self.decoder(
                prev_output_tokens,
                encoder_out=encoder_out,
                features_only=False,
                src_lengths=src_lengths,
                return_all_hiddens = False,
            )

        return decoder_out

    def freeze_encoder(self):

        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):

        for param in self.encoder.parameters():
            param.requires_grad = True

    def load_base_encoder(self, bsencoder): 

        base_encoder = torch.load( bsencoder )
        self.encoder.encoder.convolutions.load_state_dict( base_encoder.encoder.convolutions.state_dict() )
        for param in self.encoder.encoder.convolutions.parameters():
            param.requires_grad = False
        self.encoder.encoder.rnns.load_state_dict( base_encoder.encoder.rnns.state_dict() )
        for param in self.encoder.encoder.rnns.parameters():
            param.requires_grad = False

    def load_fairseq_encoder(self, fsencoder, task):

        if not hasattr(task.args, 'decoder'):
            return

        print(' - self encoder type: {}'.format(type(self.encoder)))
        print(' - self decoder type: {}'.format(type(self.decoder)))
        print(' ---')
        print(' args: {}'.format(task.args))

        '''models, _model_args = checkpoint_utils.load_model_ensemble(
            [fsencoder], 
            task=task,
        )
        fs_model = models[0]'''
        state = torch.load(fsencoder)
        
        #self.encoder.load_state_dict( state['model'].encoder )
        self.encoder = checkpoint_utils.load_pretrained_component_from_model(self.encoder, fsencoder) 

        if task.args.decoder == state['args'].decoder:
            if task.args.load_dictionary or task.args.source_dictionary:
                print(' * End2EndSLUModel: pre-initializing decoder weights from another model...')
                sys.stdout.flush()

                for k in state['model'].keys():
                    state['model'][k].requires_grad = False

                if isinstance(self.decoder, BasicDecoder):
                    '''SubstateDict = OrderedDict()
                    for key in state['model'].keys():
                        if 'decoder' in key and 'output_projection' not in key and 'output_norm' not in key:
                            SubstateDict[key[len('decoder')+1:]] = state['model'][key]
                        elif 'decoder' in key:
                            subkey = key[len('decoder')+1:]
                            SubstateDict[subkey] = self.decoder.state_dict()[subkey]
                    self.decoder.load_state_dict(SubstateDict, strict=True)'''

                    # Pre-initialize 'output_projection' matrix (self.decoder.output_projection.weight, second dimension is the size of the dictionary)
                    if task.args.target_dictionary:
                        tgt_dict = torch.load(task.args.target_dictionary)
                        for sym in tgt_dict.symbols:
                            tgt_idx = self.dict.index(sym)
                            if sym != unk_token and tgt_idx == self.dict.unk():
                                raise ValueError(' Symbol {} is unknown in current dictionary'.format(sym))
                            self.decoder.output_projection.weight.data[tgt_idx,:] = state['model']['decoder.output_projection.weight'].data[tgt_idx,:]
                            self.decoder.output_projection.bias.data[tgt_idx] = state['model']['decoder.output_projection.bias'].data[tgt_idx]
                            self.decoder.output_norm.weight.data[tgt_idx,:] = state['model']['decoder.output_norm.weight'].data[tgt_idx,:]
                            self.decoder.output_norm.bias.data[tgt_idx] = state['model']['decoder.output_norm.bias'].data[tgt_idx]
                    elif task.args.source_dictionary:
                        print(' * End2EndSLUModel, pre-initializing basic decoder with source dictionary')
                        sys.stdout.flush()

                        common = 0 
                        src_dict = torch.load(task.args.source_dictionary)
                        for sym in src_dict.symbols:
                            if sym in self.dict.symbols:
                                common += 1
                                src_idx = src_dict.index(sym)
                                tgt_idx = self.dict.index(sym) 

                                self.decoder.output_projection.weight.data[tgt_idx,:] = state['model']['decoder.output_projection.weight'].data[src_idx,:]
                                self.decoder.output_projection.bias.data[tgt_idx] = state['model']['decoder.output_projection.bias'].data[src_idx]
                                self.decoder.output_norm.weight.data[tgt_idx] = state['model']['decoder.output_norm.weight'].data[src_idx]
                                self.decoder.output_norm.bias.data[tgt_idx] = state['model']['decoder.output_norm.bias'].data[src_idx]

                        print(' * End2EndSLUModel, {} common symbols have been initialized'.format(common))
                        sys.stdout.flush()
                    else:
                        dd_size = state['model']['decoder.output_projection.weight'].size(0) 
                        self.decoder.output_projection.weight.data[:dd_size,:] = state['model']['decoder.output_projection.weight'].data
                        self.decoder.output_projection.bias.data[:dd_size] = state['model']['decoder.output_projection.bias'].data
                        self.decoder.output_norm.weight.data[:dd_size] = state['model']['decoder.output_norm.weight'].data
                        self.decoder.output_norm.bias.data[:dd_size] = state['model']['decoder.output_norm.bias'].data

                    print('   * BasicDecoder correctly pre-initialized.')
                    sys.stdout.flush()
                elif isinstance(self.decoder, ICASSPDecoderEx) or isinstance(self.decoder, LSTMDecoderEx):
                    SubstateDict = OrderedDict()
                    for key in state['model'].keys():
                        # NOTE: pre-initialize first all parameters except embeddings and output projection, which may have a different size with respect to the source domain
                        if 'decoder' in key and 'embed_tokens' not in key and 'fc_out' not in key:
                            SubstateDict[key[len('decoder')+1:]] = state['model'][key]
                        elif 'decoder' in key:  # NOTE: keep original initialization for embeddings and output projection
                            subkey = key[len('decoder')+1:]
                            SubstateDict[subkey] = self.decoder.state_dict()[subkey]

                    print('[DEBUG] pre-initializing decoder with the following state:')
                    print(SubstateDict.keys())
                    print('[DEBUG] ==========')
                    sys.stdout.flush()
                    self.decoder.load_state_dict(SubstateDict, strict=True)
 
                    if task.args.target_dictionary:
                        print(' * End2EndSLUModel, mapping embeddings and output projetction with target dictionary')
                        sys.stdout.flush()

                        #src_dict = torch.load(task.args.source_dictionary)
                        tgt_dict = torch.load(task.args.target_dictionary)
                        for sym in tgt_dict.symbols:
                            #src_idx = src_dict.index(sym)
                            tgt_idx = self.dict.index(sym)
                            if sym != unk_token and tgt_idx == self.dict.unk():
                                raise ValueError('Symbol {} is unknown in current dictionary'.format(sym))
                            self.decoder.embed_tokens.weight.data[tgt_idx,:] = state['model']['decoder.embed_tokens.weight'].data[tgt_idx,:]
                            if hasattr(self.decoder, 'fc_out'):
                                self.decoder.fc_out.weight.data[tgt_idx,:] = state['model']['decoder.fc_out.weight'].data[tgt_idx,:]
                                self.decoder.fc_out.bias.data[tgt_idx] = state['model']['decoder.fc_out.bias'].data[tgt_idx]
                    elif task.args.source_dictionary:
                        print(' * End2EndSLUModel, mapping embeddings and output projection with source dictionary')
                        sys.stdout.flush()

                        common = 0
                        # NOTE: pre-initialize embeddings and output projection for tokens in common to source and target domains.
                        src_dict = torch.load(task.args.source_dictionary)
                        for sym in src_dict.symbols:
                            if sym in self.dict.symbols:
                                common += 1
                                src_idx = src_dict.index(sym)
                                tgt_idx = self.dict.index(sym)

                                self.decoder.embed_tokens.weight.data[tgt_idx,:] = state['model']['decoder.embed_tokens.weight'].data[src_idx,:]
                                if hasattr(self.decoder, 'fc_out'):
                                    self.decoder.fc_out.weight.data[tgt_idx,:] = state['model']['decoder.fc_out.weight'].data[src_idx,:]
                                    self.decoder.fc_out.bias.data[tgt_idx] = state['model']['decoder.fc_out.bias'].data[src_idx]

                        print(' * End2EndSLUModel, {} common symbols have beed initialized'.format(common))
                        sys.stdout.flush()
                    else:
                        print(' * End2EndSLUModel, mapping embeddings and output projection as starting blocks')
                        sys.stdout.flush()

                        dd_size = state['model']['decoder.embed_tokens.weight'].size(0)
                        self.decoder.embed_tokens.weight.data[:dd_size,:] = state['model']['decoder.embed_tokens.weight'].data
                        if hasattr(self.decoder, 'fc_out'):
                            self.decoder.fc_out.weight.data[:dd_size,:] = state['model']['decoder.fc_out.weight'].data
                            self.decoder.fc_out.bias.data[:dd_size] = state['model']['decoder.fc_out.bias'].data

                    print('   * {} correctly pre-initialized.'.format(type(self.decoder)))
                    sys.stdout.flush()
            else:
                print(' * End2EndSLUModel: pre-initializing also decoder...')
                sys.stdout.flush()
                self.decoder = checkpoint_utils.load_pretrained_component_from_model(self.decoder, fsencoder)
        
        if self.args.freeze_encoder:
            print(' * End2EndSLUModel: freezing encoder parameters...')
            sys.stdout.flush()
            self.freeze_encoder()

        print(' ====================')
        sys.stdout.flush()

    def load_fairseq_decoder(self, fsencoder, task):

        print(' - self decoder type: {}'.format(type(self.decoder))) 
        print(' ---') 

        models, _model_args = checkpoint_utils.load_model_ensemble(
            [fsencoder], 
            task=task,
        )
        fs_model = models[0]
 
        print(' - loaded model decoder type: {}'.format(type(fs_model.decoder)))
        sys.stdout.flush()

        self.decoder.load_state_dict( fs_model.decoder.state_dict() )

        print(' ####################')
        sys.stdout.flush()

### REGISTER THE MODEL FOR USING IN FAIRSEQ ###
from fairseq.models import register_model_architecture

# The first argument to ``register_model_architecture()`` should be the name
# of the model we registered above (i.e., 'simple_lstm'). The function we
# register here should take a single argument *args* and modify it in-place
# to match the desired architecture.

@register_model_architecture('end2end_slu', 'end2end_slu_arch')
def end2end_slu_arch(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
    #args.load_encoder = getattr(args, 'load_encoder', 'None')
    #args.load_fairseq_encoder = getattr(args, 'load_fairseq_encoder', 'None')
    args.encoder_transformer_layers = getattr(args, 'encoder_transformer_layers', False)
    args.scheduled_sampling = getattr(args, 'scheduled_sampling', False)
    args.pyramid_hidden_layers = getattr(args, 'pyramid_hidden_layers', 1)

    args.decoder = getattr(args, 'decoder')
    if args.decoder == 'transformer':
        trans_ba(args)
    elif args.decoder == 'lstm':
        lstm_ba(args)
    elif not args.decoder in ['basic', 'ctclstm', 'icassp_lstm', 'end2end_slu', 'deliberation']:
        raise NotImplementedError































