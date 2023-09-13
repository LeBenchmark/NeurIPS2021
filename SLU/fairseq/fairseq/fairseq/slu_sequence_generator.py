# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder

class SLUSequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        max_len_a=0,
        max_len_b=200,
    ):
        """Generates translations of a given source sentence.
        A stripped-down version of sequence_generator only works with a batch size of 1
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.bos = tgt_dict.bos()

        self.blank = tgt_dict.set_blank()

        self.vocab_size = len(tgt_dict)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
        """
        model = EnsembleModel(models)
        return self._generate(model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        **kwargs
    ):
        model.eval()
        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        torch.set_printoptions(threshold=5000)
        src_len = src_tokens.size(1)

        max_len = min(
            int(self.max_len_a * src_len + self.max_len_b),
            # exclude the EOS marker
            model.max_decoder_positions() - 1,
        )

        # compute the encoder output  
        encoder_outs = model.forward_encoder(encoder_input)
        
        tokens = [self.bos]
        scores = []

        finished = False
        step = 0
        while not finished:
            # Forward step 
            print('Prefix:', tokens)
            lprobs, avg_attn_scores = model.forward_decoder(
                src_tokens.new_tensor(tokens).long().unsqueeze(0),
                encoder_outs, 
            )
            lprobs[lprobs != lprobs] = -math.inf
            # Forbid a few tokens:
            lprobs[:, self.pad] = -math.inf 
            lprobs[:, self.bos] = -math.inf
            lprobs[:, self.unk] = -math.inf
            lprobs[:, self.blank] = -math.inf

            # Pick a candidate with a score.
            cand_token_score, cand_token_index = lprobs.max(dim=-1)
            tokens.append(cand_token_index.item())
            scores.append(cand_token_score.item())
            if cand_token_index == self.eos or step > max_len:
                finished = True
            # increment the step:
            step += 1
            
        # could be better but this will do.
        return [[{
                'tokens': torch.Tensor(tokens[1:]), # Remove BOS
                'score': sum(scores)/len(scores),
                'positional_scores': torch.Tensor(scores),
                'alignment' : None,
            }]]


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )

        log_probs = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
            )
            log_probs.append(probs)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1] if len(decoder_out) > 1 else None
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)

