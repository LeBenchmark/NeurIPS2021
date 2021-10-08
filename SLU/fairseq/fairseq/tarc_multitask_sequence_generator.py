# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import math
import torch
import numpy as np

from fairseq import search, utils
from fairseq.tarc_utils import *
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.TarcMultiTaskModels import TarcMultiTaskModel

class TarcSequenceGenerator(object):
    def __init__(
        self,
        tgt_dict,
        beam_size=1,
        max_len_a=0,
        max_len_b=200,
        min_len=1,
        normalize_scores=True,
        len_penalty=1.,
        unk_penalty=0.,
        retain_dropout=False,
        temperature=1.,
        match_source_len=False,
        no_repeat_ngram_size=0,
        search_strategy=None,
        eos=None
    ):
        """Generates translations of a given source sentence.

        Args:
            tgt_dict (~fairseq.data.Dictionary): target dictionary
            beam_size (int, optional): beam width (default: 1)
            max_len_a/b (int, optional): generate sequences of maximum length
                ax + b, where x is the source length
            min_len (int, optional): the minimum length of the generated output
                (not including end-of-sentence)
            normalize_scores (bool, optional): normalize scores by the length
                of the output (default: True)
            len_penalty (float, optional): length penalty, where <1.0 favors
                shorter, >1.0 favors longer sentences (default: 1.0)
            unk_penalty (float, optional): unknown word penalty, where <0
                produces more unks, >0 produces fewer (default: 0.0)
            retain_dropout (bool, optional): use dropout when generating
                (default: False)
            temperature (float, optional): temperature, where values
                >1.0 produce more uniform samples and values <1.0 produce
                sharper samples (default: 1.0)
            match_source_len (bool, optional): outputs should match the source
                length (default: False)
        """
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos() if eos is None else eos
        self.bos = tgt_dict.bos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_penalty = unk_penalty
        self.retain_dropout = retain_dropout
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.no_repeat_ngram_size = no_repeat_ngram_size
        assert temperature > 0, '--temperature must be greater than 0'

        self.search = (
            search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        )


    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = TarcEnsembleModel(models)
        return self._generate(model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        tgt_tok_bounds = sample['net_input']['tgt_tok_bounds']
        sort_order = sample['net_input']['sort_order']

        char_flag = (not model.models[0].args.token_sequences) and model.models[0].args.char_sequences
        toks_src_tokens, char_src_tokens = encoder_input['src_tokens']

        if len(model.models[0].encoder.encoders) > 1:
            toks_src_tokens = split_on_sep(toks_src_tokens, model.models[0].sequence_separator)
            char_src_tokens = split_on_sep(char_src_tokens, model.models[0].sequence_generator)
        else:
            toks_src_tokens = [toks_src_tokens]
            char_src_tokens = [char_src_tokens]

        toks_src_lengths = (toks_src_tokens[0].ne(self.eos) & toks_src_tokens[0].ne(self.pad)).long().sum(dim=1)
        char_src_lengths = (char_src_tokens[0].ne(self.eos) & char_src_tokens[0].ne(self.pad)).long().sum(dim=1)
        src_lengths = char_src_lengths if char_flag else toks_src_lengths

        input_size = toks_src_tokens[0].size() if not char_flag else char_src_tokens[0].size()
        src_len = input_size[1]
        beam_size = self.beam_size
        num_of_tasks = model.models[0].num_of_tasks
        bsz_val = input_size[0]
        bsz = [input_size[0] for t_idx in range(num_of_tasks)]

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!' 

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz[0]).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(toks_src_tokens[0].device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers 
        scores = [toks_src_tokens[0].new(bsz[0] * beam_size, max_len + 1).float().fill_(0) for i in range(num_of_tasks)]
        scores_buf = [scores[0].clone() for i in range(num_of_tasks)]
        tokens = [toks_src_tokens[0].new(bsz[0] * beam_size, max_len + 2).long().fill_(self.pad) for i in range(num_of_tasks)]
        tokens_buf = [tokens[0].clone() for i in range(num_of_tasks)]
        for i in range(num_of_tasks):
            tokens[i][:, 0] = self.eos if bos_token is None else bos_token
        attn = [None] * num_of_tasks
        attn_buf = [None] * num_of_tasks

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples. 
        blacklist = [toks_src_tokens[0].new_zeros(bsz[0], beam_size).eq(-1) for t_idx in range(num_of_tasks)]  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[[] for i in range(bsz[0])] for j in range(num_of_tasks)]
        finished = [[False for i in range(bsz[0])] for j in range(num_of_tasks)]
        num_remaining_sent = [bsz[0] for i in range(num_of_tasks)]

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = [(torch.arange(0, bsz[0]) * beam_size).unsqueeze(1).type_as(tokens[0]) for i in range(num_of_tasks)]
        cand_offsets = [torch.arange(0, cand_size).type_as(tokens[0]) for i in range(num_of_tasks)]

        # helper function for allocating buffers on the fly
        buffers = {}
 
        def buffer(name, type_of=tokens[0]):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()   # Allocate a new empty tensor
            return buffers[name]
 
        def is_finished(task, sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[task][sent]) <= beam_size
            if len(finalized[task][sent]) == beam_size or step == max_len:
                return True
            return False
 
        def finalize_hypos(task, step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel() 

            # clone relevant token and attention tensors
            tokens_clone = tokens[task].index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn[task].index_select(0, bbsz_idx)[:, :, 1:step+2] if attn[task] is not None else None

            # compute scores per token position
            pos_scores = scores[task].index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin: List[int] = []
            prev = 0
            for f in finished[task]:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]
                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[task][sent]) < beam_size:
                    finalized[task][sent].append(get_hypo())

            newly_finished: List[int] = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[task][sent] and is_finished(task, sent, step, unfin_idx):
                    finished[task][sent] = True
                    newly_finished.append(unfin_idx)
             
            return newly_finished

        for m in model.models:
            m.decoder.reset_hidden_states() 
        for task_idx in range(num_of_tasks):
            reorder_state = None
            batch_idxs = None
            dec_shapes = None
            for m in model.models:
                m.decoder.set_active_tasks(task_idx, task_idx+1)
            for step in range(max_len + 1):  # one extra step for EOS marker
                # reorder decoder internal states based on the prev choice of beams
                if reorder_state is not None:
                    if batch_idxs is not None:
                        # update beam indices to take into account removed sentences
                        corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                        reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                    model.reorder_incremental_state(reorder_state)
                    encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state) 

                mt_target = [t[:, :step+1] for t in tokens]
                mtt_szs = [t.size(0) for t in mt_target]
                mt_target = concat_with_sep(mt_target, model.models[0].sequence_separator, shapes=dec_shapes)
                tok_char_mt_target = [mt_target, mt_target] 

                lprobs, avg_attn_scores = model.forward_decoder(
                    tok_char_mt_target, shapes=dec_shapes, tgt_tok_bounds=tgt_tok_bounds, sort_order=sort_order, encoder_outs=encoder_outs, temperature=self.temperature,
                )
 
                lprobs[task_idx][lprobs[task_idx] != lprobs[task_idx]] = -math.inf
                lprobs[task_idx][:, self.pad] = -math.inf  # never select pad
                lprobs[task_idx][:, self.unk] -= self.unk_penalty  # apply unk penalty

                # handle max length constraint
                if step >= max_len:
                    lprobs[task_idx][:, :self.eos] = -math.inf
                    lprobs[task_idx][:, self.eos + 1:] = -math.inf 

                # handle prefix tokens (possibly with different lengths) 
                if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                    raise NotImplementedError(' TarcSequenceGenerator: prefix tokens are not supported with multi-task models')
                    prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                    prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                    prefix_mask = prefix_toks.ne(self.pad)
                    lprobs[prefix_mask] = -math.inf
                    lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                        -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                    )
                    # if prefix includes eos, then we should make sure tokens and
                    # scores are the same across all beams
                    eos_mask = prefix_toks.eq(self.eos)
                    if eos_mask.any():
                        # validate that the first beam matches the prefix
                        first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                        eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                        target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                        assert (first_beam == target_prefix).all()

                        def replicate_first_beam(tensor, mask):
                            tensor = tensor.view(-1, beam_size, tensor.size(-1))
                            tensor[mask] = tensor[mask][:, :1, :]
                            return tensor.view(-1, tensor.size(-1))

                        # copy tokens, scores and lprobs from the first beam to all beams
                        tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                        scores = replicate_first_beam(scores, eos_mask_batch_dim)
                        lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
                elif step < self.min_len:
                    # minimum length constraint (does not apply if using prefix_tokens) 
                    lprobs[task_idx][:, self.eos] = -math.inf

                if self.no_repeat_ngram_size > 0:
                    # for each beam and batch sentence, generate a list of previous ngrams
                    gen_ngrams = [[{} for bbsz_idx in range(bsz[t_idx] * beam_size)] for t_idx in range(num_of_tasks)]
                    for bbsz_idx in range(bsz[task_idx] * beam_size):
                        gen_tokens = tokens[task_idx][bbsz_idx].tolist()
                        for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                            gen_ngrams[task_idx][bbsz_idx][tuple(ngram[:-1])] = \
                                    gen_ngrams[task_idx][bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

                # Record attention scores 
                if type(avg_attn_scores[task_idx]) is list:
                    avg_attn_scores[task_idx] = avg_attn_scores[task_idx][0]
                if avg_attn_scores[task_idx] is not None:
                    if attn[task_idx] is None: 
                        attn[task_idx] = scores[task_idx].new(bsz[task_idx] * beam_size, avg_attn_scores[task_idx].size(1), max_len + 2)
                        attn_buf[task_idx] = attn[task_idx].clone()
                    attn[task_idx][:, :, step + 1].copy_(avg_attn_scores[task_idx])

                scores[task_idx] = scores[task_idx].type_as(lprobs[task_idx])
                scores_buf[task_idx] = scores_buf[task_idx].type_as(lprobs[task_idx])

                # These two may actually be created and used in the for loop marked with (**) 
                self.search.set_src_lengths(src_lengths)
                if self.no_repeat_ngram_size > 0:
                    def calculate_banned_tokens(bbsz_idx):
                        # before decoding the next token, prevent decoding of ngrams that have already appeared
                        ngram_index = [tuple(tokens[t_idx][bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist()) for t_idx in range(num_of_tasks)]
                        return [gen_ngrams[t_idx][bbsz_idx].get(ngram_index[t_idx], []) for t_idx in range(num_of_tasks)]

                    if step + 2 - self.no_repeat_ngram_size >= 0:
                        # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                        banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz_val * beam_size)]
                    else:     
                        banned_tokens = [[[] for i in range(num_of_tasks)] for bbsz_idx in range(bsz_val * beam_size)]

                    for bbsz_idx in range(bsz_val * beam_size):
                        lprobs[task_idx][bbsz_idx, banned_tokens[bbsz_idx][task_idx]] = -math.inf
                
                eos_bbsz_idx = buffer('eos_bbsz_idx')
                eos_scores = buffer('eos_scores', type_of=scores[0])

                cand_scores, cand_indices, cand_beams = self.search.step(
                    step,
                    lprobs[task_idx].view(bsz[task_idx], -1, self.vocab_size),
                    scores[task_idx].view(bsz[task_idx], beam_size, -1)[:, :, :step],
                )

                # cand_bbsz_idx contains beam indices for the top candidate
                # hypotheses, with a range of values: [0, bsz*beam_size),
                # and dimensions: [bsz, cand_size]
                cand_bbsz_idx = cand_beams.add(bbsz_offsets[task_idx])

                # finalize hypotheses that end in eos, except for blacklisted ones
                # or candidates with a score of -inf
                eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
                eos_mask[:, :beam_size][blacklist[task_idx]] = 0

                # only consider eos when it's among the top beam_size indices
                torch.masked_select(
                    cand_bbsz_idx[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_bbsz_idx,
                )

                finalized_sents: List[int] = []
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(
                        cand_scores[:, :beam_size],
                        mask=eos_mask[:, :beam_size],
                        out=eos_scores,
                    )
                    if num_remaining_sent[task_idx] > 0: 
                        finalized_sents = finalize_hypos(task_idx, step, eos_bbsz_idx, eos_scores) 
                        num_remaining_sent[task_idx] -= len(finalized_sents)
 
                remain_sent_all = sum(num_remaining_sent) 
                assert remain_sent_all >= 0
                if num_remaining_sent[task_idx] == 0:
                    break
                assert step < max_len
 
                if len(finalized_sents) > 0: 
                    if num_remaining_sent[task_idx] > 0:
                        bsz[task_idx] -= len(finalized_sents)

                        # construct batch_idxs which holds indices of batches to keep for the next pass
                        batch_mask = cand_indices.new_ones(bsz_val)
                        batch_mask[cand_indices.new(finalized_sents)] = 0
                        batch_idxs = batch_mask.nonzero().squeeze(-1)

                        eos_mask = eos_mask[batch_idxs]
                        cand_beams = cand_beams[batch_idxs] 
                        bbsz_offsets[task_idx].resize_(bsz[task_idx], 1) 
                        cand_bbsz_idx = cand_beams.add(bbsz_offsets[task_idx])
                        cand_scores = cand_scores[batch_idxs]
                        cand_indices = cand_indices[batch_idxs] 

                        if prefix_tokens is not None:
                            raise NotImplementedError(' TarcSequenceGenerator: prefix tokens not supported with multi-task model')
                            prefix_tokens = prefix_tokens[batch_idxs]
                        src_lengths = src_lengths[batch_idxs]
                        blacklist[task_idx] = blacklist[task_idx][batch_idxs]
 
                        scores[task_idx] = scores[task_idx].view(bsz_val, -1)[batch_idxs].view(bsz[task_idx] * beam_size, -1)
                        scores_buf[task_idx].resize_as_(scores[task_idx])
                        tokens[task_idx] = tokens[task_idx].view(bsz_val, -1)[batch_idxs].view(bsz[task_idx] * beam_size, -1)
                        tokens_buf[task_idx].resize_as_(tokens[task_idx])
                        if attn[task_idx] is not None:
                            attn[task_idx] = attn[task_idx].view(bsz_val, -1)[batch_idxs].view(bsz[task_idx] * beam_size, attn.size(1), -1)
                            attn_buf[task_idx].resize_as_(attn[task_idx]) 
                else:
                    batch_idxs = None
  
                # Set active_mask so that values > cand_size indicate eos or
                # blacklisted hypos and values < cand_size indicate candidate
                # active hypos. After this, the min values per row are the top
                # candidate active hypos.
                active_mask = buffer('active_mask')
                eos_mask[:, :beam_size] |= blacklist[task_idx]
                torch.add(
                    eos_mask.type_as(cand_offsets[task_idx]) * cand_size,
                    cand_offsets[task_idx][:eos_mask.size(1)],
                    out=active_mask,
                )

                # get the top beam_size active hypotheses, which are just the hypos
                # with the smallest values in active_mask
                active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
                torch.topk(
                    active_mask, k=beam_size, dim=1, largest=False,
                    out=(new_blacklist, active_hypos)
                )

                # update blacklist to ignore any finalized hypos
                blacklist[task_idx] = new_blacklist.ge(cand_size)[:, :beam_size]
                assert (~blacklist[task_idx]).any(dim=1).all()

                active_bbsz_idx = buffer('active_bbsz_idx')
                torch.gather(
                    cand_bbsz_idx, dim=1, index=active_hypos,
                    out=active_bbsz_idx,
                )
                active_scores = torch.gather(
                    cand_scores, dim=1, index=active_hypos,
                    out=scores[task_idx][:, step].view(bsz[task_idx], beam_size),
                )

                active_bbsz_idx = active_bbsz_idx.view(-1)
                active_scores = active_scores.view(-1)

                # copy tokens and scores for active hypotheses
                torch.index_select(
                    tokens[task_idx][:, :step + 1], dim=0, index=active_bbsz_idx,
                    out=tokens_buf[task_idx][:, :step + 1],
                )
                torch.gather(
                    cand_indices, dim=1, index=active_hypos,
                    out=tokens_buf[task_idx].view(bsz[task_idx], beam_size, -1)[:, :, step + 1],
                )
                if step > 0:
                    torch.index_select(
                        scores[task_idx][:, :step], dim=0, index=active_bbsz_idx,
                        out=scores_buf[task_idx][:, :step],
                    )
                torch.gather(
                    cand_scores, dim=1, index=active_hypos,
                    out=scores_buf[task_idx].view(bsz[task_idx], beam_size, -1)[:, :, step],
                )

                # copy attention for active hypotheses
                if attn[task_idx] is not None:
                    torch.index_select(
                        attn[task_idx][:, :, :step + 2], dim=0, index=active_bbsz_idx,
                        out=attn_buf[task_idx][:, :, :step + 2],
                    )

                # swap buffers
                tokens[task_idx], tokens_buf[task_idx] = tokens_buf[task_idx], tokens[task_idx]
                scores[task_idx], scores_buf[task_idx] = scores_buf[task_idx], scores[task_idx]
                if attn[task_idx] is not None:
                    attn[task_idx], attn_buf[task_idx] = attn_buf[task_idx], attn[task_idx]

                # reorder incremental state in decoder
                reorder_state = active_bbsz_idx 

            # sort by score descending  
            for sent in range(len(finalized[task_idx])):
                finalized[task_idx][sent] = sorted(finalized[task_idx][sent], key=lambda r: r['score'], reverse=True)

        decoding_output = [[] for i in range(bsz_val)]
        # concatenate output sequences from different tasks
        sequence_separator = model.models[0].sequence_separator
        for i in range(bsz_val):
            beam_hyp = []
            for j in range(beam_size):
                fin_tokens = []
                fin_score = 0.0
                fin_attention = []
                fin_pos_scores = []
                for t_idx in range(num_of_tasks):
                    fin_tokens.append( finalized[t_idx][i][j]['tokens'] )
                    fin_score += finalized[t_idx][i][j]['score']
                    fin_attention.append( finalized[t_idx][i][j]['attention'] )
                    fin_pos_scores.append( finalized[t_idx][i][j]['positional_scores'] )

                beam_hyp.append( {
                    'tokens' : concat_with_sep(fin_tokens, sequence_separator),
                    'score' : fin_score,
                    'attention' : fin_attention[0] if fin_attention[0] is not None else None,
                    'alignment' : None,
                    'positional_scores' : torch.cat( fin_pos_scores, -1 ),
                } )
            decoding_output[i] = sorted(beam_hyp, key=lambda r: r['score'], reverse=True)

        return decoding_output


class TarcEnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
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
    def forward_decoder(self, tokens, shapes=None, tgt_tok_bounds=None, sort_order=None, encoder_outs=None, temperature=1.):
        if len(self.models) == 1:
            return self._decode_one(
                tokens,
                shapes,
                tgt_tok_bounds,
                sort_order,
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
                shapes,
                tgt_tok_bounds,
                sort_order,
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
                    for att_idx, att in enumerate(attn):
                        avg_attn[att_idx].add_(att)

        num_tasks = len( log_probs[0] )
        avg_probs = [torch.logsumexp(torch.stack([probs[t_idx] for probs in log_probs], dim=0), dim=0) - math.log(len(self.models)) for t_idx in range( num_tasks )]
        if avg_attn is not None:
            for att_idx in range( num_tasks ):
                avg_attn[att_idx].div_(len(self.models))
        return avg_probs, avg_attn

    def _decode_one(
        self, tokens, shapes, tgt_tok_bounds, sort_order, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, shapes=shapes, tgt_tok_bounds=tgt_tok_bounds, sort_order=sort_order, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, shapes=shapes, tgt_tok_bounds=tgt_tok_bounds, sort_order=sort_order, encoder_out=encoder_out))
 
        for dout in decoder_out[0]:
            dout = dout[0][:, -1:, :]
            if temperature != 1.:
                dout[0].div_(temperature)
        attn = decoder_out[1] if len(decoder_out) > 1 else None
        res_attn = []
        for att in attn: 
            if type(att) is dict:
                att = att.get('attn', None)
            if type(att) is list:
                att = att[0]
            if att is not None:
                att = att[:, -1, :]
            res_attn.append( att )

        res_probs = []
        for dout in zip(decoder_out[0], decoder_out[1]):
            probs = model.get_normalized_probs((dout[0][0], dout[1]), log_probs=log_probs)
            probs = probs[:, -1, :]
            res_probs.append( probs )
        return res_probs, res_attn

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


