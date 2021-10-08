#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import logging
import math
from itertools import groupby

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.tarc_utils import *
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@register_criterion('tarc_multitask_loss')
class TarcMultiTaskCriterion(FairseqCriterion):
    
    def __init__(self, args, task, sentence_avg):
        super().__init__(task)
        self.task = task
        self.args = args
    
        self.loss_function = CrossEntropyCriterion(task, False)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--use-source-side-sample-size", action="store_true", default=False,
            help=(
                "when compute average loss, using number of source tokens "
                + "as denominator. "
                + "This argument will be no-op if sentence-avg is used."
            ),
        )
    
    @classmethod
    def build_criterion(cls, args, task):

        return cls(args, task, True)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample["net_input"])
        t_idx = 0
        global_loss = None
        loss_data = None
        global_ntokens = 0
        global_nsentences = 0
        global_sample_size = 0

        toks_mt_target_l, char_mt_target_l = sample['target']
        toks_mt_target = split_on_sep(toks_mt_target_l, model.sequence_separator)
        char_mt_target = split_on_sep(char_mt_target_l, model.sequence_separator)

        assert len(net_output[0]) == len(toks_mt_target)
        assert len(net_output[0]) == len(char_mt_target)
        if (not self.args.token_sequences) and self.args.char_sequences:
            mt_target = char_mt_target
            sample_ntokens = sample['ntokens'][1]
        else:
            mt_target = toks_mt_target
            sample_ntokens = sample['ntokens'][0]
        if len(net_output[0]) > 1 and net_output[0][1] is not None:
            sample_ntokens = sample['ntokens'][1]

        o_idx = 0
        for o , attn, target in zip(net_output[0], net_output[1], mt_target):

            net_out = (o[0], attn)
            lprobs = model.get_normalized_probs(net_out, log_probs=True)
            (bsz, tgt_len) = target.size()
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = target.view(-1) 

            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index = self.loss_function.padding_idx,
                reduction = 'sum' if reduce else 'none'
            )
            ch_loss = None
            if o[1] is not None:
                ch_lprobs = model.get_normalized_probs( (o[1], attn), log_probs=True )
                ch_lprobs = ch_lprobs.view(-1, ch_lprobs.size(-1))
                char_tgt = char_mt_target[o_idx].view(-1)
                ch_loss = F.nll_loss(
                    ch_lprobs,
                    char_tgt,
                    ignore_index = self.loss_function.padding_idx,
                    reduction= 'sum' if reduce else 'none'
                )
            o_idx += 1

            if global_loss is None:
                if ch_loss is not None:
                    global_loss = (loss + ch_loss)
                    loss_data = (loss.data + ch_loss.data)
                else:
                    global_loss = loss
                    loss_data = loss.data
            else:
                if ch_loss is not None:
                    global_loss += (loss + ch_loss)
                    loss_data += (loss.data + ch_loss.data)
                else:
                    global_loss += loss
                    loss_data += loss.data

            sample_size = bsz if self.loss_function.sentence_avg else sample_ntokens[t_idx]
            global_ntokens += sample_ntokens[t_idx]
            global_nsentences += bsz
            global_sample_size += sample_size

            t_idx += 1

        logging_output = {
            'loss' : loss_data,
            'ntokens' : global_ntokens,
            'nsentences' : global_nsentences,
            'sample_size' : global_sample_size,
        }

        return global_loss, global_sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        
        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:

        return True

