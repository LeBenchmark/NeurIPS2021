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

from examples.speech_recognition.criterions.SLU_CTC_loss import compute_ctc_uer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

@register_criterion('tarc_multitask_loss')
class TarcMultiTaskCriterion(FairseqCriterion):
    
    def __init__(self, args, task, sentence_avg):
        super().__init__(task)
        self.task = task
        self.args = args

        self.curr_epoch = 1
        self.max_epoch = args.max_epoch if hasattr(args, 'max_epoch') else 120
        self.softmax_temperature = self.args.softmax_temperature if self.args.rise_temperature_at_epoch == 1 else 1.0
        if self.args.rise_temperature_strategy not in ['fix', 'linear']:
            raise NotImplementedError('{} is not a valid temperature rising strategy, allowed: fix, linear')
        if self.args.rise_temperature_strategy == 'linear':
            assert self.max_epoch > self.args.rise_temperature_at_epoch
        self.rise_temperature_factor = 0.0 if self.args.rise_temperature_strategy == 'fix' else (self.args.softmax_temperature - self.args.softmax_start_temperature) / (self.max_epoch - self.args.rise_temperature_at_epoch)
        print('[DEBUG] criterion, max-epoch: {}'.format(self.max_epoch))
        print('[DEBUG] criterion, softmax-temperature: {}'.format(self.softmax_temperature))
        print('[DEBUG] criterion, softmax-start-temperature: {}'.format(self.args.softmax_start_temperature))
        print('[DEBUG] criterion, rise-temperature-at-epoch: {}'.format(self.args.rise_temperature_at_epoch))
        print('[DEBUG] criterion, rise-temperature-strategy: {}'.format(self.args.rise_temperature_strategy))
        print('[DEBUG] criterion, rise-temperature-factor: {}'.format(self.rise_temperature_factor))
        sys.stdout.flush()

        #self.L0 = []
        #self.Li = []
        #self.ri = []
        #self.n_updates = 0
        #self.reset_L0_after = 1068
        #self.alpha = 1.0

        if self.args.speech_input:
            dictionary = task.target_dictionary
            self.blank_idx = dictionary.blank()
            self.loss_function = []
            for l_idx, l in enumerate(self.args.sub_losses):
                print(' * TarcMultiTaskCriterion, setting loss@task-{} to {}'.format(l_idx, l))
                sys.stdout.flush()

                if l == 'nll_loss':
                    self.loss_function.append( (l, CrossEntropyCriterion(task, False)) )
                elif l == 'ctc':
                    print('   * blank index for CTC loss set to {}'.format(self.blank_idx))
                    sys.stdout.flush()

                    self.loss_function.append( (l, torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)) )
                else:
                    raise NotImplementedError('Loss name {} is not supported'.format(l))
        else:
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

        if self.curr_epoch != model.get_curr_epoch():
            print('[DEBUG] criterion, curr_epoch: {}'.format(self.curr_epoch))
            print('[DEBUG] criterion, model epoch: {}'.format(model.get_curr_epoch()))
            print('[DEBUG] criterion, rise-temperature-at-epoch: {}'.format(self.args.rise_temperature_at_epoch))
            print('[DEBUG] criterion, temperature: {}'.format(self.softmax_temperature))
            sys.stdout.flush()

        if self.curr_epoch != model.get_curr_epoch() and self.args.rise_temperature_at_epoch > 1 and self.curr_epoch >= self.args.rise_temperature_at_epoch:
            if self.args.rise_temperature_strategy == 'fix' and self.curr_epoch == self.args.rise_temperature_at_epoch:
                self.softmax_temperature = self.args.softmax_temperature
                print('[DEBUG] criterion, softmax-temperature changed: {}'.format(self.softmax_temperature))
                sys.stdout.flush()
            elif self.args.rise_temperature_strategy == 'linear':
                self.softmax_temperature = self.args.softmax_start_temperature + (self.curr_epoch - self.args.rise_temperature_at_epoch) * self.rise_temperature_factor
                print('[DEBUG] criterion, softmax-temperature changed: {}'.format(self.softmax_temperature))
                sys.stdout.flush()
        self.curr_epoch = model.get_curr_epoch()

        net_output = model(**sample["net_input"]) 
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

        t_idx = 0  
        losses = []
        ERs = []
        if self.args.speech_input and len(self.loss_function) > 1:
            assert len(net_output[0]) == len(self.loss_function)
        loss_idx = 0
        for o , attn, target in zip(net_output[0], net_output[1], mt_target):
 
            net_out = (o[0] / self.softmax_temperature, attn)
            lprobs = model.get_normalized_probs(net_out, log_probs=True)
            bsz, _ = target.size() 

            (N, T, C) = lprobs.size()
            bound_input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(lprobs.device) 
            bound_target_lengths = sample["target_lengths"][0][loss_idx]
            if self.args.speech_input:
                errors, total, hyps, refs = compute_ctc_uer(
                        lprobs.detach(), target.detach(), bound_input_lengths, bound_target_lengths, self.blank_idx
                )
                ERs.append( (errors, total) )
 
                idx = loss_idx if len(self.loss_function) > 1 else 0
                if self.loss_function[idx][0] == 'nll_loss':

                    lprobs = lprobs.view(-1, lprobs.size(-1))
                    target = target.view(-1)
                    loss = F.nll_loss(
                        lprobs,
                        target,
                        ignore_index = self.loss_function[idx][1].padding_idx,
                        reduction = 'sum' if reduce else 'none'
                    )
                else:

                    #bound_lprobs = model.get_normalized_probs(net_out, log_probs=True)  
                    #(N, T, C) = lprobs.size()
                    #bound_input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(lprobs.device) 
                    #bound_target_lengths = sample["target_lengths"][0][loss_idx]

                    lprobs = lprobs.transpose(0, 1) 
                    loss = self.loss_function[idx][1](
                        lprobs,
                        target,
                        bound_input_lengths,
                        bound_target_lengths,
                    )
            else:
                #print('[DEBUG] computing loss on lprobs and target of shape (reduce={}): {} vs. {}'.format(reduce, lprobs.size(), target.size()))
                #sys.stdout.flush()

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
                ch_lprobs = model.get_normalized_probs( (o[1] / self.softmax_temperature, attn), log_probs=True )
                ch_lprobs = ch_lprobs.view(-1, ch_lprobs.size(-1))
                char_tgt = char_mt_target[t_idx].view(-1)
                ch_loss = F.nll_loss(
                    ch_lprobs,
                    char_tgt,
                    ignore_index = self.loss_function.padding_idx,
                    reduction= 'sum' if reduce else 'none'
                )
            t_idx += 1

            losses.append( (loss, ch_loss) )
            loss_idx += 1

        for t_idx, ll in enumerate(losses):
            loss, ch_loss = ll
            if global_loss is None:
                if ch_loss is not None:
                    global_loss = loss + ch_loss #(self.ri[t_idx] ** self.alpha) * (loss + ch_loss)
                    loss_data = (loss.data + ch_loss.data)
                else:
                    global_loss = loss #(self.ri[t_idx] ** self.alpha) * loss
                    loss_data = loss.data
            else:
                if ch_loss is not None:
                    global_loss += (loss + ch_loss) #(self.ri[t_idx] ** self.alpha) * (loss + ch_loss)
                    loss_data += (loss.data + ch_loss.data)
                else:
                    global_loss += loss #(self.ri[t_idx] ** self.alpha) * loss
                    loss_data += loss.data
 
            sample_size = sample_ntokens[t_idx]
            if self.args.speech_input:
                loss_idx = t_idx if len(self.loss_function) > 1 else 0
                if self.loss_function[loss_idx][0] == 'nll_loss' and self.loss_function[loss_idx][1].sentence_avg:
                    sample_size = bsz
            elif self.loss_function.sentence_avg:
                sample_size = bsz
            global_ntokens += sample_ntokens[t_idx]
            global_nsentences += bsz
            global_sample_size += sample_size 

        logging_output = {
            'loss' : loss_data,
            'ntokens' : global_ntokens,
            'nsentences' : global_nsentences,
            'sample_size' : global_sample_size,
        }

        if self.args.speech_input:
            for i in range(len(losses)):
                loss_idx = i if len(self.loss_function) > 1 else 0 

                key = self.loss_function[loss_idx][0]
                if key not in logging_output:
                    logging_output[key] = [losses[i][0].item()]
                else:
                    logging_output[key].append( losses[i][0].item() )

                if 'errors' not in logging_output:
                    logging_output['errors'] = [ERs[i][0]]
                    logging_output['total_tokens'] = [ERs[i][1]]
                else:
                    logging_output['errors'].append(ERs[i][0])
                    logging_output['total_tokens'].append(ERs[i][1])

        return global_loss, global_sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))

        total_tokens = None
        if 'total_tokens' in logging_outputs[0]:
            total_tokens = [0] * len(logging_outputs[0]['total_tokens'])

            for i in range(len(total_tokens)):
                for log in logging_outputs:
                    total_tokens[i] += log['total_tokens'][i]

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=4)
        if 'ctc' in logging_outputs[0]:
            scores = [0.0] * len(logging_outputs[0]['ctc'])
            errors = [0] * len(scores)

            for i in range(len(scores)):
                for log in logging_outputs: 
                    scores[i] += log['ctc'][i]
                    errors[i] += log['errors'][i]
            
            for i in range(len(scores)):
                metrics.log_scalar('subloss-ctc-' + str(i), scores[i] / total_tokens[i] / math.log(2), total_tokens[i], round=4)
                metrics.log_scalar('ER@' + str(i), (errors[i]*100.0)/total_tokens[i], total_tokens[i], round=4 )

        if 'nll_loss' in logging_outputs[0]:
            scores = [0.0] * len(logging_outputs[0]['nll_loss'])
            errors = [0] * len(scores)
           
            for i in range(len(scores)):
                for log in logging_outputs: 
                    scores[i] += log['nll_loss'][i]
                    errors[i] += log['errors'][i]
 
            for i in range(len(scores)):
                metrics.log_scalar('subloss-NLLloss@' + str(i), scores[i] / total_tokens[i] / math.log(2), total_tokens[i], round=4)
                metrics.log_scalar('ER@' + str(i), (errors[i]*100.0)/total_tokens[i], total_tokens[i], round=4 )

        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:

        return True

