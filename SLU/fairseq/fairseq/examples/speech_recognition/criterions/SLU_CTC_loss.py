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
from fairseq.globals import *
from fairseq.criterions import FairseqCriterion, register_criterion
from examples.speech_recognition.data.data_utils import encoder_padding_mask_to_lengths
from examples.speech_recognition.utils.wer_utils import Code, EditDistance, Token
from fairseq.criterions.cross_entropy import CrossEntropyCriterion

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def arr_to_toks(arr):
    toks = []
    for a in arr:
        toks.append(Token(str(a), 0.0, 0.0))
    return toks


def compute_ctc_uer(logprobs, targets, input_lengths, target_lengths, blank_idx):
    """
        Computes utterance error rate for CTC outputs

        Args:
            logprobs: (Torch.tensor)  N, T1, D tensor of log probabilities out
                of the encoder
            targets: (Torch.tensor) N, T2 tensor of targets
            input_lengths: (Torch.tensor) lengths of inputs for each sample
            target_lengths: (Torch.tensor) lengths of targets for each sample
            blank_idx: (integer) id of blank symbol in target dictionary

        Returns:
            batch_errors: (float) errors in the batch
            batch_total: (float)  total number of valid samples in batch
    """
    batch_errors = 0.0
    batch_total = 0.0
    hyps = []
    refs = []
    for b in range(logprobs.shape[0]):
        predicted = logprobs[b][: input_lengths[b]].argmax(1).tolist()
        target = targets[b][: target_lengths[b]].tolist()

        # dedup predictions
        predicted = [p[0] for p in groupby(predicted)]

        # remove blanks
        nonblanks = []
        for p in predicted:
            if p != blank_idx:
                nonblanks.append(p)
        predicted = nonblanks

        #dedup predictions
        #predicted = [p[0] for p in groupby(predicted)]

        # compute the alignment based on EditDistance
        alignment = EditDistance(False).align(
            arr_to_toks(predicted), arr_to_toks(target)
        )

        # compute the number of errors
        # note that alignment.codes can also be used for computing
        # deletion, insersion and substitution error breakdowns in future
        for a in alignment.codes:
            if a != Code.match:
                batch_errors += 1
        batch_total += len(target)
        hyps.append( predicted )
        refs.append( target )

    return batch_errors, batch_total, hyps, refs


#@register_criterion("slubase_ctc_loss")
class SLUBaseCTCCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(task)
        self.blank_idx = task.blank_idx
        self.pad_idx = task.label_vocab.pad() #task.target_dictionary.pad()
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

        if self.args.label_dictionary:
            self.loss_reduction = 'none'
        else:
            self.loss_reduction = 'sum'
        self.dictionary = task.label_vocab #task.target_dictionary
        self.end_concept = self.dictionary.index(slu_end_concept_mark)
        if self.end_concept == self.dictionary.unk():
            raise ValueError('End concept symbol {} is expected to be defined in the dictionary'.format(slu_end_concept_mark))

        if args.slu_end2end:
            #self.aux_loss = CrossEntropyCriterion(task, False)
            self.aux_loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)
        else:
            self.aux_loss = None
        if hasattr(self.args, 'dec_loss') and self.args.dec_loss == 'nllloss':
            print(' * End2EndSLUCriterion: using NLLLoss for the decoder')
            sys.stdout.flush()

            self.loss_function = CrossEntropyCriterion(task, False)
        else:
            print(' * End2EndSLUCriterion: using CTC Loss for the decoder')
            sys.stdout.flush()

            self.loss_function = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)
        if self.args.asr_slu_loss:
            print(' * End2EndSLUCriterion, multi-task learning: using CTC Loss for the encoder')
            sys.stdout.flush()

            self.enc_loss_function = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)

        if self.args.asr_slu_loss:
            self.asr_loss_scale = 0.75

        if self.args.label_dictionary:
            self.token_loss_discount = 0.75

            print(' * SLU_CTC_loss criterion: loading label dictionary for label loss scaling (token discount: {})'.format(self.token_loss_discount))
            sys.stdout.flush()

            self.label_lst = self.load_label_dictionary() 
            self.prob_mask = torch.ones(len(task.target_dictionary)).fill_(self.token_loss_discount)
            for l in self.label_lst:
                self.prob_mask[l] = 1.0

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

    def load_label_dictionary(self):

        f = open(self.args.label_dictionary, encoding='utf-8')
        lines = f.readlines()
        f.close()

        label_dict = {}
        for l in lines:
            label_dict[l.strip()] = 1
        label_dict[slu_start_concept_mark] = 1
        label_dict[slu_end_concept_mark] = 1

        label_lst = []
        task_vocab = self.task.target_dictionary
        for l in label_dict:
            idx = task_vocab.index(l)
            assert idx != task_vocab.unk()
            label_lst.append( idx )

        return label_lst

    def forward(self, model, sample, reduce=True, log_probs=True):
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
        net_output = (net_output[0] / self.softmax_temperature, net_output[1])
        sem_output = None
        bound_lprobs = model.get_normalized_probs(net_output, log_probs=log_probs) 
        (N, T, C) = bound_lprobs.size() 
  
        bound_input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(bound_lprobs.device) 
        bound_target_lengths = sample["target_lengths"]
        #bound_input_lengths = bound_target_lengths
        bound_targets = sample["target"]    # B x T 

        if hasattr(self.args, 'dec_loss') and self.args.dec_loss == 'nllloss':
            loss = F.nll_loss(
                    bound_lprobs.view(-1, bound_lprobs.size(-1)),
                    bound_targets.view(-1),
                    ignore_index = self.loss_function.padding_idx,
                    reduction = self.loss_reduction,
            )

            if self.loss_reduction == 'none':
                # TMP FOR DEBUG CHECK
                NN, TT = bound_targets.size()
                assert NN == N and TT == T

                loss = loss.view(N, T)
                scores = F.log_softmax(bound_lprobs, -1)
                _, preds = torch.max( scores, -1 )
                discount_mask = torch.ones(N, T).fill_(self.token_loss_discount).to(loss.device)
                # Solution 1.
                #for k in range(N*T):
                #    #for j in range(T):
                #    i = k // T
                #    j = k % T
                #    if bound_targets[i,j] in self.label_lst:
                #        discount_mask[i,j] = 1.0

                # Solution 2.
                for idx in self.label_lst:
                    discount_mask[bound_targets == idx] = 1.0

                #print('[DEBUG] =======================================================')
                #print('[DEBUG] bound targets: {}'.format(bound_targets))
                #print('[DEBUG] loss discount mask: {}'.format(discount_mask))
                #print('[DEBUG] End2EndSLUCriterion, NLL loss shape: {}, ({} x {} = {})'.format(loss.size(), N, T, N*T))
                #print('[DEBUG] End2EndSLUCriterion, preds shape: {}'.format(preds.size()))
                #sys.stdout.flush()

                loss = loss * discount_mask
                loss = torch.sum(loss)

            bound_lprobs = bound_lprobs.transpose(0, 1) # NOTE: for the compute_ctc_uer
        else:
            # N T D -> T N D (F.ctc_loss expects this) 
            bound_lprobs = bound_lprobs.transpose(0, 1) 

            loss = self.loss_function(
                    bound_lprobs,
                    bound_targets,
                    bound_input_lengths,
                    bound_target_lengths,
            )

        if isinstance(net_output[1], dict) and 'draft_out' in net_output[1]:
            draft_out = net_output[1]['draft_out']
            draft_out = (draft_out[0] / self.softmax_temperature, draft_out[1])
            draft_probs = model.get_normalized_probs(draft_out, log_probs=log_probs)
            (dN, dT, dC) = draft_probs.size()
            draft_input_lengths = torch.full(size=(dN,), fill_value=dT, dtype=torch.long).to(draft_probs.device)
            draft_probs = draft_probs.transpose(0,1)

            draft_loss = self.loss_function(
                draft_probs,
                bound_targets,
                draft_input_lengths,
                bound_target_lengths,
            )

            loss = loss + draft_loss
        elif isinstance(net_output[1], dict) and 'asr_out' in net_output[1]:

            #net_output[1]['asr_out'] = net_output[1]['asr_out'] / self.softmax_temperature
            asr_net_out = (net_output[1]['asr_out'], None)
            asr_probs = model.get_normalized_probs(asr_net_out, log_probs=log_probs)
            (T, N, D) = asr_probs.size()
            asr_prob_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long).to(asr_probs.device)
            asr_inputs = net_output[1]['asr_in'].transpose(0, 1)
            asr_input_lengths = net_output[1]['asr_in_len']
            #asr_prob_lengths = asr_input_lengths

            asr_loss = self.enc_loss_function(
                asr_probs,
                asr_inputs,
                asr_prob_lengths,
                asr_input_lengths,
            )

            loss = self.asr_loss_scale * asr_loss + loss

        if False: #self.aux_loss is not None and sem_output is not None:
            aux_net_output = (sem_output, None)
            sem_lprobs = model.get_normalized_probs(aux_net_output, log_probs=log_probs)
            sem_targets, sem_target_lengths = extract_concepts(bound_targets, self.dictionary.bos(), self.dictionary.eos(), self.dictionary.pad(), self.end_concept, move_trail=False)
            (aB, aT, aC) = sem_lprobs.size()

            sem_input_lengths = torch.full(size=(aB,), fill_value=aT, dtype=torch.long).to(sem_lprobs.device)
            if batch_first:
                sem_lprobs = sem_lprobs.transpose(0, 1)

            aux_loss = self.aux_loss(
                sem_lprobs,
                sem_targets,
                sem_input_lengths,
                sem_target_lengths
            )
            loss = loss + aux_loss

        bound_lprobs = bound_lprobs.transpose(0, 1)  # T N D -> N T D
        
        errors, total, hyps, refs = compute_ctc_uer(
            bound_lprobs.detach(), bound_targets.detach(), bound_input_lengths, bound_target_lengths, self.blank_idx
        )
        pad_count = 0
        if self.args.padded_reference:
            pad_count = N * 2

        batch_error_rate = float(errors) #/ (float(total) - pad_count)
        ber = torch.Tensor( [batch_error_rate] )
 
        sample_size = sample["ntokens"] -pad_count # if the reference is padded with bos and eos, do not account for them.
        nframes = 0
        if isinstance(sample["net_input"]["src_lengths"], tuple):
            nframes = torch.sum( sample['net_input']['src_lengths'][0]).item()
        else:
            nframes = torch.sum( sample['net_input']['src_lengths']).item()
        logging_output = {
            "loss": utils.item( ber.data ) if reduce else ber.data, #utils.item(loss.data) if reduce else loss.data,
            "ctc": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample_size,
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "errors": errors,
            "total": total - pad_count,
            "predicted" : hyps,
            "target" : refs,
            "nframes": nframes
        }

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        errors = sum(log.get("errors", 0) for log in logging_outputs)
        total = sum(log.get("total", 0) for log in logging_outputs)
        nframes = sum(log.get("nframes", 0) for log in logging_outputs)
        agg_output = {
            "loss": loss_sum / sample_size, #/ math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "nframes": nframes,
            "sample_size": sample_size,
            "err": (errors * 100.0) / total,
        }
        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)
        return agg_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item( sum(log.get("loss", 0) for log in logging_outputs) )
        ctc_sum = utils.item( sum(log.get("ctc", 0) for log in logging_outputs) )
        ntokens = utils.item( sum(log.get("ntokens", 0) for log in logging_outputs) )
        nsentences = utils.item( sum(log.get("nsentences", 0) for log in logging_outputs) )
        sample_size = utils.item( sum(log.get("sample_size", 0) for log in logging_outputs) )
        errors = utils.item( sum(log.get("errors", 0) for log in logging_outputs) )
        total = utils.item( sum(log.get("total", 0) for log in logging_outputs) )
        nframes = utils.item( sum(log.get("nframes", 0) for log in logging_outputs) )

        metrics.log_scalar('loss', loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar('ctc', ctc_sum / sample_size, sample_size, round=5)
        metrics.log_scalar('err', (errors * 100.0) / total, total, round=4)
        metrics.log_scalar('ntokens', ntokens)
        metrics.log_scalar('nsentences', nsentences)
        metrics.log_scalar('nframes', nframes)
        metrics.log_scalar('sample_size', sample_size)


@register_criterion("slu_ctc_loss")
class SLUCTCCriterion(SLUBaseCTCCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        self.blank_idx = task.blank_idx
        self.eos_idx = task.label_vocab.eos()
        #self.loss_function = torch.nn.CTCLoss(blank=self.blank_idx, reduction=self.loss_reduction, zero_infinity=True)

        '''if args.slu_end2end:
            print(' * SLUCTCCriterion, using an auxiliary loss for end2end SLU')
            sys.stdout.flush()
            #self.aux_loss = CrossEntropyCriterion(task, False)
            self.aux_loss = torch.nn.CTCLoss(blank=self.blank_idx, reduction='sum', zero_infinity=True)
        else:
            self.aux_loss = None'''

        print(' - SLUCTCCriterion, initialized blank idx as {}, reduction = {}'.format(self.blank_idx, self.loss_reduction))
        sys.stdout.flush()

    @staticmethod
    def add_args(parser):
        pass

    @classmethod
    def build_criterion(cls, args, task):

        return SLUCTCCriterion(args, task)

