# -*- coding: utf-8 -*-

# Code for adding the End2End SLU task into fairseq

import os
import re
import sys
import torch
import math
import numpy as np
from scipy.io import wavfile
from scipy import signal

from fairseq.globals import *
from fairseq.dstc_utils import turn_transcripts_and_sem, clean_sem
from fairseq.data import Dictionary, End2EndSLUDataset, FairseqDataset, data_utils, iterators
from fairseq.tasks import FairseqTask, register_task

import typing
from typing import List

import examples.speech_recognition.criterions.SLU_CTC_loss as slu_ctc

import whisper
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram

#from carbontracker.tracker import CarbonTracker
from codecarbon import EmissionsTracker

from fairseq.models.roberta import RobertaModel, CamembertModel

_ADVANCED_SPK_MARK_ = False

class SLUDictionary(Dictionary):
    """Dictionary wrapping to initialize a dictionary from a raw python dictionary"""

    def __init__(
        self,
        pad=pad_token,
        eos=EOS_tag,
        unk=unk_token,
        bos=SOS_tag,
        extra_special_symbols=None,
    ):
        super(SLUDictionary, self).__init__(pad, eos, unk, bos, extra_special_symbols)
        # The above statement does:
        #   1. self.unk_word, self.pad_word, self.eos_word = unk, pad, eos
        #   2.  self.bos_index = self.add_symbol(bos)
        #       self.pad_index = self.add_symbol(pad)
        #       self.eos_index = self.add_symbol(eos)
        #       self.unk_index = self.add_symbol(unk)
        #   3. Add the other special-symbols

        self.bos_word = bos

    def init_from_python_dict(self, dict):

        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = 0

        for v, k in enumerate(dict):
            self.add_symbol(k)

        self.bos_index = self.add_symbol(self.bos_word)
        self.pad_index = self.add_symbol(self.pad_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.unk_index = self.add_symbol(self.unk_word)

    def extend_from_python_dict(self, dict):
        for v, k in enumerate(dict):
            self.add_symbol(k)

        # just in case...
        self.bos_index = self.add_symbol(self.bos_word)
        self.pad_index = self.add_symbol(self.pad_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.unk_index = self.add_symbol(self.unk_word)

    def reset(self):
        self.symbols = []
        self.count = []
        self.indices = {}
        self.nspecial = 0
        self.bos_word = self.pad_word = self.eos_word = self.unk_word = ''
        self.bos_index = self.pad_index = self.eos_index = self.unk_index = -1

    def set_nspecial_(self):
        self.nspecial = len(self.symbols)

    def set_bos_(self, bos=SOS_tag):
        self.bos_word = bos
        self.bos_index = self.add_symbol(self.bos_word)
        return self.bos_index

    def set_pad_(self,pad=pad_token):
        self.pad_word = pad
        self.pad_index = self.add_symbol(self.pad_word)
        return self.pad_index

    def set_eos_(self, eos=EOS_tag):
        self.eos_word = eos
        self.eos_index = self.add_symbol(self.eos_word)
        return self.eos_index

    def set_unk_(self, unk=unk_token):
        self.unk_word = unk
        self.unk_index = self.add_symbol(self.unk_word)
        return self.unk_index

    def set_blank(self, blank=blank_token):
        self.blank_word = blank
        return self.add_symbol(blank)

    def blank(self):
        idx = self.index(self.blank_word)
        if idx == self.unk():
            raise ValueError('Blank symbol {} is not defined in the dictionary'.format(self.blank_word))
        return idx

def init_slu_dictionary(args) -> SLUDictionary:

    slu_dict = SLUDictionary(
        pad=pad_token,
        eos=EOS_tag,
        unk=unk_token,
        bos=SOS_tag,
        extra_special_symbols=slu_special_symbols
    )

    slu_dict.reset() 
    slu_dict.set_blank(blank_token)
    slu_dict.set_pad_(pad_token)
    slu_dict.set_eos_(EOS_tag)
    slu_dict.set_unk_(unk_token)
    slu_dict.set_bos_(SOS_tag)

    if hasattr(args, 'sequence_separator') and args.sequence_separator not in slu_special_symbols:
        slu_special_symbols[slu_special_symbols.index(seq_separator)] = args.sequence_separator

    for s in slu_special_symbols:
        slu_dict.add_symbol(s) 
    slu_dict.set_nspecial_()  # We set the tokens added so far as being special tokens.

    return slu_dict

class SLUEpochBatchIterator(iterators.EpochBatchIterating):
    """Exactly the same as EpochBatchIterator in iterators.py, except it assumes indices returned by the dataset.ordered_indices method are organized dialog level when curriculum is not used.

    Compared to :class:`torch.utils.data.DataLoader`, this iterator:

    - can be reused across multiple epochs with the :func:`next_epoch_itr`
      method (optionally shuffled between epochs)
    - can be serialized/deserialized with the :func:`state_dict` and
      :func:`load_state_dict` methods
    - supports sharding with the *num_shards* and *shard_id* arguments

    Args:
        dataset (~torch.utils.data.Dataset): dataset from which to load the data
        collate_fn (callable): merges a list of samples to form a mini-batch
        batch_sampler (~torch.utils.data.Sampler): an iterator over batches of
            indices
        seed (int, optional): seed for random number generator for
            reproducibility (default: 1).
        num_shards (int, optional): shard the data iterator into N
            shards (default: 1).
        shard_id (int, optional): which shard of the data iterator to
            return (default: 0).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means the data will be loaded in the main process
            (default: 0).
        epoch (int, optional): the epoch to start the iterator from
            (default: 1).
    """

    def __init__(
        self, dataset, dictionary, collate_fn, batch_sampler, seed=1, num_shards=1, shard_id=0,
        num_workers=0, epoch=1, curriculum=0, dialog_level=False
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.dictionary = dictionary
        self.collate_fn = collate_fn
        self.frozen_batches = tuple(batch_sampler)
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers

        self.epoch = max(epoch, 1)  # we use 1-based indexing for epochs
        self.shuffle = True
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, 'supports_prefetch', False)

        self.curriculum = curriculum
        self.dialog_level = dialog_level

    def __len__(self):
        return len(self.frozen_batches)

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""
        if self._next_epoch_itr is not None:
            return self.epoch
        elif self._cur_epoch_itr is not None and self.end_of_epoch():
            return self.epoch + 1
        else:
            return self.epoch

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        """Return a new iterator over the dataset.

        Args:
            shuffle (bool, optional): shuffle batches before returning the
                iterator (default: True).
            fix_batches_to_gpus: ensure that batches are always
                allocated to the same shards across epochs. Requires
                that :attr:`dataset` supports prefetching (default: False).
        """
        self.epoch = self.next_epoch_idx
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self._cur_epoch_itr = self._get_iterator_for_epoch(
                self.epoch, shuffle, fix_batches_to_gpus=fix_batches_to_gpus,
            )
        self.dataset.set_epoch(self.epoch)
        self.shuffle = shuffle
        return self._cur_epoch_itr

    def end_of_epoch(self) -> bool:
        """Returns whether the most recent epoch iterator has been exhausted"""
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        """The number of consumed batches in the current epoch."""
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        """Returns a dictionary containing a whole state of the iterator."""
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
            'shuffle': self.shuffle,
        }

    def load_state_dict(self, state_dict):
        """Copies the state of the iterator from the given *state_dict*."""
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            self._next_epoch_itr = self._get_iterator_for_epoch(
                self.epoch,
                shuffle=state_dict.get('shuffle', True),
                offset=itr_pos,
            )
            if self._next_epoch_itr is None:
                # we finished the epoch, increment epoch counter
                self.epoch += 1

    def _get_iterator_for_epoch(self, epoch, shuffle, fix_batches_to_gpus=False, offset=0):

        # DEBUG: Add a print message here to make sure batches are shuffled
        shuffle_flag = False
        linearized_flag = False

        def shuffle_batches(batches, seed):
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
            return batches

        def linearize_batches(batches):
            linear_batches = []
            for bb in batches:
                linear_batches.extend(bb)
            return linear_batches

        if self._supports_prefetch:
            batches = self.frozen_batches

            #print('[DEBUG] 1) SLUEpochBatchIterator, type of batches: {} (size: {})'.format(type(batches), len(batches)))
            #print('[DEBUG]     * type of batches elements: {} (size: {})'.format(type(batches[0]), len(batches[0])))
            #sys.stdout.flush()

            if not shuffle and (type(batches[0]) == list or type(batches[0]) == tuple) and (type(batches[0][0]) == list or type(batches[0][0]) == tuple):
                #print('[DEBUG] 1) SLUEpochBatchIterator: linearizing batches')
                #sys.stdout.flush()
                batches = linearize_batches(batches)
                linearized_flag = True
            shuffle_flag = False

            #print('[DEBUG] 2) SLUEpochBatchIterator, type of batches: {} (size: {})'.format(type(batches), len(batches)))
            #print('[DEBUG]     * type of batches elements: {} (size: {})'.format(type(batches[0]), len(batches[0])))
            #sys.stdout.flush()

            if shuffle and not fix_batches_to_gpus:
                batches = shuffle_batches(list(batches), self.seed + epoch)
                if (type(batches[0]) == list or type(batches[0]) == tuple) and (type(batches[0][0]) == list or type(batches[0][0]) == tuple):
                    #print('[DEBUG] 2) SLUEpochBatchIterator: linearizing batches')
                    #sys.stdout.flush()
                    batches = linearize_batches(batches)
                    linearized_flag = True
                shuffle_flag = True

            #print('[DEBUG] 3) SLUEpochBatchIterator, type of batches: {}'.format(type(batches)))
            #print('[DEBUG]     * type of batches elements: {}'.format(type(batches[0])))
            #sys.stdout.flush()

            batches = list(iterators.ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))
            self.dataset.prefetch([i for s in batches for i in s])

            if shuffle and fix_batches_to_gpus:
                batches = shuffle_batches(batches, self.seed + epoch + self.shard_id)
                if (type(batches[0]) == list or type(batches[0]) == tuple) and (type(batches[0][0]) == list or type(batches[0][0]) == tuple):
                    #print('[DEBUG] 3) SLUEpochBatchIterator: linearizing batches')
                    #sys.stdout.flush()
                    batches = linearize_batches(batches)
                    linearized_flag = True
                shuffle_flag = True

            #print('[DEBUG] 4) SLUEpochBatchIterator, type of batches: {}'.format(type(batches)))
            #print('[DEBUG]     * type of batches elements: {}'.format(type(batches[0])))
            #sys.stdout.flush()
        else:
            if shuffle:
                batches = shuffle_batches(list(self.frozen_batches), self.seed + epoch)

                '''print('[DEBUG] 5.a) SLUEpochBatchIterator, type of batches: {} (size: {})'.format(type(batches), len(batches)))
                print('[DEBUG]     * type of batches elements: {} (size: {})'.format(type(batches[0]), len(batches[0])))
                print('[DEBUG]     * type of elements of elements: {} (size: {})'.format(type(batches[0][0]), len(batches[0][0])))
                num_elems = 0
                for db in batches:
                    prev = -1
                    for tb in db:
                        num_elems += len(tb) if type(tb) == list or type(tb) == tuple else 1
                        if prev != -1:
                            assert tb == prev+1
                        prev = tb
                print('[DEBUG]     * total number of turns: {}'.format(num_elems))
                sys.stdout.flush()'''

                if (type(batches[0]) == list or type(batches[0]) == tuple) and (type(batches[0][0]) == list or type(batches[0][0]) == tuple):
                    #print('[DEBUG] 5.a) SLUEpochBatchIterator: linearizing batches')
                    #sys.stdout.flush()
                    batches = linearize_batches(batches)
                    linearized_flag = True
                shuffle_flag = True
            else:
                batches = self.frozen_batches

                '''print('[DEBUG] 5.b) SLUEpochBatchIterator, type of batches: {} (size: {})'.format(type(batches), len(batches)))
                print('[DEBUG]     * type of batches elements: {} (size: {})'.format(type(batches[0]), len(batches[0])))
                print('[DEBUG]     * type of elements of elements: {} (size: {})'.format(type(batches[0][0]), len(batches[0][0])))
                num_elems = 0
                for db in batches:
                    for tb in db:
                        num_elems += len(tb) if type(tb) == list or type(tb) == tuple else 1
                print('[DEBUG]     * total number of turns: {}'.format(num_elems))
                print('[DEBUG] Visializing a dialog:')
                mid_idx = int(len(batches)/2)
                dialog_batch = batches[mid_idx]
                for idx, turn_batch in enumerate(dialog_batch):
                    print('* {}) {}'.format(idx, self.dictionary.string(self.dataset.tgt[turn_batch[0]])))
                print('[DEBUG] -----')
                sys.stdout.flush()'''

                if (type(batches[0]) == list or type(batches[0]) == tuple) and (type(batches[0][0]) == list or type(batches[0][0]) == tuple):
                    #print('[DEBUG] 5.b) SLUEpochBatchIterator: linearizing batches')
                    #sys.stdout.flush()
                    batches = linearize_batches(batches)
                    linearized_flag = True
                shuffle_flag = False

            '''print('[DEBUG] 6) SLUEpochBatchIterator, type of batches: {} (size: {})'.format(type(batches), len(batches)))
            print('[DEBUG]     * type of batches elements: {} (size: {})'.format(type(batches[0]), len(batches[0])))
            num_elems = 0
            for tb in batches:
                num_elems += len(tb)
            print('[DEBUG]     * total number of turns: {}'.format(num_elems))
            print('[DEBUG] Visualizing a dialog after linearization:')
            dlen_limit = 50
            mid_idx = int(len(batches)/2)
            for t_idx in range(dlen_limit):
                #assert len(batches[mid_idx+t_idx]) == 5
                print('* {}) {}'.format(mid_idx+t_idx, self.dictionary.string( self.dataset.tgt[batches[mid_idx+t_idx][0]] )))
            print(' -----')
            sys.stdout.flush()'''

            batches = list(iterators.ShardedIterator(
                batches, self.num_shards, self.shard_id, fill_value=[]
            ))

        '''print('[DEBUG] SLUEpochBatchIterator, batch shuffle flag: {}'.format(shuffle_flag))
        print('[DEBUG] SLUEpochBatchIterator, type of batches after ShardedIterator: {} (size: {})'.format(type(batches), len(batches)))
        print('[DEBUG]     * type of batches elements: {} (size: {})'.format(type(batches[0]), len(batches[0])))
        num_elems = 0
        for tb in batches:
            num_elems += len(tb)
        print('[DEBUG]     * total number of turns: {}'.format(num_elems))
        sys.stdout.flush()'''

        if linearized_flag:
            #print('[DEBUG] SLUEpochBatchIterator, setting offset to 0')
            #sys.stdout.flush()
            offset = 0

            '''print('[DEBUG] SLUEpochBatchIterator,')
            print('[DEBUG]    *** Visualizing whole dialog after linearization (linear lenght: {}):'.format(len(batches)))
            mid_idx = int(len(batches)/2)
            dialog_idx = 0
            turn_idx_limit = 50
            for idx in range(turn_idx_limit):
                #assert len(batches[mid_idx+idx]) == 5
                print('[DEBUG] {}) {}'.format(idx, self.dictionary.string(self.dataset.tgt[batches[mid_idx+idx][dialog_idx]])))
                sys.stdout.flush()
            print('[DEBUG] ----------')
            sys.stdout.flush()'''
        '''else:
            print('[DEBUG] *** Visualizing whole dialog before end:')
            mid_idx = int(len(batches)/2)
            dialog_batch = batches[mid_idx]
            for i, idx in enumerate(dialog_batch):
                print('[DEBUG] {} -> {}) {}'.format(i, idx, self.dictionary.string(self.dataset.tgt[dialog_batch[i]])))
            print('[DEBUG] ----------')
            sys.stdout.flush()'''

        if offset > 0 and offset >= len(batches):
            return None

        if self.num_workers > 0:
            os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning' 

        #print('[DEBUG] SO FAR SO GOOD!')
        #sys.exit(0)

        return iterators.CountingIterator(
            torch.utils.data.DataLoader(
                self.dataset,
                collate_fn=self.collate_fn,
                batch_sampler=batches[offset:],
                num_workers=self.num_workers,
            ),
            start=offset,
        )

def create_batch_(data, curr_batch_ids):
    
    dialog_len = len( data[curr_batch_ids[0]] ) # all dialogs in this batch have this same length
    dialog_batch = []
    for i in range( dialog_len ):
        for d_idx in range(len(curr_batch_ids)):

            dialog_batch.append( (curr_batch_ids[d_idx], i) )

    return dialog_batch

def get_bogus_turn(real_turn, dd, roberta, turn_id=None, filler=bogus_token):
    # EOS_tag, User_ID
    T, C = real_turn[1].size()
    sig = torch.zeros(3, C).to(real_turn[1])   # NOTE: 20 as input length, 3 as output length, this is to garantie correct CTC loss output values
    if filler == EOD_tag:
        sig.fill_( EOD_value )
        #print('[DEBUG] get_bogus_turn: filling bogus turn with value {}'.format(EOD_value))
        #sys.stdout.flush()
    else:
        sig.fill_( EOD_value / 2.0)
    if roberta is None:
        tt = torch.LongTensor( [dd.bos(), dd.index(filler), dd.eos()] ).to(real_turn[1].device)
    else:
        tt = roberta.encode( filler ).to(real_turn[1].device)

    assert tt.size(0) == 3 and tt[0].item() == dd.bos() and tt[-1].item() == dd.eos()
    bogus_turn_id = turn_id if turn_id is not None else real_turn[0]
    return (bogus_turn_id, sig, tt, tt, tt, bogus_ID)    # NOTE: bogus_ID is defined in globals.py

def rearrange_for_dialog_level_slu_(data, bs, dlens, dd, roberta=None):

    len_sorted = sorted(dlens.keys())
    batches = []    # Batches of dialog id, not necessarily of the same length
    batch = []
    did2len = {}    # Given a dialog id (did) gets its length (in number of turns)
    batch2len = []  # For a batch index, gets the length dialogues in that batch (must) have
    for l in len_sorted:
        num_batches = int(len(dlens[l])/bs) # dlens[l] is the list of dialogues (ID) having length l (in number of turns)
        remainder = len(dlens[l]) % bs

        for bi in range(num_batches):
            for ti in range(bs):
                batch.append( dlens[l][bi*bs+ti] )
                did2len[dlens[l][bi*bs+ti]] = l
                if len(batch) == bs:    # "finalize" this batch
                    batches.append(batch)
                    batch = []
                    batch2len.append(l) # all dialogues in this batch will be padded to have this length

        for ti in range(remainder):
            batch.append( dlens[l][num_batches*bs+ti] )
            did2len[dlens[l][num_batches*bs+ti]] = l
            if len(batch) == bs:
                batches.append(batch)
                batch = []
                batch2len.append(l) # this should be correct as we are scanning len_sorted with increasing lengths

    if len(batch) > 0:
        batches.append(batch) # TODO: last batch must be treated possibly in a different way...
        max_len = 0
        for did in batch:
            did2len[did] = len(data[did])
            if max_len < len(data[did]):
                max_len = len(data[did])
        batch = []
        batch2len.append( max_len )

    # TMP DEBUG CHECK
    tot_len = 0
    for bi, b in enumerate(batches):
        if bi < len(batches)-1:
            assert len(b) == bs
        tot_len += len(b)
    assert tot_len == len(data.keys())
    for did in did2len.keys():
        assert did2len[did] == len(data[did])
    for bi, bb in enumerate(batches):
        for did in bb:
            assert len(data[did]) <= batch2len[bi]
    # END TMP DEBUG CHECK

    # Now add bogus turns to dialogues so that to have:
    #     1. all dialogues in the same batch have the same length;
    #     2. num of dialogues multiple of batch-size (this is accomplished by creating whole bogus dialogues! Hope this work...) 
    for bi, bb in enumerate(batches):
        max_len = batch2len[bi]

        tmp = max([did2len[did] for did in bb])
        assert tmp == max_len

        for did in bb:
            if did2len[did] < max_len:
                dialog = data[did]
                sample_turn = dialog[0]
                if sample_turn[1] is None:
                    found = False
                    for ii in range(len(dialog)):
                        if dialog[ii][1] is not None:
                            sample_turn = dialog[ii]
                            found = True
                            break
                    if not found:
                        raise ValueError('The input signals for current dialog are all None')
                for ii in range(did2len[did]+1, max_len+1):
                    dialog.append( get_bogus_turn(sample_turn, dd, roberta, ii) )
                    did2len[did] += 1
                #data[did] = dialog # needed ???

    # NOTE: last batch may contain less dialogues than bs, so possibly whole bogus dialogues must be added to have a batch of the same size as the others.
    if len(batches[-1]) != bs:
        bogus_did_base = 9900
        max_len = batch2len[-1]

        tmp = max( [did2len[did] for did in batches[-1]] )
        assert tmp == max_len
        tmp = max( [len(data[did]) for did in batches[-1]] )
        assert tmp == max_len

        for ii in range(bs-len(batches[-1])):
            bogus_did = str(bogus_did_base+ii+1)
            batches[-1].append(bogus_did)
            data[bogus_did] = []
            did2len[bogus_did] = 0

        assert len(batches[-1]) == bs

        for did in batches[-1]:
            if did2len[did] < max_len: # actually only the bogus dialogues should satisfy this condition !
                if str(bogus_did_base) in did:
                    assert did2len[did] == 0

                dialog = data[did]
                sample_turn = data[batches[-1][0]][0]
                if sample_turn[1] is None:
                    found = False
                    for ii in range(len(data[batches[-1][0]])):
                        if data[batches[-1][0]][ii][1] is not None:
                            sample_turn = data[batches[-1][0]][ii]
                            found = True
                            break
                    if not found:
                        raise ValueError('All input signals of the current dialog at last batch are None')
                for ii in range(did2len[did]+1, max_len+1):
                    #print('[DEBUG] rearrange_dialogs, creating bogus turn from dialog id {}: {}'.format(batches[-1][0], data[batches[-1][0]]))
                    #sys.stdout.flush()
                    dialog.append( get_bogus_turn(sample_turn, dd, roberta, ii) )
                    did2len[did] += 1
                #data[did] = dialog # needed ???

    # TMP DEBUG CHECK
    assert len(data) % bs == 0

    for ii, bb in enumerate(batches):
        dlen = batch2len[ii]
        plen = len(data[bb[0]])
        found = False
        for did in bb:
            assert len(data[did]) == dlen, 'Assertion 1: expected dialog length {}, got {}'.format(dlen, len(data[did]))
            assert len(data[did]) == plen, 'Assertion 2: expected dialog length {}, got {}'.format(plen, len(data[did]))

            if torch.sum(data[did][-1][1]).item() != 0.0:
                found = True
        if not found:
            raise ValueError('Found a batch with all zero-turns as last')
    # END TMP DEBUG CHECK

    # TODO: add here an end of dialog marker in each and every dialog so that the model can detect when to re-initialize the dialog history cache
    for did in data:
        sample_turn = data[did][0]
        if sample_turn[1] is None:
            found = False
            for ii in range(len(data[did])):
                if data[did][ii][1] is not None:
                    sample_turn = data[did][ii]
                    found = True
                    break
            if not found:
                raise ValueError('All input signals of current dialog are None')
        data[did].append( get_bogus_turn(sample_turn, dd, roberta, len(data[did])+1, filler=EOD_tag) )

    dialog_lengths = {}
    for dialog_id in data.keys():
        curr_len = len(data[dialog_id])
        if not curr_len in dialog_lengths:
            dialog_lengths[curr_len] = []
        dialog_lengths[curr_len].append( dialog_id )

    # TMP DEBUG CHECK
    for l in dialog_lengths.keys():
        assert len(dialog_lengths[l]) % bs == 0
    for bb in batches:
        found = False
        for did in bb:
            T, C = data[did][-1][1].size()
            if torch.sum(data[did][-1][1] == EOD_value).item() != T*C:
                raise ValueError('Found a batch without End-of-dialogue marker. Expected value {}, got value {}. Expected num. of EOD values {} ({} x {}), got num. of EOD values {}'.format(T*C*EOD_value, torch.sum(data[did][-1][1]), T*C, T, C, torch.sum(data[did][-1][1] == EOD_value)))
    # END TMP DEBUG CHECK

    return dialog_lengths

def create_dialog_batches_(data, batch_size, infos):

    dd = infos['dict']
    roberta=infos['roberta']
    dlevel_slu = infos['dialog_level_slu']
    normalize_dialogs = infos['normalize_dialogs']

    dialog_lengths = {}
    for dialog_id in data.keys():
        curr_len = len(data[dialog_id])
        if not curr_len in dialog_lengths:
            dialog_lengths[curr_len] = []
        dialog_lengths[curr_len].append( dialog_id )

    if normalize_dialogs:
        dialog_lengths = rearrange_for_dialog_level_slu_(data, batch_size, dialog_lengths, dd, roberta=roberta) 

    dialog_batches = []
    batch_sizes = []
    for dlen in dialog_lengths.keys():
        
        if len(dialog_lengths[dlen]) > 1:
            
            num_batches = int(len(dialog_lengths[dlen]) / batch_size)
            remainder = len(dialog_lengths[dlen]) % batch_size

            if normalize_dialogs:
                assert remainder == 0
            
            b_idx = 0
            while b_idx < num_batches:
                curr_batch_ids = dialog_lengths[dlen][b_idx*batch_size:(b_idx+1)*batch_size]
                dialog_batches.append( create_batch_(data, curr_batch_ids) )
                batch_sizes.append(batch_size)
                b_idx += 1
            
            if remainder > 0:
                curr_batch_ids = dialog_lengths[dlen][num_batches*batch_size:]
                dialog_batches.append( create_batch_(data, curr_batch_ids) )
                batch_sizes.append( remainder )
        else:
            if normalize_dialogs and batch_size > 1:
                raise ValueError()

            curr_batch_ids = dialog_lengths[dlen]
            dialog_batches.append( create_batch_(data, curr_batch_ids) )
            batch_sizes.append(1)

    return dialog_batches, batch_sizes

def read_txt(txtpath):

    f = open(txtpath, 'rb')
    lines = [l.decode('utf-8','ignore') for l in f.readlines()]
    #for line in lines:
    #    dialog_transcript = line
    if len(lines) > 0:
        dialog_transcript = lines[-1]
    else:
        print(' - End2EndSLU WARNING: found empty file {}, returning "none"'.format(txtpath))
        sys.stdout.flush()
        dialog_transcript = 'none'
    f.close()
    return dialog_transcript.strip()

def build_slu_turn(args, dialog_struct):

    slu_turn = ''
    for vals in dialog_struct.values():
        for turn_struct in vals:
            for c in turn_struct['Concepts']:
                if len(slu_turn) > 0:
                    slu_turn = slu_turn + ' '
                if len(c) == 3:
                    slu_turn = slu_turn + slu_start_concept_mark + ' ' + c[2] + ' ' + '@' + c[1] + ' ' + slu_end_concept_mark
                else:
                    slu_turn = slu_turn + slu_start_concept_mark + ' ' + c[3] + ' ' + '@' + c[1] + ' ' + slu_end_concept_mark

    return slu_turn

def parse_media_semantic_annotation(filename, attval_format=False):
    
    # Example:
    # 364_16 @null{euh} @+rang-temps{la troisième} @+temps-unite{semaine} @+temps-mois{d' août}
    
    clean_re = re.compile('^\s+|\s+$')
    p = re.compile('^(\+|\-)?(\S+)\{([^\}]+)\}')
    attval_re = re.compile('^(\+|\-)?(\S+)\[([^\]]*)\]\{([^\}]+)\}')
    
    Dialogs = {}
    f = open(filename, encoding='utf-8')
    for line in f:
        line.rstrip("\r\n")
        tokens = line.split()
        ID = tokens[0]
        #(dialog_ID,turn_ID) = ID.split("_")
        # TODO: check that the following lines of code, up to concepts_str..., are generic with any corpus ID!
        ID_tokens = ID.split("_")
        dialog_ID = "_".join(ID_tokens[:-1])
        turn_ID = ID_tokens[-1]
        concepts_str = clean_re.sub('', ' '.join(tokens[1:]))

        #print(' - parse_media_semantic_annotation, dialog_ID and turn_ID: {}, {}'.format(dialog_ID, turn_ID))
        #sys.stdout.flush()

        TurnStruct = {}
        TurnStruct['ID'] = turn_ID
        if not dialog_ID in Dialogs:
            Dialogs[dialog_ID] = []

        concept_list = []
        concepts = concepts_str.split('@')
        for c in concepts:
            if len(c) > 0 and c != 'null{} ' and c != 'null{}' and c != 'null[void]{} ' and c != 'null[void]{}':
                m = attval_re.match(c)
                if m == None:
                    if attval_format:
                        raise ValueError('wrong attribute-value format parsing {}'.format(c))
                    m = p.match(c)
                    if m == None:
                        sys.stderr.write(' - ERROR: parse_media_semantic_annotation parsing error at {} while parsing file {}\n'.format(c, filename))
                        sys.exit(1)
                    else:
                        (mode,concept,surface) = (m.group(1),m.group(2),clean_re.sub('',m.group(3)))
                        concept_list.append( (mode,concept,surface) )
                else:
                    (mode, concept, value, surface) = (m.group(1), m.group(2), m.group(3), clean_re.sub('', m.group(4)))
                    #if attval_format:
                    #    concept_list.append( (mode, concept, value, surface) )
                    #else:
                    concept_list.append( (mode, concept, value, surface) )
        TurnStruct['Concepts'] = concept_list
        Dialogs[dialog_ID].append( TurnStruct )

    f.close() 
    return Dialogs

def parse_media_filename( turn ):

    file_prefix = turn.split('/')[-1]   # e.g. dialog-611.turn-26-Machine-overlap_none.channel-1
    dialog_id = file_prefix.split('.')[0]
    turn_id = file_prefix.split('.')[1]
    turn_id_comps = turn_id.split('-')[0:3]
    turn_id = '-'.join(turn_id_comps)

    return dialog_id, turn_id

def parse_fsc_filename( turn ):

    '''components = turn.split('/')    # e.g. .../2BqVo8kVB2Skwgyb/029f6450-447a-11e9-a9a5-5dbec3b8816a
    dialog_id = components[-2]
    turn_id = components[-1]'''

    # NOTE: this is actually the parsing of DSTC format filenames, I use this function for lazyness ... :-)
    #print('[DEBUG] reading {}'.format(turn.split('/')[-1]))
    filename = turn.split('/')[-1]
    tokens = filename.split('.')
    dialog_id = tokens[0]
    turn_id = '.'.join(tokens[1:])

    return dialog_id, turn_id

def parse_filename( args, turn ):

    if args.corpus_name in ['media', 'etape']:
        return parse_media_filename(turn)
    elif args.corpus_name == 'fsc':
        return parse_fsc_filename(turn)
    else:
        raise NotImplementedError

def read_dialog_data(TurnList, args):
    
    windowtime = args.window_time 
    reduce_flag = args.reduce_signal
    lan_flag = args.w2v_language
    feat_ext = args.feature_extension

    if args.online_feature_extraction:
        print(' * read_dialog_data: extracting features online, reading .wav inputs')
        sys.stdout.flush()
    else:
        print(' * read_dialog_data: reading {} feature files'.format(feat_ext))
        sys.stdout.flush()

    dialog_data = {}
    with open(TurnList) as f:
        turns = f.readlines()
        f.close()

    n_input_features = -1
    none_tsr_idx = []
    #check_turn_idx = 0
    for turn in turns:
        turn = turn.strip()
        dialog_id, turn_id = parse_filename( args, turn )
        basename = turn.split('/')[-1]
        ttd = basename.split('.')
        ttt = turn_id.split('-')
        if args.corpus_name == 'media':
            assert len(ttd) == 3
            assert len(ttt) == 3
        elif args.corpus_name == 'fsc':
            assert len(ttd) == 2
            assert len(ttt) == 3

        if dialog_id not in dialog_data:
            dialog_data[dialog_id] = []
            check_turn_idx = 0

        #if dialog_id != 'mul0645' and dialog_id != 'pmul3937' and dialog_id != 'pmul2638':
        #    assert check_turn_idx+1 == int(turn_id.split('-')[1]), 'Turn ID clash @turn {}: expected {} vs. parsed {}'.format(turn, check_turn_idx+1, turn_id.split('-')[1])
        #check_turn_idx += 1

        # 1. Turn ID for sorting turns of the same dialog
        turn_data = []
        turn_data.append( dialog_id.strip() + '@' + turn_id.strip() )

        sig_flag = False
        trs_flag = False 

        # 2. Spectrogram of the turn audio signal (the input to the network)
        if args.online_feature_extraction or args.speech_encoder == 'sb-wav2vec2':
            sample_rate, samples = wavfile.read(turn.strip() + '.wav')
            if args.upsample:
                samples = signal.resample(samples, samples.shape[0]*2)
            spg_tsr = torch.from_numpy(samples).float() 
            sig_flag = True
        else:
            if windowtime == 20: 
                if args.use_transcription_as_input:
                    if os.path.exists(turn.strip() + feat_ext):
                        spg_tsr = torch.load(turn.strip() + feat_ext)
                        sig_flag = True
                    else:
                        spg_tsr = None
                else:
                    spg_tsr = torch.load(turn.strip() + feat_ext)
                    sig_flag = True 
            elif windowtime == 10:
                raise NotImplementedError 
            elif windowtime < 0:
                srate_str = '16kHz'
                if windowtime == -2:
                    srate_str = '8kHz'
                lan=''
                if lan_flag == 'Fr':
                    lan='-Fr'
                if windowtime > -3:
                    filename = turn.strip() + '.' + srate_str + lan + '.w2v'
                    if args.use_transcription_as_input:
                        if os.path.exists(filename):
                            spg_tsr = torch.load(filename)
                            sig_flag = True
                        else:
                            spg_tsr = None
                    else:
                        spg_tsr = torch.load(filename)
                        sig_flag = True
                else:
                    filename = turn.strip() + feat_ext
                    if args.use_transcription_as_input:
                        if os.path.exists(filename):
                            spg_tsr = torch.load(filename)
                            sig_flag = True
                        else:
                            spg_tsr = None
                    else:
                        spg_tsr = torch.load(filename)
                        sig_flag = True 
                if reduce_flag > 1:
                    raise NotImplementedError 

            if spg_tsr is not None and windowtime > -3:
                spg_tsr = spg_tsr.squeeze().permute(1,0)
            elif spg_tsr is not None:
                spg_tsr = spg_tsr.squeeze() 
            if spg_tsr is not None and len(spg_tsr.size()) != 2:
                spg_tsr = spg_tsr.unsqueeze(0) 

        if spg_tsr is not None and spg_tsr.size(0) < 3:
            print(' *** read_dialog_data: got strangely short signal {}...'.format(spg_tsr.size()))
            sys.stdout.flush()

        if spg_tsr is None or args.online_feature_extraction or (spg_tsr.size(0) > 3 and spg_tsr.size(0) < args.max_source_positions):
            if spg_tsr is not None:
                if args.speech_encoder != 'sb-wav2vec2' and n_input_features != -1 and n_input_features != spg_tsr.size(-1):
                    raise ValueError('Found input tensors with different number of features ({} != {})'.format(n_input_features, spg_tsr.size(-1)))
                elif n_input_features == -1:
                    n_input_features = spg_tsr.size(-1) 
            turn_data.append( spg_tsr )  # Spectrogram's shape is (1 x num_features x sequence_length), we make it (sequence_length x num_features) 

            # 3. The reference transcription
            turn_txt = read_txt(turn.strip() + '.txt') 
            trs_flag = True
            turn_data.append( turn_txt.strip() ) 

            # 4. The transcription "enriched" with semantic annotation 
            if args.corpus_name in ['media', 'etape']:
                if True: #user_ID in basename:
                    if args.slu_turn_format == 'media':
                        dialog_struct = parse_media_semantic_annotation( turn.strip() + '.sem' )
                        '''slu_turn = ''
                        for vals in dialog_struct.values():
                            for turn_struct in vals: 
                                for c in turn_struct['Concepts']: 
                                    if len(slu_turn) > 0:
                                        slu_turn = slu_turn + ' '
                                    slu_turn = slu_turn + slu_start_concept_mark + ' ' + c[2] + ' ' + c[1] + ' ' + slu_end_concept_mark'''
                        slu_turn = build_slu_turn(args, dialog_struct)
                    else:
                        f = open(turn.strip() + '.sem', encoding='utf-8')
                        ll = f.readlines()
                        f.close()
                        assert len(ll) == 1
                        slu_turn = ll[0].strip()

                    turn_data.append( slu_turn.strip() )
                    if user_ID in basename:
                        turn_data.append( user_ID )
                    else:
                        turn_data.append( machine_ID )
                else:
                    slu_turn = slu_start_concept_mark + ' ' + turn_txt + ' ' + machine_semantic + ' ' + slu_end_concept_mark
                    turn_data.append( slu_turn.strip() )
                    turn_data.append( machine_ID )

                if turn_id in dialog_data[dialog_id]:
                    sys.stderr.write(' *** read_dialog_data WARNING: turn_id {} is not unic\n'.format(turn_id))
            elif args.corpus_name == 'fsc':
                turn_sem = read_txt(turn.strip() + '.sem')
                #tt = turn_sem.split()
                #c1, c2, c3 = '-'.join(tt[:-2]), tt[-2], tt[-1]
                #c1 = slu_start_concept_mark + ' ' + ' '.join(tt[:-2]) + ' ' + slu_end_concept_mark
                #c2 = tt[-2]
                #c3 = tt[-1]
                turn_data.append( turn_sem ) #c1 + ' ' + c2 + ' ' + c3 )
                if user_ID in basename:
                    turn_data.append( user_ID )
                elif machine_ID in basename:
                    turn_data.append( machine_ID )
                else:
                    turn_data.append( user_ID ) # This will still work with the FSC corpus
            else:
                raise NotImplementedError 

            if hasattr(args, 'use_transcription_as_input') and args.use_transcription_as_input:
                assert sig_flag or trs_flag, 'When --use-transcription-as-input is specified at least one of input signal and transcription must be provided'
            else:
                assert sig_flag and trs_flag, 'Both input signal and corresponding transcription must be provided'
            dialog_data[dialog_id].append( turn_data )
        else:
            sys.stderr.write(' *** read_dialog_data WARNING: discarding offending length turn (size: {})'.format(spg_tsr.size(0)))

    # NOTE: I don't know why, but order of turns in dialog_data is not always coherent, so here I sort turns of each dialogue so that to respect the natural order
    print(' *** Sorting dialogue turns...')
    sys.stdout.flush()
    n_sorted = 0
    for did in dialog_data.keys():
        turn_list = sorted(dialog_data[did], key=lambda list: int(list[0].split('@')[-1].split('-')[1]))

        d_sorted = False
        for idx, td in enumerate(dialog_data[did]):
            assert idx+1 == int(turn_list[idx][0].split('@')[-1].split('-')[1]), 'Turn ID clash @dialogue {}: {} expected vs. {} found'.format(did, idx+1, int(turn_list[idx][0].split('@')[-1].split('-')[1]))
            if int(td[0].split('@')[-1].split('-')[1]) != idx+1:
                d_sorted = True
            if turn_list[idx][1] is None:
                none_tsr_idx.append( (did, idx) )
        if d_sorted:
            n_sorted += 1
        dialog_data[did] = turn_list
    print('     * {} dialogues needed sorting.'.format(n_sorted))
    sys.stdout.flush()

    if len(none_tsr_idx) > 0:
        print(' *** Adding {} bogus turns'.format(len(none_tsr_idx)))
        sys.stdout.flush()

        assert n_input_features != -1
        for did, idx in none_tsr_idx:
            #print('[DEBUG] getting tsr length from transcript {}'.format(dialog_data[did][idx][2]))
            #sys.stdout.flush()
            #trs_len = max( len(dialog_data[did][idx][2].split()), len(dialog_data[did][idx][3].split()) )
            #bogus_tsr = torch.zeros(trs_len, n_input_features)

            trs_len = len(dialog_data[did][idx][2].split()) # NOTE: * 4 assuming a Pyramidal encoder with 3 LSTM blocks (otherwise it doesn't matter). This should garantie correct CTC loss output values.
            assert trs_len > 0
            assert dialog_data[did][idx][1] is None
            assert dialog_data[did][idx][0].split('-')[-1] == 'Machine'
            dialog_data[did][idx][1] = torch.zeros(trs_len, n_input_features) 

    # TMP DEBUG CHECK
    n_none = 0
    for did in dialog_data.keys():
        for td in dialog_data[did]:
            if td[1] is None:
                n_none += 1
    if n_none > 0:
        print(' *** WARNING: found {} None input signals in read data'.format(n_none))
        sys.stdout.flush()
    return dialog_data

def get_character_level_slu_turn(args, slu_turn_tsr, vocab):

    pad_flag = args.padded_reference
    slu_turn_str = vocab.string(slu_turn_tsr)

    #print('Original turn: {}'.format(slu_turn_str))
    #print(' -----')

    tokens = slu_turn_str.split()
    char_slu_turn_str = ''
    if pad_flag:
        char_slu_turn_str = SOS_tag

    if args.corpus_name == 'fsc':
        for c in '|'.joint(tokens[:-2]):
            char_slu_turn_str = char_slu_turn_str + c + ' '
    else:
        chunk = {}
        chunk['surface'] = []
        for t in tokens:
            if t == slu_start_concept_mark:
                chunk['start'] = t
            elif t == slu_end_concept_mark:
                chunk['end'] = t
                chunk['concept'] = chunk['surface'][-1]
                chunk['surface'] = chunk['surface'][:-1]

                if chunk['concept'] in ['', ' ', '|']:
                    raise ValueError(' Found invalid concept {}'.format(chunk['concept']))

                if char_slu_turn_str != '':
                    char_slu_turn_str = char_slu_turn_str + ' '
                char_slu_turn_str = char_slu_turn_str + chunk['start'] + ' | ' # TODO: with new format replace ' ' with ' | '
                for c in '|'.join(chunk['surface']):
                    char_slu_turn_str = char_slu_turn_str + c + ' '
                char_slu_turn_str = char_slu_turn_str + ' | ' + chunk['concept'] + ' | ' + chunk['end'] + ' | '
                #char_slu_turn_str = char_slu_turn_str + ' ' + chunk['concept'] + ' ' + chunk['end'] + ' ' # TODO: with new format comment this line and uncomment the previous
                chunk = {}
                chunk['surface'] = []
            else:
                chunk['surface'].append(t)

    if pad_flag:
        char_slu_turn_str = char_slu_turn_str + EOS_tag
    else:
        char_slu_turn_str = char_slu_turn_str[:-1] # Remove trailing space

    #print(' **** Original SLU turn: {}'.format(slu_turn_str))
    #print(' * Character-level turn: {}'.format(char_slu_turn_str))
    #print(' -----')
    #sys.stdout.flush()

    return char_slu_turn_str

def references2indexes(args, data, vocab, split, roberta=None):

    for did in data.keys():
        for idx in range(len(data[did])):
            (turn_id, spg, txt_turn, slu_turn, spk_ID) = data[did][idx]
            if roberta is not None:
                char_tsr = roberta.encode( ' '.join(list(txt_turn)) )
                token_tsr = roberta.encode( txt_turn )
                slu_tsr = roberta.encode( slu_turn )
            else:
                if (hasattr(args, 'freeze_dictionary') and args.freeze_dictionary) or (hasattr(args, 'freeze_train_dictionary') and args.freeze_train_dictionary and split != 'train'):
                    char_list = [vocab.index(c) for c in txt_turn] if txt_turn is not None else []
                    token_list = [vocab.index(t) for t in txt_turn.split()] if txt_turn is not None else []
                    slu_list = [vocab.index(s) for s in slu_turn.split()]
                else: 
                    char_list = [vocab.add_symbol(c) for c in txt_turn] if txt_turn is not None else [] 
                    token_list = [vocab.add_symbol(t) for t in txt_turn.split()] if txt_turn is not None else [] 
                    slu_list = [vocab.add_symbol(s) for s in slu_turn.split()]

                if args.padded_reference:
                    char_list = [vocab.bos()] + char_list + [vocab.eos()]
                    token_list = [vocab.bos()] + token_list + [vocab.eos()]
                    slu_list = [vocab.bos()] + slu_list + [vocab.eos()]

                char_tsr = torch.LongTensor( char_list )
                token_tsr = torch.LongTensor( token_list )
                slu_tsr = torch.LongTensor( slu_list )

            data[did][idx] = (turn_id, spg, char_tsr, token_tsr, slu_tsr, spk_ID)


def feature_wise_mean_std(data, skip_all_zeros=False, sample_threshold=0.0):
    (sequence_length, num_features) = list(data.values())[0][0][1].size()
    data_mu = torch.zeros( num_features )
    data_sigma = torch.zeros( num_features )

    with torch.no_grad():
        total_features = 0
        for dialog_id in data.keys():
            randn = torch.rand(1).item()

            if randn > sample_threshold:
                for t in data[dialog_id]:
                    if t[1] is not None:
                        if not skip_all_zeros or torch.sum( t[1] ) != 0.0:
                            data_mu = data_mu + torch.sum( t[1], 0 )
                            total_features += t[1].size(0)
        data_mu = data_mu / float( total_features ) if total_features != 0 else 0.0

        for dialog_id in data.keys():
            randn = torch.rand(1).item()

            if randn > sample_threshold:
                for t in data[dialog_id]:
                    if t[1] is not None:
                        if not skip_all_zeros or torch.sum( t[1] ) != 0.0:
                            data_sigma = data_sigma + torch.sum( (t[1] - data_mu)**2, 0 )
        data_sigma = torch.sqrt( data_sigma / float(total_features-1) ) if total_features != 0 else 1.0

    return (data_mu, data_sigma)

def normalize_data_only(data, mu, sigma):
    with torch.no_grad():
        for dialog_id in data.keys():
            for t in data[dialog_id]:
                #if t[1] is not None:
                if torch.sum(t[1]) != 0.0:
                    t[1] = (t[1] - mu) / sigma


@register_task('end2end_slu')
class End2EndSLU(FairseqTask):
    
    @staticmethod
    def add_args(parser):
        
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--serialized-data', type=str,
                            help='file containing the serialized corpus (with a previous torch.save)')
        parser.add_argument('--load-dictionary', type=str,
                            help='Load the dictionary for the task symbols from the specified file (e.g. to perform transfer learning)')
        parser.add_argument('--load-embeddings', type=str, help='Load decoder token embeddings from the specified file')
        parser.add_argument('--freeze-dictionary', action='store_true', default=False, help='Freeze the loaded dictionary, that is it does not add new found symbols')
        parser.add_argument('--freeze-train-dictionary', action='store_true', default=False, help='Create the dictionary only from training data tokens')
        parser.add_argument('--freeze-embeddings', action='store_true', default=False, help='Freeze loaded pre-trained embeddings')
        parser.add_argument('--source-dictionary', type=str,
                            help='When performing transfer learning, uses this dictionary as source domain dictionary')
        parser.add_argument('--target-dictionary', type=str,
                            help='When performing transfer learning, uses this dictionary as target domain dictionary')
        parser.add_argument('--max-source-positions', default=15000000, type=int,
                            help='max input length')
        parser.add_argument('--max-target-positions', default=1600, type=int,
                            help='max input length')
        parser.add_argument('--slu-subtask', default='char', type=str,
                            help='which subtask has to be modeled (e.g. char, token, concept)')
        parser.add_argument('--user-only', action='store_true', default=False,
                            help='use only user turns (specific to SLU for dialog systems, and to the corpus MEDIA in particular)')
        parser.add_argument("--padded-reference", action='store_true', default=False,
                            help="Specify if the gold reference is padded")
        parser.add_argument('--test-mode', action='store_true', default=False,
                            help='Specify that model must be run in test mode')
        parser.add_argument('--reduce-signal', type=int, default=0,
                            help='Reduce the input signal by the given factor (only for wav2vec features)')
        parser.add_argument('--window-time', type=int, default=20,
                            help='Specify the window time, in milliseconds, used for extracting spectrograms. If negative, wav2vec features are used.')
        parser.add_argument('--w2v-language', type=str, default='En',
                            help='Specify which languege wav2vec features are from')
        parser.add_argument('--mode', type=str, default='user+machine',
                            help='Leave this unchanged!')
        parser.add_argument('--io-size-ratio', type=int, default=5,
                            help='The estimated ratio between input and output sequence lenghts')
        parser.add_argument('--bidirectional-encoder', action='store_true', default=True,
                            help='Use a bidirectional LSTM in the encoder')
        parser.add_argument('--encoder-transformer-layers', action='store_true', default=False,
                            help='Use transformer encoder layers on top of the convolution+recurrent encoder')
        parser.add_argument('--corpus-name', type=str, default='media',
                            help='Specify the corpus name to customize the data reading. 1) media (default), 2) fsc (fluent speech commands)')
        parser.add_argument('--slu-turn-format', type=str, default='media', help='Format of the slu turn read from .sem files')
        parser.add_argument( '--load-encoder', type=str, help='Load a pre-trained (basic) encoder' )
        parser.add_argument( '--load-fairseq-encoder', type=str, help='Load the encoder from a fairseq checkpoint' )
        parser.add_argument( '--load-fairseq-decoder', type=str, help='Load the decoder from a fairseq checkpoint' )
        parser.add_argument( '--slu-end2end', action='store_true', default=False, help='Add an auxiliary loss for an additional output expected in the model output structure' )
        parser.add_argument('--scheduled-sampling', action='store_true', default=False, help='Use scheduled sampling during training')
        parser.add_argument('--start-scheduled-sampling', type=int, default=2, help='Epoch starting from which scheduled sampling will be performed')
        parser.add_argument('--encoder-state-window', type=int, default=2, help='Size w of the window of encoder states attended by the cross attention')
        parser.add_argument('--pyramid-hidden-layers', type=int, default=2, help='Number of layers in the hidden LSTM stages of the pyramidal encoder')
        parser.add_argument('--prev-prediction-query', type=int, default=-1, help='Index of the decoder hidden state used as query on the previous predictions')
        parser.add_argument('--feature-extension', type=str, default='.20.0ms-spg', help='Extenion of the feature file name')
        parser.add_argument('--character-level-slu', action='store_true', default=False, help='Perform SLU from character-level transcription')
        parser.add_argument('--constrained-output-length', type=int, default=-1, help='Constrain the decoded output of the LSTM decoder to the specified length')
        parser.add_argument('--max-padding-len', type=int, default=60, help='Max length of input sequences that are padded in order to avoid input/output length missmatch')
        parser.add_argument('--max-padding-ratio', type=float, default=7.0, help='Factor by which input sequences shorter than max-padding-len are stretched to avoid input/output length missmatch')
        parser.add_argument('--padding-active', action='store_true', default=False, help='Determine if padding should be done or not')
        parser.add_argument('--unfreeze-encoder-at-cer', type=float, default=110.00, help='Unfreeze encoder parameters for training when the dev CER goes below the specified value')
        parser.add_argument('--attention-type', type=str, default='std', help='Attention type to use: standard (std, default), mha2 (multi-head attention with 2 heads), mha4 (multi-head attention with 4 heads)')
        parser.add_argument('--dialog-level-slu', action='store_true', default=False, help='Re-arrange turns and dialogues to perform dialog-level SLU')
        parser.add_argument('--normalize-dialog-batches', action='store_true', default=False, help='Rearrange dialogues so that to have all batches of the same size')
        parser.add_argument('--parse-rich-semantic', action='store_true', default=False, help='Use enriched/annotated semantic sequences instead of the standard .sem annotation files')
        parser.add_argument('--dialog-batches', action='store_true', default=False, help='Use a single dialog as a batch')
        parser.add_argument('--use-dialog-history', action='store_true', default=False, help='If set to True, the model will exploit dialog history')
        parser.add_argument('--context-discount', type=float, default=1.0, help='Context discount when using "sum" as context fusion method')
        parser.add_argument('--context-fusion', type=str, default='gating', help='Method for integrating the context information: sum (default), gating')
        parser.add_argument('--trs-fusion', action='store_true', default=False, help='If True, use a gate for speech and transcription fusion, otherwise it uses a simple sum')
        parser.add_argument('--context-size', type=int, default=8, help='Number of turns used as context')
        parser.add_argument('--context-first-turns', type=int, default=6, help='Number of turns used in the context and taken at the beginning of the dialog, the remainder are the last previous turns')
        parser.add_argument('--speech-encoder', type=str, default='ziggurat', help='Structure of speech encoder: ziggurat (NeurIPS 2021 paper), deep-speech (ICASSP 2020 paper)')
        parser.add_argument('--se-size', type=int, default=1024, help='Specifies the size of features extracted with the speech-encoder when using a speechbrain wav2vec2 model')
        parser.add_argument('--model-beam', type=int, default=1, help='At generation phase and when using dialog context, tells to the model the beam size so that the model can manage correctly the dialog history (generate.py does not seem to add the beam argument to the namespace)')
        parser.add_argument('--print-history-attn-weights', action='store_true', default=False, help='Print dialog history attention weights of the current decoded sequence, along with current target and history turn tokens')
        parser.add_argument('--label-dictionary', type=str, help='Load a task specific label dictionary, e.g. to perform discounting on non-label tokens in the decoder')
        parser.add_argument('--non-label-discounting', type=float, default=1.0, help='Discounting factor for non-label tokens in the decoder')
        parser.add_argument('--use-transcription-as-input', action='store_true', default=False, help='Use transcription of input signal as input if it is available, in addition to audio signal if it is available')
        parser.add_argument('--use-source-context', action='store_true', default=False, help='Use the source-side context. This option must be used together with options for dialog-level SLU.')
        parser.add_argument('--roberta-model', type=str, help='Specify an absolute path to a Roberta model used to encode transcriptions')
        parser.add_argument('--asr-slu-loss', action='store_true', default=False, help='Optimize the model with respect to both ASR (encoder) and SLU (decoder). Transcription must be provided!')
        parser.add_argument('--dec-loss', type=str, default='ctc', help='Specifies the loss for the decoder: ctc (default), nllloss (cross entropy)')
        parser.add_argument('--online-feature-extraction', action='store_true', default=False, help='Perform feature extraction online for each wav')
        parser.add_argument('--feature-extractor', type=str, help='Feature extractor (e.g. Wav2Vec model) for online feature extraction')
        parser.add_argument('--fe-size', type=str, default='small', help='Argument for feature extractor (e.g. for whisper it can be "small", "medium", or "large")')
        parser.add_argument('--upsample', action='store_true', default=False, help='Upsample signals when performing online feature extraction')
        parser.add_argument('--freeze-fe', action='store_true', default=False, help='If set to True freeze the speech-encoder when using a speechbrain wav2vec2 model')
        parser.add_argument('--sb-model-path', type=str, help='Path to the directory (hub) containing the speechbrain wav2vec2 model')
        parser.add_argument('--softmax-temperature', type=float, default=1.0, help="Use the specified temperature when computing the output (log-)probabilities with the SLU CTC loss")
        parser.add_argument('--softmax-start-temperature', type=float, default=1.0, help='Starting softmax temperature when rising temperature with linear strategy')
        parser.add_argument('--rise-temperature-at-epoch', type=int, default=1, help='Rise softmax temperature at the specified epoch')
        parser.add_argument('--rise-temperature-strategy', type=str, default='fix', help='Choose the strategy used to rise softmax temperature: fix (default), linear (increase the temperature linearly starting from the specified epoch up to the specified temperature)')
        parser.add_argument('--model-start-point', type=str, help='Load a pretrained model from the specified file')
        parser.add_argument('--decoder-norm-layers', action='store_true', default=False, help='If set, apply normalisation layers in the decoder')

        parser.add_argument('--load-dstc-ex-data', type=str, help='Ad-hoc argument to load the extra 100xtraining data for the DSTC challenge')

    @classmethod
    def setup_task(cls, args, **kwargs):
        
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just initialize the label dictionary

        label_vocab = init_slu_dictionary(args)
        if args.load_dictionary:
            print(' - Loading serialized dictionary...')
            sys.stdout.flush()
            
            label_vocab = torch.load(args.load_dictionary)

        print('| [label] dictionary: {} types'.format(len(label_vocab)))
        
        return End2EndSLU(args, label_vocab)

    def __init__(self, args, label_vocab):
        
        super().__init__(args)

        self.tmp_corpus = {}
        self.spk_abs_val = 1.0
        self.num_speakers = 2
        if args.dialog_level_slu:
            self.num_speakers = 3
        self.tracker_epoch = -1
        self.epoch_no_improvement = 0
        self.patience = args.patience if hasattr(args, 'patience') else 0
        self.last_epoch = False
        self.ended_tracking = False
        self.max_epoch = args.max_epoch if hasattr(args, 'max_epoch') else 0
        self.tracker = EmissionsTracker(output_dir=args.save_dir if hasattr(args, 'save_dir') else './') #CarbonTracker(epochs=args.max_epoch, monitor_epochs=-1)

        self.batching_change = False
        self.batching_changed = True

        self.label_vocab = label_vocab
        self.num_features = 81

        self.blank_idx = label_vocab.blank()
        self.sos_tag_idx = label_vocab.index( SOS_tag )
        self.eos_tag_idx = label_vocab.index( EOS_tag )
        self.slu_start_concept_idx = label_vocab.index( slu_start_concept_mark )
        self.slu_end_concept_idx = label_vocab.index( slu_end_concept_mark ) 

        if self.args.roberta_model:
            print(' * End2EndSLU, loading RoBERTa model from: {}'.format(self.args.roberta_model))
            sys.stdout.flush()

            if 'camembert' in self.args.roberta_model:
                self.roberta = CamembertModel.from_pretrained( self.args.roberta_model ) 
            else:
                tokens = self.args.roberta_model.split('/')
                mpath = '/'.join(tokens[:-1]) + '/'
                mmodel = tokens[-1]
                self.roberta = RobertaModel.from_pretrained(mpath, checkpoint_file=mmodel)
            self.roberta = self.roberta.eval()
            for param in self.roberta.parameters():
                param.requires_grad = False
 
            #self.blank_idx = self.get_roberta_token_index(blank_token)
        else:
            self.roberta = None

        if self.args.model_start_point:
            print(' * End2EndSLU, loading model starting point from {}'.format(self.args.model_start_point))
            sys.stdout.flush()

            self.model_start_point = torch.load(self.args.model_start_point)

        if self.args.online_feature_extraction:
            if self.args.feature_extractor == 'whisper':
                self.fe = whisper.load_model( self.args.fe_size )
            else:
                raise NotImplementedError()
        else:
            self.fe = None

        if self.args.speech_encoder == 'sb-wav2vec2':
            self.args.feature_extension = '.wav'

        # Scheduled sampling state and needed values
        self.criterion = args.criterion
        self.scheduled_sampling = args.scheduled_sampling
        if self.scheduled_sampling:
            print(' - End2EndSLU task: using scheduled sampling')
            sys.stdout.flush()
        self.nsteps = 0
        self.best_train_loss = LOSS_INIT_VALUE
        self.curr_train_loss = LOSS_INIT_VALUE
        self.best_dev_loss = LOSS_INIT_VALUE
        self.curr_dev_loss = LOSS_INIT_VALUE
        self.dev_errors = 0
        self.dev_tokens = 0
        self.best_dev_er = ER_INIT_VALUE
        self.curr_dev_er = ER_INIT_VALUE
        self.tf_tries = 0
        self.ss_tries = 0
        self.tf_freedom_tries = 0
        self.ss_freedom_tries = 0
        self.switch_to_come = False
        self.init_ss_threshold = 0.99

        self.curr_epoch = 0
        self.max_tf_epochs = 5          # Max # of epochs with teacher forcing
        self.max_ss_epochs = 5          # Max # of epochs with scheduled sampling
        self.tf_patience = 3            # # of epochs of tf training without improvement
        self.ss_patience = 3            # Same for scheduled sampling
        self.forced_switch_epoch = 2    # Force switching between tf and ss training after this number of epochs at the beginning of training

    def set_scheduled_sampling(self, ss_val=False):
        self.scheduled_sampling = ss_val

    def switch_scheduled_sampling(self):
        self.scheduled_sampling = not self.scheduled_sampling

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True
            
        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                    gradient
                - logging outputs to display while training
        """

        loss, sample_size, logging_output = super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        self.curr_train_loss += loss.item()
        self.nsteps += 1

        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        self.curr_dev_loss += loss.item()
        if isinstance(criterion, slu_ctc.SLUCTCCriterion):
            self.dev_errors += logging_output['errors']
            self.dev_tokens += logging_output['total']

        if self.last_epoch and not self.ended_tracking:
            self.tracker.stop()
            self.ended_tracking = True

            print('[DEBUG] CO2 emission tracker stopped.')
            sys.stdout.flush()

        return loss, sample, logging_output

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""

        model.set_curr_epoch( epoch )
        if self.tracker_epoch == -1:
            print('[DEBUG] CO2 emission tracker epoch started')
            sys.stdout.flush()

            self.tracker.start()
        else:
            print('[DEBUG] CO2 emission tracker progress...')
            sys.stdout.flush() 
        self.tracker_epoch = epoch 

        n_ss_up = 0
        self.curr_epoch += 1
        if self.curr_epoch <= self.args.curriculum:
            self.datasets['train'].curriculum(value=True)
            #print('[DEBUG] End2EndSLU, setting curriculum learning to True')
            #sys.stdout.flush()
        if self.curr_epoch > self.args.curriculum:
            self.datasets['train'].curriculum(value=False)
            #print('[DEBUG] End2EndSLU, setting curriculum learning to False')
            #sys.stdout.flush()

            if self.args.curriculum == 0 or (not self.batching_changed and not self.batching_change and self.args.use_dialog_history):
                print('[DEBUG] End2EndSLU, instructing the model for dialog history use!')
                sys.stdout.flush()

                model.decoder.use_dialog_history(value=True)

            if self.args.curriculum > 0 and self.batching_changed:
                print('[DEBUG] End2EndSLU, batching change scheduled...')
                sys.stdout.flush()
                self.batching_change = True
                self.batching_changed = False

        if epoch > 1:
            self.init_ss_threshold *= 0.98
            if self.scheduled_sampling and epoch >= self.args.start_scheduled_sampling:
                n_ss_up = model.get_scheduled_sampling_updates()
                model.set_scheduled_sampling(ss_val=True, epoch=epoch)

            if self.nsteps > 0: # if we reload a checkpoint, epoch can be > 1, but nsteps is 0
                self.curr_train_loss /= self.nsteps
                self.curr_dev_loss /= self.nsteps
            if self.dev_tokens > 0:
                self.curr_dev_er = float(self.dev_errors) / float(self.dev_tokens)

            if self.criterion == 'slu_ctc_loss':
                print(' ----------')
                if self.scheduled_sampling:
                    print(' Dev error rate summary after epoch {} (scheduled sampling updates: {})'.format(epoch-1, n_ss_up))
                else:
                    print(' Dev error rate summary after epoch {}'.format(epoch-1))
                print(' * Current dev error rate: {:.4f} (best {:.4f})'.format(self.curr_dev_er, self.best_dev_er))
                print(' ----------')
                sys.stdout.flush()

            if self.best_train_loss > self.curr_train_loss:
                self.best_train_loss = self.curr_train_loss
            if self.best_dev_loss > self.curr_dev_loss:
                self.best_dev_loss = self.curr_dev_loss
            if self.best_dev_er > self.curr_dev_er:
                self.best_dev_er = self.curr_dev_er
                self.epoch_no_improvement = 0
            else:
                self.epoch_no_improvement += 1
                # Reset some ss state values
            #else:
                # Set some ss values

            if self.args.unfreeze_encoder_at_cer != 110.00 and self.curr_dev_er <= self.args.unfreeze_encoder_at_cer:
                print(' * End2EndSLU, dev cer {} reached, unfreezing encoder'.format(self.curr_dev_er))
                sys.stdout.flush()

                model.unfreeze_encoder()

        if self.epoch_no_improvement >= self.patience or self.tracker_epoch == self.max_epoch:
            print('[DEBUG] stopping CO2 emission tracker')
            sys.stdout.flush()

            self.last_epoch = True
            #self.tracker.stop()

        self.curr_train_loss = 0.0
        self.curr_dev_loss = 0.0
        self.curr_dev_er = 0.0
        self.dev_errors = 0
        self.dev_tokens = 0
        self.nsteps = 0

    def add_speaker_marker(self, feats, spk):
        """
            Add leading and trailing vectors as speaker marker for distinguishing input features from different speakers.
            When using the MEDIA corpus it is important to distinguish between the Machine, which is much easier to recognize, and a User.
        """

        #num_speakers = 2
        channel_per_speaker = 2
        #spk_abs_val = 1.0

        dims = feats.size()
        if len(dims) > 1:
            (src_len, dim) = feats.size()
        else:
            return feats

        #eod_mark = torch.sum( feats == EOD_value ).item()
        if _ADVANCED_SPK_MARK_:
            speaker_mark = torch.zeros(src_len, channel_per_speaker*self.num_speakers).to(feats)
        spk_val = 0.0
        if spk == user_ID:
            spk_val = +self.spk_abs_val 
            if _ADVANCED_SPK_MARK_:
                speaker_mark[:,:channel_per_speaker] = spk_val #if eod_mark == 0 else EOD_value
        elif spk == machine_ID:
            spk_val = -self.spk_abs_val
            if _ADVANCED_SPK_MARK_:
                speaker_mark[:,channel_per_speaker:2*channel_per_speaker] = spk_val #if eod_mark == 0 else EOD_value
        elif spk == bogus_ID: 
            spk_val = -2.0*self.spk_abs_val
            if _ADVANCED_SPK_MARK_:
                speaker_mark[:,2*channel_per_speaker:] = spk_val #if eod_mark == 0 else EOD_value
        else:
            raise NotImplementedError

        if _ADVANCED_SPK_MARK_:
            feats = torch.cat([feats, speaker_mark], 1)
            dim = dim + channel_per_speaker*self.num_speakers
        #if eod_mark > 0:
        #    spk_val = EOD_value
        padder = torch.zeros(3, dim).fill_(spk_val).to(feats)
        return torch.cat( [padder, feats, padder], 0 )

        #spk_mark = torch.zeros_like(feats).fill_(spk_val)
        #return feats + spk_mark

    def pad_short_sequences(self, corpus, split):

        def all_zeros(t:torch.Tensor):
            return torch.sum(t) == 0.0

        length_missmatch = 0
        missmatches = []
        total = 0
        max_turn_length = 0
        comp_t_idx = 0
        if self.args.slu_subtask == 'char':
            comp_t_idx = 2
        elif self.args.slu_subtask == 'token':
            comp_t_idx = 3
        elif self.args.slu_subtask == 'concept':
            comp_t_idx = 4 
        in_red_factor = (self.args.num_lstm_layers-1)*2 if hasattr(self.args, 'num_lstm_layers') else 4 

        # 1. First scan, detect sequences shorter than the expected output, once reduced by the LSTM pyramidal architecture
        for did in corpus[split]:
            total = total + len(corpus[split][did])
            for t_idx in range(len(corpus[split][did])):
                t = corpus[split][did][t_idx]

                if self.args.character_level_slu:
                    char_slu_turn = get_character_level_slu_turn(self.args, t[4], self.label_vocab)
                    char_t = char_slu_turn.split()

                    if len(char_t) > max_turn_length:
                        max_turn_length = len(char_t)
                        if max_turn_length > 1024:
                            print(' ### Got strangely long turn ({}): {}'.format(max_turn_length, char_slu_turn))

                    slu_t = torch.LongTensor( [self.label_vocab.add_symbol(c) for c in char_t] ) 
                    if not all_zeros(t[1]) and t[1].size(0)//in_red_factor < slu_t.size(0): 
                        length_missmatch += 1
                        missmatches.append( (t[1].size(0)//in_red_factor, slu_t.size(0)) )

                    turn_tuple = (t[0], t[1], t[2], t[3], slu_t, t[5])
                    corpus[split][did][t_idx] = turn_tuple
                else:
                    if t[comp_t_idx].size(0) > max_turn_length:
                        max_turn_length = t[comp_t_idx].size(0) 

                    if not all_zeros(t[1]) and t[1].size(0)//in_red_factor < t[comp_t_idx].size(0):
                        length_missmatch += 1
                        missmatches.append( (t[1].size(0)//in_red_factor, t[comp_t_idx].size(0)) ) 

        if split == 'train': # Ignore input/output length missmatch statistics if this is the test set
            max_len = 0
            max_ratio = 0.0
            for p in missmatches:
                if p[1] > max_len:
                    max_len = p[1]
                if p[1]/p[0] > max_ratio:
                    max_ratio = p[1]/p[0]
            if max_len > self.args.max_padding_len:
                self.args.max_padding_len = max_len
            if max_ratio > self.args.max_padding_ratio:
                self.args.max_padding_ratio = max_ratio 

        # 2. Second scan, pad short input sequences.
        padded_sequences = 0
        if split == 'train' and not (self.args.decoder == 'ctclstm' and not self.args.load_fairseq_encoder): # NOTE: we pad only training examples and only if we are not performing end2end training.
            for did in corpus[split]: 
                for t_idx in range(len(corpus[split][did])):
                    t = corpus[split][did][t_idx]

                    input_tsr = t[1]
                    if not all_zeros(t[1]) and t[1].size(0)//in_red_factor < t[comp_t_idx].size(0):
 
                        pad_len = t[comp_t_idx].size(0) * in_red_factor
                        T, C = t[1].size()
                        lratio = float(pad_len) / float(T)

                        input_tsr = torch.zeros(pad_len,C).to(t[1])
                        for i in range(pad_len):
                            input_tsr[i,:] = t[1][int(i/lratio),:].clone()
                        padded_sequences += 1

                    if not all_zeros(input_tsr):
                        assert input_tsr.size(0)//in_red_factor >= t[comp_t_idx].size(0), ' input-output sequence lengths missmatch after padding: {} vs. {}'.format(input_tsr.size(0)//in_red_factor, t[comp_t_idx].size(0))

                    turn_tuple = (t[0], input_tsr, t[2], t[3], t[4], t[5])
                    corpus[split][did][t_idx] = turn_tuple
 
        print(' * End2EndSLU: padded {} sequences'.format(padded_sequences))
        sys.stdout.flush()

        return corpus[split], (length_missmatch, missmatches, max_turn_length, total)

    def create_batches_as_dialogs(self, data, batch_info, batch_sizes, split, slu_mode_info):

        ids = []
        speakers = []
        ratios = []
        sources = []
        src_lengths = []
        targets = []
        tgt_lengths = []
        global_turn_idx = 0
        idx_batches = []

        if slu_mode_info['use_transcription'] or slu_mode_info['asr_slu_loss']:
            assert self.args.slu_subtask == 'concept', 'transcriptions are allowed as additional input only for SLU'

        for did in data.keys():

            batched_turns_idx = []
            for tid in range( len(data[did]) ):

                turn = data[did][tid]
                ids.append(turn[0])
                speakers.append(turn[-1])
                feats = turn[1]
                if feats is not None:
                    feats = self.add_speaker_marker(feats, turn[-1])
                if slu_mode_info['use_transcription'] or slu_mode_info['asr_slu_loss']:
                    sources.append( (feats, turn[3]) )
                else:
                    sources.append( feats )
                sig_len = feats.size(0) if feats is not None else 0
                if slu_mode_info['use_transcription'] or slu_mode_info['asr_slu_loss']:
                    src_lengths.append( (sig_len, turn[3].size(0)) )
                else:
                    src_lengths.append( sig_len )

                batched_turns_idx.append( (global_turn_idx, turn[-1]) )
                global_turn_idx += 1

                if self.args.slu_subtask == 'char':
                    targets.append( turn[2] )
                    tgt_lengths.append( turn[2].size(0) )

                    ratios.append( turn[2].size(0) / feats.size(0) )

                elif self.args.slu_subtask == 'token':
                    targets.append( turn[3] )
                    tgt_lengths.append( turn[3].size(0) )

                    ratios.append( turn[3].size(0) / feats.size(0) )

                elif self.args.slu_subtask == 'concept':
                    targets.append( turn[4] )
                    tgt_lengths.append( turn[4].size(0) )

                    ratios.append( turn[4].size(0) / feats.size(0) )

                else:
                    raise NotImplementedError

            idx_batches.append( batched_turns_idx )
        return idx_batches, (ids, speakers, sources, src_lengths, targets, tgt_lengths), (ratios, global_turn_idx)

    def rearrange_dialog_batches(self, data, batch_info, batch_sizes, split, slu_mode_info):

        ratios = []
        ids = []
        speakers = []
        sources = []
        src_lengths = []
        targets = []
        tgt_lengths = []
        global_turn_idx = 0
        idx_batches = []

        if slu_mode_info['use_transcription'] or slu_mode_info['asr_slu_loss']:
            assert self.args.slu_subtask == 'concept', 'Transcriptions are allowed as additional input only for SLU'

        #num_zeros = []
        #for dialog_id in data.keys():
        for batch, batch_size in zip(batch_info, batch_sizes):
            #for turn in turns:
            batched_turns_idx = []
            curr_turn_batch = []
            dialog_batch_size = batch_size #bsz
            #while len(batch) % dialog_batch_size != 0:
            #    dialog_batch_size -= 1
            #if dialog_batch_size == 0:
            #    raise ValueError('FATAL ERROR: zero dialog batch size detected')
            #print('[DEBUG] End2EndSLU.rearrange_dialog_batches, current dialog batch size: {}, total length {}'.format(dialog_batch_size, len(batch)))
            #sys.stdout.flush()
            for (did, idx) in batch:
                turn = data[did][idx]
                #if (split == 'train') or (turn[-1] == user_ID) or slu_mode_info['dialog_level_slu'] or self.args.corpus_name != 'media':
                if not self.args.user_only or (self.args.user_only and turn[-1] == user_ID) or slu_mode_info['dialog_level_slu']:
                    #if turn[-1] == user_ID:
                    # if (not self.args.user_only and (split == 'train' or turn[-1] == user_ID)) or (self.args.user_only and turn[-1] == user_ID)
                    ids.append(turn[0])
                    speakers.append(turn[-1])
                    feats = turn[1]
                    if feats is not None:
                        feats = self.add_speaker_marker(feats, turn[-1])
                    #tmp = int((feats == 0.0).sum())
                    #if tmp > 0:
                    #    num_zeros.append( tmp )
                    if slu_mode_info['use_transcription'] or slu_mode_info['asr_slu_loss']:
                        sources.append( (feats, turn[3]) )
                    else:
                        sources.append( feats )
                    sig_len = feats.size(0) if feats is not None else 0
                    if slu_mode_info['use_transcription'] or slu_mode_info['asr_slu_loss']:
                        src_lengths.append( (sig_len, turn[3].size(0)) )
                    else:
                        src_lengths.append( sig_len )

                    '''print(' - End2EndSLU task data shapes:')
                    print('   - feats shape: {}'.format(feats.size()))
                    print('   - char seq shape: {}'.format(turn[2].size()))
                    print('   - token seq shape: {}'.format(turn[3].size()))
                    print('   - slu seq shape: {}'.format(turn[4].size()))
                    print(' -----')
                    sys.stdout.flush()'''

                    if self.args.dialog_level_slu:
                        curr_turn_batch.append( (global_turn_idx, turn[-1]) )
                        if len(curr_turn_batch) >= dialog_batch_size:
                            batched_turns_idx.append( curr_turn_batch )
                            curr_turn_batch = []
                    else:
                        batched_turns_idx.append( (global_turn_idx, turn[-1]) )
                    global_turn_idx += 1

                        #print('[DEBUG] End2ENdSLU.rearrange_dialog_batches: stored batch of size {} (batch size reset to {})'.format(len(idx_batches[-1]), len(batched_turns_idx)))
                        #sys.stdout.flush()

                    if self.args.slu_subtask == 'char':
                        targets.append( turn[2] )
                        tgt_lengths.append( turn[2].size(0) )

                        ratios.append( turn[2].size(0) / feats.size(0) )

                    elif self.args.slu_subtask == 'token':
                        targets.append( turn[3] )
                        tgt_lengths.append( turn[3].size(0) )

                        ratios.append( turn[3].size(0) / feats.size(0) )

                    elif self.args.slu_subtask == 'concept':
                        targets.append( turn[4] )
                        tgt_lengths.append( turn[4].size(0) )

                        ratios.append( turn[4].size(0) / feats.size(0) )

                    else:
                        raise NotImplementedError
                else:
                    data[did][idx] = None

            if split == 'train':
                assert len(curr_turn_batch) == 0, 'ERROR, wrong remainder size in current batch: {}. Expected 0 with batch size {} and total dialog length {}'.format(len(batched_turns_idx), dialog_batch_size, len(batch))
            if self.args.dialog_level_slu:
                if len(curr_turn_batch) > 0:
                    assert len(curr_turn_batch) == bsz
                    batched_turns_idx.append( curr_turn_batch )
                if len(batched_turns_idx) > 0:
                    idx_batches.append( batched_turns_idx )

                '''print('[DEBUG]### End2EndSLU, storing {} dialog batches of size {}'.format(len(batched_turns_idx), len(batched_turns_idx[0])))
                print('[DEBUG]###     * type of idx_batches: {}'.format(type(idx_batches))) # Expected List
                print('[DEBUG]###     * type of elements of idx_batches: {} (size: {})'.format(type(idx_batches[-1]), len(idx_batches[-1])))  # Expected List
                print('[DEBUG]###     * type of elements of elements of idx_batches: {}, length {}'.format(type(idx_batches[-1][0]), len(idx_batches[-1][0])))   # Expected List
                print('[DEBUG]###     * type of deepest elements: {}'.format(type(idx_batches[-1][0][0])))  # Expected tuple
                sys.stdout.flush()'''
            else:
                idx_batches.append( batched_turns_idx )

        return idx_batches, (ids, speakers, sources, src_lengths, targets, tgt_lengths), (ratios, global_turn_idx)

    def _tmp_count_corpus_split(self, data):

        n_turns = 0
        n_tgt_tokens = 0
        for did in data.keys():
            n_turns += len(data[did])
            for tt in data[did]:
                n_tgt_tokens += tt[4].size(0)

        print('[TMP DEBUG] _tmp_count_corpus_split: {} turns, {} tokens'.format(n_turns, n_tgt_tokens))
        sys.stdout.flush()

    def get_roberta_token_index(self, token):

        assert self.roberta is not None
        tt = self.roberta.encode(token)
        assert tt.size(0) == 3 and tt[0].item() == self.roberta.task.source_dictionary.bos() and tt[-1].item() == self.roberta.task.source_dictionary.eos()
        return tt[1].item()

    def load_dstc_extra_data(self, data_file, original_data, vocab):

        # Format example:
        # line_nr: 1 dialog_id: 0-tpa-sng01856.json turn_id: 3 text: user: no i just need to make sure it's cheap and i need parking state: hotel-parking=yes; hotel-pricerange=cheap; hotel-type=hotel

        dialog_data = turn_transcripts_and_sem(data_file, mode='train', save_turns=False)

        first_key = list(dialog_data.keys())[0]
        print('[DEBUG] read {} extra dialogs'.format(len(dialog_data)))
        #print('[DEBUG] first dialog data (key: {}):'.format( first_key ))
        #print(dialog_data[first_key])
        print('[DEBUG] -----')
        sys.stdout.flush()

        ok = list(original_data.keys())[0]
        num_features = original_data[ok][0][1].size(1)

        extra_data = {}
        for did in dialog_data.keys():

            if did in original_data:
                raise ValueError('DID {} is already defined (and should not!)'.format(did))
            orig_did = did.split('-')[-1]
            assert orig_did in original_data

            #orig_tidxs = [int(original_data[orig_did][i][0].split('@')[-1].split('-')[1]) for i in range(len(original_data[orig_did]))]
            #original_data[orig_did] = sorted(original_data[orig_did], key=lambda list: int(list[0].split('@')[-1].split('-')[1]))
            #print('[DEBUG] ********************')
            #print('[DEBUG] Comparing dialogues {} (lengths: original {} vs. extra-data {})'.format(orig_did, len(original_data[orig_did]), len(dialog_data[did])))
            #print('[DEBUG]  * original data tid sequence: {}'.format([original_data[orig_did][i][0].split('@')[-1].split('-')[1] for i in range(len(original_data[orig_did]))]))
            #print('[DEBUG] -----')
            #print('[DEBUG]  * extra data tid sequence: {}'.format([td['turn_id'] for td in dialog_data[did]]))
            #print('[DEBUG] -----')
            #sys.stdout.flush()

            turn_list = []
            are_identical = True
            for tidx, td in enumerate(dialog_data[did]):

                orig_tidx = original_data[orig_did][tidx][0].split('@')[-1].split('-')[1]
                assert int(orig_tidx) == tidx+1, 'Turn ID missmatch, original {} vs. extra-data {} @ {}'.format(orig_tidx, tidx+1, did)
                spk = 'User' if td['spk'] == 'user:' else 'Machine'
                TID = did.strip() + '@Turn-' + str(td['turn_id']).strip() + '-' + spk
                sem = clean_sem( td['sem'] )
                if original_data[orig_did][tidx][3] == sem:
                    spg = original_data[orig_did][tidx][1]
                    if spg is None:
                        print(' *** WARNING: detected None input signal coming from original data @{}'.format(TID))
                        sys.stdout.flush()
                else:
                    #spg_len = len(sem.split()) * 4 + 1
                    spg = torch.zeros(4, num_features)
                assert spg is not None, 'Found None input signal @{}'.format(TID)

                #print('[DEBUG] *****')
                #print('[DEBUG] original {} vs. extra-data {} turn {} comparison'.format(orig_did, did, tidx))
                #print('[DEBUG]    * ID: {} vs. {}'.format(original_data[orig_did][tidx][0], TID))
                #print('[DEBUG]    * Txt: {} vs. {}'.format(original_data[orig_did][tidx][2], td['txt']))
                #print('[DEBUG]    * Sem >!{}!<: {} vs. {}'.format('MATCH' if original_data[orig_did][tidx][3] == sem else 'MISSMATCH', original_data[orig_did][tidx][3], sem))
                #print('[DEBUG]    * Spk: {} vs. {}'.format(original_data[orig_did][tidx][4], spk))
                #print('[DEBUG]')
                #print('[DEBUG] --------------------')
                #sys.stdout.flush()

                #if did not in original_data:
                #    original_data[did] = []
                #original_data[did].append( [TID, spg, td['txt'], sem, td['spk']] )
                are_identical = are_identical and (original_data[orig_did][tidx][2].strip() == td['txt'].strip())
                turn_list.append( [TID, spg, td['txt'], sem, spk] )
            if not are_identical and len(turn_list) > 0:
                extra_data[did] = turn_list
            elif len(turn_list) == 0:
                raise ValueError('Dailogue {} looks empty'.format(did))
            else:
                print(' *** dialogue {} in DSTC extra data is redundant, skipping...'.format(did))
                sys.stdout.flush()

        return extra_data

    def load_dataset(self, split, **kwargs):
    
        """Load a given dataset split (e.g., train, valid, test)."""

        corpus = {}
        my_split = split
        if my_split == 'valid':
            my_split = 'dev'    # For compatibility with the old End2End SLU system (ICASSP 2020) 

        if (not self.args.load_dstc_ex_data or split != 'train') and os.path.exists(self.args.serialized_data + '.' + my_split) and os.path.isfile(self.args.serialized_data + '.' + my_split): 

            print(' - Loading serialized {} split...'.format(my_split))
            sys.stdout.flush()

            corpus[my_split] = torch.load(self.args.serialized_data + '.' + my_split)
        else:
            print(' - Reading dialog data')
            print('   * NOTE: reading SLU turn in {} format'.format(self.args.slu_turn_format))

            extra_data = None
            if 'train' not in self.tmp_corpus:
                print('   - reading train data...')
                sys.stdout.flush()
                train_data = read_dialog_data( self.args.data + '/train_prefixes.lst', self.args )
                print('   - read {} train dialogues'.format(len(train_data)))

                if self.args.load_dstc_ex_data:
                    print('    * reading DSTC extra data...')
                    sys.stdout.flush()

                    extra_data = self.load_dstc_extra_data( self.args.load_dstc_ex_data, train_data, self.label_vocab )
                self.tmp_corpus['train'] = train_data
            else:
                train_data = self.tmp_corpus['train']

            if 'valid' not in self.tmp_corpus:
                print('   - reading dev data...')
                sys.stdout.flush()
                dev_data = read_dialog_data( self.args.data + '/dev_prefixes.lst', self.args )
                self.tmp_corpus['valid'] = dev_data
                print('   - read {} dev dialogues'.format(len(dev_data)))
            else:
                dev_data = self.tmp_corpus['valid']

            if 'test' not in self.tmp_corpus:
                print('   - reading test data...')
                sys.stdout.flush()
                test_data = read_dialog_data( self.args.data + '/test_prefixes.lst', self.args )
                self.tmp_corpus['test'] = test_data
                print('   - read {} test dialogs'.format(len(test_data)))
            else:
                test_data = self.tmp_corpus['test']

            preprocess_data = False
            if not self.args.online_feature_extraction and self.args.speech_encoder != 'sb-wav2vec2':
                print(' - Normalizing input data...' )
                sys.stdout.flush() 
                if not self.args.load_dstc_ex_data and 'norm_var' not in self.tmp_corpus:
                    print('  - computing mean and std of training data...')
                    sys.stdout.flush()

                    threshold=0.0
                    #if self.args.load_dstc_ex_data:
                    #    threshold = 0.9
                    (train_mu, train_sigma) = feature_wise_mean_std( train_data, sample_threshold=threshold )
                    self.tmp_corpus['norm_var'] = (train_mu, train_sigma)
                    preprocess_data = True

                    print('   * mean: max={}, min={}, mean={}'.format(torch.max(train_mu), torch.min(train_mu), torch.mean(train_mu)))
                    print('   * std : max={}, min={}, mean={}'.format(torch.max(train_sigma), torch.min(train_sigma), torch.mean(train_sigma)))
                    sys.stdout.flush()
                elif 'norm_var' in self.tmp_corpus:
                    (train_mu, train_sigma) = self.tmp_corpus['norm_var']

                if preprocess_data:
                    print('  - z-scaling data feature-wise...')
                    sys.stdout.flush()

                    normalize_data_only(train_data, train_mu, train_sigma)
                    print('    * train done.')
                    sys.stdout.flush()
                    if extra_data is not None:
                        normalize_data_only(extra_data, train_mu, train_sigma)
                        # NOTE: merges train_data and extra_data
                        for did in extra_data.keys():
                            assert did not in train_data
                            train_data[did] = extra_data[did]
                        print('     * extra data done.')
                        sys.stdout.flush()

                    normalize_data_only(dev_data, train_mu, train_sigma)
                    print('    * dev done.')
                    sys.stdout.flush()

                    normalize_data_only(test_data, train_mu, train_sigma)
                    print('    * test done.')
                    sys.stdout.flush()

            if 'norm_var' not in self.tmp_corpus and extra_data is not None:
                for did in extra_data.keys():
                    assert did not in train_data
                    train_data[did] = extra_data[did]

            print(' - Mapping tokens to indexes...')
            sys.stdout.flush()
            if preprocess_data or 'norm_var' not in self.tmp_corpus:
                references2indexes(self.args, train_data, self.label_vocab, 'train', roberta=None) 
                references2indexes(self.args, dev_data, self.label_vocab, 'dev', roberta=None)
                references2indexes(self.args, test_data, self.label_vocab, 'test', roberta=None)
            print('   - Dictionary size: {}'.format(len(self.label_vocab)))
            sys.stdout.flush()

            corpus = {}
            corpus['train'] = train_data
            corpus['dev'] = dev_data
            corpus['test'] = test_data
            corpus['dictionary'] = self.label_vocab

            print(' - Saving serialized corpus...')
            sys.stdout.flush()
            if not self.args.load_dstc_ex_data:
                torch.save(train_data, self.args.serialized_data + '.train' )
            else:
                print('   * Loaded extra DSTC data, skipping train data serialization (too big!)')
                sys.stdout.flush()

            torch.save(dev_data, self.args.serialized_data + '.dev' )
            torch.save(test_data, self.args.serialized_data + '.test' )
            if self.args.load_dictionary:
                torch.save(self.label_vocab, self.args.load_dictionary)
            else:
                torch.save(self.label_vocab, self.args.serialized_data + '.dict' )

        if not 'dictionary' in corpus:
            if self.args.load_dictionary:
                corpus['dictionary'] = self.label_vocab
            else:
                corpus['dictionary'] = torch.load(self.args.serialized_data + '.dict')
            self.label_vocab = corpus['dictionary'] 
            print(' - Loaded dictionary size: {}'.format(len(self.label_vocab)))
            sys.stdout.flush() 

        if my_split == 'train' and 'test' in corpus:
            corpus['test'] = None
        if 'train' in self.tmp_corpus:
            self.tmp_corpus['test'] = None
        assert self.blank_idx == self.label_vocab.blank() or self.blank_idx == self.get_roberta_token_index(blank_token)
        assert self.sos_tag_idx == self.label_vocab.index( SOS_tag )
        assert self.eos_tag_idx == self.label_vocab.index( EOS_tag )
        assert self.slu_start_concept_idx == self.label_vocab.index( slu_start_concept_mark )
        assert self.slu_end_concept_idx == self.label_vocab.index( slu_end_concept_mark )

        # data for each turn are: turn-id, signal-tensor, char-sequence (int) tensor, token-sequence (int) tensor, concept-sequence (int) tensor, speaker-id (machine or user)
        # data are organized as a dict of lists: the dict is indexed with dialog-id, the value is the list of turns for that dialog, each turn structure contains data described above.  

        #debug_idx = 0
        #if hasattr(self.args, 'num_lstm_layers'):
        print(' * End2EndSLU: checking for input-output sequence length incompatibility...')
        sys.stdout.flush()

        #print('[DEBUG] dict size before: {}'.format(len(self.label_vocab)))

        cc, infos = self.pad_short_sequences(corpus, my_split)
        corpus[my_split] = cc
        length_missmatch, missmatches, max_turn_length, total = infos 

        #print('[DEBUG] dict size after: {}'.format(len(self.label_vocab)))
        #sys.stdout.flush()

        if length_missmatch > 0:
            msg = ''
            if my_split == 'train':
                msg = ' (and solved by padding)'
            if self.args.character_level_slu:
                print('   *** End2EndSLU: {} out of {} input-output length incompatibility detected{} converting to character-level SLU output'.format(length_missmatch, total, msg))
                print('     * Missmatches: {}'.format(missmatches))
                print('   *** End2EndSLU: max character-level turn length is {}'.format(max_turn_length))
                sys.stdout.flush()
            else:
                print('   *** End2EndSLU: {} out of {} input-output length incompatibility detected{}'.format(length_missmatch, total, msg))
                print('     * Missmatches: {}'.format(missmatches))
                print('   *** End2EndSLU: max turn length is {}'.format(max_turn_length))
                sys.stdout.flush()
        else:
            print('   Done.')
            sys.stdout.flush() 

        if 'train' in corpus:
            print(' - Filtering training examples where input sequence is longer than {}'.format(self.args.max_source_positions))
            sys.stdout.flush()

            filtered = 0
            new_split = {}
            for did in corpus['train'].keys():
                new_dd = []
                for tt in corpus['train'][did]:
                    if tt[1].size(0) <= self.args.max_source_positions:
                        new_dd.append( tt )
                    else:
                        filtered += 1
                if len(new_dd) > 0:
                    new_split[did] = new_dd

            corpus['train'] = new_split
            print('    * Filtered {} turns.'.format(filtered))
            sys.stdout.flush()

        print(' - Reorganizing {} dialog data...'.format(my_split))
        if self.args.user_only:
            print('   - Loading user turns only.') 

        dialogs = corpus[my_split]

        # TMP for DEBUG
        #for did in dialogs.keys():
            #for turn in dialogs[did]:
                #print(' * tensor size: {}'.format(turn[1].size()))
                #sys.stdout.flush()
                #assert turn[1].size(-1) == 1024 and len(turn[1].size()) == 2, 'Found tensor with wrong dimensions: {}'.format(turn[1].size())
        # End of TMP for DEBUG

        bsz = 1
        #if my_split == 'train':
        while( bsz < self.args.max_sentences ):
            bsz = bsz * 2
        if bsz > self.args.max_sentences:
            bsz = int(bsz / 2)
        if bsz < self.args.max_sentences:
            bsz = self.args.max_sentences

        print(' - Approximating batch size to {} from {}'.format(bsz, self.args.max_sentences))
        sys.stdout.flush()
        slu_mode_info = {}
        slu_mode_info['dict'] = self.label_vocab #if self.roberta is None else self.roberta.task.source_dictionary
        slu_mode_info['roberta'] = None #self.roberta
        slu_mode_info['dialog_level_slu'] = self.args.dialog_level_slu if hasattr(self.args, 'dialog_level_slu') else False
        slu_mode_info['normalize_dialogs'] = self.args.normalize_dialog_batches if hasattr(self.args, 'normalize_dialog_batches') else False
        slu_mode_info['use_transcription'] = self.args.use_transcription_as_input if hasattr(self.args, 'use_transcription_as_input') else False
        slu_mode_info['asr_slu_loss'] = self.args.asr_slu_loss if hasattr(self.args, 'asr_slu_loss') else False
        batch_info, batch_sizes = create_dialog_batches_(dialogs, bsz, slu_mode_info) 

        #print('[DEBUG] batch_sizes content: {}'.format(batch_sizes))
        #tmp_lst = [len(el) for el in batch_info]
        #print('[DEBUG] batch_info batch sizes: {} (total: {})'.format(tmp_lst, sum(tmp_lst))) 
        #sys.stdout.flush()

        if self.args.dialog_batches:
            idx_batches, split_data, info = self.create_batches_as_dialogs(dialogs, batch_info, batch_sizes, my_split, slu_mode_info)
        else:
            idx_batches, split_data, info = self.rearrange_dialog_batches(dialogs, batch_info, batch_sizes, my_split, slu_mode_info)

            #print('[DEBUG] idx_batches structure:')
            #for i in range( len(idx_batches) ):
            #    print('[DEBUG]   * Size of current dialog batch: {}'.format(len(idx_batches[i])))
            #    for j in range( len(idx_batches[i]) ):
            #        print('[DEBUG]     ** Size of current turn batch: {}'.format(len(idx_batches[i][j])))
            #sys.stdout.flush()
            #sys.exit(0)

        ids, speakers, sources, src_lengths, targets, tgt_lengths = split_data
        ratios, global_turn_idx = info 

        print(' * End2EndSLU, reorganized {} turns for split {}'.format(global_turn_idx, my_split))
        #print('   - Detected {} input tensors containing zeros: {}'.format(len(num_zeros), num_zeros))
        sys.stdout.flush()
        src_lengths_tsr = torch.FloatTensor( [t[0] for t in src_lengths] ) if hasattr(self.args, 'use_transcription_as_input') and self.args.use_transcription_as_input else torch.FloatTensor( src_lengths )
        mean = torch.mean(src_lengths_tsr)
        median = torch.median(src_lengths_tsr)
        std = torch.std( src_lengths_tsr )
        max_len = torch.max( src_lengths_tsr )
        print(' - {} data statistics:'.format(my_split))
        print('   * Total # of turns: {}'.format(len(sources)))
        print('   * Mean source length: {:.2f}'.format(mean.item()))
        print('     - Std of source length: {:.2f}'.format(std.item()))
        print('     - Median source length: {}'.format(median.item()))
        print('     - Max. source length: {}'.format(max_len.item()))
        if len(ratios) > 1:
            rmean = sum(ratios)/len(ratios)
            rstd = math.sqrt( sum([(el - rmean)**2 for el in ratios]) / (len(ratios)-1) )
            print('   * Mean target/source length ratio: {:.4f} (+/- {:.4f})'.format(rmean, rstd))
            print('   * Max. target/source length ratio: {:.4f}'.format(max(ratios)))
            print('   * Min. target/source length ratio: {:.4f}'.format(min(ratios)))
            sys.stdout.flush() 

        # Reorganize turns based on increasing source length for curriculum learning 
        '''src_info = [(i, sources[i].size(0)) for i in range(len(sources))]
        sorted_structure = sorted(src_info, key=lambda tuple: tuple[1])
        sources = [sources[t[0]] for t in sorted_structure]
        src_lengths = [src_lengths[t[0]] for t in sorted_structure]
        targets = [targets[t[0]] for t in sorted_structure]
        tgt_lengths = [tgt_lengths[t[0]] for t in sorted_structure]'''
        sample_source = None
        if self.args.use_transcription_as_input or self.args.asr_slu_loss:
            found = False
            for tt in sources:
                if tt[0] is not None:
                    sample_source = tt[0]
                    found = True
                    break
            if not found:
                raise ValueError('At least one input signal is expected to be not None')
        else:
            found = False
            for t in sources:
                if t is not None:
                    sample_source = t
                    found = True
                    break
            if not found:
                raise ValueError('At least one input signal is expected to be not None')

        self.num_features = sample_source.size(-1)

        input_feed = True
        if split != 'test' or 'train' not in self.datasets:
            self.datasets[split] = End2EndSLUDataset.End2EndSLUDataset(
                args=self.args,
                split=split,
                src=sources,
                src_sizes=src_lengths,
                tgt=targets,
                tgt_sizes=tgt_lengths,
                tgt_dict=self.label_vocab,
                idx_structure=idx_batches,
                left_pad_target=False,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions, 
                input_feeding=input_feed,
                shuffle=True,
                RoBERTa=None,
                feature_extractor=self.fe,
                turn_ids=ids,
                turn_spk=speakers,
                spk_abs_val=self.spk_abs_val,
            )
        #self.datasets[split].set_split_id(value=split)

        if split == 'train' and self.args.curriculum > 0:
            #print('[DEBUG] End2EndSLU, setting curriculum learning to True for train split (curriculum epoch {})'.format(self.args.curriculum))
            #sys.stdout.flush()
            self.datasets['train'].curriculum(value=True)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        if self.args.use_transcription_as_input:
            #if self.roberta is None:
            return self.label_vocab
            #else:
            #    return self.roberta.task.source_dictionary
        else:
            return None

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        #if self.roberta is None:
        return self.label_vocab
        #else:
        #    return self.roberta.task.source_dictionary

    # We could override this method if we wanted more control over how batches
    # are constructed, but it's not necessary for this tutorial since we can
    # reuse the batching provided by LanguagePairDataset.
    #
    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=1):

        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
        Returns:
            SLUEpochBatchIterator: a batched iterator over the
                given dataset split
        """

        #print('[DEBUG] End2EndSLU.get_batch_iterator: num_shards, shard_id, num_workers: {}, {}, {}'.format(num_shards, shard_id, num_workers))
        #sys.stdout.flush()

        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if not self.args.model_start_point and dataset in self.dataset_to_epoch_iter and not self.batching_change:
            #print('[DEBUG] End2EndSLU.get_batch_iterator: returning buffered epoch iterator for split {} @epoch {}'.format(dataset.get_split_id(), self.curr_epoch))
            #sys.stdout.flush()
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        '''print('[DEBUG] 1) End2EndSLU.get_batch_iterator, got {} size indices structure @split {}:'.format(len(indices), dataset.get_split_id()))
        print('[DEBUG]    * type of indices: {}'.format(type(indices)))
        print('[DEBUG]    * type of indices element: {} (size: {})'.format(type(indices[0]), len(indices[0])))
        sys.stdout.flush()'''

        # filter examples that are too large
        if max_positions is not None and not self.args.use_transcription_as_input and not self.args.dialog_level_slu and not self.args.dialog_batches:
            if hasattr(self, 'training'):
                #max_positions = 2000000 # NOTE: try not to skeep long sequences at validation phase
                indices = data_utils.filter_by_size(
                    indices,
                    dataset,
                    max_positions,
                    raise_exception=(not ignore_invalid_inputs),
                )

        '''print('[DEBUG] 2) End2EndSLU.get_batch_iterator, got {} size indices structure @split {}:'.format(len(indices), dataset.get_split_id()))
        print('[DEBUG]    * type of indices: {}'.format(type(indices)))
        print('[DEBUG]    * type of indices element: {} (size: {})'.format(type(indices[0]), len(indices[0])))
        sys.stdout.flush()'''

        # create mini-batches with given size constraints 
        if type(indices[0]) != list and type(indices[0]) != tuple: #not self.args.dialog_level_slu or (self.curr_epoch <= self.args.curriculum and dataset.get_split_id() == 'train'):
            #print('[DEBUG] End2EndSLU.get_batch_iterator: batchizing indices @split {}'.format(dataset.get_split_id()))
            #sys.stdout.flush()

            batch_sampler = data_utils.batch_by_size(
                indices,
                dataset.num_tokens,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            batch_sampler = indices

            #print('[DEBUG] End2EndSLU.get_batch_iterator: keeping original indices as batch sampler for split {}'.format(dataset.get_split_id()))
            #sys.stdout.flush()

        '''print('[DEBUG] End2EndSLU.get_batch_iterator, got {} size batch sampler @split {}:'.format(len(batch_sampler), dataset.get_split_id()))
        print('[DEBUG]    * type of batch sampler: {}'.format(type(batch_sampler)))
        print('[DEBUG]    * type of batch sampler element: {} (size: {})'.format(type(batch_sampler[0]), len(batch_sampler[0])))
        print('[DEBUG]    * type of elements of elements: {} (size: {})'.format(type(batch_sampler[0][0]), len(batch_sampler[0][0])))
        print('[DEBUG] *** Visualizing a dialog:')
        mid_idx = int(len(batch_sampler)/2)
        dialog_batch = batch_sampler[mid_idx]
        for idx, turn_batch in enumerate(dialog_batch):
            print('* {}) {}'.format(idx, self.label_vocab.string(dataset.tgt[turn_batch[0]])))
        print('[DEBUG] -----')
        sys.stdout.flush()'''

        # return a reusable, sharded iterator
        epoch_iter = SLUEpochBatchIterator(
            dataset=dataset,
            dictionary=self.label_vocab,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            curriculum=self.args.curriculum if hasattr(self.args, 'curriculum') else 0,
            dialog_level=self.args.dialog_level_slu
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        if dataset.get_split_id() == 'train':
            self.batching_change = False

        #print('[DEBUG] End2EndSLU.get_batch_iterator: returning new epoch iterator for data split {} @epoch {}'.format(dataset.get_split_id(), self.curr_epoch))
        #sys.stdout.flush()

        return epoch_iter

    #def build_generator(self, args):
    #    from fairseq.slu_sequence_generator import SLUSequenceGenerator

    #    print(' * End2End SLU task: using SLUSequenceGenerator')
    #    sys.stdout.flush()

    #    return SLUSequenceGenerator(
    #        self.target_dictionary,
    #        max_len_a=getattr(args, "max_len_a", 0),
    #        max_len_b=getattr(args, "max_len_b", 200),
    #    )



























