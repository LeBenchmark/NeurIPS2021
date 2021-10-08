

import sys
import logging

import numpy as np
import torch

from . import data_utils, FairseqDataset
from fairseq.tarc_utils import *

# Use this to show warnings (anything else ?) at the command line
logger = logging.getLogger(__name__)

def collate_tokens_ex(values, pad_idx, bos_idx=None, eos_idx=None, left_pad=False, move_trail=False):
    """ Convert a list of 1d tensors into a padded 2d tensor.
        This is a generalization of the funcion in fairseq.data_utils which can either move eos to the beginning,
        or bos to the end (for backward decoders)
    """

    assert not ((bos_idx is not None) and (eos_idx is not None)), 'collate_tokens_ex: either bos index or eos index must be not None, got both None'

    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    for i, v in enumerate(values):
        copy_tensor_for_collate(v, res[i][size - len(v):] if left_pad else res[i][:len(v)], bos_idx, eos_idx, move_trail)
    return res

def collate(
            samples, pad_idx, bos_idx, eos_idx, sequence_separator, left_pad_source=False, left_pad_target=False,
            input_feeding=True, granularity_flags=(False, True)
):

    if len(samples) == 0:
        return {}

    def merge_tokens(key, g_idx, left_pad, bos=None, eos=None, move_trail=False):
        res = []
        for idx in range(len(samples[0][key][g_idx])): # All samples have the same structure, thus we can safely choose the one at index 0 to know how many tensors are associated to 'key'
            res.append( collate_tokens_ex(
                                        [s[key][g_idx][idx] for s in samples],
                                        pad_idx, bos_idx=bos, eos_idx=eos, left_pad=left_pad, move_trail=move_trail,
                        )
            )
        return res 

    id = torch.LongTensor([s['id'] for s in samples]) 
    toks_src_tokens = merge_tokens('source', g_idx=0, left_pad=left_pad_source)
    char_src_tokens = merge_tokens('source', g_idx=1, left_pad=left_pad_source) 

    # sort by descending source length
    toks_src_lengths = []
    char_src_lengths = []
    src_tok_bounds = []
    assert len( samples[0]['source'][0] ) == len( samples[0]['source'][1] )
    for idx in range( len( samples[0]['source'][0] ) ):
        toks_src_lengths.append( torch.LongTensor([s['source'][0][idx].ne(pad_idx).long().sum() for s in samples]) ) 
        char_src_lengths.append( torch.LongTensor([s['source'][1][idx].ne(pad_idx).long().sum() for s in samples]) )
        src_tok_bounds.append( [s['src_tok_bounds'][idx] for s in samples] )

    bogus, sort_order = toks_src_lengths[0].sort(descending=True)
    bogus, char_sort_order = char_src_lengths[0].sort(descending=True) 

    if (not granularity_flags[0]) and granularity_flags[1]: 
        sort_order = char_sort_order
    toks_src_lengths = [t.index_select(0, sort_order) for t in toks_src_lengths]
    char_src_lengths = [t.index_select(0, sort_order) for t in char_src_lengths]
    id = id.index_select(0, sort_order)
    toks_src_tokens = [t.index_select(0, sort_order) for t in toks_src_tokens]
    char_src_tokens = [t.index_select(0, sort_order) for t in char_src_tokens]
    for li in range(len(src_tok_bounds)):
        src_tok_bounds[li] = [src_tok_bounds[li][i] for i in sort_order]

    toks_target = merge_tokens('target', g_idx=0, left_pad=left_pad_target)
    char_target = merge_tokens('target', g_idx=1, left_pad=left_pad_target)
    toks_target = [t.index_select(0, sort_order).type(torch.LongTensor) for t in toks_target]
    char_target = [t.index_select(0, sort_order).type(torch.LongTensor) for t in char_target]

    toks_tgt_lengths = []
    char_tgt_lengths = []
    assert len( samples[0]['target'][0] ) == len( samples[0]['target'][1] )
    for idx in range( len( samples[0]['target'][0] ) ):
        toks_tgt_lengths.append( torch.LongTensor([s['target'][0][idx].size(0) for s in samples]).index_select(0, sort_order) )
        char_tgt_lengths.append( torch.LongTensor([s['target'][1][idx].size(0) for s in samples]).index_select(0, sort_order) ) 
    toks_ntokens = []
    char_ntokens = []
    tgt_tok_bounds = []
    for idx in range( len( samples[0]['target'][0] ) ):
        toks_ntokens.append( sum(len(s['target'][0][idx]) for s in samples) )
        char_ntokens.append( sum(len(s['target'][1][idx]) for s in samples) )
        tgt_tok_bounds.append( [s['tgt_tok_bounds'][idx] for s in samples] )
    for li in range(len(tgt_tok_bounds)):
        tgt_tok_bounds[li] = [tgt_tok_bounds[li][i] for i in sort_order]

    toks_prev_output_tokens = None
    char_prev_output_tokens = None
    toks_next_output_tokens = None
    char_next_output_tokens = None
    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        toks_prev_output_tokens = merge_tokens(
            'target',
            g_idx=0,
            left_pad=left_pad_target,
            bos=None,
            eos=eos_idx,
            move_trail=True,
        )
        toks_prev_output_tokens = [t.index_select(0, sort_order).type(torch.LongTensor) for t in toks_prev_output_tokens]

        char_prev_output_tokens = merge_tokens(
            'target',
            g_idx=1,
            left_pad=left_pad_target,
            bos=None,
            eos=eos_idx,
            move_trail=True,
        )
        char_prev_output_tokens = [t.index_select(0, sort_order).type(torch.LongTensor) for t in char_prev_output_tokens]

        # This is created but not used for now...
        toks_next_output_tokens = merge_tokens(
            'target',
            g_idx=0,
            left_pad=left_pad_target,
            bos=bos_idx,
            eos=None,
            move_trail=True,
        )
        toks_next_output_tokens = [t.index_select(0, sort_order).type(torch.LongTensor) for t in toks_next_output_tokens]

        char_next_output_tokens = merge_tokens(
            'target',
            g_idx=1,
            left_pad=left_pad_target,
            bos=bos_idx,
            eos=None,
            move_trail=True,
        )
        char_next_output_tokens = [t.index_select(0, sort_order).type(torch.LongTensor) for t in char_next_output_tokens]
 
    assert len(toks_src_tokens) == len(toks_src_lengths)
    assert len(toks_target) == len(toks_tgt_lengths)
    assert len(toks_target) == len(toks_ntokens)
    assert len(toks_target) == len(toks_prev_output_tokens)
    assert len(toks_target) == len(toks_next_output_tokens)

    assert len(char_src_tokens) == len(char_src_lengths)
    assert len(char_target) == len(char_tgt_lengths)
    assert len(char_target) == len(char_ntokens)
    assert len(char_target) == len(char_prev_output_tokens)
    assert len(char_target) == len(char_next_output_tokens)
 
    toks_src_tokens = concat_with_sep(toks_src_tokens, sequence_separator)
    char_src_tokens = concat_with_sep(char_src_tokens, sequence_separator)
    toks_target = concat_with_sep(toks_target, sequence_separator)
    char_target = concat_with_sep(char_target, sequence_separator)
    toks_prev_output_tokens = concat_with_sep(toks_prev_output_tokens, sequence_separator)
    char_prev_output_tokens = concat_with_sep(char_prev_output_tokens, sequence_separator)
    toks_next_output_tokens = concat_with_sep(toks_next_output_tokens, sequence_separator)
    char_next_output_tokens = concat_with_sep(char_next_output_tokens, sequence_separator)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': [toks_ntokens, char_ntokens],
        'net_input': {
            'src_tokens': [toks_src_tokens, char_src_tokens],
            'src_lengths': [toks_src_lengths, char_src_lengths],
            'src_tok_bounds': src_tok_bounds,
            'sort_order': [sort_order, char_sort_order],
        },
        'target': [toks_target, char_target],
        'target_lengths' : [toks_tgt_lengths, char_tgt_lengths],
    }
    if toks_prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = [toks_prev_output_tokens, char_prev_output_tokens]
        batch['net_input']['tgt_tok_bounds'] = tgt_tok_bounds

    return batch


class TarcMultiTaskDataset(FairseqDataset):
    """
        Dataset for MultiTask training as conceived in the corpus TArC context.
        
        Args:
            src (torch.utils.data.Dataset): source dataset to wrap
            src_sizes (List[int]): source sentence lengths
            src_dict (~fairseq.data.Dictionary, optional): source vocabulary
            tgt (torch.utils.data.Dataset, optional): target dataset to wrap
            tgt_sizes (List[int], optional): target sentence lengths
            tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
            left_pad_target (bool, optional): pad target tensors on the left side
                (default: False).
            max_source_positions (int, optional): max number of tokens in the
                source sentence (default: 1024).
            max_target_positions (int, optional): max number of tokens in the
                target sentence (default: 1024).
            shuffle (bool, optional): shuffle dataset elements before batching
                (default: True).
            input_feeding (bool, optional): create a shifted version of the targets
                to be passed into the model for teacher forcing (default: True).
            append_eos_to_target (bool, optional): if set, appends eos to end of
                target if it's absent (default: False).
            append_bos (bool, optional): if set, appends bos to the beginning of
                source/target sentence.
        """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt, tgt_sizes, tgt_dict,
        sequence_separator,
        left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        append_eos_to_target=False,
        append_bos=False, eos=None,
        keep_data_order=False,
        granularity_flags=None,
    ):

        self.token_sequences = granularity_flags[0] if granularity_flags is not None else False
        self.char_sequences = granularity_flags[1] if granularity_flags is not None else False 

        self.src = src
        self.tgt = tgt 
        self.src_sizes = []
        for g_idx in range( len(src_sizes)-1 ):
            self.src_sizes.append([])
            for t_idx in range( len(src_sizes[g_idx]) ):
                self.src_sizes[g_idx].append( np.array( src_sizes[g_idx][t_idx] ) )
        self.src_sizes.append([])
        assert len(src_sizes[-1]) == 1
        self.src_sizes[-1].append( src_sizes[-1][0] ) 
        self.tgt_sizes = []
        for g_idx in range( len(tgt_sizes)-1 ):
            self.tgt_sizes.append([])
            for t_idx in range( len(tgt_sizes[g_idx]) ):
                self.tgt_sizes[g_idx].append( np.array( tgt_sizes[g_idx][t_idx] ) ) 
        self.tgt_sizes.append([])
        num_tasks = len( tgt_sizes[0] )
        for g_idx in range( len(tgt_sizes) ):
            assert len(tgt_sizes[g_idx]) == num_tasks
        for t_idx in range( len(tgt_sizes[-1]) ):
            self.tgt_sizes[-1].append( tgt_sizes[-1][t_idx] )

        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.sequence_separator = sequence_separator
        self.keep_data_order = keep_data_order
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.bos = tgt_dict.bos()
        self.eos = tgt_dict.eos()

    def __getitem__(self, index): 
        tgt_item = [ [l[index] for l in self.tgt[0]], [l[index] for l in self.tgt[1]] ]
        src_item = [ [l[index] for l in self.src[0]], [l[index] for l in self.src[1]] ]

        src_tok_bounds = [ l[index] for l in self.src_sizes[-1] ]
        tgt_tok_bounds = [ l[index] for l in self.tgt_sizes[-1] ] 

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos()
            if self.tgt[0][0][index][-1] != eos:
                tgt_item = [ [torch.cat(l[index], torch.LongTensor([eos])) for l in self.tgt[0]], [torch.cat(l[index], torch.LongTensor([eos])) for l in self.tgt[1]] ]

        if self.append_bos:
            bos = self.tgt_dict.bos()
            if self.tgt[0][0][index][0] != bos:
                tgt_item = [ [torch.cat(torch.LongTensor([bos]), l[index]) for l in self.tgt[0]], [torch.cat(torch.LongTensor([bos]), l[index]) for l in self.tgt[1]] ]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'src_tok_bounds': src_tok_bounds,
            'tgt_tok_bounds': tgt_tok_bounds
        }

        return example

    def __len__(self):
        return len(self.src[0][0])

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.
            
        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch with the following keys:
            
                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:
                
                    - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                    - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                    - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                    
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                target sentence of shape `(bsz, tgt_len)`. Padding will appear
                on the left if *left_pad_target* is ``True``.
        """

        return collate(
            samples, pad_idx=self.tgt_dict.pad(), bos_idx=self.bos, eos_idx=self.eos, sequence_separator=self.sequence_separator,
            left_pad_source=False, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            granularity_flags=(self.token_sequences, self.char_sequences),
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
         
        return max([t[index] for t in self.src_sizes[0]] + [t[index] for t in self.tgt_sizes[0]])

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
 
        return ( max([t[index] for t in self.src_sizes[0]]), max([t[index] for t in self.tgt_sizes[0]]) )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
            
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))

        if self.keep_data_order:
            print(' - TarcMultiTaskDataset: keeping original data order')
            sys.stdout.flush()
            return np.arange(len(self))
        else:
            if self.tgt_sizes[0][-1] is not None:
                indices = indices[np.argsort(self.tgt_sizes[0][-1][indices], kind='mergesort')]
            return indices[np.argsort(self.src_sizes[0][0][indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        flags = [getattr(l, 'supports_prefetch', False) for l in self.src[0]] + [getattr(l, 'supports_prefetch', False) for l in self.src[1]] + [getattr(l, 'supports_prefetch', False) for l in self.tgt[0]] + [getattr(l, 'supports_prefetch', False) for l in self.tgt[1]]
 
        return sum(flags) == len(flags)

    def prefetch(self, indices):
        [ [l.prefetch(indices) for l in src] for src in self.src ]
        [ [l.prefetch(indices) for l in tgt] for tgt in self.tgt ]





































