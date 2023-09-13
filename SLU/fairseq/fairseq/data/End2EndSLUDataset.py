

import sys
import logging

import numpy as np
import torch

import whisper
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, pad_or_trim, log_mel_spectrogram
from fairseq.globals import user_ID, machine_ID, bogus_ID

from . import data_utils, FairseqDataset

# Use this to show warnings (anything else ?) at the command line
logger = logging.getLogger(__name__)

def add_speaker_marker(feats, spk, spk_abs_val):
    """
        Add leading and trailing vectors as speaker marker for distinguishing input features from different speakers.
        When using the MEDIA corpus it is important to distinguish between the Machine, which is much easier to recognize, and a User.
    """

    #spk_abs_val = 2.0
    channel_per_speaker = 2
    (src_len, dim) = feats.size() 
    spk_val = 0.0
    if spk == user_ID:
        spk_val = +spk_abs_val  
    elif spk == machine_ID:
        spk_val = -spk_abs_val 
    elif spk == bogus_ID: 
        spk_val = -2.0*spk_abs_val 
    else:
        raise NotImplementedError

    padder = torch.zeros(3, dim).fill_(spk_val).to(feats)
    return torch.cat( [padder, feats, padder], 0 )

def collate_features(values, pad_idx, eos_idx=None, left_pad=False):
    """
    Convert a list of 2d tensors into a padded 3d tensor. 2d tensors are expected to be (length, dim)
    This function is intended to process speech input features, that is raw audio, spectrograms, etc.
    It thus does not make sense to pad or to add bos or eos symbols.
    """
    size = max(v.size(0) for v in values)
    dims = values[0].size()
    if len(dims) > 1:
        dim = values[0].size(1)
        res = torch.zeros(len(values), size, dim).to(values[0])
    else:
        dim = 1
        res = torch.zeros(len(values), size).to(values[0])
    
    def copy_tensor(src, dst):
        assert dst.size(0) == src.size(0)
        if len(dst.size()) > 1:
            assert dst.size(1) == src.size(1)
        dst.copy_(src)

    if len(dims) > 1:
        for i, v in enumerate(values):
            copy_tensor(v, res[i, size - v.size(0):,:] if left_pad else res[i, :v.size(0),:])
    else:
        for i, v in enumerate(values):
            copy_tensor(v, res[i, size - v.size(0):] if left_pad else res[i, :v.size(0)])
    return res

def collate_tokens_ex(values, pad_idx, bos_idx=None, eos_idx=None, left_pad=False, move_trail=False):
    """ Convert a list of 1d tensors into a padded 2d tensor.
        This is a generalization of the funcion in fairseq.data_utils which can either move eos to the beginning,
        or bos to the end (for backward decoders)
    """

    assert not ((bos_idx is not None) and (eos_idx is not None)), 'collate_tokens_ex: either bos index or eos index must be not None, got both None'

    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()

        #print('[DEBUG] copy_tensor:')
        #print('[DEBUG]  * src: {}'.format(src))
        #print('[DEBUG]  * bos_idx, eos_idx, move_trail: {}, {}, {}'.format(bos_idx, eos_idx, move_trail))
        #print(' ----------')
        #sys.stdout.flush()

        if move_trail:
            if bos_idx is not None:
                assert src[0] == bos_idx
                dst[-1] = bos_idx
                dst[:-1] = src[1:]
            else:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def collate(
            samples, pad_idx, bos_idx, eos_idx, left_pad_source=False, left_pad_target=False,
            input_feeding=True, tgt_dict=None,
):

    if len(samples) == 0:
        return {}

    def merge_features(key, left_pad):
        return collate_features(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad,
        )

    def merge_features_ex(key, left_pad, bos=None, eos=None, move_trail=False, tgt_dict=None):
        src_sig = collate_features([s[key][0] for s in samples], pad_idx, eos_idx, left_pad) 
        src_trs = collate_tokens_ex([s[key][1] for s in samples], pad_idx, bos_idx=bos, eos_idx=eos, left_pad=left_pad, move_trail=move_trail) 
        return (src_sig, src_trs)

    def merge_tokens(key, left_pad, bos=None, eos=None, move_trail=False):
        return collate_tokens_ex(
            [s[key] for s in samples],
            pad_idx, bos_idx=bos, eos_idx=eos, left_pad=left_pad, move_trail=move_trail,
        )

    '''print(' *** collate sample shape:')
    for s in samples:
        print(' - source shape: {}'.format(s['source'].size()))
        print(' - target shape: {}'.format(s['target'].size()))
        print(' ---')
    print(' *****')
    sys.stdout.flush()'''

    use_trs_flag = isinstance(samples[0]['source'], tuple)

    id = torch.LongTensor([s['id'] for s in samples])
    strid = [s['strid'] for s in samples]
    src_signals, src_tokens = None, None
    if use_trs_flag:

        #print(' * collate, merging features and transcriptions')
        
        src_signals, src_tokens = merge_features_ex('source', left_pad=left_pad_source, tgt_dict=tgt_dict) 
    else:
        #print(' * collate, merging features')

        src_tokens = merge_features('source', left_pad=left_pad_source)

    #print(' * collate, merged features: {}'.format(src_tokens.size()))
    #sys.stdout.flush()

    # sort by descending source length
    src_lengths, src_tok_lengths = None, None
    if use_trs_flag:
        src_lengths = torch.LongTensor( [s['source'][0].size(0) for s in samples] )
        src_tok_lengths = torch.LongTensor( [s['source'][1].size(0) for s in samples] )
        #src_lengths = (src_sig_lengths, src_trs_lengths)
    else:
        src_lengths = torch.LongTensor([s['source'].size(0) for s in samples])
    #src_lengths, sort_order = src_lengths.sort(descending=True)
    #id = id.index_select(0, sort_order)
    #src_tokens = src_tokens.index_select(0, sort_order)

    target = merge_tokens('target', left_pad=left_pad_target)
    #target = target.index_select(0, sort_order).type(torch.LongTensor)
    tgt_lengths = torch.LongTensor([s['target'].size(0) for s in samples])
    #tgt_lengths = tgt_lengths.index_select(0, sort_order)
    ntokens = sum(len(s['target']) for s in samples)

    prev_output_tokens = None
    next_output_tokens = None
    if input_feeding:
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge_tokens(
            'target',
            left_pad=left_pad_target,
            bos=None,
            eos=eos_idx,
            move_trail=True,
        )
        #prev_output_tokens = prev_output_tokens.index_select(0, sort_order).type(torch.LongTensor)

        # This is created but not used for now...
        next_output_tokens = merge_tokens(
            'target',
            left_pad=left_pad_target,
            bos=bos_idx,
            eos=None,
            move_trail=True,
        )
        #next_output_tokens = next_output_tokens.index_select(0, sort_order).type(torch.LongTensor)

    batch = {
        'id': id,
        'strid': strid,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_signals': src_signals,
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_tok_lengths': src_tok_lengths,
        },
        'target': target,
        'target_lengths' : tgt_lengths,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    #if next_output_tokens is not None:
    #    batch['net_input']['next_output_tokens'] = next_output_tokens

    '''print(' *** collate debug:')
    print('   * target shape: {}'.format(target.size()))
    print('   * target_lengths: {}'.format(tgt_lengths))
    print('   * prev_output_tokens (prevot) shape: {}'.format(prev_output_tokens.size()))
    print('')
    print('   * target: {}'.format(target))
    print(' -----')
    print('   * prevot: {}'.format(prev_output_tokens))
    print(' *****')
    sys.exit(0)'''

    return batch


class End2EndSLUDataset(FairseqDataset):
    """
        A pair of torch.utils.data.Datasets. First containing feature tensors (e.g. wav signals, spectrograms, wav2vec features, etc.), second containing the desired output (e.g. characters, tokens, concepts, etc.)
        
        Args:
            src (torch.utils.data.Dataset): source dataset to wrap
            src_sizes (List[int]): source sentence lengths
            tgt (torch.utils.data.Dataset, optional): target dataset to wrap
            tgt_sizes (List[int], optional): target sentence lengths
            tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
            idx_structure (torch.utils.data.Dataset): indexes of sources for training preserving sources structures (e.g. dialogs, documents, etc.)
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
        self, args, split, src, src_sizes,
        tgt, tgt_sizes, tgt_dict,
        idx_structure,
        left_pad_target=False,
        max_source_positions=15000000, max_target_positions=10000,
        shuffle=True, input_feeding=True,
        append_eos_to_target=False,
        append_bos=False, eos=None,
        RoBERTa=None,
        feature_extractor=None,
        turn_ids=None,
        turn_spk=None,
        spk_abs_val=1.0,
    ):

        self.data_split_id = split
        self.args = args
        self.ids = turn_ids
        self.spk = turn_spk
        self.spk_abs_val = spk_abs_val
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        if RoBERTa is not None:
            self.tgt_dict = RoBERTa.task.source_dictionary
        else:
            self.tgt_dict = tgt_dict 
        if self.args.online_feature_extraction:
            assert feature_extractor is not None
            assert turn_spk is not None
            if self.args.feature_extractor == 'whisper':
                self.model = feature_extractor #whisper.load_model( self.args.fe_size )
                self.model = self.model.cuda()
            else:
                raise NotImplementedError()
        #self.idx_batches = idx_structure

        self.idx_spk_batches = idx_structure
        self.idx_batches = []
        #print('[DEBUG] End2ENDSLUDataset, type of idx_structure: {} (size: {})'.format(type(idx_structure), len(idx_structure)))
        #sys.stdout.flush()
        for dialog_batch in idx_structure:

            #print('[DEBUG] End2EndSLUDataset, type of elements in idx_structure: {} (size: {})'.format(type(dialog_batch), len(dialog_batch)))
            #sys.stdout.flush()

            if args.dialog_level_slu:
                tmp_d = []
                for turn_batch in dialog_batch:

                    #print('[DEBUG] End2EndSLUDataset, type of elements of elements: {} (size: {})'.format(type(turn_batch), len(turn_batch)))
                    #print('[DEBUG] End2EndSLUDataset, type of deepest elements: {}'.format(type(turn_batch[0])))
                    #sys.stdout.flush()

                    tmp_b = []
                    for turn_info in turn_batch:
                        tmp_b.append( turn_info[0] )
                    tmp_d.append( tmp_b )
                self.idx_batches.append( tmp_d )

                '''print('[DEBUG]### End2EndSLUdataset, reorganizing batches @split {}'.format(self.data_split_id))
                print('[DEBUG]###     * type of self.idx_batches: {} (size: {})'.format(type(self.idx_batches), len(self.idx_batches)))
                print('[DEBUG]###     * type of elements: {} (length: {})'.format(type(self.idx_batches[-1]), len(self.idx_batches[-1])))
                print('[DEBUG]###     * type of elements of elements: {} (length: {})'.format(type(self.idx_batches[-1][0]), len(self.idx_batches[-1][0])))
                print('[DEBUG]###     * type of deepest elements: {}'.format(type(self.idx_batches[-1][0][0])))''' 
            else:
                self.idx_batches.append( [t[0] for t in dialog_batch] )

        '''if args.dialog_level_slu:
            assert len(self.idx_batches) == len(idx_structure)
            for didx in range(len(idx_structure)):
                assert len(idx_structure[didx]) == len(self.idx_batches[didx])
                for tidx in range(len(idx_structure[didx])):
                    assert len(idx_structure[didx][tidx]) == len(self.idx_batches[didx][tidx])
                    for idx in range(len(idx_structure[didx][tidx])):
                        assert idx_structure[didx][tidx][idx][0] == self.idx_batches[didx][tidx][idx]

            mid_idx = int(len(self.idx_batches)/2)
            print('[DEBUG] End2EndSLUDataset, *** Visualizing a whole dialog:')
            dialog_batch = self.idx_batches[mid_idx]
            dialog_idx = 0
            for tidx in range(len(dialog_batch)):
                print('[DEBUG] {}) {}'.format(tidx, self.tgt_dict.string(self.tgt[dialog_batch[tidx][dialog_idx]])))
                sys.stdout.flush()
            print('[DEBUG] ---------------')
            sys.stdout.flush()
        if args.dialog_batches:
            mid_idx = int(len(self.idx_batches)/2)
            print('[DEBUG] End2EndSLUDataset, *** Visualizing a whole dialog @initialization:')
            dialog_batch = self.idx_batches[mid_idx]
            for i, idx in enumerate(dialog_batch):
                print('[DEBUG] {} -> {}) {}'.format(i, idx, self.tgt_dict.string(self.tgt[dialog_batch[i]])))
            print('[DEBUG] ----------')
            sys.stdout.flush()'''

        # Curriculum solution 1: sort all turns by length, without regard to the speaker
        lengths = [(i, t[0].size(0)) for i, t in enumerate(src)] if isinstance(src[0], tuple) else [(i, t.size(0)) for i,t in enumerate(src)]
        sorted_structure = sorted(lengths, key=lambda tuple: tuple[1])
        self.curriculum_indices = [t[0] for t in sorted_structure]

        # Curriculum solution 2: sort turns by length separating machine and user turns, machine turns are put first as they are all from the same speaker and thus simpler. 
        '''m_idx = []
        u_idx = []
        for s in idx_structure:
            m_idx.extend( [t[0] for t in s if t[1] == 'Machine'] )
            u_idx.extend( [t[0] for t in s if t[1] == 'User'] )

        #u_idx = [t[0] for t in self.idx_spk_batches if t[1] == 'User']
        m_lengths = [(i, src[i].size(0)) for i in m_idx]
        u_lengths = [(i, src[i].size(0)) for i in u_idx] 
        m_sorted_structure = sorted(m_lengths, key=lambda tuple: tuple[1])
        m_sorted_idx = [t[0] for t in m_sorted_structure]
        u_sorted_structure = sorted(u_lengths, key=lambda tuple: tuple[1])
        u_sorted_idx = [t[0] for t in u_sorted_structure]
        self.curriculum_indices = m_sorted_idx + u_sorted_idx'''

        assert len(src) == len(self.curriculum_indices)

        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.append_eos_to_target = append_eos_to_target
        self.append_bos = append_bos
        self.bos = self.tgt_dict.bos()
        self.eos = self.tgt_dict.eos() #(eos if eos is not None else tgt_dict.eos()) 

        # For DEBUGGING
        '''self.epochs_id_order = []
        self.id_order = []'''

    def curriculum(self, value=False):
        self.shuffle = not value

    def set_split_id(self, value='train'):
        print('[DEBUG] End2EndSLUDataset, data split id set to: {}'.format(value))
        sys.stdout.flush()

        self.data_split_id = value
        return self.data_split_id

    def get_split_id(self):
        return self.data_split_id

    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]

        if self.args.online_feature_extraction:

            if self.args.feature_extractor == 'whisper':
                mel = log_mel_spectrogram(src_item.squeeze()) 
                dtype = torch.float32 
                decode_options = {"fp16": False}
                segment = pad_or_trim(mel, N_FRAMES).to(self.model.device).to(dtype)
                single = segment.ndim == 2
                if single:
                    segment = segment.unsqueeze(0) 

                if segment.shape[-2:] != (self.model.dims.n_audio_ctx, self.model.dims.n_audio_state):
                    with torch.no_grad():
                        src_item = self.model.encoder(segment)
                        src_item = src_item.squeeze()
                        if len(src_item.size()) < 2:
                            src_item = src_item.unsqueeze(0)
                else:
                    raise ValueError('encoder is expected to be always called here!')
            else:
                raise NotImplementedError()

            src_item = add_speaker_marker(src_item, self.spk[index], self.spk_abs_val)

        #print('[DEBUG] __getitem__, sending input sequence of shape: {}'.format(src_item.size()))
        #sys.exit(0)

        #print('[DEBUG] End2EndSLUDataset.getitem: demanded index {}'.format(index))
        #sys.stdout.flush()

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos()
            if self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos()
            if self.tgt[index][0] != bos:
                tgt_item = torch.cat([torch.LongTensor([bos]), self.tgt[index]])

        # For DEBUGGING
        '''if index in self.id_order:
            if len(self.epochs_id_order) > 0:
                equal = True
                for idx in range(len(self.epochs_id_order[-1])):
                    if self.epochs_id_order[-1][idx] != self.id_order[idx]:
                        equal = False
                        break
                if equal:
                    print(' *** End2EndSLUDataset: data order is the same between last two epochs')
                    sys.stdout.flush()
                else:
                    print(' *** End2EndSLUDataset: data order changed in the last two epochs')
                    sys.stdout.flush()
            self.epochs_id_order.append( [el for el in self.id_order] )
            self.id_order = []
        self.id_order.append(index)'''
 
        #print(' * __getitem__: src_item sizes: {}, {}'.format(src_item[0].size(), src_item[1].size()))
        #sys.stdout.flush()

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'strid': self.ids[index],
        }

        return example

    def __len__(self):
        return len(self.src)

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
            samples, pad_idx=self.tgt_dict.pad(), bos_idx=self.bos, eos_idx=self.eos,
            left_pad_source=False, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, tgt_dict=self.tgt_dict,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        if len(self.src_sizes.shape) == 2:
            return max(self.src_sizes[index,0], self.tgt_sizes[index])
        else:
            return max(self.src_sizes[index], self.tgt_sizes[index])

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if len(self.src_sizes.shape) == 2:
            return (self.src_sizes[index,0], self.tgt_sizes[index])
        else:
            return (self.src_sizes[index], self.tgt_sizes[index])

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
            
        if self.shuffle:
            batch_shuffle_idx = np.random.permutation(len(self.idx_batches))
            #[id for sid in shuffle for id in lists[sid]]
            if not self.args.dialog_level_slu and not self.args.dialog_batches:
                indices = np.array([idx for sidx in batch_shuffle_idx for idx in self.idx_batches[sidx]])
            elif self.args.dialog_batches:
                indices = []
                for idx in batch_shuffle_idx:
                    indices.append( self.idx_batches[idx] )

                '''print('[DEBUG] End2EndSLUDataset, using dialog-batches shuffled indices:')
                print('[DEBUG]     * type of indices: {} (size: {})'.format(type(indices), len(indices)))
                print('[DEBUG]     * type of elements of indices: {} (size: {})'.format(type(indices[0]), len(indices[0])))
                print('[DEBUG]   *** Visualizing a whole dialog in ordered_indices:')
                mid_idx = int(len(indices)/2)
                dialog_batch = indices[mid_idx]
                for i, idx in enumerate(dialog_batch):
                    print('[DEBUG] {} -> {}) {}'.format(i, idx, self.tgt_dict.string(self.tgt[dialog_batch[i]])))
                print('[DEBUG] ----------')
                sys.stdout.flush()'''
            else:
                indices = []
                for idx in batch_shuffle_idx:
                    indices.append( self.idx_batches[idx] ) # NOTE: this allows to keep original dialog structures, so this should allow to perform dialog-level SLU when --dialog-level-slu is used

                for el in indices:
                    assert type(el) == list
                    for b in el:
                        assert type(b) == list
                        for tt in b:
                            assert type(tt) != list and type(tt) != tuple

                print('[DEBUG] End2EndSLUDataset, using shuffled indices @split {}'.format(self.data_split_id))
                '''print('[DEBUG]     *** Visualizing a whole dialog in ordered_indices:')
                mid_idx = int(len(indices)/2)
                dialog_batch = indices[mid_idx]
                dialog_idx = 0
                for tidx in range(len(dialog_batch)):
                    print('[DEBUG] {}) {}'.format(tidx, self.tgt_dict.string(self.tgt[dialog_batch[tidx][dialog_idx]])))
                print(' ----------')'''
                sys.stdout.flush()

            #indices = np.random.permutation(len(self)) 
        else:
            #indices = np.arange(len(self))
            if self.args.curriculum > 0:
                indices = np.array(self.curriculum_indices)
            else:
                indices = self.idx_batches

            print('[DEBUG] End2EndSLUDataset, using curriculum indices @split {}'.format(self.data_split_id))
            sys.stdout.flush()

            return indices

        if not self.args.dialog_level_slu and not self.args.dialog_batches:
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
            if len(self.src_sizes.shape) == 2:
                return indices[np.argsort(self.src_sizes[indices,0], kind='mergesort')]
            else:
                return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            return indices

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False))
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        self.tgt.prefetch(indices)





































