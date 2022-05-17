# -*- coding: utf-8 -*-

# Code for adding the End2End SLU task into fairseq

import os
import re
import sys
import torch
import math

from fairseq.globals import *
from fairseq.data import Dictionary, End2EndSLUDataset
from fairseq.tasks import FairseqTask, register_task

import examples.speech_recognition.criterions.SLU_CTC_loss as slu_ctc

#from carbontracker.tracker import CarbonTracker
from codecarbon import EmissionsTracker

_ADVANCED_SPK_MARK_ = True

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

def create_batch_(data, curr_batch_ids):
    
    dialog_len = len( data[curr_batch_ids[0]] ) # all dialogs in this batch have this same length
    dialog_batch = []
    for i in range( dialog_len ):
        for d_idx in range(len(curr_batch_ids)):

            dialog_batch.append( (curr_batch_ids[d_idx], i) )

    return dialog_batch

def get_bogus_turn(real_turn, dd, turn_id=None):
    # EOS_tag, User_ID
    T, C = real_turn[1].size()
    sig = torch.zeros(1, C).to(real_turn[1])
    tt = torch.LongTensor( [dd.eos()] )
    bogus_id = turn_id if turn_id is not None else real_turn[0]
    return (bogus_id, sig, tt, tt, tt, bogus_ID)

def rearrange_for_dialog_level_slu_(data, bs, dlens, dd):

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

    # Now add bogus turns to dialogues so that to have: 1. all dialogues in the same batch of the same length; 2. num of dialogues multiple of batch-size (this is accomplished by creating whole bogus dialogues! Hope this work...) 
    for bi, bb in enumerate(batches):
        max_len = batch2len[bi]

        tmp = max([did2len[did] for did in bb])
        assert tmp == max_len

        for did in bb:
            if did2len[did] < max_len:
                dialog = data[did]
                for ii in range(did2len[did]+1, max_len+1):
                    dialog.append( get_bogus_turn(dialog[0], dd, ii) )
                #data[did] = dialog # needed ???

    # NOTE: last batch may contain less dialogues than bs, so possibly whole bogus dialogues must be added to have a batch of the same size as the others.
    if len(batches[-1]) != bs:
        did = 9900
        max_len = batch2len[-1]

        tmp = max( [did2len[did] for did in batches[-1]] )
        assert tmp == max_len

        for ii in range(bs-len(batches[-1])):
            bogus_did = str(did+ii+1)
            batches[-1].append(bogus_did) 
            data[bogus_did] = []
            did2len[bogus_did] = 0

        assert len(batches[-1]) == bs

        for did in batches[-1]:
            if did2len[did] < max_len: # actually only the bogus dialogues should satisfy this condition !
                assert did2len[did] == 0

                dialog = data[did]
                for ii in range(did2len[did]+1, max_len+1):
                    dialog.append( get_bogus_turn(batches[-1][0], dd, ii) )
                #data[did] = dialog # needed ???

    # TMP DEBUG CHECK
    assert len(data) % bs == 0

    for ii, bb in enumerate(batches):
        dlen = batch2len[ii]
        plen = len(data[bb[0]])
        found = False
        for did in bb:
            assert len(data[did]) == dlen
            assert len(data[did]) == plen

            if torch.sum(data[did][-1][1]).item() != 0.0:
                found = True
        if not found:
            raise ValueError('Found a batch with all zero-turns as last')
    # END TMP DEBUG CHECK

    # TODO: add here an end of dialog marker in each and every dialog so that the model can detect when to re-initialize the dialog history cache
    for did in data:
        data[did].append( get_bogus_turn(data[did][0], dd, len(data[did])+1) )

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
            if torch.sum(data[did][-1][1]) != 0.0:
                found = True
        if found:
            raise ValueError('Found a batch without End-of-dialogue marker')
    # END TMP DEBUG CHECK

    return dialog_lengths

def create_dialog_batches_(data, batch_size, infos):

    dd = infos['dict']
    dlevel_slu = infos['dialog_level_slu']

    dialog_lengths = {}
    for dialog_id in data.keys():
        curr_len = len(data[dialog_id])
        if not curr_len in dialog_lengths:
            dialog_lengths[curr_len] = []
        dialog_lengths[curr_len].append( dialog_id )

    if dlevel_slu:
        raise NotImplementedError()

        dialog_lengths = rearrange_for_dialog_level_slu_(data, batch_size, dialog_lengths, dd)

    dialog_batches = []
    for dlen in dialog_lengths.keys():
        
        if len(dialog_lengths[dlen]) > 1:
            
            num_batches = int(len(dialog_lengths[dlen]) / batch_size)
            remainder = len(dialog_lengths[dlen]) % batch_size

            if dlevel_slu:
                assert remainder == 0
            
            b_idx = 0
            while b_idx < num_batches:
                curr_batch_ids = dialog_lengths[dlen][b_idx*batch_size:(b_idx+1)*batch_size]
                dialog_batches.append( create_batch_(data, curr_batch_ids) )
                b_idx += 1
            
            if remainder > 0:
                curr_batch_ids = dialog_lengths[dlen][num_batches*batch_size:]
                dialog_batches.append( create_batch_(data, curr_batch_ids) )
        else:
            if dlevel_slu:
                raise ValueError()

            curr_batch_ids = dialog_lengths[dlen]
            dialog_batches.append( create_batch_(data, curr_batch_ids) )

    return dialog_batches

def read_txt(txtpath):
 
    f = open(txtpath, 'rb')
    lines = [l.decode('utf-8','ignore') for l in f.readlines()]
    for line in lines:
        dialog_transcript = line
    f.close()
    return dialog_transcript.strip()

def parse_media_semantic_annotation(filename):
    
    # Example:
    # 364_16 @null{euh} @+rang-temps{la troisième} @+temps-unite{semaine} @+temps-mois{d' août}
    
    clean_re = re.compile('^\s+|\s+$')
    p = re.compile('^(\+|\-)?(\S+)\{([^\}]+)\}')
    
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
            if len(c) > 0 and c != 'null{} ' and c != 'null{}':
                m = p.match(c)
                if m == None:
                    sys.stderr.write(' - ERROR: parse_media_semantic_annotation parsing error at {} while parsing file {}\n'.format(c, filename))
                    sys.exit(1)
                else:
                    (mode,concept,surface) = (m.group(1),m.group(2),clean_re.sub('',m.group(3)))
                    concept_list.append( (mode,concept,surface) )
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

    components = turn.split('/')    # e.g. .../2BqVo8kVB2Skwgyb/029f6450-447a-11e9-a9a5-5dbec3b8816a
    dialog_id = components[-2]
    turn_id = components[-1]

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

    if windowtime < 0:
        srate_str = '16kHz'
        if windowtime == -2:
            srate_str = '8kHz'
        if windowtime > -3:
            print(' * read_dialog_data: reading {} wave2vec features...'.format(srate_str))
        elif windowtime == -3:
            print(' * read_dialog_data: reading FlowBERT features ({} files)...'.format(feat_ext))
        elif windowtime == -4:
            print(' * read_dialog_data: reading W2V2 base features ({} files)...'.format(feat_ext))
        elif windowtime == -5:
            print(' * read_dialog_data: reading FlowBERT base features ({} files)...'.format(feat_ext))
        elif windowtime == -6:
            print(' * read_dialog_data: reading W2V2 large features ({} files)...'.format(feat_ext))
        elif windowtime == -7:
            print(' * read_dialog_data: reading XLSR53 large features ({} files)...'.format(feat_ext))
        else:
            raise NotImplementedError()
        sys.stdout.flush()
    else:
        print(' * read_dialog_data: reading {} feature files'.format(feat_ext))
        sys.stdout.flush()

    dialog_data = {}
    with open(TurnList) as f:
        turns = f.readlines()
        f.close()

    for turn in turns:
        dialog_id, turn_id = parse_filename( args, turn ) 

        if dialog_id not in dialog_data:
            dialog_data[dialog_id] = [] 

        # 1. Turn ID for sorting turns of the same dialog
        turn_data = []
        turn_data.append( turn_id )

        # 2. Spectrogram of the turn audio signal (the input to the network)
        if windowtime == 20:
            #file_ext = '.8kHz.20.0ms-spg'
            #file_ext = '.8kHz.20.0ms-mfcc'
            #if args.corpus_name == 'fsc':
            #    file_ext = '.16kHz' + file_ext
            spg_tsr = torch.load(turn.strip() + feat_ext)
        elif windowtime == 10:
            raise NotImplementedError
            #spg_tsr = torch.load(turn.strip() + '.10.0ms-spg')
        elif windowtime < 0:
            srate_str = '16kHz'
            if windowtime == -2:
                srate_str = '8kHz'
            lan=''
            if lan_flag == 'Fr':
                lan='-Fr'
            if windowtime > -3:
                spg_tsr = torch.load(turn.strip() + '.' + srate_str + lan + '.w2v')
            else:
                spg_tsr = torch.load(turn.strip() + feat_ext)
            '''elif windowtime == -3:
                spg_tsr = torch.load(turn.strip() + '.fbert')
            elif windowtime == -4: 
                spg_tsr = torch.load(turn.strip() + '.ebert')
            elif windowtime == -5:
                spg_tsr = torch.load(turn.strip() + '.bbert')
            elif windowtime == -6:
                spg_tsr = torch.load(turn.strip() + '.bebert')
            elif windowtime == -7:
                spg_tsr = torch.load(turn.strip() + '.xbert')'''
            if reduce_flag > 1:
                raise NotImplementedError
                #spg_tsr = Sf.reduce( spg_tsr.squeeze(), reduce_flag )

        if windowtime > -3:
            spg_tsr = spg_tsr.squeeze().permute(1,0)
        else:
            spg_tsr = spg_tsr.squeeze() 
        if len(spg_tsr.size()) != 2:
            spg_tsr = spg_tsr.unsqueeze(0)

        #print(' - read_dialog_data, read spectrogram shape: {}'.format(spg_tsr.size()))
        #sys.stdout.flush()
        #sys.exit(0)

        if spg_tsr.size(0) < 3:
            print(' *** read_dialog_data: got strangely short signal {}...'.format(spg_tsr.size()))
            sys.stdout.flush()

        if spg_tsr.size(0) > 3 and spg_tsr.size(0) < args.max_source_positions:
            turn_data.append( spg_tsr )  # Spectrogram's shape is (1 x num_features x sequence_length), we make it (sequence_length x num_features) 

            # 3. The reference transcription 
            turn_txt = read_txt(turn.strip() + '.txt')
            turn_data.append( turn_txt )

            # 4. The transcription "enriched" with semantic annotation
            basename = turn.split('/')[-1]
            if args.corpus_name in ['media', 'etape']:
                if user_ID in basename:
                    dialog_struct = parse_media_semantic_annotation( turn.strip() + '.sem' )
                    slu_turn = ''
                    for vals in dialog_struct.values():
                        for turn_struct in vals: 
                            for c in turn_struct['Concepts']:
                                if len(slu_turn) > 0:
                                    slu_turn = slu_turn + ' '
                                slu_turn = slu_turn + slu_start_concept_mark + ' ' + c[2] + ' ' + c[1] + ' ' + slu_end_concept_mark
                    turn_data.append( slu_turn )
                    turn_data.append( user_ID )
                else:
                    slu_turn = slu_start_concept_mark + ' ' + turn_txt + ' ' + machine_semantic + ' ' + slu_end_concept_mark
                    turn_data.append( slu_turn )
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
                turn_data.append( user_ID )
            else:
                raise NotImplementedError 

            dialog_data[dialog_id].append( turn_data )

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

def references2indexes(args, data, vocab):

    for did in data.keys():
        for idx in range(len(data[did])):
            (turn_id, spg, txt_turn, slu_turn, spk_ID) = data[did][idx]
            char_list = [vocab.add_symbol(c) for c in txt_turn]
            token_list = [vocab.add_symbol(t) for t in txt_turn.split()]
            slu_list = [vocab.add_symbol(s) for s in slu_turn.split()]

            if args.padded_reference:
                char_list = [vocab.bos()] + char_list + [vocab.eos()]
                token_list = [vocab.bos()] + token_list + [vocab.eos()]
                slu_list = [vocab.bos()] + slu_list + [vocab.eos()]

            char_tsr = torch.LongTensor( char_list )
            token_tsr = torch.LongTensor( token_list )
            slu_tsr = torch.LongTensor( slu_list )

            data[did][idx] = (turn_id, spg, char_tsr, token_tsr, slu_tsr, spk_ID)


def feature_wise_mean_std(data):
    (sequence_length, num_features) = list(data.values())[0][0][1].size()
    data_mu = torch.zeros( num_features )
    data_sigma = torch.zeros( num_features )

    with torch.no_grad():
        total_features = 0
        for dialog_id in data.keys():
            for t in data[dialog_id]:
                data_mu = data_mu + torch.sum( t[1], 0 )
                total_features += t[1].size(0)
        data_mu = data_mu / float( total_features )

        for dialog_id in data.keys():
            for t in data[dialog_id]:
                data_sigma = data_sigma + torch.sum( (t[1] - data_mu)**2, 0 )
        data_sigma = torch.sqrt( data_sigma / float(total_features-1) )

    return (data_mu, data_sigma)

def normalize_data_only(data, mu, sigma):
    with torch.no_grad():
        for dialog_id in data.keys():
            for t in data[dialog_id]:
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
        parser.add_argument('--source-dictionary', type=str,
                            help='When performing transfer learning, uses this dictionary as source domain dictionary')
        parser.add_argument('--target-dictionary', type=str,
                            help='When performing transfer learning, uses this dictionary as target domain dictionary')
        parser.add_argument('--max-source-positions', default=10000, type=int,
                            help='max input length')
        parser.add_argument('--max-target-positions', default=1600, type=int,
                            help='max input length')
        parser.add_argument('--slu-subtask', default='char', type=str,
                            help='which subtask has to be modeled (e.g. char, token, concept)')
        parser.add_argument('--user-only', action='store_true',
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
        parser.add_argument('--dialog-level-slu', action='store_true', default=False, help='Re-arrange turns and dialogues to perform dialog-level SLU')
        parser.add_argument('--speech-encoder', type=str, default='ziggurat', help='Structure of speech encoder: ziggurat (NeurIPS 2021 paper), deep-speech (ICASSP 2020 paper)')

    @classmethod
    def setup_task(cls, args, **kwargs):
        
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just initialize the label dictionary
        
        label_vocab = SLUDictionary(
            pad=pad_token,
            eos=EOS_tag,
            unk=unk_token,
            bos=SOS_tag,
            extra_special_symbols=[blank_token, machine_semantic, slu_start_concept_mark, slu_end_concept_mark, tok_separator, 'ã']
        )
        # NOTE: we are ogliged to initialize the Dictionary as it is requested in Fairseq,
        #       but then we reset it and re-add symbols so that to have index 0 for the blank token, as needed by the CTC loss.
        #       (Added later: the index of the blank token to be passed to the CTC loss can be actually customized)
        label_vocab.reset() 
        label_vocab.set_blank(blank_token)
        label_vocab.set_pad_(pad_token)
        label_vocab.set_eos_(EOS_tag)
        label_vocab.set_unk_(unk_token)
        label_vocab.set_bos_(SOS_tag)
 
        label_vocab.add_symbol(machine_semantic)
        label_vocab.add_symbol(slu_start_concept_mark)
        label_vocab.add_symbol(slu_end_concept_mark)
        label_vocab.add_symbol(tok_separator)
        label_vocab.add_symbol('ã')
        label_vocab.set_nspecial_()  # We set the tokens added so far as being special tokens.

        if args.load_dictionary:
            print(' - Loading serialized dictionary...')
            sys.stdout.flush()
            
            label_vocab = torch.load(args.load_dictionary)

        print('| [label] dictionary: {} types'.format(len(label_vocab)))
        
        return End2EndSLU(args, label_vocab)

    def __init__(self, args, label_vocab):
        
        super().__init__(args)

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

        self.label_vocab = label_vocab
        self.num_features = 81

        self.blank_idx = label_vocab.index( blank_token )
        self.sos_tag_idx = label_vocab.index( SOS_tag )
        self.eos_tag_idx = label_vocab.index( EOS_tag )
        self.slu_start_concept_idx = label_vocab.index( slu_start_concept_mark )
        self.slu_end_concept_idx = label_vocab.index( slu_end_concept_mark )

        '''print(' ***** End2EndSLU task initialization...')
        print(' *** SLU task: blank-idx {}'.format(self.blank_idx))
        print(' *** SLU task: sos_tag_idx {}'.format(self.sos_tag_idx))
        print(' *** SLU task: eos_tag_idx {}'.format(self.eos_tag_idx))
        print(' *** SLU task: slu_start_concept_idx {}'.format(self.slu_start_concept_idx))
        print(' *** SLU task: slu_end_concept_idx {}'.format(self.slu_end_concept_idx))
        print(' *** SLU task: pad index {}'.format(self.label_vocab.index(pad_token)))
        sys.stdout.flush()'''

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
        if self.curr_epoch <= 3:
            self.datasets['train'].curriculum(value=True)
            #print(' End2EndSLU, setting curriculum learning to True')
            #sys.stdout.flush()
        if self.curr_epoch > 3:
            self.datasets['train'].curriculum(value=False)
            #print(' End2EndSLU, setting curriculum learning to False')
            #sys.stdout.flush() 

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
        (src_len, dim) = feats.size()
        if _ADVANCED_SPK_MARK_:
            speaker_mark = torch.zeros(src_len, channel_per_speaker*self.num_speakers).to(feats)
        spk_val = 0.0
        if spk == user_ID:
            spk_val = +self.spk_abs_val
            if _ADVANCED_SPK_MARK_:
                speaker_mark[:,:channel_per_speaker] = spk_val
        elif spk == machine_ID:
            spk_val = -self.spk_abs_val
            if _ADVANCED_SPK_MARK_:
                speaker_mark[:,channel_per_speaker:2*channel_per_speaker] = spk_val
        elif spk == bogus_ID: 
            spk_val = -2.0*self.spk_abs_val
            if _ADVANCED_SPK_MARK_:
                speaker_mark[:,2*channel_per_speaker:] = spk_val
        else:
            raise NotImplementedError

        if _ADVANCED_SPK_MARK_:
            feats = torch.cat([feats, speaker_mark], 1)
            dim = dim + channel_per_speaker*self.num_speakers
        padder = torch.zeros(3, dim).fill_(spk_val).to(feats)
        return torch.cat( [padder, feats, padder], 0 )

        #spk_mark = torch.zeros_like(feats).fill_(spk_val)
        #return feats + spk_mark

    def pad_short_sequences(self, corpus, split):

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
        in_red_factor = (self.args.num_lstm_layers-1)*2 if hasattr(self.args, 'num_lstm_layers') else 2

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
                    if t[1].size(0)//in_red_factor < slu_t.size(0): 
                        length_missmatch += 1
                        missmatches.append( (t[1].size(0)//in_red_factor, slu_t.size(0)) )

                    turn_tuple = (t[0], t[1], t[2], t[3], slu_t, t[5])
                    corpus[split][did][t_idx] = turn_tuple
                else:
                    if t[comp_t_idx].size(0) > max_turn_length:
                        max_turn_length = t[comp_t_idx].size(0)

                    if t[1].size(0)//in_red_factor < t[comp_t_idx].size(0):
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

        '''if split == 'train' and len(missmatches) > 0:
            print(' * End2EndSLU: padding sequences shorter than {}, stretching by a factor {}'.format(self.args.max_padding_len, self.args.max_padding_ratio))
            sys.stdout.flush()
            self.args.padding_active = True'''

        # 2. Second scan, pad short input sequences.
        padded_sequences = 0
        if split == 'train' and not (self.args.decoder == 'ctclstm' and not self.args.load_fairseq_encoder): # NOTE: we pad only training examples and only if we are not performing end2end training.
            for did in corpus[split]: 
                for t_idx in range(len(corpus[split][did])):
                    t = corpus[split][did][t_idx]

                    input_tsr = t[1]
                    if t[1].size(0)//in_red_factor < t[comp_t_idx].size(0):
 
                        pad_len = t[comp_t_idx].size(0) * in_red_factor
                        T, C = t[1].size()
                        lratio = float(pad_len) / float(T)

                        input_tsr = torch.zeros(pad_len,C).to(t[1])
                        for i in range(pad_len):
                            input_tsr[i,:] = t[1][int(i/lratio),:].clone()
                        padded_sequences += 1

                   
                    assert input_tsr.size(0)//in_red_factor >= t[comp_t_idx].size(0), ' input-output sequence lengths missmatch after padding: {} vs. {}'.format(input_tsr.size(0)//in_red_factor, t[comp_t_idx].size(0))

                    turn_tuple = (t[0], input_tsr, t[2], t[3], t[4], t[5])
                    corpus[split][did][t_idx] = turn_tuple
 
        print(' * End2EndSLU: padded {} sequences'.format(padded_sequences))
        sys.stdout.flush()

        return corpus[split], (length_missmatch, missmatches, max_turn_length, total)

    def load_dataset(self, split, **kwargs):
    
        """Load a given dataset split (e.g., train, valid, test)."""

        corpus = {}
        my_split = split
        if my_split == 'valid':
            my_split = 'dev'    # For compatibility with the old End2End SLU system (ICASSP 2020) 

        if os.path.exists(self.args.serialized_data + '.' + my_split) and os.path.isfile(self.args.serialized_data + '.' + my_split): 

            print(' - Loading serialized {} split...'.format(my_split))
            sys.stdout.flush()

            corpus[my_split] = torch.load(self.args.serialized_data + '.' + my_split)

            '''if task.args.target_dictionary:
                        tgt_dict = torch.load(task.args.target_dictionary)
                        for sym in tgt_dict.symbols:
                            tgt_idx = self.dict.index(sym)
                            if sym != unk_token and tgt_idx == self.dict.unk():'''

            '''if self.args.load_dictionary and my_split == 'dev':
                tgt_dict = torch.load(self.args.target_dictionary)
                new_symbols = 0
                for sym in tgt_dict.symbols:
                    tgt_idx = self.label_vocab.index(sym)
                    if tgt_idx == self.label_vocab.unk() and sym != unk_token:
                        print(' * End2EndSLU, adding token {} to the dictionary'.format(sym))
                        sys.stdout.flush()

                        new_symbols += 1
                        self.label_vocab.add_symbol(sym)

                if new_symbols > 0:
                    torch.save(self.label_vocab, self.args.load_dictionary)
                else:
                    print(' * End2EndSLU, no new symbol added to the dictionary')
                    sys.stdout.flush()
                corpus['dictionary'] = self.label_vocab

                print(' * End2EndSLU, new dictionary size: {}'.format(len(self.label_vocab)))
                sys.stdout.flush()'''
        else:
            print(' - Reading dialog data')
            print('   - reading train data...')
            sys.stdout.flush()
            train_data = read_dialog_data( self.args.data + '/train_prefixes.lst', self.args )

            print('   - reading dev data...')
            sys.stdout.flush()
            dev_data = read_dialog_data( self.args.data + '/dev_prefixes.lst', self.args )

            print('   - reading test data...')
            sys.stdout.flush()
            test_data = read_dialog_data( self.args.data + '/test_prefixes.lst', self.args )

            print( ' - Normalizing input data...' )
            sys.stdout.flush()
            (train_mu, train_sigma) = feature_wise_mean_std( train_data )
            normalize_data_only(train_data, train_mu, train_sigma)
            normalize_data_only(dev_data, train_mu, train_sigma)
            normalize_data_only(test_data, train_mu, train_sigma)

            print(' - Mapping tokens to indexes...')
            sys.stdout.flush()
            references2indexes(self.args, train_data, self.label_vocab)
            references2indexes(self.args, dev_data, self.label_vocab)
            references2indexes(self.args, test_data, self.label_vocab)
            print('   - Dictionary size: {}'.format(len(self.label_vocab)))
            sys.stdout.flush()

            corpus = {}
            corpus['train'] = train_data
            corpus['dev'] = dev_data
            corpus['test'] = test_data
            corpus['dictionary'] = self.label_vocab

            print(' - Saving serialized corpus...')
            sys.stdout.flush()
            torch.save(train_data, self.args.serialized_data + '.train' )
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
        self.blank_idx = self.label_vocab.index( blank_token )
        self.sos_tag_idx = self.label_vocab.index( SOS_tag )
        self.eos_tag_idx = self.label_vocab.index( EOS_tag )
        self.slu_start_concept_idx = self.label_vocab.index( slu_start_concept_mark )
        self.slu_end_concept_idx = self.label_vocab.index( slu_end_concept_mark )

        # data for each turn are: turn-id, signal-tensor, char-sequence (int) tensor, token-sequence (int) tensor, concept-sequence (int) tensor, speaker-id (machine or user)
        # data are organized as a dict of lists: the dict is indexed with dialog-id, the value is the list of turns for that dialog, each turn structure contains data described above.

        #debug_idx = 0
        #if hasattr(self.args, 'num_lstm_layers'):
        print(' * End2EndSLU: checking for input-output sequence length incompatibility...')
        sys.stdout.flush()

        print('[DEBUG] dict size before: {}'.format(len(self.label_vocab)))

        cc, infos = self.pad_short_sequences(corpus, my_split)
        corpus[my_split] = cc
        length_missmatch, missmatches, max_turn_length, total = infos

        print('[DEBUG] dict size after: {}'.format(len(self.label_vocab)))
        sys.stdout.flush()

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

        print(' - Reorganizing {} dialog data...'.format(my_split))
        if self.args.user_only:
            print('   - Loading user turns only.') 

        dialogs = corpus[my_split]
        bsz = 1
        #if my_split == 'train':
        while( bsz < self.args.max_sentences ):
            bsz = bsz * 2
        if bsz > self.args.max_sentences:
            bsz = int(bsz / 2)
        if self.args.max_sentences == 5:
            bsz = 5

        print(' - Approximating batch size to {} from {}'.format(bsz, self.args.max_sentences))
        sys.stdout.flush()
        slu_mode_info = {}
        slu_mode_info['dict'] = self.label_vocab
        slu_mode_info['dialog_level_slu'] = self.args.dialog_level_slu if hasattr(self.args, 'dialog_level_slu') else False
        batch_info = create_dialog_batches_(dialogs, bsz, slu_mode_info)
        idx_batches = []

        ratios = []
        sources = []
        src_lengths = []
        targets = []
        tgt_lengths = []
        global_turn_idx = 0
        #for dialog_id in dialogs.keys():
        for batch in batch_info:
            #for turn in turns:
            batched_turns_idx = []
            for (did, idx) in batch:
                turn = dialogs[did][idx]
                if (my_split == 'train') or (turn[-1] == user_ID) or slu_mode_info['dialog_level_slu']:
                    #if turn[-1] == user_ID:
                    # if (not self.args.user_only and (my_split == 'train' or turn[-1] == user_ID)) or (self.args.user_only and turn[-1] == user_ID)
                    feats = turn[1]
                    feats = self.add_speaker_marker(feats, turn[-1])
                    sources.append( feats )
                    src_lengths.append( feats.size(0) ) 

                    '''print(' - End2EndSLU task data shapes:')
                    print('   - feats shape: {}'.format(feats.size()))
                    print('   - char seq shape: {}'.format(turn[2].size()))
                    print('   - token seq shape: {}'.format(turn[3].size()))
                    print('   - slu seq shape: {}'.format(turn[4].size()))
                    print(' -----')
                    sys.stdout.flush()'''

                    batched_turns_idx.append( (global_turn_idx, turn[-1]) )
                    #batched_turns_idx.append( global_turn_idx )
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
                else: 
                    dialogs[did][idx] = None

            if len(batched_turns_idx) > 0:
                idx_batches.append( batched_turns_idx )

        print(' - Reorganized {} turns for split {}'.format(global_turn_idx, my_split))
        sys.stdout.flush()
        src_lengths_tsr = torch.FloatTensor( src_lengths )
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
        rmean = sum(ratios)/len(ratios)
        rstd = math.sqrt( sum([(el - rmean)**2 for el in ratios]) / (len(ratios)-1) )
        print('   * Mean target/source length ratio: {:.4f} (+/- {:.4f})'.format(rmean, rstd))
        print('   * Max. target/source length ratio: {:.4f}'.format(max(ratios)))
        print('   * Min. target/source length ratio: {:.4f}'.format(min(ratios)))
        sys.stdout.flush()
    
        # Reorganize turns based on increasing source length for curriculum learning
        #src_info = []
        #for i in range(len(sources)):
        #    src_info.append( (i, sources[i].size(0)) )
        src_info = [(i, sources[i].size(0)) for i in range(len(sources))]
        sorted_structure = sorted(src_info, key=lambda tuple: tuple[1])
        sources = [sources[t[0]] for t in sorted_structure]
        src_lengths = [src_lengths[t[0]] for t in sorted_structure]
        targets = [targets[t[0]] for t in sorted_structure]
        tgt_lengths = [tgt_lengths[t[0]] for t in sorted_structure]
        self.num_features = sources[0].size(-1)

        input_feed = True 
        self.datasets[split] = End2EndSLUDataset.End2EndSLUDataset(
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
        )

        #if split == 'train':
        #    self.datasets['train'].curriculum(value=True)

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.label_vocab

    # We could override this method if we wanted more control over how batches
    # are constructed, but it's not necessary for this tutorial since we can
    # reuse the batching provided by LanguagePairDataset.
    #
    # def get_batch_iterator(
    #     self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
    #     ignore_invalid_inputs=False, required_batch_size_multiple=1,
    #     seed=1, num_shards=1, shard_id=0,
    # ):
    #     (...)

    #def build_generator(self, args):
    #    from fairseq.slu_sequence_generator import SLUSequenceGenerator

    #    print(' * End2End SLU task: using SLUSequenceGenerator')
    #    sys.stdout.flush()

    #    return SLUSequenceGenerator(
    #        self.target_dictionary,
    #        max_len_a=getattr(args, "max_len_a", 0),
    #        max_len_b=getattr(args, "max_len_b", 200),
    #    )



























