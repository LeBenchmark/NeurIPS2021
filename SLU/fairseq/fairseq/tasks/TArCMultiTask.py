# Code for MultiTask TArC annotation/collection

import os
import re
import sys
import csv
import glob
import torch

from fairseq.tarc_utils import *
from fairseq.data import Dictionary, TarcMultiTaskDataset
from fairseq.tasks import FairseqTask, register_task

from typing import Dict, List

LOSS_INIT_VALUE=999999.9
ER_INIT_VALUE=LOSS_INIT_VALUE

CLS_COL_IDX=1
POS_COL_IDX=4
def set_pos_col_idx(args):
    if args.sub_task == 'tarc-ext':
        return 3
    elif args.sub_task == 'tarc-full':
        return 4
    elif args.sub_task == 'madar-trs-ex':
        return 2
    elif args.sub_task == 'madar-trs-full':
        return 3
    elif args.sub_task == 'tarc-full-npos':
        return 5
    else:
        return 9999999

# Define some utility functions for reading the TArC multi-task data format (tabular format)
def process_morpho(tag):
    res_tags = []
    for c in tag:
        res_tags.append( '@' + c )
    return res_tags

def append_tags( tags, sep ):
    res_tags = []
    for i in range(len(tags)):
        if i == 0:
            res_tags.append( tags[i] )
        else:
            res_tags.append( sep + tags[i] )
    return res_tags

def process_micro_pos(tag):
    
    res_tags = []
    if tag != '_':
        micro_tags = tag.split('_')
        res_tags.extend( append_tags( micro_tags[:-1], '_' ) )
        if len(micro_tags) >= 2 and (micro_tags[-2] == 'ADJ' or micro_tags[-2] == 'PRON') and micro_tags[-1][0] in ['1', '2', '3']:
            morpho_tags = process_morpho( micro_tags[-1] )
            morpho_tags[0] = morpho_tags[0].replace('@', '_') 
            res_tags.extend( morpho_tags )
        else:
            if micro_tags[-1] == 'SG' or micro_tags[-1] == 'PL':
                res_tags.append( '_' + micro_tags[-1] )
            else:
                if micro_tags[-1] != ':':
                    nano_tags = micro_tags[-1].split(':')
                    if len(nano_tags) > 1:
                        if len(nano_tags) != 2:
                            sys.stderr.write(' - Micro POS error: {}\n'.format(micro_tags[-1]))
                            sys.exit(1)
                        res_tags.append( '_' + nano_tags[0] )
                        morpho_tags = process_morpho(nano_tags[1])
                        morpho_tags[0] = morpho_tags[0].replace('@', ':') 
                        res_tags.extend( morpho_tags )
                    else:
                        if len(micro_tags) > 1:
                            res_tags.append( '_' + micro_tags[-1] )
                        else:
                            res_tags.append( micro_tags[-1] )
                else:
                    res_tags.append( micro_tags[-1] )
    else:
        res_tags.append( tag )
    return res_tags

def process_suffix_pos(tag):
    
    res_tags = []
    if tag != '-':
        tag_tokens = tag.split('-')
        if len(tag_tokens) > 1:
            for i in range(len(tag_tokens)):
                t = tag_tokens[i]
                if len(t) > 2 and t[:2] == 'IV' and not (len(t) > 6 and t[:6] == 'IVSUFF'):
                    res_tags.append('IV')
                    morpho_tags = process_morpho( t[2:] )
                    res_tags.extend( morpho_tags )
                else:
                    suffix_tokens = process_micro_pos(t)
                    if i > 0:
                        suffix_tokens[0] = '-' + suffix_tokens[0]
                    res_tags.extend( suffix_tokens )
        else:
            res_tags.extend( process_micro_pos(tag) )
    else:
        res_tags.append(tag)
    return res_tags

def process_macro_pos(tag):
    
    res_tags = []
    if tag != '+':
        tag_tokens = tag.split('+') 
        for i in range(len(tag_tokens)):
            t = tag_tokens[i]
            micro_pos = process_suffix_pos(t)
            if i > 0:
                res_tags.append('+')
            res_tags.extend(micro_pos)
    else:
        res_tags.append(tag)
    return res_tags

def process_pos_tag(tag, args, pad_flag=True, rep_flag=False):  
    res_tags = []
    pos_tokens = tag.split(']')
    if len(pos_tokens) > 1: 
        if len(pos_tokens) != 2:
            sys.stderr.write(' - POS format error (splitting at ]): {}\n'.format(tag))
            sys.exit(1)
        pos_pos_tokens = pos_tokens[0].split('[')
        if len(pos_pos_tokens) > 1 and len(pos_pos_tokens[0]) > 0:  # Case like 'IV2S-IV+[PREP+PRON_1S]IVSUFF_IO:1S'
            if len(pos_pos_tokens) != 2:
                sys.stderr.write(' - POS format error (splitting at [): {}\n'.format(tag))
                sys.exit(1) 

            pref = pos_pos_tokens[0]
            pref_tokens = process_macro_pos( pref ) 
            infix = pos_pos_tokens[1]
            infix_tokens = process_macro_pos( infix )
            infix_tokens = ['+['] + infix_tokens + [']'] 
            post = pos_tokens[1]
            post_tokens = process_macro_pos( post )
            
            res_tags.extend(pref_tokens)
            res_tags.extend( infix_tokens )
            res_tags.extend( post_tokens ) 
        elif len(pos_pos_tokens) > 1:   # Case like '[NOUN_QUANT]ADV'
            if len(pos_pos_tokens) != 2:
                sys.stderr.write(' - POS format error (splitting at [, with first part empty): {}\n'.format(tag))
                sys.exit(1) 

            pref = pos_pos_tokens[1]
            pref_tokens = process_macro_pos( pref )
            pref_tokens = ['['] + pref_tokens + [']']
            post = pos_tokens[1]
            post_tokens = process_macro_pos( post )
            res_tags.extend( pref_tokens )
            res_tags.extend( post_tokens ) 
        else:
            sys.stderr.write(' - POS format error, possibly unbalanced [] at pos {}\n'.format(tag))
            sys.exit(1)
    else:   # "Normal" case (normal for Arabic people...) 
        pos_tokens = process_macro_pos( tag )
        res_tags.extend( pos_tokens )
    if pad_flag:
        return [start_token] + res_tags + [end_token]
    else:
        return res_tags

def create_character_list(token, args, pad_flag=True, rep_flag=False):
    if (not (token in fillers + latin_fillers)) and rep_flag and args.sub_task in ['tarc-ext', 'tarc-full', 'tarc-full-npos']:
        token = replace_all_num(token)
        token = replace_all_pun(token)
        token = replace_all_sym(token)
    elif (not (token in fillers + latin_fillers)) and args.sub_task in ['tarc-ext', 'tarc-full', 'tarc-full-npos']:
        token = replace_all_Lpun(token)
        token = replace_all_Lsym(token)

    tok_lst = []
    tok_idx = 0
    for t in token.split():
        if tok_idx > 0:
            tok_lst.append( space_token )
        for c in t:
            tok_lst.append( c )
        tok_idx += 1

    if args.sub_task in ['tarc-ext', 'tarc-full', 'tarc-full-npos']:
        seq_fillers = detect_fillers(tok_lst, fillers + latin_fillers)
        if len(seq_fillers) > 0:
            new_lst = []
            prev_start = 0
            for t in seq_fillers:
                new_lst = new_lst + tok_lst[prev_start:t[0]] + [''.join(tok_lst[t[0]:t[1]])]
                prev_start = t[1]
            new_lst = new_lst + tok_lst[t[1]:] 
            tok_lst = new_lst
 
    if pad_flag:
        return [start_token] + tok_lst + [end_token]
    else:
        return tok_lst

def process_tokenization( token, args, pad_flag=True, rep_flag=False ):
    
    res_tok = [] 
    if token != '+':
        tok_tokens = token.split('+')
        if len(tok_tokens) > 1:
            res_tok.append(start_token)
            for i in range(len(tok_tokens)):
                t = tok_tokens[i] 
                if len(t) > 0:
                    if i > 0:
                        res_tok.append( '+' )
                    res_tok.extend( create_character_list( t, args, pad_flag=False, rep_flag=rep_flag ) ) 
                else:
                    sys.stderr.write('process_tokenization FORMAT ERROR: found empy token splitting on + >{}< (probably you must replace multiple + in your data)\n'.format(token))
                    sys.exit(1)
            res_tok.append(end_token)
        else: 
            res_tok.extend( create_character_list( token, args, pad_flag=True, rep_flag=rep_flag ) )
    else:
        res_tok.append(start_token)
        res_tok.append(token)
        res_tok.append(end_token)

    return res_tok

def tiger_label_processing(label, args, pad_flag=True, rep_flag=False):

    if label != '$.':
        label_tokens = label.split('.')
    else:
        label_tokens = [label] 

    if pad_flag:
        return [start_token] + label_tokens + [end_token]
    else:
        return label_tokens

def no_processing(token, args, pad_flag=True, rep_flag=False):
    return [token]

madar_translation = (create_character_list, no_processing, create_character_list)
madar_ex_translation = (create_character_list, no_processing, process_pos_tag, create_character_list)
madar_full_translation = (create_character_list, no_processing, create_character_list, process_pos_tag, create_character_list)
tarc_processing_base = (create_character_list, process_tokenization, process_pos_tag)   # Processing functions for raw tunisian, tokenized tunisian and POS
tarc_processing_ext = (create_character_list, no_processing, process_tokenization, process_pos_tag)    # Processing functions for arabish, tunisian, tokenized tunisian and POS
tarc_processing_full = (create_character_list, no_processing, create_character_list, process_tokenization, process_pos_tag)
tarc_processing_full_npos = (create_character_list, no_processing, no_processing, create_character_list, process_tokenization, process_pos_tag)
tarc_substep_1 = (create_character_list, no_processing, create_character_list)
tarc_substep_2 = (create_character_list, process_tokenization)
tarc_substep_3 = (process_tokenization, process_pos_tag)
tiger_mt_processing = (create_character_list, no_processing, no_processing)
tiger_mtext_processing = (create_character_list, no_processing, tiger_label_processing)
tiger4_mt_processing = (create_character_list, no_processing, no_processing, no_processing)

# NOTE: used to decide if tokens and chars representations must be computed and merged when their usage is specified from command line
granularity_merging_flags = {}
granularity_merging_flags['tiger-mt'] = (True, False, False)
granularity_merging_flags['tiger-mt-ext'] = (True, False, True)
granularity_merging_flags['tiger4-mt'] = (True, False, False, False)
granularity_merging_flags['madar-trs'] = (True, False, True)
granularity_merging_flags['madar-trs-ex'] = (True, False, True, True)
granularity_merging_flags['madar-trs-full'] = (True, False, True, True, True)
granularity_merging_flags['tarc-base'] = (True, False, True)
granularity_merging_flags['tarc-ext'] = (True, False, True, True)
granularity_merging_flags['tarc-full'] = (True, False, True, True, True)
granularity_merging_flags['tarc-full-npos'] = (True, False, False, True, True, True)
granularity_merging_flags['tarc-substep1'] = (True, False, True)
granularity_merging_flags['tarc-substep2'] = (True, True)
granularity_merging_flags['tarc-substep3'] = (True, True)

def choose_column_processing(num_columns, args):
    
    column_processing = None
    if args.sub_task == 'tiger-mt':
        print(' - TArCMultiTask, processing mode set to tiger-mt')
        sys.stdout.flush()
        column_processing = tiger_mt_processing
    elif args.sub_task == 'tiger-mt-ext':
        print(' - TArCMultiTask, processing mode set to tiger-mt-ext')
        sys.stdout.flush()
        column_processing = tiger_mtext_processing
    elif args.sub_task == 'tiger4-mt':
        print(' - TArCMultiTask, processing mode set to tiger4-mt')
        sys.stdout.flush()
        column_processing = tiger4_mt_processing
    elif args.sub_task == 'madar-trs':
        print(' - TArCMultiTask, processing mode set to madar-trs')
        sys.stdout.flush()
        column_processing = madar_translation
    elif args.sub_task == 'madar-trs-ex':
        print(' - TArCMultiTask, processing mode set to madar-trs-ex')
        sys.stdout.flush()
        column_processing = madar_ex_translation
    elif args.sub_task == 'madar-trs-full':
        print(' - TArCMultiTask, processing mode set to madar-trs-full')
        sys.stdout.flush()
        column_processing = madar_full_translation
    elif args.sub_task == 'tarc-base':
        print(' - TArCMultiTask, processing mode set to tarc-base')
        sys.stdout.flush()
        column_processing = tarc_processing_base
    elif args.sub_task == 'tarc-ext':
        print(' - TArCMultiTask, processing mode set to tarc-ext')
        sys.stdout.flush()
        column_processing = tarc_processing_ext
    elif args.sub_task == 'tarc-full':
        print(' - TArCMultiTask, processing mode set to tarc-full')
        sys.stdout.flush()
        column_processing = tarc_processing_full
    elif args.sub_task == 'tarc-full-npos':
        print(' - TArCMultiTask, processing mode set to tarc-full-npos')
        column_processing = tarc_processing_full_npos
    elif args.sub_task == 'tarc-substep1':
        print(' - TArCMultiTask, processing mode set to tarc-substep1')
        sys.stdout.flush()
        column_processing = tarc_substep_1
    elif args.sub_task == 'tarc-substep2':
        print(' - TArCMultiTask, processing mode set to tarc-substep2')
        sys.stdout.flush()
        column_processing = tarc_substep_2
    elif args.sub_task == 'tarc-substep3':
        print(' - TArCMultiTask, processing mode set to tarc-substep3')
        sys.stdout.flush()
        column_processing = tarc_substep_3
    elif args.sub_task == 'base':
        print(' - TArCMultiTask, processing mode set to base')
        sys.stdout.flush()
        column_processing = [no_processing for i in range(num_columns)]
    else:
        print(' - TArCMultiTask, setting default processing mode (no processing)')
        sys.stdout.flush()
        column_processing = [no_processing for i in range(num_columns)] 

    return column_processing

def check_column_processing(num_columns, args):
    
    if num_columns == 3 and not args.sub_task in ['tiger-mt', 'tiger-mt-ext', 'tarc-base', 'base', 'madar-trs', 'tarc-substep1']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif num_columns == 2 and not args.sub_task in ['tarc-substep2', 'tarc-substep3', 'base']:
        raise ValueError(' 2 columns are expected with sub-task tarc-substep2|3')
    elif num_columns == 4 and not args.sub_task in ['tiger4-mt', 'tarc-ext', 'base','madar-trs-ex']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif num_columns == 5 and not args.sub_task in ['tarc-full','madar-trs-full', 'base']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif num_columns == 6 and not args.sub_task in ['tarc-full-npos', 'base']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif (not num_columns in [2, 3, 4, 5, 6]) and (not args.sub_task in ['base']):
        raise ValueError(' Unexpected number of columns in input file, possible values: 2, 3, 4, 5 unless base is specified. Got {}'.format(num_columns))

def apply_global_filling(args, curr_seq, cls_seq, Afillers, Lfillers, OFFcols, idx):

    (fillerFOR, fillerEMO) = Afillers
    (LfillerFOR, LfillerEMO) = Lfillers
    (CLS_COL_IDX, POS_COL_IDX) = OFFcols

    if args.sub_task in ['tarc-ext', 'tarc-full', 'madar-trs', 'madar-trs-ex', 'madar-trs-full', 'tarc-full-npos'] and idx != CLS_COL_IDX and idx != POS_COL_IDX:
        for s_idx in range(len(curr_seq)):
            if cls_seq[s_idx] == 'foreign':
                if (args.sub_task == 'tarc-full' and idx == 0) or (args.sub_task == 'tarc-full-npos' and idx == 0) or (args.sub_task == 'madar-trs' and idx == 2) or (args.sub_task == 'madar-trs-ex' and idx == 3) or (args.sub_task == 'madar-trs-full' and idx == 4):
                    curr_seq[s_idx] = LfillerFOR
                else:
                    curr_seq[s_idx] = fillerFOR
            elif cls_seq[s_idx] == 'emotag':
                if (args.sub_task == 'tarc-full' and idx == 0) or (args.sub_task == 'tarc-full-npos' and idx == 0) or (args.sub_task == 'madar-trs' and idx == 2) or (args.sub_task == 'madar-trs-ex' and idx == 3) or (args.sub_task == 'madar-trs-full' and idx == 4):
                    curr_seq[s_idx] = LfillerEMO
                else:
                    curr_seq[s_idx] = fillerEMO

    return curr_seq

def _add_t2c_entry_safe(D: Dict[str, List[str]], key: str, val: List[str]) -> Dict[str, List[str]]:

    if not (key in D):
        D[key] = val
    else:
        if len(D[key]) != len(val):
            print(' - WARNING: adding ambigous value to entry {}: 1. {} VS. 2. {}'.format(key, D[key], val))
            sys.stdout.flush()
        else:
            check_flag = True
            for i in range(len(val)):
                if val[i] != D[key][i]:
                    check_flag = False
                    break
            if not check_flag:
                print(' - WARNING: adding ambigous value to entry {}: 1. {} VS. 2. {}'.format(key, D[key], val))
                sys.stdout.flush()

    return D

def read_tarc_tabular_data(filename, args): 

    with open(filename, 'rb') as f:
        lines = [l.decode('utf-8') for l in f.readlines()]
        f.close() 

    POS_COL_IDX = set_pos_col_idx(args)
    test_flag = False
    if filename[-4:] == 'test':
        test_flag = True

    # NOTE: The following are lists to keep token-level and character-level information.
    #       char_seq_lengths and char_seq_lens are used to store start and end indices of tokens in the character-level information,
    #       so that when using both, the model can retrieve the list of embeddings of characters belonging to a given token, and mix them properly.
    char_sequences = []
    char_seq_lengths = []
    tok_sequences = []
    char_seqs = []
    char_seq_lens = []
    tok_seqs = []

    start_idx = 0
    num_columns = 0
    all_sequences = []
    curr_sequences = []
    for line in lines:
        tokens = line.rstrip(' \r\n').split('\t')

        #print(' - Read tokens ({}): {}'.format(len(tokens), tokens))
        #sys.stdout.flush()

        if len(tokens) > 1 and num_columns == 0:
            num_columns = len(tokens)
            all_sequences = [[] for i in range(num_columns)]
            curr_sequences = [[] for i in range(num_columns)]
        if len(tokens) > 1:

            if len(tokens) != num_columns:
                raise IOError(' Wrong data format, found different number of columns at different input lines ({} VS. {})'.format(len(tokens), num_columns))

            for i in range(num_columns):
                curr_sequences[i].append(tokens[i])
        else:
            for i in range(num_columns):
                curr_seq = apply_global_filling(args, curr_sequences[i], curr_sequences[CLS_COL_IDX], [fillerFOR, fillerEMO], [LfillerFOR, LfillerEMO], [CLS_COL_IDX, POS_COL_IDX], i) 
                all_sequences[i].append( curr_seq )

            curr_sequences = [[] for i in range(num_columns)] 

    print(' *** TarcMultiTask, read {} sequences'.format(len(all_sequences)))
    print(' *** TarcMultiTask, detected {} tasks in data'.format(num_columns-1))
    sys.stdout.flush()

    column_processing = choose_column_processing( num_columns, args )
    check_column_processing( num_columns, args )

    total_tokens = 0
    assert len(all_sequences) == num_columns
    num_sequences = len(all_sequences[0])
    for i in range(num_columns):
        assert len(all_sequences[i]) == num_sequences
    if args.sub_task in ['tarc-ext', 'tarc-full', 'tarc-full-npos']:
        for i in range(num_sequences):
            seq_len = len(all_sequences[0][i])
            for j in range(num_columns):
                assert len(all_sequences[j][i]) == seq_len

    def init_llist(size):
        return [[] for i in range(size)]

    masked_tokens = 0
    token_offsets = []
    token2components = [{} for t_idx in range(num_columns)]
    for ts_idx in range(num_sequences):

        curr_seq_len = len(all_sequences[0][ts_idx])
        for s_idx in range(curr_seq_len):
            tokens = []
            for t_idx in range(num_columns):
                tokens.append( all_sequences[t_idx][ts_idx][s_idx] ) 

            assert len(tokens) == num_columns

            if len(char_sequences) == 0:
                char_sequences = init_llist( num_columns )
            if len(tok_sequences) == 0:
                tok_sequences = init_llist( num_columns )
            if len(char_seq_lengths) == 0:
                char_seq_lengths = init_llist( num_columns )
            if len(char_seqs) == 0:
                char_seqs = init_llist( num_columns )
            if len(char_seq_lens) == 0:
                char_seq_lens = init_llist( num_columns )
            if len(token_offsets) == 0:
                token_offsets = [0 for i in range(len(tokens))]
            if len(tok_seqs) == 0:
                tok_seqs = init_llist( num_columns )
            for idx in range(num_columns):
                if idx > 0 and args.ignore_test_output and test_flag:
                    process_res = no_processing(tokens[idx], args)
                else:
                    rep_flag = True
                    if (idx == 0 and (args.sub_task == 'tarc-full' or args.sub_task == 'tarc-full-npos')) or (idx == 2 and args.sub_task == 'madar-trs') or (args.sub_task == 'madar-trs-ex' and idx == 3) or (args.sub_task == 'madar-trs-full' and idx == 4):
                        rep_flag = False
                    process_res = column_processing[idx]( tokens[idx], args, rep_flag=rep_flag )

                token2components[idx] = _add_t2c_entry_safe(token2components[idx], tokens[idx], process_res) 

                char_seqs[idx].extend( process_res )
                char_seq_lens[idx].append( (token_offsets[idx], token_offsets[idx]+len(process_res)) )
                token_offsets[idx] += len(process_res)
                tok_seqs[idx].append( tokens[idx] )

        total_tokens += len(char_seqs[0])
        for idx in range(len(char_sequences)):
            char_sequences[idx].append( char_seqs[idx] )
            char_seq_lengths[idx].append( char_seq_lens[idx] )
            tok_sequences[idx].append( tok_seqs[idx] ) 
        char_seqs = []
        char_seq_lens = []
        token_offsets = []
        tok_seqs = []

    print(' - Size of char_sequences: {}'.format(len(char_sequences)))
    sys.stdout.flush()
    for ci in range(len(char_sequences)):
        print('   - Size of dim {}: {}'.format(ci, len(char_sequences[ci])))
        sys.stdout.flush()

    print(' - Loaded {} sequences, {} tokens (masked {}) from {}'.format( len(char_sequences[0]), total_tokens, masked_tokens, filename ))
    sys.stdout.flush()

    return (tok_sequences, char_sequences, char_seq_lengths, token2components)

def read_tarc_parallel_data( file_prefix, args ):

    input_file = file_prefix + '.input'
    output_files = glob.glob(file_prefix + '.output*')
 
    POS_COL_IDX = set_pos_col_idx(args)
    test_flag = False
    if file_prefix[-4:] == 'test':
        test_flag = True

    input_data = []
    f = open(input_file, encoding='utf-8')
    data = f.readlines()
    f.close()
    total_tokens = 0
    for s in data:
        tokens = s.rstrip(' \r\n').split()
        total_tokens += len(tokens)
        input_data.append(tokens)
    print(' - TArCMultiTask, read {} sequences, {} tokens from input data'.format(len(input_data), total_tokens))
    sys.stdout.flush()

    output_data = [[] for i in range(len(output_files))]
    for i in range( len(output_files) ):
        f = open(output_files[i], encoding='utf-8')
        data = f.readlines()
        f.close()

        total_tokens = 0
        s_idx = 0
        for s in data:
            tokens = s.rstrip(' \r\n').split()
            total_tokens += len(tokens)
            if i > 0:
                i_seq = input_data[s_idx]
                input_data[s_idx] = apply_global_filling(args, i_seq, output_data[CLS_COL_IDX-1][s_idx], [fillerFOR, fillerEMO], [LfillerFOR, LfillerEMO], [CLS_COL_IDX, POS_COL_IDX], 0)
                tokens = apply_global_filling(args, tokens, output_data[CLS_COL_IDX-1][s_idx], [fillerFOR, fillerEMO], [LfillerFOR, LfillerEMO], [CLS_COL_IDX, POS_COL_IDX], i+1)
            output_data[i].append( tokens )
            s_idx += 1
        print(' - TArCMultiTask, read {} sequences, {} tokens from output data {}'.format(len(output_data[i]), total_tokens, i))
        sys.stdout.flush()
 
    for i in range( len(output_data) ):
        assert len(input_data) == len(output_data[i])
        #for s_idx in range(len(input_data)):
        #    assert len( input_data[s_idx] ) == len( output_data[i][s_idx] )

    num_columns = len(output_data)+1
    column_processing = choose_column_processing(num_columns, args)
    check_column_processing(num_columns, args)

    total_tokens = 0
    # NOTE: The following are lists to keep token-level and character-level information.
    #       char_seq_lengths and char_seq_lens are used to store start and end indices of tokens in the character-level information,
    #       so that when using both, the model can retrieve the list of embeddings of characters belonging to a given token, and mix them properly.
    def init_llist(size):
        return [[] for i in range(size)]
    lsize = len(output_data)+1
    char_sequences = init_llist( lsize )
    char_seq_lengths = init_llist( lsize )
    tok_sequences = init_llist( lsize )
    char_seqs = init_llist( lsize )
    char_seq_lens = init_llist( lsize )
    tok_seqs = init_llist( lsize )
    token_offsets = [0 for i in range( lsize )]
    token2components = [{} for t_idx in range(num_columns)]
    for i in range( len(input_data) ):
        c_idx = 0
        for t in input_data[i]:
            rep_flag = True
            if (c_idx == 0 and (args.sub_task == 'tarc-full' or args.sub_task == 'tarc-full-npos')) or (c_idx == 2 and args.sub_task == 'madar-trs') or (args.sub_task == 'madar-trs-ex' and c_idx == 3) or (args.sub_task == 'madar-trs-full' and c_idx == 4):
                rep_flag = False
            #process_res = column_processing[idx]( tokens[idx], args, rep_flag=rep_flag )
            tok_res = column_processing[c_idx](t, args, rep_flag=rep_flag)
            total_tokens += len(tok_res)
            char_seqs[c_idx].extend( tok_res )
            char_seq_lens[c_idx].append( (token_offsets[c_idx], token_offsets[c_idx]+len(tok_res)) )
            token_offsets[c_idx] += len(tok_res) 
            tok_seqs[c_idx].append( t )

            token2components[c_idx] = _add_t2c_entry_safe(token2components[c_idx], t, tok_res) 

        char_sequences[c_idx].append( char_seqs[c_idx] )
        char_seq_lengths[c_idx].append( char_seq_lens[c_idx] )
        tok_sequences[c_idx].append( tok_seqs[c_idx] )
        c_idx += 1
        for j in range(len(output_data)):
            for t in output_data[j][i]: 
                if args.ignore_test_output and test_flag:
                    process_res = no_processing(t)
                else:
                    rep_flag = True
                    if (c_idx == 0 and (args.sub_task == 'tarc-full' or args.sub_task == 'tarc-full-npos')) or (c_idx == 2 and args.sub_task == 'madar-trs') or (args.sub_task == 'madar-trs-ex' and c_idx == 3) or (args.sub_task == 'madar-trs-full' and c_idx == 4):
                        rep_flag = False 
                    process_res = column_processing[c_idx](t, args, rep_flag=rep_flag)
                char_seqs[c_idx].extend( process_res )
                char_seq_lens[c_idx].append( (token_offsets[c_idx], token_offsets[c_idx]+len(process_res)) )
                token_offsets[c_idx] += len(process_res)
                tok_seqs[c_idx].append(t)

                token2components[c_idx] = _add_t2c_entry_safe(token2components[c_idx], t, process_res)

            char_sequences[c_idx].append( char_seqs[c_idx] )
            char_seq_lengths[c_idx].append( char_seq_lens[c_idx] )
            tok_sequences[c_idx].append( tok_seqs[c_idx] )
            c_idx += 1
        char_seqs = init_llist( lsize )
        char_seq_lens = init_llist( lsize )
        tok_seqs = init_llist( lsize )
        token_offsets = [0 for i in range(lsize)]

    print(' - TArCMultiTask, read {} sequences, {} tokens'.format(len(char_sequences[0]), total_tokens))
    sys.stdout.flush()

    return (tok_sequences, char_sequences, char_seq_lengths, token2components)

def map_tokens(data, dict, pad_flag):

    tensors = []
    for s in data:
        tok_lst = []
        for t in s: 
            tok_lst.append( dict.add_symbol(t) )
        if pad_flag:
            tok_lst = [dict.bos()] + tok_lst + [dict.eos()] 
        tensors.append( torch.LongTensor( tok_lst ) )

    return tensors

@register_task('tarc_multitask')
class TarcMultiTask(FairseqTask):
    
    @staticmethod
    def add_args(parser):
        
        # Add some command-line arguments for specifying where the data is
        # located and the maximum supported input length.
        parser.add_argument('data', metavar='FILE',
                            help='Main input file in tabular format. It must contain at least 2 columns; the first column is interpreted as the input, all the other columns are interpreted as outputs, unless --num-of-tasks and --task-indeces specify otherwise')
        parser.add_argument('--num-of-tasks', type=int, default=-1,
                            help='Number of tasks to learn simultaneously. It must correspond at most to the number of columns minus one in the main input file')
        parser.add_argument('--task-indeces', type=str, default='1:',
                            help='Indeces of columns in the main input file corresponding to the learnable tasks. It is represented as an interval string S:E')
        parser.add_argument('--additinal-data', type=str,
                            help='A semicolon separated list of files for multi-task learning')
        parser.add_argument('--data-description', type=str,
                            help='A string describing the content of each additinal input data in the format [N,S:E;]+, where N is the number of tasks, S:E the indeces of the learnable tasks')
        parser.add_argument('--pad-reference', action='store_true', default=False,
                            help='Specify to pad sequences with start and end of sequence markers')
        parser.add_argument('--max-source-positions', type=int, default=2048,
                            help='Maximum number of items in source sequences')
        parser.add_argument('--max-target-positions', type=int, default=2048,
                            help='Maximum number of items in target sequences')
        parser.add_argument('--sub-task', type=str, default='base',
                            help='Which multi-task problem to address: 1) base (default, no pre-processing) 2) tiger-mt, 2) tarc-base 3) tarc-ext')
        parser.add_argument('--data-format', type=str, default='tabular',
                            help='Format of input data: 1) tabular (default), 2) parallel')
        parser.add_argument('--sequence-separator', type=str, default='_SEQ_SEP_',
                            help='Used to separate output sequences from different tasks')
        parser.add_argument('--serialized-data', type=str,
                            help='Load data from a previously created serialization of the whole corpus. If it does not exist, read the whole corpus and serialize it.')
        parser.add_argument('--ignore-test-output', action='store_true', default=False,
                            help='Don\'t apply pre-processing to test output(s)')
        parser.add_argument('--keep-data-order', action='store_true', default=False,
                            help='Keep data in the original order, that is does not sort sequences by length.')
        parser.add_argument('--reverse-sequences', action='store_true', default=False,
                            help='Reverse sequences, that is they will be processed from right to left (e.g. for Arabic processing)')
        parser.add_argument('--reverse-tokens', action='store_true', default=False,
                            help='Reverse tokens individually, that is they will be processed from last to first char (e.g. for Arabic processing)')
        parser.add_argument('--load-madar-model', type=str, default='None',
                            help='Task specific for --sub-task=tarc-full, pre-trained model loading for parameter pre-initialization')
        parser.add_argument('--load-madar-data', type=str, default='None',
                            help='Task specific for --sub-task=tarc-full, serialized data loading for embedding pre-initialization')
        parser.add_argument('--token-sequences', action='store_true', default=False,
                            help='Use token-level information for modelling sequences')
        parser.add_argument('--char-sequences', action='store_true', default=False,
                            help='Use character-level information for modelling sequences')
        parser.add_argument('--double-learning', action='store_true', default=False,
                            help='Learn the model from both token and character representations')

    @classmethod
    def setup_task(cls, args, **kwargs):
        
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just initialize the label dictionary
        
        input_vocab = Dictionary(
                                 pad=pad_token,
                                 eos=eos_token,
                                 unk=unk_token,
                                 bos=bos_token,
                                 extra_special_symbols=[start_token, end_token]
        )
        output_vocab = input_vocab

        print('| [token] dictionary: {} types'.format(len(input_vocab)))
        print('| [label] dictionary: {} types'.format(len(output_vocab)))
        if (not args.token_sequences) and (not args.char_sequences):
            args.char_sequences = True

        if args.token_sequences:
            print('  - TArCMultiTask, using token-level information in sequences')
            sys.stdout.flush()
        if args.char_sequences:
            print('  - TArCMultiTask, using character-level information in sequences')
            sys.stdout.flush()
        
        return TarcMultiTask(args, input_vocab, output_vocab)

    def __init__(self, args, input_vocab, output_vocab):
        
        super().__init__(args)
        self.args = args

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
 
        self.sequence_separator = self.input_vocab.add_symbol(args.sequence_separator)
        self.token2components_tsr = []
        self.granularity_merging_flags = {}
        self.double_learning = args.double_learning 
        self.splits = {}
        self.num_of_inputs = 1 

    def set_granularity_merging_flags(self, g_flags):
        self.granularity_merging_flags = g_flags

    def get_granularity_merging_flags(self):
        return self.granularity_merging_flags

    def _t2c_to_tsr(self, t2c: Dict[str, List[str]], dict: Dictionary) -> Dict[int, torch.LongTensor]:
 
        res_dict = {}
        for k in t2c.keys():
            res_dict[dict.add_symbol(k)] = torch.LongTensor( [dict.add_symbol(v) for v in t2c[k]] )
        return res_dict

    def load_dataset(self, split, **kwargs):
    
        """Load a given dataset split (e.g., train, valid, test)."""

        if len(self.datasets) == 0:
            if self.args.serialized_data is not None and os.path.exists(self.args.serialized_data) and os.path.isfile(self.args.serialized_data):
                print(' - TArCMultiTask, reading serialized data from {}...'.format(self.args.serialized_data))
                sys.stdout.flush()
                self.splits = torch.load( self.args.serialized_data )
                self.input_vocab = self.splits['vocab']
                self.output_vocab = self.input_vocab
                self.token2components_tsr = self.splits['token2components'] 
            else:
                print(' - TArCMultiTask, creating dictionaries from whole corpus...')
                sys.stdout.flush()
 
                token2components = []
                for my_split in ['train', 'dev', 'test']:
                    data_sequences = []
                    if self.args.data_format == 'tabular':
                        data_sequences = read_tarc_tabular_data( self.args.data + '.' + my_split, self.args) 
                    elif self.args.data_format == 'parallel': 
                        data_sequences = read_tarc_parallel_data( self.args.data + '.' + my_split, self.args )
                    else:
                        raise NotImplementedError(' unsupported data format {}'.format(self.args.data_format))

                    if len(token2components) == 0:
                        token2components = [{bos_token : [bos_token], eos_token : [eos_token]} for t_idx in range(len(data_sequences[3]))]
                    for t_idx in range(len(data_sequences[3])):
                        token2components[t_idx].update( data_sequences[3][t_idx] )

                    if self.args.token_sequences and self.args.char_sequences:
                        assert len(data_sequences[0]) == len(data_sequences[1])
                        assert len(data_sequences[1]) == len(data_sequences[2])

                    print(' - load_dataset, read {} data split'.format(my_split))
                    print('   - got {} tasks'.format(len(data_sequences[0])-1))
                    sys.stdout.flush()

                    tensors = []
                    lengths = []
                    tok_tensors = []
                    tok_lengths = []
                    char_seq_lengths = [[] for i in range(len(data_sequences[2]))]
                    for d_idx in range( len(data_sequences[0]) ):   
                        tok_tt = map_tokens(data_sequences[0][d_idx], self.input_vocab, self.args.pad_reference)
                        tok_tensors.append( tok_tt )
                        tok_ll = torch.LongTensor( [t.size(0) for t in tok_tt] )
                        tok_lengths.append( tok_ll )
                       
                        seq_tt = []
                        for s in data_sequences[2][d_idx]:
                            if self.args.pad_reference:
                                s = [(p[0]+1, p[1]+1) for p in s]
                            char_seq_lengths[d_idx].append( torch.LongTensor( s ) )    # As many pairs as tokens...
                      
                        tt = map_tokens(data_sequences[1][d_idx], self.input_vocab, self.args.pad_reference)
                        tensors.append( tt )
                        ll = torch.LongTensor([t.size(0) for t in tt])
                        lengths.append(ll) 
                    self.splits[my_split] = ([tok_tensors, tensors], [tok_lengths, lengths, char_seq_lengths])
 
                for t_idx in range(len(token2components)): 
                    self.token2components_tsr.append( self._t2c_to_tsr(token2components[t_idx], self.input_vocab) ) 

                self.splits['vocab'] = self.input_vocab
                self.splits['token2components'] = self.token2components_tsr 
                if self.args.serialized_data is not None: 
                    print(' - TArCMultiTask, serializing data to file {}...'.format(self.args.serialized_data))
                    sys.stdout.flush()
                    torch.save(self.splits, self.args.serialized_data)

        assert len(self.splits) == 5 and 'train' in self.splits.keys() and 'dev' in self.splits.keys() and 'test' in self.splits.keys()
        print(' - TArCMultiTask, instantiating current split {}'.format(split))
        sys.stdout.flush() 
        my_split = split
        if split == 'valid':
            my_split = 'dev' 

        tensors, lengths = self.splits[my_split]

        print(' - Split {} data lengths statistics:'.format(split))
        for idx, ll in enumerate(lengths[1]):
            print('\t- Max. length @{}: {}'.format(idx, torch.max(ll).item()))
            print('\t- Min. length @{}: {}'.format(idx, torch.min(ll).item())) 
            print('\t-----')
        print(' - Dictionary sizes: {}, {}'.format(len(self.input_vocab), len(self.output_vocab)))
        print(' _______________')
 
        num_of_tasks = len(tensors[0])-1
        self.num_of_inputs = len(tensors[0]) - num_of_tasks
        self.args.num_of_tasks = num_of_tasks
        bound_idx = len(tensors[0])-num_of_tasks
 
        if 'base' not in granularity_merging_flags:
            granularity_merging_flags['base'] = tuple([False for i in range(num_of_tasks+1)])
        if self.args.sub_task not in granularity_merging_flags:
            granularity_merging_flags[self.args.sub_task] = tuple([False for i in range(num_of_tasks+1)]) 
        self.set_granularity_merging_flags(granularity_merging_flags[self.args.sub_task])

        sources = [tensors[0][0:bound_idx], tensors[1][0:bound_idx]] 
        src_lengths = [lengths[0][0:bound_idx], lengths[1][0:bound_idx], lengths[2][0:bound_idx]]
        targets = [tensors[0][bound_idx:], tensors[1][bound_idx:]] 
        tgt_lengths = [lengths[0][bound_idx:], lengths[1][bound_idx:], lengths[2][bound_idx:]]

        print(' - Tarc MultiTask, learning with {} input(s) (lengths: {}), {} different outputs (num. of tasks: {}, lengths: {})'.format(len(sources), len(src_lengths), len(targets), self.args.num_of_tasks, len(tgt_lengths)))
        sys.stdout.flush() 

        input_feed = True 
        self.datasets[split] = TarcMultiTaskDataset.TarcMultiTaskDataset(
            src=sources,
            src_sizes=src_lengths,
            src_dict=self.input_vocab,
            tgt=targets,
            tgt_sizes=tgt_lengths,
            sequence_separator=self.sequence_separator,
            tgt_dict=self.output_vocab,
            left_pad_target=False,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions, 
            input_feeding=input_feed,
            keep_data_order=self.args.keep_data_order,
            granularity_flags=(self.args.token_sequences, self.args.char_sequences),
        ) 

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and the "target"
        # has max length 1.
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.input_vocab

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.output_vocab

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

    def build_generator(self, args):
        
        from fairseq.tarc_multitask_sequence_generator import TarcSequenceGenerator 

        return TarcSequenceGenerator(
            self.target_dictionary,
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
        )



























