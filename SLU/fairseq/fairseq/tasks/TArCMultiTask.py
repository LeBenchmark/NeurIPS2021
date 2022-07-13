# Code for MultiTask TArC annotation/collection

import os
import re
import sys
import csv
import glob
import torch

from fairseq.tarc_utils import *
from fairseq.data import TarcMultiTaskDataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.End2EndSLU import SLUDictionary, init_slu_dictionary

from nltk.tree import Tree
from typing import Dict, List

LOSS_INIT_VALUE=999999.9
ER_INIT_VALUE=LOSS_INIT_VALUE

punct = [',', '.', ':', '\'\'', '$', '``', '-LRB-', '-RRB-']

CLS_COL_IDX=1
POS_COL_IDX=4
def set_pos_col_idx(args):
    if args.sub_task == 'tarc-ext':
        return 3
    elif args.sub_task in ['tarc-ext-npos', 'tarc-full', 'tarc-lemma']:
        return 4
    elif args.sub_task == 'madar-trs-ex':
        return 2
    elif args.sub_task in ['madar-trs-full']:
        return 3
    elif args.sub_task == 'madar-trs-full-ex':
        return 4
    elif args.sub_task in ['tarc-full-npos', 'tarc-lemma-full']:
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
    if (not (token in fillers + latin_fillers)) and rep_flag and ((hasattr(args, 'apply_filling') and args.apply_filling) or args.sub_task in ['tarc-ext', 'tarc-ext-npos', 'tarc-full', 'tarc-full-npos', 'tarc-lemma', 'tarc-lemma-full']):
        token = replace_all_num(token)
        token = replace_all_pun(token)
        token = replace_all_sym(token)
    elif (not (token in fillers + latin_fillers)) and ((hasattr(args, 'apply_filling') and args.apply_filling) or args.sub_task in ['tarc-ext', 'tarc-ext-npos', 'tarc-full', 'tarc-full-npos', 'tarc-lemma', 'tarc-lemma-full']):
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

    if (hasattr(args, 'apply_filling') and args.apply_filling) or args.sub_task in ['tarc-ext', 'tarc-ext-npos', 'tarc-full', 'tarc-full-npos', 'tarc-lemma', 'tarc-lemma-full']:
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

opmap = {}
opmap['nop'] = no_processing
opmap['char'] = create_character_list
opmap['tok'] = process_tokenization
opmap['pos'] = process_pos_tag
opmap['Tpos'] = tiger_label_processing
madar_translation = (create_character_list, no_processing, create_character_list)
madar_ex_translation = (create_character_list, no_processing, process_pos_tag, create_character_list)
madar_full_translation = (create_character_list, no_processing, process_tokenization, process_pos_tag, create_character_list)
madar_fullex_translation = (create_character_list, no_processing, create_character_list, process_tokenization, process_pos_tag)
tarc_processing_base = (create_character_list, process_tokenization, process_pos_tag)   # Processing functions for raw tunisian, tokenized tunisian and POS
tarc_processing_ext = (create_character_list, no_processing, process_tokenization, process_pos_tag)    # Processing functions for arabish, tunisian, tokenized tunisian and POS
tarc_processing_ext_npos = ('char', 'nop', 'tok', 'nop', 'pos')
tarc_processing_full = (create_character_list, no_processing, create_character_list, process_tokenization, process_pos_tag)
tarc_processing_full_npos = (create_character_list, no_processing, no_processing, create_character_list, process_tokenization, process_pos_tag)
tarc_lemmatization = ('char', 'nop', 'char', 'tok', 'pos', 'char')
tarc_lemmatization_full = ('char', 'nop', 'char', 'char', 'tok', 'pos')
tarc_substep_1 = (create_character_list, no_processing, create_character_list)
tarc_substep_2 = (create_character_list, process_tokenization)
tarc_substep_3 = (process_tokenization, process_pos_tag)
tiger_mt_processing = (create_character_list, no_processing, no_processing)
tiger_mtext_processing = (create_character_list, no_processing, tiger_label_processing)
tiger4_mt_processing = (create_character_list, no_processing, no_processing, no_processing)
wsj_processing = ('char', 'nop', 'nop')

# NOTE: used to decide if tokens and chars representations must be computed and merged when their usage is specified from command line
granularity_merging_flags = {}
granularity_merging_flags['tiger-mt'] = (True, False, False)
granularity_merging_flags['tiger-mt-ext'] = (True, False, True)
granularity_merging_flags['tiger4-mt'] = (True, False, False, False)
granularity_merging_flags['madar-trs'] = (True, False, True)
granularity_merging_flags['madar-trs-ex'] = (True, False, True, True)
granularity_merging_flags['madar-trs-full'] = (True, False, True, True, True)
granularity_merging_flags['madar-trs-full-ex'] = (True, False, True, True, True)
granularity_merging_flags['tarc-base'] = (True, False, True)
granularity_merging_flags['tarc-ext'] = (True, False, True, True)
granularity_merging_flags['tarc-full'] = (True, False, True, True, True)
granularity_merging_flags['tarc-full-npos'] = (True, False, False, True, True, True)
granularity_merging_flags['tarc-substep1'] = (True, False, True)
granularity_merging_flags['tarc-substep2'] = (True, True)
granularity_merging_flags['tarc-substep3'] = (True, True)

def choose_column_processing(num_columns, args):

    gflags = None
    column_processing = None
    if args.sub_task == 'base':
        print(' - TArCMultiTask, processing mode set to base')
        sys.stdout.flush()

        gflags = {}
        column_processing = []
        gflags[args.sub_task] = []
        for ti in range(num_columns):
            if ti == 0 and (args.token_sequences and args.char_sequences):
                column_processing.append( opmap['char'] )
                gflags[args.sub_task].append(True)
            elif args.char_sequences and not args.token_sequences:
                column_processing.append( opmap['char'] )
                gflags[args.sub_task].append(False)
            else:
                column_processing.append( opmap['nop'] )
                gflags[args.sub_task].append(False)
    elif args.sub_task == 'tiger-mt':
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
    elif args.sub_task == 'madar-trs-full-ex':
        print(' - TArCMultiTask, processing mode set to madar-trs-full-ex')
        sys.stdout.flush()
        column_processing = madar_fullex_translation
    elif args.sub_task == 'tarc-base':
        print(' - TArCMultiTask, processing mode set to tarc-base')
        sys.stdout.flush()
        column_processing = tarc_processing_base
    elif args.sub_task == 'tarc-ext':
        print(' - TArCMultiTask, processing mode set to tarc-ext')
        sys.stdout.flush()
        column_processing = tarc_processing_ext
    elif args.sub_task == 'tarc-ext-npos':
        print(' - TArCMultiTask, processing mode set to tarc-ext-npos')
        sys.stdout.flush()
        gflags = {}
        column_processing = []
        gflags[args.sub_task] = []
        for s in tarc_processing_ext_npos:
            column_processing.append( opmap[s] )
            if s == 'nop':
                gflags[args.sub_task].append( False )
            else:
                gflags[args.sub_task].append( True )
    elif args.sub_task == 'tarc-lemma':
        print(' - TArCMultiTask, processing mode set to tarc-lemma')
        sys.stdout.flush()
        gflags = {}
        column_processing = []
        gflags[args.sub_task] = []
        for s in tarc_lemmatization:
            column_processing.append( opmap[s] )
            if s == 'nop':
                gflags[args.sub_task].append( False )
            else:
                gflags[args.sub_task].append( True )
    elif args.sub_task == 'tarc-lemma-full':
        print(' - TArCMultiTask, processing mode set to tarc-lemma-full')
        sys.stdout.flush()
        gflags = {}
        column_processing = []
        gflags[args.sub_task] = []
        for s in tarc_lemmatization_full:
            column_processing.append( opmap[s] )
            if s == 'nop':
                gflags[args.sub_task].append( False )
            else:
                gflags[args.sub_task].append( True )
    elif args.sub_task == 'wsj':
        print(' - TArCMultiTask, processing mode set to wsj')
        sys.stdout.flush()

        gflags = {}
        column_processing = []
        gflags[args.sub_task] = []
        for s in wsj_processing:
            column_processing.append( opmap[s] )
            if s == 'nop':
                gflags[args.sub_task].append(False)
            else:
                gflags[args.sub_task].append(True) 
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

    return column_processing, gflags

def check_column_processing(num_columns, args):
    
    if num_columns == 3 and not args.sub_task in ['wsj', 'tiger-mt', 'tiger-mt-ext', 'tarc-base', 'base', 'madar-trs', 'tarc-substep1']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif num_columns == 2 and not args.sub_task in ['tarc-substep2', 'tarc-substep3', 'base']:
        raise ValueError(' 2 columns are expected with sub-task tarc-substep2|3')
    elif num_columns == 4 and not args.sub_task in ['tiger4-mt', 'tarc-ext', 'base','madar-trs-ex']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif num_columns == 5 and not args.sub_task in ['tarc-full', 'tarc-ext-npos', 'madar-trs-full', 'madar-trs-full-ex', 'base']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif num_columns == 6 and not args.sub_task in ['tarc-full-npos', 'base', 'tarc-lemma', 'tarc-lemma-full']:
        raise ValueError(' wrong num. of columns in input data for processing mode {}'.format(args.sub_task))
    elif (not num_columns in [2, 3, 4, 5, 6]) and (not args.sub_task in ['base']):
        raise ValueError(' Unexpected number of columns in input file, possible values: 2, 3, 4, 5 unless base is specified. Got {}'.format(num_columns))

def apply_global_filling(args, curr_seq, cls_seq, Afillers, Lfillers, OFFcols, idx):
    '''
        Put fillers in place of tokens for columns corresponding to a class 'foreign' and 'emotag', except for the POS column.
        For Arabizi a filler in latin character is used (Lfiller(FOR|EMO))
    '''

    (fillerFOR, fillerEMO) = Afillers
    (LfillerFOR, LfillerEMO) = Lfillers
    (CLS_COL_IDX, POS_COL_IDX) = OFFcols

    if ((hasattr(args, 'apply_filling') and args.apply_filling) or args.sub_task in ['tarc-ext', 'tarc-ext-npos', 'tarc-full', 'madar-trs', 'madar-trs-ex', 'madar-trs-full', 'madar-trs-full-ex', 'tarc-full-npos', 'tarc-lemma', 'tarc-lemma-full']) and idx != CLS_COL_IDX and idx != POS_COL_IDX:
        for s_idx in range(len(curr_seq)):
            if cls_seq[s_idx] == 'foreign' and (not curr_seq[s_idx] in [bos_token, pad_token, eos_token, unk_token, start_token, end_token, space_token, separator_]):
                # NOTE: (hasattr(args, 'apply_filling') and args.apply_filling works if input sequences are in Arabizi. TODO: modify to be generic
                if (hasattr(args, 'apply_filling') and args.apply_filling) or (args.sub_task == 'tarc-full' and idx == 0) or (args.sub_task == 'tarc-full-npos' and idx == 0) or (args.sub_task == 'madar-trs' and idx == 2) or (args.sub_task == 'madar-trs-ex' and idx == 3) or (args.sub_task == 'madar-trs-full' and idx == 4) or (args.sub_task == 'madar-trs-full-ex' and idx == 2) or (args.sub_task in ['tarc-lemma', 'tarc-lemma-full'] and idx == 0):
                    curr_seq[s_idx] = LfillerFOR
                else:
                    curr_seq[s_idx] = fillerFOR
            elif cls_seq[s_idx] == 'emotag' and (not curr_seq[s_idx] in [bos_token, pad_token, eos_token, unk_token, start_token, end_token, space_token, separator_]):
                # NOTE: same as above
                if (hasattr(args, 'apply_filling') and args.apply_filling) or (args.sub_task == 'tarc-full' and idx == 0) or (args.sub_task == 'tarc-full-npos' and idx == 0) or (args.sub_task == 'madar-trs' and idx == 2) or (args.sub_task == 'madar-trs-ex' and idx == 3) or (args.sub_task == 'madar-trs-full' and idx == 4) or (args.sub_task == 'madar-trs-full-ex' and idx == 2) or (args.sub_task in ['tarc-lemma', 'tarc-lemma-full'] and idx == 0):
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

def add_indices_to_dict(dd, tensors):
    for t in tensors:
        for el in list(t):
            if el.item() not in dd:
                dd[el.item()] = 1

def add_entries_to_inverse_dict(inverse_dict, global_dict, tensors):

    for t in tensors:
        tmp_str = global_dict.string(t)
        for tk in tmp_str.split():
            if global_dict.index(tk) not in inverse_dict:
                inverse_dict[global_dict.index(tk)] = tk

def add_entries_to_inverse_dict_from_trees(inverse_dict, global_dict, tree_format, tensors):

    if tree_format == 'wsj':
        raise NotImplementedError
    else:
        for t in tensors:
            tmp_str = global_dict.string(t)
            for tk in tmp_str.split():
                if '(' in tk or ')' in tk:
                    if global_dict.index(tk) not in inverse_dict:
                        inverse_dict[global_dict.index(tk)] = tk

def add_tree_tokens_to_dict_no_leaves(output_dict, tree_token_mapping, tt):

    if isinstance(tt, Tree):
        t_idx = tree_token_mapping.index(tt.label())
        if t_idx not in output_dict:
            #print(' *** Adding {} to output dictionary (index {})'.format(tt.label(), t_idx))
            #sys.stdout.flush()
            output_dict[t_idx] = 1
        for child in tt:
            add_tree_tokens_to_dict_no_leaves(output_dict, tree_token_mapping, child)

def create_output_dict_from_trees(output_dict, tree_token_mapping, tree_format, tree_tensors):

    for t in tree_tensors:
        tree_str = tree_token_mapping.string(t)
        if tree_format == 'wsj':
            ttree = Tree.fromstring(tree_str)
            add_tree_tokens_to_dict_no_leaves(output_dict, tree_token_mapping, ttree)
        elif tree_format == 'GAFL':
            tokens = tree_str.split()
            for t in tokens:
                if '(' in t or ')' in t:
                    t_idx = tree_token_mapping.index(t)
                    if t_idx not in output_dict:
                        #print(' * Adding {} to the output dict'.format(t))
                        #sys.stdout.flush()

                        output_dict[t_idx] = 1
        else:
            raise NotImplementedError(' tree format {} undefined ')

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
                #all_sequences[i].append( curr_sequences[i] )

            curr_sequences = [[] for i in range(num_columns)] 

    print(' *** TarcMultiTask, read {} sequences'.format(len(all_sequences)))
    print(' *** TarcMultiTask, detected {} tasks in data'.format(num_columns-1))
    sys.stdout.flush()

    column_processing, gflags = choose_column_processing( num_columns, args )
    if gflags is not None:
        granularity_merging_flags[args.sub_task] = gflags[args.sub_task]
    check_column_processing( num_columns, args )

    total_tokens = 0
    assert len(all_sequences) == num_columns
    num_sequences = len(all_sequences[0])
    for i in range(num_columns):
        assert len(all_sequences[i]) == num_sequences
    if args.sub_task in ['tarc-ext', 'tarc-ext-npos', 'tarc-full', 'tarc-full-npos']:
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
                if (idx > 0 and args.ignore_test_output and test_flag) or tokens[idx] in [bos_token, pad_token, eos_token, unk_token, start_token, end_token, space_token, separator_]:
                    process_res = no_processing(tokens[idx], args)
                else:
                    rep_flag = True
                    # NOTE: see note above concerning apply_filling flag
                    if (hasattr(args, 'apply_filling') and args.apply_filling) or (idx == 0 and (args.sub_task == 'tarc-full' or args.sub_task == 'tarc-full-npos' or args.sub_task == 'tarc-lemma' or args.sub_task == 'tarc-lemma-full')) or (idx == 2 and args.sub_task == 'madar-trs') or (args.sub_task == 'madar-trs-ex' and idx == 3) or (args.sub_task == 'madar-trs-full' and idx == 4) or (args.sub_task == 'madar-trs-full-ex' and idx == 2):
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

    # 1. Read input sequences
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

    # 2. Read output sequences
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
        print(' - TArCMultiTask, read {} sequences, {} tokens from output data {}'.format(len(output_data[i]), total_tokens, output_files[i]))
        sys.stdout.flush()
 
    for i in range( len(output_data) ):
        assert len(input_data) == len(output_data[i])
        #for s_idx in range(len(input_data)):
        #    assert len( input_data[s_idx] ) == len( output_data[i][s_idx] )

    num_columns = len(output_data)+1
    column_processing, gflags = choose_column_processing(num_columns, args)
    if gflags is not None:
        granularity_merging_flags[args.sub_task] = gflags[args.sub_task]
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
            # NOTE: see note above concerning apply_filling flag
            if (hasattr(args, 'apply_filling') and args.apply_filling) or (c_idx == 0 and (args.sub_task == 'tarc-full' or args.sub_task == 'tarc-full-npos' or args.sub_task == 'tarc-lemma' or args.sub_task == 'tarc-lemma-full')) or (c_idx == 2 and args.sub_task == 'madar-trs') or (args.sub_task == 'madar-trs-ex' and c_idx == 3) or (args.sub_task == 'madar-trs-full' and c_idx == 4) or (args.sub_task == 'madar-trs-full-ex' and c_idx == 2):
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
                if args.ignore_test_output and test_flag or t in [bos_token, pad_token, eos_token, unk_token, start_token, end_token, space_token, separator_]:
                    process_res = no_processing(t)
                else:
                    rep_flag = True
                    # NOTE: see note above concerning apply_filling flag
                    if (hasattr(args, 'apply_filling') and args.apply_filling) or (c_idx == 0 and (args.sub_task == 'tarc-full' or args.sub_task == 'tarc-full-npos' or args.sub_task == 'tarc-lemma' or args.sub_task == 'tarc-lemma-full')) or (c_idx == 2 and args.sub_task == 'madar-trs') or (args.sub_task == 'madar-trs-ex' and c_idx == 3) or (args.sub_task == 'madar-trs-full' and c_idx == 4) or (args.sub_task == 'madar-trs-full-ex' and c_idx == 2):
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

def map_tokens(args, data, dict, pad_flag, split):

    tensors = []
    for s in data:
        tok_lst = []
        for t in s:
            if args.freeze_dictionary or (args.freeze_train_dictionary and split != 'train'):
                tok_lst.append( dict.index(t) )
            else:
                tok_lst.append( dict.add_symbol(t) )
        if pad_flag:
            tok_lst = [dict.bos()] + tok_lst + [dict.eos()] 
        tensors.append( torch.LongTensor( tok_lst ) )

    return tensors

def check_input_definiteness(tt, d_dict, m_dict):

    special_symbols = ['_FOREIGN_', '_SYM_', '_PUNC_', '_EMOTAG_']

    mismatches = {}
    for t in tt:
        t_str = d_dict.string(t)
        for tok in t_str.split():

            if tok in special_symbols:
                print('   * Found special symbol {} in sequence: {}'.format(tok, t_str))
                sys.stdout.flush()

            id1 = d_dict.index(tok)
            assert id1 != d_dict.unk()
            id2 = m_dict.index(tok)

            if id1 != id2:
                if tok not in mismatches:
                    mismatches[tok] = [(id1, id2)]
                else:
                    for tpl in mismatches[tok]:
                        if id1 != tpl[0] or id2 != tpl[1]:
                            mismatches[tok].append( (id1, id2) )

    return mismatches


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
        parser.add_argument('--force-input-decoding', action='store_true', default=False,
                            help='Force the model to decode the input sequence when tackling tasks where the input sequence must be predicted (e.g. constituent syntactic parsing)')
        parser.add_argument('--reverse-input', action='store_true', default=False,
                            help='Reverse input sequences (useful e.g. for seq-to-seq syntactic parsing)')
        parser.add_argument('--tree-format', type=str, default='wsj', help='Specific to constituency syntactic parsing, specifies the format of the output trees (wsj, or GAFL)')
        parser.add_argument('--input-lm', type=str, help='Load a language model for extracting representations of the input sequences')
        parser.add_argument('--lm-data', type=str, help='Serialized data used to train the language model, used to keep the same dictionary')
        parser.add_argument('--use-transformer-layers', action='store_true', default=False, help='Add Transformer layers in the LSTM encoder')
        parser.add_argument('--transformer-layers', type=int, default=3, help='Number of Transformer layers in the LSTM encoder')
        parser.add_argument('--load-transformer-layers', type=str, help='Load pre-trained transformer layers in the LSTM encoder')
        parser.add_argument('--freeze-transformer-layers', action='store_true', default=False, help='Freeze pre-trained Transformer layers in LSTM encoder')
        parser.add_argument('--apply-filling', action='store_true', default=False, help='Apply filler substitution whatever the subtask')
        parser.add_argument('--save-embeddings', action='store_true', default=False, help='Save the token embeddings of the best model')
        parser.add_argument('--load-dictionary', type=str, help='Load a pre-defined dictionary from the specified file')
        parser.add_argument('--load-embeddings', type=str, help='Load embeddings from the specified file')
        parser.add_argument('--freeze-dictionary', action='store_true', default=False, help='Don\'t add symbols from additionally read data to the dictionary')
        parser.add_argument('--freeze-train-dictionary', action='store_true', default=False, help='Construct the dictionary only from training data tokens')

    @classmethod
    def setup_task(cls, args, **kwargs):
        
        # Here we can perform any setup required for the task. This may include
        # loading Dictionaries, initializing shared Embedding layers, etc.
        # In this case we'll just initialize the label dictionary
       
        
        #blank_token = "__"
        #pad_char = pad_token 
        #machine_semantic = 'MachineSemantic'
        #slu_start_concept_mark = '_SOC_'
        #slu_end_concept_mark = '_EOC_'
        #tok_separator = '|'
        #user_ID = 'User'
        #machine_ID = 'Machine'
        #bogus_ID = '_Bogus_'
        #EOD_tag = '_EOD_'

        #input_vocab = Dictionary(
        #                         pad=pad_token,
        #                         eos=eos_token,
        #                         unk=unk_token,
        #                         bos=bos_token,
        #                         extra_special_symbols=[start_token, end_token, args.sequence_separator, blank_token, machine_semantic, slu_start_concept_mark, slu_end_concept_mark, tok_separator, user_ID, machine_ID, bogus_ID, EOD_tag]
        #)
        input_vocab = init_slu_dictionary(args) # NOTE: defined in fairseq.tasks.End2EndSLU
        if hasattr(args, 'load_dictionary') and args.load_dictionary:

            if args.serialized_data is not None and os.path.exists(args.serialized_data + '.train') and os.path.isfile(args.serialized_data + '.train'):
                print('')
                print(' ######################################################################################################################################################')
                print(' ### TArCMultiTask CRITICAL WARNING: loading pre-defined dictionary, did you regenerate serialized data with this dictionary to have correct indexing ?')
                print(' ######################################################################################################################################################')
                print('')
                sys.stdout.flush()

            input_vocab = torch.load(args.load_dictionary)
        output_vocab = input_vocab

        if hasattr(args, 'input_lm') and args.input_lm:
            if not args.lm_data:
                raise ValueError('LM data are mandatory when using a language model to encode input sequences')

            lm_data = torch.load(args.lm_data)
            input_vocab = lm_data['vocab']
            output_vocab = input_vocab

            if args.serialized_data is not None and os.path.exists(args.serialized_data + '.train') and os.path.isfile(args.serialized_data + '.train'):
                print('')
                print(' ######################################################################################################################################################')
                print(' ### TArCMultiTask CRITICAL WARNING: loading pre-defined dictionary, did you regenerate serialized data with this dictionary to have correct indexing ?')
                print(' ######################################################################################################################################################')
                print('')
                sys.stdout.flush()

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
        self.input_dict = {} if args.force_input_decoding else None
        self.output_dict = {} if args.force_input_decoding else None
        self.punct_dict = {} if args.force_input_decoding else None
        self.inverse_input_dict = {} if self.input_dict is not None else None
        self.inverse_output_dict = {} if self.output_dict is not None else None

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        if hasattr(args, 'lm_data') and args.lm_data:
            #self.lm_vocab = Dictionary(
            #                     pad=pad_token,
            #                     eos=eos_token,
            #                     unk=unk_token,
            #                     bos=bos_token,
            #                     extra_special_symbols=[start_token, end_token, args.sequence_separator]
            #)
            self.lm_vocab = init_slu_dictionary(args)
            self.lm_vocab.update( input_vocab )
 
        self.sequence_separator = self.input_vocab.index(args.sequence_separator)
        self.token2components_tsr = []
        self.granularity_merging_flags = {}
        self.double_learning = args.double_learning 
        self.splits = {}
        #if hasattr(args, 'load_dictionary') and args.load_dictionary:
        #    self.splits['vocab'] = input_vocab
        self.num_of_inputs = 1 

    def set_granularity_merging_flags(self, g_flags):
        self.granularity_merging_flags = g_flags

    def get_granularity_merging_flags(self):
        return self.granularity_merging_flags

    def _t2c_to_tsr(self, t2c: Dict[str, List[str]], dict: SLUDictionary) -> Dict[int, torch.LongTensor]:
 
        res_dict = {}
        for k in t2c.keys():
            res_dict[dict.add_symbol(k)] = torch.LongTensor( [dict.add_symbol(v) for v in t2c[k]] )
        return res_dict

    def load_dataset(self, split, **kwargs):
    
        """Load a given dataset split (e.g., train, valid, test)."""

        my_split = split
        if my_split == 'valid':
            my_split = 'dev'
 
        if self.args.serialized_data is not None and os.path.exists(self.args.serialized_data + '.' + my_split) and os.path.isfile(self.args.serialized_data + '.' + my_split):
            print(' - TArCMultiTask, reading serialized data from {} ...'.format(self.args.serialized_data + '.' + my_split))
            sys.stdout.flush()
            self.splits[my_split] = torch.load( self.args.serialized_data + '.' + my_split )
            if 'vocab' not in self.splits:
                print('   * reading dictionary ...')
                sys.stdout.flush()

                if not (hasattr(self.args, 'load_dictionary') and self.args.load_dictionary):
                    self.input_vocab = torch.load( self.args.serialized_data + '.vocab' ) #self.splits['vocab']
                self.output_vocab = self.input_vocab

                print('   * reading token-to-components map ...')
                sys.stdout.flush()

                self.token2components_tsr = torch.load(self.args.serialized_data + '.token2components' ) #self.splits['token2components']
                self.splits['vocab'] = self.input_vocab
                self.splits['token2components'] = self.token2components_tsr
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
                    tk_idx = 0
                    #if not self.args.token_sequences:
                    #    tk_idx = 1
                    tok_tt = map_tokens(self.args, data_sequences[tk_idx][d_idx], self.input_vocab, self.args.pad_reference, my_split)
                    tok_tensors.append( tok_tt )
                    tok_ll = torch.LongTensor( [t.size(0) for t in tok_tt] )
                    tok_lengths.append( tok_ll )
                       
                    seq_tt = []
                    for s in data_sequences[2][d_idx]:
                        if self.args.pad_reference:
                            s = [(p[0]+1, p[1]+1) for p in s]
                        char_seq_lengths[d_idx].append( torch.LongTensor( s ) )    # As many pairs as tokens...

                    ch_idx = 1
                    #if not self.args.char_sequences:
                    #    ch_idx = 0
                    tt = map_tokens(self.args, data_sequences[ch_idx][d_idx], self.input_vocab, self.args.pad_reference, my_split)
                    tensors.append( tt )
                    ll = torch.LongTensor([t.size(0) for t in tt])
                    lengths.append(ll) 
                self.splits[my_split] = ([tok_tensors, tensors], [tok_lengths, lengths, char_seq_lengths])
 
            for t_idx in range(len(token2components)): 
                self.token2components_tsr.append( self._t2c_to_tsr(token2components[t_idx], self.input_vocab) ) 

            self.splits['vocab'] = self.input_vocab
            self.splits['token2components'] = self.token2components_tsr 
            if self.args.serialized_data is not None: 
                print(' - TArCMultiTask, serializing data...')
                sys.stdout.flush()
                for spt in self.splits.keys():
                    print('   * Saving {}'.format( self.args.serialized_data + '.' + spt ))
                    sys.stdout.flush()
                    torch.save(self.splits[spt], self.args.serialized_data + '.' + spt)

        assert my_split in self.splits.keys() and 'vocab' in self.splits.keys() and 'token2components' in self.splits.keys()
        print(' - TArCMultiTask, instantiating current split {}'.format(split))
        sys.stdout.flush() 

        tensors, lengths = self.splits[my_split]
        if hasattr(self.args, 'lm_data') and self.args.lm_data:
            print(' * TArCMultiTask, checking input symbols definiteness (model dict size {}, vs. LM dict size {})...'.format(len(self.input_vocab), len(self.lm_vocab)))
            sys.stdout.flush()
            if self.args.token_sequences:
                print('   * checking token definiteness...')
                sys.stdout.flush()
                miss = check_input_definiteness(tensors[0][0], self.input_vocab, self.lm_vocab)
                if len(miss) > 0:
                    raise ValueError('There are {} undefined symbols in the language model dictionary: {}'.format(len(miss), miss))
            if self.args.char_sequences:
                print('   * checking character definiteness...')
                sys.stdout.flush()
                miss = check_input_definiteness(tensors[1][0], self.input_vocab, self.lm_vocab)
                if len(miss) > 0:
                    raise ValueError('There are {} undefined symbols in the language model dictionary: {}'.format(len(miss), miss))

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
  
        if self.args.sub_task not in granularity_merging_flags:
            _, gflags = choose_column_processing(num_of_tasks+1, self.args)
            if gflags is not None:
                granularity_merging_flags[self.args.sub_task] = gflags[self.args.sub_task]
            check_column_processing(num_of_tasks+1, self.args)
        self.set_granularity_merging_flags(granularity_merging_flags[self.args.sub_task]) 
 
        if self.input_dict is not None:
            self.inverse_input_dict[self.input_vocab.bos()] = '<bos>'
            self.inverse_input_dict[self.input_vocab.eos()] = '<eos>'

            for tok in punct:
                t_idx = self.input_vocab.index(tok)
                if t_idx == self.input_vocab.unk():
                    print(' - TArCMultiTask WARNING: punctuation {} is not defined in the dictionary'.format(tok))
                    sys.stdout.flush()
                if t_idx not in self.punct_dict:
                    self.punct_dict[t_idx] = 1
            add_indices_to_dict(self.input_dict, tensors[0][0])
            add_entries_to_inverse_dict(self.inverse_input_dict, self.input_vocab, tensors[0][0])
            if my_split != 'train':
                trn_tensors, trn_lenghts = self.splits['train']
                add_indices_to_dict(self.input_dict, trn_tensors[0][0])
                add_entries_to_inverse_dict(self.inverse_input_dict, self.input_vocab, trn_tensors[0][0])

            '''for t in tensors[0][0]:
                tmp_str = self.input_vocab.string(t)
                for tk in tmp_str.split():
                    if self.input_vocab.index(tk) not in self.inverse_input_dict:
                        print(' *** Associating {} -> {}'.format(self.input_vocab.index(tk), tk))
                        sys.stdout.flush()
                        self.inverse_input_dict[self.input_vocab.index(tk)] = tk
            if my_split != 'train':
                trn_tensors, trn_lengths = self.splits['train']
                for t in trn_tensors[0][0]:
                    tmp_str = self.input_vocab.string(t)
                    for tk in tmp_str.split():
                        if self.input_vocab.index(tk) not in self.inverse_input_dict:
                            print(' *** Associating {} -> {}'.format(self.input_vocab.index(tk), tk))
                            sys.stdout.flush()
                            self.inverse_input_dict[self.input_vocab.index(tk)] = tk'''

        if self.output_dict is not None:
            self.inverse_output_dict[self.input_vocab.bos()] = '<bos>'
            self.inverse_output_dict[self.input_vocab.eos()] = '<eos>'

            if self.args.tree_format == 'wsj':
                for ii in range(bound_idx, len(tensors[0][bound_idx:])+1):
                    if ii == len(tensors[0][bound_idx:]):
                        create_output_dict_from_trees(self.output_dict, self.input_vocab, self.args.tree_format, tensors[0][ii])
                        add_entries_to_inverse_dict_from_trees(self.inverse_output_dict, self.input_vocab, self.args.tree_format, tensors[0][ii])
                    else:
                        add_indices_to_dict(self.output_dict, tensors[0][ii])
                        add_entries_to_inverse_dict(self.inverse_output_dict, self.input_vocab, tensors[0][ii])
            else:
                create_output_dict_from_trees(self.output_dict, self.input_vocab, self.args.tree_format, tensors[0][-1])
                add_entries_to_inverse_dict_from_trees(self.inverse_output_dict, self.input_vocab, self.args.tree_format, tensors[0][-1])
 
            print(' - TArCMultiTask, created input dictionary of size {}'.format(len(self.input_dict)))
            print(' - TArCMultiTask, created output dictionary of size {}'.format(len(self.output_dict)))
            print(' - TArCMultiTask, created inverse input dictionary of size {}'.format(len(self.inverse_input_dict)))
            print(' - TArCMultiTask, created inverse output dictionary of size {}'.format(len(self.inverse_output_dict)))
            sys.stdout.flush()

        sources = [tensors[0][0:bound_idx], tensors[1][0:bound_idx]]
        if self.args.reverse_input:
            print(' - TArCMultiTask: reversing input sequences...')
            sys.stdout.flush()

            for ii in range(len(sources[0])):
                for t_idx in range(len(sources[0][ii])):
                    sources[0][ii][t_idx] = sources[0][ii][t_idx].flip(dims=[0])
                    #sources[1][ii][t_idx] = sources[1][ii][t_idx].flip(dims=[0])
                    

        src_lengths = [lengths[0][0:bound_idx], lengths[1][0:bound_idx], lengths[2][0:bound_idx]]
        targets = [tensors[0][bound_idx:], tensors[1][bound_idx:]] 
        tgt_lengths = [lengths[0][bound_idx:], lengths[1][bound_idx:], lengths[2][bound_idx:]]

        print(' - Tarc MultiTask, learning with {} input(s) (lengths: {}), {} different outputs (num. of tasks: {}, lengths: {})'.format(len(sources), len(src_lengths), len(targets), self.args.num_of_tasks, len(tgt_lengths)))
        sys.stdout.flush() 

        #print(' -----')
        #print(' * First token sequence: {}'.format(self.input_vocab.string(sources[0][0][0])))
        #print(' -----')
        #print(' * First char sequence: {}'.format(self.input_vocab.string(sources[1][0][0])))
        #print(' -----') 

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

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""

        if epoch > 1 and hasattr(self.args, 'save_embeddings') and self.args.save_embeddings:
            print('[DEBUG] * TArCMultiTask: saving current model embeddings')
            sys.stdout.flush()

            torch.save(model.encoder.encoders[0].embed_tokens, self.args.serialized_data + '.emb')

    def build_generator(self, args):
        
        from fairseq.tarc_multitask_sequence_generator import TarcSequenceGenerator 

        return TarcSequenceGenerator(
            self.target_dictionary,
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
        )



























