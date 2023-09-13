
import sys
import os
from glob import iglob
import h5py
import scipy.io.wavfile
import tqdm

TRAINTXTFILE = "/home/getalp/dinarelm/work/data/DSTC/train.tts-verbatim.2022-07-27.txt"
VALTXTFILE = "/home/getalp/dinarelm/work/data/DSTC/dev-dstc11.tts-verbatim.2022-07-27.txt"
TESTTXTFILE="/home/getalp/dinarelm/work/data/DSTC/test.turn.unseen.12h.v2.txt"

def pcm2wav(filepath):
    """
    Given a filepath to a h5p file (.hd5) we read the pcm audio of each turn and convert it to wav format to hear it.
    """
    data = h5py.File(filepath, 'r')
    for group in list(data.keys()):
        # For each group (=turn?) we select the pcm audio and convert it to wav
        audio_pcm = data[group]['audio'][:]
        # FIXME: the turn id's index is strangely handled, seems to be an offset of one except on the first turn
        turn = group.split(' ')[-1]
        dialogue = group.split(' ')[-3].split('.')[0]
        #if int(turn) > 1:
        #    turn = str(int(turn) - 1)
        #print("Converting " + group + " : " + data[group].attrs['hyp'])
        newPath = os.path.join(os.path.dirname(filepath), dialogue + ".Turn-" + turn + "-User.wav")
        scipy.io.wavfile.write(newPath, 16000, audio_pcm)

def get_ids(splits: list):
    dialogue = splits[3].split(".")[0].strip()  # Basically, it removes the .json extension
    
    # NOTE: see the FIXME in pcm2wav for understanding why turn id is computed this way
    #turn = "1" if splits[5] == "1" else str(int(splits[5]) - 1) 
    #turn = turn.strip()

    turn = splits[5].strip()

    return dialogue, turn

def clean_sem(sem: str):

    return sem.replace("=", " ").replace(";", "").strip()

machine_marker = '_machine_'
machine_sem = machine_marker + '=void; void=' + machine_marker
machine_sem_noval = machine_marker + ' void void ' + machine_marker
punctuation = ['.', ',', ';', ':', '!', '?']

def get_trs_and_sem(splits: list):

    if 'state:' in splits[-1]:
        transcript = splits[-1].split("state:")[0][6:]
        if len(transcript) == 0 or transcript == ' ':
            transcript = '<empty>'

        #print('[DEBUG] removing punctuation from: {}'.format(transcript))
        #sys.stdout.flush()

        for p in punctuation:
            transcript = transcript.replace(p + ' ', ' ' + p + ' ')
        if transcript[-1] in punctuation and transcript[-2] != ' ':
            transcript = transcript + ' '
            transcript[-1] = transcript[-2]
            transcript[-2] = ' '
        transcript = transcript.replace('\'', ' \'')
        transcript = transcript.replace('  ', ' ')
        transcript = transcript.strip() 
        sem = splits[-1].split("state:")[-1].strip()
    else:
        transcript = ''
        sem = ''

    if len(transcript) == 0:
        transcript = 'void'
    if len(sem) == 0:
        sem = 'void=void'

    return transcript, sem

def sem_dict_to_str(sd):

    res = ''
    for k, v in sd.items():
        res = res + k + '=' + v + '; '
    return res[:-2]

def get_sem_dict(a):

    #print('[DEBUG] get_sem_dict, extracting dict from: {}'.format(a))
    #sys.stdout.flush()

    sd = {}
    for tt in a.strip().split(';'):
        assert '=' in tt, 'Wrong annotation format: {}'.format(tt)
        k, v = tt.strip().split('=')
        sd[k] = v
    return sd

def dict_diff(d1, d2, val_flag):  # NOTE: set(d1) ^ set(d2) does not seem to work correctly

    diffd = {}
    for k in d1.keys():
        if k not in d2 or (not val_flag and d2[k] != d1[k]):
            diffd[k] = d1[k]
    return diffd

def remove_values(sem):

    sd = get_sem_dict(sem)
    for k in sd.keys():
        sd[k] = ''
    return sem_dict_to_str(sd)

def remove_dialog_history(a, data, did, tid, spk, val_flag):

    if spk == 'agent:':
        return a
    else:   
        ctd = get_sem_dict(a)

        #print('[DEBUG] remove_dialog_history, current turn sem dict: {}'.format(ctd))
        #sys.stdout.flush()

        htd = {}
        prev_tid = int(tid)-1
        for td in data[did]:
            if td['spk'] == 'user:': 
                htd.update( get_sem_dict( td['sem'] ) ) 

        #print('[DEBUG] remove_dialog_history, history turns sem dict: {}'.format(htd))
        #sys.stdout.flush()

        if len(htd) > 0: 
            diffd = dict_diff(ctd, htd, val_flag)
            filled = False

            #print('[DEBUG] remove_dialog_history, difference dict: {}'.format(diffd))
            #sys.stdout.flush()

            if len(diffd) == 0:
                diffd = {'void': 'void'}
                filled = True 
            new_sem = sem_dict_to_str(diffd)

            #print('[DEBUG] remove_dialog_history, final sem: {}'.format(new_sem))
            #sys.stdout.flush()

            return new_sem
        else:
            return a


def turn_transcripts_and_sem(TXTFILE: str, mode="train", tts_id="tpa", save_turns=True):

    remove_history = True  # If set to True, remove history from the turn annotation, so that each turn's annotation is made of the current turn annotation only.
    remove_values_flag = False

    #if mode=='train':
    dialog_data = {}
    complete_dialog_data = {}
    with open(TXTFILE, 'r') as fullDialogues:
        for line in fullDialogues:
            # The file is formated as line_nr: X dialog_id: Y turn_id: Z text: user: A state: B
            # We limit the number of splits to 7 to not split the text
            splits = line.split(' ', 7)
            if splits[0] != 'END_OF_DIALOG\n':
                dialogue, turn = get_ids(splits)

                #print('[DEBUG] dialog, turn: {}, {}'.format(dialogue, turn))
                #print('[DEBUG] -----')
                #sys.stdout.flush()

                transcript, sem = get_trs_and_sem(splits) 

                check_tokens = splits[-1].split()
                assert splits[6] == 'text:', 'Got wrong first check token: {}'.format(splits[6])
                assert check_tokens[0] == 'user:' or check_tokens[0] == 'agent:', 'Uknown speaker in check tokens: {}'.format(check_tokens[0])

                # write the transcript in a txt file and the semantic annotation in a .sem file FIXME for now only one tts
                if mode == 'train':
                    if check_tokens[0] == 'user:':
                        new_path_transcript = os.path.join(os.path.dirname(TXTFILE), "train", tts_id, dialogue + ".Turn-" + turn + "-User.txt")
                        new_path_sem = os.path.join(os.path.dirname(TXTFILE), "train", tts_id, dialogue + ".Turn-" + turn + "-User.sem")
                    elif check_tokens[0] == 'agent:':
                        new_path_transcript = os.path.join(os.path.dirname(TXTFILE), "train", tts_id, dialogue + ".Turn-" + turn + "-Machine.txt")
                        new_path_sem = os.path.join(os.path.dirname(TXTFILE), "train", tts_id, dialogue + ".Turn-" + turn + "-Machine.sem")
                        transcript = machine_marker + ' ' + transcript + ' ' + machine_marker
                        sem = machine_sem
                        if remove_values_flag:
                            sem = machine_sem_noval
                    else:
                        raise ValueError('unknown speaker found at dialog {}, turn {}: {}'.format(dialogue, turn, check_tokens[0]))
                elif 'dev' in mode or 'test' in mode:
                    corpus_split = mode 
                    if check_tokens[0] == 'user:':
                        new_path_transcript = os.path.join(os.path.dirname(TXTFILE), corpus_split, dialogue + ".Turn-" + turn + "-User.txt")
                        new_path_sem = os.path.join(os.path.dirname(TXTFILE), corpus_split, dialogue + ".Turn-" + turn + "-User.sem") 
                    elif check_tokens[0] == 'agent:':
                        new_path_transcript = os.path.join(os.path.dirname(TXTFILE), corpus_split, dialogue + ".Turn-" + turn + "-Machine.txt")
                        new_path_sem = os.path.join(os.path.dirname(TXTFILE), corpus_split, dialogue + ".Turn-" + turn + "-Machine.sem") 
                        transcript = machine_marker + ' ' + transcript + ' ' + machine_marker
                        sem = machine_sem
                        if remove_values_flag:
                            sem = machine_sem_noval
                    else:
                        raise ValueError('Found wrong speaker at line >>>{}<<<: {}'.format(line, check_tokens[0]))

                sem_noval = None
                if remove_history and int(turn) > 1:
                    sem = remove_dialog_history(sem, complete_dialog_data, dialogue, turn, check_tokens[0], remove_values_flag)
                if remove_values_flag and check_tokens[0] == 'user:':
                    sem_noval = remove_values(sem)
                if dialogue not in dialog_data:
                    dialog_data[dialogue] = []
                    complete_dialog_data[dialogue] = []
                td = {'turn_id': int(turn)}
                td['spk'] = check_tokens[0]
                td['txt'] = transcript
                td['sem'] = sem_noval if sem_noval is not None else sem
                dialog_data[dialogue].append( td )

                td = {'turn_id': int(turn)}
                td['spk'] = check_tokens[0]
                td['txt'] = transcript
                td['sem'] = sem
                complete_dialog_data[dialogue].append(td)

                if save_turns:
                    with open(new_path_transcript, "w") as transcript_f, open(new_path_sem, "w") as sem_f:
                        print('[DEBUG] writing transcript and sem in mode {}:'.format(mode))
                        print('[DEBUG]    * {}'.format(transcript))
                        print('[DEBUG]    * {}'.format( clean_sem(sem_noval) if sem_noval is not None else clean_sem(sem) ))
                        print(' ----------')

                        transcript_f.write(transcript)
                        sem_f.write( clean_sem(sem_noval) if sem_noval is not None else clean_sem(sem) )
    return dialog_data

