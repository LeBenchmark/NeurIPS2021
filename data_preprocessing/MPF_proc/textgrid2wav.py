#!/usr/bin/python3
"""
Date: 05/02/2021
Author: Marcely 

"""



import sys, glob
from utils import *


def get_ids(textgrid):
    non_tiers = ["Silences", "silences", "Silence", "silence", "Commentaires", "com", "Com", "COM", "commentaires", "Commentaire"]
    ids = textgrid.tierNameList
    for n_tier in non_tiers:
        try:
            ids.remove(n_tier)
        except ValueError:
            pass
    return ids

def is_valid_transcription(text):
    '''      intervals [700]:
            xmin = 2564.9758850514654
            xmax = 2566.096048978448
            text = "(Rires)."
        intervals [692]:
            xmin = 2548.932246871463
            xmax = 2549.7272019164184
            text = "<Et euh>."
        intervals [693]:
            xmin = 2549.7272019164184
            xmax = 2551.1364404052024
            text = ""
        intervals [694]:
            xmin = 2551.1364404052024
            xmax = 2556.050707955834
            text = "Ah ouais donc tu aimes bien euh comment ça s'appelle (.) qui passe sur la une là les Experts."
    three examples of intervals, the goal is to remove the first three
    '''
    if text == "" or text == "(Rire)" or ((text[0] == "(") and (text[-2:] == ").")) or ((text[0] == "<") and (text[-2:] == ">.")):
        #catches empty transcriptions or trannscriptions in the format (text). and <text>. 
        return False
    return True

def process_pairs(wav_folder, textgrid_folder, output_folder, json_file_name):
    wav_paths = glob.glob(wav_folder + "/*")
    for wav_path in wav_paths:
        file_prefix = wav_path.split("/")[-1].split(".")[0]
        textgrid = load_textgrid(textgrid_folder + "/" + file_prefix + ".TextGrid")
        speakers = get_ids(textgrid)
        audio = load_wav(wav_path)
        for speaker in speakers:
            speaker_segments = textgrid.tierDict[speaker].entryList #praatio.tgio.IntervalTier list of Intervals
            sentence_id = 1
            for segment in speaker_segments: 
                # e.g. Interval(start=432.64378279278293, end=434.1553833260748, label='&= juron.')
                trans = segment.label
                if is_valid_transcription(trans):
                    start = float(segment.start)
                    end = float(segment.end)
                    segment_id = file_prefix + '_s_' + str(sentence_id) + "_spk_" + speaker
                    duration = end - start
                    if duration > 1 and duration < 30:
                        segment_audio = slice_wav(start, end, audio)
                        segment_audio_path = output_folder + "/" + segment_id + ".wav"
                        write_wav(segment_audio_path, segment_audio)
                        json_entry = create_json_entry(segment_audio_path, trans, duration, speaker, "U")
                        write_json_entry(output_folder, json_file_name, segment_id, json_entry)
                        sentence_id += 1
                    else:
                        with open("removed_sentences.tsv","a") as log_file:
                            log_file.write("\t".join([str(duration),trans, segment_id]) + "\n")


 

if __name__ == "__main__":
    process_pairs(sys.argv[1], sys.argv[2], sys.argv[3], "MPF_FULL.json")

