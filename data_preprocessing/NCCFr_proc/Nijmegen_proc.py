#!/usr/bin/python3
"""
Date: 09/02/2021
Author: Marcely 
"""

import argparse, glob
from utils import *

def retrieve_speaker_info(file_name):
    d = dict()
    with open(file_name) as csv_file:
        for line in csv_file:
            line = line.split(",")
            spk_id = line[0]
            if spk_id != "ID": #HEADER
                gender = line[1] if len(line[1]) > 0 else "U"
                d[spk_id] = gender
    return d

def filter_ids(tg_ids):
    #print(tg_ids)
    filtered = list()
    for tg_id in tg_ids:
        if not tg_id == "background":
            filtered.append(tg_id)
    return filtered

def process(args):
    spk_dict = retrieve_speaker_info(args.metadata_file)
    #print(spk_dict)
    wav_files = glob.glob(args.wav_folder + "/*")
    for wav_file in wav_files:
        file_prefix = wav_file.split("/")[-1].split(".wav")[0]
        textgrid = load_textgrid(args.textgrids_folder + "/" + file_prefix + ".TextGrid")
        speakers = filter_ids(textgrid.tierNameList)
        for speaker in speakers:
            speaker_segments = textgrid.tierDict[speaker].entryList #praatio.tgio.IntervalTier list of Intervals
            sentence_id = 1
            audio = load_wav(wav_file)
            for segment in speaker_segments: 
                # e.g. Interval(start=432.64378279278293, end=434.1553833260748, label='&= juron.')
                trans = segment.label
                start = float(segment.start)
                end = float(segment.end)
                segment_id = file_prefix + '_s_' + str(sentence_id) + "_spk_" + speaker
                duration = end - start
                #print(duration)
                if duration > 1 and duration < 30 :
                    #print("accepted")
                    segment_audio = slice_wav(start, end, audio)
                    segment_audio_path = args.output_folder + "/" + segment_id + ".wav"
                    write_wav(segment_audio_path, segment_audio)
                    try:
                        json_entry = create_json_entry(segment_audio_path, trans, duration, speaker, spk_dict[speaker])
                    except KeyError:
                        json_entry = create_json_entry(segment_audio_path, trans, duration, speaker, "U")
                    write_json_entry(args.output_folder, args.json_file_name, segment_id, json_entry)
                    sentence_id += 1
                else:
                    with open("removed_sentences.tsv","a") as log_file:
                        log_file.write("\t".join([str(duration),trans, segment_id]) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-folder', help="Folder containing the wav files")
    parser.add_argument('--textgrids-folder', help="Folder containing the textgrid files")
    parser.add_argument('--metadata-file', help="File containing speaker information")
    parser.add_argument('--output-folder', help="Output folder for json files")
    parser.add_argument('--json-file-name', help="Output folder for json files")
    args=parser.parse_args()
    process(args)
