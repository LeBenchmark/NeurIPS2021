"""
Date: 08/02/2021
Author: Marcely 
"""

from utils import *
import glob, argparse

SUFFIX =  "_one_channel.wav"

def get_transcription(file_path):
    return [line.strip("\n") for line in open(file_path)][0]

def process(args):
    ids = glob.glob(args.transcription_folder + "/*")
    for transcription_id in ids:
        transcription = get_transcription(transcription_id)
        s_id = transcription_id.replace(".txt", "").split("/")[-1]
        audio = load_wav(args.wav_folder + "/" + s_id + SUFFIX)
        duration = audio.duration_seconds
        if 1 <= duration < 30 :
            wav_file_path = args.output_folder + "/" + s_id + ".wav"
            write_wav(wav_file_path, audio)
            new_entry = create_json_entry(wav_file_path, transcription, duration, "NONE", "U")
            write_json_entry(args.output_folder, args.json_file_name, s_id, new_entry)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav-folder', help="Folder containing the files")
    parser.add_argument('--transcription-folder', help="Folder containing the metadata")
    parser.add_argument('--output-folder', help="Output folder for json files")
    parser.add_argument('--json-file-name', help="Output folder for json files")
    args=parser.parse_args()
    process(args)


