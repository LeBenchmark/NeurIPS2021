#!/usr/bin/python3
"""
Date: 15/12/2020
Author: Solène

Construction des fichiers json pour les train/dev/test du corpus MLS comprenant:
- le chemin relatif des fichiers sons
- la transcription
- la durée du son
- le speaker ID """

import json
import argparse
import subprocess
from subprocess import Popen, PIPE
import time
from decimal import *
import re

parser=argparse.ArgumentParser(prog='PROG', description=("\nConstruction des fichiers json pour les train/dev/test du corpus MLS\n"
			"INput: {train,dev or test} folder (string) \n"
			"OUTput: {train,dev,test}/file.json"),
					usage='%(prog)s directory output_file\n \n')
parser.add_argument('directory', help='train, dev or test folder name', type=str)
parser.add_argument('output_file', help='output_file name without extension', type=str)
args=parser.parse_args() #args.stage

filename=args.directory+'/'+args.output_file+'.json'
data={}
segment_dict={}
gender_file={}


#on lit le fichier segments
segments_file=open(args.directory+'/segments.txt', mode='r')
for segment in segments_file:
    segment=segment.split('\t')
    audio_name=segment[0]
    start_time=segment[2]
    end_time=segment[3]
    duration=float(end_time)-float(start_time)
    #duration=Decimal(end_time)-Decimal(start_time)
    #duration=float(round(duration,2)) --> pour arrondir les durées à 0,xx
    segment_dict[audio_name]=duration
segments_file.close()

#on lit le fichier meta-info
meta_file=open(file="metainfo.txt",mode='r')
find_data=re.compile(r"(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\n")
for line in meta_file:
    if find_data.match(line):
        found_line=find_data.match(line)

        ## récupération des groupes
        spk=found_line.group(1).strip()
        gender=found_line.group(2).strip()
        gender_file[spk]=gender
meta_file.close()

#on lit le fichier transcripts
transcript_file=open(args.directory+'/transcripts.txt', mode='r')
for line in transcript_file:
    line=line.split('\t')
    audioFile_ID=line[0]
    speaker_ID=audioFile_ID.split('_')[0]
    book_ID=audioFile_ID.split('_')[1]
    transcription=line[1]
    if audioFile_ID in segment_dict:
        duration=segment_dict[audioFile_ID]
    if speaker_ID in gender_file:
        gender=gender_file[speaker_ID]
    data[audioFile_ID]={
        'path': './audio/'+speaker_ID+'/'+book_ID+'/'+audioFile_ID+'.wav',
        'trans': transcription.strip(),
        'duration': duration,
        'spk_id': speaker_ID,
        'spk_gender' : gender
    }
transcript_file.close()



with open(filename, mode='w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2, separators=(',', ': '))

