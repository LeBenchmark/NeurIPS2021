#!/usr/bin/python3
"""
Date: 04/05/2021
Author: Solène

Construction du fichier json pour le corpus "voxpopuli", unlabelled + transcribed data:
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

from progress.bar import Bar,ChargingBar
import soundfile as sf
import os

parser=argparse.ArgumentParser(prog='PROG', description=("\nConstruction du fichier json pour le corpus 'voxpopuli', unlabelled or transcribed data\n"
			"INput: fr folder (string) \n"
			"OUTput: data.json"),
					usage='%(prog)s directory output_file\n \n')
parser.add_argument('directory', help='data folder name', type=str)
parser.add_argument('output_file', help='data.json (without extension)', type=str)
args=parser.parse_args() #args.stage

jsonfilename=args.directory+'/'+args.output_file+'.json'
data={}
#segment_dict={}
#gender_file={}
count=True
counter=0
countfile=0

print("ETAPE 1: on récolte les données \(nom de fichier, durée, spk_id...\)")
#parcourir les dossiers par année depuis "wav"
#|unlabelled_data
#   |-fr
#       |-wav
#           |-2020
#           |-2019
#           |-...
#               |file.wav


for subdir, dirs, files in os.walk("./"):
    if count==True:
        numDirs=len(dirs)
        dirBar= Bar('Processing '+str(numDirs)+' directories ',max=numDirs) # on créé la barre des dirs
        count=False
          
#for i in range(numDirs): #pour chaque dir
for subdir, dirs, files in os.walk("./"):
    if counter >=1:
        #print(len(files)) # on regarde le nbre de fichiers
        print("")
        bar=Bar('Processing files', max=len(files)) #on créé une barre des fichiers

        for file in files: #lire ses fichiers
            if file.endswith(".wav"):
                file_path="./"+file[0:4]+"/"+file
                f=sf.SoundFile(file_path)
                sound_size=len(f)
                samplerate=f.samplerate
                duration=sound_size/samplerate
                filename=os.path.splitext(file)
                data[filename[0]]={'path': file_path,
                                'duration': duration,
                            'spk_id': '',
                            'spk_gender': 'U',
                            'trans': ''}
                bar.next() # on avance la barre des fichiers
        bar.finish()
        dirBar.next() # on avance la barre des dirs
    counter+=1

dirBar.finish()
##################################################
# on écrit dans le json
print("Etape 2: écrire dans le json")
with open(jsonfilename, mode='w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2, separators=(',', ': '))

##################################################
# #on lit le fichier segments
# segments_file=open(args.directory+'/segments.txt', mode='r')
# for segment in segments_file:
#     segment=segment.split('\t')
#     audio_name=segment[0]
#     start_time=segment[2]
#     end_time=segment[3]
#     duration=float(end_time)-float(start_time)
#     #duration=Decimal(end_time)-Decimal(start_time)
#     #duration=float(round(duration,2)) --> pour arrondir les durées à 0,xx
#     segment_dict[audio_name]=duration
# segments_file.close()

# #on lit le fichier meta-info
# meta_file=open(file="metainfo.txt",mode='r')
# find_data=re.compile(r"(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\|(.*?)\n")
# for line in meta_file:
#     if find_data.match(line):
#         found_line=find_data.match(line)

#         ## récupération des groupes
#         spk=found_line.group(1).strip()
#         gender=found_line.group(2).strip()
#         gender_file[spk]=gender
# meta_file.close()

# #on lit le fichier transcripts
# transcript_file=open(args.directory+'/transcripts.txt', mode='r')
# for line in transcript_file:
#     line=line.split('\t')
#     audioFile_ID=line[0]
#     speaker_ID=audioFile_ID.split('_')[0]
#     book_ID=audioFile_ID.split('_')[1]
#     transcription=line[1]
#     if audioFile_ID in segment_dict:
#         duration=segment_dict[audioFile_ID]
#     if speaker_ID in gender_file:
#         gender=gender_file[speaker_ID]
#     data[audioFile_ID]={
#         'path': './audio/'+speaker_ID+'/'+book_ID+'/'+audioFile_ID+'.wav',
#         'trans': transcription.strip(),
#         'duration': duration,
#         'spk_id': speaker_ID,
#         'spk_gender' : gender
#     }
# transcript_file.close()





