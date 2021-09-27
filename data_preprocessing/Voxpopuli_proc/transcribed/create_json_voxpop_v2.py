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
parser.add_argument('--directory', help='data folder name', type=str, default="./")
parser.add_argument('--json_file', help='file.json', type=str, default="data.json")
parser.add_argument('--tsv_file', type=str, help='tsv file path', default="")
args=parser.parse_args() #args.stage

#get arguments
json_filename=args.directory+'/'+args.json_file
tsv_path=args.tsv_file


data={}
count=True
counter=0
tsv_data={}
tsv_file_exists=False
transDataNum=0

print("ETAPE 1: on récolte les données \(nom de fichier, durée, spk_id...\)")
#parcourir les dossiers par année depuis "wav"
#|(un)labelled_data
#   |-fr
#       |-wav
#           |-2020
#           |-2019
#           |-...
#           |-train.tsv
#               |file.wav

#si on a un fichier tsv (transcriptions), l'ouvrir et mettre les données dans tsv_dict
if tsv_path!="":
    tsv_file_exists=True
    tsv_file=open(tsv_path, mode='r')
for line in tsv_file:
    line=line.split("\t")
    wav_id=line[0]
    trans=line[1]
    trans_norm=line[2]
    spk_id=line[3]
    subset=line[4]
    tsv_data[wav_id]={'trans': trans_norm,
                        'spk_id': spk_id}
print("tsv dict created")


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

                if tsv_file_exists==True:
                    #on ne conserve que les données qui ont une transcription dans le tsv
                    if filename[0] in tsv_data.keys(): 
                        data[filename[0]]={'path': file_path,
                                        'duration': duration,
                                        'spk_id': tsv_data[filename[0]]['spk_id'],
                                        'spk_gender': 'U',
                                        'trans': tsv_data[filename[0]]['trans']}
                        transDataNum+=1

                if tsv_file_exists==False:
                    #on conserve toutes les données
                    data[filename[0]]={'path': file_path,
                                        'duration': duration,
                                        'spk_id': '',
                                        'spk_gender': 'U',
                                        'trans': ''}

                bar.next() # on avance la barre des fichiers
        bar.finish()
        print("num of transcribed files kept: "+str(transDataNum))
        transDataNum=0
        dirBar.next() # on avance la barre des dirs
    counter+=1

dirBar.finish()
tsv_file.close()
##################################################
# on écrit dans le json
print("Etape 2: écrire dans le json")
with open(json_filename, mode='w', encoding='utf-8') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2, separators=(',', ': '))






