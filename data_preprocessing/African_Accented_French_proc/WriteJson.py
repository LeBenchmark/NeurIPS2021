'''

Sina ALISAMIR
2020-2021
'''

import os, glob
from funcs import get_files_in_path, printProgressBar
from pydub import AudioSegment
import json
import csv

def main():
    wavsPath = "../wavs/**/"
    theFiles = get_files_in_path(wavsPath)
    
    allFilesInfo = {}
    for i, filePath in enumerate(theFiles):
        audio_file = AudioSegment.from_wav(filePath)
        if audio_file.duration_seconds < 1 or audio_file.duration_seconds > 30: continue
        fileName, myDict = makeDict(filePath)
        allFilesInfo[fileName] = myDict
        printProgressBar(i + 1, len(theFiles), prefix = 'Processing Files:', suffix = 'Complete')
    with open('../data.json', 'w') as fp:
        json.dump(allFilesInfo, fp,  indent=4, ensure_ascii=False)
            
def makeDict(filePath):
    fileName = os.path.basename(filePath)[:-4]
    trans = getTranscript(filePath)
    spk_id = ""
    gender = "U"
    if "_m_" in filePath.split(os.path.sep)[-2]: gender = "M"
    if "_f_" in filePath.split(os.path.sep)[-2]: gender = "F"
    audio_file = AudioSegment.from_wav(filePath)
    duration = audio_file.duration_seconds
    myDict = {
        "path"       : filePath[1:],
        "trans"      : trans,
        "duration"   : duration,
        "spk_id"     : spk_id,
        "spk_gender" : gender
    }
    return fileName, myDict

def getTranscript(wavPath):
    fileName = os.path.basename(wavPath)[:-4]
    # print(wavPath)
    transcript = ""
    if wavPath.split(os.path.sep)[2] == "dev":
        transPath = "../transcripts/dev/niger_west_african_fr/transcripts.txt"
        with open(transPath, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            transList = list(spamreader)
        for i, trans in enumerate(transList):
            transWav = os.path.basename(trans[0])[:-4]
            if transWav == fileName:
                trans = ' '.join(trans[1:])
                transcript = trans
    elif wavPath.split(os.path.sep)[2] == "devtest":
        transPath = "../transcripts/devtest/ca16_read/conditioned.txt"
        with open(transPath, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            transList = list(spamreader)
        for i, trans in enumerate(transList):
            transWav = os.path.basename(trans[0])
            if transWav == fileName:
                trans = ' '.join(trans[1:])
                transcript = trans
    elif wavPath.split(os.path.sep)[2] == "test":
        transPath = "../transcripts/test/ca16/prompts.txt"
        with open(transPath, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            transList = list(spamreader)
        for i, trans in enumerate(transList):
            transWav = os.path.basename(trans[0])
            if transWav == fileName:
                trans = ' '.join(trans[1:])
                transcript = trans
    elif wavPath.split(os.path.sep)[2] == "train":
        subFolder = wavPath.split(os.path.sep)[3]
        baseWav = os.path.basename(wavPath)[:-4]
        # print(subFolder, baseWav)
        if subFolder == "ca16":
            if "conv" in baseWav: 
                transPath = "../transcripts/train/ca16_conv/transcripts.txt"
            else:
                transPath = "../transcripts/train/ca16_read/conditioned.txt"
        else:
            transPath = "../transcripts/train/yaounde/fn_text.txt"
        with open(transPath, newline='\n') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ')
            transList = list(spamreader)
        for i, trans in enumerate(transList):
            if subFolder == "yaounde":
                subsubFolder = wavPath.split(os.path.sep)[4]
                if not subsubFolder in trans[0]: continue
            transWav = os.path.basename(trans[0])
            if transWav[-4] == '.': transWav = transWav[:-4]
            if transWav == fileName:
                trans = ' '.join(trans[1:])
                transcript = trans
    return transcript

if __name__== "__main__":
    main()