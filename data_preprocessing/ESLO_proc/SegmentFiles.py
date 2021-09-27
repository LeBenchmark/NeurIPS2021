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
    wavsPath = "../wav/"
    turnPath = "../wav_turns/"
    segPath = "../train1/segments"
    utt2spkPath = "../train1/utt2spk"
    txtPath = "../train1/text"

    if not os.path.exists(turnPath): os.makedirs(turnPath)
    # theFiles = get_files_in_path(wavsPath)
    fileNames, uttIDs, times = getSegTimes(segPath)
    print(fileNames[0:3], times[0:3])
    
    allFilesInfo = {}
    for i, fileName in enumerate(fileNames):
        printProgressBar(i + 1, len(fileNames), prefix = 'Processing Files:', suffix = 'Complete')
        wavPath = os.path.join(wavsPath, fileName + ".wav")
        
        if not os.path.exists(wavPath): continue
        if times[i][2] < 1 or times[i][2] > 30: continue
        audio_file = AudioSegment.from_wav(wavPath)
        
        spkID = getSpeakerID(uttIDs[i], utt2spkPath)
        trs = getTxt(uttIDs[i], txtPath)

        segmentName = fileName+"_"+str(times[i][0])+"_"+str(times[i][1])
        segmentPath = os.path.join(turnPath, segmentName+".wav")
        newAudio = audio_file[times[i][0]*1000:times[i][1]*1000]
        newAudio.export(segmentPath, format="wav")
        myDict = {
            "path"       : segmentPath,
            "trans"      : trs,
            "duration"   : str(times[i][2]),
            "spk_id"     : spkID,
            "spk_gender" : "U"
        }
        allFilesInfo[segmentName] = myDict
        # printProgressBar(i + 1, len(fileNames), prefix = 'Processing Files:', suffix = 'Complete')
    with open('../data.json', 'w') as fp:
        json.dump(allFilesInfo, fp,  indent=4, ensure_ascii=False)
            
def getSegTimes(segmentPath):
    with open(segmentPath, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        segments = list(reader)
    fileNames = []
    times = []
    uttIDs = []
    for segment in segments:
        uttIDs.append(segment[0])
        fileNames.append(segment[0].split('#')[0])
        start = float(segment[-2])
        end = float(segment[-1])
        times.append([start, end, end-start])
    return fileNames, uttIDs, times

def getSpeakerID(uttID, utt2spkPath):
    with open(utt2spkPath, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        segments = list(reader)
    for segment in segments:
        if segment[0] == uttID: return segment[1]

def getTxt(uttID, txtPath):
    with open(txtPath, newline='\n') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        segments = list(reader)
    for segment in segments:
        if segment[0] == uttID: return segment[1]

if __name__== "__main__":
    main()