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
    wavsPath = "../PMDOM2FR_wavs/**/wav/**"
    theFiles = get_files_in_path(wavsPath)
    
    allFilesInfo = {}
    for i, filePath in enumerate(theFiles):
        audio_file = AudioSegment.from_wav(filePath)
        spk_ids, spk_genders, times, trs = getSegments(filePath)
        directory = os.path.dirname(filePath)
        directory = os.path.join(directory, "..", "wavs_turns")
        if not os.path.exists(directory): os.makedirs(directory)
        fileName = os.path.basename(filePath)[:-4]
        for j, time in enumerate(times):
            if time[2] < 1 or time[2] > 30: continue
            segmentName = fileName+"_"+str(time[0])+"_"+str(time[1])
            segmentPath = os.path.join(directory, segmentName+".wav")
            newAudio = audio_file[time[0]*1000:time[1]*1000]
            newAudio.export(segmentPath, format="wav")
            myDict = {
                "path"       : segmentPath,
                "trans"      : trs[j],
                "duration"   : str(time[2]),
                "spk_id"     : spk_ids[j],
                "spk_gender" : spk_genders[j]
            }
            allFilesInfo[segmentName] = myDict
        printProgressBar(i + 1, len(theFiles), prefix = 'Processing Files:', suffix = 'Complete')
    with open('../data.json', 'w') as fp:
        json.dump(allFilesInfo, fp,  indent=4, ensure_ascii=False)
            
def getSegments(wavPath):
    trsPath = wavPath.replace("/wav/", "/trs/")
    trsPath = trsPath.replace(".wav", ".trs")
    with open(trsPath, newline='\n', encoding='ISO-8859-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        trsList = list(reader)
    Turns = []
    Speakers = []
    syncs = []; counter=0
    for i, trsLine in enumerate(trsList):
        if len(trsLine) > 0:
            if trsLine[0] == "<Turn": Turns.append(trsLine); counter=i
            if trsLine[0] == "<Speaker": Speakers.append(trsLine)
            if trsLine[0] == "</Turn>": syncs.append(trsList[counter+1:i-1])
    trs = getTrsFromSyncs(syncs)
    spk_ids, spk_genders, times = getInfoOnSegs(Turns, Speakers)
    return spk_ids, spk_genders, times, trs

def getInfoOnSegs(Turns, Speakers):
    spk_ids = []
    spk_genders = []
    times = []
    for turn in Turns:
        for i, item in enumerate(turn):
            if "startTime" in item: 
                try:
                    start = float(item.split('"')[1:2][0])
                except:
                    start = float(turn[i+1].split('"')[0])
            if "endTime" in item: 
                try:
                    end = float(item.split('"')[1:2][0])
                except:
                    end = float(turn[i+1].split('"')[0])
        times.append([start, end, end-start])
        for spk in Speakers:
            spkId = spk[1][4:-1]
            for item in turn:
                if "speaker" in item: 
                    itemId = item.split('"')[1:2][0]
            if itemId != spkId: continue
            spkName = spk[2].split('"')[1:2][0]
            spkGndr = spk[3].split('"')[1:2][0]
            spk_ids.append(spkName)#spkId+"_"+spkName
            spk_genders.append("M") if spkGndr == "male" else spk_genders.append("F")
            
    return spk_ids, spk_genders, times

def getTrsFromSyncs(syncs):
    alltrs = []
    for item in syncs:
        trs = []
        for i, sync in enumerate(item):
            if len(sync) == 0:continue
            if sync[0] == "<Sync":
                if i > 0: 
                    if item[i-2][0] == "<Sync": 
                        trs.append(item[i-1])
        trsStr = ""
        for j, st in enumerate(trs):
            if j != 0: trsStr = trsStr + " "
            trsStr = trsStr + " ".join(st)
        alltrs.append(trsStr)
    return alltrs

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

if __name__== "__main__":
    main()