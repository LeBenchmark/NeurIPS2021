'''

Sina ALISAMIR
2020-2021
'''


import os, glob
from funcs import get_files_in_path, printProgressBar
from pydub import AudioSegment
import json

def main():
    wavsPath = "../wavs/**/"
    theFiles = get_files_in_path(wavsPath)

    allFilesInfo = {}
    for i, filePath in enumerate(theFiles):
        audio_file = AudioSegment.from_wav(filePath)
        if audio_file.duration_seconds < 1: continue
        fileName, myDict = makeDict(filePath)
        allFilesInfo[fileName] = myDict
    with open('../data.json', 'w') as fp:
        json.dump(allFilesInfo, fp,  indent=4, ensure_ascii=False)
            
def makeDict(filePath):
    fileName = os.path.basename(filePath)[:-4]
    spk_id = fileName[0:2]
    gender = "M"
    if int(spk_id)%2 == 0:
        gender = "F"
    trans = getTranscript(fileName)
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

def getTranscript(fileName):
    trans = ""
    if fileName[-1] == "1": trans = "Un cheval fou dans mon jardin"
    if fileName[-1] == "2": trans = "Deux ânes aigris au pelage brun"
    if fileName[-1] == "3": trans = "Trois cygnes aveugles au bord du lac"
    if fileName[-1] == "4": trans = "Quatre vieilles truies éléphantesques"
    if fileName[-1] == "5": trans = "Cinq pumas fiers et passionnés"
    if fileName[-1] == "6": trans = "Six ours aimants domestiqués"
    return trans

if __name__== "__main__":
    main()