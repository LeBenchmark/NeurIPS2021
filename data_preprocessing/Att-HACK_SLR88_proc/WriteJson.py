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
    txtPath = "../Volumes/CLEM_HDD/IRCAM/Open_SLR/txt/"
    theFiles = get_files_in_path(wavsPath)

    allFilesInfo = {}
    for i, filePath in enumerate(theFiles):
        audio_file = AudioSegment.from_wav(filePath)
        if audio_file.duration_seconds < 1: continue
        fileName, myDict = makeDict(filePath, txtPath)
        allFilesInfo[fileName] = myDict
        printProgressBar(i + 1, len(theFiles), prefix = 'Processing Files:', suffix = 'Complete')
    with open('../data.json', 'w') as fp:
        json.dump(allFilesInfo, fp,  indent=4, ensure_ascii=False)
            
def makeDict(filePath, txtPath):
    fileName = os.path.basename(filePath)[:-4]
    fileTrs = os.path.join(txtPath, fileName+".txt")
    with open(fileTrs) as f:
        trs = f.readlines()[0]
    spk_id = fileName[0:3]
    gender = fileName[0]
    audio_file = AudioSegment.from_wav(filePath)
    duration = audio_file.duration_seconds
    myDict = {
        "path"       : filePath[1:],
        "trans"      : trs,
        "duration"   : duration,
        "spk_id"     : spk_id,
        "spk_gender" : gender
    }
    return fileName, myDict

if __name__== "__main__":
    main()