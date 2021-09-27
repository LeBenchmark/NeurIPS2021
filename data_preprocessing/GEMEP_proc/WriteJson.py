'''

Sina ALISAMIR
2020-2021
'''


import os, glob
from funcs import get_files_in_path, printProgressBar
from pydub import AudioSegment
import json

def main():
    wavsPath = "../GEMEP_wavs/**/"
    theFiles = get_files_in_path(wavsPath)

    allFilesInfo = {}
    for i, filePath in enumerate(theFiles):
        audio_file = AudioSegment.from_wav(filePath)
        if audio_file.duration_seconds < 1: continue
        fileName, myDict = makeDict(filePath)
        allFilesInfo[fileName] = myDict
        printProgressBar(i + 1, len(theFiles), prefix = 'Processing Files:', suffix = 'Complete')
    with open('data.json', 'w') as fp:
        json.dump(allFilesInfo, fp,  indent=4)
            
def makeDict(filePath):
    fileName = os.path.basename(filePath)[:-4]
    spk_id = fileName[0:3]
    gender = fileName[0]
    audio_file = AudioSegment.from_wav(filePath)
    duration = audio_file.duration_seconds
    genders = {"W01":"M", "W02":"F", "W03":"M", "W04":"M", "W05":"M", "W06":"F", "W07":"F", "W08":"M", "W09":"F", "W10":"F"}
    myDict = {
        "path"       : filePath,
        "trans"      : "",
        "duration"   : duration,
        "spk_id"     : spk_id,
        "spk_gender" : genders[spk_id]
    }
    return fileName, myDict

if __name__== "__main__":
    main()