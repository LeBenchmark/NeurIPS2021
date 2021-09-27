import os, glob
from funcs import get_files_in_path, printProgressBar
from pydub import AudioSegment
import json

def main():
    wavsPath = "/mnt/HD-Storage/Databases/Audiocite_processed/**/"
    theFiles = get_files_in_path(wavsPath)

    allFilesInfo = {}
    total_duration = 0
    for i, filePath in enumerate(theFiles):
        audio_file = AudioSegment.from_wav(filePath)
        if audio_file.duration_seconds < 1 or audio_file.duration_seconds > 30: continue
        fileName, myDict, duration = makeDict(filePath)
        total_duration += duration
        allFilesInfo[fileName] = myDict
        printProgressBar(i + 1, len(theFiles), prefix = 'Processing Files:', suffix = 'Complete')
    with open('../data.json', 'w') as fp:
        json.dump(allFilesInfo, fp,  indent=4, ensure_ascii=False)
    print("total duration in seconds: ", total_duration)

def makeDict(filePath):
    fileName = os.path.basename(filePath)[:-4]
    audio_file = AudioSegment.from_wav(filePath)
    duration = audio_file.duration_seconds
    myDict = {
        "path"       : filePath[1:],
        "trans"      : "",
        "duration"   : duration,
        "spk_id"     : "",
        "spk_gender" : "U"
    }
    return fileName, myDict, duration

if __name__== "__main__":
    main()