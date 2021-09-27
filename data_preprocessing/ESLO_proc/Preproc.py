'''

Sina ALISAMIR
2020-2021
'''

import os, glob
from funcs import get_files_in_path, printProgressBar

def main():
    path = "../PMDOM2FR/**/"
    newPath = "../PMDOM2FR_wavs/"
    theFiles = get_files_in_path(path)

    for i, filePath in enumerate(theFiles):
        # Making wav files
        fileNewPath = filePath.replace("/PMDOM2FR/", "/PMDOM2FR_wavs/")
        makeDirFor(fileNewPath)
        os.system('sox ' + filePath + ' -r 16000 -c 1 -b 16 -e signed-integer ' + fileNewPath)
        # Copying trs files
        fileTrsPath = filePath.replace("/wav/", "/trs/")
        fileTrsPath = fileTrsPath.replace(".wav", ".trs")
        fileTrsDesPath = fileTrsPath.replace("/PMDOM2FR/", "/PMDOM2FR_wavs/")
        makeDirFor(fileTrsDesPath)
        os.system('cp ' + fileTrsPath + ' ' + fileTrsDesPath)
        printProgressBar(i + 1, len(theFiles), prefix = 'Transforming Files:', suffix = 'Complete')
    
def makeDirFor(filePath):
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__== "__main__":
    main()