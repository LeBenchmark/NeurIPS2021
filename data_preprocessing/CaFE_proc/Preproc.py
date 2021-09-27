'''

Sina ALISAMIR
2020-2021
'''


import os, glob
from funcs import get_files_in_path, printProgressBar

def main():
    path = "../CaFE_48k/**/"
    newPath = "../wavs/"
    theFiles = get_files_in_path(path)

    for i, filePath in enumerate(theFiles):
        fileNewPath = filePath.replace("/CaFE_48k/", "/wavs/")
        directory = os.path.dirname(fileNewPath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.system('sox ' + filePath + ' -r 16000 -c 1 -b 16 -e signed-integer ' + fileNewPath)
        printProgressBar(i + 1, len(theFiles), prefix = 'Transforming Files:', suffix = 'Complete')
    
if __name__== "__main__":
    main()