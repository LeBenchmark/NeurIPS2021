## VAD related imports
from VAD_Module import VAD_Module
# from Models import FeatureModel

## Segmenter import
from Segmenter import Segmenter

## Other imports
import os, glob, argparse


def main(wavsFolder, outputFolder):
    path = os.path.join(wavsFolder, "**", "*.wav")
    theFiles = glob.glob(path, recursive=True)

    for filePath in theFiles:
        vad = VAD_Module(smoothedWin=0.05, mergeWin=0.5, modelPath="./Models/Recola_46_MFB_standardized_LinTanh/model.pth")  #"Recola_46_MFB_standardized_LinTanh" or "Recola_46_MFB_standardized_GRU_32-1"
        times = vad.timesFromFile(filePath)
        print("VAD is done for", filePath)

        segmenter = Segmenter(savePath=outputFolder)
        outPath = os.path.split(filePath.replace(wavsFolder, outputFolder))[0]
        print("outPath", outPath)
        segmenter.segmentFile(times, filePath, outPath=outPath)
        print("Segmentation is done for", filePath)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wavsFolder', '-i', help="path to input folder containing wav files", default="") # e.g. "./WavFiles"
    parser.add_argument('--outputFolder', '-o', help="path to output folder containing segmented wav files", default="") # e.g. "./segments"
    
    args = parser.parse_args()
    Flag = False
    if args.wavsFolder == "": Flag = True
    if args.outputFolder == "": Flag = True
    if Flag:
        parser.print_help()
    else:
        main(args.wavsFolder, args.outputFolder)