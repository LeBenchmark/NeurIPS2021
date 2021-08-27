import os, sys, argparse, json, glob
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from DataProcess.Preprocess import main as Preprocess
from DataProcess.makeJson import main as makeJson
from DataProcess.repartitionJson import main as repartitionJson
from DataProcess.addAnnots import main as addAnnots
from Experiments.Experiment import Experiment
from Torch_Runners.Run_main import Runner
import pandas as pd
import numpy as np

def main(wavsFolder, annotsFolder, partitions, outFolder):
    """
    example:
        python AlloSat.py -w "/home/getalp/alisamis/Datasets/Original/AlloSat_corpus/audio" -a "/home/getalp/alisamis/Datasets/Original/AlloSat_corpus/annotations/labels" -p "/home/getalp/alisamis/Datasets/Original/AlloSat_corpus/" -o "/home/getalp/alisamis/Datasets/Processed/AlloSat"
    """
    
    ## Preprocess wav files
    print("Preprocess wav files...")
    newFolder = os.path.join(outFolder, "Wavs")
    Preprocess(wavsFolder, newFolder, False, "wav")

    ## Create json file
    makeJson(outFolder, [])
    jsonPath = os.path.join(outFolder, "data.json")
    trainList, devList, testList = getParts(partitions)
    outJson = jsonPath
    print("Partitioning data...")
    repartitionJson([], jsonPath, False, trainList, devList, testList, outJson)

    ## Create labels
    annotTargetFolder = os.path.join(outFolder, "Annots", "labels")
    makeAnnot(annotsFolder, annotTargetFolder)

    ## add Annots
    dataJsonPath = os.path.join(outFolder, "data.json")
    genres = ["satisfaction"]
    annotsList = ["labels"]
    headers = ["satisfaction"]
    addAnnots(annotsList, headers, genres, dataJsonPath)

    ## Set the experiments
    ExpPath     = os.path.join(outFolder, "Results", "LeBenchmark")
    expJsonPath = os.path.join(ExpPath, "experiments.json")
    setExperiments(outFolder, dataJsonPath, ExpPath, expJsonPath)

    ## Run the experiments
    
    runner = Runner(expJsonPath)
    runner.main()



def setExperiments(outFolder, dataJsonPath, ExpPath, expJsonPath):
    if os.path.exists(expJsonPath): os.remove(expJsonPath) # To remove from the json file, all the experiments before will be removed 
    device = "cuda:1"
    ignoreExisting = True # ignores already defined experiments, if "False", would overwrite 
    seeds = [0, 1, 2]
    testRun = False
    testConcated = True
    onlineFeat = True
    resampleTarget = True

    models = {
        "LinTanh": [""],
        "GRU": ["32-1", "64-1"]
        }
    feats = ["MFB_standardized", "W2V2_large_libri960", "W2V2_large_xlsr_53_56k", "W2V2_mls_french_base", "W2V2_mls_french_large", "W2V2_2.6k_base", "W2V2_2952h_base", "W2V2_2952h_large", "W2V2_7K_base", "W2V2_7K_large"] #"MFB_standardized", "wav2vec2-xlsr_53", "wav2vec2-FlowBERT_2952h_base"
    featModelPaths = {"MFB_standardized": "",
                      "W2V2_large_libri960": "/home/getalp/alisamis/Models/libri960_big.pt",
                      "W2V2_large_xlsr_53_56k": "/home/getalp/alisamis/Models/xlsr_53_56k.pt",
                      "W2V2_mls_french_base":   "/home/getalp/alisamis/Models/wav2vec2.0_models/mls_french_base/17_05_2021/checkpoint_best.pt",
                      "W2V2_mls_french_large":  "/home/getalp/alisamis/Models/wav2vec2.0_models/mls_french_large/08_03_2021/checkpoint_best.pt",
                      "W2V2_2.6k_base": "/home/getalp/alisamis/Models/wav2vec2.0_models/2.6K_base/24_06_2021/checkpoint_best.pt",
                      "W2V2_2952h_base":  "/home/getalp/alisamis/Models/wav2vec2.0_models/2952h_base/17_05_2021/checkpoint_best.pt",
                      "W2V2_2952h_large": "/home/getalp/alisamis/Models/wav2vec2.0_models/2952h_large/18_05_2021/checkpoint_best.pt",
                      "W2V2_7K_base":  "/home/getalp/alisamis/Models/wav2vec2.0_models/7K_base/24_05_2021/checkpoint_best.pt",
                      "W2V2_7K_large":  "/home/getalp/alisamis/Models/wav2vec2.0_models/7K_large/25_05_2021/checkpoint_best.pt",
                     }
    annots = ["labels"]
    for seed in seeds:
        for model in models:
            for param in models[model]:
                for feat in feats:
                    for annot in annots:
                        expID = annot + "_" + feat + "_" + model + "_" + param + "_" + str(seed)
                        myExp = Experiment(expID)
                        myExp.expPath = os.path.join(ExpPath, myExp.ID)
                        myExp.genre = "classic"
                        featModelPath = ""
                        featModelNorm = True
                        featModelMaxDur = 15.98
                        if "MFB" in feat:
                            myExp.inputDim = 40
                        if "W2V2" in feat:
                            if "base" in feat: 
                                myExp.inputDim = 768
                                featModelNorm = False
                            if "large" in feat: 
                                myExp.inputDim = 1024
                                featModelNorm = True
                        myExp.data = {
                            "path": dataJsonPath, # path to data.json file
                            "devPath": dataJsonPath,
                            "testPaths": [dataJsonPath],
                            "annotation": annot, # path to data.json file (if needed, can be left as "")
                            "feature": feat, # which feature to read (if needed, e.g. if not wav as input, can be left as "")
                            "featModelPath": featModelPaths[feat],
                            "featModelNorm": featModelNorm,
                            "featModelMaxDur": featModelMaxDur,
                            }

                        myExp.seed = seed
                        myExp.testRun = testRun
                        myExp.testConcated = testConcated
                        myExp.onlineFeat = onlineFeat
                        myExp.resampleTarget = resampleTarget
                        myExp.model = model
                        myExp.modelParams = param
                        myExp.device = device
                        myExp.criterion = "CCC"
                        myExp.optimizer = "Adam"
                        myExp.learningRate = 0.001
                        myExp.maxEpoch = 250
                        myExp.tolerance = 15
                        myExp.minForTolerance = 50
                        myExp.outputDim = 1
                        myExp.metrics = ["CCC", "RMSE"]
                        myExp.evaluation = {}
                        myExp.saveToJson(expJsonPath, ignoreExisting)

def getParts(path):
    trainListPath = os.path.join(path, "train.txt")
    devListPath   = os.path.join(path, "dev.txt"  )
    testListPath  = os.path.join(path, "test.txt" )
    trainList = pd.read_csv(trainListPath, delimiter = "\t", header=None).to_numpy()
    trainList = trainList.reshape(trainList.shape[0])
    devList = pd.read_csv(devListPath, delimiter = "\t", header=None).to_numpy()
    devList = devList.reshape(devList.shape[0])
    testList = pd.read_csv(testListPath, delimiter = "\t", header=None).to_numpy()
    testList = testList.reshape(testList.shape[0])
    return trainList, devList, testList
    
def arffs2csvs(arffFolder, csvFolder, mainFolder, outFolder):
    filePath = os.path.join(arffFolder, "**", "*.arff")
    filesPaths = glob.glob(filePath, recursive=True)
    outFiles = [filePath.replace(mainFolder, outFolder).replace(os.path.split(arffFolder)[1], csvFolder).replace(".arff", ".csv") for filePath in filesPaths]
    for i, filePath in enumerate(filesPaths):
        outPath = outFiles[i]
        directory = os.path.dirname(outPath)
        if not os.path.exists(directory): os.makedirs(directory)
        arff2csv(filePath, csv_path=outPath)

def arff2csv(arff_path, csv_path=None, _encoding='utf8'):
    with open(arff_path, 'r', encoding=_encoding) as fr:
        attributes = []
        if csv_path is None:
            csv_path = arff_path[:-4] + 'csv'  # *.arff -> *.csv
        write_sw = False
        with open(csv_path, 'w', encoding=_encoding) as fw:
            for line in fr.readlines():
                if write_sw:
                    if line == "": print("emp")
                    if line != "\n": fw.write(line)
                elif '@data' in line:
                    fw.write(','.join(attributes) + '\n')
                    write_sw = True
                elif '@attribute' in line:
                    attributes.append(line.split()[1])  # @attribute attribute_tag numeric
    print("Convert {} to {}.".format(arff_path, csv_path))

def makeAnnot(origFolder, outFolder):
    if not os.path.exists(outFolder): os.makedirs(outFolder)
    allPaths = glob.glob(os.path.join(origFolder, '*'), recursive=True)
    for filePath in sorted(allPaths):
        AfterLastSlash = filePath.rfind(os.path.split(filePath)[-1])
        fileName = filePath[AfterLastSlash:]
        new_path = os.path.join(outFolder, fileName)
        df = pd.read_csv(filePath, delimiter=';')
        header = df.keys()
        out = df.to_numpy()[:,1:].astype('float64')
        df = pd.DataFrame(data=out)
        df.to_csv(new_path, header=header[1:], index=False)

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotsFolder', '-a', help="path to folder contatining annotations files", default="")
    parser.add_argument('--wavsFolder', '-w', help="path to folder containing wav files", default="")
    parser.add_argument('--partitions', '-p', help="path to folder containing partitioning files i.e. train.txt, dev.txt, test.txt", default="")
    parser.add_argument('--outFolder', '-o', help="path to folder containing output files", default="")
    
    args = parser.parse_args()
    Flag = False
    if args.annotsFolder == "": Flag = True
    if args.wavsFolder == "":  Flag = True
    if args.partitions == "":  Flag = True
    if args.outFolder == "":   Flag = True
    if Flag:
        parser.print_help()
    else:
        main(args.wavsFolder, args.annotsFolder, args.partitions, args.outFolder)

