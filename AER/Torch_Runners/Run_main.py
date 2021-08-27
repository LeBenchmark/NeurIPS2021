import os, sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
from Experiments.Experiment import Experiment
import json
import argparse
from Torch_Runners.modelWrapper import ModelWrapper
from DataProcess.DataReader import DataReader
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os, sys
from Utils.Metrics import getMetric
from Utils.Funcs import csvReader, printProgressBar, smooth

class Runner(object):
	"""
	This class is used to run experiments! Based on indicated parameters, this main class routes to scripts for training and testing models.
	Example:
		python Run_main.py -j "../../Experiments/Recola_46_11/experiments.json"
	"""
	def __init__(self, jsonPath):
		super(Runner, self).__init__()
		self.jsonPath = jsonPath
		self.getExpIDs()

	def loadExperiment(self, expID):
		self.experiment = Experiment(expID, loadPath=self.jsonPath)
		if self.experiment.testRun: self.experiment.maxEpoch = 1

	def loadWrapper(self, getBest=False):
		wrapper = ModelWrapper(savePath=self.experiment.expPath, device=self.experiment.device, seed=self.experiment.seed, printLvl=10)
		wrapper.setModel(self.experiment.model, self.experiment.inputDim, self.experiment.outputDim, self.experiment.modelParams)
		wrapper.setOptimizer(self.experiment.optimizer, learningRate=self.experiment.learningRate)
		wrapper.setCriterion(self.experiment.criterion)
		wrapper.loadCheckpoint()
		if getBest: wrapper.loadBestModel()
		self.wrapper = wrapper

	def loadDatasets(self):
		self.datasetTrain = DataReader(self.experiment.data["path"], onlineFeat=self.experiment.onlineFeat, resampleTarget=self.experiment.resampleTarget)
		self.datasetTrain.setDatasetClassic("train", self.experiment.data["feature"], self.experiment.data["annotation"])
		if self.experiment.testRun: self.datasetTrain.keepOneOnly()
		self.datasetDev = DataReader(self.experiment.data["devPath"], onlineFeat=self.experiment.onlineFeat, resampleTarget=self.experiment.resampleTarget)
		self.datasetDev.setDatasetClassic("dev", self.experiment.data["feature"], self.experiment.data["annotation"])
		if self.experiment.data["featModelPath"] != "" and self.experiment.onlineFeat: 
			self.datasetTrain.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], device=self.experiment.device)
			print("fairseq model for training data loaded")
			# self.datasetDev.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], cuda=self.cuda)
			self.datasetDev.reUseModelFeat(self.datasetTrain.featModel, normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], device=self.experiment.device)
			print("fairseq model for development data loaded")
		if self.experiment.testRun: self.datasetDev.keepOneOnly()

	def main(self):
		for expID in self.expIDs:
			self.loadExperiment(expID)
			self.loadWrapper()
			self.loadDatasets()
			print("--- Experiment with ID:", self.experiment.ID, "---")
			if self.experiment.stage == "ready":
				self.trainModel()
				self.experiment.stage = "trained"
				self.experiment.saveToJson(self.jsonPath)
				self.saveBestModel()
			if self.experiment.stage == "trained":
				print("Model is trained!")
				self.currentDataPath = self.experiment.data["devPath"]
				self.writeModelOutput(partition="dev")
				for path in self.experiment.data["testPaths"]:
					self.currentDataPath = path
					self.writeModelOutput()
				self.experiment.stage = "readyForTest"
				self.experiment.saveToJson(self.jsonPath)
			if self.experiment.stage == "readyForTest":
				print("Model outputs are extracted and ready for test!")
				evaluations = {}
				self.currentDataPath = self.experiment.data["devPath"]
				evaluations["dev"] = self.testModel(partition="dev")
				for path in self.experiment.data["testPaths"]:
					self.currentDataPath = path
					evaluations[path] = self.testModel()
				self.experiment.evaluation = evaluations
				self.experiment.stage = "tested"
				self.experiment.saveToJson(self.jsonPath)
			if self.experiment.stage == "tested":
				print("Model is tested! results:", str(self.experiment.evaluation))

	def saveBestModel(self):
		self.loadWrapper(getBest=True)
		path = os.path.join(self.experiment.expPath, "model.pth")
		print("path", path)
		torch.save(self.wrapper.model, path)

	# def getWrapper(self, getBest=False):
	# 	wrapper = ModelWrapper(savePath=self.experiment.expPath, device=self.experiment.device, seed=self.experiment.seed, printLvl=10)
	# 	wrapper.setModel(self.experiment.model, self.experiment.inputDim, self.experiment.outputDim, self.experiment.modelParams)
	# 	wrapper.setOptimizer(self.experiment.optimizer, learningRate=self.experiment.learningRate)
	# 	wrapper.setCriterion(self.experiment.criterion)
	# 	wrapper.loadCheckpoint()
	# 	if getBest: wrapper.loadBestModel()
	# 	return wrapper


	def trainModel(self):
		print("Training model ...")
		# if not "classification" in self.experiment.genre:
		# 	self.experiment.outputDim = tar.shape[1]
		self.wrapper.trainModel(self.datasetTrain, self.datasetDev, batchSize= self.experiment.batchSize, maxEpoch= self.experiment.maxEpoch, 
						   loadBefore=True, tolerance = self.experiment.tolerance, 
						   minForTolerance= self.experiment.minForTolerance, limitTrainData=self.experiment.limitTrainData, limitDevData=self.experiment.limitDevData)
		self.wrapper.saveLogToCSV()
		

	def writeModelOutput(self, partition="test"):
		# print("Writing outputs ...")
		if "classification" in self.experiment.genre:
			self.writeOutForClassification(partition=partition)
		else:
			self.writeOutForRegression(partition=partition)
	

	def writeOutForRegression(self, partition="test"):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.experiment.onlineFeat, resampleTarget=self.experiment.resampleTarget)
		dataset.setDatasetFeatOnly(partition, self.experiment.data["feature"])
		if self.experiment.data["featModelPath"] != "" and self.experiment.onlineFeat: 
			dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], device=self.experiment.device)
			print("fairseq model for writting model outputs loaded")
		if self.experiment.testRun: dataset.keepOneOnly()
		dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
		self.loadWrapper(getBest=True)
		modelOutPath = os.path.join(self.wrapper.savePath, "outputs")
		if not os.path.exists(modelOutPath): os.makedirs(modelOutPath)
		for idx, (ID, feat) in enumerate(dataloader):
			output = self.wrapper.forwardModel(feat)
			output = output.detach().cpu().numpy()
			# print(ID, feat.shape, output.shape)
			savePath = os.path.join(modelOutPath, ID[0]+".csv")
			headers = ["output_"+str(i) for i in range(output.shape[2])]
			df = pd.DataFrame(output[0], columns = headers)
			df.to_csv(savePath, index=False)
			printProgressBar(idx+1, len(dataloader), prefix = 'Writing outputs:', suffix = '', length = "fit")

	def writeOutForClassification(self, partition="test"):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.experiment.onlineFeat, resampleTarget=self.experiment.resampleTarget)
		dataset.setDatasetFeatOnly(partition, self.experiment.data["feature"])
		if self.experiment.data["featModelPath"] != "" and self.experiment.onlineFeat: 
			dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], device=self.experiment.device)
			print("fairseq model for writting model outputs loaded")
		if self.experiment.testRun: dataset.keepOneOnly()
		dataloader = DataLoader(dataset=dataset,   batch_size=1, shuffle=False)
		self.loadWrapper(getBest=True)
		modelOutPath = os.path.join(self.wrapper.savePath, "outputs")
		if not os.path.exists(modelOutPath): os.makedirs(modelOutPath)
		headers = []
		outputs = []
		for idx, (ID, feat) in enumerate(dataloader):
			printProgressBar(idx+1, len(dataloader), prefix = 'Writing outputs:', suffix = '', length = "fit")
			output = self.wrapper.forwardModel(feat)
			output = output.detach().cpu().numpy()
			# print(ID, feat.shape, output.shape)
			outputs = output if len(outputs)==0 else np.concatenate((outputs,output))
			headers.append(ID[0])
			# print(len(headers), outputs.shape)
		savePath = os.path.join(modelOutPath, "outputs.csv")
		df = pd.DataFrame(np.transpose(outputs), columns = headers)
		df.to_csv(savePath, index=False)


	def testModel(self, partition="test"):
		if "classification" in self.experiment.genre:
			return self.testClassification(partition=partition)
		else:
			return self.testRegression(partition=partition)


	def testRegression(self, partition="test"):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.experiment.onlineFeat, resampleTarget=self.experiment.resampleTarget)
		dataset.setDatasetAnnotOnly(partition, self.experiment.data["annotation"])
		firstID1 = list(dataset.dataPart.keys())[0]
		try:
			firstID2 = list(dataset.dataPart[firstID1]["annotations"])[0]
		except:
			print("Warning: No test target found for", firstID1)
			return
		headers = dataset.dataPart[firstID1]["annotations"][firstID2]["headers"]
		# print(headers)
		if self.experiment.testRun: dataset.keepOneOnly()
		IDs = dataset.dataPart.keys()
		# for key in self.experiment.metrics:
		modelOutPath = os.path.join(self.experiment.expPath, "outputs")
		evaluations = {}
		evaluation = {}
		allTars = []
		allOuts = []
		results = np.zeros((len(self.experiment.metrics), len(headers), len(IDs)))
		for idx, ID in enumerate(IDs):
			savePath = os.path.join(modelOutPath, ID+".csv")
			outputs = pd.read_csv(savePath).to_numpy()
			targets = dataset.targetReader(ID)
			# RESAMPLE OUTPUT TO TARGETS FOR TESTING!
			if self.experiment.resampleTarget: 
				from Utils.Funcs import reshapeMatrix
				outputs = reshapeMatrix(outputs, len(targets))
				# print("targets.shape", targets.shape, outputs.shape)
			for dim in range(targets.shape[1]):
				output = outputs[:, dim]
				target = targets[:, dim]
				# while target.shape[0] > output.shape[0]: output = np.append(output, outputs[-1])
				# while target.shape[0] < output.shape[0]: output = outputs[:target.shape[0]].reshape(target.shape[0])
				while target.shape[0] != output.shape[0]: output = outputs.reshape(target.shape[0])
				if self.experiment.testConcated: allTars+=list(target);  allOuts+=list(output)
				for k, key in enumerate(self.experiment.metrics):
					result = getMetric(target, output, metric=key)
					results[k, dim, idx] = result
			printProgressBar(idx+1, len(IDs), prefix = 'Testing model:', suffix = '', length = "fit")

		for k, key in enumerate(self.experiment.metrics):
			for dim in range(targets.shape[1]): 
				if self.experiment.testConcated: 
					evaluation[headers[dim]] = getMetric(np.array(allTars), np.array(allOuts), metric=key)
					if key=="AUC": # write fpr & tpr to plot ROCs!
						from sklearn import metrics
						fpr, tpr, thresholds = metrics.roc_curve(np.array(allTars), np.array(allOuts))
						fpr =  reshapeMatrix(np.expand_dims(fpr, axis=1), 100)
						tpr =  reshapeMatrix(np.expand_dims(tpr, axis=1), 100)
						savePath = os.path.join(self.experiment.expPath, "ROC_resampled_"+str(dim)+"_"+os.path.split(self.currentDataPath)[-1]+".csv")
						np.savetxt(savePath, [np.squeeze(fpr), np.squeeze(tpr)], delimiter=",")
				else:
					evaluation[headers[dim]] = {}
					evaluation[headers[dim]]['mean'] = np.mean(results[k, dim])
					evaluation[headers[dim]]['std'] = np.std(results[k, dim])
					evaluation[headers[dim]]['min'] = np.min(results[k, dim])
					evaluation[headers[dim]]['max'] = np.max(results[k, dim])
			evaluations[key] = evaluation.copy()
		return evaluations


	def testClassification(self, partition="test"):
		dataset = DataReader(self.currentDataPath, onlineFeat=self.experiment.onlineFeat, resampleTarget=self.experiment.resampleTarget)
		dataset.setDatasetClassic(partition, self.experiment.data["feature"], self.experiment.data["annotation"])
		if self.experiment.data["featModelPath"] != "" and self.experiment.onlineFeat: 
			dataset.getModelFeat(self.experiment.data["featModelPath"], normalised=self.experiment.data["featModelNorm"], maxDur=self.experiment.data["featModelMaxDur"], device=self.experiment.device)
			print("fairseq model for testing model outputs loaded")
		inp, tar = dataset[0]
		self.experiment.inputDim = inp.shape[1]
		firstID1 = list(dataset.dataPart.keys())[0]
		firstID2 = list(dataset.dataPart[firstID1]["annotations"])[0]
		headers = dataset.dataPart[firstID1]["annotations"][firstID2]["headers"]
		# print(headers)
		wrapper = getWrapper(self.experiment, seed=self.experiment.seed, getBest=True)
		modelOutPath = os.path.join(wrapper.savePath, "outputs")
		savePath = os.path.join(modelOutPath, "outputs.csv")
		outputsCSV = pd.read_csv(savePath)
		if self.experiment.testRun: dataset.keepOneOnly()
		IDs = dataset.dataPart.keys()
		AllOuts = []
		AllTars = []
		for idx, ID in enumerate(IDs):
			outputs = outputsCSV[ID].to_numpy()
			targets = dataset.targetReader(ID)
			AllOuts.append(np.argmax(outputs))
			AllTars.append(targets[0,0])
			# print(np.argmax(outputs), targets[0,0])
			printProgressBar(idx+1, len(IDs), prefix = 'Testing model :', suffix = '', length = "fit")
			# if idx > 50: break
		target = np.array(AllTars)
		output = np.array(AllOuts)
		evaluation = {}
		for key in self.experiment.metrics:
			evaluation[key] = getMetric(target, output, metric=key)
		self.experiment.evaluation = evaluation
		confMat = confMatrix(target, output, numTars=self.experiment.outputDim)
		# print(confMatrix(target, output, numTars=experiment.outputDim))
		savePath = os.path.join(wrapper.savePath, "confMat.csv")
		np.savetxt(savePath, confMat, delimiter=",")
		return evaluation


	def getExpIDs(self):
		with open(self.jsonPath, 'r') as jsonFile: 
		    jsonDict = json.load(jsonFile)
		    expIDs = jsonDict.keys()
		self.expIDs = expIDs


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', '-j', help="path to experiments.json file")
    
    args = parser.parse_args()
    Flag = False
    if args.json is None: Flag=True
    if Flag:
        parser.print_help()
    else:
        runner = Runner(args.json)
        runner.main()
