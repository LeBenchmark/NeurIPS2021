import torch
import torch.nn as nn
from torch.autograd import Variable
from pydub import AudioSegment
from python_speech_features import logfbank
import numpy as np
# from Models import FeatureModel

class VAD_Module(object):
    """
    VAD module to do voice activity detection on audio files.
    Example:
        path = "./test.wav"
        vad = VAD_Module()
        times = vad.timesFromFile(path)
        # This will return a list of times [start, end] (in seconds) of all the voices detected for the file at path, e.g. [[0.0, 3.39], [3.72, 5.48], [7.04, 7.87], [8.24, 8.65]].
    """
    def __init__(self, smoothedWin=0.5, mergeWin=0.5, modelPath=""):
        self.smoothedWin = smoothedWin
        self.modelPath = modelPath
        self.mergeWin = mergeWin*100
    
    def timesFromFile(self, path):
        times = self.timesFromAudio(path)
        return times
    
    def timesFromAudio(self, audioPath):
        out = self.VadFromAudio(audioPath)
        times = self.getTimes(out)# Get the start and end times of each detected speech segment
        return times

    def VadFromAudio(self, audioPath):
        model = torch.load(self.modelPath, map_location="cpu")
        inputs = self.getFeatsFromWav(audioPath, winlen=0.025, winstep=0.01)
        inputs = (inputs - np.mean(inputs)) / np.std(inputs)
        feats = torch.FloatTensor(inputs)
        feats = Variable(feats).unsqueeze(0)
        out = model(feats)# Getting the output of the model based on input features
        out = out[:,:,:].view(out.size()[0]*out.size()[1]*out.size()[2])
        out = out.cpu().data.numpy()
        if self.smoothedWin>0: out = self.hysteresis(out, win=self.smoothedWin*100)# Smoothing the output of the model with a window of 0.5s (here it means 50 samples based on feature extraction process)
        out = self.mergeOuts(out)
        return out

    def getFeatsFromWav(self, path, winlen=0.025, winstep=0.01):
        audio_file = AudioSegment.from_wav(path)
        sig = np.array(audio_file.get_array_of_samples())
        sig = sig / 32767.0 # scaling for 16-bit integer
        rate = audio_file.frame_rate
        fbank_feat = logfbank(sig, rate, winlen=winlen, winstep=winstep, nfilt=40, nfft=2028, lowfreq=0, highfreq=None, preemph=0.97) #, winfunc=np.hanning
        return fbank_feat

    def getTimes(self, out, fs=0.01):
        ins = []
        outs = []
        last = 0
        for i, o in enumerate(out):
            if o == 1 and last != 1: ins.append(i)
            if o == -1 and last == 1: outs.append(i)
            last = o
        if out[-1] == 1: outs.append(len(out)-1)
        times = []
        for i, _ in enumerate(outs):
            times.append([round(ins[i]*fs,3), round(outs[i]*fs,3)])
        return times

    def smooth(self, sig, win=25*1):
        import numpy as np
        mysig = sig.copy()
        aux = int(win/2)
        for i in range(aux, len(mysig)-aux):
            value = np.mean(sig[i-aux:i+aux])
            mysig[i] = 1 if value > 0 else -1
        mysig[:aux] = 1 if np.mean(sig[:aux]) > 0 else -1
        mysig[-aux:] = 1 if np.mean(sig[-aux:]) > 0 else -1
        return mysig

    def hysteresis(self, sig, win=25*1, bottom=-0.8, top=0):
        import numpy as np
        mysig = sig.copy()
        aux = int(win/2)
        mysig[0] = 1 if mysig[0] > top else -1
        for i in range(1, len(mysig)):
            if mysig[i] >= top:
                mysig[i] = 1
            if mysig[i] >= bottom and mysig[i] < top:
                if mysig[i-1] == 1: 
                    mysig[i] = 1
                else:
                    mysig[i] = -1
            if mysig[i] < bottom:
                mysig[i] = -1
        start = 0
        for i in range(1, len(mysig)):
            if mysig[i] == 1 and mysig[i-1] == -1: start = i
            if mysig[i] == -1 and mysig[i-1] == 1: 
                if i-start < win: 
                    for j in range(start, i):
                        mysig[j] = -1 
        return mysig

    def mergeOuts(self, out):
        myOut = out
        counter = 0; shouldCount = False; startC = 0
        for i in range(1, len(out)):
            if out[i-1] == 1 and out[i] == -1: 
                shouldCount = True
                startC = i
            if shouldCount:
                counter += 1
            if out[i-1] == -1 and out[i] == 1: 
                shouldCount = False
                if counter < self.mergeWin:
                    for j in range(startC, i):
                        myOut[j] = 1
                counter = 0
        return myOut
