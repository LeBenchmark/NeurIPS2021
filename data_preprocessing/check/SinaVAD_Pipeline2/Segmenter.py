# import librosa
import os
import soundfile as sf

class Segmenter(object):
    """
    Segmenting audio files based on an input 
    """
    def __init__(self, savePath='./segments'):
        self.savePath = savePath
        
    def segmentFile(self, times, path, outPath="", sr=16000):
        data, sr = sf.read(path)
        for time in times:
            segment = data[int(time[0]*sr):int(time[1]*sr)]
            originalName = os.path.split(path)[-1][:-4]
            fileName = originalName + '_' + str(time[0]) + '_' + str(time[1]) + '.wav'
            segmentPath = os.path.join(self.savePath, originalName, fileName)
            if outPath != "": segmentPath = os.path.join(outPath, originalName, fileName)
            if not os.path.exists(os.path.dirname(segmentPath)): os.makedirs(os.path.dirname(segmentPath))
            sf.write(segmentPath, segment, sr)
