
import os
import sys

from scipy import signal
from scipy.io import wavfile
import numpy as np

import torch
import fairseq
import torch.nn.functional as F
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc

import resampy

def split_wav_signal(infos, signal, chunk_duration):

    overlap = infos['split_overlap']
    sample_rate = infos['sample_rate']

    chunks = []
    offset = 0
    bound = int(chunk_duration*sample_rate) 

    overlap_samples = int(overlap * sample_rate)
    while bound <= len(signal):
        chunks.append( signal[offset:bound] )
        if bound == len(signal):
            return chunks
        offset += (int(chunk_duration*sample_rate) - overlap_samples)
        bound = min(len(signal), offset+int(chunk_duration*sample_rate)) 

    return chunks

def extract_w2v2_features(model, samples):

    if isinstance(model, Wav2VecCtc):
        poker_face = torch.BoolTensor(1).fill_(False).to(samples.device)
        res = model.w2v_encoder.w2v_model.extract_features(source=samples.view(1,-1), padding_mask=poker_face, mask=False)  
        y = res[0] 
    else:
        res = model(samples.view(1,-1), padding_mask=None, mask=False, features_only=True)
        y = res['x']

    return y

def extract_features_from_splits(infos, threshold, model, samples):

    chunks = split_wav_signal(infos, samples, threshold)
    feats = []
    for c in chunks:
        ff = extract_w2v2_features(model, c)
        feats.append(ff)

    tsr = torch.cat( feats, dim=1 )

    return tsr

def main():
    upsample = False
    new_sample_rate = 16000
    upsample_factor = 2
    cuda = False
    extract_features=False
    model_size = 'large'
    add_ext=True
    input_ext = '.wav'
    file_ext = '.tmp-tokSLU'
    duration_threshold = 28.0
    duration_margin = 1.0
    chunk_overlap = 0.020
    overwrite_arguments = {}
    device = 'cuda:1' if cuda else 'cpu' 
 
    prefix_list='/home/getalp/dinarelm/work/data/PortMEDIA/DialogTextAndWav/test_prefixes.lst'

    f = open(prefix_list, encoding='utf-8')
    lines = f.readlines()
    f.close()
    prefixes = [l.rstrip() for l in lines]

    sv_starting_point='/home/getalp/dinarelm/work/data/Exchance/wav2vec/models/FlowBERT-7K_large.pt'    
    flowbert_path='/home/getalp/dinarelm/work/data/models/wav2vec2.0/MEDIA-finetuned/MEDIA-fromFlowBERT-7klarge_SLU-token-tuned.pt'  

    feat_norm = True if model_size == 'large' else False
    if feat_norm and extract_features:
        print(' - Normalizing {} model features...'.format(model_size))
        sys.stdout.flush()
    if upsample:
        pp = prefixes[0]
        if add_ext:
            pp = pp + input_ext
        sample_rate, samples = wavfile.read(pp, mmap = True)
        upsample_factor = int(new_sample_rate / sample_rate)

        print(' - Upsampling input signals from {} to {} samples per second...'.format(sample_rate, new_sample_rate))
        sys.stdout.flush()

    if extract_features:
        print(' - Loading model...')
        sys.stdout.flush()

        if sv_starting_point is not None:
            overwrite_arguments['model'] = {'w2v_path': sv_starting_point}
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([flowbert_path], overwrite_arguments)
        model = model[0] 
        model.eval()
        if cuda:
            model = model.cuda()
    else:
        model = None

    for prefix in prefixes:
        if add_ext:
            wav_file = prefix + input_ext
        else:
            wav_file = prefix
            prefix = prefix[:-4]

        print(' - Processing file {}'.format(wav_file.split('/')[-1]))
        sys.stdout.flush()

        sample_rate, samples = wavfile.read(wav_file,mmap = True)
        if len(samples.shape) > 2:
            raise NotImplementedError(' Not defined yet how to extract features from multiple (>2) channel signals')
        if len(samples.shape) > 1: 
            n1 = np.linalg.norm(samples[:,0], axis=0)
            n2 = np.linalg.norm(samples[:,1], axis=0)
 
            if n1 > n2:
                samples = samples[:,0]
            else:
                samples = samples[:,1]

        assert len(samples.shape) == 1
        if upsample:
            resamples = signal.resample(samples, samples.shape[0]*upsample_factor)  # Up-sampling to 16kHz
            assert len(resamples.shape) == 1 and len(resamples) == upsample_factor*len(samples)
            samples = resamples
            sample_rate = upsample_factor*sample_rate
            assert sample_rate == new_sample_rate

        if extract_features:
            samples = torch.from_numpy(samples).float().squeeze()
            if cuda:
                samples = samples.cuda()
            if feat_norm:
                samples = F.layer_norm(samples, samples.shape) 

            duration = len(samples)/sample_rate
            if duration > duration_threshold + duration_margin:
                extract_params = {'sample_rate': sample_rate, 'split_overlap': chunk_overlap}
                y = extract_features_from_splits(extract_params, duration_threshold, model, samples)
            else:
                y = extract_w2v2_features(model, samples) 

            torch.save(y.detach().to('cpu'), prefix + file_ext, _use_new_zipfile_serialization=False)
        elif upsample:
            wavfile.write(prefix + file_ext, sample_rate, samples)
        else:
            print('   * {}: samples {}, duration {} sec.'.format(wav_file.split('/')[-1], len(samples), len(samples)/sample_rate))
            feat_file = prefix + file_ext
            if os.path.isfile(feat_file):
                tt = torch.load(feat_file)
                print('   * {}: feature tensor size: {}'.format('/'.join(wav_file.split('/')[-3:]), tt.size()))

if __name__ == '__main__':
    main()

