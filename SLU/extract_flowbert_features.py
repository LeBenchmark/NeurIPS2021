
import os
import sys

from scipy import signal
from scipy.io import wavfile
import numpy as np

import torch
import fairseq
import torch.nn.functional as F
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc

def split_wav_signal(infos, signal, chunk_duration):

    overlap = infos['split_overlap']
    sample_rate = infos['sample_rate']

    chunks = []
    offset = 0
    bound = int(chunk_duration*sample_rate) 

    overlap_samples = int(overlap * sample_rate)
    while bound <= len(signal):
        chunks.append( signal[offset:bound] )

        #print(' extracted chunk between {} - {}'.format(offset, bound))
        #sys.stdout.flush()

        if bound == len(signal):
            return chunks
        offset += (int(chunk_duration*sample_rate) - overlap_samples)
        bound = min(len(signal), offset+int(chunk_duration*sample_rate))

        #print(' shifted offset and bound: {} - {}'.format(offset, bound))
        #sys.stdout.flush()

    #print(' chunking done.')
    #sys.stdout.flush()

    return chunks

def extract_w2v2_features(model, samples):

    if isinstance(model, Wav2VecCtc):
        #print(' - extracting features with tuned model...')
        #sys.stdout.flush()

        poker_face = torch.BoolTensor(1).fill_(False).to(samples.device)
        res = model.w2v_encoder.w2v_model.extract_features(source=samples.view(1,-1), padding_mask=poker_face, mask=False)
        y = res[0]
    else:
        #print(' - extracting features with w2v2 model...')
        #sys.stdout.flush()

        res = model(samples.view(1,-1), padding_mask=None, mask=False, features_only=True)
        y = res['x']

    return y

def extract_features_from_splits(infos, threshold, model, samples):

    chunks = split_wav_signal(infos, samples, threshold)

    #print(' - extracting features...')
    #sys.stdout.flush()

    feats = []
    for c in chunks:
        ff = extract_w2v2_features(model, c)

        #print(' extracted features of size: {}'.format(ff.size()))

        feats.append(ff)

    tsr = torch.cat( feats, dim=1 )

    #print(' returning features of size: {}'.format(tsr.size()))
    #sys.stdout.flush()

    return tsr

def main():
    upsample = True
    cuda = False
    extract_features=True
    model_size = 'large'
    add_ext=True
    file_ext = '.7kl-tslu'
    duration_threshold = 30.0
    duration_margin = 1.0
    chunk_overlap = 0.040
    overwrite_arguments = {}
    device = 'cuda:1' if cuda else 'cpu'

    prefix_list='/home/getalp/dinarelm/work/data/MEDIA-Original/semantic_speech_aligned_corpus/media_all.lst'
    #prefix_list='/home/getalp/dinarelm/work/data/FluentSpeechCommands/fluent_speech_commands_dataset/wavs/all_prefixes.lst'
    #prefix_list='/home/getalp/dinarelm/work/data/PortMEDIA/DialogTextAndWav/all_prefixes.lst'

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
        print(' - Upsampling input signals...')
        sys.stdout.flush()

    if extract_features:
        print(' - Loading model...')
        sys.stdout.flush()

        if sv_starting_point is not None:
            overwrite_arguments['model'] = {'w2v_path': sv_starting_point}
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([flowbert_path], overwrite_arguments) #, arg_overrides=overwrite_arguments)
        model = model[0]
        #state = torch.load(flowbert_path)
        #model.load_state_dict( state['model'] )
        model.eval()
        model = model.to(device)
    else:
        model = None

    for prefix in prefixes:
        if add_ext:
            wav_file = prefix + '.wav'
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

        if upsample:
            samples = signal.resample(samples, samples.shape[0]*2)  # Up-sampling to 16kHz
            sample_rate = 2*sample_rate

        if extract_features:
            samples = torch.from_numpy(samples).float().squeeze() 
            samples = samples.to(device)
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

