# Automatic Speech-to-text Translation (AST)

Automatic speech-to-text translation (AST) consists in translating a speech utterance in a source language to a text in a target language. Here we are interested in translating directly from speech in French to text in another language. In the following, we describe the steps to reproduce our AST results presented in the paper (Section 5.3). 

# Table of Contents
**1. [AST results](#1-ast-results)**  
**2. [Dataset and installation](#2-dataset-and-installation)**   
&nbsp;&nbsp;&nbsp;&nbsp;2.1. [Dataset](#21-dataset)   
&nbsp;&nbsp;&nbsp;&nbsp;2.2. [Installation](#22-installation)  
**3. [Feature preparation](#3-feature-preparation)**  
&nbsp;&nbsp;&nbsp;&nbsp;3.1. [Task-agnostic pre-training](#31-task-agnostic-pre-training)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.1. [log-Mel filterbank features](#311-log-mel-filterbank-features)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.2. [wav2vec features](#312-wav2vec-features)   
&nbsp;&nbsp;&nbsp;&nbsp;3.2. [Self-supervised fine-tuning on mTEDx](#32-self-supervised-fine-tuning-on-mtedx)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.1. [Perform self-supervised fine-tuning on mTEDx](#321-perform-self-supervised-fine-tuning-on-mtedx)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.2. [Extract features from obtained wav2vec models](#322-extract-features-from-obtained-wav2vec-models)   
&nbsp;&nbsp;&nbsp;&nbsp;3.3. [Supervised fine-tuning for ASR on mTEDx](#33-supervised-fine-tuning-for-asr-on-mtedx)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.1. [Perform supervised fine-tuning for ASR on mTEDx](#331-perform-supervised-fine-tuning-for-asr-on-mtedx)   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.2. [Extract features from obtained wav2vec models](#332-extract-features-from-obtained-wav2vec-models)   
**4. [Training ST models](#4-training-st-models)**  
**5. [Decoding](#5-decoding)**  

# 1. AST results
The following table (corresponding to Table 5 in the paper) shows the BLEU scores on the valid and test sets of multilingual TEDx (mTEDx) using different types of speech features. Note that these results are obtained from **bilingual** ST models trained on the respective datasets.

The baselines in our experiments are models using log-Mel filterbank features (`MFB`). For models using wav2vec features, there are 3 main blocks corresponding to features extracted from ***(a) task-agnostic pre-training***, *(b) self-supervised fine-tuning on mTEDx*, and *(c) supervised fine-tuning for ASR on mTEDx*. The two latter methods belong to the ***task-specific pre-training*** category. The highest value in each block is <ins>underlined</ins>, while the best value in each column is highlighted in **bold**.

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="3">Valid data</th>
      <th colspan="3">Test data</th>
	  <th colspan="2">Links to models</th>
    </tr>
  </thead>
    <thead>
    <tr>
      <th>Input features</th>
      <th>fr-en</th>
      <th>fr-es</th>
      <th>fr-pt</th>
      <th>fr-en</th>
      <th>fr-es</th>
      <th>fr-pt</th>
      <th>wav2vec</th>
	  <th>ST model</th>
    </tr>
    </thead>
  <tbody>
    <tr>
	 <td>MFB</td>
	 <td>1.15</td>
	 <td>0.67</td>
	 <td>0.61</td>
	 <td>1.10</td>
	 <td>0.87</td>
	 <td>0.32</td>
	 <td>Na</td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_log_mel_fbank.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_log_mel_fbank.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_log_mel_fbank.pt?download=1>fr-pt</a></td>
</tr>
    <tr>
      <th colspan="9">(a) Task agnostic pre-training</th>
    </tr>
<tr>
	 <td>En-base</td>
	 <td>5.54</td>
	 <td>1.30</td>
	 <td>0.54</td>
	 <td>5.20</td>
	 <td>1.47</td>
	 <td>0.38</td>
	 <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_en_base.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_en_base.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_en_base.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>En-large</td>
	 <td>4.11</td>
	 <td>1.67</td>
	 <td>0.32</td>
	 <td>3.56</td>
	 <td>2.29</td>
	 <td>0.43</td>
	 <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_en_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_en_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_en_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-1K-base</td>
	 <td>9.18</td>
	 <td>5.09</td>
	 <td>0.39</td>
	 <td>8.98</td>
	 <td>5.64</td>
	 <td>0.49</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-1K-base>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_fr_1k_base.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_fr_1k_base.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_fr_1k_base.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-1K-large</td>
	 <td>15.31</td>
	 <td>13.74</td>
	 <td>8.29</td>
	 <td>14.46</td>
	 <td>14.77</td>
	 <td>9.37</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-1K-large>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_fr_1k_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_fr_1k_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_fr_1k_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-2.6K-base</td>
	 <td>15.09</td>
	 <td>13.27</td>
	 <td>4.72</td>
	 <td>14.69</td>
	 <td>14.04</td>
	 <td>5.51</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-2.6K-base>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_fr_2.6k_base.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_fr_2.6k_base.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_fr_2.6k_base.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-3K-base</td>
	 <td>15.05</td>
	 <td>13.19</td>
	 <td>4.44</td>
	 <td>14.80</td>
	 <td>14.27</td>
	 <td>4.72</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-base>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_fr_3k_base.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_fr_3k_base.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_fr_3k_base.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-3K-large</td>
	 <td>17.94</td>
	 <td>16.40</td>
	 <td>8.64</td>
	 <td>18.00</td>
	 <td>18.12</td>
	 <td>9.55</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_fr_3k_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_fr_3k_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_fr_3k_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-7K-base</td>
	 <td>15.13</td>
	 <td>12.78</td>
	 <td>2.65</td>
	 <td>14.50</td>
	 <td>13.61</td>
	 <td>2.66</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-base>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_fr_7k_base.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_fr_7k_base.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_fr_7k_base.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-7K-large</td>
	 <td><ins>19.23</td>
	 <td><ins>17.59</td>
	 <td><ins>9.68</td>
	 <td><ins>19.04</td>
	 <td><ins>18.24</td>
	 <td><ins>10.98</td>
	 <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_fr_7k_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_fr_7k_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_fr_7k_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>XLSR-53-large</td>
	 <td>7.81</td>
	 <td>0.49</td>
	 <td>0.43</td>
	 <td>6.75</td>
	 <td>0.52</td>
	 <td>0.36</td>
	 <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_xlsr53.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_xlsr53.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_xlsr53.pt?download=1>fr-pt</a></td>
</tr>
	<tr>
      <th colspan="9">(b) Task specific pre-training (self-supervised on mTEDX)</th>
    </tr>
<tr>
	 <td>Fr-3K-large</td>
	 <td>18.54</td>
	 <td>16.40</td>
	 <td>8.81</td>
	 <td>18.38</td>
	 <td>17.84</td>
	 <td>10.57</td>
	 <td><a href=https://zenodo.org/record/5502094/files/mtedx_fr2en_self_supervised_ft_w2v2fr_3k_large.pt?download=1>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_self_supervised_ft_3k_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_self_supervised_ft_3k_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_self_supervised_ft_3k_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-7K-large</td>
	 <td><ins>19.65</td>
	 <td><ins>17.53</td>
	 <td><ins>9.35</td>
	 <td><ins>19.36</td>
	 <td><ins>18.95</td>
	 <td><ins>10.94</td>
	 <td><a href=https://zenodo.org/record/5502094/files/mtedx_fr2en_self_supervised_ft_w2v2fr_7k_large.pt?download=1>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_self_supervised_ft_7k_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_self_supervised_ft_7k_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_self_supervised_ft_7k_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>XLSR-53-large</td>
	 <td>6.83</td>
	 <td>0.54</td>
	 <td>0.34</td>
	 <td>6.75</td>
	 <td>0.34</td>
	 <td>0.29</td>
	 <td><a href=https://zenodo.org/record/5502094/files/mtedx_fr2en_self_supervised_ft_xlsr53.pt?download=1>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_self_supervised_ft_xlsr53.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_self_supervised_ft_xlsr53.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_self_supervised_ft_xlsr53.pt?download=1>fr-pt</a></td>
</tr>
</tr>
	<tr>
      <th colspan="9">(c) Task specific pre-training (fine-tuned for ASR on mTEDX)</th>
    </tr>
<tr>
	 <td>Fr-3K-large</td>
	 <td>21.09</td>
	 <td>19.28</td>
	 <td>14.40</td>
	 <td>21.34</td>
	 <td>21.18</td>
	 <td>16.66</td>
	 <td><a href=https://zenodo.org/record/5502094/files/mtedx_fr2en_supervised_ft_w2v2fr_3k_large.pt?download=1>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_supervised_ft_3k_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_supervised_ft_3k_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_supervised_ft_3k_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>Fr-7K-large</td>
	 <td><b>21.41</td>
	 <td>20.32</td>
	 <td><b>15.14</td>
	 <td><b>21.69</td>
	 <td><b>21.57</td>
	 <td><b>17.43</td>
	 <td><a href=https://zenodo.org/record/5502094/files/mtedx_fr2en_supervised_ft_w2v2fr_7k_large.pt?download=1>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_supervised_ft_7k_large.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_supervised_ft_7k_large.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_supervised_ft_7k_large.pt?download=1>fr-pt</a></td>
</tr>
<tr>
	 <td>XLSR-53-large</td>
	 <td>21.09</td>
	 <td><b>20.38</td>
	 <td>14.56</td>
	 <td>20.68</td>
	 <td>21.14</td>
	 <td>17.21</td>
	 <td><a href=https://zenodo.org/record/5502094/files/mtedx_fr2en_supervised_ft_xlsr53.pt?download=1>Download</a></td>
	 <td><a href=https://zenodo.org/record/5502207/files/mtedx_fr2en_supervised_ft_xlsr53.pt?download=1>fr-en</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2es_supervised_ft_xlsr53.pt?download=1>fr-es</a>,<a href=https://zenodo.org/record/5502207/files/mtedx_fr2pt_supervised_ft_xlsr53.pt?download=1>fr-pt</a></td>
</tr>
  </tbody>
</table>


[`En-base/large`](https://arxiv.org/abs/2006.11477) and [`XLSR-53`](https://arxiv.org/abs/2006.13979) are off-the-shelf wav2vec models trained on English and multilingual speech, respectively. The ones whose prefixes are `Fr` are the wav2vec models that we trained on our collected French datasets of different sizes (1K, 2.6K, 3K, and 7K). Except for the one trained on 2.6K hours, each model has both `base` and `large` configurations.

**NOTE:** For the two task-specific pre-training methods (self-supervised and supervised fine-tuning on mTEDx), since the French speech is overlapped between the language pairs, we selected the pair having the most speech data (`fr-en`) to perform task-specific pre-training and used the obtained models to extract features for the remaining pairs (`fr-es` and `fr-pt`). For a fair comparison, we did not use additional data augmentation technique nor ASR encoder pre-training in the experiments.


# 2. Dataset and installation
## 2.1. Dataset
We selected subsets having French as the source language in the large multilingual speech-to-text dataset [multilingual TEDx](https://arxiv.org/abs/2102.01757). 
Our benchmark covers translation directions from French (`fr`) to three target languages: English (`en`), Portugese (`pt`), and Spanish (`es`). 
The training sizes (in hours) are shown in the following table.

| Dataset     | fr-en | fr-es | fr-pt | Link |
| ----- | ----- | ----- | ----- | ----- |
| mTEDx       |   50      |  38   | 25    | [Download](https://www.openslr.org/resources/100) |

After downloading data, please unzip and save them under `${MTEDX_ROOT}`.


## 2.2. Installation
The experiments are performed using `Python 3.8.2`, `torch 1.8.1`, `torchaudio 0.8.1`. Our implementation is based on [fairseq S2T](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text). 
Please clone [our fork](https://github.com/formiel/fairseq/tree/LeBenchmark) (`LeBenchmark` branch)
as there are modifications made for LeBenchmark:

```bash
git clone https://github.com/formiel/fairseq.git
cd fairseq
git checkout LeBenchmark
```

Then install it in your environment:
```bash
pip install -e . 
```
(remove `-e` in the above if you don't want to install in the editable mode).

In addition, please also install NVIDIA's `apex` library as instructed in [`fairseq`](https://github.com/pytorch/fairseq#requirements-and-installation).
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

Finally, the following libraries are necessary for this recipe: **sndfile**, **ffmpeg**, **pandas**, **soundfile**, **sentencepiece** and **torchaudio**. These can be installed as follows.

* With sudo privileges:
```bash
sudo apt-get install libsndfile1-dev ffmpeg 
pip install pandas soundfile sentencepiece torchaudio
```

* On a virtual environment:
```bash
conda install -y libsndfile ffmpeg
pip install pandas soundfile sentencepiece torchaudio
```

# 3. Feature preparation
In the following, please have separate `${MTEDX_ROOT}` for each type of features since the output files (features, manifest, dictionary etc.) will be overwitten if they are under the same folder. For example, below is the structure of the downloaded data after extracting:

```bash
${DOWNLOAD_DIR}
└──fr-en
    └──data
    	└──train
	└──valid
	└──test
└──fr-es
    └──data
    	└──train
	└──valid
	└──test
...
```
Then you can create a folder named `${MTEDX_ROOT}` having similar structure as above for each type of feature and create symlinks to the data downloaded for each ${MTEDX_ROOT} folder as below.
```bash
${MTEDX_ROOT}
└──fr-en
    └──data -> ${DOWNLOAD_DIR}/fr-en/data
└──fr-es
    └──data -> ${DOWNLOAD_DIR}/fr-es/data
...
```

## 3.1. Task-agnostic pre-training
### 3.1.1. log-Mel filterbank features
```bash
python examples/speech_to_text/prep_mtedx_data.py --data-root ${MTEDX_ROOT} \
						  --vocab-type unigram \
						  --vocab-size 1000 \
						  --task st
```
### 3.1.2. wav2vec features
Before extracting features from wav2vec models, it is necessary to convert `.flac` files to `.wav` files. You can use `ffmpeg` for such conversion under the `AST/tools` directory in this repo.
```bash
bash tools/flac2wav.sh $FLAC_DIR ${MTEDX_ROOT}/wav
```
where `$FLAC_DIR` is path to the directory containing `.flac` files. 

```bash
python examples/speech_to_text/prep_mtedx_data_w2v_feats.py \
	--data-root ${MTEDX_ROOT} \
	--vocab-type unigram --vocab-size 1000 --task st \
	--use-w2v-feats \
	--w2v-path ${W2V2_PATH} \
	--src fr --tgt ${TGT_LANG}
```
where:
- `${W2V2_PATH}` is path to the wav2vec 2.0 model from which you want to extract features, 
- `${TGT_LANG}` is chosen among `[en, es, pt]`.

**IMPORTANT:** If you extract features from `large` models, please add `--normalize-signal` to the above command line.


## 3.2. Self-supervised fine-tuning on mTEDx
The input to wav2vec models needs to be single channel with sampling rate of 16kHz. Therefore, we first need to downsample the audio files before training. Similar to [Section 3.1.2](#312-wav2vec-features), you can run the following command to convert `.flac` files to `.wav` files.

```bash
bash tools/flac2wav.sh $FLAC_DIR ${MTEDX_ROOT}/wav
```
where `$FLAC_DIR` is path to the directory containing `.flac` files. 

### 3.2.1. Perform self-supervised fine-tuning on mTEDx
#### (1) Split audio files
Since it is recommended to split each file into separate files each having smaller length when training wav2vec models, we first split the audio files (for each talk) into smaller files (each containing one sentence) based on the segment information provided in the released mTEDx dataset.
```bash
bash examples/speech_to_text/split_wav_files.sh ${INPUT_DIR} ${OUTPUT_SPLIT_DIR} ${SEGMENT_FILE}
```
where 
- `${INPUT_DIR}` is path to the folder containing audio files to be split,
- `${OUTPUT_SPLIT_DIR}` is where you want to store the resulting split files,
- `${SEGMENT_FILE}` is path to the segment file.


#### (2) Prepare input data for training
The input `.tsv` file to wav2vec training has the following format: 
```
/path/to/audio/folder
filename0.wav	nframes
filename1.wav	nframes
...
```
To prepare data according to this format, please run the following command to first obtain the manifest files
```bash
python examples/speech_to_text/prep_mtedx_data_w2v_feats.py \
	--data-root ${MTEDX_ROOT} \
	--vocab-type unigram --vocab-size 1000 \
	--task st --src fr  --tgt en \
	--get-manifest-only
```
Then run
```bash
python examples/speech_to_text/prep_ft_w2v2.py --audio-root ${AUDIO_ROOT} --tsv-path ${TSV_PATH} --dest ${DATA_DIR}
```
where 
- `${AUDIO_ROOT}` is path to the folder where split audio files from (1) are saved, 
- `${TSV_PATH}` is path to the `.tsv` files (`train_st.tsv`, `valid_st.tsv`, and `test_st.tsv`) obtained as above, 
- `${DATA_DIR}` is where you want to store the output files, including the `.tsv` file having the above format, the `.ltr` and `.wrd` files which include the transcripts pre-tokenized at the letter and word level, repectively. This `${DATA_DIR}` will be the input folder for task-specific pre-training (both self-supervised and supervised one).


#### (3) Training wav2vec model
**NOTE:** The self-supervised fine-tuning on mTEDx is resumed from the last optimizer's state of the corresponding pre-trained model, hence the number of updates will be picked up from where it left off previously. For example, your self-supervised fine-tuning should start at step around 180k for `Fr-1K-base`, 158k for `Fr-1K-large`, and around 496K or 500K for the remaining wav2vec `Fr` models. The `max_update` in the configuration file is hence the sum of previous training steps in the pre-trained model and the training steps to be performed on the task data. All of the self-supervised fine-tuned models in our experiments were trained for an additional 20K steps on `fr-en` pair of mTEDx.

To perform self-supervised fine-tuning on mTEDx, please run the following command:
```bash
fairseq-hydra-train \
	common.tensorboard_logdir=${TENSORBOARD_DIR} \
	checkpoint.save_dir=${SAVE_DIR} \
	checkpoint.restore_file=${PRETRAINED_W2V2_PATH} \
	checkpoint.reset_meters=true \
	task.data=${DATA_DIR} \
	--config-dir NeurIPS2021/AST/configs \
	--config-name ${MODEL_CONFIG}
```
where 
- `$TENSORBOARD_DIR$` is path to save the tensorboard, 
- `${SAVE_DIR}` is path to save the checkpoints, 
- `${MODEL_CONFIG}` is the training configuration. The main hyperparameters are the same as in `example/wav2vec/config/pretraining/wav2vec2_large_librivox.yaml` for self-supervised fine-tuning and `example/wav2vec/config/pretraining/vox100h.yaml` for supervised fine-tuning. Please refer to `NeurIPS2021/AST/configs` for the configuration files that we used in our experiments.

**IMPORTANT:** If your are resuming the training from a previous job (in case the previous run is stopped for some reason, such as time limit etc.), please modify the `checkpoint.restore_file` to be the last checkpoint (`checkpoint_last.pt`) of the previous training so that the model continues to train properly. Otherwise, it will load the pre-trained model from `${PRETRAINED_W2V2_PATH}` and run the previous training again.


### 3.2.2. Extract features from obtained wav2vec models
Please follow [Section 3.1.2](#312-wav2vec-features) to extract features from the obtained self-supervised fine-tuned wav2vec models. `$W2V2_PATH` is the path to the best checkpoint (`checkpoint_best.pt`) obtained from the training in [Section 3.2.1](#321-perform-self-supervised-fine-tuning-on-mtedx).


## 3.3. Supervised fine-tuning for ASR on mTEDx
### 3.3.1. Perform supervised fine-tuning for ASR on mTEDx
Please follow step (1) and (2) in [Section 3.2.1](#321-perform-self-supervised-fine-tuning-on-mtedx) to split the audio and prepare the `.tsv` and `.ltr` files for training.

#### (3) Learn dictionary
For supervised fine-tuning, we also need to have the dictionary. To learn the dictionary on the transcripts, please run the following command:
```bash
fairseq-preprocess --dataset-impl mmap --trainpref ${DATA_DIR}/train.ltr  --only-source  --thresholdsrc 0
```
then copy the obtained dictionary to `$DATA_DIR/dict.ltr.txt`.

#### (4) Training wav2vec model
To perform supervised fine-tuning for ASR on mTEDx, please run the following command:
```bash
fairseq-hydra-train \
	common.tensorboard_logdir=${TENSORBOARD_DIR} \
	checkpoint.save_dir=${SAVE_DIR} \
	task.data=${DATA_DIR} \
	model.w2v_path=${PRETRAINED_W2V2_PATH} \
	--config-dir NeurIPS2021/AST/configs \
	--config-name ${MODEL_CONFIG}
```


### 3.3.2. Extract features from obtained wav2vec models
Please refer to [Section 3.1.2](#312-wav2vec-features) for the feature extraction step.

**NOTE:** Please add `--w2v-ctc` to the command line in [Section 3.1.2](#312-wav2vec-features) to extract features from supervised fine-tuned wav2vec models.


# 4. Training ST models

To train a speech-to-text translation model on the extracted features,
run the following command:

```bash
fairseq-train ${MTEDX_ROOT}/${LANG_PAIR} \
	--train-subset train_st \
	--valid-subset valid_st\
	--config-yaml config_st.yaml \
	--save-dir ${ST_SAVE_DIR} \
	--num-workers 4 \
	--max-tokens 40000 \
	--max-source-positions 150000 \
	--max-target-positions 8192 \
	--task speech_to_text \
	--criterion label_smoothed_cross_entropy \
	--report-accuracy \
	--max-epoch 500 \
	--arch s2t_transformer_xs \
	--optimizer adam \
	--lr 2e-3 \
	--lr-scheduler inverse_sqrt \
	--warmup-updates 10000 \
	--clip-norm 10.0 \
	--seed 1 \
	--log-interval 1000 \
	--update-freq 8 \
	--tensorboard-logdir ${TENSORBOARD_DIR}
```
where 
- `${LANG_PAIR}` is the language pair (for example, `fr-en`, `fr-es`, or `fr-pt`) on which to train the models. 
- `${ST_SAVE_DIR}` is the path to save checkpoints.

**IMPORTANT:** 
1. Please add `--use-linear-before-cnn` when training ST models using features extracted from wav2vec models.
2. **Multi-GPU training:** Training on multiple GPUs requires some modifications of the above command:
- Replace `fairseq-train` with `python -u -m torch.distributed.launch --nproc_per_node=${NGPUS_PER_NODE} $(which fairseq-train)` where `${NGPUS_PER_NODE}` is the number of GPUs.
- Scale the effective batch size accordingly. For example, on 4 GPUs, you can set `--update-freq 2` (instead of `--update-freq 8`).


# 5. Decoding
To decode using a trained model (with weight-averaging over the last 10 checkpoints), run the following commands:

```bash
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
python scripts/average_checkpoints.py \
    --inputs ${ST_SAVE_DIR} \
    --num-epoch-checkpoints 10 \
    --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${MTEDX_ROOT}/${LANG_PAIR} \
    --config-yaml config_st.yaml \
    --gen-subset ${GEN_SUBSET} \
    --task speech_to_text \
    --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --max-tokens 50000 --beam 5 --scoring sacrebleu \
    --results-path ${RESULT_PATH}
```
where:
- `${GEN_SUBSET}` is the name of the subset you want to decode.
- `${RESULT_PATH}` is the path where you want to save the decoding results.
