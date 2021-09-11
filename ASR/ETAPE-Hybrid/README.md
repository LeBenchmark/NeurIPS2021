# Hybrid DNN-HMM ASR models 
Under construction...

Recipe scripts to train TDNN-F models in [Kaldi](https://github.com/kaldi-asr/kaldi) using hires and [Fairseq](https://github.com/pytorch/fairseq) wav2vec (2.0) features on the ETAPE-1,2 French corpus.

## Data

ETAPE-1,2 French corpus: 30 hours of French radio and TV data

https://catalogue.elra.info/en-us/repository/browse/ELRA-E0046/

## Results

#### Results obtained using a scoring script ...link to be added... (with normalization)  and reported in https://openreview.net/pdf?id=TSvj5dmuSd, Table 2:

...table to be added...

#### Results with the default Kaldi scoring (without normalization):

<table>
  <thead>
    <tr>
      <th colspan="1">Language Model (LM)</th>
      <th colspan="2">LM ETAPE</th>
      <th colspan="2">LM ESTER-1,2+EPAC</th>
    </tr>
  </thead>
  <thead>
    <tr>
      <th>Features</th>
      <th>Dev</th>
      <th>Test</th>
      <th>Dev</th>
      <th>Test</th>
    </tr>
  </thead>
   
  <tbody>
   <tr>
    <td>hires MFCC</td>
    <td>39.28</td>
    <td>40.89</td>
    <td>35.60</td>
    <td>37.73</td>
   </tr>
     <tr>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt>W2V2-En-large</a></td>
    <td>39.93</td>
    <td>42.30</td>
    <td>36.18</td>
    <td>38.75</td>
   </tr>
   <tr>
    <td><a href=https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt>XLSR-53-large</a></td>
    <td>36.36</td>
    <td>38.19</td>
    <td>32.81</td>
    <td>35.17</td>
   </tr>
    <tr> 
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-1K-base>W2V2-Fr-1K-base</a></td>
    <td>40.79</td>
    <td>43.75</td>
    <td>37.61</td
    <td>41.11</td>
   </tr>
    <tr> 
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-1K-large>W2V2-Fr-1K-large</a></td>
    <td>39.84</td>
    <td>42.06</td>
    <td>36.31</td>
    <td>39.09</td>
   </tr>
   <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-2.6K-base>W2V2-Fr-2.6K-base</a></td>
    <td>33.83</td>
    <td>35.78</td>
    <td>30.34</td
    <td>33.05</td>
   </tr>
   <tr> 
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-3K-large>W2V2-Fr-3K-large</a></td>
    <td>32.19</td>
    <td>33.87</td>
    <td>28.53</td>
    <td>30.77</td>
   </tr>
    <tr> 
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-base>W2V2-Fr-7K-base</a></td>
    <td>33.47</td>
    <td>35.09</td>
    <td>30.16</td>
    <td>32.65</td>
   </tr>
   <tr> 
    <td><a href=https://huggingface.co/LeBenchmark/wav2vec2-FR-7K-large>W2V2-Fr-7K-large</a></td>
    <td>30.08</td>
    <td>31.52</td>
    <td>26.58</td>
    <td>28.73</td>
   </tr>
  </tbody>
</table>
