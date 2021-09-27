# LeBenchmark: Data preprocessing 

This folder contains all the scripts necessary for transforming the raw data from the different french corpora into json files, input of the SSL training. Note that this process includes not only data normalization, but also audio normalization.

The following table presents the downloading links and licenses for the corpora.

| **Corpus (subcorpus) name**        | **Identifier (ISLRN, DOI...)** | **Size\*** |  **Modality**   | **Dataset use** | **License**       |
|:-----------------------------------|:-------------------------------|:----------:|:---------------:|:---------------:|:------------------|
| [African Accented French][]        | SLR57                          |    22 h    | speech, written |       SSL       | Apache 2.0        |
| [Allosat][]                        |                                |    37 h    | speech, written |       AER       | CC                |
| [Att-HACK][]                       | SLR88                          | \>300 sent | speech, written |       SSL       | CC BY-NC-ND       |
| [CaFE][]                           | 10.5281/zenodo.1478765         |    1 h     | speech, written |       SSL       | CC-BY-NC-SA 4.0   |
| [CFPP2000 (CEFC complement)][]     |                                |    20 h    | speech, written |       SSL       | CC BY-NC-SA 3.0   |
| [CommonVoice fr_604h_2020-06-22][] |                                |   604 h    | speech, written |       ASR       | CC 0              |
| [EPAC][]                           | 483-703-007-740-8              |   1677 h   | speech, written |       SSL       | ELRA NC           |
| [ESLO (ESLO2)][]                   |                                |  \>400 h   | speech, written |       SSL       | CC BY-NC-SA 4.0   |
| [ETAPE][]                          | 425-777-374-455-4              |    30 h    | speech, written |       ASR       | ELRA NC           |
| [GEMEP][]                          |                                |   0.9 h    |     speech      |       SSL       | academic only, NC |
| [MaSS][]                           |                                |   ≈ 20 h   | speech, written |       SSL       | MIT License       |
| [MEDIA][]                          | 699-856-029-354-6              |  1,258 d   | speech, written |       SLU       | ELRA NC           |
| [MLS (French)][]                   |                                |  1,096 h   | speech, written |       SSL       | CC BY 4.0         |
| [MPF][]                            |                                |    78 h    | speech, written |       SSL       | CC BY-NC-SA 4.0   |
| [mTEDx (fr-\*)][]                            | SLR100            | 25h - 50h |   speech, written   | AST | CC BY-NC-ND 4.0            |
| [NCCFr][]                                    |                   |   35 h    | Multimedia, written | SSL | academic only, NC          |
| [Portmedia (PM_DOM)][]                       | 135-793-959-390-8 |  40.5 h   |   speech, written   | SSL | ELRA NC                    |
| [RECOLA][]                                   |                   |   9.5 h   | Multimedia, written | AER | End User License Agreement |
| [TCOF][]                                     |                   |   146 h   |   speech, written   | SSL | CC BY-NC-SA                |
| [Voxpopuli unlabeled][]                      |                   | ≈ 4.5k h  |       speech        | SSL | CC0                        |
| [Voxpopuli transcribed][Voxpopuli unlabeled] |                   |  ≈ 215 h  |   speech, written   | SSL | CC0                        |

  [African Accented French]: https://www.openslr.org/57/
  [Allosat]: https://lium.univ-lemans.fr/allosat/
  [Att-HACK]: http://www.openslr.org/88/
  [CaFE]: https://zenodo.org/record/1478765#.YR5ZlFs6-00
  [CFPP2000 (CEFC complement)]: http://cfpp2000.univ-paris3.fr/index.html
  [CommonVoice fr_604h_2020-06-22]: https://commonvoice.mozilla.org/en/datasets
  [EPAC]: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0305/
  [ESLO (ESLO2)]: http://eslo.huma-num.fr/index.php
  [ETAPE]: https://catalogue.elra.info/en-us/repository/browse/ELRA-E0046/
  [GEMEP]: https://www.unige.ch/cisa/gemep
  [MaSS]: https://github.com/getalp/mass-dataset
  [MEDIA]: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0272/
  [MLS (French)]: http://www.openslr.org/94/
  [MPF]: https://www.ortolang.fr/market/corpora/mpf
  [MPF]: https://www.ortolang.fr/market/corpora/mpf
  [mTEDx (fr-\*)]: http://www.openslr.org/100
  [NCCFr]: https://mirjamernestus.nl/Ernestus/NCCFr/index.php
  [Portmedia (PM_DOM)]: https://catalogue.elra.info/en-us/repository/browse/ELRA-S0371/
  [RECOLA]: https://diuf.unifr.ch/main/diva/recola/download.html
  [TCOF]: https://www.ortolang.fr/market/corpora/tcof?path=%2FCorpus%2FAdultes
  [Voxpopuli unlabeled]: https://github.com/facebookresearch/voxpopuli


- For questions regarding the preprocessing of **African Accented French, Att_Hack, CaFE, ESLO, PORTMEDIA**. 

Contact: _sina.alisamir at univ-grenoble-alpes.fr_


- For questions regarding the preprocessing of **MLS, CFPP2000, TCOF, VOXPOPULI**.

Contact: _solene.evain at univ-grenoble-alpes.fr_


- For questions regarding the preprocessing of **EPAC, MPF, MaSS, NCCFr**.

Contact: _marcely.zanon-boito at univ-avignon.fr_

