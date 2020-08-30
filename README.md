# LTMU
- High-Performance Long-Term Tracking with Meta-Updater(**CVPR2020 Oral && Best Paper Nomination**).

![LTMU figure](framework.jpg)

## Introduction 
Our Meta-updater can be easily embedded into other online-update algorithms(Dimp, ATOM, ECO, RT-MDNet...) to make their online-update more accurately in long-term tracking task. [More info](https://zhuanlan.zhihu.com/p/130322874).
### [DiMP](https://github.com/visionml/pytracking)
| Tracker            | LaSOT(AUC)    | VOT2020 LT(F) | VOT2018 LT(F) | TLP(AUC) |
|:-----------   |:----------------:|:----------------:|:----------------:|:----------------:|
| [**RT-MDNet**](https://github.com/IlchaeJung/RT-MDNet)| 0.335               |0.338             |0.367             |0.276             |
| **RT-MDNet+MU**| 0.354               |0.396             |0.407             |0.337             |
| [**ATOM**](https://github.com/visionml/pytracking)| 0.511               |0.497             |0.510             |0.399             |
| [**ATOM+MU**](https://github.com/Daikenan/LTMU/tree/master/ATOM_MU)    | 0.541               |0.620             |0.628             |0.473             |
| [**DiMP**](https://github.com/visionml/pytracking)| 0.568               |0.573             |0.587             |0.514             |
| [**DiMP+MU**](https://github.com/Daikenan/LTMU/tree/master/DiMP_MU)    | 0.594               |0.641             |0.649             |0.564             |
| **[PrDiMP**](https://github.com/visionml/pytracking)| 0.612               |0.632             |0.631             |0.535             |
| **PrDiMP+MU**  | 0.615               |0.661             |0.675             |0.582             |
| [**SuperDiMP**](https://github.com/visionml/pytracking)| 0.646               |0.647             |0.667             |0.552             |
| **SuperDiMP+MU| 0.658               |0.704             |0.707             |0.595             |
| [D3S](https://github.com/alanlukezic/d3s)        | -                   |-                 |-                 |-                 |
| D3S+MU     | -                   |-                 |-                 |-                 |
| [ECO](https://github.com/visionml/pytracking)        | -                   |-                 |-                 |-                 |
| ECO+MU     | -                   |-                 |-                 |-                 |
| [MDNet](https://github.com/hyeonseobnam/py-MDNet)        | -                   |-                 |-                 |-                 |
| MDNet+MU     | -                   |-                 |-                 |-                 |

## Paper link
- [Google Drive](https://drive.google.com/open?id=14CGBaVl8sNIYRi0tQ5E_wsjpHiINu9Jk)
- [Baidu Yun](https://pan.baidu.com/s/1jhPOdYoNRVD30Mr5okkv2g)   提取码：kexg
## Citation
Please cite the above publication if you use the code or compare with the ASRCF tracker in your work. Bibtex entry:
```
@inproceedings{Dai_2020_CVPR,
author = {Kenan Dai, Yunhua Zhang, Dong Wang, Jianhua Li, Huchuan Lu, Xiaoyun Yang},
title = {{High-Performance Long-Term Tracking with Meta-Updater},
booktitle = {CVPR},
year = {2020}
}
```
## Results
### LTMU
- [LaSOT](https://drive.google.com/open?id=1sfNUgUcjb29-RkjA1buv7eAziEOn5ece)
- [OxUvALT](https://drive.google.com/open?id=1dAyYSpAJhMd6mFE2uRPblCwkciuA2fUf)
- [TLP](https://drive.google.com/open?id=1Heg_Pwv021pl47ekHM40H1H2tn3KjF4I)
- [VOTLT(18&19)](https://drive.google.com/open?id=1Wh4MTEavqUs4FZtH7jGJQsdSAR0ThdeA)
