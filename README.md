# LTMU
- High-Performance Long-Term Tracking with Meta-Updater(**CVPR2020 Oral**).

![LTMU figure](framework.jpg)

## Introduction 
Our Meta-updater can be easily embedded into other online-update algorithms(Dimp, ATOM, ECO, RT-MDNet...) to make their online-update more accurately in long-term tracking task. [More info](https://zhuanlan.zhihu.com/p/130322874).
### Dimp
| LaSOT            | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| Dimp50       | 0.568            |      -       |
| Dimp50+MU(k=1)       | 0.594            |      -       |

| TLP            | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| Dimp50       | 0.514            |      0.522       |
| Dimp50+MU(k=1)       | 0.564            |      0.575       |

| VOT2019 LT            | F    | TP | TR |
|:-----------   |:----------------:|:----------------:|:----------------:|
| Dimp50       | 0.5727            |    0.6225         |0.5302|
| Dimp50+MU(k=1)       | 0.6415            |     0.6871        |    0.6006|

| VOT2018 LT            | F    | TP | TR |
|:-----------   |:----------------:|:----------------:|:----------------:|
| Dimp50       | 0.5869            |    0.6356         |0.5452|
| Dimp50+MU(k=1)       | 0.6493            |     0.6978        |    0.6072|

### RT-MDNet
| LaSOT            | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| RT-MDNet       | 0.325            |  0.319           |
| RT-MDNet+MU(k=2)       | 0.366            |  0.353           |
### ATOM
Coming soon
### ECO
Coming soon
### MDNet
Coming soon
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
[LaSOT](https://drive.google.com/open?id=1sfNUgUcjb29-RkjA1buv7eAziEOn5ece)
[OxUvALT](https://drive.google.com/open?id=1dAyYSpAJhMd6mFE2uRPblCwkciuA2fUf)
[TLP](https://drive.google.com/open?id=1Heg_Pwv021pl47ekHM40H1H2tn3KjF4I)
[VOTLT(18&19)](https://drive.google.com/open?id=1Wh4MTEavqUs4FZtH7jGJQsdSAR0ThdeA)
