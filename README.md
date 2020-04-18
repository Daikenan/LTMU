# LTMU
- High-Performance Long-Term Tracking with Meta-Updater(**CVPR2020 Oral**).

## Introduction 
Our Meta-updater can be easily embedded into other online update algorithms(Dimp, ATOM, ECO, RT-MDNet...) to make them more accurate and long-term online update. 
### Dimp
| LaSOT            | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| Dimp50       | 0.568            |             |
| Dimp50+MU       | 0.594            |             |

| VOT2019 LT            | F    | TP | TR |
|:-----------   |:----------------:|:----------------:|:----------------:|
| Dimp50       | 0.5727            |    0.6225         |0.5302|
| Dimp50+MU       | 0.6415            |     0.6871        |    0.6006|
### RT-MDNet
| LaSOT            | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|
| RT-MDNet       | 0.325            |  0.319           |
| RT-MDNet+MU       | 0.366            |  0.353           |

## Paper link
- [Google Drive](https://drive.google.com/open?id=14CGBaVl8sNIYRi0tQ5E_wsjpHiINu9Jk)
- [Baidu Yun](https://pan.baidu.com/s/1jhPOdYoNRVD30Mr5okkv2g)   提取码：kexg
## Citation
Please cite the above publication if you use the code or compare with the ASRCF tracker in your work. Bibtex entry:
```
@InProceedings{Dai_2019_CVPR,  
  author = {Dai, Kenan and Wang, Dong and Lu, Huchuan and Sun, Chong and Li, Jianhua},  
  title = {Visual Tracking via Adaptive Spatially-Regularized Correlation Filters},  	
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  	
  month = {June},  
  year = {2019}  
}  
``` 
