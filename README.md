# LTMU
- High-Performance Long-Term Tracking with Meta-Updater(**CVPR2020 Oral**).

## abstract

Long-term visual tracking has drawn increasing attention because it is much closer to practical applications
than short-term tracking. Most top-ranked long-term trackers adopt the offline-trained Siamese architectures, thus,
they cannot benefit from great progress of short-term trackers with online update. However, it is quite risky to straightforwardly introduce online-update-based trackers to
solve the long-term problem, due to long-term uncertain
and noisy observations. In this work, we propose a novel offline-trained Meta-Updater to address an important but unsolved problem: Is the tracker ready for updating
in the current frame? The proposed meta-updater can effectively integrate geometric, discriminative, and appearance cues in a sequential manner, and then mine the sequential information with a designed cascaded LSTM module. Our meta-updater learns a binary output to guide the
tracker’s update and can be easily embedded into different trackers. This work also introduces a long-term tracking framework consisting of an online local tracker, an online verifier, a SiamRPN-based re-detector, and our meta-updater. Numerous experimental results on the VOT2018LT,
VOT2019LT, OxUvALT, TLP and LaSOT benchmarks show
that our tracker performs remarkably better than other competing algorithms.

## Paper link
- [Google Drive](https://drive.google.com/open?id=1yt9dTifaS_CjzRW6GvdQ2yvA-KC50z7G)
- [Baidu Yun](https://pan.baidu.com/s/1V0_5522WJmWJ94bU97qPRg)   提取码：fpcm
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
