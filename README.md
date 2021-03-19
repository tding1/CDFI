# CDFI (Compression-Driven-Frame-Interpolation)

[Paper]() (Coming soon...)

[Tianyu Ding*](https://www.tianyuding.com), [Luming Liang*](https://scholar.google.com/citations?user=vTgdAS4AAAAJ&hl=en), [Zhihui Zhu](http://mysite.du.edu/~zzhu61/index.html), Ilya Zharkov

IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**), 2021

## Introduction

We propose a **C**ompression-**D**riven network design for **F**rame **I**nterpolation (**CDFI**), that leverages model compression to significantly reduce the model size while achieving superior performance. Concretely, we first compress  [AdaCoF](https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_AdaCoF_Adaptive_Collaboration_of_Flows_for_Video_Frame_Interpolation_CVPR_2020_paper.html) and show that a 10X compressed AdaCoF performs similarly as its original counterpart; then we further improve upon this compressed model with simple modifications. 

- We achieve a significant performance gain with only a quarter in size compared with the original AdaCoF

  |                    |            Vimeo-90K            |           Middlebury            |           UCF101-DVF            | Size  |
  | :----------------: | :-----------------------------: | :-----------------------------: | :-----------------------------: | :---: |
  |                    |        PSNR, SSIM, LPIPS        |        PSNR, SSIM, LPIPS        |        PSNR, SSIM, LPIPS        |       |
  |       AdaCoF       |       34.38, 0.974, 0.019       |       35.74, 0.979, 0.019       |     35.20, **0.967**, 0.019     | 21.8M |
  | Compressed AdaCoF  |       34.15, 0.973, 0.020       |       35.46, 0.978, 0.019       |     35.14, **0.967**, 0.019     | 2.45M |
  |      AdaCoF+       |       34.58, 0.975, 0.018       |       36.12, 0.981, 0.017       |     35.19, **0.967**, 0.019     | 22.9M |
  | Compressed AdaCoF+ |       34.46, 0.975, 0.019       |       35.76, 0.979, 0.019       |     35.16, **0.967**, 0.019     | 2.56M |
  |  Our Final Model   | **35.19**, **0.978**, **0.010** | **37.17**, **0.983**, **0.008** | **35.24**, **0.967**, **0.015** | 4.98M |

- Our final model also performs favorably against other state-of-the-arts (details refer to our paper)

- The proposed framework is generic and can be easily transferred to other DNN-based frame interpolation method

<p align="center">
  <img src="figs/cdfi_fps_160.gif" />
</p>

The above GIF is a demo of using our method to genenrate slow motion video, which increases the FPS from 5 to 160. We also provide a long video demonstration [here](https://www.youtube.com/watch?v=KEUcw4xoB5E) (redirect to YouTube).

## Environment

- CUDA 11.0
- python 3.8.3

- torch 1.6.0
- torchvision 0.7.0
- cupy 7.7.0
- scipy 1.5.2
- numpy 1.19.1
- Pillow 7.2.0
- scikit-image 0.17.2

## Test Pre-trained Models

Download repository:

~~~bash
$ git clone https://github.com/tding1/CDFI.git
~~~

### Testing data

For user convenience, we already provide the [Middlebury](https://vision.middlebury.edu/flow/data/) and [UCF101-DVF](https://github.com/liuziwei7/voxel-flow) test datasets in our repository, which can be found under directory `test_data/`.

### Evaluation metrics

We use the built-in functions in `skimage.metrics` to compute the PSNR and SSIM, for which the higher the better. We also use [LPIPS](https://arxiv.org/abs/1801.03924), a newly proposed metric that measures perceptual similarity, for which the smaller the better. For user convenience, we include the implementation of LPIPS in our repo under `lpips_pytorch`, which is a slightly modifed version of [here](https://github.com/S-aiueo32/lpips-pytorch) (with an updated squeezenet backbone).

### Test our pre-trained CDFI model

~~~bash
$ python test.py --gpu_id 0
~~~

By default, it will load our pre-trained model  `checkpoints/CDFI_adacof.pth`. It will print the quatitative results on both Middlebury and UCF101-DVF, and the interpolated images will be saved under `test_output/cdfi_adacof/`.

### Test the compressed AdaCoF

~~~bash
$ python test_compressed_adacof.py --gpu_id 0 --kernel_size 5 --dilation 1
~~~

By default, it will load our pre-trained model  `checkpoints/compressed_adacof_F_5_D_1.pth`. It will print the quatitative results on both Middlebury and UCF101-DVF, and the interpolated images will be saved under `test_output/compressed_adacof_F_5_D_1/`.

### Test the compressed AdaCoF+

~~~bash
$ python test_compressed_adacof.py --gpu_id 0 --kernel_size 11 --dilation 2
~~~

By default, it will load our pre-trained model  `checkpoints/compressed_adacof_F_11_D_2.pth`. It will print the quatitative results on both Middlebury and UCF101-DVF, and the interpolated images will be saved under `test_output/compressed_adacof_F_11_D_2/`.

## Training Our Model

### Training data

We use the [Vimeo-90K](https://arxiv.org/abs/1711.09078) triplet dataset for video frame interpolation task, which is relatively large (more than 32 GB).

~~~bash
$ wget http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
$ unzip vimeo_triplet.zip
$ rm vimeo_triplet.zip
~~~

### Start training

~~~bash
$ python train.py --gpu_id 0 --data_dir path/to/your/downloaded/vimeo_triplet/
~~~

It will generate an unique ID for each training, and all the intermediate results/records will be saved under `model_weights/<training id>/`. There are many other training options, e.g., `--lr`, `--epochs`, `--loss` and so on, can be found in `train.py`.

## Applying CDFI to New Models



## Citation

Coming soon...

## Acknowledgements

The code is larged based on  [HyeongminLEE/AdaCoF-pytorch](https://github.com/HyeongminLEE/AdaCoF-pytorch) and [baowenbo/DAIN](https://github.com/baowenbo/DAIN).













