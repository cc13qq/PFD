## Code for paper "Detecting Adversarial Data Using Perturbation Forgery". 

## Get Started

We use official CIFAR10 and ImageNet100 datasets for training. 
All of the adversarial data are generated using torchattacks.
The training and testing datasets are available [here](https://drive.google.com/drive/folders/1LNanBnj8_g34vhWl6ny8uWH48kG7HoCp?usp=sharing).
Download these datasets and unzip them in the ```/data``` folder.

Our codebase accesses the datasets from `./data/` and checkpoints from `./results/checkpoints/` by default.
```
├── ...
├── data
│   
├── checkpoints
│   
├── results
│   
├── main.py
├── ...
```

Generate noise by gen_noise.py and extract distribution by gen_dist.py. You can directly download the distribution [here](https://pan.baidu.com/s/11gzCNm6S3eqzBmSegk0JaQ?pwd=z433). 
Put distributions under 'data/dist/'.

### Checkpoints
Download these checkpoints in the ```/checkpoints``` folder.

| Dataset | Attack | Checkpoints |
| --- | --- | --- |
| CIFAR-10 | Linf | [ckpt](https://drive.google.com/file/d/1h_WA_ox5yOtwR8got0IvxiCZUP_tWDR4/view?usp=sharing) |
| ImageNet100 | Linf | [ckpt](https://drive.google.com/file/d/1z6qO4ABCM8xNYuPq5XwZujmcev04Wxun/view?usp=sharing) |
| ImageNet100 | Generative | [ckpt](https://drive.google.com/file/d/1-ar86SVwSg3D42rOju-LLadjei4tk-Ar/view?usp=sharing) |


# Train Detector
### Train detector against gradient-based adversarial attacks on CIFAR-10
python main.py \
--config configs/datasets/general/DIS_CIFAR10.yml \
configs/pipelines/train/DIS_train_CIFAR10.yml \
--force_merge True\
--preprocessor.name base

### Train detector against gradient-based adversarial attacks on ImageNet100
python main.py \
--config configs/datasets/general/DIS_ImageNet100.yml \
configs/pipelines/train/DIS_train_ImageNet100.yml \
--force_merge True\
--preprocessor.name ImageNet

# Test
### Test detector against gradient-based Linf adversarial attacks on CIFAR-10
python main.py \
--config configs/datasets/general/DIS_CIFAR10.yml \
configs/pipelines/test/DIS_test_CIFAR10.yml \
--force_merge True\
--preprocessor.name base 

### Test detector against gradient-based Linf adversarial attacks on ImageNet100
python main.py \
--config configs/datasets/general/DIS_ImageNet100.yml \
configs/pipelines/test/DIS_test_ImageNet100.yml \
--force_merge True\
--preprocessor.name ImageNet 

### Test detector against generative-based adversarial attacks on ImageNet100
python main.py \
--config configs/datasets/general/DIS_GAN_Diffusion.yml \
configs/pipelines/test/DIS_test_ImageNet100_generative.yml \
--force_merge True\
--preprocessor.name ImageNet 


```
## Dependencies
python 3.8.8, PyTorch = 1.10.0, cudatoolkit = 11.7, torchvision, tqdm, scikit-learn, mmcv, numpy, opencv-python, dlib, Pillow
```
