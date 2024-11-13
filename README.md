## Code for paper "Detecting Adversarial Data Using Perturbation Forgery". 

## Get Started

Datasets are CIFAR10 and ImageNet100.

Our codebase accesses the datasets from `./data/` and checkpoints from `./results/checkpoints/` by default.
```
├── ...
├── data
│   
├── results
│   
├── main.py
├── ...
```

All of the adversarial data are generated using torchattacks.

Generate noise distribution by gen_dist.py or download from [here](https://pan.baidu.com/s/11gzCNm6S3eqzBmSegk0JaQ?pwd=z433). 
Put distributions under 'data/dist/'.

### Train Detector
python main.py \
--config configs/datasets/general/DIS_CIFAR10.yml \
configs/pipelines/train/DIS_train_CIFAR10.yml \
--force_merge True\
--preprocessor.name ImageNet 


### Test
python main.py \
--config configs/datasets/general/DIS_CIFAR10.yml \
configs/pipelines/train/DIS_test_CIFAR10.yml \
--force_merge True\
--preprocessor.name ImageNet 

The checkpoints are coming soon...

```
## Dependencies
python 3.8.8, PyTorch = 1.10.0, cudatoolkit = 11.7, torchvision, tqdm, scikit-learn, mmcv, numpy, opencv-python, dlib, Pillow
```
