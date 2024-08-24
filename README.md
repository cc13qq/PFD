## Code for paper "Detecting Adversarial Data via Perturbation Forgery". 

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


```
## Dependencies
python 3.8.8, PyTorch = 1.10.0, cudatoolkit = 11.7, torchvision, tqdm, scikit-learn, mmcv, numpy, opencv-python, dlib, Pillow
```
