# KTL_project_10_Thyroid_Segmentation

This repository contains a PyTorch implementation of a Tyroid segmentation deep learning model. The steps for training and testing the model are outlined below.

## Directory Structure
용량문제로 서버에만 데이터 올려두겠습니다.
Below is the directory structure of the project:
```bash
├── data
│   ├── train
│   │   ├── images
│   │   │   ├── img1.jpg
│   │   │   ├── img2.jpg
│   │   │   ├── ...
│   │   └── masks
│   │   │   ├── mask1.jpg
│   │   │   ├── mask2.jpg
│   │   │   ├── ...
│   ├── val
│   │   ├── images
│   │   └── masks
│   └── test
│       ├── images
│       └── masks
├── checkpoints
│   ├── metric_logger.json
│   └── model_statedict.pth
├── dataset.py
├── environment.yml
├── loss.py
├── model.py
├── test.py
├── test_results
│   └── metric_df.csv
├── train.py
└── util.py
```

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Other dependencies can be installed using `environment.yml`
  
## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/ktlProject0/KTL_project_10_Thyroid_Segmentation.git
cd KTL_project_10_Thyroid_Segmentation
```
 - You can create a new Conda environment using `conda env create -f environment.yml`.

### ThyroidSegmentation train/test

- Train a model:
To train the model, run the following script. You can modify training hyperparameters (such as epochs, learning rate, batch size) in train.py.
```bash
#if you run with gpu:
  python train.py --data_direc ./data --cuda
#else you run with cpu:
  python train.py --data_direc ./data
```
- Test the model:
Once the model is trained, you can evaluate its performance on the test set by running the test.py script.
```bash
#if you run with gpu:
  python test.py --data_direc ./data/ --cuda
#else you run with cpu:
  python test.py --data_direc ./data
```

## Saving Results

During the testing process, the following will be saved automatically in the `test_results/` directory:

1. **Training and Validation Learning Curve**:  
   A plot of the training and validation loss across epochs is saved as `learning_graph_dice_coefficient.png` and `learning_graph_cross_enropy.png`. This graph helps visualize how the model performance evolves during the training process.

2. **Test Accuracy and Loss**:  
   The accuracy and loss from the test set evaluation are saved in a `metric_df.csv` file. The CSV file will include columns such as `Cross Entropy Loss` and `Dice Coefficient Score` for better tracking of the model's performance.
   
These results can be used for further analysis and tuning of the model.
