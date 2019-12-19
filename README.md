# RealMix
*Towards Realistic Semi-Supervised Deep Learning Algorithms*

[![License](http://img.shields.io/badge/license-APACHE2-blue.svg)](LICENSE.txt)

Code for RealMix. The technique provides state-of-the-art results for semi-supervised learning benchmarks and is able to surpass baseline performance when the unlabeled and labeled distributions mismatch. This can also be run with your own dataset.

## Paper

Paper available at: "[RealMix: Towards Realistic Semi-Supervised Deep Learning Algorithms](https://arxiv.org/abs/________)" by Varun Nair, Javier Fuentes Alonso, and Tony Beltramelli.

To cite our work, please use the citation provided below:

```
@article{nair2019realmix,
  title={RealMix: Towards Realistic Semi-Supervised Deep Learning Algorithms},
  author={Nair, Varun and Fuentes Alonso, Javier and Beltramelli, Tony},
  journal={arXiv preprint arXiv:____________},
  year={2019}
}
```

Read more about research conducted at Uizard Technologies at [https://uizard.io/research](https://uizard.io/research).

## Acknowledgement

Code adapted in generous amounts from:
- "[MixMatch - A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)" by David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel.
- "[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)" by Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le.

## 1. Setup

### 1.1. Install dependencies and create environment variables

```bash
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3 --system-site-packages env3
. env3/bin/activate
pip install -r requirements.txt

# Setup environment variables
export ML_DATA="path to where you want the datasets saved"
export PYTHONPATH=$PYTHONPATH:.
```

`ML_DATA` is a shell environment variable that should point to the location where the datasets are installed. The second export command should be run from the repository folder.

### 1.2.0 Install benchmark datasets and create splits

Create training splits for CIFAR-10 and SVHN:

```
# Download datasets
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py

# Create semi-supervised subsets
for seed in 1 2 3 4 5; do
    for size in 250 500 1000 2000 4000; do
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --label_split_size=$size $ML_DATA/SSL/svhn $ML_DATA/svhn-train.tfrecord &
        CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --label_split_size=$size $ML_DATA/SSL/cifar10 $ML_DATA/cifar10-train.tfrecord &
    done
    wait
done

```

### 1.2.1 Install custom datasets and create training splits

To install your own dataset, it needs to be structured and renamed appropriately. First, have the training, unlabeled, and test data stored in separate directories, with each directory containing the respective images stored by class. 
Example file structure:

- mydataset/
    - train/
        - dog/
        - cat/
        - fox/
    - unlabeled/
    - test/
        - dog/
        - cat/
        - fox/

Next, once the above file structure is created, use ```./scripts/rename_files.py``` to rename your files. **Note: This will rename your files in-place, keeping a backup of your dataset is recommended.**
Example usage:

```
python scripts/rename\_files.py --targetdir $ML_DATA/mydataset/ --index 0 --filetype png
```

The above script will rename every png to the format {number}\_{class_name}.png . You can specify a different index for the numbering to start at and also specify the filetype as jpg. The index has no significance for training, used only for your reference to identify which indices are used in the training, unlabeled, and test set.

Finally, create the TFRecord files and training splits. Note, this assumes your dataset directory (Ex: mydataset/) is located in the directory that was assigned to $ML_DATA in step 1. Example usage:

```
CUDA_VISIBLE_DEVICES= ./scripts/create_datasets.py custom --name mydataset --traindir $ML_DATA/mydataset/train/ --unlabeldir $ML_DATA/mydataset/unlabeled/ --testdir $ML_DATA/mydataset/test/

for seed in 1 2 3 4 5; do
    CUDA_VISIBLE_DEVICES= scripts/create_split.py --seed=$seed --label_split_size=450 --label_size=500 $ML_DATA/SSL/mydataset $ML_DATA/mydataset-train.tfrecord
done
```

The ```--label_size``` flag ```scripts/create_split.py``` specifies the number of images in your training directory. A training split that is smaller that the total amount of training images available is done by setting the ```--labeled_split_size``` flag to the desired number of images, otherwise ```--label_size``` should be equal to ```--labeled_split_size```.

In practice for custom datasets, we set the ```--label_split_size``` flag to ```--label_size``` minus the number of desired validation images.


### 1.3. Augment Data

RealMix uses several data augmentations on the labeled and unlabeled sets created by ```scripts/make_aug_copy.py```. This can generate a tfrecord file that is quite large, **make sure you have at least 5-10 GB of free space available**. An example is shown below to generate augmentations for just one particular seed and split size: 

```
python scripts/make_aug_copy.py --aug_dataset=cifar10.1@250 --unlabel_size=49750 --aug_copy=50 --augment=cutout
```

This will generate a new dataset with the name of cifar10_aug50.1@250, and is the first step described in the algorithm of the paper.

Available augmentations are cifar10, svhn, stl-10, color, and cutout, with best results are achieved with using cutout. More augmentations (```--aug_copy```) are generally better, but increase the amount of storage necessary. 


## 2. Running

### 2.1. Training using RealMix 

RealMix utilizies leading ideas in semi-supervised learning to achieve state-of-the-art results. These include entropy minimization, data-augmentation, MixUp, exponential moving average parameters, training signal annealing (TSA), and out-of-distribution masking. For the intuition and theory behind each of the concepts, please read the provided paper. Tips for hyperparameters that correspond to each are as follows:

- Entropy Minimization/Distribution Sharpening: This hyperparameter has a default value that works well across many datasets and doesn't require tuning.

- Data Augmentation: Different than the augmentation performed in section 1.3, this augmentation is used in the consistency training loss. **Note: If cutout was used in section 1.3, use another type of augmentation during training to avoid overly strong augmentations.**

- MixUp: "--beta=0.75". This hyperparameter can be tuned for each dataset, and specifies the value used in the beta distribution that generates MixUp probabilties.

- Exponential Moving Average Parameters: This hyperparameter has a default value that works well across many datasets and doesn't require tuning.

- TSA: See Xie et. al, 2019 for more information on how this works. Available values for ```--tsa``` are none, linear_schedule, exp_schedule, and log_schedule.

Example of training on cifar10, shuffled with `seed=1`, 250 labeled samples, 500 validation samples, cifar10 augmentation, tsa with linear_schedule - Gives SOTA.

```bash
CUDA_VISIBLE_DEVICES=0 python realmix.py --filters=32 --dataset=cifar10_aug50.3@450-50 --w_match=75 --beta=0.75 --custom_dataset=True --augment=cifar10 --tsa=linear_schedule
```

Available labeled sizes and seeds are the ones that were created using the ```scripts/create_split.py```, re-run with different ```--size``` and ```--seed``` arguments to generate different labeled sizes and randomized splits. If you are training with the same dataset several times, change the default flag values in realmix.py libml/train.py - this will shorten the command needed to run training.

Training will currently print accuracy (train, validation, test), precision, recall, and f1-score to tensorboard, see section 3 for more. 

To print the accuracy per class to tensorboard, add the ```--class_mapping``` flag to the training command and specify the json containing a class mapping for your dataset. This file was created while running the scripts/create_datasets.py and should be in your repo directory.

### 2.2. Training baselines

You can compare the performance of this semi-supervised learning approach by running a baseline on just the training data. Example usage:

```
python fully_supervised/fs_baseline.py --train_dir experiments/fs --dataset=mydataset-50 --wd=0.02 --smoothing=0.001 --nclass=20 --img_size=32 --custom_dataset --train_record mydataset-train.tfrecord --test_record mydataset-test.tfrecord
```

This would run a Wide-ResNet-28 model on using a validation size of 50, 20 classes, and images of 32x32. The wd and smoothing parameters are recommended to be used as above, and the --custom_dataset flag is necessary to use any dataset other than CIFAR10 and SVHN.

To run a baseline model that also uses MixUp: 

```
python fully_supervised/fs_mixup.py --train_dir experiments/fs --dataset=mydataset-50 --wd=0.02 --nclass=20 --img_size=32 --custom_dataset --train_record mydataset-train.tfrecord --test_record mydataset-test.tfrecord
```

**Note**: Do not name your dataset with a "-" inside of it, the above baselines will not be able to run if your dataset is named with a "-". The "-" is used to denote the number of validation images.

### 2.3 Training Out-of-Distribution Experiments

To train out-of-distribution experiments, first create the CIFAR10 datasets as described above. Then, we filter the datasets to simulate varying levels of mismatch using ```filter_dataset.py``` as described below. 

```
python filter_dataset.py --class_filter=2,3,4,5,6,7 --filter_dataset=$ML_DATA/SSL/cifar10.1@250-label.tfrecord --save_name=$ML_DATA/SSL/cifar6.1@250-label.tfrecord

python filter_dataset.py --class_filter=2,3,4,5,6,7 --filter_dataset=$ML_DATA/cifar10-test.tfrecord --save_name=$ML_DATA/cifar6-test.tfrecord

# Unlabeled Data Filtering for 0% Mismatch
python filter_dataset.py --class_filter=4,5,6,7 --filter_dataset=$ML_DATA/SSL/cifar10.1@250-unlabel.tfrecord --save_name=$ML_DATA/SSL/cifar6_mismatch0.1@250-unlabel.tfrecord

# Unlabeled Data Filtering for 25% Mismatch
python filter_dataset.py --class_filter=0,5,6,7 --filter_dataset=$ML_DATA/SSL/cifar10.1@250-unlabel.tfrecord --save_name=$ML_DATA/SSL/cifar6_mismatch25.1@250-unlabel.tfrecord

# Unlabeled Data Filtering for 50% Mismatch
python filter_dataset.py --class_filter=0,1,6,7 --filter_dataset=$ML_DATA/SSL/cifar10.1@250-unlabel.tfrecord --save_name=$ML_DATA/SSL/cifar6_mismatch50.1@250-unlabel.tfrecord

# Unlabeled Data Filtering for 75% Mismatch
python filter_dataset.py --class_filter=0,1,8,7 --filter_dataset=$ML_DATA/SSL/cifar10.1@250-unlabel.tfrecord --save_name=$ML_DATA/SSL/cifar6_mismatch75.1@250-unlabel.tfrecord

# Unlabeled Data Filtering for 100% Mismatch
python filter_dataset.py --class_filter=0,1,8,9 --filter_dataset=$ML_DATA/SSL/cifar10.1@250-unlabel.tfrecord --save_name=$ML_DATA/SSL/cifar6_mismatch100.1@250-unlabel.tfrecord

```

Then, use the augmentation script from above to create several augmentations on the unlabeled datasets. 
Also, be sure to copy the test file to the same name as the unlabeled set for the algorithm to detect the test file.

## 3. Evaluation


### 3.1. Monitoring training progress

You can point tensorboard to the training folder (by default it is `--train_dir=./experiments`) to monitor the training
process:

```bash
tensorboard.sh --port 6007 --logdir experiments/
```

Tensorboard will log accuracy (train, val, test), precision, recall, and f1-score, as well as the per-class accuracy if ```--class_mapping``` is specified (see section 2.1). 

### 3.2. Evaluate model

Example usage:

```bash
CUDA_VISIBLE_DEVICES=0 python realmix.py --filters=32 --dataset=mydataset_aug50.3@450-50 --w_match=75 --beta=0.75 --nclass=20 --custom_dataset=True --augment=color --perform_inference --inference_dir=$ML_DATA/mydataset/test --inference_ckpt=./experiments/mydataset_aug50.3@450-50/<model_name>/tf/model.ckpt-07798784 
```

Evaluation is specified by the ```--perform_inference, --inference_dir, --inference_ckpt``` flags. ```--inference_dir``` is the directory that contains the images to be evaluated on and ```--inference_ckpt``` is the specific checkpoint used. Be sure to replace <model_name> with the name of your full model directory, in this example found in ./experiments/mydataset.3@450-50/. 

The above script will print the overall accuracy and generate mydataset.3@450-50_pred.npy, which contains the predictions on the inference set. Use this npy file to compute further statistics.

### 3.3. Checkpoint accuracy

To compute the median accuracy of the last 20 checkpoints, run:

```bash
# Following the previous example in which we trained cifar10.3@250-5000, extracting accuracy:
./scripts/extract_accuracy.py experiments/mydataset_aug50.3@450-50/<model_name>/
# The command above will create a stats/accuracy.json file in the model folder.
# The format is JSON so you can either see its content as a text file or process it to your liking.
```

## References

[1] David Berthelot, Nicholas Carlini, Ian Goodfellow, Nicolas Papernot, Avital Oliver and Colin Raffel, "MixMatch - A Holistic Approach to Semi-Supervised Learning", in arXiv:1905.02249, 2019.

[2] Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, and Quoc V. Le, "Unsupervised Data Augmentation for Consistency Training", in arXiv:1904.12848v2, 2019.
