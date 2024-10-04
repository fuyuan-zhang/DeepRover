# DeepRover: A Query-Efficient Blackbox Attack for Deep Neural Networks

This is the official code for the FSE 2023 paper "DeepRover: A Query-Efficient Blackbox Attack for Deep Neural Networks" by Fuyuan Zhang et al.

## Prerequisites
* Python (3.10.12)
* Pytorch (2.0.1)
* CUDA (12.6)

## Datasets and Models

### Datasets

You should create the folders `data/CIFAR-10` and `data/SVHN` to store the CIFAR-10 and SVHN datasets, respectively.
To load the CIFAR-10 dataset, download it from the `TRADES GitHub repository` and put it in the `data/CIFAR-10` folder.
For the SVHN dataset, simply download the standard SVHN dataset and put it in the `data/SVHN folder`.
To load the ImageNet dataset, download it and put it in the `imagenet_dataset` folder.

### Models

For the CIFAR-10 and SVHN models:
Both the undefended and defended models for the CIFAR-10 and SVHN datasets are trained using the TRADES approach from the 
`TRADES GitHub repository`. We utilize their code to train the natural (undefended) models for both CIFAR-10 and SVHN, 
as well as the implementation of the TRADES defense approach to train the defended models for these datasets.

For the ImageNet models:
We utilize pretrained Inception v3 and ResNet-50 models on ImageNet.

The `TRADES GitHub repository` is available at https://github.com/yaodongyu/TRADES

## Running the Code

`deep_rover.py` is the main module that implements the DeepRover approach without refinement. `deep_rover_refinement.py` is the main module that implements DeepRover with refinement.

In `deep-rover-experimernts.sh`, I have listed all the commands (with parameters and arguments) needed to reproduce the results from our paper.

For example, to attack the ResNet-50 Imagenet network using DeepRover without refinement, you can run the following command:

```python
python deep_rover.py --model=imagenet_resnet \
                      --gpu='0,1' --n_ex=1000 --p=0.105 --eps=5.0 --n_iter=10000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=44 --radius=5 --min_s=7
```

To attack the ResNet-50 Imagenet network using DeepRover with refinement, you can run the following command:

```python
python deep_rover_refinement.py --model=imagenet_resnet \
                                --gpu='0,1' --n_ex=1000 --p=0.105 --eps=5.0 --n_iter=10000 \
                                --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=44 --radius=5 --min_s=7
```

## Experimental Results

In the `results` folder, I have included the expected experimental results from using the current implementation of 
DeepRover to attack the CIFAR-10, SVHN, and ImageNet networks. You are welcome to reproduce our results and compare them with the 
experimental results provided in the `results` folder.

## Contact
Please contact fuyuanzhang@163.com if you have any questions about the code or reproducing the experimental results from our paper.

### Citation
```
@inproceedings{ZhangHu2023,
  author       = {Zhang, Fuyuan and Hu, Xinwen and Ma, Lei and Zhao, Jianjun},
  title        = {DeepRover: A Query-Efficient Blackbox Attack for Deep Neural Networks},
  booktitle    = {ESEC/FSE},
  publisher    = {ACM}
  pages        = {1384--1394},
  year         = {2023},
}
```