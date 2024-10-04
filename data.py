# Copyright 2024 Fuyuan Zhang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from imagenet_data import ImageNetKaggle
import torchvision

def random_select(x_test_list, y_test_list, n_ex_to_load):

    np.random.seed(50)

    index_list = []
    for index in range(0, len(x_test_list)):
        index_list.append(index)

    if n_ex_to_load > len(x_test_list):
        n_ex_to_load = len(x_test_list)

    image_num = 0
    x_test_select = []
    y_test_select = []
    while image_num < n_ex_to_load:
        index = np.random.randint(0, len(index_list))
        sel_index = index_list[index]
        x_test_select.append(x_test_list[sel_index])
        y_test_select.append(y_test_list[sel_index])
        index_list.remove(sel_index)
        image_num = image_num + 1

    x_test = np.array(x_test_select)
    y_test = np.array(y_test_select)
    return x_test, y_test

def load_cifar10_trades(n_ex_to_load):
    X_data = np.load('./data/CIFAR-10/cifar10_X.npy')
    Y_data = np.load('./data/CIFAR-10/cifar10_Y.npy')

    X_data = np.transpose(X_data, axes=[0, 3, 1, 2])

    x_test_list = np.ndarray.tolist(X_data)
    y_test_list = np.ndarray.tolist(Y_data)

    x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)
    return x_test, y_test

def load_svhn(n_ex_to_load):
    test_dataset = torchvision.datasets.SVHN(root='./data/SVHN',
                                              split='test',
                                              transform=transforms.ToTensor(),
                                              download=True)
    x_test_list = []
    y_test_list = []

    for image, label in test_dataset:
        image_array = np.array(image)
        x_test_list.append(image_array)
        y_test_list.append(label)

    x_test, y_test = random_select(x_test_list, y_test_list, n_ex_to_load)
    return x_test, y_test

def load_imagenet(n_ex, size=224):
    imagenet_size = size
    imagenet_path = "imagenet_dataset"
    val_transform = transforms.Compose(
        [
            transforms.Resize(imagenet_size),
            transforms.CenterCrop(imagenet_size),
            transforms.ToTensor()
        ]
    )

    torch.manual_seed(200)

    dataset = ImageNetKaggle(imagenet_path, "val", val_transform)
    imagenet_loader = DataLoader(
        dataset,
        batch_size=n_ex,
        num_workers=8,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )

    x_test, y_test = next(iter(imagenet_loader))

    return np.array(x_test, dtype=np.float32), np.array(y_test)

batch_size_dictionary = {'cifar10': 512,
                         'svhn': 512,
                         'imagenet': 64,
}


