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

import numpy as np

def percentage_block(p_init, iteration):

    decrease = 0.5
    if 30 < iteration <= 100:
        p = p_init * (decrease ** 1)
    elif 100 < iteration <= 300:
        p = p_init * (decrease ** 2)
    elif 300 < iteration <= 500:
        p = p_init * (decrease ** 3)
    elif 500 < iteration <= 1000:
        p = p_init * (decrease ** 4)
    elif 1000 < iteration <= 2000:
        p = p_init * (decrease ** 5)
    elif 2000 < iteration <= 4000:
        p = p_init * (decrease ** 6)
    elif 4000 < iteration <= 6000:
        p = p_init * (decrease ** 7)
    elif 6000 < iteration <= 8000:
        p = p_init * (decrease ** 8)
    elif 8000 < iteration <= 10000:
        p = p_init * (decrease ** 9)
    else:
        p = p_init

    return p

def mutation_vector_l2(x, y, r):
    mutation = np.zeros([x, y])

    x_c = np.random.randint(0, x)
    y_c = np.random.randint(0, y)

    radius = r
    for counter_x in range(max(0, x_c - radius), min(x, x_c + radius + 1)):
        for counter_y in range(max(0, y_c - radius), min(y, y_c + radius + 1)):
            distance_power = (counter_x - x_c)**2 + (counter_y - y_c)**2
            if distance_power == 0.0:
                mutation[counter_x, counter_y] = 1.5
            else:
                mutation[counter_x, counter_y] = 0.8/distance_power

    mutation /= np.sqrt(np.sum(mutation ** 2, keepdims=True))

    return mutation

def mutation_vectors_block_l2(s,num, r):
    mutation = np.zeros([s, s])
    num_mutation_vectors = num

    while np.sqrt(np.sum(mutation ** 2, keepdims=True)) == 0.0:
        i = 0
        while i < num_mutation_vectors:
            mutation = mutation + mutation_vector_l2(s,s,r)
            i=i+1

    mutation /= np.sqrt(np.sum(mutation ** 2, keepdims=True))

    return mutation