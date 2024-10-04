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

import argparse
import time
import numpy as np
import data
import models
import os
import utils
from datetime import datetime
from functions import percentage_block
from functions import mutation_vectors_block_l2
from deep_rover import deep_rover

np.set_printoptions(precision=5, suppress=True)

def deep_rover_single(model, x, y, correctly_classified, eps, n_iters, p_init, seed_num, local_density, init_s, seed_vect_num, radius, min_s, log):
    """ to attack only one image using deep-rover """
    np.random.seed(80)

    min_val, max_val = 0, 1
    c, h, w = x.shape[1:]
    x, y = x[correctly_classified], y[correctly_classified]
    x, y = x[:1], y[:1]
    num_total_images = x.shape[0]

    # the initilization step

    initial_seeds = seed_num
    initial_square = init_s
    initial_blocks = np.zeros(x.shape)
    s = initial_square
    i = 0
    while i < initial_seeds:
        h_c = np.random.randint(0, h - s)
        w_c = np.random.randint(0, w - s)
        initial_blocks[:, :, h_c:h_c + s, w_c:w_c + s] += mutation_vectors_block_l2(s,
                                                                                                            seed_vect_num,
                                                                                                            radius).reshape(
            [1, 1, s, s]) * np.random.choice([-1, 1], size=[x.shape[0], c, 1, 1])
        i = i + 1

    x_mutated = np.clip(x + initial_blocks / np.sqrt(np.sum(initial_blocks ** 2, axis=(1, 2, 3), keepdims=True)) * eps, 0, 1)

    logits = model.predict(x_mutated)
    objective = model.objective(y, logits)
    n_queries = np.ones(x.shape[0])
    for index in range(n_iters):
        to_attack = (objective > 0.0)

        if to_attack == [True]:

            x_index, x_mutated_index = x[to_attack], x_mutated[to_attack]
            y_index, objective_index = y[to_attack], objective[to_attack]
            objective_index = objective[to_attack]
            mutation_index = x_mutated_index - x_index

            p = percentage_block(p_init, index)
            min_size = min_s
            s = max(int(np.sqrt(p * h * w)), min_size)

            # one block for mutation insertion
            h_c = np.random.randint(0, h - s)
            w_c = np.random.randint(0, w - s)
            mutation_insertion_block = np.zeros(x_index.shape)
            mutation_insertion_block[:, :, h_c:h_c + s, w_c:w_c + s] = 1.0

            # multiple blocks for mutation deletion

            if p >= 0.005:
                s2 = int(s // 5)
                if s2 != 0:
                    num_of_s2 = int(s // s2) ** 2
                else:
                    s2 = s
                    num_of_s2 = 1
            else:
                s2 = s
                num_of_s2 = 1

            i = 0
            mutation_deletion_blocks = np.zeros(x_index.shape)
            while i < num_of_s2:
                h_c_2 = np.random.randint(0, h - s2)
                w_c_2 = np.random.randint(0, w - s2)

                mutation_deletion_blocks[:, :, h_c_2:h_c_2 + s2, w_c_2:w_c_2 + s2] = 1.0
                mutation_index[:, :, h_c_2:h_c_2 + s2, w_c_2:w_c_2 + s2] = 0.0
                i = i + 1

            # to compute perturbation distance of various blocks and the mutated image
            distance_insertion_block = np.sqrt(
                np.sum(((x_mutated_index - x_index) * mutation_insertion_block) ** 2, axis=(2, 3), keepdims=True))
            distance_mutated_image = np.sqrt(np.sum((x_mutated_index - x_index) ** 2, axis=(1, 2, 3), keepdims=True))
            all_blocks = np.maximum(mutation_insertion_block, mutation_deletion_blocks)
            distance_all_blocks = np.sqrt(np.sum((mutation_index * all_blocks) ** 2, axis=(2, 3), keepdims=True))

            # to perform mutation insertion and mutation deletion to mutated images
            num_of_noise = local_density * (seed_num * p // 1 + 1)

            new_mutations = np.ones([x_index.shape[0], c, s, s])
            new_mutations = new_mutations * mutation_vectors_block_l2(s, num_of_noise, radius).reshape([1, 1, s, s])
            new_mutations *= np.random.choice([-1, 1], size=[x_index.shape[0], c, 1, 1])
            old_mutations = mutation_index[:, :, h_c:h_c + s, w_c:w_c + s] / (1e-10 + distance_insertion_block)
            new_mutations = old_mutations + new_mutations
            new_mutations = new_mutations / np.sqrt(np.sum(new_mutations ** 2, axis=(2, 3), keepdims=True)) * (
                    np.maximum(eps ** 2 - distance_mutated_image ** 2, 0) / c + distance_all_blocks ** 2) ** 0.5
            mutation_index[:, :, h_c:h_c + s, w_c:w_c + s] = new_mutations + 0

            x_new = x_index + mutation_index / np.sqrt(np.sum(mutation_index ** 2, axis=(1, 2, 3), keepdims=True)) * eps
            x_new = np.clip(x_new, min_val, max_val)
            distance_mutated_image = np.sqrt(np.sum((x_new - x_index) ** 2, axis=(1, 2, 3), keepdims=True))
            logits = model.predict(x_new)
            objective_new = model.objective(y_index, logits)

            to_mutate = objective_new < objective_index
            objective[to_attack] = to_mutate * objective_new + ~to_mutate * objective_index

            to_mutate = np.reshape(to_mutate, [-1, *[1] * len(x.shape[:-1])])
            x_mutated[to_attack] = to_mutate * x_new + ~to_mutate * x_mutated_index
            n_queries[to_attack] += 1

        accuracy = (objective > 0.0).sum() / num_total_images

        if accuracy == 0:
            break

    distance_mutated_image = np.sqrt(np.sum((x_mutated - x) ** 2, axis=(1, 2, 3), keepdims=True))
    log.print('L2 Distance: {:.2f}'.format(np.mean(distance_mutated_image)))
    success = (objective < 0.0)

    return n_queries, x_mutated, success

def bisect_search(model, orig_image, adversary, true_class, tolerance=0.0001):
    """true_class is in one-hot encoding"""
    up_image = adversary
    low_image = orig_image
    counter = 0
    distance = np.sqrt(np.sum(np.power(up_image - low_image, 2), keepdims=False))

    while (distance > tolerance):
        mid_image = (up_image + low_image) / 2
        predict = np.argmax(model.predict(mid_image))
        counter = counter + 1
        if true_class[0][predict]:
            low_image = mid_image
        else:
            up_image = mid_image
        distance = np.sqrt(np.sum(np.power(up_image - low_image, 2), keepdims=False))

    return up_image, counter


def deep_rover_refine(model, x, y, true_class, success, adv, query_num, correctly_classified, eps, n_iters, p_init,
                      seed_num, local_density, init_s, seed_vect_num, radius, min_s, log):
    """x is supposed to be correctly classified images, hence y and true_class should be identical"""
    record = []
    existing_query = query_num

    if success:
        distance = eps
        while True:
            image, queries = bisect_search(model,x,adv,true_class)
            existing_query = existing_query + queries
            new_distance = np.sqrt(np.sum(np.power(image - x, 2), keepdims=False))
            record.append((existing_query,new_distance))
            if distance-new_distance > 0.01 and existing_query<n_iters:
                distance = new_distance
                queries,adv,success_attack = deep_rover_single(model, x, y, correctly_classified, distance, n_iters, p_init,
                                                                     seed_num, local_density, init_s, seed_vect_num, radius, min_s,log)
                existing_query = existing_query + queries
                if success_attack == False:
                    return True, image, record, new_distance
            else:
                log.print('refinement stops')
                return True, image, record, new_distance
    else:
        log.print('Image {} is not attacked successfully. Skip refining step'.format(j))
        return False, x, [(record,0)], eps

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters.')
    parser.add_argument('--model', type=str, default='cifar10-undefended', choices=models.models_defined,
                        help='name of models')
    parser.add_argument('--exp_results', type=str, default='results', help='the folder to save experimental results.')
    parser.add_argument('--gpu', type=str, default='0', help='specify the ids of gpus')
    parser.add_argument('--n_ex', type=int, default=5, help='the number of correctly classified input images to attack.')
    parser.add_argument('--p', type=float, default=0.1,help='percentage of pixels in an initial block.')
    parser.add_argument('--eps', type=float, default=0.15, help='the L2 distance.')
    parser.add_argument('--n_iter', type=int, default=10000, help='total number of queries.')
    parser.add_argument('--seed_num', type=int, default=200, help='Number of initial noise.')
    parser.add_argument('--local_density', type=float, default=3.0,
                        help='this is used to increase num of local noise')
    parser.add_argument('--seed_vect_num', type=int, default=1,
                        help='Number of vectors per noise in initialization.')
    parser.add_argument('--init_s', type=int, default=1, help='square size of initial noise')
    parser.add_argument('--radius', type=int, default=5, help='radius of each vector')
    parser.add_argument('--min_s', type=int, default=3, help='minimum square size of blocks')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    dataset = 'cifar10' if 'cifar10' in args.model else 'svhn' if 'svhn' in args.model else 'imagenet'
    timestamp = str(datetime.now())[:10] + '_' + str(datetime.now())[11:13] + '-' + str(datetime.now())[
                                                                                    14:16] + '-' + str(
        datetime.now())[
                                                                                                   17:19]
    basic_info = '{} model={} dataset={} n_ex={} eps={} p={} n_iter={}'.format(
        timestamp, args.model, dataset, args.n_ex, args.eps, args.p, args.n_iter)
    batch_size = data.batch_size_dictionary[dataset]

    n_cls = 1000 if dataset == 'imagenet' else 10
    gpu_memory = 0.99

    log_path = '{}/{}.log'.format(args.exp_results, basic_info)
    log = utils.Logger(log_path)
    log.print('Basic Parameters: {}'.format(basic_info))

    n_ex_to_load = 2 * args.n_ex

    if args.model in ['cifar10-defended-trades', 'cifar10-undefended-trades']:
        x_test, y_test = data.load_cifar10_trades(n_ex_to_load)
    else:
        if dataset == 'svhn':
            x_test, y_test = data.load_svhn(n_ex_to_load)
        else:
            if dataset == 'imagenet':
                if args.model != 'imagenet_inception':
                    x_test, y_test = data.load_imagenet(n_ex_to_load)
                else:
                    x_test, y_test = data.load_imagenet(n_ex_to_load, size=299)

    print('model name:', args.model)

    if args.model in ['cifar10-defended-trades', 'cifar10-undefended-trades',
                      'svhn-defended-trades', 'svhn-undefended-trades']:
        model = models.CustomModel_Trades(args.model, batch_size, gpu_memory)
    else:
        model = models.CustomModel(args.model, batch_size, gpu_memory)

    logits_clean = model.predict(x_test)
    correctly_classified = logits_clean.argmax(1) == y_test

    log.print('Clean accuracy: {:.2%}'.format(np.mean(correctly_classified)))
    y_target_onehot = utils.dense_to_onehot(y_test, n_cls=n_cls)

    num_total_input = x_test.shape[0]
    print('sampled', num_total_input, 'input images')
    x, y = x_test[correctly_classified], y_target_onehot[correctly_classified]
    x, y = x[:args.n_ex], y[:args.n_ex]
    num_total_images = x.shape[0]
    print('select', num_total_images, 'correctly classified images')

    n_queries, x_adv, _, success = deep_rover(model, x, y, args.eps, args.n_iter,
                                            args.p,
                                            args.seed_num, args.local_density, args.init_s, args.seed_vect_num,
                                            args.radius,args.min_s, log)


    """the code above run deep_rover and derived results after attack
       the code below perform refinement process. we refine adversarial examples one by one.
    """

    """n_queries refers to query numbers spent on attack x,
       x refers to correctly classified images, 
       y refers to corresponding labels in one-hot encoding,
       x_adv refers to corresponding adversarial examples (if attack is successful),
       success indicates whether the attack is successful
    """

    """regarding refinement, we need to know the new distance after refinement and queries spent for refinement"""

    success_refine_list = []
    refine_images = []
    record_list = []
    new_distance_list = []

    j = 0
    while j < x.shape[0]:

        if success[j:j + 1] == [True]:
            log.print(' ')
            log.print('Start refining image {}'.format(j))
            success_refine, image_refined, record, new_distance = deep_rover_refine(model, x[j:j+1],
                                                                                       y[j:j+1], y[j:j+1], success[j:j+1],
                                                                                       x_adv[j:j+1],
                                                                                       n_queries[j:j+1], [True],
                                                                                       args.eps, args.n_iter,args.p,
                                                                                       args.seed_num, args.local_density, args.init_s, args.seed_vect_num,
                                                                                       args.radius, args.min_s, log)

            success_refine_list.append(success_refine)
            refine_images.append(image_refined)
            record_list.append(record)
            new_distance_list.append(new_distance)

        j = j + 1

    log.print(' ')
    log.print('Average L2 Distance after Refinement: {:.2f}'.format(np.array(new_distance_list).mean()))

