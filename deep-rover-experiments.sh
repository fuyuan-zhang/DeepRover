############################## for cifar 10 neural networks ###############################################
### parameters for cifar10 undefended network without refinement
python deep_rover.py --model=cifar10-undefended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=400 --local_density=1.0 --seed_vect_num=1 --init_s=6 --radius=5 --min_s=3

### parameters for cifar10 undefended network with refinement
python deep_rover_refinement.py --model=cifar10-undefended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=400 --local_density=1.0 --seed_vect_num=1 --init_s=6 --radius=5 --min_s=3

### parameters for cifar10 defended network without refinement
python deep_rover.py --model=cifar10-defended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=400 --local_density=2.0 --seed_vect_num=3 --init_s=12 --radius=5 --min_s=3


### parameters for cifar10 defended network with refinement
python deep_rover_refinement.py --model=cifar10-defended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=400 --local_density=2.0 --seed_vect_num=3 --init_s=12 --radius=5 --min_s=3


############################################################################################################


############################## for svhn neural networks ####################################################
### parameters for SVHN undefended network without refinement
python deep_rover.py --model=svhn-undefended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=12 --radius=3 --min_s=3


### parameters for SVHN undefended network with refinement
python deep_rover_refinement.py --model=svhn-undefended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=12 --radius=3 --min_s=3


### parameters for SVHN defended network without refinement
python deep_rover.py --model=svhn-defended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=12 --radius=3 --min_s=3


### parameters for SVHN defended network with refinement
python deep_rover_refinement.py --model=svhn-defended-trades \
                      --gpu='0,1' --n_ex=1000 --p=0.11 --eps=5.0 --n_iter=20000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=12 --radius=3 --min_s=3


############################################################################################################


############################## for ImageNet neural networks ################################################
### parameters for ImageNet Inception v3 network without refinement
python deep_rover.py --model=imagenet_inception \
                      --gpu='0,1' --n_ex=1000 --p=0.103 --eps=5.0 --n_iter=10000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=2 --init_s=60 --radius=5 --min_s=7


### parameters for ImageNet Inception v3 network with refinement
python deep_rover_refinement.py --model=imagenet_inception \
                      --gpu='0,1' --n_ex=1000 --p=0.103 --eps=5.0 --n_iter=10000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=2 --init_s=60 --radius=5 --min_s=7


### parameters for ImageNet ResNet-50 network without refinement
python deep_rover.py --model=imagenet_resnet \
                      --gpu='0,1' --n_ex=1000 --p=0.105 --eps=5.0 --n_iter=10000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=44 --radius=5 --min_s=7


### parameters for ImageNet ResNet-50 network with refinement
python deep_rover_refinement.py --model=imagenet_resnet \
                      --gpu='0,1' --n_ex=1000 --p=0.105 --eps=5.0 --n_iter=10000 \
                      --seed_num=300 --local_density=3.0 --seed_vect_num=1 --init_s=44 --radius=5 --min_s=7
############################################################################################################
