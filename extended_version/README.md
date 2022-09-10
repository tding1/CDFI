# Code for extended paper of CDFI

## Preparation

You need to complete the environment installation as documented in the parent directory. To run the experiments listed here, you need to change the working directory to this one.

## Pruning the baseline model

~~~bash
python pruning.py \
        --model adacof \
        --kernel_size 5 \
        --dilation 1 \
        --checkpoint ../checkpoints/adacof_F_5_D_1.pth \
        --optimizer OBProxSG \
        --Np 10 \
        --lr 1e-2 \
        --lambda_ 1e-4 \
        --data_dir /data/vimeo_triplet/ \
        --num_training_samples 1000 \
        --test_input ../test_data/middlebury_others/input/ \
        --test_gt ../test_data/middlebury_others/gt/
~~~