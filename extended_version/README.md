# Code for extended paper of CDFI

## Preparation

You need to complete the environment installation as documented in the parent directory. To run the experiments listed here, you need to change working directory to this one.

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
        --data_dir path/to/vimeo_triplet \
        --num_training_samples 1000 \
        --test_input ../test_data/middlebury_others/input/ \
        --test_gt ../test_data/middlebury_others/gt/ \
        --gpu_id 0 \
        --uid adacof_pruned_lr_1e-2_lam_1e-4_Np_10 \
        --force
~~~

Note that for pruning an existing (pre-trained) model, you need to provide the checkpoint by using `--checkpoint`. The L1 optimizer should be set to `OBProxSG` (you can also use your favorite sparsity-inducing optimizer as needed). `Np` is the number of Proximal Steps the optimizer runs, `lr` is the learning rate, and `lambda_` is the regularization parameter. Details of the hyperparameter settings for the experiments can be found in the paper.

## Get the sparsity report from the pruned model

~~~bash
python get_sparsity_report.py \
        --model adacof \
        --checkpoint ../checkpoints/adacof_pruned_lr_1e-2_lam_1e-4_Np_10.pth \
        --kernel_size 5 \
        --dilation 1 \
        --inactive
~~~

You will get the sparsity analysis for each layer like the following:

~~~bash
                               Name                 Density  In  Out   In' Out'
                   get_kernel.moduleConv1.0.weight   0.858    6   32    5   29
                   get_kernel.moduleConv1.2.weight   0.762   32   32   27   27
                   get_kernel.moduleConv1.4.weight   0.761   32   32   27   27
                   get_kernel.moduleConv2.0.weight   0.762   32   64   27   55
                   get_kernel.moduleConv2.2.weight   0.755   64   64   55   55
                   get_kernel.moduleConv2.4.weight   0.751   64   64   55   55
                   get_kernel.moduleConv3.0.weight   0.781   64  128   56  113
                   get_kernel.moduleConv3.2.weight   0.779  128  128  112  112
                   get_kernel.moduleConv3.4.weight   0.762  128  128  111  111
                   get_kernel.moduleConv4.0.weight   0.698  128  256  106  213
                   get_kernel.moduleConv4.2.weight   0.609  256  256  199  199
                   get_kernel.moduleConv4.4.weight   0.554  256  256  190  190
                   get_kernel.moduleConv5.0.weight   0.372  256  512  156  312
                   get_kernel.moduleConv5.2.weight   0.178  512  512  215  215
                   get_kernel.moduleConv5.4.weight   0.140  512  512  191  191
                 get_kernel.moduleDeconv5.0.weight   0.141  512  512  192  192
                 get_kernel.moduleDeconv5.2.weight   0.192  512  512  224  224
                 get_kernel.moduleDeconv5.4.weight   0.133  512  512  186  186
               get_kernel.moduleUpsample5.1.weight   0.070  512  512  135  135
                 get_kernel.moduleDeconv4.0.weight   0.234  512  256  247  123
                 get_kernel.moduleDeconv4.2.weight   0.285  256  256  136  136
                 get_kernel.moduleDeconv4.4.weight   0.321  256  256  144  144
               get_kernel.moduleUpsample4.1.weight   0.313  256  256  143  143
                 get_kernel.moduleDeconv3.0.weight   0.621  256  128  201  100
                 get_kernel.moduleDeconv3.2.weight   0.647  128  128  102  102
                 get_kernel.moduleDeconv3.4.weight   0.687  128  128  106  106
               get_kernel.moduleUpsample3.1.weight   0.562  128  128   95   95
                 get_kernel.moduleDeconv2.0.weight   0.739  128   64  110   55
                 get_kernel.moduleDeconv2.2.weight   0.703   64   64   53   53
                 get_kernel.moduleDeconv2.4.weight   0.735   64   64   54   54
               get_kernel.moduleUpsample2.1.weight   0.701   64   64   53   53
                 get_kernel.moduleWeight1.0.weight   0.808   64   64   57   57
                 get_kernel.moduleWeight1.2.weight   0.798   64   64   57   57
                 get_kernel.moduleWeight1.4.weight   0.765   64   25   55   21
                 get_kernel.moduleWeight1.7.weight   0.841   25   25   22   22
                  get_kernel.moduleAlpha1.0.weight   0.781   64   64   56   56
                  get_kernel.moduleAlpha1.2.weight   0.788   64   64   56   56
                  get_kernel.moduleAlpha1.4.weight   0.763   64   25   55   21
                  get_kernel.moduleAlpha1.7.weight   0.803   25   25   22   22
                   get_kernel.moduleBeta1.0.weight   0.782   64   64   56   56
                   get_kernel.moduleBeta1.2.weight   0.801   64   64   57   57
                   get_kernel.moduleBeta1.4.weight   0.783   64   25   56   22
                   get_kernel.moduleBeta1.7.weight   0.810   25   25   22   22
                 get_kernel.moduleWeight2.0.weight   0.800   64   64   57   57
                 get_kernel.moduleWeight2.2.weight   0.793   64   64   56   56
                 get_kernel.moduleWeight2.4.weight   0.767   64   25   56   21
                 get_kernel.moduleWeight2.7.weight   0.836   25   25   22   22
                  get_kernel.moduleAlpha2.0.weight   0.786   64   64   56   56
                  get_kernel.moduleAlpha2.2.weight   0.790   64   64   56   56
                  get_kernel.moduleAlpha2.4.weight   0.765   64   25   55   21
                  get_kernel.moduleAlpha2.7.weight   0.810   25   25   22   22
                   get_kernel.moduleBeta2.0.weight   0.784   64   64   56   56
                   get_kernel.moduleBeta2.2.weight   0.799   64   64   57   57
                   get_kernel.moduleBeta2.4.weight   0.795   64   25   57   22
                   get_kernel.moduleBeta2.7.weight   0.812   25   25   22   22
               get_kernel.moduleOcclusion.0.weight   0.815   64   64   57   57
               get_kernel.moduleOcclusion.2.weight   0.812   64   64   57   57
               get_kernel.moduleOcclusion.4.weight   0.809   64   64   57   57
               get_kernel.moduleOcclusion.7.weight   0.778   64    1   56    0

~~~