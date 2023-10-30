# CLOVER
This is an implemention for our SIGKDD 2022 paper based on Pytorch.

[**Comprehensive Fair Meta-learned Recommender System**](https://arxiv.org/abs/2206.04789)

by Tianxin Wei, Jingrui He.

## Introduction

It's a general framework to comprehensively mitigate the unfairness issues in the cold-start recommender systems. 

## Requirements

- python==3.6
- torch==1.9.0
- numpy==1.19.5
- tqdm==4.64.0
- scikit-learn==0.24.2

Other versions of these packages may also work.

```python
python maml.py --cuda 0 --rerun --num_epoch 50 --loss 1 --seed 53
```

For our fair adversarial meta-learned recommende system, run:

```python
# just the first adversarial loss function without external information
python maml.py --cuda 0 --rerun --adv 1 --fm --num_epoch 50 --loss 1 --out 0 --outer 0 --inner_fc 0 --seed 53 --disable_inner_max

# just the first adversarial loss function with external information
python maml.py --cuda 0 --rerun --adv 2 --fm --num_epoch 50 --loss 1 --out 5 --outer 0 --inner_fc 0 --seed 53 --disable_inner_max

# the two adversarial loss function together
python maml.py --cuda 0 --rerun --adv 2 --adv2 1 --fm --num_epoch 50 --loss 1 --out 5 --out2 2 --outer 0 --inner_fc 0 --seed 53 --disable_inner_max --normalize --item_adv
```

The hyperparameters need to be further finetuned for each dataset to obtain better performance. In these commands, "adv" denotes the tradeoff factor of the first adversarial loss and adv2 represents the second. "loss=1" denotes use the cross-entropy loss. The argument "out" controls the input information to the adversarial network. "disable_inner_max" is to disable the optimization of task T_2 in the inner loop. "disable_inner_adv" is to disable the optimization of task T_1.

The final results are the average of five independent runs (seeds) with the best model. For the AUC metric, it is tested using a trained model to infer the sensitive attribute with function "fair_fine" in the "maml.py" file.

## Acknowledges

Part of our code references the [MeLU_pytorch](https://github.com/waterhorse1/MELU_pytorch) code repo. Many thanks for their public code.
