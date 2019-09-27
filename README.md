# A Dual Approach to Scalable Verification of Deep Networks

This repository contains a simple implementation of the dual verification
formulation descriped in the paper: "A Dual Approach to Scalable Verification
of Deep Networks.", https://arxiv.org/abs/1803.06567.

The code analyses a pre-trained TensorFlow classifier network, and attempts
to prove that test examples remain correctly classified for _all_ perturbations
of the inputs up to some threshold.

## Installation

Deep-verify can be installed with the following command:

```bash
pip install git+https://github.com/deepmind/deep-verify`
```

Deep-verify will work with both the CPU and GPU version of tensorflow and
dm-sonnet, but to allow for that it does not list Tensorflow as a requirement,
so it is necessary to ensure that Tensorflow and Sonnet are installed
separately.

## Usage

This following command pre-trains a non-robust two-layer classifier on MNIST,
and verifies it with epsilon set to 0.02:

```bash
cd deep-verify/examples
python verify.py --model=tiny --epsilon=0.02
```

This following commands use interval-bound-propagation to pre-train a small
robust conv-net on MNIST with epsilon set to 0.1, and then verifies it with
the same epsilon:

```bash
cd interval-bound-propagation/examples
python train.py --model=small --epsilon=0.1 \
    --output_dir=/tmp/small_model --num_steps=60001
cd deep-verify/examples
python verify.py --model=small --epsilon=0.1 \
    --pretrained_model_path=/tmp/small_model/model-60000
```

## Giving credit

If you use this code in your work, we ask that you cite this paper:

Krishnamurthy Dvijotham, Robert Stanforth, Sven Gowal, Timothy Mann,
and Pushmeet Kohli. "A Dual Approach to Scalable Verification of
Deep Networks." _in UAI, 2018, pp. 550–559_.

## Acknowledgements

In addition to the people involved in the original publication, we would like
to thank Chongli Qin for her contributions.

