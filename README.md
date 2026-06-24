This repository is the official implementation of "Sleep-CL: A Plug-and-Play Offline Consolidation Framework for Continual Learning via Neural Manifold Regulation"

# Overview
Continual learning (CL) enables models to learn sequential tasks without forgetting previously acquired knowledge, but catastrophic forgetting remains severe under non-stationary
data streams.

![comparison](https://github.com/yunkwankim/Sleep-CL/blob/main/Comp_CL_v4_1.jpg)

We propose Sleep-CL, a Plug-and-Play offline consolidation framework that augments, rather than replaces, the wake-phase optimization of existing CL agents. Inspired by
sleep-dependent memory consolidation, Sleep-CL inserts a structured post-learning stage composed of synaptic downregulation, non-rapid-eye-movement-inspired global consolidation, spindle-gated local replay, sharp-wave ripple style targeted replay, and adaptive model merging.

![Proposed](https://github.com/yunkwankim/Sleep-CL/blob/main/Proposed_framework_v2_1.jpg)

# Important License Notice
This repository is adapted from two original repositories:

### Repository A
- Original repository: [(https://github.com/danikiyasseh/CLOPS/)]
- Original author(s): [Kiyasseh, Dani and Zhu, Tingting and Clifton, David]
- Original license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

### Repository B
- Original repository: [(https://github.com/zqiao11/TSCIL)]
- Original author(s): [Zhongzheng Qiao, Quang Pham, Zhen Cao, Hoang H Le, P.N.Suganthan, Xudong Jiang, Ramasamy Savitha]
- Original license: Apache License 2.0

This version includes modifications for training configuration, and evaluation.
This modified version is also provided for non-commercial research purposes only.

# Requirements

To install requirements:

```pip install -r requirements.txt```

# Dataset

Download

The datasets can be downloaded from the follwing links:

1. [HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
2. [UWave](https://www.timeseriesclassification.com/description.php?Dataset=UWaveGestureLibraryAll)
3. [Dailysports](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities)
4. [WIDSM](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
5. [Chapman](https://figshare.com/collections/ChapmanECG/4560497/2)
6. [PhysioNet2020](https://moody-challenge.physionet.org/2020/)
7. [Split-CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)

# Continual Learning (CL) Algorithms

## CL Methods

1. [LwF](https://arxiv.org/abs/1606.09282)
2. [EWC](https://arxiv.org/abs/1612.00796)
3. [ER](https://arxiv.org/abs/1811.11682)
4. [DER](https://arxiv.org/abs/2004.07211)
5. [MAS](https://arxiv.org/abs/1711.09601)
6. [ASER](https://arxiv.org/abs/2009.00093)
7. [SI](https://arxiv.org/abs/1703.04200)
8. [CLOPS](https://www.nature.com/articles/s41467-021-24483-0)
9. [Generative Replay](https://arxiv.org/abs/1705.08690)

## Sleep-inspired CL Methods

1. [SIESTA](https://arxiv.org/abs/2303.10725)
2. [WSCL](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10695036)
3. [SRC](https://www.nature.com/articles/s41467-022-34938-7)


