# VDL-Surrogate
The source code for our IEEE VIS 2022 paper "VDL-Surrogate: A View-Dependent Latent-based Model for Parameter Space Exploration of Ensemble Simulations". 
This branch is for the Nyx dataset.

## Getting Started

### View-Dependent Latent Generation

<img src="https://github.com/trainsn/VDL-Surrogate/blob/Nyx/image/overview(a1).jpg" width="70%">

Given a sampled view-dependent data and a selected viewpoint, we train a train a Ray AntoEncoder (RAE):

```
cd rae
python main.py --root dataset \
               --direction x, y, or z \
               --sn \ 
               --weighted \ 
               --data-size L_0 \
               --img-size H(==W) \
               --ch 64 \
               --load-batch 8 \
               --batch-size 2816 \
               --check-every 16 \
               --log-every 50
```

### VDL-Predictor Training

<img src="https://github.com/trainsn/VDL-Surrogate/blob/Nyx/image/overview(a2).jpg" width="35%">

We train a VDL-Predictor, which takes the simulation parameters as input and output predicted view-dependent latent representations:

```
cd vdl_predictor
python main.py --root dataset \
               --direction x, y, or z \
               --sn \ 
               --data-size L_0 \
               --img-size H(==W) \
               --ch 64 \
               --batch-size 1 \
               --check-every 25 \
               --log-every 20
```




