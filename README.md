# VDL-Surrogate
The source code for our IEEE VIS 2022 paper "VDL-Surrogate: A View-Dependent Latent-based Model for Parameter Space Exploration of Ensemble Simulations". 
This branch is for the Nyx dataset.

## Getting Started

### View-Dependent Latent Generation

<img src="https://github.com/trainsn/VDL-Surrogate/blob/Nyx/image/overview(a1).jpg" width="80%">

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

<img src="https://github.com/trainsn/VDL-Surrogate/blob/Nyx/image/overview(a2).jpg" width="40%">

Given the same selected viewpoint, we train a VDL-Predictor, which takes the simulation parameters as input and output predicted view-dependent latent representations:

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

###  Model Inference

<img src="https://github.com/trainsn/VDL-Surrogate/blob/Nyx/image/overview(b).jpg" width="23%">

Given the same selected viewpoint, we feed a new simulation parameter into the corresponding trained VDL-Predictor for a predicted view-dependent latent presentation and decode the latent representation by the trained RAE decoder to data space for visualization.

To evaluate VDL-Surrogate on the testing dataset, run 
```
cd vdl_predictor
python eval.py --root dataset \
               --direction x, y, or z \
               --sn \
               --data-size L_0 \
               --img-size H(==W) \
               --ch 64 \
               --resume path_to_trained_VDL-Predictor \
               --ae-resume path_to_trained_RAE \  
               --batch-size 2048 
```

To predict the simulation output given a particular simulation parameter setting, run 

```
cd vdl_predictor
python infer.py --root dataset \
               --direction x, y, or z \
               --sn \
               --data-size L_0 \
               --img-size H(==W) \
               --ch 64 \
               --resume path_to_trained_VDL-Predictor \
               --ae-resume path_to_trained_RAE \  
               --batch-size 2048 \
               --omm OmM \
               --omb OmB \
               --h H
```

Having the predicted view-dependent data, consider using the following repo for visualizatoin: 
[VolumeRenderer_Nyx](https://github.com/trainsn/VolumeRenderer/tree/nyx_vr_fuse).

To evalute the quality of generated visualizion images, run:
```
cd vdl_predictor
python eval_img.py --root path/to/dataset/root \
                   --tf transfer_function_id \ 
                   --mode sub_directory/to/images
```

## Citation

If you use this code for your research, please cite our paper.
```
@article{shi2022vdl,
  title={VDL-Surrogate: A View-Dependent Latent-based Model for Parameter Space Exploration of Ensemble Simulations},
  author={Shi, Neng and Xu, Jiayi and Li, Haoyu and Guo, Hanqi and Woodring, Jonathan and Shen, Han-Wei},
  journal={arXiv preprint arXiv:2207.13091},
  year={2022}
}
```

## Acknowledgments
Our code is inspired by [InSituNet](https://github.com/hewenbin/insitu_net).
