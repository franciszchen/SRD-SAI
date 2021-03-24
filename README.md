# SRD-SAI

This repository is an official PyTorch implementation of the paper **"Super-Resolution Enhanced Medical Image Diagnosis with Sample Affinity Interaction"** [[paper](https://www.researchgate.net/publication/348851387_Super-Resolution_Enhanced_Medical_Image_Diagnosis_with_Sample_Affinity_Interaction)] from **TMI 2021**.

<div align=center><img width="700" src=/fig/framework.png></div>

## Dependencies
* Python 3.6
* PyTorch >= 1.3.0
* numpy

## Quickstart 
* Train the SRD-SAI framework:
```python
python ./train_srd.py --job_type S --batch_size 8 --optim Adam --num_workers 4 --lr 5e-4 --lr_decay_interval 30 --lr_decay_gamma 0.5 --weight_decay 1e-5 --epochs 350 --dual_ratio 100 --regular_ratio 100 --rank_ratio 1 --ensemble_ce_ratio 2.0 --super_ce_ratio 1.0 --low_ce_ratio 1.0 --aux_ce_ratio 1.0 --resolution 128 --cross_str cross0 --lr_factor_srnet 0.01
```

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@article{chen2021super,
  title={Super-Resolution Enhanced Medical Image Diagnosis with Sample Affinity Interaction},
  author={Chen, Zhen and Guo, Xiaoqing and Woo, Peter YM and Yuan, Yixuan},
  journal={IEEE Transactions on Medical Imaging},
  year={2021},
  publisher={IEEE}
}
```
