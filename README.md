# ScoreNet
PyTorch implementation of ["ScoreNet: Learning Non-Uniform Attention and Augmentation for
Transformer-Based Histopathological Image Classification"](https://openaccess.thecvf.com/content/WACV2023/papers/Stegmuller_ScoreNet_Learning_Non-Uniform_Attention_and_Augmentation_for_Transformer-Based_Histopathological_Image_WACV_2023_paper.pdf).

## Datasets
### BRACS
The [BRACS](https://www.bracs.icar.cnr.it/) dataset is organized as follows:
```
tree -d BRACS/BRACS_RoI/previous
BRACS/BRACS_RoI/previous
├── test
│   ├── 0_N
│   ├── 1_PB
│   ├── 2_UDH
│   ├── 3_ADH
│   ├── 4_FEA
│   ├── 5_DCIS
│   └── 6_IC
├── train
│   ├── 0_N
│   ├── 1_PB
│   ├── 2_UDH
│   ├── 3_ADH
│   ├── 4_FEA
│   ├── 5_DCIS
│   └── 6_IC
└── val
    ├── 0_N
    ├── 1_PB
    ├── 2_UDH
    ├── 3_ADH
    ├── 4_FEA
    ├── 5_DCIS
    └── 6_IC
```
Note that to be able to compare with existing baselines we used the "previous" version of the dataset.
### BACH
The [BACH](https://zenodo.org/record/3632035) dataset is organized as follows:
```
tree -d ICIAR2018_BACH_Challenge/Photos/ 
ICIAR2018_BACH_Challenge/Photos/
├── Benign
├── InSitu
├── Invasive
└── Normal

```

## Acknowledgment
This code relies on some elements of [DINO](https://github.com/facebookresearch/dino) and the accompanying code of [Differentiable Patch Selection for Image Recognition](https://github.com/google-research/google-research/tree/master/ptopk_patch_selection).

## Cite
```
@inproceedings{stegmuller2023scorenet,
  title={Scorenet: Learning non-uniform attention and augmentation for transformer-based histopathological image classification},
  author={Stegm{\"u}ller, Thomas and Bozorgtabar, Behzad and Spahr, Antoine and Thiran, Jean-Philippe},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={6170--6179},
  year={2023}
}
```