# Revisiting Evolution-aware &amp; free protein language models as protein function predictors

## Env:
### Jax(Alphafold2):

https://github.com/kalininalab/alphafold_non_docker

### Pytorch(ESM-1b,MSA-Transformer):
1. As a prerequisite, you must have PyTorch installed(https://pytorch.org/get-started/locally/).

2. pip install fair-esm  # latest release, OR:

   pip install git+https://github.com/facebookresearch/esm.git

## data:
### For SSP & Contact map:
ESMStructuralSplitDataset: 
| Name   | Description                                                                   | URL                                                                   |
|--------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| splits | train/valid splits                                                            | https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz |
| pkl    | pkl objects containing sequence, SSP labels, distance map, and 3d coordinates | https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz    |
| msas   | a3m files containing MSA for each domain                                      | https://dl.fbaipublicfiles.com/fair-esm/structural-data/msas.tar.gz   |

from https://github.com/facebookresearch/esm

For Contact map Test: CAMEO(https://www.cameo3d.org/)

### For pretrain ESM-1b & MSA-Transformer
Alphafold2 training data: 

https://registry.opendata.aws/openfold/ 

from Openfold(https://github.com/aqlaboratory/openfold)








```bibtex
@article{hu2022exploring,
  title={Exploring evolution-aware \&-free protein language models as protein function predictors},
  author={Hu, Mingyang and Yuan, Fajie and Yang, Kevin K and Ju, Fusong and Su, Jin and Wang, Hui and Yang, Fei and Ding, Qiuyang},
  journal={arXiv preprint arXiv:2206.06583},
  year={2022}
}
```
