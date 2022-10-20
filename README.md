# Revisiting-PLMs
Revisiting Evolution-based &amp; free protein language models
## Env:
Jax:

https://github.com/kalininalab/alphafold_non_docker

Esm:

pip install fair-esm  # latest release, OR:

pip install git+https://github.com/facebookresearch/esm.git

## data:
ESMStructuralSplitDataset: 
| Name   | Description                                                                   | URL                                                                   |
|--------|-------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| splits | train/valid splits                                                            | https://dl.fbaipublicfiles.com/fair-esm/structural-data/splits.tar.gz |
| pkl    | pkl objects containing sequence, SSP labels, distance map, and 3d coordinates | https://dl.fbaipublicfiles.com/fair-esm/structural-data/pkl.tar.gz    |
| msas   | a3m files containing MSA for each domain                                      | https://dl.fbaipublicfiles.com/fair-esm/structural-data/msas.tar.gz   |

from https://github.com/facebookresearch/esm

Alphafold2 training data: 

https://registry.opendata.aws/openfold/ from Openfold(https://github.com/aqlaboratory/openfold)
