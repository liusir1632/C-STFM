## Introduction
A cross-temporal meteorological prediction model for multivariate forecasting


## Overview

* `openstl/api` contains an experiment runner.
* `openstl/core` contains core training plugins and metrics.
* `openstl/datasets` contains datasets and dataloaders.
* `openstl/methods/` contains training methods for various video prediction methods.
* `openstl/models/` contains the main network architectures of various video prediction methods.
* `openstl/modules/` contains network modules and layers.
* `tools/` contains the executable python files tools/train.py and tools/test.py with possible arguments for training, validating, and testing pipelines.

## Installation

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```shell
conda env create -f environment.yml
conda activate STFM
```

<details close>
<summary>Dependencies</summary>

* argparse
* dask
* decord
* fvcore
* hickle
* lpips
* matplotlib
* netcdf4
* numpy
* opencv-python
* packaging
* pandas
* python<=3.10.8
* scikit-image
* scikit-learn
* torch
* timm
* tqdm
* xarray==0.19.0
</details>

## Getting Started

An example of training SimVP+gSTA on 2m temperature dataset.
```shell
python main.py -d t2m -m SimVP --model_type gsta --lr 1e-3 --ex_name t2m_simvp_gsta


C-STFM：
On the basis of successfully running simvp, change the /models/simvp.py file to the svp76.py file, keep the path consistent and run it. All changes to C-STFM are made in svp76.py.



```

<p align="right">(<a href="#top">back to top</a>)</p>




## Contact

If you have any questions, feel free to contact us through email. Enjoy!

- Jun Liu (17356582635@163.com), QingHai University

<p align="right">(<a href="#top">back to top</a>)</p>
