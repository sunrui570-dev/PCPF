# Towards Robust UDA Semantic Segmentation with Finer Prototype Construction and Pseudo-Label Filtering

## Brief
This is the implementation of paper: Towards Robust UDA Semantic Segmentation with Finer Prototype Construction and Pseudo-Label Filtering

<span style="color:gray;">⚠ Note: The source code is currently incomplete and will be fully released once the manuscript is accepted by the journal.</span>

## Overview Framework
<img src="resources/overview.png" width="900">


## Dependency and Installation
Ubuntu 20.04
Python 3.8.5
Cuda version 11.0.3

1. Create Conda Environment

```shell
conda create --name pcpr python=3.8.5
conda activate pcpr
```

2. The requirements can be installed with:

```shell
pip install -r requirements.txt 
pip install mmcv-full==1.3.7  # requires the other packages to be installed first
```

3. Please, download the MiT-B5 ImageNet weights provided by [SegFormer](https://github.com/NVlabs/SegFormer?tab=readme-ov-file#training)
from their [OneDrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ) and put them in the folder `pretrained/`.

## Dataset Setup

|      |  Dataset   | Download Link                                                |
| :--: | :--------: | ------------------------------------------------------------ |
|  1   | Cityscapes | [Link](https://www.cityscapes-dataset.com/downloads/)        |
|  2   |    GTA     | [Link](https://download.visinf.tu-darmstadt.de/data/from_games/) |
|  3   |  Synthia   | [Link](http://synthia-dataset.net/downloads/)                |




## Training

```shell
python run_experiments.py --config configs/pcpr/gtaHR2csHR_pcpr.py
```

The logs and checkpoints are stored in `work_dirs/`.

## Inference
```shell
python -m tools.test ${CONFIG_FILE} ${CHECKPOINT_FILE} --show-dir ${SHOW_DIR}_HR --opacity 1 
```

## Results
<img src="resources/result2.png" width="900">

## Results
<img src="resources/results1.png" width="1200">

## License & Acknowledgements
We are very grateful for these excellent works: [MIC](https://github.com/lhoyer/MIC),[HRDA](https://github.com/lhoyer/HRDA),[DAFormer](https://github.com/lhoyer/DAFormer),[MMSegmentation](https://github.com/open-mmlab/mmsegmentation),[SegFormer](https://github.com/NVlabs/SegFormer),[DACS](https://github.com/vikolss/DACS). Please follow their respective licenses for usage and redistribution. Thanks for their awesome works.

## Contact
Feel free to contact me if there is any question. (Rui Sun: sunrui4509@163.com)


