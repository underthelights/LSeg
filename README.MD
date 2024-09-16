# Language-driven Grasp Detection (LGrasp)
The repo contains PyTorch Implementation of paper [Language-Driven Grasp Detection](https://arxiv.org/abs/2201.03546). 

## Usage
### Successed Environment
I ran on a Google Colab T4. You can run [training_google_colab.ipynb](https://github.com/trislee02/lseg/blob/main/training_google_colab.ipynb) on Google Colab with a hosted runtime T4.

- `GPU`: NVIDIA Tesla T4
- `nvcc --version`: cuda_12.2.r12.2/compiler.33191640_0
- `CUDA Version`: 12.2
- `torch.__version__` : 2.1.0+cu121
- `torchaudio.__version__` : 2.1.0+cu121
- `torchvision.__version__` : 0.16.0+cu121
- `torchmetrics.__version__`: 0.7.0
- `pytorch-lightning` : 1.9.0
- `torch-encoding` git+https://github.com/zhanghang1989/PyTorch-Encoding@c959dab8312b637fcc7edce83607acb4b0f82645

### Installation
#### Create a Conda environment
```
conda create -n lang-seg
conda activate lang-seg
```
#### Install libraries
```
conda install ipython
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning==1.9.0
pip install opencv-python
pip install imageio
pip install ftfy regex tqdm
pip install altair==4.0.0
pip install streamlit==1.11.0
pip install protobuf==3.20.3
pip install timm
pip install tensorboardX
pip install matplotlib
pip install test-tube
pip install wandb
pip install torchmetrics==0.7.0
pip install git+https://github.com/zhanghang1989/PyTorch-Encoding/
pip install git+https://github.com/openai/CLIP.git
```

## Acknowledgement
Thanks to the code base from [Lang-Seg](https://github.com/isl-org/lang-seg), [DPT](https://github.com/isl-org/DPT), [Pytorch_lightning](https://github.com/PyTorchLightning/pytorch-lightning), [CLIP](https://github.com/openai/CLIP), [Pytorch Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [Streamlit](https://streamlit.io/), [Wandb](https://wandb.ai/site)
