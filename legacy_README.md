To install NCCL 2.26 (with timestamps in the log)
```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt update
apt install libnccl2=2.26.* libnccl-dev=2.26.*
```

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH


Build env from scratch:
```
conda create -n ece1770 python=3.11
conda activate ece1770
conda config --env --add channels conda-forge
conda config --env --add channels nvidia
conda config --env --add channels pytorch
conda config --env --set channel_priority flexible
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda install -c conda-forge nccl=2.26 -y
```