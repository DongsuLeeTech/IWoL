# IWoL

---

### Metadrive


1. Create conda environment for MetaDrive
   ```bash
   cd ~/IWoL/MetaDrive
   conda env create -f environment.yaml
   conda activate [your env name]
   ```

2. MetaDrive installation (Metadrive version: https://drive.google.com/file/d/13XeZ2h9h9g-To3phFurFBmADHujnkvIT/view?usp=sharing)
   ```bash
   cd ~/
   git clone https://github.com/metadriverse/metadrive.git
   cd metadrive
   pip install -e .
   ```

3. Install dependencies
   ```bash
   cd ~/IWoL/MetaDrive
   pip install -e .
   pip install -r requirements.txt
   pip install wandb setproctitle absl-py tensorboardX imageio wandb[media]
   ```

4. Test Metadrive running example
   ```bash
   cd onpolicy/scrips/train_metadrive_scripts/
   sh train_inter_small_ImIWoL.sh
   ```

#### [Trouble shooting: simplpbr version] 
If you encounter issues related to `panda3d-simplepbr`, try downgrading it to version 0.9.x:
   ```bash
   pip uninstall -y panda3d-simplepbr
   pip install --no-cache-dir "panda3d-simplepbr==0.9.*"
   ```   

If you encounter other version issues, we recommend reinstalling specific versions of numpy, scipy, matplotlib, and seaborn:
   ```bash
   pip install --upgrade --force-reinstall "numpy==1.26.4" "scipy==1.10.1" "matplotlib<3.9" "seaborn<0.13"
   ```

---

### Robotarium

1. Create conda environment for MetaDrive
   ```bash
   cd ~/IWoL/Robotarium
   conda env create -f environment.yaml
   conda activate [your env name]
   ```
   
2. Install dependencies
   ```bash
   pip install -e .
   pip install -r requirements.txt # ignore pip version conflict
   pip install wandb gym etproctitle absl-py tensorboardX tensorflow imageio wandb[media] matplotlib cvxopt scipy seaborn psutil
   ```
   
#### Trouble shooting
If you encounter issues with `Generator` from `typing_extensions`, try installing it via conda:
   ```bash
   pip install --upgrade "typing_extensions>=4.7"
   ```

---

### MQE

1. Create conda environment
   ```bash
   conda create -n mqe python=3.8
   ``` 

2. Install PyTorch and Isaac Gym.
    
    - Install appropriate PyTorch version from https://pytorch.org/.
      > pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    
    - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym. Ubuntu20.04 and Python3.8 recommended.
      > tar -xzvf IsaacGym_Preview_4_Package.tar.gz cd isaacgym/python && pip install -e .

3. Install dependencies
   ```bash
   cd ~/IWoL/MQE/
   pip install -e .
   pip install wandb gym etproctitle absl-py tensorboardX tensorflow imageio wandb[media] matplotlib cvxopt scipy seaborn psutil
   ```

4. Train MQE
    ```bash
    cd ~/IWoL/MQE/
    python train_imiwol.py --headless --sim_device cuda:0 --rl_device cuda:0
    ```

#### Trouble shooting
If you encounter issues with `RuntimeError: Ninja is required to load C++ extension` not found, install it via conda:
   ```bash
  conda install -y ninja
   ```

---

### Bidexhands

1. Create conda environment
   ```bash
   conda create -n bidexhands python=3.8
   ``` 

2. Install PyTorch and Isaac Gym.
    
    - Install appropriate PyTorch version from https://pytorch.org/.
      > pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
    
    - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym. Ubuntu20.04 and Python3.8 recommended.
      > tar -xzvf IsaacGym_Preview_4_Package.tar.gz cd isaacgym/python && pip install -e .

3. Install dependencies
   ```bash
   cd ~/IWoL/BiDexHands/
   pip install -e .
   pip install wandb gym etproctitle absl-py tensorboardX tensorflow imageio wandb[media] matplotlib cvxopt scipy seaborn psutil
   ```

4. Train BiDexHands
   ```bash
   train.py --task=ShadowHandTwoCatchUnderarm --algo=imiwol --num_envs=250 --headless --compute_device_id 0 --sim_device cuda:0 --graphics_device_id 0
   ```

#### Trouble shooting
If you encounter issues with `RuntimeError: Ninja is required to load C++ extension` not found, install it via conda:
   ```bash
  conda install -y ninja
   ```
