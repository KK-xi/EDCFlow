# EDCFlow
<img width="2042" height="894" alt="image" src="https://github.com/user-attachments/assets/8f7eb70c-9fdc-4e5e-bd5e-b6f910ae2604" />

This repository contains the source code for our paper: **[EDCFlow: Exploring Temporally Dense Difference Maps for Event-based  Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf#:~:text=In%20this%20work%2C%20we%20take%20advantage%20of%20the,achieve%20high-quality%20flow%20es-timation%20at%20a%20higher%20resolution.)** in CVPR2025.


# Requirements
The code has been tested with PyTorch 1.12 and Cuda 11.3.

    conda create --name edcflow
    conda activate edcflow
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

# Pretrained wieghts
Pretrained model was saved in

    chekpoints/final_checkpoint.pth.tar

# Required Data
DSEC datasets and MVSEC datasets:

Similar to [TMA](https://github.com/ispc-lab/TMA/tree/main?tab=readme-ov-file#dsec-dataset-preparation), we use pre-generated event volumes and flows saved in .npz files.

# Training

    python DSEC_train_main.py --data_root data_path
    
Please choose your own datasets folder name. The checkpoints will be saved in `checkpoints/'.

# Evaluation

    python evaluate.py --data_root data_path

Please modify the corresponding checkponts path name.

# Citation
If you find this codebase helpful for your research, please cite our paper:

    @inproceedings{liu2025edcflow,
    title={EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation},
    author={Liu, Daikun and Cheng, Lei and Wang, Teng and Sun, Changyin},
    booktitle={CVPR},
    year={2025},
    }



