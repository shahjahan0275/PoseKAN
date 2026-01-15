# PoseKAN
[to be updated]

# [PoseKAN: Adaptive Graph Kolmogorov-Arnold Network for 3D Human Pose Estimation](https://doi.org/10.48550/arXiv.2511.08809) [International Conference on 3D Vision 2026]

This repository contains the official PyTorch implementation of the  flexible graph Kolmogorov-Arnold Network (PoseKAN), a novel framework that extends KANs to graph-based learning for 3D human pose estimation authored by Abu Taib Mohammed Shahjahan and A. Ben Hamza. If you discover our code to be valuable for your research, kindly consider including the following citation:

 
 ```
@article{shahjahan2026PoseKAN,
   title={PoseKAN: Adaptive Graph Kolmogorov-Arnold Network for 3D Human Pose Estimation},
   author={Shahjahan, Abu Taib Mohammed and Hamza, A Ben},
   conference={International Conference on 3D Vision 2026},
   year={2026},      
 }
```

## Network Architecture

<div align="center">
  <img src="https://raw.githubusercontent.com/shahjahan0275/PoseKAN/main/demo/PoseKAN_Architecture.png" 
       alt="PoseKAN_Architecture" width="600">
</div>

The PyTorch implementation for PoseKAN


## Qualitative and quantitative results

![Greeting](https://github.com/shahjahan0275/PoseKAN/blob/main/demo/smoking.gif)

## Results on Human3.6M 	

|      Method       |  MPJPE (P1)   | PA-MPJPE (P2) |
| ------------------| ------------- | ------------- |
|      [SemGCN](https://github.com/garyzhao/SemGCN)      |    57.6mm     |      -        |
|    [High-order](https://github.com/ZhimingZo/HGCN)     |    55.6mm     |    43.7mm     |
|     [HOIF-Net](https://github.com/happyvictor008/Higher-Order-Implicit-Fairing-Networks-for-3D-Human-Pose-Estimation)      |    54.8mm     |    42.9mm     |
| [Weight Unsharing](https://github.com/tamasino52/Any-GCN)  |    52.4mm     |    41.2mm     |
|      [MM-GCN](https://github.com/JaeYungLee/MM_GCN)       |    51.7mm     |    40.3mm     |
|     [Modulated](https://github.com/ZhimingZo/Modulated-GCN)     |    49.4mm     |    39.1mm     |
|     [GraphMLP](https://github.com/Vegetebird/GraphMLP)     |    48.0mm     |    38.4mm     |
|       Ours        |    46.7mm     |    38.3mm     |

## Quick Start

This repository is built upon Python v3.8 and Pytorch v1.8.0 on Ubuntu 20.04.4 LTS. All experiments are conducted on a single NVIDIA RTX 4500 GPU with 20GB of memory.

## Dependencies

Please make sure you have the following dependencies installed:

- PyTorch >= 1.8.0
- NumPy
- Matplotlib

## Dataset

Our model is evaluated on [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [MPI-INF-3DHP](http://vision.imar.ro/human3.6m/description.php) datasets. Please put the datasets in <span style="background-color: #f0f0f0">./dataset</span> directory.

### Human3.6M
2D detections for Human3.6M dataset are provided by [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) Pavllo et al.

### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset in the same way as [PoseAug](https://github.com/jfzhang95/PoseAug). Please refer to [DATASETS.md](https://github.com/jfzhang95/PoseAug/blob/main/DATASETS.md) to prepare the dataset file.

## Training from Scratch
To initiate the training of our model, utilizing the identified 2D keypoints (HR-Net) along with pose refinement, please execute the following command:
```
python main_graph.py  --pro_train 1 --beta 0.2 --k hrn --batchSize 512 --hid_dim 384 --save_model 1  --save_dir './checkpoint/train_result/' --post_refine --save_out_type post --show_protocol2
```
To initiate the training of our model, utilizing the ground truth 2D keypoints excluding pose refinement and non-local layer, please execute the following command:
```
python main_graph.py  --pro_train 1 --beta 0.2 --k gt --batchSize 512 --hid_dim 384 --save_model 1 --save_dir './checkpoint/train_result/' --show_protocol2   
```

## Evaluation

### Human3.6M
To evaluate our model using the detected 2D keypoints (HR-Net) with pose refinement, please run the following command:
```
python main_graph.py -k hrn --beta 0.2 --batchSize 512 --hid_dim 384 --previous_dir './checkpoint/train_result/' --save_out_type post --save_dir './checkpoint/test_result/' --gsnet_gcn_reload 1 --module_gsnet_model [model_gsnet].pth --post_refine --post_refine_reload 1 --post_refine_model `[model_post_refine]`.pth --show_protocol2 --nepoch 2   
```

To evaluate our model using the ground truth 2D keypoints without incorporating pose refinement and and non-local layer, please run the following command:
```
python main_graph.py -k gt --beta 0.2 --batchSize 512 --hid_dim 384 --previous_dir './checkpoint/train_result/' --save_dir './checkpoint/test_result/' --save_out_type xyz --gsnet_gcn_reload 1 --module_gsnet_model `[model_gsnet]`.pth --show_protocol2 --nepoch 2   
```

## Evaluating Our Pre-trained Models

The pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1gV2Dzuz3fJah3C56wnPr52-FBJEZuoQM). Put them in the `./checkpoint/train_result` directory.

To evaluate our pre-trained model using the ground truth 2D keypoints, please run:

```
python main_graph.py -k gt --beta 0.2 --batchSize 512 --hid_dim 384 --previous_dir './checkpoint/train_result/' --save_dir './checkpoint/test_result/' --save_out_type xyz --gsnet_gcn_reload 1 --module_gsnet_model model_gsnet_gcn_7_eva_xyz_3649.pth --show_protocol2 --nepoch 2  
```

# Acknowledgement

Our code makes references to the following repositories.
- [Modulated GCN](https://github.com/ZhimingZo/Modulated-GCN).
- [SemGCN](https://github.com/garyzhao/SemGCN).
- [PoseAug](https://github.com/jfzhang95/PoseAug).
- [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).

We thank the authors for sharing their code and kindly request that you also acknowledge their contributions by citing their work.
