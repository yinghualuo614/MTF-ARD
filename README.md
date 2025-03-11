## Robust Person Re-identification via Multi-Teacher Adversarial Distillation with Logit Fusion
This repository is the official PyTorch implementation of "Robust Person Re-identification via Multi-Teacher Adversarial Distillation with Logit Fusion" , submitted to The Visual Computer.
## Datasets
| Datasets       | Link                                                                                   |
|----------------|----------------------------------------------------------------------------------------|
| Market-1501    | [Download](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ) |
| Dukemtmc-reid  | [Download](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view)      |

The processed datasets are available for download from the above links.
## Models
The teacher models are trained using the PGD method and are available for download on [Google Drive](https://drive.google.com/file/d/1AveC0s5LuWQVb5zGsMo78IWg64g6hZ9q/view?usp=sharing).

## Requirements
The codes are evaluated under the following environment settings and libraries:
- OS: Ubuntu
- GPU: NVIDIA RTX2080Ti
- CUDA 12.1 (or later)
- Python 3.10
- Pytorch 2.2.1 (or later)
- Tensorvision 0.17.1
- scipy
- tqdm
- prettytable
- accelerate

## Configuration files

- For path settings with datasets, see [pytorch_reid_models/reid_models/data/datasets/dataset_paths.yaml](pytorch_reid_models/reid_models/data/datasets/dataset_paths.yaml)
- For model parameters setting, see [pytorch_reid_models/reid_models/modeling/models_config.yaml](pytorch_reid_models/reid_models/modeling/models_config.yaml)
- For fast-reid models settings, see [pytorch_reid_models/reid_models/modeling/third_party_models/FastReID/configs](pytorch_reid_models/reid_models/modeling/third_party_models/FastReID/configs)
  

## Train
Before training, you need to update the teacher model weight path in lines 360-368 to the correct path, select the dataset to be run in line 379, select the name of the student model to be built in line 396, and then run the script with the command:

```bash
python reid_MTF-ARD.py
```

## Test
Before testing, you need to update the test model weight path in line 150 to the correct path, and select the relevant dataset and model.

```bash
python eval_attack.py
```

## Relevant links

- [light-reid] https://github.com/wangguanan/light-reid
- [ABD-Net] https://github.com/VITA-Group/ABD-Net
- [AGW-Net] https://github.com/mangye16/ReID-Survey/
- [AP-Net] https://github.com/CHENGY12/APNet
- [DeepPersonReid] https://github.com/KaiyangZhou/deep-person-reid
- [Fast-ReID] https://github.com/JDAI-CV/fast-reid
- [ReidStrongBaseline] https://github.com/michuanhaohao/reid-strong-baseline
- [TransReID] https://github.com/damo-cv/TransReID

## References

If you use any part of this code in your research, please cite our paper:
Citations are updated when the preprint version of our article is published on Research Square.
