## Robust Person Re-identification via Multi-Teacher Adversarial Distillation with Logit Fusion

## Datasets
| Datasets       | Link                                                                                   |
|----------------|----------------------------------------------------------------------------------------|
| Market-1501    | [Download](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ) |
| Dukemtmc-reid  | [Download](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view)      |

## Models
The teacher models are trained using the PGD method and are available for download on [Google Drive](https://drive.google.com/file/d/1AveC0s5LuWQVb5zGsMo78IWg64g6hZ9q/view?usp=sharing).

## Requirements
- CUDA 12.1 (or later)
- Python 3.10
- Pytorch 2.2.1 (or later)
- Tensorvision 0.17.1
- scipy
- tqdm
- accelerate

## Training
Before starting the training, update the weight path of the teacher model in lines 366-377 to the correct path, and then proceed to run the script.

```bash
python reid-MTF-ARD.py
```

## Acknowledge

- [light-reid] https://github.com/wangguanan/light-reid
- [ABD-Net] https://github.com/VITA-Group/ABD-Net
- [AGW-Net] https://github.com/mangye16/ReID-Survey/
- [AP-Net] https://github.com/CHENGY12/APNet
- [DeepPersonReid] https://github.com/KaiyangZhou/deep-person-reid
- [Fast-ReID] https://github.com/JDAI-CV/fast-reid
- [ReidStrongBaseline] https://github.com/michuanhaohao/reid-strong-baseline
- [TransReID] https://github.com/damo-cv/TransReID
