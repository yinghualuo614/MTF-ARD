## Robust Person Re-identification via Multi-Teacher Adversarial Distillation with Logit Fusion
The code of paper  Robust Person Re-identification via Multi-Teacher Adversarial Distillation with Logit Fusion in The Visual Computer
![MTF-ARD Method](https://github.com/yinghualuo614/MTF-ARD/blob/master/reid_attack/MTF-ARD-method.png)
We propose  Multi-Teacher Adversarial Distillation with Logit Fusion (MTF-ARD), a novel method that leverages multiple adversarial teacher models to provide comprehensive and student-friendly guidance. Our method adaptively assigns weights to different teacher models based on their predictive confidence and their similarity with the student model. Specifically, the predictive confidence is obtained based on the Triplet loss between the predicted distribution of the teacher model and the ground-truth, and the similarity is assessed through the channel-wise cosine similarity between the final layer features of the teacher model and the student model. By integrating information from multiple teacher models, the student model is transferred the richer and more robust adversarial knowledge, and its generalization ability against adversarial attacks is significantly improved. More details can be found in the [Reid_MTF-ARD.py](https://github.com/yinghualuo614/MTF-ARD/blob/master/reid_MTF-ARD.py).
## Datasets
| Datasets       | Link                                                                                   |
|----------------|----------------------------------------------------------------------------------------|
| Market-1501    | [Download](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ) |
| Dukemtmc-reid  | [Download](https://drive.google.com/file/d/1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O/view)      |

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

- For path settings with datasets, see [pytorch_reid_models/reid_models/data/datasets/dataset_paths.yaml](reid_models/data/datasets/dataset_paths.yaml)
- For model parameters setting, see [pytorch_reid_models/reid_models/modeling/models_config.yaml](reid_models/modeling/models_config.yaml)
- For fast-reid models settings, see [pytorch_reid_models/reid_models/modeling/third_party_models/FastReID/configs](reid_models/modeling/third_party_models/FastReID/configs)
  

## Train
Before training, you need to update the teacher model weight path in lines 352-360 to the correct path, select the dataset to be run in line 371, select the name of the student model to be built in line 388, and then run the script with the command:

```bash
python reid_MTF-ARD.py
```

## Test
Before testing, you need to update the test model weight path in line 150 to the correct path, and select the relevant dataset and model.

```bash
python eval_attack.py
```

## Experiment
The following is the experimental demonstration of ResNet18 as a student model on the Market-1501 dataset.
| 防御方法   |  FGSM (Rank-1) | FGSM (mAP) | MI-FGSM (Rank-1) | MI-FGSM (mAP) | PGD (Rank-1) | PGD (mAP) | BIM (Rank-1) | BIM (mAP) |
|------------|---------------|------------|------------------|---------------|--------------|-----------|--------------|-----------|
| PGD_AT     | 57.8%     | 34.0%     | 54.2%     | 31.1%     | 52.2%     | 30.1%     | 50.6%     | 20.9%      |
| ARD        | 69.4%     | 47.1%     | 59.0%     | 38.8%     | 53.4%     | 34.8%     | 52.8%     | 35.1%      |
| IAD        | 69.0%     | 46.5%     | 59.8%     | 39.5%     | 55.8%     | 35.3%     | 56.6%     | 37.2%      |
| RSLAD      | 69.4%     | 46.2%     | 60.6%     | 40.5%     | 54.0%     | 35.9%     | 54.2%     | 35.7%      |
| MTARD      | 72.4%     | 52.1%     | 59.0%     | 43.0%     | 56.2%     | 39.9%     | 55.2%     | 41.0%      |
| AdaAD      | 65.8%     | 44.3%     | 58.4%     | 38.1%     | 56.2%     | 35.9%     | 53.8%     | 35.7%      |
| DGAD       | 67.2%     | 44.7%     | 60.0%     | 38.4%     | 56.8%     | 36.2%     | 55.8%     | 35.6%      |
| Fair-MTARD | 66.4%     | 45.3%     | 63.4%     | 40.3%     | 60.6%     | 37.9%     | 59.2%     | 37.9%      |
| B-MTARD    | 72.4%     | 51.4%     | 61.6%     | 43.0%     | 55.0%     | 39.8%     | 57.4%     | 40.1%      |
| AVER       | 71.0%     | 49.7%     | 60.2%     | 42.4%     | 56.8%     | 39.9%     | 56.2%     | 39.6%      |
| MTF-ARD    | **73.2%** | ​**53.8%** | ​**66.0%** | ​**47.3%** | ​**62.8%** | ​**44.0%** | ​**60.2%** | ​**43.2%**  |

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
