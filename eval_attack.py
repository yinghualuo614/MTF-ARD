import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
from tqdm.auto import tqdm
from prettytable import PrettyTable
from torch.utils import data

from pytorch_reid_models.reid_models.data import build_test_datasets
from pytorch_reid_models.reid_models.data.build import build_train_dataset
from pytorch_reid_models.reid_models.evaluate import Estimator
from pytorch_reid_models.reid_models.modeling import _build_reid_model
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_attack.MY_attack import FGSM, MIFGSM, PGD, BIM


def evaluate_reid(
    q_dataset,
    g_dataset,
    adv_q_datasets,
    target_model,
):
    q_dataloader = data.DataLoader(q_dataset, batch_size=128, num_workers=8)
    g_dataloader = data.DataLoader(g_dataset, batch_size=128, num_workers=8)

    adv_q_dataloaders = [
        data.DataLoader(adv_q_dataset, batch_size=128, num_workers=8)
        for adv_q_dataset in adv_q_datasets
    ]

    estimator = Estimator(target_model, g_dataloader)
    cmc, mAP, mINP = estimator(q_dataloader)
    adv_metrics = [
        estimator(adv_q_dataloader) for adv_q_dataloader in adv_q_dataloaders
    ]

    results = PrettyTable(field_names=["", "top1", "top5", "mAP", "mINP"])
    results.add_row(
        [
            "origin",
            f"{cmc[0]:.3f}",
            f"{cmc[4]:.3f}",
            f"{mAP:.3f}",
            f"{mINP:.3f}",
        ]
    )
    for idx, (adv_cmc, adv_mAP, adv_mINP) in enumerate(adv_metrics, start=1):
        results.add_row(
            [
                f"adv{idx}",
                f"{adv_cmc[0]:.3f}",
                f"{adv_cmc[4]:.3f}",
                f"{adv_mAP:.3f}",
                f"{adv_mINP:.3f}",
            ]
        )
    return str(results)


def generate_adv(q_dataset, attacker, input_label=False):
    all_adv_imgs, all_pids, all_camids = [], [], []
    q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
    for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
        imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
        if input_label:
            adv_imgs = attacker(imgs, pids, camids)
        else:
            adv_imgs = attacker(imgs)
        all_adv_imgs.append(adv_imgs.cpu())
        all_pids.append(pids.cpu())
        all_camids.append(camids.cpu())
    all_adv_imgs = torch.cat(all_adv_imgs)
    all_pids = torch.cat(all_pids)
    all_camids = torch.cat(all_camids)

    return data.TensorDataset(all_adv_imgs, all_pids, all_camids)


def generate_uap_adv(q_dataset, attacker, input_label=False):
    all_adv_imgs, all_pids, all_camids = [], [], []
    t_dataset = build_train_dataset([q_dataset.name], per_dataset_num=800)
    uap = attacker(t_dataset)
    q_dataloader = data.DataLoader(q_dataset, batch_size=32, num_workers=8)
    for imgs, pids, camids in tqdm(q_dataloader, desc="Generate adv", leave=False):
        imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()
        adv_imgs = torch.clamp(imgs + uap, 0, 1)
        all_adv_imgs.append(adv_imgs.cpu())
        all_pids.append(pids.cpu())
        all_camids.append(camids.cpu())
    all_adv_imgs = torch.cat(all_adv_imgs)
    all_pids = torch.cat(all_pids)
    all_camids = torch.cat(all_camids)

    return data.TensorDataset(all_adv_imgs, all_pids, all_camids)


# def test(test_dataset, model):
#     model.eval()
#     q_dataset, g_dataset = test_dataset
#     attacker1 = DIM(model, alpha=2 / 255, steps=10)
#     adv_q_dataset1 = generate_adv(q_dataset, attacker1)
#
#     # agent_model_name = "bagtricks_inception_v3_fastreid"
#     # agent_model = build_reid_model(agent_model_name, q_dataset.name)
#     # agent_model.cuda().eval().requires_grad_(False)
#     attacker2 = TIM(model, alpha=2 / 255, steps=10)
#     adv_q_dataset2 = generate_adv(q_dataset, attacker2)
#
#     results = evaluate_reid(
#         q_dataset, g_dataset, [adv_q_dataset1, adv_q_dataset2], model
#     )
#     return results

def MY_test(test_dataset, model, weights_path = None):
    model.eval()
    q_dataset, g_dataset = test_dataset

    #The default setting for EPS is 8/255
    attacker1 = FGSM(model)
    adv_q_dataset1 = generate_adv(q_dataset, attacker1)

    attacker2 = MIFGSM(model, alpha=2 / 255, steps=10)
    adv_q_dataset2 = generate_adv(q_dataset, attacker2)

    attacker3 = PGD(model, alpha=2 / 255, steps=10)
    adv_q_dataset3 = generate_adv(q_dataset, attacker3)

    attacker4 = BIM(model, alpha=2 / 255, steps=10)
    adv_q_dataset4 = generate_adv(q_dataset, attacker4)

    # attacker5 = DIM(model, alpha=2 / 255, steps=10)
    # adv_q_dataset5 = generate_adv(q_dataset, attacker5)
    #
    # attacker6 = TIM(model, alpha=2 / 255, steps=10)
    # adv_q_dataset6 = generate_adv(q_dataset, attacker6)

    results = evaluate_reid(
        q_dataset, g_dataset, [adv_q_dataset1, adv_q_dataset2, adv_q_dataset3, adv_q_dataset4], model)

    return results

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    #Fill in the model file path you want to test
    weights_path = "/reid_defense/logs/pgd_at/bagtricks_R50_4/market1501-bagtricks_R50_fastreid-120.pth"

    set_seed(42)
    logger = setup_logger("__main__")
    # build test dataset
    dataset_name = "market1501"
    test_dataset = build_test_datasets(dataset_names=[dataset_name], query_num=500)[
        dataset_name
    ]

    # build target model
    model_name = "bagtricks_with_distiller_student_R18_fastreid"
    num_classes_dict = {"dukemtmcreid": 702, "market1501": 751, "msmt17": 1041}
    model = _build_reid_model(
        model_name, num_classes=num_classes_dict[dataset_name], weights_path=weights_path
    )
    model.cuda().eval().requires_grad_(False)

    # results = test(test_dataset, model)
    # logger.info(f"evaluate results:\n" + results)

    results = MY_test(test_dataset, model)
    logger.info(f"evaluate results:\n" + results)



    # attack1
    # agent_model_name = "bagtricks_inception_v3_fastreid"
    # agent_model = build_reid_model(agent_model_name, dataset_name)
    # agent_model.cuda().eval().requires_grad_(False)
    # attacker1 = TIM(agent_model, alpha=1 / 255, steps=50)
    # adv_q_dataset1 = generate_adv(q_dataset, attacker1)
    #
    # # attack2
    # # attacker2 = Bandits(target_model)
    # # adv_q_dataset2 = generate_adv(q_dataset, attacker2, input_label=True)
    #
    # agent_model_names = [
    #     "bagtricks_inception_v3_fastreid",
    #     "bagtricks_inception_v4_fastreid",
    #     "bagtricks_inception_resnet_v2_fastreid",
    #     "bagtricks_mobilenet_v3_large_fastreid",
    # ]
    # agent_models = [
    #     build_reid_model(agent_model_name, dataset_name).cuda()
    #     for agent_model_name in agent_model_names
    # ]
    # for m in agent_models:
    #     m.requires_grad_(False)
    # attacker2 = MGAATIM(agent_models)
    # adv_q_dataset2 = generate_adv(q_dataset, attacker2)
    #
    # # attack3
    # attacker3 = MI(target_model)
    # adv_q_dataset3 = generate_adv(q_dataset, attacker3)
    #
    # # attack4
    # attacker4 = MIUAP(target_model,epoch=10)
    # adv_q_dataset4 = generate_uap_adv(q_dataset, attacker4)

    # evaluate
    # results = evaluate_reid(
    #     q_dataset,
    #     g_dataset,
    #     [adv_q_dataset1, adv_q_dataset2, adv_q_dataset3, adv_q_dataset4],
    #     target_model,
    # )
    # logger.info(f"ReID Metrics: \n" + results)


if __name__ == "__main__":
    main()
