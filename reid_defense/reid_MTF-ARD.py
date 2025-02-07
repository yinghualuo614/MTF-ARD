
import os
import sys
import time
import datetime

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from pathlib import Path

import accelerate
from scipy.stats import norm
import math
import torch
import torch.nn as nn
from pytorch_metric_learning.losses import TripletMarginLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_metric_learning.miners import BatchEasyHardMiner
from torchvision.utils import save_image
from tqdm.auto import tqdm
import torch.nn.functional as F

from pytorch_reid_models.reid_models.data import (
    build_test_datasets,
    build_train_dataloader,
)
from pytorch_reid_models.reid_models.modeling import _build_reid_model
from pytorch_reid_models.reid_models.utils import set_seed, setup_logger
from reid_defense.eval_attack import MY_test


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = (
        torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    )  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pair wise distance between samples, shape [N, M]
      is_pos: positive index with shape [N, M]
      is_neg: negative index with shape [N, M]
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N]
    dist_ap, _ = torch.max(dist_mat * is_pos, dim=1)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N]
    dist_an, _ = torch.min(dist_mat * is_neg + is_pos * 1e9, dim=1)

    return dist_ap, dist_an


def weighted_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the weighted positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      is_pos:
      is_neg:
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
    """
    assert len(dist_mat.size()) == 2

    is_pos = is_pos
    is_neg = is_neg
    dist_ap = dist_mat * is_pos
    dist_an = dist_mat * is_neg

    weights_ap = softmax_weights(dist_ap, is_pos)
    weights_an = softmax_weights(-dist_an, is_neg)

    dist_ap = torch.sum(dist_ap * weights_ap, dim=1)
    dist_an = torch.sum(dist_an * weights_an, dim=1)

    return dist_ap, dist_an


def triplet_loss(embedding, targets, margin, norm_feat, hard_mining, reduction='mean'):
    r"""Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    if norm_feat:
        dist_mat = cosine_dist(embedding, embedding)
    else:
        dist_mat = euclidean_dist(embedding, embedding)

    # For distributed training, gather all features from different process.
    # if comm.get_world_size() > 1:
    #     all_embedding = torch.cat(GatherLayer.apply(embedding), dim=0)
    #     all_targets = concat_all_gather(targets)
    # else:
    #     all_embedding = embedding
    #     all_targets = targets

    N = dist_mat.size(0)
    is_pos = (
        targets.view(N, 1).expand(N, N).eq(targets.view(N, 1).expand(N, N).t()).float()
    )
    is_neg = (
        targets.view(N, 1).expand(N, N).ne(targets.view(N, 1).expand(N, N).t()).float()
    )

    if hard_mining:
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)
    else:
        dist_ap, dist_an = weighted_example_mining(dist_mat, is_pos, is_neg)

    y = dist_an.new().resize_as_(dist_an).fill_(1)

    if margin > 0:
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin, reduction=reduction)
    else:
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        # fmt: off
        if loss == float('Inf'): loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        # fmt: on

    return loss

def fea_logit_weight_calculate(trans_feat_s_list, mid_feat_t_list, output_feat_t_list, pids, teacher_output, stu_logit):
    bsz = pids.shape[0]
    loss_cosine_t_list = [F.cosine_similarity(mid_fea_s.view(mid_fea_s.size(0), mid_fea_s.size(1), -1), mid_fea_t.view(mid_fea_s.size(0), mid_fea_s.size(1), -1), dim=2).mean(-1) for mid_fea_s, mid_fea_t in zip(trans_feat_s_list, mid_feat_t_list)]
    num_teacher = len(loss_cosine_t_list)
    loss_triplet_t_list = torch.stack(loss_cosine_t_list, dim=0)
    fea_weight = F.softmax(1 * loss_triplet_t_list, dim=0)
    teacher_logit = torch.zeros_like(teacher_output[0])
    for weight, tea_logits in zip(fea_weight, teacher_output):
        teacher_logit += weight.view(weight.shape[0], 1) * tea_logits

    loss = jsdiv_loss_test(stu_logit, teacher_logit).sum()
    loss /= (1.0 * bsz)
    return loss, fea_weight

class CalWeight(nn.Module):
    def __init__(self, feat_s, feat_t_list):
        super(CalWeight, self).__init__()

        s_channel = feat_s.shape[1]
        for i in range(len(feat_t_list)):
            t_channel = feat_t_list[i].shape[1]
            setattr(self, 'embed'+str(i), Embed(s_channel, t_channel, 2, False))


    def forward(self, feat_s, feat_t_list, model_t_list=None):
        trans_feat_s_list = []
        output_feat_t_list = []
        stu_trans_logit_list = []
        s_H = feat_s.shape[2]
        s_W = feat_s.shape[-1]

        for i, mid_feat_t in enumerate(feat_t_list):
            t_H = mid_feat_t.shape[2]
            t_W = mid_feat_t.shape[-1]
            if s_H >= t_H:
                feat_s = F.adaptive_avg_pool2d(feat_s, (t_H, t_W))
            else:
                feat_s = F.interpolate(feat_s, size=(t_H, t_W), mode='bilinear')
            trans_feat_s = getattr(self, 'embed'+str(i))(feat_s)
            trans_feat_s_list.append(trans_feat_s)
            output_logit_s, output_feat_t = model_t_list[i][-1](trans_feat_s, is_s_to_t=True)
            # if i == 0:
            #     print(model_t_list[0][-1])
            output_feat_t_list.append(output_feat_t)
            stu_trans_logit_list.append(output_logit_s)
        return trans_feat_s_list, output_feat_t_list, stu_trans_logit_list

class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128, factor=2, convs=False):
        super(Embed, self).__init__()
        self.convs = convs
        if self.convs:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_in//factor, kernel_size=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in//factor, dim_in//factor, kernel_size=3, padding=1),
                nn.BatchNorm2d(dim_in//factor),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in//factor, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)
            )
        else:
            self.transfer = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=1),
                nn.BatchNorm2d(dim_out),
                nn.ReLU(inplace=True)
            )


    def forward(self, x):
        x = self.transfer(x)
        return x


def _kldiv_test(y_s, y_t, t):
    p_s = F.log_softmax(y_s / t, dim=1)
    p_t = F.softmax(y_t / t, dim=1)
    loss = (F.kl_div(p_s, p_t, reduction="none") * (t ** 2)).sum(-1)

    return loss


def jsdiv_loss_test( y_s, y_t, t=16):
    loss = (_kldiv_test(y_s, y_t, t) + _kldiv_test(y_t, y_s, t)) / 2
    return loss

def _kldiv(y_s, y_t, t):
    p_s = F.log_softmax(y_s / t, dim=1)
    p_t = F.softmax(y_t / t, dim=1)
    loss = F.kl_div(p_s, p_t, reduction="sum") * (t ** 2) / y_s.shape[0]
    return loss


def jsdiv_loss( y_s, y_t, t=16):
    loss = (_kldiv(y_s, y_t, t) + _kldiv(y_t, y_s, t)) / 2
    return loss

def set_bn_dropout_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1 or classname.find("Dropout") != -1:
        m.eval()

KL_loss = nn.KLDivLoss()
XENT_loss = nn.CrossEntropyLoss()
alpha = 1
temp = 1

def train_overhaul(
    accelerator,
    student_model,
    module_list,
    nat_teacher_model,
    train_loader,
    optimizer,
    miner,
    criterion_triplet,
    criterion_cross,
    max_epoch,
    epoch,
    adv_steps,
    alpha=2 / 255,
    eps=4 / 255,
):
    student_model.train()
    for module in module_list:
        module.train()
    for adv_teacher_model in module_list[1:]:
        for param in adv_teacher_model[1].parameters():
            param.requires_grad_(False)
    model_t_list = module_list[1:]

    bar = tqdm(
        train_loader,
        total=len(train_loader),
        desc=f"Epoch[{epoch}/{max_epoch}]",
        leave=False,
    )
    for batch_idx, (imgs, pids, camids) in enumerate(bar):
        imgs, pids, camids = imgs.cuda(), pids.cuda(), camids.cuda()

        student_model.apply(set_bn_dropout_eval)
        # adv_teacher_model.apply(set_bn_dropout_eval)
        adv_imgs = imgs.clone() + torch.empty_like(imgs).uniform_(-1e-2, 1e-2)
        for _ in range(adv_steps):
            adv_imgs.requires_grad_(True)

            logits, feats = student_model(adv_imgs)
            loss_triplet = criterion_triplet(feats, pids)
            loss_cross = criterion_cross(logits, pids)
            loss = loss_triplet + loss_cross

            grad = torch.autograd.grad(loss, adv_imgs)[0]

            # Update adversaries
            adv_imgs = adv_imgs.detach() + alpha * grad.sign()
            delta = torch.clamp(adv_imgs - imgs, min=-eps, max=eps)
            adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()
        student_model.train()

        student_nat_logits, student_nat_feat,  student_nat_feats = student_model(imgs, overhaul = True)

        student_adv_logits, student_adv_feat,  student_adv_feats = student_model(adv_imgs, overhaul = True)

        loss_student_adv_cross = criterion_cross(student_adv_logits, pids)
        loss_student_adv_triplet = criterion_triplet(student_adv_feat, pids)

        feat_t_list = []
        logit_t_list = []


        loss_triplet_t_list = [triplet_loss(feat_t[-1], pids, margin=0.3, norm_feat=False, hard_mining=True, reduction='none') for feat_t in feat_t_list]
        loss_t = torch.stack(loss_triplet_t_list, dim=0)
        logit_weight = (1.0 - F.softmax(1 * loss_t, dim=0)) / (len(model_t_list) - 1)

        bsz = student_adv_logits.shape[0]
        logit_fusion = torch.zeros_like(student_adv_logits)
        for weight, logit_t in zip(logit_weight, logit_t_list):
            logit_fusion += weight.view(weight.shape[0], 1) * logit_t
        loss_logit = jsdiv_loss_test(student_adv_logits, logit_fusion).sum()
        loss_logit /= (1.0 * bsz)

        mid_feat_t_list = [feat_t[-2] for feat_t in feat_t_list]

        trans_feat_s_list, output_feat_t_list, stu_trans_list = module_list[0](student_adv_feats[-1], mid_feat_t_list,
                                                               model_t_list)

        loss_fea, fea_weight = fea_logit_weight_calculate(trans_feat_s_list, mid_feat_t_list, stu_trans_list, pids, logit_t_list, student_adv_logits)

        loss = 0.05 * loss_student_adv_cross + 0.05 * loss_student_adv_triplet + 0.60 * loss_logit + 0.30 * loss_fea


        optimizer.zero_grad(True)
        loss.backward()
        # accelerator.backward(loss)
        optimizer.step()

        acc = (student_nat_logits.max(1)[1] == pids).float().mean()
        bar.set_postfix_str(f"loss:{loss.item():.1f} " f"acc:{acc.item():.1f}")
        bar.update()
    bar.close()


def main():
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"

    #teacher_path
    adv_tea_weight1 = '/reid_defense/logs/pgd_at/bagtricks_R50_4/market1501-bagtricks_R50_fastreid-120.pth'
    adv_tea_weight2 = '/reid_defense/logs/pgd_at/bagtricks_osnet_x1_0_fastreid_4/market1501-bagtricks_osnet_x1_0_fastreid-120.pth'
    adv_tea_weight3 = '/reid_defense/logs/pgd_at/bagtricks_R101_ibn_fastreid_4/market1501-bagtricks_R101_ibn_fastreid-120.pth'
    nat_tea_weight = '/reid_defense/logs/train/normal_train/market1501/market1501-bagtricks_R50_fastreid-120.pth'

    duke_tea_adv1 = '/reid_defense/logs/pgd_at/bagtricks_R50_4/dukemtmcreid-bagtricks_R50_fastreid-120.pth'
    duke_tea_adv2 = '/reid_defense/logs/pgd_at/bagtricks_osnet_x1_0_fastreid_4/dukemtmcreid-bagtricks_osnet_x1_0_fastreid-120.pth'
    duke_tea_adv3 = '/reid_defense/logs/pgd_at/bagtricks_R101_ibn_fastreid_batchsize128/dukemtmcreid-bagtricks_R101_ibn_fastreid-120.pth'
    duke_tea_nat = '/reid_defense/logs/train/normal_train/dukemtmcreid/dukemtmcreid-bagtricks_R101_ibn_fastreid-120.pth'

    new_duke_xunlian = '/reid_defense/logs/pgd_at/bagtricks_R101_ibn_fastreid_4_3/dukemtmcreid-bagtricks_R101_ibn_fastreid-120.pth'

    setup_logger(name="pytorch_reid_models.reid_models")
    logger = setup_logger(name="__main__")

    seed = 42
    set_seed(seed)

    accelerator = accelerate.Accelerator(mixed_precision="no")

    dataset_name = "market1501"
    test_dataset = build_test_datasets(dataset_names=[dataset_name], query_num=500)[
        dataset_name
    ]
    train_loader = build_train_dataloader(
        dataset_names=[dataset_name],
        transforms=["randomflip", "randomcrop", "rea"],
        batch_size=64,
        sampler="pk",
        num_instance=4,
    )

    num_classes_dict = {"dukemtmcreid": 702, "market1501": 751, "msmt17": 1041}
    num_classes = num_classes_dict[dataset_name]


    # 搭建学生模型
    student_model_name = "bagtricks_with_distiller_student_R34_fastreid"
    student_model = _build_reid_model(
        student_model_name,
        num_classes=num_classes
    ).cuda()
    student_model = accelerator.prepare(student_model)

    model_t_list = []

    
    adv_teacher1_model_name = "bagtricks_R50_fastreid"
    # Make sure load pretrained model
    os.environ["pretrain"] = "1"
    adv_teacher1_model = _build_reid_model(
        adv_teacher1_model_name,
        num_classes=num_classes,
        weights_path=adv_tea_weight1
    ).cuda()
    adv_teacher1_model = accelerator.prepare(adv_teacher1_model)
    model_t_list.append(adv_teacher1_model)

    adv_teacher2_model_name = "bagtricks_osnet_x1_0_fastreid"
    adv_teacher2_model = _build_reid_model(
        adv_teacher2_model_name,
        num_classes=num_classes,
        weights_path=adv_tea_weight2
    ).cuda()
    adv_teacher2_model = accelerator.prepare(adv_teacher2_model)
    model_t_list.append(adv_teacher2_model)

    adv_teacher3_model_name = "bagtricks_R101_ibn_fastreid"
    adv_teacher3_model = _build_reid_model(
        adv_teacher3_model_name,
        num_classes=num_classes,
        weights_path=adv_tea_weight3
    ).cuda()
    adv_teacher3_model = accelerator.prepare(adv_teacher3_model)
    model_t_list.append(adv_teacher3_model)

    data = torch.randn(2, 3, 256, 128).cuda()

    feat_t_list = []
    student_model.eval()
    for model_t in model_t_list:
        model_t.eval()
    feat_s, _ = student_model(data, is_feat=True)

    feat_t_list = [feat_t[-1] for feat_t in feat_t_list]
    cal_weight = CalWeight(feat_s[-1], feat_t_list)

    module_list = nn.ModuleList([])
    module_list.append(cal_weight)
    module_list.extend(model_t_list).cuda()


    nat_teacher_model_name = "bagtricks_R50_fastreid"
    # Make sure load pretrained model
    os.environ["pretrain"] = "1"
    nat_teacher_model = _build_reid_model(
        nat_teacher_model_name,
        num_classes=num_classes,
        weights_path=nat_tea_weight
    ).cuda()
    nat_teacher_model = accelerator.prepare(nat_teacher_model)

    adv_steps = 4
    max_epoch = 120
    optimizer = torch.optim.Adam(student_model.parameters(), lr=3.5e-4, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=7e-7)

    miner = BatchEasyHardMiner()
    criterion_triplet = TripletMarginLoss(margin=0.3)
    criterion_cross = nn.CrossEntropyLoss(label_smoothing=0.1)

    save_dir = Path(f"logs/train/resnet34/reid_MY-ARD_new_weight_yuan_0.05_0.05_0.60_0.30/five/market1501")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, max_epoch + 1):
        train_overhaul(
            accelerator,
            student_model,
            module_list,
            nat_teacher_model,
            train_loader,
            optimizer,
            miner,
            criterion_triplet,
            criterion_cross,
            max_epoch,
            epoch,
            adv_steps,
        )

        scheduler.step()

        if epoch % 10 == 0:
            torch.save(
                student_model[1].state_dict(),
                save_dir / f"{dataset_name}-{student_model_name}-{epoch}.pth",
            )
            results = MY_test(test_dataset, student_model)
            logger.info(f"Epoch {epoch:0>2} evaluate results:\n" + results)

    end_time = time.time()
    train_time = str(datetime.timedelta(seconds=end_time - start_time))
    logger.info(f"Finished. Total time (h:m:s): " + train_time)


if __name__ == "__main__":
    main()
