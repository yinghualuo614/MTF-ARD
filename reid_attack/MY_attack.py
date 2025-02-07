
from functools import partial

import kornia as K
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision.utils import save_image
from tqdm.auto import tqdm

def set_bn_dropout_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1 or classname.find("Dropout") != -1:
        m.eval()


class FGSM:
    def __init__(
        self,
        attacked_model,
        eps=8 / 255,
        random_start=True,
    ):
        self.attacked_model = attacked_model.apply(set_bn_dropout_eval)
        self.attacked_model.eval()
        self.eps = eps
        self.random_start = random_start

        self.device = next(attacked_model.parameters()).device

    def input_diversity(self, x):
        img_size = x.shape[-1]
        img_resize = int(img_size * self.resize_rate)

        if self.resize_rate < 1:
            img_size = img_resize
            img_resize = x.shape[-1]

        rnd = torch.randint(
            low=img_size, high=img_resize, size=(1,), dtype=torch.int32
        ).item()
        ratio = x.shape[2] / x.shape[3]
        rescaled = F.interpolate(
            x, size=[int(rnd * ratio), rnd], mode="bilinear", align_corners=False
        )
        h_rem = int((img_resize - rnd) * ratio)
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem, size=(1,), dtype=torch.int32).item()
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem, size=(1,), dtype=torch.int32).item()
        pad_right = w_rem - pad_left

        padded = F.pad(
            rescaled,
            [pad_left, pad_right, pad_top, pad_bottom],
            value=0,
        )

        return padded if torch.rand(1) < self.diversity_prob else x

    def forward(self, images):
        images = images.detach().to(self.device)

        criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=self.device)
        )
        criterion_cross = torch.nn.CrossEntropyLoss()
        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        feats = self.attacked_model(images)

        adv_images.requires_grad = True

        adv_feats = self.attacked_model(adv_images)

        # Calculate loss
        loss = criterion(adv_feats, feats)

        # loss = criterion_cross(adv_logits, pids)

        # Update adversarial images
        grad = torch.autograd.grad(loss, adv_images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

    def __call__(self, images):
        return self.forward(images)



class MIFGSM:
    def __init__(
        self,
        attacked_model,
        eps=8 / 255,
        alpha=1 / 255,
        steps=10,
        decay=1.0,
        random_start=True,
    ):
        self.attacked_model = attacked_model
        self.attacked_model.eval()
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.random_start = random_start

        self.device = next(attacked_model.parameters()).device

    def forward(self, images):
        images = images.detach().to(self.device)

        criterion = criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=self.device)
        )

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        feats = self.attacked_model(images)
        for _ in range(self.steps):
            adv_images.requires_grad = True

            adv_feats = self.attacked_model(adv_images)

            # Calculate loss
            loss = criterion(adv_feats, feats)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images)[0]

            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def __call__(self, images):
        return self.forward(images)

class PGD:
    def __init__(
        self,
        attacked_model,
        eps=8 / 255,
        alpha=1 / 255,
        steps=10,
        random_start=True,
    ):
        self.attacked_model = attacked_model
        self.attacked_model.eval()
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start

        self.device = next(attacked_model.parameters()).device

    def forward(self, images):
        images = images.detach().to(self.device)

        criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=self.device)
        )

        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        feats = self.attacked_model(images)
        for _ in range(self.steps):
            adv_images.requires_grad = True

            adv_feats = self.attacked_model(adv_images)

            # Calculate loss
            loss = criterion(adv_feats, feats)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def __call__(self, images):
        return self.forward(images)

class BIM:
    def __init__(
        self,
        attacked_model,
        eps=8 / 255,
        alpha=1 / 255,
        steps=10,
        random_start=True,
    ):
        self.attacked_model = attacked_model
        self.attacked_model.eval()
        self.eps = eps
        self.steps = steps
        self.alpha = alpha
        self.random_start = random_start

        self.device = next(attacked_model.parameters()).device

    def forward(self, images):
        images = images.detach().to(self.device)

        criterion = criterion = partial(
            torch.nn.CosineEmbeddingLoss(), target=torch.ones(1, device=self.device)
        )

        adv_images = images.clone().detach()
        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        feats = self.attacked_model(images)
        for _ in range(self.steps):
            adv_images.requires_grad = True

            adv_feats = self.attacked_model(adv_images)

            # Calculate loss
            loss = criterion(adv_feats, feats)

            # Update adversarial images
            grad = torch.autograd.grad(loss, adv_images)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

    def __call__(self, images):
        return self.forward(images)

