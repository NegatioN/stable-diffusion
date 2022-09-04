import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from transformers import AutoFeatureExtractor


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


ckpt = "/root/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
cfg="/root/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
config = OmegaConf.load(cfg)
model = load_model_from_config(config, ckpt)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
seed_everything(0)


batch_size = 1
prompt = "A dummy"
data = [batch_size * [prompt]]
precision_scope = autocast #if opt.precision=="autocast" else nullcontext

%%time
c = model.get_learned_conditioning(prompt[0])
c
size = 256
n_samples, C, H, W, factor = 1, 4, size, size, 8
scale, ddim_eta, ddim_steps = 8, 0.0, 1
start_code = torch.randn([n_samples, C, H // factor, W // factor], device=device)
sampler = PLMSSampler(model)
uc = None
if scale != 1.0:
    uc = model.get_learned_conditioning(batch_size * [""])

%%time
with torch.no_grad():
    with precision_scope("cuda"):
        with model.ema_scope():
            shape = [C, H // factor, W // factor]
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                conditioning=c,
                                                batch_size=n_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                x_T=start_code)

sampler??