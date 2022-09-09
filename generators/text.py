import hashlib
import os.path
import time

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from einops import rearrange
from imwatermark import WatermarkEncoder
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import autocast
from tqdm import trange, tqdm
from transformers import AutoFeatureExtractor

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from repositories.query import Query
from repositories.sampler import SamplerEnum


class TextGenerator:
    config: str = "configs/stable-diffusion/v1-inference.yaml"
    model_checkpoint_path: str = "models/ldm/stable-diffusion-v1/model.ckpt"
    batch_size: int = 1
    n_rows: int = 0
    latent_channels: int = 4
    down_sampling_factor: int = 8
    ddim_eta: float = 0.0
    samples: int = 1
    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    output_path: str = "./images"

    def seed(self, seed: int):
        seed_everything(seed)

    def load_model_from_config(self, config, ckpt, verbose=False):
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

    def create_watermark_encoder(self):
        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

        return wm_encoder

    def put_watermark(self, img, wm_encoder=None):
        if wm_encoder is not None:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = wm_encoder.encode(img, 'dwtDct')
            img = Image.fromarray(img[:, :, ::-1])
        return img

    def load_replacement(self, x):
        try:
            hwc = x.shape
            y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
            y = (np.array(y) / 255.0).astype(x.dtype)
            assert y.shape == x.shape
            return y
        except Exception:
            return x

    def numpy_to_pil(self, images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def check_safety(self, x_image):
        safety_checker_input = self.safety_feature_extractor(self.numpy_to_pil(x_image), return_tensors="pt")
        x_checked_image, has_nsfw_concept = self.safety_checker(images=x_image,
                                                                clip_input=safety_checker_input.pixel_values)
        assert x_checked_image.shape[0] == len(has_nsfw_concept)
        for i in range(len(has_nsfw_concept)):
            if has_nsfw_concept[i]:
                x_checked_image[i] = self.load_replacement(x_checked_image[i])
        return x_checked_image, has_nsfw_concept

    def generate(self, query: Query):
        self.seed(query.seed)
        config = OmegaConf.load(self.config)
        model = self.load_model_from_config(config, self.model_checkpoint_path)
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)

        if query.sampler == SamplerEnum.plms:
            sampler = PLMSSampler(model)
        elif query.sampler == SamplerEnum.ddim:
            sampler = DDIMSampler(model)
        else:
            raise ValueError('Invalid sampler')

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        base_count = len(os.listdir(self.output_path))

        data = [self.batch_size * [query.prompt]]
        precision_scope = autocast

        start_code = None

        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()

                    for n in trange(query.iterations, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            unconditional_conditioning = None

                            if query.scale != 1.0:
                                unconditional_conditioning = model.get_learned_conditioning(self.batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            conditioning = model.get_learned_conditioning(prompts)
                            shape = [
                                self.latent_channels,
                                query.height // self.down_sampling_factor,
                                query.height // self.down_sampling_factor
                            ]

                            samples_ddim, _ = sampler.sample(
                                S=query.sampling_steps,
                                conditioning=conditioning,
                                batch_size=self.samples,
                                shape=shape,
                                verbose=False,
                                unconditional_guidance_scale=query.scale,
                                unconditional_conditioning=unconditional_conditioning,
                                eta=self.ddim_eta,
                                x_T=start_code
                            )

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            x_checked_image, has_nsfw_concept = self.check_safety(x_samples_ddim)

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img = self.put_watermark(img, self.create_watermark_encoder())

                                img.save(os.path.join(self.output_path, query.filename()))
                                base_count += 1
                    toc = time.time()
