import sys
sys.path.append(".")
sys.path.append("..")

from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import DDIMScheduler, AutoencoderKL, StableDiffusionPipeline
import numpy as np
from PIL import Image
import os
import copy

from model.conv import Conv
from model.res import Res


class Base_warper:
    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.raw_vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
        self.raw_vae.to(self.device)

        # Finetune latent
        self.finetune_latent_model = None
        if config.finetune_latent:
            if config.finetune_latent_model.lower() == "cnn":
                self.finetune_latent_model = Conv()
            elif config.finetune_latent_model.lower() == "resnet":
                self.finetune_latent_model = Res()
            else:
                raise NotImplementedError(f"Not implemented model {self.finetune_latent_model}")
            self.finetune_latent_model.to(self.device)

        self.set_optimizer()
        self.set_scheduler()

    def load_ldm(self, config):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        MY_TOKEN = ''
        if config.ldm_version == "1.4":
            ldm_token = "CompVis/stable-diffusion-v1-4"
        elif config.ldm_version == "1.5":
            ldm_token = "runwayml/stable-diffusion-v1-5"
        elif config.ldm_version == "2.1":
            ldm_token = "stabilityai/stable-diffusion-2-1"
        ldm_stable = StableDiffusionPipeline.from_pretrained(ldm_token, use_auth_token=MY_TOKEN, scheduler=scheduler)

        return ldm_stable
    
    @torch.no_grad()
    def raw_latent2image(self, latent):
        latent = 1 / self.raw_vae.config.scaling_factor * latent
        img = self.raw_vae.decode(latent).sample
        img = (img / 2 + 0.5).clamp(0, 1)
        return img
        
    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.get_parameters(), lr=0.0001)

    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=50, T_mult=2, eta_min=0)


    # def load_checkpoint(self, ckpt_path, load_ckpt=-1):
    #     state_dict = None
        
    #     if load_ckpt == -1:
    #         ckpt_dir = os.listdir(ckpt_path)
    #         ckpt_dir.sort(key=lambda x: x.split("_")[-1])
            
    #         if ckpt_dir:
    #             ckpt_path = os.path.join(ckpt_path, ckpt_dir[-1])

    #             print(f"Loaded {ckpt_dir[-1]} epoch checkpoint")
    #             state_dict = torch.load(ckpt_path)
    #         else:
    #             print(f"No checkpoint found in {ckpt_path}")

    #     else:
    #         ckpt_path = os.path.join(ckpt_path, f"state_{load_ckpt}.pt")
            
    #         if os.path.isfile(ckpt_path):            
    #             print(f"Loaded {load_ckpt} epoch checkpoint")
    #             state_dict = torch.load(ckpt_path)
    #         else:
    #             print(f"No checkpoint found in {ckpt_path}")
            
    #     return state_dict
            
        
    # def save_checkpoint(self, epoch, ckpt_path):
    #     ckpt_filename = ""
    #     if self.config.finetune_decoder:
    #         ckpt_filename += "dec_"
    #         ckpt_filename += f"{self.config.finetune_decoder_model}_"
    #         if self.config.control_layers != None:
    #             ckpt_filename += f"{self.config.control_layers}_"
    #     if self.config.finetune_latent:
    #         ckpt_filename += "latent_"
    #         if self.config.finetune_latent_model == "resnet":
    #             ckpt_filename += "resnet_"
    #         elif self.config.finetune_latent_model == "cnn":
    #             ckpt_filename += "cnn_"

    #     ckpt_filename += f"ep_{epoch}.pt"

    #     save_ckpt_path = os.path.join(ckpt_path, ckpt_filename)

    #     save_params = {
    #             "epoch": epoch,
    #             "optimizer": self.optimizer.state_dict(),
    #     }

    #     if self.config.finetune_decoder:
    #         if self.config.finetune_decoder_model == "direct":
    #             save_params["decoder"] = self.model.vae.decoder.state_dict()
    #         elif self.config.finetune_decoder_model == "controlnet":
    #             save_params["vae_controlnet"] = self.model.state_dict()

    #     if self.config.finetune_latent:
    #         save_params[self.config.finetune_latent_model] = self.finetune_latent_model.state_dict()
        
    #     torch.save(
    #         save_params,
    #         save_ckpt_path,
    #     )
        
    #     print(f"Saved current state {epoch} at {ckpt_path}")
