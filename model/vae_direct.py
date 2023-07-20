import sys
sys.path.append(".")
sys.path.append("..")

from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import os
import tyro
from dataclasses import dataclass
from model.my_diffusers.models import AutoencoderKL_Pretrained 
from warper.so2_warper import SO2_warper
from utils.train_utils import weight_reset


class VAE_direct(SO2_warper):
    def __init__(self, config):
        self.device = config.device
        self.autoencoder_pretrained = AutoencoderKL_Pretrained.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)

        # Freeze encoder, enable decoder
        for name, param in self.autoencoder_pretrained.encoder.named_parameters():
            param.requires_grad_(False)
        for name, param in self.autoencoder_pretrained.decoder.named_parameters():
            param.requires_grad_(True)
        
        if config.reset_decoder:
            print("WARNING: Reset decoder")
            self.autoencoder_pretrained.decoder.apply(weight_reset)

        super().__init__(config)

    def get_parameters(self):
        _param = []
        print("Training parameters : VAE decoder")
        _param += list(self.autoencoder_pretrained.decoder.parameters())

        return _param

    def latent2image(self, src_latent):
        # NOTE: Image should be normalized -1 1
        src_latent = 1 / self.autoencoder_pretrained.config.scaling_factor * src_latent

        imgs = self.autoencoder_pretrained.decode(src_latent).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

        
    def image2latent(self, img):
        # imgs: [B, 3, H, W]
        img = 2 * img - 1

        posterior = self.autoencoder_pretrained.encode(img).latent_dist
        latents = posterior.sample() * self.autoencoder_pretrained.config.scaling_factor

        return latents

    def load_checkpoint(self, ckpt_path, load_ckpt=-1):
        state_dict = None
        
        ckpt_path = os.path.join(ckpt_path, load_ckpt)
            
        if os.path.isfile(ckpt_path):            
            print(f"Loaded {load_ckpt} epoch checkpoint")
            state_dict = torch.load(ckpt_path)
            assert "direct_decoder" in state_dict
            self.autoencoder_pretrained.decoder.load_state_dict(state_dict["direct_decoder"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
            self.scheduler.load_state_dict(state_dict["scheduler"])
            return state_dict["epoch"]
        
        else:
            print(f"No checkpoint found in {ckpt_path}")
            return 0
            
        
    def save_checkpoint(self, epoch, ckpt_path):
        ckpt_filename = ""
        if self.config.finetune_decoder:
            ckpt_filename += "dec_"
            ckpt_filename += f"{self.config.finetune_decoder_model}_"

        ckpt_filename += f"ep_{epoch}.pt"

        save_ckpt_path = os.path.join(ckpt_path, ckpt_filename)

        save_params = {
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
        }

        save_params["direct_decoder"] = self.autoencoder_pretrained.decoder.state_dict()

        torch.save(
            save_params,
            save_ckpt_path,
        )
        
        print(f"Saved current state {epoch} at {ckpt_path}")



@dataclass
class VAE_direct_config:
   # Training config
   device: int = 7
   reset_decoder: bool = True

if __name__ == "__main__":
    from torchinfo import summary
    config = tyro.cli(VAE_direct_config)
    
    img = Image.open("/home/jh27kim/warp_latent/dataset/afhq/val/cat/image/flickr_cat_000008.jpg")
    img.save("./in.png")
    np_img = (np.array(img) / 255.0).astype(np.float32)
    torch_img = torch.tensor(np_img)
    torch_img = torch_img.permute(2, 0, 1).unsqueeze(0).to(config.device)

    raw_decoder = VAE_direct(config)
    latent = raw_decoder.image2latent(torch_img)
    img = raw_decoder.latent2image(latent)
    img = img.squeeze(0).permute(1, 2, 0)
    img = img.detach().cpu().numpy() * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save("./out3.png")