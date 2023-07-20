import sys
import os

sys.path.append(".")
sys.path.append("..")

from model.my_diffusers.models import AutoencoderKL_Pretrained 
from PIL import Image
import torch
import tyro
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from warper.so2_warper import SO2_warper
from utils.lora_utils.lora import inject_trainable_lora_extended
import itertools


class VAE_lora_dec(SO2_warper):
    def __init__(self, config) -> None:
        self.device = config.device
        self.autoencoder_pretrained = AutoencoderKL_Pretrained.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)

        self.target_replace_module = {"UNetMidBlock2D", "UpDecoderBlock2D"}
        self.lora_params, train_names = inject_trainable_lora_extended(self.autoencoder_pretrained.decoder, 
                                                                       target_replace_module=self.target_replace_module, 
                                                                       verbose=True,
                                                                       r=config.lora_rank)
        
        super().__init__(config) 

    def get_parameters(self):
        _params = itertools.chain(*self.lora_params)
        return _params


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
            assert "decoder_lora" in state_dict
            self.autoencoder_pretrained.decoder.load_state_dict(state_dict["decoder_lora"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
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
        }

        save_params["decoder_lora"] = self.autoencoder_pretrained.decoder.state_dict()

        torch.save(
            save_params,
            save_ckpt_path,
        )
        
        print(f"Saved current state {epoch} at {ckpt_path}")
    

@dataclass
class VAE_direct_config:
   # Training config
   device: int = 5

   # Fine tuning 
   finetune_decoder: bool = True
   finetune_decoder_model: str = "lora" # direct, controlnet, lora, controlnet_dec
   lora_rank: int = 16



if __name__ == "__main__":
    from torchinfo import summary
    config = tyro.cli(VAE_direct_config)
    vae = VAE_lora_dec(config)

    img = Image.open("/home/jh27kim/warp_latent/dataset/afhq/val/cat/image/flickr_cat_000008.jpg")
    img.save("./in.png")
    np_img = (np.array(img) / 255.0).astype(np.float32)
    torch_img = torch.tensor(np_img)
    torch_img = torch_img.permute(2, 0, 1).unsqueeze(0).to(config.device)
    print(torch.max(torch_img), torch.min(torch_img))

    latent = vae.image2latent(torch_img)
    img = vae.latent2image(latent)
    img = img.squeeze(0).permute(1, 2, 0)
    img = img.detach().cpu().numpy() * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save("./out3.png")
