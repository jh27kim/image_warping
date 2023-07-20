import sys
import os

sys.path.append(".")
sys.path.append("..")

from model.my_diffusers.models import AutoencoderKL_ControlNet_Decoder
from model.my_diffusers.models import AutoencoderKL_Pretrained 
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from warper.so2_warper import SO2_warper


class VAE_controlNet_dec(SO2_warper):
    def __init__(self, config) -> None:
        self.device = config.device

        self.autoencoder_controlnet_dec = AutoencoderKL_ControlNet_Decoder.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        self.autoencoder_pretrained = AutoencoderKL_Pretrained.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)

        super().__init__(config)
        
        # Enable ControlNet encoder/zero conv gradients
        for name, params in self.autoencoder_controlnet_dec.named_parameters():
            if "post_quant_conv" in name or "encoder" in name or "quant_conv" in name:
                params.requires_grad_(False)
            elif "decoder" in name or "zero_conv" in name:
                params.requires_grad_(True)

        # Disable VAE decoder gradients
        for name, params in self.autoencoder_pretrained.named_parameters():
            params.requires_grad_(False)
        
    def get_parameters(self):
        _params = []
        _params += self.autoencoder_controlnet_dec.get_parameters()
        return _params

    def latent2image(self, src_latent, tar_latent):
        # NOTE: Image should be normalized -1 1
        src_latent = 1 / self.autoencoder_pretrained.config.scaling_factor * src_latent
        tar_latent = 1 / self.autoencoder_pretrained.config.scaling_factor * tar_latent

        tar_sample, condition = self.autoencoder_controlnet_dec.decode(z=tar_latent, 
                                                                       src_latent=src_latent,
                                                                       control_config=self.config)
            
        imgs = self.autoencoder_pretrained.decode(src_latent, condition).sample
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
            assert "vae_controlnet" in state_dict
            self.autoencoder_controlnet_dec.load_state_dict(state_dict["decoder"])
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
            if self.config.control_layers != None:
                ckpt_filename += f"{self.config.control_layers}_"
            if self.config.control_scale != 1.0:
                ckpt_filename += f"scale_{self.config.control_scale}_"

        ckpt_filename += f"ep_{epoch}.pt"

        save_ckpt_path = os.path.join(ckpt_path, ckpt_filename)

        save_params = {
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
        }

        save_params["vae_controlnet"] = self.autoencoder_controlnet_dec.state_dict()

        torch.save(
            save_params,
            save_ckpt_path,
        )
        
        print(f"Saved current state {epoch} at {ckpt_path}")
    

if __name__ == "__main__":
    from torchinfo import summary
    device = 1
    vae = VAE_controlNet_dec()

    img = Image.open("/home/jh27kim/warp_latent/dataset/afhq/val/cat/image/flickr_cat_000008.jpg")
    img.save("./in.png")
    np_img = (np.array(img) / 255.0).astype(np.float32)
    torch_img = torch.tensor(np_img)
    torch_img = torch_img.permute(2, 0, 1).unsqueeze(0).to(device)
    print(torch.max(torch_img), torch.min(torch_img))

    # summary(vae, input_size=(1, 4, 64, 64))

    latent = vae.image2latent(torch_img)
    img = vae.latent2image(latent, latent)
    img = img.squeeze(0).permute(1, 2, 0)
    img = img.detach().cpu().numpy() * 255.0
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img.save("./out3.png")
