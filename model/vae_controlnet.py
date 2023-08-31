import sys
sys.path.append(".")
sys.path.append("..")

from model.my_diffusers.models import AutoencoderKL_ControlNet
from model.my_diffusers.models import AutoencoderKL_Pretrained 
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
from warper.so2_warper import SO2_warper


class VAE_controlNet(SO2_warper):
    def __init__(self, 
                 config=None) -> None:
        self.device = config.device

        self.autoencder_controlnet = AutoencoderKL_ControlNet.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)
        self.autoencoder_pretrained = AutoencoderKL_Pretrained.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(self.device)

        super().__init__(config)
        
        # Enable ControlNet encoder/zero conv gradients
        for name, params in self.autoencder_controlnet.named_parameters():
            if "post_quant_conv" in name: 
                params.requires_grad_(False)
            elif "encoder" in name or "quant_conv" in name or "zero_conv" in name:
                params.requires_grad_(True)
            elif "decoder" in name:
                params.requires_grad_(False)
        
        # Enable VAE decoder gradients
        for name, params in self.autoencoder_pretrained.named_parameters():
            if "decoder" in name or "post_quant_conv" in name:
                params.requires_grad_(True)
            elif "encoder" in name or "quant_conv" in name:
                params.requires_grad_(False)
        
    def get_parameters(self):
        _params = []
        _params += self.autoencder_controlnet.get_parameters()
        return _params

    def latent2image(self, latents, tar_img):
        # NOTE: Image should be normalized -1 1
        condition = self.autoencder_controlnet(tar_img) # Jaihoon 2023.07.10 Retrieve condition params
        
        latents = 1 / self.autoencoder_pretrained.config.scaling_factor * latents
        latent_mid = latents + condition.pop() # Jaihoon 2023.07.10

        imgs = self.autoencoder_pretrained.decode(latent_mid, condition).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    def image2latent(self, src_img):
        # imgs: [B, 3, H, W]
        src_img = 2 * src_img - 1
        posterior = self.autoencoder_pretrained.encode(src_img).latent_dist
        latents = posterior.sample() * self.autoencoder_pretrained.config.scaling_factor

        return latents

        
    # def forward(self, x):
    #     control_outs = self.autoencder_controlnet(x)
    #     res = self.autoencoder_pretrained(sample=x, 
    #                                       condition=control_outs)

    #     return res.sample
    

if __name__ == "__main__":
    from torchinfo import summary
    device = 1
    vae_controlnet = VAE_controlNet()

    img = Image.open("/home/jh27kim/warp_latent/dataset/afhq/val/cat/image/flickr_cat_000008.jpg")
    img.save("./in.png")
    np_img = (np.array(img) / 255.0).astype(np.float32)
    torch_img = torch.tensor(np_img)
    torch_img = torch_img.permute(2, 0, 1).unsqueeze(0).to(device)
    print(torch.max(torch_img), torch.min(torch_img))

    # out = vae_controlnet(torch_img)
    # out = (out / 2 + 0.5).clamp(0, 1)
    # out = out.squeeze(0).permute(1, 2, 0)
    # out = out.detach().cpu().numpy()
    # out = (out * 255.0).astype(np.uint8)
    # out = Image.fromarray(out)
    # out.save("./out.png")
    # print("saved image")

    params = vae_controlnet.get_parameters()
    print(len(params))

#     diffusion_vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
#     summary(diffusion_vae, input_size=(1, 3, 512, 512))

    # NOTE: Plug in dummy torch_img as condition variable
    latent, condition = vae_controlnet.image2latent(torch_img, torch_img)
    out2 = vae_controlnet.latent2image(latent, condition)
    out2 = out2.squeeze(0).permute(1, 2, 0)
    out2 = out2.detach().cpu().numpy()
    out2 = (out2 * 255.0).astype(np.uint8)
    print(out2.shape)
    out2 = Image.fromarray(out2)
    out2.save("./out2.png")
    

# if __name__ == "__main__":
#     from torchinfo import summary
#     device = 2

#     img = Image.open("/home/jh27kim/warp_latent/dataset/afhq/val/cat/image/flickr_cat_000008.jpg")
#     np_img = np.array(img).astype(np.float32)
#     torch_img = torch.tensor((np_img - 0.5) * 2.0)
#     torch_img = torch_img.permute(2, 0, 1).unsqueeze(0).to(device)

    # ControlNet Autoencoder
    # autoencder_controlnet = AutoencoderKL_ControlNet.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    # summary(autoencder_controlnet, input_size=(1, 3, 512, 512))
    # control_outs = autoencder_controlnet(torch_img)

    # Pretrained Autoencoder
    # autoencoder_pretrained = AutoencoderKL_Pretrained.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    # summary(autoencoder_pretrained, input_size=(1, 3, 512, 512), condition=control_outs)
    # outs = autoencoder_pretrained(torch_img, control_outs)
    
    # model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda")
    # summary(model, input_size=(1, 3, 512, 512))
