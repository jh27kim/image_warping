import torch
from PIL import Image
import numpy as np
from math import log10, sqrt

def image2latent(image, vae):
    with torch.no_grad():
        image = (image - 0.5) * 2.0
        if image.dim == 3:
            image = image.permute(2, 0, 1).unsqueeze(0).to(vae)
        
        latents = vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
    return latents

def latent2image(latents, vae):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
        
    return image

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    image = np.array(Image.open(image_path))[:, :, :3] / 255.0
    image = image.astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    
    return image

def torch_to_pil(img):
    if img.dim() == 4:
        img = img.squeeze(0)
        
    img = img.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * 255.0).astype(np.uint8)
    img = Image.fromarray(img)
    
    return img

def mse(original, predicted):
    mse = np.mean((original - predicted) ** 2)
    return mse

def mse_psnr(original, predicted):
    mse_val = mse(original, predicted)
    if(mse_val == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse_val))
    return mse_val, psnr

def cos_sim(A, B):
    return np.dot(A, B)/(np.linalg.norm(A)*np.linalg.norm(B))

def sub_vec(A, B):
    return np.linalg.norm(A - B) / np.linalg.norm(A)