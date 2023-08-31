from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import shutil
from torch.optim.adam import Adam
from PIL import Image
import json
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchgeometry as tgm
import cv2


class DDIM_inversion:
    def __init__(self, 
                 model, 
                 NUM_DDIM_STEPS = 50):
        
        scheduler = DDIMScheduler(beta_start=0.00085, 
                                  beta_end=0.012, 
                                  beta_schedule="scaled_linear", 
                                  clip_sample=False,
                                  set_alpha_to_one=False)

        self.NUM_DDIM_STEPS = NUM_DDIM_STEPS
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        
    @property
    def scheduler(self):
        return self.model.scheduler
    
    
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        # if return_type == 'np':
        # if False
        #     image = (image / 2 + 0.5).clamp(0, 1)
        #     image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        #     image = (image * 255).astype(np.uint8)
        return image

    
    @torch.no_grad()
    def image2latent(self, image):
        # with torch.no_grad():
        #     if type(image) is Image:
        #         image = np.array(image)
        #     if type(image) is torch.Tensor and image.dim() == 4:
        #         latents = image
        #     else:
        #         image = torch.from_numpy(image).float() / 127.5 - 1
        #         image = image.permute(2, 0, 1).unsqueeze(0).to(self.model)
        #         latents = self.model.vae.encode(image)['latent_dist'].mean
        #         latents = latents * 0.18215

        with torch.no_grad():
            image = 2 * image - 1

            posterior = self.model.vae.encode(image).latent_dist
            latents = posterior.sample() * 0.18215

        return latents

    
    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
        
        return self.context
    
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    
    @torch.no_grad()
    def reverse_step(self, latents, prompt_embeds, start_t=0, guidance_scale=7.5):
        start_index = None
        for i, t in enumerate(self.model.scheduler.timesteps):
            if int(t.item()) == int(start_t):
                start_index = i
                break
                
        assert start_index != None
        
        timesteps = self.model.scheduler.timesteps[start_index:]
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.model.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.model.scheduler.step(noise_pred, t, latents).prev_sample
        
        return latents

    
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = {"0": latent}
        latent = latent.clone().detach()
        
        for i in range(self.NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent[str(t.item())] = latent
            
        return all_latent
    
    
    @torch.no_grad()
    def ddim_invert(self, image):
        latent = self.image2latent(image)
        ddim_latents = self.ddim_loop(latent)
        return ddim_latents
    
    
    # def invert(self, image_path: str, prompt: str):
    #     self.init_prompt(prompt)
    #     image_gt = load_512(image_path)
    #     ddim_latents = self.ddim_invert(image_gt)
        
    #     return image_gt, ddim_latents
    