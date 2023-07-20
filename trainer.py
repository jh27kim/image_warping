from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from model.vae_controlnet import VAE_controlNet
from model.vae_direct import VAE_direct
from model.vae_controlnet_dec import VAE_controlNet_dec
from model.vae_lora import VAE_lora_dec
import numpy as np
from PIL import Image
import json
import json
import os
import random
from torch.utils.data import DataLoader
import tyro
from dataclasses import dataclass
import wandb
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
from utils.transformation_utils import *

from data.dataloader import CustomImageDataset


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

def mse_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return mse, psnr


def prepare_dataloader(config, train=True):
    metadata = dict()
    if train:
        batch_size = config.train_batch_size
        with open(os.path.join(config.train_datadir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        shuffle = True
    else:
        batch_size = config.test_batch_size
        with open(os.path.join(config.test_datadir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        shuffle = False
    
    if train and config.use_single_image:
        # Single image 
        batch_size = 1
        selected_index = random.randint(0, len(metadata))
        single_img = metadata[str(selected_index)]
        for k, v in metadata.items():
            metadata[k] = single_img
        shuffle = False
            
    custom_image_dataset = CustomImageDataset(metadata)
    custom_dataloader = DataLoader(custom_image_dataset, batch_size=batch_size, shuffle=shuffle)

    return custom_dataloader

def add_gradient(target_model):
    _out = 0.0
    for n, p in target_model.named_parameters():
        if p.grad != None:
            _out += p.grad.sum()

    return _out

@dataclass
class TrainerConfig:
   # Training config
   seed: int = 42
   num_devices: int = 1
   device: int = 0
   train_batch_size: int = 1
   train_datadir: str = "./dataset/afhq/train/cat"
   max_epoch: int = 10
   use_single_image: bool = False
   use_single_pose: bool = False

   # Fine tuning 
   finetune_latent: bool = False
   finetune_decoder: bool = True
   reset_decoder: bool = True
   finetune_decoder_model: str = "lora" # direct, controlnet, lora, controlnet_dec
   finetune_latent_model: str = None # cnn, resnet
   control_layers: str = "all" # all, end
   control_scale: float = 1.0
   lora_rank: int = 32

   # Logging & Checkpoint
   ckpt_path: str = "./ckpt"
   load_ckpt: str = None
   debug_mode: bool = True
   debug_visualize_step: int = 100
   val_visualize_step: int = 50
   log_wandb: bool = True
   load_chekpoint: str = "none" # latest, epoch number, none
   
   # Diffusion model config
   ldm_version: str = "1.4"
   num_ddim_steps: int = 50
   guidance_scale: int = 7.5
   max_num_word: int = 77

   # Warping config
   warping: str = "SO2" # SO2, PERS2
   SO2_angle: int = 180
   SO2_translate: int = 10
   scale: float = 1.0
   perspective_displacement: int = 5

   # Test config
   test_batch_size: int = 1
   test_datadir: str = "./dataset/afhq/val/cat"
   val_step: int = 5000
   

def train(config):
    if config.finetune_decoder:
        if config.finetune_decoder_model == "direct":
            warper = VAE_direct(config)

        elif config.finetune_decoder_model == "controlnet":
            warper = VAE_controlNet(config)

        elif config.finetune_decoder_model == "controlnet_dec":
            warper = VAE_controlNet_dec(config)

        elif config.finetune_decoder_model == "lora":
            warper = VAE_lora_dec(config)

    # Load dataset
    train_dataloader = prepare_dataloader(config, train=True)
    test_dataloader = prepare_dataloader(config, train=False)
    TOTAL_TRAIN_DATASET = len(train_dataloader)
    TOTAL_TEST_DATASET = len(test_dataloader)

    L2_loss = torch.nn.MSELoss()

    # Load the last checkpoint
    START_EPOCH = 0
    if config.load_chekpoint != "none":
        START_EPOCH = warper.load_checkpoint(config.ckpt_path, config.load_ckpt)

    total_loss = 1e9
    for ep in range(START_EPOCH, config.max_epoch):
        train_loss = 0.0
        val_loss = 0.0
        
        for i, train_data in enumerate(train_dataloader):
            output_dict = apply_transformation(warper, config, train_data, return_all=config.debug_mode)
            SO2_warp_img = output_dict["SO2_warp_img"]
            tar_img = output_dict["tar_img"]

            warper.optimizer.zero_grad()
            loss = L2_loss(SO2_warp_img, tar_img)
            loss.backward()
            warper.optimizer.step()
            train_loss += loss.item()

            if config.debug_mode:
                debug_log = {
                'train/train loss': loss.item()
                } 
                debug_log['train/train learning rate'] = warper.optimizer.param_groups[0]['lr']
                if not i or (i+1) % config.debug_visualize_step == 0:
                    debug_log["train/train source_img"] = wandb.Image(torch_to_pil(output_dict["src_img"][0, ...]))
                    debug_log["train/train warped_img"] = wandb.Image(torch_to_pil(SO2_warp_img[0, ...]))
                    debug_log["train/train target_img"] = wandb.Image(torch_to_pil(tar_img[0, ...]))

                    SO2_warp_img_no_train = warper.raw_latent2image(output_dict["SO2_warp_latent"])
                    debug_log["train/train no_finetune"] = wandb.Image(torch_to_pil(SO2_warp_img_no_train[0, ...]))

                wandb.log(debug_log)
                print("debug logged", (ep*TOTAL_TRAIN_DATASET)+i+1)

            if (i+1) % config.val_step == 0:
                metric_dict = {
                    "ssim_raw": [],
                    "ssim_finetune": [],
                    "mse_raw": [],
                    "mse_finetune": [],
                    "psnr_raw": [],
                    "psnr_finetune": [],
                }

                with torch.no_grad():
                    for j, val_data in enumerate(test_dataloader):
                        output_dict = apply_transformation(warper, config, val_data, return_all=True)
                        SO2_warp_img = output_dict["SO2_warp_img"]
                        tar_img = output_dict["tar_img"]
                        SO2_warp_latent = output_dict["SO2_warp_latent"]
                        src_img = output_dict["src_img"]

                        loss = L2_loss(SO2_warp_img, tar_img)
                        val_loss += loss.item()
                        
                        if config.log_wandb:
                            test_log = {
                                'validation/test loss': loss.item()
                            }
                            SO2_warp_img_no_train = warper.raw_latent2image(SO2_warp_latent)

                            res_raw_numpy = np.array(torch_to_pil(SO2_warp_img_no_train))
                            res_finetune_numpy = np.array(torch_to_pil(SO2_warp_img))
                            res_target_numpy = np.array(torch_to_pil(tar_img))

                            mse_raw, psnr_raw = mse_psnr(res_target_numpy, res_raw_numpy)
                            mse_finetune, psnr_finetune = mse_psnr(res_target_numpy, res_finetune_numpy)
                            ssim_raw = ssim(res_target_numpy, res_raw_numpy, channel_axis=-1)
                            ssim_finetune = ssim(res_target_numpy, res_finetune_numpy, channel_axis=-1)

                            test_log['validation/test SSIM raw'] = ssim_raw
                            test_log['validation/test SSIM finetune'] = ssim_finetune
                            test_log['validation/test PSNR raw'] = psnr_raw
                            test_log['validation/test PSNR finetune'] = psnr_finetune
                            test_log['validation/test MSE raw'] = mse_raw
                            test_log['validation/test MSE finetune'] = mse_finetune

                            metric_dict["ssim_raw"].append(ssim_raw)
                            metric_dict["ssim_finetune"].append(ssim_finetune)
                            metric_dict["psnr_raw"].append(psnr_raw)
                            metric_dict["psnr_finetune"].append(psnr_finetune)
                            metric_dict["mse_raw"].append(mse_raw)
                            metric_dict["mse_finetune"].append(mse_finetune)

                            if (j + 1) % config.val_visualize_step == 0:
                                test_log["validation/test source_img"] = wandb.Image(torch_to_pil(src_img[0, ...]))
                                test_log["validation/test target_img"] = wandb.Image(torch_to_pil(tar_img[0, ...]))
                                test_log["validation/test warped_img"] = wandb.Image(torch_to_pil(SO2_warp_img[0, ...]))

                                test_log["validation/test no_finetune"] = wandb.Image(torch_to_pil(SO2_warp_img_no_train[0, ...]))

                            # wandb.log(test_log, step=(ep*TOTAL_TEST_DATASET)+j+1)
                            wandb.log(test_log)
                            print("logged", (ep*TOTAL_TEST_DATASET)+j+1)

                    val_loss /= TOTAL_TEST_DATASET
                    print(f"Epoch: {ep} Validation loss: {val_loss} Total loss: {total_loss}")

        train_loss /= TOTAL_TRAIN_DATASET
        print(f"Epoch: {ep} Train loss: {train_loss}")

        warper.scheduler.step()

        if (ep+1) == config.max_epoch or val_loss < total_loss:
            total_loss = val_loss
            warper.save_checkpoint(epoch=ep, ckpt_path=config.ckpt_path)

        for k, v in metric_dict.items():
            print(k, np.mean(np.array(v)))



def main(config):
    if config.log_wandb:
        wandb.login()
        wandb.init(project="afhq-SO2-test",
                config={"batch_size": config.train_batch_size})

    train(config)

def entrypoint():   
    main(tyro.cli(TrainerConfig))


if __name__ == "__main__":
    entrypoint()