from torchgeometry.core.conversions import deg2rad
from torchgeometry.core.homography_warper import homography_warp

import torch
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch.nn.functional as F

def apply_transformation(warper, config, data_dict, return_all=False):    
    src_img = data_dict["image"].to(config.device)
    src_latent = warper.image2latent(src_img)

    # Warp latent and image
    if config.warping == "SO2":
        warped_latent = apply_rigid_body(src_latent, warper, config, isImage=False)
        warped_image = apply_rigid_body(src_img, warper, config, isImage=True)
    
    elif config.warping == "PERS2":
        warped_latent = apply_perspective(src_latent, warper, config, isImage=False)
        warped_image = apply_perspective(src_img, warper, config, isImage=True)
        
    # Selectively pass another layer 
    if config.finetune_latent:
        warped_latent_refined = warper.finetune_latent_model(warped_latent)
    else:
        warped_latent_refined = warped_latent

    if config.finetune_decoder_model == "controlnet_dec":
        SO2_warp_img = warper.latent2image(src_latent, warped_latent_refined)
    else:
        SO2_warp_img = warper.latent2image(warped_latent_refined)
    
    return_dict = {
        "tar_img": warped_image,
        "SO2_warp_img": SO2_warp_img, 
    }

    if return_all:
        return_dict["SO2_warp_latent"] = warped_latent_refined
        return_dict["src_img"] = src_img

    return return_dict


def apply_rigid_body(src, warper, config, isImage=False):
    batch_size = src.shape[0]

    angle = (torch.rand(batch_size) - 0.5) * 2 # -1 ~ 1
    angle *= config.SO2_angle
    scale = torch.ones(batch_size) * config.scale

    center = torch.ones(batch_size, 2)
    center[..., 0] = src.shape[3] / 2  # x
    center[..., 1] = src.shape[2] / 2  # y

    translate = (torch.rand_like(center) - 0.5) * 2.0 # -1 ~ 1
    translate *= config.SO2_translate

    if config.use_single_pose:
        angle = (torch.ones(angle.shape) * 90).to(config.device)
        scale = torch.ones(scale.shape).to(config.device)
        translate = torch.ones(translate.shape).to(config.device) * config.SO2_translate

    if isImage:
        translate *= 8 # 1 latent pixel == 8x8 patch

    rigid_mat = warper.get_rotation_matrix2d(center=center.to(config.device), 
                                              angle=angle.to(config.device), 
                                              scale=scale.to(config.device), 
                                              translate=translate.to(config.device))

    _, _, src_h, src_w = src.shape
    transformed_output = warper.warp_affine(src, 
                                            rigid_mat, 
                                            dsize=(src_h, src_w), 
                                            interpolation_mode="bilinear", 
                                            padding_mode="reflection")
    
    return transformed_output

def apply_perspective(src, warper, config, isImage=False):
    src_size = src.shape[2:]
    src_w, src_h = src_size

    points_src = torch.FloatTensor([[[0, 0], [src_w-1, 0], [0, src_h-1], [src_w - 1, src_h - 1]]]).to(config.device)
    points_dir = torch.FloatTensor([[[1, 1], [-1, 1], [1, -1], [-1, -1]]]).to(config.device)
    points_displacement = torch.rand_like(points_src).to(config.device) * (config.perspective_displacement)
    if config.use_single_pose:
        points_displacement = torch.ones(points_displacement.shape).to(config.device) * config.perspective_displacement

    points_displacement *= points_dir
    if isImage:
        points_displacement *= 8  # 1 latent pixel == 8x8 patch
    points_dst = points_src + points_displacement

    pers_mat = warper.get_perspective_transform(points_src, points_dst)
    transformed_output = warper.warp_perspective(src, 
                                                 pers_mat, 
                                                 dsize=(src_size[0], src_size[1]), 
                                                 interpolation_mode="bilinear", 
                                                 padding_mode="reflection")
    
    return transformed_output