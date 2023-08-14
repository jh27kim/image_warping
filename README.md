# Image warping & refinement
This repository provides 2D and 3D warping source codes. 

## Image warping

Supports $SE(2)$ and $SO(2)$ transformation warping and refinement using a U-Net network.

For 2D transformation task, use any available dataset. Here we used [AFHQ dataset](https://www.v7labs.com/open-datasets/afhq). 

|                  Input image          |           Warped image                |       Latent warp + refined image     |
|:-------------------------------------:|:-------------------------------------:|:-------------------------------------:|
| ![](https://github.com/jh27kim/image_warping/assets/58447982/7d82cca0-6c74-456b-ad15-cbde6e17cd67) | ![](https://github.com/jh27kim/image_warping/assets/58447982/8892419b-7477-4195-a2d2-ad291085f543) | ![](https://github.com/jh27kim/image_warping/assets/58447982/b4c71b68-9469-4cbe-a59f-6336aa96d5af) | 



## Depth warping

Supports depth warping given ground truth / predicted depth map. Many of the code is borrowed from [here](https://github.com/NagabhushanSN95/Pose-Warping)

### Depth warping formulation

$T_{src \rightarrow dst} = T_{src}^{-1} \cdot T_{dst}$

$T_{proj} = K_{dst} \cdot T_{src \rightarrow dst}$

$X_{reproj} = K_{src}^{-1} \cdot X_{src}$

$X_{warped} = N(K_{dst} \cdot (T_{src}^{-1} \cdot T_{dst}) \cdot K_{src}^{-1} \cdot X_{src})$


Ground truth depth map is hard to acquire in practice. 
You can either extract depth from a mesh, or use 3D scanned depth map from publicly available dataset like [OmniObject3D](https://omniobject3d.github.io/).

|                  Input image          |          360 3D object synthesis     |
|:-------------------------------------:|:-------------------------------------:|
| <img width="256" height="256" src="https://github.com/jh27kim/image_warping/assets/58447982/c0278d83-f5ee-461f-81b6-8797d9395466"> | ![](https://github.com/jh27kim/image_warping/assets/58447982/8b4e1589-0920-4363-87aa-11c9df1ad11d) | 
| ![](https://github.com/jh27kim/image_warping/assets/58447982/3f75d86a-11ef-42c3-b2ba-81d8bbc56460) | ![](https://github.com/jh27kim/image_warping/assets/58447982/d3e9c7d0-f650-41b7-8ded-17297300240c) | 


