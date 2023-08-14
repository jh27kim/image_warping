# Image warping & refinement
This repository provides 2D and 3D warping source codes. 

## Image warping

Supports $SE(2)$ and $SO(2)$ transformation warping and refinement using a U-Net network.

For 2D transformation task, use any available dataset. Here we used [AFHQ dataset](https://www.v7labs.com/open-datasets/afhq). 

|                  Input image          |           Warped image                |       Latent warp + refined image     |
|:-------------------------------------:|:-------------------------------------:|:-------------------------------------:|
| ![](https://github.com/jh27kim/image_warping/assets/58447982/7d82cca0-6c74-456b-ad15-cbde6e17cd67) | ![](https://github.com/jh27kim/image_warping/assets/58447982/8892419b-7477-4195-a2d2-ad291085f543) | ![](https://github.com/jh27kim/image_warping/assets/58447982/b4c71b68-9469-4cbe-a59f-6336aa96d5af) | 



## Depth warping

Supports depth warping given ground truth / predicted depth map.

Ground truth depth map is hard to acquire in practice. 
You can either extract depth from a mesh, or use 3D scanned depth map from publicly available dataset like [OmniObject3D](https://omniobject3d.github.io/).

|                  Input image          |          360 3D object synthesis     |
|:-------------------------------------:|:-------------------------------------:|
| ![](https://github.com/jh27kim/image_warping/assets/58447982/e7ac2356-1609-48c2-ad72-a88f2e954cef) | ![](https://github.com/jh27kim/image_warping/assets/58447982/51943cf7-7cfc-4e61-a8fc-20983c65499d) | 

![ezgif com-resize](https://github.com/jh27kim/image_warping/assets/58447982/e655da4b-48a0-4a87-a2b6-401393c86d5a)
