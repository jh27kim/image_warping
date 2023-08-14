# Image warping & refinement
This repository provides 2D and 3D warping source codes. 

## Image warping

Supports $SE(2)$ and $SO(2)$ transformation warping and refinement using a U-Net network.

For 2D transformation task, use any available dataset. Here we used [AFHQ dataset](https://www.v7labs.com/open-datasets/afhq). 

<p align="center">
  <img width='400' src=""/>
  <img width='400' src=""/>
</p>

## Depth warping

Supports depth warping given ground truth / predicted depth map.

Ground truth depth map is hard to acquire in practice. 
You can either extract depth from a mesh, or use 3D scanned depth map from publicly available dataset like [OmniObject3D](https://omniobject3d.github.io/).

<p align="center">
  <img width='400' src="https://github.com/jh27kim/image_warping/assets/58447982/e7ac2356-1609-48c2-ad72-a88f2e954cef"/>
  <img width='400' src="https://github.com/jh27kim/image_warping/assets/58447982/525695c8-18f5-4950-a9e8-0e8158ff698f"/>
</p>

<p align="center">
  <img width='400' src="https://github.com/jh27kim/image_warping/assets/58447982/51943cf7-7cfc-4e61-a8fc-20983c65499d"/>
</p>
