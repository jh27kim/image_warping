from torch.utils.data import Dataset
import numpy.typing as npt
from PIL import Image
import json
import torch
import numpy as np

from typing import List, Dict
from jaxtyping import Float
from torch import Tensor
from pathlib import Path



class CustomImageDataset(Dataset):
    def __init__(self, metadata, transform=None, target_transform=None):
        self.metadata = metadata        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.metadata)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        
        image_filepath = self.metadata[str(image_idx)]["filepath"]
        pil_image = Image.open(image_filepath).resize((512, 512))
        
#         if self.scale_factor != 1.0:
#             width, height = pil_image.size
#             newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
#             pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
            
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
            
        assert image.shape[0] == image.shape[1] == 512
        
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        
        return image

    def get_image(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        image = image.permute(2, 0, 1)
        
        assert image.shape[0] == 3
        return image

    def get_data(self, image_idx: int) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx, "image": image}
#         data.update(metadata)
        
        return data

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        """
        Returns image filenames for this dataset.
        The order of filenames is the same as in the Cameras object for easy mapping.
        """

        return self.image_filenames