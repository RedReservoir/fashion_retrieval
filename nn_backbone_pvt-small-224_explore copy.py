from PIL import Image
import requests

import torch
from torchvision import transforms

from transformers import PvtModel, AutoImageProcessor


if __name__ == "__main__":

    # Create pre-trained model

    model = PvtModel.from_pretrained("Zetatech/pvt-small-224")

    # Load image (any other image can be used)

    img_url = "https://images.pexels.com/photos/949670/pexels-photo-949670.jpeg"
    img_raw = Image.open(requests.get(img_url, stream=True).raw)
    img_original = transforms.ToTensor()(img_raw)

    ##
    ## Evaluate with image size 3 x 224 x 224
    ##

    # Create image pre-processor
    
    ctsrbm_image_transform = AutoImageProcessor.from_pretrained("Zetatech/pvt-small-224")
    ctsrbm_image_transform_corr = lambda t: torch.from_numpy(ctsrbm_image_transform(t).pixel_values[0])

    # Pre-process image

    img_preprocessed = ctsrbm_image_transform_corr(img_original)[None, :]
    print("standard resolution input shape: ", img_preprocessed.shape)

    # PVT forward pass

    output = model(img_preprocessed).last_hidden_state
    print("standard resolution output shape:", output.shape)

    ##
    ## Evaluate with image size 3 x 448 x 448
    ##

    # Create image pre-processor
    
    ctsrbm_image_transform = AutoImageProcessor.from_pretrained("Zetatech/pvt-small-224")
    ctsrbm_image_transform.size["height"] = 448
    ctsrbm_image_transform.size["width"] = 448
    ctsrbm_image_transform_corr = lambda t: torch.from_numpy(ctsrbm_image_transform(t).pixel_values[0])

    # Pre-process image

    img_preprocessed = ctsrbm_image_transform_corr(img_original)[None, :]
    print("higher resolution input shape:   ", img_preprocessed.shape)

    # PVT forward pass

    output = model(img_preprocessed).last_hidden_state
    print("higher resolution output shape:  ", output.shape)
    