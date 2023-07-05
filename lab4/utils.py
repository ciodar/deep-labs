import numpy as np
import PIL.Image as Image

NORM_MEAN = np.array([0.5, 0.5, 0.5])
NORM_STD = np.array([0.5, 0.5, 0.5])

def get_pil_image(nparray, normalized=True):
    npimg = nparray.transpose(1, 2, 0)
    if normalized:
        npimg = (npimg * NORM_STD[None,None]) + NORM_MEAN[None,None]
    npimg = np.clip(npimg, a_min=0.0, a_max=1.0)
    return Image.fromarray((npimg * 255).astype(np.uint8))