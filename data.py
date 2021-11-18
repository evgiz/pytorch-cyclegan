
import skimage.io
import numpy as np
import torchvision


def _local_img_loader(path):
    image = skimage.io.imread(path, as_gray=False)

    # Convert from RGBA to RGB
    if image.shape[2] == 4:
        image = skimage.color.rgba2rgb(image)

    # Byte to float
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0

    return image.astype(np.float32)


def get_local(folder, transform):
    return torchvision.datasets.ImageFolder(
        root=f"./data/{folder}/",
        loader=_local_img_loader,
        transform=transform
    )