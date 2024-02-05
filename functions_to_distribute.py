import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def generate_random_depth_mask(depth):

    # Scale depth to [0, 1]
    depth = depth / np.max(depth)

    # Initialize mask
    mask = np.zeros((depth.shape[0], depth.shape[1]))

    # Define focal range
    window = random.uniform(0.2,0.4)
    focal_point = random.uniform(0,1)

    if focal_point >= 1-window:
        depth1 = 1-window
        depth2 = 1
    elif focal_point <= window:
        depth1 = 0
        depth2 = window
    else:
        depth1 = focal_point - window
        depth2 = focal_point + window

    # Infill depths outside of focal range
    mask[depth < depth1] = 1
    mask[depth > depth2] = 1

    return mask

if __name__ == "__main__":

    file = "depth.png"

    # Example usage
    depth = np.array(Image.open(file).convert("L"))
    mask = generate_random_depth_mask(depth)
    plt.imsave("mask.png", mask, cmap="gray")
