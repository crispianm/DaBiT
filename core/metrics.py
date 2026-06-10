import cv2
import numpy as np
from skimage.metrics import structural_similarity


def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).
    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
    Returns:
        float: psnr result.
    """

    assert (
        img1.shape == img2.shape
    ), f"Image shapes are different: {img1.shape}, {img2.shape}."

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def calc_psnr_and_ssim(img1, img2):
    """Calculate PSNR and SSIM for images.
    img1: ndarray, range [0, 255]
    img2: ndarray, range [0, 255]
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    psnr = calculate_psnr(img1, img2)
    ssim = structural_similarity(
        img1,
        img2,
        data_range=255,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )

    return psnr, ssim


def crop_8x8(img):
    """Centre-crop to a multiple of 32 (with a 16px margin), as used by FMA-Net's
    tOF computation. Returns the crop and its top-left offset."""
    ori_h, ori_w = img.shape[0], img.shape[1]

    h = (ori_h // 32) * 32
    w = (ori_w // 32) * 32
    while h > ori_h - 16:
        h -= 32
    while w > ori_w - 16:
        w -= 32

    y = (ori_h - h) // 2
    x = (ori_w - w) // 2
    return img[y : y + h, x : x + w], y, x


def calculate_tof(pre_comp, comp, pre_gt, gt):
    """Temporal optical-flow error (tOF), following FMA-Net
    (https://github.com/KAIST-VICLab/FMA-Net/blob/main/utils.py).

    Measures how closely the optical flow of the prediction between two
    consecutive frames matches the optical flow of the ground truth.
    Inputs are RGB frames with range [0, 255]; lower is better.
    """
    pre_comp_g = cv2.cvtColor(pre_comp.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    comp_g = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    pre_gt_g = cv2.cvtColor(pre_gt.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gt_g = cv2.cvtColor(gt.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    target_OF = cv2.calcOpticalFlowFarneback(
        pre_gt_g, gt_g, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    output_OF = cv2.calcOpticalFlowFarneback(
        pre_comp_g, comp_g, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )

    target_OF, _, _ = crop_8x8(target_OF)
    output_OF, _, _ = crop_8x8(output_OF)

    OF_diff = np.absolute(target_OF - output_OF)
    OF_diff = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1))  # per-pixel L2 norm
    return OF_diff.mean()
