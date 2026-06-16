"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

    Evaluate precomputed predictions (skip inference):
        python test.py --precomputed_dir ./my_predictions --dataroot ./datasets/facades --name facades_pix2pix

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import argparse
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
import torch
import matplotlib.pyplot as plt
import torch_fidelity
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from lpips import LPIPS
from collections import defaultdict
from PIL import Image
import numpy as np


def calculate_ssim(img1, img2):
    """Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (torch.Tensor): The first image tensor of shape (C, H, W).
        img2 (torch.Tensor): The second image tensor of shape (C, H, W).

    Returns:
        float: The SSIM value between the two images.
    """
    img1 = (img1 + 1) / 2  # Assuming input is in the range [-1, 1]
    img2 = (img2 + 1) / 2
    ssim_value = ssim(img1, img2, data_range=1.0)
    ms_ssim_value = ms_ssim(img1, img2, data_range=1.0)
    return ssim_value.item(), ms_ssim_value.item()


def calculate_psnr(img1, img2):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (torch.Tensor): The first image tensor of shape (C, H, W).
        img2 (torch.Tensor): The second image tensor of shape (C, H, W).

    Returns:
        float: The PSNR value between the two images.
    """
    img1 = (img1 + 1) / 2  # Assuming input is in the range [-1, 1]
    img2 = (img2 + 1) / 2
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr_value.item()


class LPIPSM(torch.nn.Module):
    def __init__(self, net='alex'):
        super(LPIPSM, self).__init__()
        self.net = LPIPS(net=net).cuda()

    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2  # Assuming input is in the range [-1, 1]
        img2 = (img2 + 1) / 2
        lpips_value = self.net(img1.cuda(), img2.cuda())
        return lpips_value.item()


def load_image_as_tensor(path, crop_size=210, resize=256):
    """Load a PNG/JPG image from disk, center-crop then resize, return a [-1, 1] float tensor of shape (1, C, H, W).
    
    Args:
        path (str):       Path to the image file.
        crop_size (int):  Size of the center crop (square). Default: 256.
        resize (int):     Size to resize the cropped image to (square). Default: 256.
    """
    img = Image.open(path).convert('RGB')

    # # Center crop
    # w, h = img.size
    # left = (w - crop_size) // 2
    # top  = (h - crop_size) // 2
    # img  = img.crop((left, top, left + crop_size, top + crop_size))

    # # Resize
    # img = img.resize((resize, resize), Image.BICUBIC)

    arr = np.array(img).astype(np.float32) / 255.0  # [0, 1]
    arr = arr * 2 - 1                                # [-1, 1]
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # (1, C, H, W)


def run_precomputed(opt, pred_dir, gt_dir, out_dir, LPIPS_metric):
    """Evaluate predictions that are already saved to disk.

    Expects pred_dir to contain image files whose names match those in gt_dir.
    Images are assumed to be saved in [0, 1] PNG format (standard saves).
    """
    SUPPORTED = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    pred_files = sorted([
        f for f in os.listdir(pred_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ])

    if not pred_files:
        raise FileNotFoundError(f"No images found in precomputed_dir: {pred_dir}")

    metrics_vals = defaultdict(list)

    for fname in tqdm(pred_files, desc="Evaluating precomputed predictions"):
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            print(f"[WARNING] No matching ground-truth for {fname}, skipping.")
            continue

        pred = load_image_as_tensor(pred_path)
        gt   = load_image_as_tensor(gt_path)

        ssim_value, ms_ssim_value = calculate_ssim(gt, pred)
        psnr_value  = calculate_psnr(gt, pred)
        lpips_value = LPIPS_metric(gt, pred)

        metrics_vals['SSIM'].append(ssim_value)
        metrics_vals['MS-SSIM'].append(ms_ssim_value)
        metrics_vals['PSNR'].append(psnr_value)
        metrics_vals['LPIPS'].append(lpips_value)

    return metrics_vals, pred_dir   # fidelity metrics will use pred_dir directly


def run_model_inference(opt, out_dir, LPIPS_metric):
    """Run the model and save fake_B images; return per-image metrics and the output dir."""
    dataset = create_dataset(opt)
    model   = create_model(opt)
    metrics_vals = defaultdict(list)

    fake_dir = os.path.join(out_dir, 'fake_B')
    os.makedirs(fake_dir, exist_ok=True)

    for i, data in enumerate(tqdm(dataset, desc="Running model inference")):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)
            model.parallelize()
            if opt.eval:
                model.eval()

        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        ssim_value, ms_ssim_value = calculate_ssim(visuals['real_B'], visuals['fake_B'])
        psnr_value  = calculate_psnr(visuals['real_B'], visuals['fake_B'])
        lpips_value = LPIPS_metric(visuals['real_B'], visuals['fake_B'])

        metrics_vals['SSIM'].append(ssim_value)
        metrics_vals['MS-SSIM'].append(ms_ssim_value)
        metrics_vals['PSNR'].append(psnr_value)
        metrics_vals['LPIPS'].append(lpips_value)

        # Save side-by-side strip (real_A | fake_B | real_B)
        strip = torch.cat([v[0] for v in visuals.values()], 2)
        strip_path = os.path.join(out_dir, f'{i + 1}.png')
        plt.imsave(strip_path, (strip.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

        # Save fake_B alone for fidelity metrics
        fake = visuals['fake_B'][0]
        fake_path = os.path.join(fake_dir, f'{i + 1}.png')
        plt.imsave(os.path.join(opt.results_dir, name + '_results', 'pred', f'{i + 1}_pred.png'), (fake.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

        plt.imsave(os.path.join(opt.results_dir, name + '_results', 'gt', f'{i + 1}_gt.png'), (visuals['real_B'][0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

        plt.imsave(os.path.join(opt.results_dir, name + '_results', 'input', f'{i + 1}_inp.png'), (visuals['real_A'][0].cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

    return metrics_vals, os.path.join(opt.results_dir, name + '_results', 'pred')


if __name__ == '__main__':
    # -----------------------------------------------------------------------
    # Parse the extra --precomputed_dir flag before TestOptions sees argv,
    # so we can decide whether to instantiate the model at all.
    # -----------------------------------------------------------------------
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--precomputed_dir', type=str, default=None,
                            help='Path to a directory of already-generated predictions. '
                                 'If set, model inference is skipped and images are loaded '
                                 'directly from this folder for evaluation.')
    pre_parser.add_argument('--gt_dir', type=str, default=None,
                            help='Ground-truth image directory used when --precomputed_dir '
                                 'is set. Defaults to <dataroot>/testB if not provided.')
    pre_args, remaining_argv = pre_parser.parse_known_args()

    import sys
    sys.argv = [sys.argv[0]] + remaining_argv  # let TestOptions parse the rest

    opt = TestOptions().parse()
    opt.num_threads  = 0
    opt.batch_size   = 1
    opt.serial_batches = True
    opt.no_flip      = True
    opt.display_id   = -1
    opt.use_val_data = True

    opt.results_dir  = "./evaluation_results/"
    name             = opt.name if not pre_args.precomputed_dir else os.path.basename(pre_args.precomputed_dir.rstrip('/'))
    out_dir          = os.path.join(opt.results_dir, name + '_results', 'test')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.results_dir, name + '_results', 'pred'), exist_ok=True)
    os.makedirs(os.path.join(opt.results_dir, name + '_results', 'gt'), exist_ok=True)
    os.makedirs(os.path.join(opt.results_dir, name + '_results', 'input'), exist_ok=True)

    LPIPS_metric = LPIPSM(net='alex')

    # -----------------------------------------------------------------------
    # Branch: precomputed predictions  vs  live model inference
    # -----------------------------------------------------------------------
    if pre_args.precomputed_dir:
        gt_dir = pre_args.gt_dir or os.path.join(opt.dataroot, 'testB')
        print(f"[INFO] Evaluating precomputed predictions from: {pre_args.precomputed_dir}")
        print(f"[INFO] Ground-truth directory:                  {gt_dir}")
        metrics_vals, fid_input_dir = run_precomputed(opt, pre_args.precomputed_dir, gt_dir, out_dir, LPIPS_metric)
        fid_gt_dir = gt_dir
    else:
        print(f"[INFO] Running model inference for: {opt.name}")
        metrics_vals, fid_input_dir = run_model_inference(opt, out_dir, LPIPS_metric)
        fid_gt_dir = "/hddstore/janmesh/nnUNet_raw/Dataset003_DresdenMultiOnly/imagesTs"

    # -----------------------------------------------------------------------
    # Aggregate pixel-level metrics
    # -----------------------------------------------------------------------
    metrics_dict = {
        metric_name: sum(vals) / len(vals)
        for metric_name, vals in metrics_vals.items()
    }

    # # -----------------------------------------------------------------------
    # # Fidelity metrics (FID, KID, IS, Precision/Recall)
    # # -----------------------------------------------------------------------
    # fidelity_metrics_dict = torch_fidelity.calculate_metrics(
    #     input1=fid_input_dir,
    #     input2=fid_gt_dir,
    #     cuda=True,
    #     isc=True,
    #     fid=True,
    #     kid=True,
    #     kid_subset_size=50,
    #     prc=True,
    #     verbose=False,
    # )
    # metrics_dict.update(fidelity_metrics_dict)

    # -----------------------------------------------------------------------
    # Write results
    # -----------------------------------------------------------------------
    metrics_path = os.path.join(opt.results_dir, name + '_results', 'metrics_test.txt')
    with open(metrics_path, 'w') as f:
        for metric_name, metric_value in metrics_dict.items():
            f.write(f"{metric_name}: {metric_value}\n")

    print(f"\n[INFO] Metrics saved to: {metrics_path}")
    for k, v in metrics_dict.items():
        print(f"  {k}: {v}")