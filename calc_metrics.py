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

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
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

def calculate_ssim(img1, img2):
    """Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (torch.Tensor): The first image tensor of shape (C, H, W).
        img2 (torch.Tensor): The second image tensor of shape (C, H, W).

    Returns:
        float: The SSIM value between the two images.
    """
    # Ensure the input images are in the range [0, 1]
    img1 = (img1 + 1) / 2  # Assuming input is in the range [-1, 1]
    img2 = (img2 + 1) / 2


    # Calculate SSIM
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
    # Ensure the input images are in the range [0, 1]
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



if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    #opt.num_test = float("inf") ## some max value... we need all for evaluations
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    opt.results_dir = "./evaluation_results/"
    os.makedirs(os.path.join(opt.results_dir, opt.name + '_results', 'test'), exist_ok=True)

    metrics_vals = defaultdict(list)
    LPIPS_metric = LPIPSM(net='alex')

    for i, data in enumerate(tqdm(dataset)):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            model.parallelize()
            if opt.eval:
                model.eval()
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        ssim_value, ms_ssim_value = calculate_ssim(visuals['real_B'], visuals['fake_B'])
        psnr_value = calculate_psnr(visuals['real_B'], visuals['fake_B'])
        lpips_value = LPIPS_metric(visuals['real_B'], visuals['fake_B'])
        metrics_vals['SSIM'].append(ssim_value)
        metrics_vals['MS-SSIM'].append(ms_ssim_value)
        metrics_vals['PSNR'].append(psnr_value)
        metrics_vals['LPIPS'].append(lpips_value)

        img_path = model.get_image_paths()     # get image paths
        result = torch.cat([v[0] for v in visuals.values()], 2)
        path = os.path.join(opt.results_dir, opt.name + '_results', 'test',
                        str(i + 1) + '.png')
        plt.imsave(path, (result.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

    metrics_dict = {metric_name: sum(metric_values) / len(metric_values) for metric_name, metric_values in metrics_vals.items()}

    fidelity_metrics_dict = torch_fidelity.calculate_metrics(
        input1=os.path.join(opt.results_dir, opt.name + '_results', 'test'), 
        input2=os.path.join(opt.dataroot, opt.phase + 'B'), 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=True, 
        prc=True, 
        verbose=False,
    )

    metrics_dict.update(fidelity_metrics_dict)

    with open(os.path.join(opt.results_dir, opt.name + '_results', 'metrics_test.txt'), 'w') as f:
        for metric_name, metric_value in metrics_dict.items():
            f.write(f"{metric_name}: {metric_value}\n")
