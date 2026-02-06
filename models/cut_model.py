import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
import torch
import torch.nn as nn
import torch.nn.functional as F


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        #self.loss_names += ['red'] #['red', 'color']
        #self.visual_names += ['edge_gen', 'edge_gt']

        ## set default loss weights
        for name in self.loss_names:
            setattr(self, 'lambda_' + name, 1.0)

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']


        self.opt.num_classes = 12 # set number of classes for segmentation mask
        # define networks (both generator and discriminator)
        # opt.input_nc = 3
        # opt.output_nc = 2
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        self.netF_masked = networks.define_F(opt.input_nc, "masked_sample", opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(3+self.opt.num_classes, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)


            # opt.color_num_bins = 16
            # opt.color_emb_dim = 32
            # self.color_embedder = nn.Sequential( ## Add this to optimizer too
            #     nn.Linear(opt.color_num_bins * 3, 128),
            #     nn.ReLU(),
            #     nn.Linear(128, opt.color_emb_dim)
            # )

            # self.criterionColor = ColorLoss(
            #     embedder=self.color_embedder,
            #     patch_size=32,
            #     num_bins=opt.color_num_bins,
            #     emb_dim=opt.color_emb_dim
            # ).to(self.device)

            self.criterionEdge = EdgeLoss(alpha=0.99).to(self.device)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionRed = torch.nn.MSELoss().to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            # self.optimizer_C = torch.optim.Adam(self.color_embedder.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # self.optimizers.append(self.optimizer_C)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF_masked.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        # self.optimizer_C.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        # self.optimizer_C.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_mask = input['A_mask' if AtoB else 'B_mask'].to(self.device)
        self.real_B_mask = input['B_mask' if AtoB else 'A_mask'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.real_A_mask_onehot = F.one_hot(self.real_A_mask.long(), num_classes=self.opt.num_classes).permute(0, 3, 1, 2).float()
        self.real_B_mask_onehot = F.one_hot(self.real_B_mask.long(), num_classes=self.opt.num_classes).permute(0, 3, 1, 2).float()

    def set_loss_weights(self, lambdas):
        for name, value in lambdas.items():
            if name in self.loss_names:
                setattr(self, 'lambda_' + name, value)

        

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        self.real_mask_onehot = torch.cat((self.real_A_mask_onehot, self.real_B_mask_onehot), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_mask_onehot
        #self.real = torch.cat([self.real, self.real_mask_onehot], dim=1)
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])
                self.real_mask_onehot = torch.flip(self.real_mask_onehot, [3])

        ## Used when generating two-channel output
        # self.fake_green_blue = self.netG(self.real)
        # self.fake = torch.cat([self.real[:,0:1,:,:], self.fake_green_blue], dim=1)   

        self.fake = self.netG(self.real, self.real_mask_onehot)

        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(torch.cat([fake, self.real_A_mask_onehot], dim=1))
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(torch.cat([self.real_B, self.real_B_mask_onehot], dim=1))
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(torch.cat([fake, self.real_A_mask_onehot], dim=1))
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE_masked, self.loss_NCE = self.calculate_masked_NCE_loss(self.real_A, self.fake_B, mask=self.real_A_mask, mask_onehot=self.real_A_mask_onehot)
        else:
            self.loss_NCE_masked, self.loss_NCE = 0.0, 0.0
        # print("loss_NCE_masked:", self.loss_NCE_masked, "loss_NCE:", self.loss_NCE)
        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y_masked, self.loss_NCE_Y = self.calculate_masked_NCE_loss(self.real_B, self.idt_B, mask=self.real_B_mask, mask_onehot=self.real_B_mask_onehot)
            # print("loss_NCE_Y_masked:", self.loss_NCE_Y_masked, "loss_NCE_Y:", self.loss_NCE_Y)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.25 + (self.loss_NCE_masked + self.loss_NCE_Y_masked) * 0.25
        else:
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y_masked) * 0.5

        #self.loss_red = self.criterionRed(self.fake[:, 0:1, :, :], self.real[:, 0:1, :, :]) * 0.1


        #self.loss_color = (self.criterionColor(self.fake_B, self.real_A_mask) + self.criterionColor(self.idt_B, self.real_B_mask) if self.opt.nce_idt else 0.0) * 1.0
        #self.loss_edge, self.edge_gen, self.edge_gt = self.criterionEdge(self.real_A, self.real_A_mask, self.fake_B)
                                                

        self.loss_G = self.loss_G_GAN + loss_NCE_both #+ self.loss_red #+ self.loss_edge * self.lambda_edge
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def calculate_masked_NCE_loss(self, src, tgt, mask=None, mask_onehot=None):
        feat_q = self.netG(tgt, mask_onehot, self.nce_layers, encode_only=True)
        
        resized_masks = []
        for f_q in feat_q:
            _, _, h, w = f_q.shape
            resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=(h, w), mode='nearest')
            resized_masks.append(resized_mask)
        
        feat_k = self.netG(src, mask_onehot, self.nce_layers, encode_only=True)


        total_nce_loss = 0.0
        classes = torch.unique(mask) 
        for c in classes:
            # if c == 0:
            #     continue  # skip background
            total_nce_loss += self.calculate_single_NCE_loss(feat_q, feat_k, resized_masks, c.item())

        avg_nce_loss = total_nce_loss / len(classes) # (len(classes) - 1) if len(classes) > 1 else total_nce_loss

        full_nce_loss = self.calculate_single_NCE_loss(feat_q, feat_k, resized_masks, class_idx=None)


        return avg_nce_loss, full_nce_loss


    def calculate_single_NCE_loss(self, feat_q, feat_k, reshaped_mask, class_idx=None):
        """Calculate NCE loss between src and tgt"""

        n_layers = len(self.nce_layers)
        all_masks = []

        if class_idx is not None:
            for layer_idx in range(len(feat_q)):
                mask_c = (reshaped_mask[layer_idx] == class_idx).float()
                all_masks.append(mask_c.squeeze(1))


        
        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k_pool, sample_ids = self.netF_masked(feats=feat_k, num_patches=self.opt.num_patches, 
                                                   patch_ids=None, masks=all_masks if class_idx is not None else None)
        feat_q_pool, _ = self.netF_masked(feats=feat_q, num_patches=self.opt.num_patches, 
                                          patch_ids=sample_ids, masks=all_masks if class_idx is not None else None)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()
        # print("total_nce_loss single", class_idx, ":", total_nce_loss, "over", n_layers, "layers")
        return total_nce_loss / n_layers


class ColorLoss(nn.Module):
    def __init__(
        self,
        embedder,
        patch_size=16,
        num_bins=16,
        sigma=0.05,
        purity_thresh=0.5,
        emb_dim=64
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_bins = num_bins
        self.sigma = sigma
        self.purity_thresh = purity_thresh

        in_dim = 3 * num_bins
        self.embedder = embedder

        # Memory bank - larger size for more diversity
        self.alpha = 0.5
        self.margin = 0.5


    # --------------------------------------------------

    def extract_patches(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        return patches.reshape(B, -1, C, p, p)

    def extract_mask_patches(self, mask):
        B, H, W = mask.shape
        p = self.patch_size
        patches = mask.unfold(1, p, p).unfold(2, p, p)
        return patches.reshape(B, -1, p, p)

    # --------------------------------------------------

    def soft_histogram(self, x):
        """
        x: (N,) values in [0,1]
        """
        centers = torch.linspace(0, 1, self.num_bins, device=x.device)
        diff = x[:, None] - centers[None, :]
        w = torch.exp(-0.5 * (diff / self.sigma) ** 2)
        hist = w.sum(dim=0)
        return hist / (hist.sum() + 1e-6)

    # --------------------------------------------------

    def forward_old(self, img, mask):
        """
        img:  (B, 3, H, W)
        mask: (B, H, W)
        """
        img_patches = self.extract_patches(img)
        mask_patches = self.extract_mask_patches(mask)

        B, N, _, _, _ = img_patches.shape

        all_embeddings = []
        all_labels = []

        for b in range(B):
            for n in range(N):
                patch = img_patches[b, n]
                patch_mask = mask_patches[b, n]

                # dominant class + purity
                flat = patch_mask.view(-1)
                dominant_class = flat.mode().values.item()
                purity = (flat == dominant_class).float().mean()

                if purity < self.purity_thresh:
                    continue

                # masked histogram
                mask_c = (patch_mask == dominant_class)
                if mask_c.sum() < 10:
                    continue

                hists = []
                for c in range(3):
                    pixels = patch[c][mask_c]
                    hists.append(self.soft_histogram(pixels))

                hist = torch.cat(hists, dim=0)  # (3*num_bins)
                emb = self.embedder(hist.unsqueeze(0)).squeeze(0)

                all_embeddings.append(emb)
                all_labels.append(dominant_class)

        if len(all_embeddings) < 2:
            return torch.tensor(0.0, device=img.device, requires_grad=True)

        embeddings = torch.stack(all_embeddings)
        labels = torch.tensor(all_labels, device=img.device, dtype=torch.long)

        unique_labels, counts = labels.unique(return_counts=True)
        print(f"Unique classes: {len(unique_labels)}, Counts: {counts}")

        return info_nce_loss(embeddings, labels)
    
    def forward(self, img, mask):
        """
        img:  (B, 3, H, W)
        mask: (B, H, W)
        """
        img_patches = self.extract_patches(img)
        mask_patches = self.extract_mask_patches(mask)

        B, N, _, p, _ = img_patches.shape

        img_patches = img_patches.view(B*N, 3, p, p)
        mask_patches = mask_patches.view(B*N, p, p)

        flat = mask_patches.view(B*N, -1) # (BN, p*p)

        num_classes = int(flat.max()) + 1
        counts = torch.zeros(B*N, num_classes, device=flat.device)


        counts.scatter_add_(
            1,
            flat,
            torch.ones_like(flat, dtype=torch.float)
            )


        dominant = counts.argmax(dim=1) # (BN,)
        purity = counts.max(dim=1).values / flat.shape[1] # (BN,)
        valid = purity >= self.purity_thresh

        mask_c = mask_patches == dominant[:, None, None] # (BN, p, p)
        # valid &= mask_c.view(B*N, -1).sum(dim=1) >= 10

        centers = torch.linspace(0, 1, self.num_bins, device=img.device)

        hists = []
        for c in range(3):
            # Shape: (BN, p, p)
            channel_patches = img_patches[:, c]
            
            # Apply mask and get variable-length pixels per patch
            # We need to handle this carefully since each patch has different valid pixel counts
            
            # Expand to (BN, p*p) and mask
            pixels_flat = channel_patches.view(B*N, -1)  # (BN, p*p)
            mask_flat = mask_c.view(B*N, -1)  # (BN, p*p)
            
            # Set invalid pixels to 0 (won't contribute due to masking in histogram)
            pixels_masked = pixels_flat * mask_flat
            
            # Compute histogram with proper masking
            diff = pixels_masked[..., None] - centers  # (BN, p*p, num_bins)
            w = torch.exp(-0.5 * (diff / self.sigma) ** 2)
            
            # Only sum over valid pixels
            w = w * mask_flat[..., None]  # Zero out invalid positions
            
            hist = w.sum(dim=1)  # (BN, num_bins)
            hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-6)
            hists.append(hist)

        hist = torch.cat(hists, dim=1)  # (BN, 3*num_bins)
        hist = hist[valid]
        labels = dominant[valid]

        num_valid = hist.shape[0]

        # # In forward_new, before returning loss:
        # unique_labels, counts = labels.unique(return_counts=True)
        # print(f"Valid patches: {num_valid}, Unique classes: {len(unique_labels)}, Counts: {counts}")

        loss = 0
        count = 0
        
        # For each class, enforce histogram consistency
        for c in labels.unique():
            mask_c = labels == c
            if mask_c.sum() < 2:
                continue
            
            # Get all histograms for this class
            hists_c = hist[mask_c]  # (N_c, num_bins*3)
            
            # Mean histogram for this class
            mean_hist = hists_c.mean(dim=0, keepdim=True)
            
            # Push all toward mean (reduce intra-class variance)
            loss += F.mse_loss(hists_c, mean_hist.expand_as(hists_c))
            count += 1
        
        return loss / max(count, 1)


    
def info_nce_loss(embeddings, labels, temperature=0.1):
    """
    embeddings: (M, D)
    labels:     (M,)
    """
    embeddings = F.normalize(embeddings, dim=1)
    sim = embeddings @ embeddings.t() / temperature  # (M, M)

    labels = labels.unsqueeze(1)
    mask_pos = labels.eq(labels.t()).float()
    
    # Exclude diagonal from positive pairs
    eye = torch.eye(len(labels), device=labels.device)
    mask_pos = mask_pos * (1 - eye)
    
    # ============ CRITICAL FIX ============
    # Only keep samples that have at least one positive pair
    has_positive = mask_pos.sum(dim=1) > 0
    
    if has_positive.sum() < 2:
        # Not enough samples with positives
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
    
    # Filter to only samples with positives
    embeddings = embeddings[has_positive]
    labels = labels[has_positive]
    sim = sim[has_positive][:, has_positive]
    mask_pos = mask_pos[has_positive][:, has_positive]
    eye = torch.eye(len(embeddings), device=embeddings.device)
    # ======================================
    
    # Negative mask (everything except positives and diagonal)
    mask_neg = 1 - labels.eq(labels.t()).float()
    
    # For numerical stability
    logits_max = torch.max(sim * (mask_neg + eye), dim=1, keepdim=True)[0]
    logits = sim - logits_max.detach()
    
    # Compute log prob
    exp_logits = torch.exp(logits) * (1 - eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    
    # Average over positive pairs
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)
    
    return -mean_log_prob_pos.mean()




class EdgeLoss(nn.Module):
    def __init__(self, alpha=0.99):
        super(EdgeLoss, self).__init__()
        kx = torch.tensor([[ 3., 0., -3.],
                            [10., 0.,-10.],
                            [ 3., 0., -3.]], dtype=torch.float32)
        ky = torch.tensor([[ 3., 10., 3.],
                            [ 0., 0., 0.],
                            [-3.,-10., -3.]], dtype=torch.float32)

        self.register_buffer('filter_x', kx.view(1,1,3,3))
        self.register_buffer('filter_y', ky.view(1,1,3,3))

        self.alpha = alpha
    
    def edge_from_red(self, R):
        gx = F.conv2d(R, self.filter_x, padding=1)
        gy = F.conv2d(R, self.filter_y, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)
    
    def edge_from_segmentation(self, seg):
        if seg.dim() == 3:
            seg = seg.unsqueeze(1)  # (B,1,H,W)
        seg = seg.float()
        dx = torch.abs(seg[:, :, :, 1:] - seg[:, :, :, :-1])
        dy = torch.abs(seg[:, :, 1:, :] - seg[:, :, :-1, :])
        edge = torch.zeros_like(seg)
        edge[:, :, :, 1:] += dx
        edge[:, :, 1:, :] += dy
        edge = (edge > 0).float()
        return edge
    
    def combine_edges(self, E_seg, E_red, alpha=0.999):
        return alpha * E_seg + (1 - alpha) * E_red
    
    def color_edge_from_rgb(self, rgb):
        R = rgb[:, 0:1]
        G = rgb[:, 1:2]
        B = rgb[:, 2:3]
        C1 = G - R
        C2 = B - R
        gx1 = F.conv2d(C1, self.filter_x, padding=1)
        gy1 = F.conv2d(C1, self.filter_y, padding=1)
        gx2 = F.conv2d(C2, self.filter_x, padding=1)
        gy2 = F.conv2d(C2, self.filter_y, padding=1)
        edge = torch.sqrt(gx1**2 + gy1**2 + gx2**2 + gy2**2 + 1e-6)
        return edge

    def forward(self, x, seg, gen):

        E_red = self.edge_from_red(x[:, 0:1])
        E_seg = self.edge_from_segmentation(seg)  
        E_gt = self.combine_edges(E_seg, E_red, alpha=self.alpha)

        E_gen = self.color_edge_from_rgb(gen)      

        #return torch.mean(E_gt * (1.0 - torch.tanh(E_gen))) # F.l1_loss(E_gen, E_gt)
    
        loss_pos = torch.mean(E_gt * torch.abs(E_gen - E_gt))
        loss_neg = torch.mean((1.0 - E_gt) * E_gen)

        return loss_pos + 0.1 * loss_neg, E_gen, E_gt



