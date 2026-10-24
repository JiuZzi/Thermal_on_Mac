"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import random
import time
import numpy as np
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp
from util.training_progress import recovery_offset


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    if opt.stop_after_epoch is not None and not (opt.epoch_count <= opt.stop_after_epoch <= opt.n_epochs + opt.n_epochs_decay):
        raise ValueError('stop_after_epoch must be within the configured epoch range')
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Training random seed: {opt.seed}")
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")
    total_iters = opt.resume_total_iters
    resume_skip = recovery_offset(total_iters, dataset_size, opt.epoch_count, opt.batch_size,
                                  opt.continue_train,
                                  torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if total_iters:
        generator = getattr(model, "netG_A", None)
        generator = getattr(generator, "module", generator)
        counter = getattr(generator, "color_loss_updates", None)
        if counter is not None and int(counter) != total_iters:
            raise ValueError("Loaded checkpoint update counter differs from resume_total_iters")
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    if total_iters:
        print(f"Weight recovery: {total_iters} saved updates; epoch {opt.epoch_count}, "
              f"skip {resume_skip} batch positions. Optimizer, replay pool and RNG states were not saved.", flush=True)
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = resume_skip if epoch == opt.epoch_count else 0
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            if epoch == opt.epoch_count and i < resume_skip:
                if i % 1000 == 0:
                    print(f"Recovery: reading/skipping saved batch positions {i}/{resume_skip}", flush=True)
                continue
            if epoch == opt.epoch_count and resume_skip and i == resume_skip:
                print(f"Recovery: next optimizer update is {total_iters + 1}", flush=True)
            iter_start_time = time.time()  # timer for computation per iteration
            t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)
                if hasattr(model, "save_guidance_diagnostics"):
                    model.save_guidance_diagnostics(epoch, total_iters)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        stopping = opt.stop_after_epoch is not None and epoch == opt.stop_after_epoch
        if epoch % opt.save_epoch_freq == 0 or stopping:  # also preserve a complete checkpoint before early stopping
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")
        if stopping:
            print(f"Stopped after complete epoch {epoch}; original {opt.n_epochs}+{opt.n_epochs_decay} schedule preserved.")
            break

    cleanup_ddp()
