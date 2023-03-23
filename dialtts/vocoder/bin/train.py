#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Train Model."""

import os
import sys
import argparse
import logging

import matplotlib
import numpy as np
import soundfile as sf
import yaml
import torch
import torch.nn.functional as F

from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import dialtts.vocoder
import dialtts.vocoder.models
import dialtts.vocoder.optimizers
import dialtts.vocoder.lr_scheduler

from dialtts.vocoder.datasets import AudioMelSCPDataset
from dialtts.vocoder.layers import MultiResolutionSTFTLoss
from dialtts.vocoder.layers import PQMF

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel GAN Vocoder training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        pqmf,
        config,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" models.
            criterion (dict): Dict of criterions. It must contrain "ce" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "generator" optimizer.
            scheduler (dict): Dict of schedulers. It must contrain "generator" scheduler.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pqmf = pqmf
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        
        self.multiband = self.config["generator_params"]["out_channels"] > 1

    def run(self):
        """Run training."""
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logging.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        if self.config["distributed"]:
            state_dict["model"] = {
                "generator": self.model["generator"].module.state_dict(),
                "discriminator": self.model["discriminator"].module.state_dict(),
            }
        else:
            state_dict["model"] = {
                "generator": self.model["generator"].state_dict(),
                "discriminator": self.model["discriminator"].state_dict(),
            }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model["generator"].module.load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].module.load_state_dict(state_dict["model"]["discriminator"])
        else:
            self.model["generator"].load_state_dict(state_dict["model"]["generator"])
            self.model["discriminator"].load_state_dict(state_dict["model"]["discriminator"])
        
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(state_dict["optimizer"]["generator"])
            self.optimizer["discriminator"].load_state_dict(state_dict["optimizer"]["discriminator"])
            self.scheduler["generator"].load_state_dict(state_dict["scheduler"]["generator"])
            self.scheduler["discriminator"].load_state_dict(state_dict["scheduler"]["discriminator"])

    def _train_step(self, batch):
        """Train model one step."""
        
        # parse batch
        x = batch[0].to(self.device)
        y = batch[1].to(self.device)
        
        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](x, self.steps)
        
        if self.multiband:
            y_mb_ = y_
            y_ = self.pqmf.synthesis(y_mb_)
        
        # total loss of generator
        gen_loss = 0
        
        # multi-resolution stft loss
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        self.total_train_loss["train/spectral_convergence_loss"] += sc_loss.item()
        self.total_train_loss["train/log_stft_magnitude_loss"] += mag_loss.item()
        gen_loss += self.config["lambda_stft"] * (sc_loss + mag_loss)
        
        # sub-band multi-resolution stft loss
        if self.config["use_suband_stft_loss"] and self.multiband:
            y_mb = self.pqmf.analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2)) # (B, C, T)-> (B*C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2)) # (B, C, T)-> (B*C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["stft"](y_mb_, y_mb)
            self.total_train_loss["train/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            self.total_train_loss["train/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            gen_loss += self.config["lambda_stft"] * (sub_sc_loss + sub_mag_loss)
        
        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_)
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](p_[i], p_[i].new_ones(p_[i].size()))
            adv_loss /= float(i + 1)
            self.total_train_loss["train/adversarial_loss"] += adv_loss.item()
            lambda_adv = min((self.steps - self.config["discriminator_train_start_steps"]) / 100000.0, 1.0) * self.config["lambda_adv"]
            gen_loss += lambda_adv * adv_loss
            
        # total loss
        self.total_train_loss["train/generator_loss"] += gen_loss.item()

        # update generator
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        if self.config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config["generator_grad_norm"])
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()
        
        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # for standard discriminator
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_.detach())
            
            # multi-discriminator loss
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](p[i], p[i].new_ones(p[i].size()))
                fake_loss += self.criterion["mse"](p_[i], p_[i].new_zeros(p_[i].size()))
            real_loss /= float(i + 1)
            fake_loss /= float(i + 1)
            dis_loss = real_loss + fake_loss

            self.total_train_loss["train/real_loss"] += real_loss.item()
            self.total_train_loss["train/fake_loss"] += fake_loss.item()
            self.total_train_loss["train/discriminator_loss"] += dis_loss.item()
            
            # update discriminator
            self.optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if self.config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config["discriminator_grad_norm"])
            self.optimizer["discriminator"].step()
            self.scheduler["discriminator"].step()
        
        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            if self.config["rank"] == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logging.info(f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
                     f"({self.train_steps_per_epoch} steps per epoch).")

        # record learning rate
        if self.config["rank"] == 0:
            lr_per_epoch = defaultdict(float)
            for key in self.scheduler:
                lr_per_epoch[f"learning_rate/{key}"] = self.scheduler[key].get_last_lr()[0]
            self._write_to_tensorboard(lr_per_epoch)

        # needed for shuffle in distributed training
        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        
        # parse batch
        x = batch[0].to(self.device)
        y = batch[1].to(self.device)
        
        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](x)
        
        if self.multiband:
            y_mb_ = y_
            y_ = self.pqmf.synthesis(y_mb_)
        
        # total loss of generator
        gen_loss = 0
        
        # multi-resolution stft loss
        sc_loss, mag_loss = self.criterion["stft"](y_.squeeze(1), y.squeeze(1))
        self.total_eval_loss["eval/spectral_convergence_loss"] += sc_loss.item()
        self.total_eval_loss["eval/log_stft_magnitude_loss"] += mag_loss.item()
        gen_loss += self.config["lambda_stft"] * (sc_loss + mag_loss)
        
        # sub-band multi-resolution stft loss
        if self.config["use_suband_stft_loss"] and self.multiband:
            y_mb = self.pqmf.analysis(y)
            y_mb = y_mb.view(-1, y_mb.size(2)) # (B, C, T)-> (B*C, T)
            y_mb_ = y_mb_.view(-1, y_mb_.size(2)) # (B, C, T)-> (B*C, T)
            sub_sc_loss, sub_mag_loss = self.criterion["stft"](y_mb_, y_mb)
            self.total_eval_loss["eval/sub_spectral_convergence_loss"] += sub_sc_loss.item()
            self.total_eval_loss["eval/sub_log_stft_magnitude_loss"] += sub_mag_loss.item()
            gen_loss += self.config["lambda_stft"] * (sub_sc_loss + sub_mag_loss)
        
        # adversarial loss
        if self.steps > self.config["discriminator_train_start_steps"]:
            p_ = self.model["discriminator"](y_, y_mags_)
            adv_loss = 0.0
            for i in range(len(p_)):
                adv_loss += self.criterion["mse"](p_[i], p_[i].new_ones(p_[i].size()))
            adv_loss /= float(i + 1)
            self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
            lambda_adv = min((self.steps - self.config["discriminator_train_start_steps"]) / 100000.0, 1.0) * self.config["lambda_adv"]
            gen_loss += lambda_adv * adv_loss
            
        # total loss
        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()
        
        #######################
        #    Discriminator    #
        #######################
        if self.steps > self.config["discriminator_train_start_steps"]:
            # for standard discriminator
            p = self.model["discriminator"](y)
            p_ = self.model["discriminator"](y_)
            
            # multi-discriminator loss
            real_loss = 0.0
            fake_loss = 0.0
            for i in range(len(p)):
                real_loss += self.criterion["mse"](p[i], p[i].new_ones(p[i].size()))
                fake_loss += self.criterion["mse"](p_[i], p_[i].new_zeros(p_[i].size()))
            real_loss /= float(i + 1)
            fake_loss /= float(i + 1)
            dis_loss = real_loss + fake_loss

            self.total_eval_loss["eval/real_loss"] += real_loss.item()
            self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
            self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()
        
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logging.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.data_loader["dev"], desc="[eval]"), 1):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logging.info(f"(Steps: {self.steps}) Finished evaluation "
                     f"({eval_steps_per_epoch} steps per epoch).")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logging.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")
        
        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt
        
        def _demphasis(y, alpha=self.config.get("alpha", 0)):
            if alpha <= 0: return y
            y_ = np.zeros_like(y)
            for i in range(1, len(y)):
                y_[i] = y[i-1] + alpha * y_[i-1]
            return y_

        # generate
        x = batch[0].to(self.device)
        y = batch[1]
        y_ = self.model["generator"](x)
        
        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (r, g) in enumerate(zip(y, y_), 1):
            # convert to ndarray
            r = r.view(-1).cpu().numpy().flatten() # groundtruth
            g = g.view(-1).cpu().numpy().flatten() # generated
            
            # de-emphasis
            r = _demphasis(r, self.config.get("alpha", 0))
            g = _demphasis(g, self.config.get("alpha", 0))
            
            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(2, 1, 1)
            plt.plot(r)
            plt.title("groundtruth speech")
            plt.subplot(2, 1, 2)
            plt.plot(g)
            plt.title(f"generated speech @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            r = np.clip(r, -1, 1)
            g = np.clip(g, -1, 1)
            sf.write(figname.replace(".png", "_ref.wav"), r,
                     self.config["sampling_rate"], "PCM_16")
            sf.write(figname.replace(".png", "_gen.wav"), g,
                     self.config["sampling_rate"], "PCM_16")

            if idx >= self.config["num_save_intermediate_results"]:
                break

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint(
                os.path.join(self.config["outdir"], f"checkpoint-{self.steps}steps.pkl"))
            logging.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config["eval_interval_steps"] == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config["log_interval_steps"] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config["log_interval_steps"]
                logging.info(f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.")
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config["train_max_steps"]:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_steps=16000,
        hop_size=160,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of input signal in batch.
            hop_size (int): Hop size of each frame.

        """
        hop_size *= 2
        if batch_max_steps % hop_size != 0:
            batch_max_steps += -(batch_max_steps % hop_size)
        assert batch_max_steps % hop_size == 0 and batch_max_steps // hop_size > 1
        hop_size //= 2
        
        self.batch_max_steps = batch_max_steps
        self.batch_max_frames = batch_max_steps // hop_size
        self.hop_size = hop_size

        # set useful values in random cutting
        self.start_offset = 0
        self.end_offset = -self.batch_max_frames

    def __call__(self, batch):
        """Convert into batch tensors.
        """
        # check length
        batch = [self._adjust_length(*b) for b in batch]
        xs, cs = [b[0] for b in batch], [b[1] for b in batch]

        # make batch with random cut
        c_lengths = [len(c) for c in cs]
        start_frames = np.array([np.random.randint(
            self.start_offset, cl + self.end_offset) for cl in c_lengths])
        x_starts = start_frames * self.hop_size
        x_ends = x_starts + self.batch_max_steps
        c_starts = start_frames
        c_ends = start_frames + self.batch_max_frames
        batch_y = np.array([x[start:end] for x, start, end in zip(xs, x_starts, x_ends)])
        batch_c = np.array([c[start:end] for c, start, end in zip(cs, c_starts, c_ends)])
        
        # convert to tensor
        batch_y = torch.tensor(batch_y, dtype=torch.float).unsqueeze(1) # (B, 1, t)
        batch_c = torch.tensor(batch_c, dtype=torch.float).transpose(2, 1) # (B, C, T)
        
        return batch_c, batch_y

    def _adjust_length(self, x, c):
        # x: (t,)
        # c: (T, C)
        assert len(c) * self.hop_size == len(x), f"{len(c)} {len(x)}"
        while len(c) <= self.batch_max_frames:
            c =  np.concatenate((c,c), axis=0)
            x =  np.concatenate((x,x), axis=0)
        assert len(c) * self.hop_size == len(x)
        return (x, c)


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Neural Vocoder (See detail in dialtts/vocoder/bin/train.py).")
    parser.add_argument("--train-scp", type=str, required=True,
                        help="train.scp file for training. .")
    parser.add_argument("--dev-scp", type=str, required=True,
                        help="valid.scp file for validation. ")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--config", type=str, required=True,
                        help="yaml format configuration file.")
    parser.add_argument("--pretrain", default="", type=str, nargs="?",
                        help="checkpoint file path to load pretrained params. (default=\"\")")
    parser.add_argument("--resume", default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explictly specify.")
    args = parser.parse_args()

    args.distributed = False
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(args.rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # suppress logging for distributed training
    if args.rank != 0:
        sys.stdout = open(os.devnull, "w")

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = dialtts.vocoder.__version__   # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    dataset_params = {
        "mel_channels": config["num_mel"],
        "hop_size": config["hop_size"],
        "sampling_rate_threshold": config["sampling_rate"],
    }
    train_dataset = AudioMelSCPDataset(
        scpfn=args.train_scp,
        **dataset_params,
        mel_length_threshold=None
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = AudioMelSCPDataset(
        scpfn=args.dev_scp,
        **dataset_params,
        mel_length_threshold=config["batch_max_steps"] // config["hop_size"]
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater = Collater(
        batch_max_steps=config["batch_max_steps"],
        hop_size=config["hop_size"],
    )
    sampler = {"train": None, "dev": None}
    if args.distributed:
        # setup sampler for distributed training
        from torch.utils.data.distributed import DistributedSampler
        sampler["train"] = DistributedSampler(
            dataset=dataset["train"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )
        sampler["dev"] = DistributedSampler(
            dataset=dataset["dev"],
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=False,
        )
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            sampler=sampler["train"],
            pin_memory=config["pin_memory"],
        ),
        "dev": DataLoader(
            dataset=dataset["dev"],
            shuffle=False if args.distributed else True,
            collate_fn=collater,
            batch_size=config["batch_size"],
            num_workers=1,
            sampler=sampler["dev"],
            pin_memory=config["pin_memory"],
        ),
    }

    # define models
    generator_class = getattr(dialtts.vocoder.models, config["generator_type"])
    discriminator_class = getattr(dialtts.vocoder.models, config["discriminator_type"])
    model = {
        "generator": generator_class(**config["generator_params"]).to(device),
        "discriminator": discriminator_class(**config["discriminator_params"]).to(device),
    }
    logging.info(model["generator"])
    logging.info(model["discriminator"])
    
    # print parameters
    total_params, trainable_params, nontrainable_params = 0, 0, 0
    for param in model["generator"].parameters():
        num_params = np.prod(param.size())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            nontrainable_params += num_params
    logging.info(f"Total parameters of Generator: {total_params}")
    logging.info(f"Trainable parameters of Generator: {trainable_params}")
    logging.info(f"Non-trainable parameters of Generator: {nontrainable_params}\n")
    
    total_params, trainable_params, nontrainable_params = 0, 0, 0
    for param in model["discriminator"].parameters():
        num_params = np.prod(param.size())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            nontrainable_params += num_params
    logging.info(f"Total parameters of Discriminator: {total_params}")
    logging.info(f"Trainable parameters of Discriminator: {trainable_params}")
    logging.info(f"Non-trainable parameters of Discriminator: {nontrainable_params}\n")
    
    # define criterion and optimizers
    criterion = {
        "stft": MultiResolutionSTFTLoss(
            **config["stft_loss_params"]).to(device),
        "mae": torch.nn.L1Loss().to(device),
        "mse": torch.nn.MSELoss().to(device),
    }
    
    generator_optimizer_class = getattr(dialtts.vocoder.optimizers, config["generator_optimizer_type"])
    discriminator_optimizer_class = getattr(dialtts.vocoder.optimizers, config["discriminator_optimizer_type"])
    optimizer = {
        "generator": generator_optimizer_class(
            model["generator"].parameters(),
            **config["generator_optimizer_params"],
        ),
        "discriminator": discriminator_optimizer_class(
            model["discriminator"].parameters(),
            **config["discriminator_optimizer_params"],
        ),
    }
    generator_scheduler_class = getattr(dialtts.vocoder.lr_scheduler, config["generator_scheduler_type"])
    discriminator_scheduler_class = getattr(dialtts.vocoder.lr_scheduler, config["discriminator_scheduler_type"])
    scheduler = {
        "generator": generator_scheduler_class(
            optimizer=optimizer["generator"],
            **config["generator_scheduler_params"],
        ),
        "discriminator": discriminator_scheduler_class(
            optimizer=optimizer["discriminator"],
            **config["discriminator_scheduler_params"],
        ),
    }
    
    # define PQMF object
    pqmf = PQMF(config["num_band"], taps=config["taps"], beta=config["beta"]).to(device)
    
    if args.distributed:
        # wrap model for distributed training
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
        model["generator"] = DistributedDataParallel(model["generator"])
        model["discriminator"] = DistributedDataParallel(model["discriminator"])

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        pqmf=pqmf,
        config=config,
        device=device,
    )

    # load pretrained parameters from checkpoint
    if len(args.pretrain) != 0:
        trainer.load_checkpoint(args.pretrain, load_only_params=True)
        logging.info(f"Successfully load parameters from {args.pretrain}.")

    # resume from checkpoint
    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logging.info(f"Successfully resumed from {args.resume}.")
    
    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(config["outdir"], f"checkpoint-{trainer.steps}steps.pkl"))
        logging.info(f"Successfully saved checkpoint @ {trainer.steps} steps.")
        logging.info(f"KeyboardInterrupt @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
