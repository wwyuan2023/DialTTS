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


import dialtts.acoustic
import dialtts.acoustic.models
import dialtts.acoustic.optimizers
import dialtts.acoustic.lr_scheduler

from dialtts.acoustic.datasets import TextMelSCPDataset

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


class Trainer(object):
    """Customized trainer module for Parallel Acoustic model training."""

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
        self.config = config
        self.device = device
        self.writer = SummaryWriter(config["outdir"])
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
    
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
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)
        
        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x, y, steps=self.steps)
        
        # parse inputs and outputs
        spk_id, text_target, duration_target = x
        mel_target = y
        mel_outputs, (duration_prediction, energy_target, energy_prediction), \
            (text_mask, mel_mask) = y_
        
        # total loss of generator
        gen_loss = 0
        
        # duration loss
        dur_loss = self.criterion["L1_mean"](
            duration_prediction.masked_select(text_mask),
            duration_target.masked_select(text_mask).log()
        )
        self.total_train_loss["train/dur_loss"] += dur_loss.item()
        gen_loss += dur_loss * self.config["lambda_dur"]
        
        # energy loss
        text_mask = text_mask.unsqueeze(1) # (B,1,N)
        energy_loss = self.criterion["L2_mean"](
            energy_prediction.masked_select(text_mask),
            energy_target.masked_select(text_mask)
        )
        self.total_train_loss["train/energy_loss"] += energy_loss.item()
        gen_loss += energy_loss * self.config["lambda_energy"]
        
        # mel loss
        B, C, T = mel_target.size()
        mel_mask = mel_mask.unsqueeze(-1).expand(-1,-1,C) # (B,T,C)
        mel_target = mel_target.transpose(1,2).masked_select(mel_mask).view(-1,C) # (L,C)
        for i in range(len(mel_outputs)):
            mel_outputs[i] = mel_outputs[i].transpose(1,2).masked_select(mel_mask).view(-1,C) # (L,C)
            mel_loss = self.criterion["L1_mean"](mel_outputs[i], mel_target)
            self.total_train_loss[f"train/mel_loss{i}"] += mel_loss.item()
            gen_loss += mel_loss
        
        # adversarial loss
        L = mel_target.size(0)
        mel_cutlen = L - L % B
        mel_output = None
        if self.steps > self.config["discriminator_train_start_steps"]:
            # select mask data
            mel_output = mel_outputs[-1][:mel_cutlen].view(B,-1,C).transpose(1,2) # (B,C,T)
            
            # for standard discriminator
            p_ = self.model["discriminator"](mel_output)
            
            # discriminator loss
            if not isinstance(p_, list): # one-discriminator
                adv_loss = self.criterion["L2_mean"](p_, p_.new_ones(p_.size()))
            else: # multi-discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.criterion["L2_mean"](p_[i], p_[i].new_ones(p_[i].size()))
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
            # select mask data
            mel_target = mel_target[:mel_cutlen].view(B,-1,C).transpose(1,2) # (B,C,T')
            if mel_output is None:
                mel_output = mel_outputs[-1][:mel_cutlen].view(B,-1,C).transpose(1,2) # (B,C,T')
            
            # for standard discriminator
            p = self.model["discriminator"](mel_target)
            p_ = self.model["discriminator"](mel_output.detach())
            
            # discriminator loss
            if not isinstance(p, list): # one-discriminator
                real_loss = self.criterion["L2_mean"](p, p.new_ones(p.size()))
                fake_loss = self.criterion["L2_mean"](p_, p_.new_zeros(p.size()))
            else: # multi-discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.criterion["L2_mean"](p[i], p[i].new_ones(p[i].size()))
                    fake_loss += self.criterion["L2_mean"](p_[i], p_[i].new_zeros(p_[i].size()))
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
        x, y = batch
        x = tuple([x_.to(self.device) for x_ in x])
        y = y.to(self.device)
        
        #######################
        #      Generator      #
        #######################
        y_ = self.model["generator"](*x, y, steps=self.steps)
        
        # parse inputs and outputs
        spk_id, text_target, duration_target = x
        mel_target = y
        mel_outputs, (duration_prediction, energy_target, energy_prediction), \
            (text_mask, mel_mask) = y_
        
        # total loss of generator
        gen_loss = 0
        
        # duration loss
        dur_loss = self.criterion["L1_mean"](
            duration_prediction.masked_select(text_mask),
            duration_target.masked_select(text_mask).log()
        )
        self.total_eval_loss["eval/dur_loss"] += dur_loss.item()
        gen_loss += dur_loss * self.config["lambda_dur"]
        
        # energy loss
        text_mask = text_mask.unsqueeze(1) # (B,1,N)
        energy_loss = self.criterion["L2_mean"](
            energy_prediction.masked_select(text_mask),
            energy_target.masked_select(text_mask)
        )
        self.total_eval_loss["eval/energy_loss"] += energy_loss.item()
        gen_loss += energy_loss * self.config["lambda_energy"]
        
        # mel loss
        B, C, T = mel_target.size()
        mel_mask = mel_mask.unsqueeze(-1).expand(-1,-1,C) # (B,T,C)
        mel_target = mel_target.transpose(1,2).masked_select(mel_mask).view(-1,C) # (L,C)
        for i in range(len(mel_outputs)):
            mel_outputs[i] = mel_outputs[i].transpose(1,2).masked_select(mel_mask).view(-1,C) # (L,C)
            mel_loss = self.criterion["L1_mean"](mel_outputs[i], mel_target)
            self.total_eval_loss[f"eval/mel_loss{i}"] += mel_loss.item()
            gen_loss += mel_loss
        
        # adversarial loss
        L = mel_target.size(0)
        mel_cutlen = L - L % B
        mel_output = None
        if self.steps > self.config["discriminator_train_start_steps"]:
            # select mask data
            mel_output = mel_outputs[-1][:mel_cutlen].view(B,-1,C).transpose(1,2) # (B,C,T)
            
            # for standard discriminator
            p_ = self.model["discriminator"](mel_output)
            
            # discriminator loss
            if not isinstance(p_, list): # one-discriminator
                adv_loss = self.criterion["L2_mean"](p_, p_.new_ones(p_.size()))
            else: # multi-discriminator
                adv_loss = 0.0
                for i in range(len(p_)):
                    adv_loss += self.criterion["L2_mean"](p_[i], p_[i].new_ones(p_[i].size()))
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
            # select mask data
            mel_target = mel_target[:mel_cutlen].view(B,-1,C).transpose(1,2) # (B,C,T')
            if mel_output is None:
                mel_output = mel_outputs[-1][:mel_cutlen].view(B,-1,C).transpose(1,2) # (B,C,T')
            
            # for standard discriminator
            p = self.model["discriminator"](mel_target)
            p_ = self.model["discriminator"](mel_output.detach())
            
            # discriminator loss
            if not isinstance(p, list): # one-discriminator
                real_loss = self.criterion["L2_mean"](p, p.new_ones(p.size()))
                fake_loss = self.criterion["L2_mean"](p_, p_.new_zeros(p.size()))
            else: # multi-discriminator
                real_loss = 0.0
                fake_loss = 0.0
                for i in range(len(p)):
                    real_loss += self.criterion["L2_mean"](p[i], p[i].new_ones(p[i].size()))
                    fake_loss += self.criterion["L2_mean"](p_[i], p_[i].new_zeros(p_[i].size()))
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
        
        # generate
        x_batch, y_batch = batch
        x_batch = tuple([x.to(self.device) for x in x_batch])
        y_batch = y_batch.to(self.device)
        y_batch_ = self.model["generator"](*x_batch, y_batch)
        
        mel_target = y_batch
        mel_outputs, _, _ = y_batch_
        
        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, ys in enumerate(zip(mel_target, *mel_outputs), 1):
            # convert to ndarray
            yy = [y.data.cpu().numpy() for y in ys]
            n = len(yy)
            
            # plot figure and save it
            figname = os.path.join(dirname, f"{idx}.png")
            plt.subplot(n, 1, 1)
            plt.pcolor(yy[0])
            plt.title("groundtruth melspectrogram")
            for i in range(1, 1+len(yy)):
                plt.subplot(n, 1, i+1)
                plt.pcolor(yy[i])
                plt.title(f"generated melspectrogram{i} @ {self.steps} steps")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

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
        text_channels,
        mel_channels,
        batch_max_steps=1000,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum frame in batch.

        """
        self.text_channels = text_channels
        self.mel_channels = mel_channels
        self.batch_max_steps = batch_max_steps

    def __call__(self, batch):
        """Convert into batch tensors.
        """
        B = len(batch)
        max_text_len = max([x[1].shape[1] for x in batch])
        max_mel_len = max([x[-1].shape[1] for x in batch])
        
        assert max_mel_len <= self.batch_max_steps
        
        # right zero-pad text/dur to `max_text_len`, and also mel to `max_mel_len`
        text_padded = np.zeros(shape=(B,self.text_channels,max_text_len), dtype=np.float32)
        dur_padded = np.zeros(shape=(B,max_text_len), dtype=np.int64)
        mel_padded = np.zeros(shape=(B,self.mel_channels,max_mel_len), dtype=np.float32)
        spk_id = np.zeros(shape=(B,), dtype=np.int64)
        for i, (spkid, text, dur, mel) in enumerate(batch):
            text_padded[i,:,:text.shape[1]] = text
            dur_padded[i,:dur.shape[0]] = dur
            mel_padded[i,:,:mel.shape[1]] = mel
            spk_id[i] = int(spkid)
        
        # convert to tensor
        spk_id = torch.tensor(spk_id, dtype=torch.long) # (B,)
        text_padded = torch.tensor(text_padded, dtype=torch.float).transpose(1,2) # (B,N,c)
        dur_padded = torch.tensor(dur_padded, dtype=torch.float) # (B,N)
        mel_padded = torch.tensor(mel_padded, dtype=torch.float) # (B,d,T)
        
        return (spk_id, text_padded, dur_padded), mel_padded


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train Parallel TTS (See detail in dialtts/acoustic/bin/train.py).")
    parser.add_argument("--train-scp", type=str, required=True,
                        help="train.scp file for training.")
    parser.add_argument("--dev-scp", type=str, required=True,
                        help="valid.scp file for validation. ")
    parser.add_argument("--train-mlf", "--mlf", type=str, required=True,
                        help="mlf file for training and validation. ")
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
    config["version"] = dialtts.acoustic.__version__   # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # get dataset
    dataset_params = {
        "text_channels": config["num_text"],
        "mel_channels": config["num_mel"],
        "hop_size_ms": config["hop_size_ms"],
        "max_mel_length": config["batch_max_steps"],
    }
    train_dataset = TextMelSCPDataset(
        scpfn=args.train_scp,
        mlffn=args.train_mlf,
        **dataset_params,
    )
    logging.info(f"The number of training files = {len(train_dataset)}.")
    dev_dataset = TextMelSCPDataset(
        scpfn=args.dev_scp,
        mlffn=args.train_mlf,
        **dataset_params,
    )
    logging.info(f"The number of development files = {len(dev_dataset)}.")
    dataset = {
        "train": train_dataset,
        "dev": dev_dataset,
    }

    # get data loader
    collater = Collater(
        config["num_text"], config["num_mel"], 
        config["batch_max_steps"],
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
    generator_class = getattr(dialtts.acoustic.models, config["generator_type"])
    discriminator_class = getattr(dialtts.acoustic.models, config["discriminator_type"])
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
        "L1_sum": torch.nn.SmoothL1Loss(reduction="sum", beta=0.001).to(device),
        "L1_mean": torch.nn.SmoothL1Loss(reduction="mean", beta=0.001).to(device),
        "L2_sum": torch.nn.MSELoss(reduction="sum").to(device),
        "L2_mean": torch.nn.MSELoss(reduction="mean").to(device),
    }
    
    generator_optimizer_class = getattr(dialtts.acoustic.optimizers, config["generator_optimizer_type"])
    discriminator_optimizer_class = getattr(dialtts.acoustic.optimizers, config["discriminator_optimizer_type"])
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
    generator_scheduler_class = getattr(dialtts.acoustic.lr_scheduler, config["generator_scheduler_type"])
    discriminator_scheduler_class = getattr(dialtts.acoustic.lr_scheduler, config["discriminator_scheduler_type"])
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

