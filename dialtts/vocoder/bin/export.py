#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export Neural vocoder Model checkpoint."""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml

import dialtts.vocoder
import dialtts.vocoder.models


def main():
    
    parser = argparse.ArgumentParser(
        description="Export Neural vocoder Generator (See detail in dialtts/vocoder/bin/export.py).")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", default=None, type=str,
                        help="yaml format configuration file.")
    parser.add_argument("--convert", default=0, type=int,
                        help="convert to torch script if setting nonzero.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()
    
    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
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

    # load config
    config_path = args.config
    if config_path is None:
        dirname = os.path.dirname(args.checkpoint)
        config_path = os.path.join(dirname, "config.yml")
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # save config to outdir
    config["version"] = dialtts.vocoder.__version__   # add version info
    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    # load generator
    generator_class = getattr(
        dialtts.vocoder.models,
        config["generator_type"]
    )
    generator = generator_class(**config["generator_params"])
    generator.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu")["model"]["generator"]
    )
    logging.info(generator)
    
    # print parameters
    total_params, trainable_params, nontrainable_params = 0, 0, 0
    for param in generator.parameters():
        num_params = np.prod(param.size())
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        else:
            nontrainable_params += num_params
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")
    logging.info(f"Non-trainable parameters: {nontrainable_params}\n")
    
    # save generator to outdir
    checkpoint_path = os.path.join(args.outdir, "checkpoint.pkl")
    state_dict = {
        "model": {"generator": generator.state_dict()}
    }
    torch.save(state_dict, checkpoint_path)
    logging.info(f"Successfully export generator parameters from [{args.checkpoint}] to [{checkpoint_path}].")
    
    # export Torch Script
    if args.convert != 0:
        # load parameters
        generator_class = getattr(
            dialtts.vocoder.models,
            config["generator_type"]
        )
        generator = generator_class(**config["generator_params"])
        generator.load_state_dict(
            torch.load(args.checkpoint, map_location="cpu")["model"]["generator"],
            strict=False,
        )
        if hasattr(generator, "remove_weight_norm"): generator.remove_weight_norm()
        generator.eval()
        
        # export
        batch_size = 1
        dummy_input = torch.zeros(batch_size, config["num_mel"], 1)
        traced = torch.jit.trace(generator, dummy_input)
        
        script_path = os.path.join(args.outdir, config["generator_type"].lower() + ".script")
        torch.jit.save(traced, script_path)
        logging.info(f"Successfully convert to torch script module: [{script_path}].\n\n")
        


if __name__ == "__main__":
    main()
