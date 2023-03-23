# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import fnmatch
import os
import sys

import numpy as np
import torch
import yaml



def find_files(root_dir, query="*.wav"):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))

    return files


def get_mask_from_lengths(lengths, max_len=None, device=None):
    # lengths: (B,...)
    if max_len is None:
        max_len = torch.max(lenths).item()
    ids = torch.arange(0, max_len, device=device)
    mask = (ids <lengths.unsqueeze(1)).bool()
    return mask # (B,...,max_len)


def load_model(checkpoint, config=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import dialtts.acoustic.models

    # get model and load parameters
    model_class = getattr(
        dialtts.acoustic.models,
        config["generator_type"]
    )
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    return model


