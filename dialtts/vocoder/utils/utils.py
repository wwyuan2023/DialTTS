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


def encode_mulaw_numpy(x, mu=1024):
    """FUNCTION TO PERFORM MU-LAW ENCODING

    Args:
        x (ndarray): audio signal with the range from -1 to 1
        mu (int): quantized level

    Return:
        (ndarray): quantized audio signal with the range from 0 to mu - 1
    """
    mu = mu - 1
    fx = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return np.floor((fx + 1) / 2 * mu + 0.5).astype(np.int64)


def encode_mulaw_torch(x, mu=1024):
    """FUNCTION TO PERFORM MU-LAW ENCODING

    Args:
        x (ndarray): audio signal with the range from -1 to 1
        mu (int): quantized level

    Return:
        (ndarray): quantized audio signal with the range from 0 to mu - 1
    """
    mu = mu - 1
    fx = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / np.log(1 + mu) # log1p(x) = log_e(1+x)
    return torch.floor((fx + 1) / 2 * mu + 0.5).long()


def decode_mulaw_numpy(y, mu=1024):
    """FUNCTION TO PERFORM MU-LAW DECODING

    Args:
        x (ndarray): quantized audio signal with the range from 0 to mu - 1
        mu (int): quantized level

    Return:
        (ndarray): audio signal with the range from -1 to 1
    """
    #fx = 2 * y / (mu - 1.) - 1.
    mu = mu - 1
    fx = y / mu * 2 - 1
    x = np.sign(fx) / mu * ((1 + mu) ** np.abs(fx) - 1)
    return np.clip(x, a_min=-1, a_max=0.999969482421875)


def decode_mulaw_torch(y, mu=1024):
    """FUNCTION TO PERFORM MU-LAW DECODING

    Args:
        x (ndarray): quantized audio signal with the range from 0 to mu - 1
        mu (int): quantized level

    Return:
        (ndarray): audio signal with the range from -1 to 1
    """
    #fx = 2 * y / (mu - 1.) - 1.
    mu = mu - 1
    fx = y / mu * 2 - 1
    x = torch.sign(fx) / mu * ((1 + mu) ** torch.abs(fx) - 1)
    return torch.clamp(x, min=-1, max=0.999969482421875)

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
    import dialtts.vocoder.models

    # get model and load parameters
    model_class = getattr(
        dialtts.vocoder.models,
        config["generator_type"]
    )
    model = model_class(**config["generator_params"])
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    return model


