#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Decode with trained vocoder Generator."""

import sys, os
import torch
import yaml
import numpy as np

import dialtts.acoustic
from dialtts.acoustic.utils import load_model


# default checkpoint path
default_checkpoint_path = os.path.join(dialtts.acoustic.__path__[0], "checkpoint", "checkpoint.pkl")
        

class NeuralAcoustic(object):
    def __init__(self, checkpoint_path=None, config_path=None, device=None):
        
        if checkpoint_path is None:
            checkpoint_path = default_checkpoint_path
    
        # setup config
        if config_path is None:
            dirname = os.path.dirname(checkpoint_path)
            config_path = os.path.join(dirname, "config.yml")
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)
        
        self.num_mel = self.config["num_mel"]
        
        # setup device
        if device is not None:
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # setup model
        model = load_model(checkpoint_path, self.config)
        if hasattr(model, "remove_weight_norm"):
            model.remove_weight_norm()
        self.model = model.eval().to(self.device)
        
        # alias inference
        self.inference = self.infer

    @torch.no_grad()
    def infer(self, spkid, text, duration_rate=1.0):
        # spkid: (B=1,)
        # text: (B=1,N,c)
        
        mel, dur = self.model.infer(spkid.to(self.device), text.to(self.device), duration_rate=duration_rate) # (B,C,T)/(B,N)
        
        return mel, dur


def main():
    
    import argparse
    import logging
    import time
    
    import soundfile as sf
    import librosa
    
    from tqdm import tqdm
    from dialtts.acoustic.utils import find_files
    
    
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Decode dumped features with trained Neural Acoustic Generator "
                    "(See detail in dialtts/acoustic/bin/infer.py).")
    parser.add_argument("--feats-scp", "--scp", type=str, required=True,
                        help="feats.scp file. ")
    parser.add_argument("--spk-id", "--spk", default=1, type=int,
                        help="speaker Id. (default=1)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="directory to save generated speech.")
    parser.add_argument("--checkpoint", default=None, type=str, 
                        help="checkpoint file to be loaded.")
    parser.add_argument("--config", default=None, type=str,
                        help="yaml format configuration file. if not explicitly provided, "
                             "it will be searched in the checkpoint directory. (default=None)")
    parser.add_argument("--verbose", type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # setup model
    model = NeuralAcoustic(args.checkpoint, args.config)

    # setup config
    config = model.config
    
    # get feature files
    features = dict()
    with open(args.feats_scp) as fid:
        for line in fid:
            line = line.strip()
            if line == "" or line[0] == "#": continue
            lines = line.split('|')
            utt_id = os.path.splitext(os.path.basename(lines[0]))[0]
            spk_id = int(lines[1]) if len(lines) > 1 else int(args.spk_id)
            features[utt_id] = (spk_id, lines[0])
    logging.info(f"The number of features to be decoded = {len(features)}.")

    # start generation
    total_rtf = 0.0
    for idx, (utt_id, (spkid, vecfn)) in enumerate(features.items(), 1):
        start = time.time()
        
        # load feature
        text = np.fromfile(vecfn, dtype=np.float32).reshape(-1, config["num_text"])
        
        # inference
        text = torch.from_numpy(text).unsqueeze(0) # (B=1,N,c)
        spkid = torch.tensor((spkid,), dtype=torch.long)
        mel, dur = model.infer(spkid, text) # (T,)
        
        # save feature
        mel = mel.squeeze(0).cpu().numpy().T # (T,C)
        mel.tofile(os.path.join(args.outdir, f"{utt_id}.msc{mel.shape[1]}"))
        
        rtf = (time.time() - start) / (len(mel) * config["hop_size_ms"] / 1000)
        total_rtf += rtf

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
