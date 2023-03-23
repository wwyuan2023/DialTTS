#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Decode with trained vocoder Generator."""

import sys, os
import torch
import yaml
import numpy as np

import dialtts.vocoder
from dialtts.vocoder.utils import load_model
from dialtts.vocoder.layers import PQMF


# default checkpoint path
default_checkpoint_path = os.path.join(dialtts.vocoder.__path__[0], "checkpoint", "checkpoint.pkl")
        

class NeuralVocoder(object):
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
        self.hop_size = self.config["hop_size"]
        self.sampling_rate = self.config["sampling_rate"]
        
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
        model.remove_weight_norm()
        self.model = model.eval().to(self.device)
        
        # define PQMF object
        self.pqmf = PQMF(self.config["num_band"], taps=self.config["taps"], beta=self.config["beta"]).to(device)
        
        # alias inference
        self.inference = self.infer

    @torch.no_grad()
    def infer(self, x):
        # tensor, (B=1, C, T)
        
        x = self.model.infer(x.to(self.device)) # (B=1,c,t)
        x = self.pqmf.synthesis(x) # (B=1,1,t*2)
        
        return x


def main():
    
    import argparse
    import logging
    import time
    
    import soundfile as sf
    import librosa
    
    from tqdm import tqdm
    from dialtts.vocoder.utils import find_files
    
    
    """Run decoding process."""
    parser = argparse.ArgumentParser(
        description="Generating speech signal with trained Neural Vocoder Generator."
                    "(See detail in dialtts/vocoder/bin/infer.py).")
    parser.add_argument("--feats-scp", "--scp", default=None, type=str,
                        help="feats.scp file. "
                             "you need to specify either feats-scp or dumpdir.")
    parser.add_argument("--dumpdir", default=None, type=str,
                        help="directory including feature files. "
                             "you need to specify either feats-scp or dumpdir.")
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
    model = NeuralVocoder(args.checkpoint, args.config)

    # setup config
    config = model.config
    
    # check arguments
    if (args.feats_scp is not None and args.dumpdir is not None) or \
            (args.feats_scp is None and args.dumpdir is None):
        raise ValueError("Please specify either --dumpdir or --feats-scp.")

    # get feature files
    features = dict()
    if args.dumpdir is not None:
        for melfn in find_files(args.dumpdir, f"*.msc{config[num_mel]}"):
            utt_id = os.path.splitext(os.path.basename(melfn))[0]
            features[utt_id] = melfn
        logging.info("From {} find {} feature files.".format(args.dumpdir, len(features)))
    else:
        with open(args.feats_scp) as fid:
            for line in fid:
                line = line.strip()
                if line == "" or line[0] == "#": continue
                utt_id = os.path.splitext(os.path.basename(line))[0]
                features[utt_id] = line
        logging.info("From {} find {} feature files.".format(args.feats_scp, len(features)))
    logging.info(f"The number of features to be decoded = {len(features)}.")

    # start generation
    total_rtf = 0.0
    for idx, (utt_id, melfn) in enumerate(features.items(), 1):
        start = time.time()
        
        # load feature
        mel = np.fromfile(melfn, dtype=np.float32).reshape(-1, config["num_mel"]).T
        
        # inference
        mel = torch.from_numpy(mel).unsqueeze(0) # (B=1,C,T)
        y = model.infer(mel) # (B=1,C=1,t)
        
        # save audio
        y = y.view(-1).cpu().numpy()
        sf.write(os.path.join(args.outdir, f"{utt_id}.wav"),
            y, config["sampling_rate"], "PCM_16"
        )
        
        rtf = (time.time() - start) / (len(y) / config["sampling_rate"])
        total_rtf += rtf

    # report average RTF
    logging.info(f"Finished generation of {idx} utterances (RTF = {total_rtf / idx:.03f}).")


if __name__ == "__main__":
    main()
