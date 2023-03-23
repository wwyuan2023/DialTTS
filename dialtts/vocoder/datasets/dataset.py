# -*- coding: utf-8 -*-

import os
import logging
import warnings
import numpy as np
import soundfile as sf
import torch

from torch.utils.data import Dataset


def load_scpfn(filename, delimter="|"):
    scplist = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                scplist.append(line.split(delimter))
    return scplist


def load_melfn(filename, dim):
    return np.fromfile(filename, dtype=np.float32).reshape(-1, dim)


class AudioMelSCPDataset(Dataset):
    """PyTorch compatible audio signal and mel feature dataset based on scp files."""

    def __init__(
        self,
        scpfn,
        mel_channels,
        hop_size,
        sampling_rate_threshold,
        mel_length_threshold=None,
        return_utt_id=False,
    ):
        """Initialize dataset.
        """
        _scplist = load_scpfn(scpfn) # [(wavfn,melfn), ...]
        self.scplist = []
        drop, count = 0, 0
        for files in _scplist:
            melfn, wavfn = files[0], files[1]
            mel = load_melfn(melfn, mel_channels)
            audio, sr = sf.read(wavfn)
            
            if (sr != sampling_rate_threshold):
                drop += 1
                logging.warning(f"Drop [{wavfn}]/[{melfn}]: sampling_rate={sr}, sampling_rate_threshold={sampling_rate_threshold}")
                continue
            
            if mel_length_threshold is not None and mel.shape[0] <= mel_length_threshold:
                drop += 1
                logging.warning(f"Drop [{wavfn}]/[{melfn}]: mel_length_threshold={mel_length_threshold}, mel_length={mel.shape[0]}")
                continue
            
            if abs(len(audio) - mel.shape[0]*hop_size) > 5 * hop_size:
                drop += 1
                logging.warning(f"Drop [{wavfn}]/[{melfn}]: audio_length={len(audio)}, mel_length={mel.shape[0]}")
                continue
            
            count +=1
            self.scplist += [ tuple([melfn, wavfn]) ]
        logging.warning(f"{drop} files are dropped, {count} files are loaded!")
        
        self.mel_channels = mel_channels
        self.hop_size = hop_size
        self.return_utt_id = return_utt_id
    
    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).

        """
        melfn, wavfn = self.scplist[idx]
        mel = load_melfn(melfn, self.mel_channels)
        audio, sr = sf.read(wavfn)
        
        # normalize audio signal to be [-1, 1]
        audio = audio.astype(np.float32)
        audio /= abs(audio).max() # [-1, 1]
        
        # check length
        diff =  mel.shape[0] * self.hop_size - len(audio)
        if diff > 0:
            audio = np.pad(audio, [0,diff], mode="constant", constant_values=0)
        elif diff < 0:
            audio = audio[:mel.shape[0]*self.hop_size]
        
        assert mel.shape[0] * self.hop_size == len(audio)
        
        if self.return_utt_id:
            utt_id = os.path.splitext(os.path.basename(wavfn))[0]
            items = utt_id, audio, mel
        else:
            items = audio, mel

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.scplist)


