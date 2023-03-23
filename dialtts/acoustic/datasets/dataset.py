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

def load_mlffn(filename, ncols=3):
    mlf = dict()
    with open(filename, encoding='utf-8') as fd:
        for line in fd:
            line = line.strip()
            if len(line) == 0 or line[0] == '#': continue
            if line[:3] == '"*/' and line[-5:] == '.lab"': # "*/0001.lab"
                utt_id = line[3:-5]
                phn_list = list()
                for line in fd:
                    line = line.strip()
                    if len(line) == 0 or line[0] == '#': continue
                    if line == '.':break
                    # format: (start,end,phoneme), eg: 630 700 CNm
                    phn_list.append(line.split())
                    assert len(phn_list[-1]) == ncols, f"{line}"
                mlf[utt_id] = np.array(phn_list)
            else:
                assert 0, f"Format error: {line}\n"
    return mlf

def load_binfn(filename, dim):
    return np.fromfile(filename, dtype=np.float32).reshape(-1, dim)


class TextMelSCPDataset(Dataset):
    """PyTorch compatible text and mel feature dataset based on scp files."""

    def __init__(
        self,
        scpfn,
        mlffn,
        text_channels,
        mel_channels,
        hop_size_ms,
        max_mel_length=None,
        return_utt_id=False,
    ):
        """Initialize dataset.
        """
        _scp = load_scpfn(scpfn, "|") # [(vecfn,melfn,spkid), ...]
        _mlf = load_mlffn(mlffn, 3)
        
        self.scplist = []
        for vecfn, melfn, spkid in _scp:
            utt_id = os.path.splitext(os.path.basename(vecfn))[0]
            assert utt_id in _mlf, f"{utt_id} is not in mlf\n"
            num = len(_mlf[utt_id]) # phoneme number
            vec = load_binfn(vecfn, text_channels)
            assert num == len(vec), f"{utt_id}, {num}, {vec.shape}, {_mlf[utt_id]}"
            phnlab = np.delete(_mlf[utt_id], 2, axis=1).astype(np.float32) # delete phoneme name
            dur = (phnlab[:,1] - phnlab[:,0]) / hop_size_ms # phoneme frame number, shape=(N,), dtype=float32
            num = dur.sum() # total duration
            mel = load_binfn(melfn, mel_channels)
            assert abs(num - len(mel)) <= 5, f"{utt_id}, {num}, {len(mel)}\n"
            self.scplist += [ tuple([utt_id, int(spkid), vecfn, melfn, dur]) ]
        logging.warning(f"{len(self.scplist)} files are loaded!")
        
        self.text_channels = text_channels
        self.mel_channels = mel_channels
        self.hop_size_ms = hop_size_ms
        self.max_mel_length = max_mel_length
        self.return_utt_id = return_utt_id
    
    def _convert_duration(self, dur):
        _dur = np.zeros(shape=dur.shape, dtype=np.int64)
        diff = 0.
        for i in range(dur.shape[0]):
            d = max(1, int(dur[i] + diff + 0.5))
            diff += dur[i] - d
            _dur[i] = d
        return _dur
    
    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            int: Speaker Id .
            ndarray: Text encoding vector (c,N).
            ndarray: Phoneme duration (N,).
            ndarray: Mel spectrogram (d,T).
        """
        utt_id, spk_id, vecfn, melfn, dur = self.scplist[idx]
        
        spk_id = int(spk_id)
        #dur = (dur + 0.5).astype(np.int64) # float32 -> int64
        dur = self._convert_duration(dur)
        
        vec = load_binfn(vecfn, self.text_channels)
        mel = load_binfn(melfn, self.mel_channels)
        assert len(vec) == len(dur), f"{len(vec)}, {len(dur)}"
        
        if self.max_mel_length is not None and mel.shape[0] > self.max_mel_length:
            while True:
                start = np.random.randint(0, vec.shape[0]-2)
                end = np.random.randint(start+1, vec.shape[0])
                if dur[start:end].sum() < self.max_mel_length:
                    vec = vec[start:end]
                    mel_start, mel_end = dur[:start].sum(), dur[:end].sum()
                    mel = mel[mel_start:mel_end]
                    dur = dur[start:end]
                    assert dur.sum() == mel.shape[0], f"{dur.sum()}, {mel.shape[0]}"
                    break
        
        if mel.shape[0] >= dur.sum():
            mel = mel[:dur.sum()]
        else:
            d = dur.sum() - mel.shape[0]
            mel = np.pad(mel, ((0,d),(0,0)), "edge")
        
        vec = vec.T # (c,N)
        mel = mel.T # (d,T)
        assert vec.shape[1] == len(dur)
        assert mel.shape[1] == dur.sum()
        
        if self.return_utt_id:
            items = utt_id, spk_id, vec, dur, mel
        else:
            items = spk_id, vec, dur, mel

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.scplist)


class TextSCPDataset(Dataset):
    """PyTorch compatible text and mel feature dataset based on scp files."""

    def __init__(
        self,
        scpfn,
        text_channels,
        return_utt_id=False,
    ):
        """Initialize dataset.
        """
        _scp = load_scpfn(scpfn, "|") # [(vecfn,spkid), ...]
        
        self.scplist = []
        for vecfn, spkid in _scp:
            utt_id = os.path.splitext(os.path.basename(vecfn))[0]
            self.scplist += [ tuple([utt_id, int(spkid), vecfn]) ]
        logging.warning(f"{len(self.scplist)} files are loaded!")
        
        self.text_channels = text_channels
        self.return_utt_id = return_utt_id
    
    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            int: Speaker Id .
            ndarray: Text encoding vector (c,N).
            ndarray: Phoneme duration (N,).
            ndarray: Mel spectrogram (d,T).
        """
        utt_id, spk_id, vecfn = self.scplist[idx]
        
        spk_id = int(spk_id)
        vec = load_binfn(vecfn, self.text_channels)
        
        if self.return_utt_id:
            items = utt_id, spk_id, vec
        else:
            items = spk_id, vec

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.scplist)

