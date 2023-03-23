#coding: utf-8

import os, sys
import fnmatch
import numpy as np
import librosa

###########################
sample_rate = 8000
mels_dim = 48
n_fft = 1024
frame_size = 48 * int(sample_rate / 1000)   # 48 ms
hop_size = 12 * int(sample_rate / 1000)     # 12 ms
fmin = 70    # Hz
fmax = 4000  # Hz
###########################


if __name__ == "__main__":
    
    file_scp = 'files_id.scp'
    in_wav_dir = 'in_wav_dir'
    out_feat_dir = 'out_mel_dir'
    
    file_scp, in_wav_dir, out_feat_dir = os.sys.argv[1:4]
    
    # load file id 
    files_id = []
    with open(file_scp, 'rt') as fid:
        for line in fid:
            line = line.strip()
            if line == '': continue
            files_id.append(line)
    
    # if `file_scp` is null, walk `in_wav_dir`
    if len(files_id) == 0:
        for root, dirnames, filenames in os.walk(in_wav_dir):
            for filename in fnmatch.filter(filenames, "*.wav"):
                files_id.append(filename[:-len('.wav')])
    
    for i in range(len(files_id)):
        wavfn = os.path.join(in_wav_dir, files_id[i]+'.wav')
        sys.stdout.write('extract from %s ...' % wavfn)
        # read wav
        y, sr = librosa.load(wavfn, sr=sample_rate)
        # add random
        y += np.random.random(size=y.shape) * 2e-6 - 1e-6
        # normalization
        y /= max(abs(y))
        # extract mel-frequency spectrum coefs
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_size, win_length=frame_size, center=True)
        S = np.abs(D)
        S = librosa.feature.melspectrogram(y=None, sr=sr, S=S, n_mels=mels_dim, n_fft=n_fft, hop_length=hop_size, power=1.0, fmin=fmin, fmax=fmax)
        S = np.clip(S, np.exp(-10), np.exp(10))
        log_S = np.log(S)
        mel = log_S.transpose(1,0).astype(np.float32) # (T,d)
        # write ot featfn
        featfn = os.path.join(out_feat_dir, files_id[i]+'.msc'+str(mels_dim))
        with open(featfn, 'wb') as fid:
            mel.tofile(fid)
        sys.stdout.write('done.\n')
