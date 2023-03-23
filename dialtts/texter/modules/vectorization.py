#coding: utf-8

import os, sys
import numpy as np

from dialtts.texter import Config
from dialtts.texter.utils import GPOS, SenType, Syllable, Phoneme, Lang
from dialtts.texter.utils import SegText
from dialtts.texter.utils import is_punctuation


def one_hot(index, n_class):
    # assert index < n_class, f"{index} < {n_class}"
    hot = [0. for _ in range(n_class)]
    hot[index%n_class] = 1.
    return hot


class Vectorization(object):
    def __init__(self, loglv=0):
        self.loglv = loglv
        
        self.max_tone_number = Config.max_tone_number
        self.max_syllable_number = Config.max_syllable_number
    
    def __call__(self, batch_utt_id, batch_segtext):
        
        _batch_utt_id, _btch_segtext, _batch_vector = [], [], []
        for (utt_id, segtext) in zip(batch_utt_id, batch_segtext):
            utt_id, segtext, vector = self.process(utt_id, segtext)
            if len(segtext) > 0:
                _batch_utt_id.append(utt_id)
                _btch_segtext.append(segtext)
                _batch_vector.append(vector)
        
        return _batch_utt_id, _btch_segtext, _batch_vector
    
    def process(self, utt_id, segtext):
        
        segtext = self._add_sentype(segtext)
        segtext = self._insert_sil(segtext)
        
        outputs = []
        pws = 0 # position of word in sentence
        for i in range(len(segtext)):
            #print(f"!!!!!!!!!! {utt_id}  ", segtext.segtext[i])
            py = segtext.get_py(i)
            if py is None or len(py) == 0:
                continue
            cx, lang, stype = segtext.get_cx(i), segtext.get_lang(i), segtext.get_sentype(i)
            outputs += self.vectoring(lang, py, cx, stype, pws)
            if len(py) == 1 and Syllable.is_sil(py[0]):
                pws = 0
            else:
                pws += len(py)
        
        if self.loglv > 0:
            line = f'Vectorization: {utt_id}    ' + segtext.printer() + '\n'
            sys.stderr.write(line)
        
        vector = np.array(outputs, dtype=np.float32)
        
        return utt_id, segtext, vector
    
    def _insert_sil(self, segtext):
        # 句末是否为sil
        for i in range(len(segtext)-1, -1, -1):
            py = segtext.get_py(i)
            if py is not None and len(py) > 0:
                break;
        py = segtext.get_py(i)
        if not Syllable.is_sil(py[0]):
            segtext.append() # 追加空元素
            segtext.set_wd(-1, "。")
            segtext.set_py(-1, ["sil0"])
            segtext.set_cx(-1, "w")
            segtext.set_lang(-1, segtext.get_lang(-2))
            segtext.set_sentype(-1, segtext.get_sentype(-2))
        
        # 开头是否为sil
        for i in range(len(segtext)):
            py = segtext.get_py(i)
            if py is not None and len(py) > 0:
                break;
        py = segtext.get_py(i)
        if not Syllable.is_sil(py[0]):
            segtext.insert(0) # 插入空元素
            segtext.set_wd(0, SenType.idx2sentype(segtext.get_sentype(1)))
            segtext.set_py(0, ["sil0"])
            segtext.set_cx(0, "w")
            segtext.set_lang(0, segtext.get_lang(1))
            segtext.set_sentype(0, segtext.get_sentype(1))
        
        return segtext
    
    def _add_sentype(self, segtext):
        
        stype = 0
        for i in range(len(segtext)-1, -1, -1):
            wd = segtext.get_wd(i)
            cx = segtext.get_cx(i)
            if(GPOS.is_punc(cx) and is_punctuation(wd)):
                stype = SenType.sentype2idx(wd)
            segtext.set_sentype(i, stype)
        
        return segtext
    
    def vectoring(self, lang, pys, cx, stype, pws):
        outs = []
        py_num = len(pys)
        for j in range(py_num):
            py = pys[j]
            py, tone = py[:-1], int(py[-1])
            phns = Syllable.s2p(py) # "wu" -> ("CNuw")
            phn_num = len(phns)
            for k in range(phn_num):
                vec = []
                phn = phns[k]
                vec += Phoneme.one_hot(phn)                     # phoneme via one-hot. [0, 90)
                vec += one_hot(tone, self.max_tone_number)      # tone via one-hot. [90, 96)
                vec += one_hot(1 if k + 1 == phn_num else 0, 2) # syllable boundary via one-hot. [96, 98)
                vec += one_hot(1 if j + 1 == py_num else 0, 2)  # word boundary via one-hot. [98, 100)
                vec += GPOS.one_hot(cx)                         # GPOS via one-hot. [100, 119)
                vec += SenType.one_hot(stype)                   # sentence type via one-hot. [119, 129)
                vec += one_hot(pws+j, self.max_syllable_number) # position of syllable in sentence via one-hot. [129, 192)
                outs.append(vec)
        return outs
    
    def devectoring(self, vector:np.ndarray):
        # decode vector list to prosody dict
        prosody = list()
        if len(vector) == 0: return prosody
        
        vector = vector.astype(np.int32)
        wr = np.arange(vector.shape[1])
        
        for i in range(len(vector)):
            prosody.append(dict())
            prosody[-1]["phoneme"] = Phoneme.idx2phn(np.sum(vector[i,:90] * wr[:90]))
            prosody[-1]["tone"] = int(sum(vector[i,90:96] * wr[:6]))
            prosody[-1]["syllable_boundary"] = int(sum(vector[i,96:98] * wr[:2]))
            prosody[-1]["word_boundary"] = int(sum(vector[i,98:100] * wr[:2]))
            prosody[-1]["cx"] = GPOS.idx2gpos(sum(vector[i,100:119] * wr[:19]))
            prosody[-1]["sent_type"] = SenType.idx2sentype(sum(vector[i,119:129] * wr[:10]))
            prosody[-1]["position"] = int(sum(vector[i,129:192] * wr[:63]))
        
        return prosody


def vectorization(fid=sys.stdin):
    assert len(sys.argv) > 1, "outdir must be given!"
    outdir = sys.argv[1]
    
    loglv = 0 if len(sys.argv) <= 2 else int(sys.argv[2])
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    vectorization = Vectorization(loglv)
    
    while True:
        # split batch
        batch_utt_id, batch_utt_text = [], []
        for line in fid:
            line = line.strip()
            if line == '': continue
            for i in range(len(line)):
                if line[i].isspace():
                    break
            utt_id, utt_text = line[:i].strip(), line[i:].strip()
            batch_utt_id.append(utt_id)
            batch_utt_text.append(utt_text)
            if len(batch_utt_id) >= Config.batch_size:
                break
        
        if len(batch_utt_id) == 0: break
        
        # vectorization
        batch_segtext = []
        for (utt_id, utt_text) in zip(batch_utt_id, batch_utt_text):
            segtext = SegText(utt_text)
            batch_segtext.append(segtext)
        batch_utt_id, batch_segtext, batch_vector = vectorization(batch_utt_id, batch_segtext)
        
        # save and output
        for (utt_id, segtext, vector) in zip(batch_utt_id, batch_segtext, batch_vector):
            outpath = os.path.join(outdir, f"{utt_id}.vec{vector.shape[1]}")
            sys.stderr.write(f"Save to {outpath}, shape={vector.shape}\n")
            vector.tofile(outpath)
            
            # checking
            if loglv >= 2:
                prosody = vectorization.devectoring(vector)
                for p in prosody:
                    print(p)


if __name__ == "__main__":
    
    vectorization()
    
