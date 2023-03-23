# coding: utf-8

import os, sys
import fnmatch

import textgrid

from dialtts.texter.utils import SegText, Syllable


def get_phn_align(filename):
    tgo = textgrid.TextGrid.fromFile(filename)
    phn_iters = tgo.getFirst('phones')
    
    phnlist = [] # 3-cols: start, end, text
    for it in phn_iters:
        start_ms, end_ms, text = int(it.minTime * 1000), int(it.maxTime * 1000), it.mark
        if text in ['', 'sil', 'sp']:
            text = 'sil'
            if len(phnlist) > 0 and phnlist[-1][2] == 'sil': # combine break
                phnlist[-1][1] = end_ms
                continue
        phnlist.append([start_ms, end_ms, text])
    
    return phnlist

def process_utterance(utt_id, segtext, phnlist):
    _segtext, _phnlist = SegText(), [] # output
    
    phnlab = phnlist.pop(0)
    _phnlist.append(phnlab) # 句首的sil直接拷贝
    assert phnlab[2] == "sil"
    
    pys = segtext.get_py(-1) # 判断是否sil结尾，否则添加一个
    if len(pys) != 1 or pys[0] != "sil0":
        segtext.append(segtext[-1].copy())
        segtext.set_wd(-1, "。")
        segtext.set_py(-1, ["sil0"])
        segtext.set_cx(-1, "w")
    
    for i in range(len(segtext)):
        _segtext.append(segtext[i].copy())
        pys = _segtext.get_py(-1)
        cx = _segtext.get_cx(-1)
        lang = _segtext.get_lang(-1)
        
        if pys is None or len(pys) == 0:
            #if cx == 'w': # 是符号，但不是分句标点
            #    if phnlist[0][2] == "sil" and phnlist[0][1] - phnlist[0][0] >= 150: # 静音大于150ms
            #        _segtext.set_wd(-1, "，")  # 替换成逗号
            #        _segtext.set_py(-1, ["sil0"])
            #        _phnlist.append(phnlist.pop(0))
            continue
        
        if len(pys) == 1 and Syllable.is_sil(pys[0]):
            if len(phnlist) <= 0:
                print("ERROR!!!!!!!!!! ", _segtext, _phnlist, segtext[i:], phnlist)
                assert 0
            if phnlist[0][2] == "sil":
                _phnlist.append(phnlist.pop(0))
            else:
                _segtext.pop(-1)
            continue
        
        if phnlist[0][2] == "sil": # 词边界静音
            if phnlist[0][1] - phnlist[0][0] >= 150: # 大于150ms
                _segtext.insert(-1, _segtext[-1].copy()) # 插入一个逗号
                _segtext.set_wd(-2, "，")
                _segtext.set_py(-2, ["sil0"])
                _segtext.set_cx(-2, "w")
                _phnlist.append(phnlist.pop(0))
            else:
                phnlist[1][0] = phnlist[0][0] # 合并到后面的音素
                phnlist.pop(0)
        
        # 词到音素的拆分
        for py in pys:
            py, tone = py[:-1], int(py[-1:])
            phns = Syllable.s2p(py) # "wu" -> ["CNuw"]
            for k in range(len(phns)):
                if phns[k] == phnlist[0][2]:
                    _phnlist.append(phnlist.pop(0))
                else:
                    print("!!!!!!!!!!!!!!!")
                    print(_segtext)
                    print(_phnlist)
                    print(f"utt_id={utt_id}, i={i}, k={k}, py={py}, tone={tone}, phns[k]={phns[k]},,,,,,")
                    assert 0
        continue
    
    # check 
    if len(phnlist) > 0:
        print("????????????????????")
        print(_segtext)
        print(_phnlist)
        print(phnlist)
        assert 0
    
    return utt_id, _segtext, _phnlist


def main(textgrid_dir, labfn, labfn_out, mlfn_out):
    
    # find all .TextGrid files
    textgrid_files = dict()
    for root, dirnames, filenames in os.walk(textgrid_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, "*.TextGrid"):
            basename = os.path.splitext(filename)[0]
            textgrid_files[basename] = os.path.join(textgrid_dir, filename)
    print(f"{textgrid_dir}: {len(textgrid_files)} textgrid files!")
    
    # load .lab from stdin
    batch_utt_id, batch_utt_text = [], []
    with open(labfn_in, "rt") as fid:
        for line in fid:
            line = line.strip()
            if line == "": continue
            for i in range(len(line)):
                if line[i].isspace():
                    break
            utt_id, utt_text = line[:i].strip(), line[i:].strip()
            if len(utt_text) == 0: continue
            if utt_id not in textgrid_files:
                print(f"utt_id={utt_id}: There is no corresponding textgrid file!")
                continue
            batch_utt_id.append(utt_id)
            batch_utt_text.append(utt_text)
    print(f"{labfn_in}: {len(batch_utt_id)} files!")
    
    # the data on both sides, discard the unmatched
    _batch_utt_id, _batch_utt_text, _batch_utt_phn = [], [], []
    for (utt_id, utt_text) in zip(batch_utt_id, batch_utt_text):
        print(f"Process {utt_id}:")
        # get phoneme alignment
        phnlist = get_phn_align(textgrid_files[utt_id])
        #print(phnlist)
        assert phnlist[0][2] == "sil" and phnlist[-1][2] == "sil"
        # parse utterance
        segtext = SegText(utt_text)
        # processing
        utt_id, segtext, phnlist = process_utterance(utt_id, segtext, phnlist)
        
        _batch_utt_id.append(utt_id)
        _batch_utt_text.append(segtext.printer())
        phn_str = ''
        for p in phnlist:
            phn_str += f"{p[0]} {p[1]} {p[2]}\n"
        _batch_utt_phn.append(phn_str)
        #print("{}, {}".format(_batch_utt_text[-1], _batch_utt_phn[-1]))
        print(f"Processing {utt_id} done!")
    
    # write .lab/.mlf file
    with open(labfn_out, 'wt') as f:
        for (utt_id, utt_text) in zip(_batch_utt_id, _batch_utt_text):
            f.write(f"{utt_id} {utt_text}\n")
    with open(mlfn_out, 'wt') as f:
        f.write("#!MLF!#\n")
        f.write("#timestap unit: ms\n")
        for (utt_id, phn_list) in zip(_batch_utt_id, _batch_utt_phn):
            f.write(f"\"*/{utt_id}.lab\"\n {phn_list}.\n")


if __name__ == "__main__":
    
    textgrid_dir, labfn_in = sys.argv[1], sys.argv[2]
    labfn_out, mlfn_out = sys.argv[3], sys.argv[4]
    
    main(textgrid_dir, labfn_in, labfn_out, mlfn_out)
    