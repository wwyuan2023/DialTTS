# coding: utf-8

import os, sys

class Config:
    batch_size = 8
    
    max_word_length = 10
    max_tone_number = 6
    max_phoneme_number = 8
    max_syllable_number = 63
    
    cn_user_dict_path = "resouces/cn/dict/user_add.txt"
    cn_dict_path = "resouces/cn/dict.v1"
    en_dict_path = "resouces/en/dict.v1"
    
    special_symbols_cn = "resouces/cn/special_symbols_cn.lex"
    polyphone_cn = "resouces/cn/polyphone_cn.lex"
    

