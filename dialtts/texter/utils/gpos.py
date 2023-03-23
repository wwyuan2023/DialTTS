# coding: utf-8

class GPOS:
    gpos_num = 19
    gpos_list = (
        'a',  # 形容词
        'c',  # 连词
        'd',  # 副词
        'f',  # 方位词
        'i',  # 成语
        'm',  # 数词
        'n',  # 普通名词
        'nr', # 人名
        'nx', # 外语
        'nz', # 专有名词
        'o',  # 拟声词
        'p',  # 介词
        'q',  # 量词
        'r',  # 代词
        't',  # 时间名词
        'u',  # 助词
        'v',  # 动词
        'w',  # 标点符号
        'y',  # 语气词
    )
    gpos_dict = {
        'a': 0,
        'c': 1,
        'd': 2,
        'f': 3,
        'i': 4,
        'm': 5,
        'n': 6,
        'nr': 7,
        'nx': 8,
        'nz': 9,
        'o': 10,
        'p': 11,
        'q': 12,
        'r': 13,
        't': 14,
        'u': 15,
        'v': 16,
        'w': 17,
        'y': 18,
    }
    
    @staticmethod
    def gpos2idx(gpos):
        return GPOS.gpos_dict.get(gpos, 6)
    
    @staticmethod
    def idx2gpos(idx):
        if -GPOS.gpos_num <= idx < GPOS.gpos_num:
            return GPOS.gpos_list[idx]
        return 'n'
    
    @staticmethod
    def is_punc(gpos): # punctuation mark
        if gpos == 'w': return True
        return False
    
    @staticmethod
    def one_hot(gpos):
        hot = [0. for _ in range(GPOS.gpos_num)]
        hot[GPOS.gpos2idx(gpos)] = 1.
        return hot


if __name__ == "__main__":
    
    print(GPOS.one_hot('n'))
    
