# coding: utf-8

class Syllable:
    syl_dict = {
        "sil": ("sil",),
        "a": ("CNaa",),
        "ai": ("CNay", "CNib",),
        "an": ("CNae", "CNn",),
        "ang": ("CNah", "CNng",),
        "ao": ("CNah", "CNub",),
        "ba": ("CNb", "CNaa",),
        "bai": ("CNb", "CNay", "CNib",),
        "ban": ("CNb", "CNae", "CNn",),
        "bang": ("CNb", "CNah", "CNng",),
        "bao": ("CNb", "CNah", "CNub",),
        "bei": ("CNb", "CNei", "CNib",),
        "ben": ("CNb", "CNax", "CNn",),
        "beng": ("CNb", "CNoe", "CNng",),
        "bi": ("CNb", "CNiy",),
        "bian": ("CNb", "CNil", "CNeh", "CNn",),
        "biao": ("CNb", "CNil", "CNah", "CNub",),
        "bie": ("CNb", "CNil", "CNee",),
        "bin": ("CNb", "CNih", "CNn",),
        "bing": ("CNb", "CNih", "CNng",),
        "bo": ("CNb", "CNul", "CNao",),
        "bu": ("CNb", "CNuw",),
        "ca": ("CNc", "CNaa",),
        "cai": ("CNc", "CNay", "CNib",),
        "can": ("CNc", "CNae", "CNn",),
        "cang": ("CNc", "CNah", "CNng",),
        "cao": ("CNc", "CNah", "CNub",),
        "ce": ("CNc", "CNea",),
        "cei": ("CNc", "CNei", "CNib",),
        "cen": ("CNc", "CNax", "CNn",),
        "ceng": ("CNc", "CNoe", "CNng",),
        "cha": ("CNch", "CNaa",),
        "chai": ("CNch", "CNay", "CNib",),
        "chan": ("CNch", "CNae", "CNn",),
        "chang": ("CNch", "CNah", "CNng",),
        "chao": ("CNch", "CNah", "CNub",),
        "che": ("CNch", "CNea",),
        "chen": ("CNch", "CNax", "CNn",),
        "cheng": ("CNch", "CNoe", "CNng",),
        "chi": ("CNch", "CNizh",),
        "chong": ("CNch", "CNoh", "CNng",),
        "chou": ("CNch", "CNoh", "CNub",),
        "chu": ("CNch", "CNuw",),
        "chua": ("CNch", "CNul", "CNaa",),
        "chuai": ("CNch", "CNul", "CNay", "CNib",),
        "chuan": ("CNch", "CNul", "CNae", "CNn",),
        "chuang": ("CNch", "CNul", "CNah", "CNng",),
        "chui": ("CNch", "CNul", "CNei", "CNib",),
        "chun": ("CNch", "CNul", "CNax", "CNn",),
        "chuo": ("CNch", "CNul", "CNao",),
        "ci": ("CNc", "CNiz",),
        "cong": ("CNc", "CNoh", "CNng",),
        "cou": ("CNc", "CNoh", "CNub",),
        "cu": ("CNc", "CNuw",),
        "cuan": ("CNc", "CNul", "CNae", "CNn",),
        "cui": ("CNc", "CNul", "CNei", "CNib",),
        "cun": ("CNc", "CNul", "CNax", "CNn",),
        "cuo": ("CNc", "CNul", "CNao",),
        "da": ("CNd", "CNaa",),
        "dai": ("CNd", "CNay", "CNib",),
        "dan": ("CNd", "CNae", "CNn",),
        "dang": ("CNd", "CNah", "CNng",),
        "dao": ("CNd", "CNah", "CNub",),
        "de": ("CNd", "CNea",),
        "dei": ("CNd", "CNei", "CNib",),
        "den": ("CNd", "CNax", "CNn",),
        "deng": ("CNd", "CNoe", "CNng",),
        "di": ("CNd", "CNiy",),
        "dia": ("CNd", "CNil", "CNaa",),
        "dian": ("CNd", "CNil", "CNeh", "CNn",),
        "diao": ("CNd", "CNil", "CNah", "CNub",),
        "die": ("CNd", "CNil", "CNee",),
        "ding": ("CNd", "CNih", "CNng",),
        "diu": ("CNd", "CNil", "CNoh", "CNub",),
        "dong": ("CNd", "CNoh", "CNng",),
        "dou": ("CNd", "CNoh", "CNub",),
        "du": ("CNd", "CNuw",),
        "duan": ("CNd", "CNul", "CNae", "CNn",),
        "dui": ("CNd", "CNul", "CNei", "CNib",),
        "dun": ("CNd", "CNul", "CNax", "CNn",),
        "duo": ("CNd", "CNul", "CNao",),
        "e": ("CNea",),
        "ei": ("CNei", "CNib",),
        "en": ("CNax", "CNn",),
        "eng": ("CNoe", "CNng",),
        "er": ("CNaar",),
        "fa": ("CNf", "CNaa",),
        "fan": ("CNf", "CNae", "CNn",),
        "fang": ("CNf", "CNah", "CNng",),
        "fei": ("CNf", "CNei", "CNib",),
        "fen": ("CNf", "CNax", "CNn",),
        "feng": ("CNf", "CNoe", "CNng",),
        "fiao": ("CNf", "CNil", "CNah", "CNub",),
        "fo": ("CNf", "CNul", "CNao",),
        "fou": ("CNf", "CNoh", "CNub",),
        "fu": ("CNf", "CNuw",),
        "ga": ("CNg", "CNaa",),
        "gai": ("CNg", "CNay", "CNib",),
        "gan": ("CNg", "CNae", "CNn",),
        "gang": ("CNg", "CNah", "CNng",),
        "gao": ("CNg", "CNah", "CNub",),
        "ge": ("CNg", "CNea",),
        "gei": ("CNg", "CNei", "CNib",),
        "gen": ("CNg", "CNax", "CNn",),
        "geng": ("CNg", "CNoe", "CNng",),
        "gong": ("CNg", "CNoh", "CNng",),
        "gou": ("CNg", "CNoh", "CNub",),
        "gu": ("CNg", "CNuw",),
        "gua": ("CNg", "CNul", "CNaa",),
        "guai": ("CNg", "CNul", "CNay", "CNib",),
        "guan": ("CNg", "CNul", "CNae", "CNn",),
        "guang": ("CNg", "CNul", "CNah", "CNng",),
        "gui": ("CNg", "CNul", "CNei", "CNib",),
        "gun": ("CNg", "CNul", "CNax", "CNn",),
        "guo": ("CNg", "CNul", "CNao",),
        "ha": ("CNh", "CNaa",),
        "hai": ("CNh", "CNay", "CNib",),
        "han": ("CNh", "CNae", "CNn",),
        "hang": ("CNh", "CNah", "CNng",),
        "hao": ("CNh", "CNah", "CNub",),
        "he": ("CNh", "CNea",),
        "hei": ("CNh", "CNei", "CNib",),
        "hen": ("CNh", "CNax", "CNn",),
        "heng": ("CNh", "CNoe", "CNng",),
        "hong": ("CNh", "CNoh", "CNng",),
        "hou": ("CNh", "CNoh", "CNub",),
        "hu": ("CNh", "CNuw",),
        "hua": ("CNh", "CNul", "CNaa",),
        "huai": ("CNh", "CNul", "CNay", "CNib",),
        "huan": ("CNh", "CNul", "CNae", "CNn",),
        "huang": ("CNh", "CNul", "CNah", "CNng",),
        "hui": ("CNh", "CNul", "CNei", "CNib",),
        "hun": ("CNh", "CNul", "CNax", "CNn",),
        "huo": ("CNh", "CNul", "CNao",),
        "ji": ("CNj", "CNiy",),
        "jia": ("CNj", "CNil", "CNaa",),
        "jian": ("CNj", "CNil", "CNeh", "CNn",),
        "jiang": ("CNj", "CNil", "CNah", "CNng",),
        "jiao": ("CNj", "CNil", "CNah", "CNub",),
        "jie": ("CNj", "CNil", "CNee",),
        "jin": ("CNj", "CNih", "CNn",),
        "jing": ("CNj", "CNih", "CNng",),
        "jiong": ("CNj", "CNil", "CNoh", "CNng",),
        "jiu": ("CNj", "CNil", "CNoh", "CNub",),
        "ju": ("CNj", "CNvw",),
        "juan": ("CNj", "CNvl", "CNeh", "CNn",),
        "jue": ("CNj", "CNvl", "CNee",),
        "jun": ("CNj", "CNvl", "CNih", "CNn",),
        "ka": ("CNk", "CNaa",),
        "kai": ("CNk", "CNay", "CNib",),
        "kan": ("CNk", "CNae", "CNn",),
        "kang": ("CNk", "CNah", "CNng",),
        "kao": ("CNk", "CNah", "CNub",),
        "ke": ("CNk", "CNea",),
        "kei": ("CNk", "CNei", "CNib",),
        "ken": ("CNk", "CNax", "CNn",),
        "keng": ("CNk", "CNoe", "CNng",),
        "kong": ("CNk", "CNoh", "CNng",),
        "kou": ("CNk", "CNoh", "CNub",),
        "ku": ("CNk", "CNuw",),
        "kua": ("CNk", "CNul", "CNaa",),
        "kuai": ("CNk", "CNul", "CNay", "CNib",),
        "kuan": ("CNk", "CNul", "CNae", "CNn",),
        "kuang": ("CNk", "CNul", "CNah", "CNng",),
        "kui": ("CNk", "CNul", "CNei", "CNib",),
        "kun": ("CNk", "CNul", "CNax", "CNn",),
        "kuo": ("CNk", "CNul", "CNao",),
        "lv": ("CNl", "CNvw",),
        "lve": ("CNl", "CNvl", "CNee",),
        "lue": ("CNl", "CNvl", "CNee",),
        "la": ("CNl", "CNaa",),
        "lai": ("CNl", "CNay", "CNib",),
        "lan": ("CNl", "CNae", "CNn",),
        "lang": ("CNl", "CNah", "CNng",),
        "lao": ("CNl", "CNah", "CNub",),
        "le": ("CNl", "CNea",),
        "lei": ("CNl", "CNei", "CNib",),
        "leng": ("CNl", "CNoe", "CNng",),
        "li": ("CNl", "CNiy",),
        "lia": ("CNl", "CNil", "CNaa",),
        "lian": ("CNl", "CNil", "CNeh", "CNn",),
        "liang": ("CNl", "CNil", "CNah", "CNng",),
        "liao": ("CNl", "CNil", "CNah", "CNub",),
        "lie": ("CNl", "CNil", "CNee",),
        "lin": ("CNl", "CNih", "CNn",),
        "ling": ("CNl", "CNih", "CNng",),
        "liu": ("CNl", "CNil", "CNoh", "CNub",),
        "lo": ("CNl", "CNao",),
        "long": ("CNl", "CNoh", "CNng",),
        "lou": ("CNl", "CNoh", "CNub",),
        "lu": ("CNl", "CNuw",),
        "luan": ("CNl", "CNul", "CNae", "CNn",),
        "lun": ("CNl", "CNul", "CNax", "CNn",),
        "luo": ("CNl", "CNul", "CNao",),
        "ma": ("CNm", "CNaa",),
        "mai": ("CNm", "CNay", "CNib",),
        "man": ("CNm", "CNae", "CNn",),
        "mang": ("CNm", "CNah", "CNng",),
        "mao": ("CNm", "CNah", "CNub",),
        "me": ("CNm", "CNea",),
        "mei": ("CNm", "CNei", "CNib",),
        "men": ("CNm", "CNax", "CNn",),
        "meng": ("CNm", "CNoe", "CNng",),
        "mi": ("CNm", "CNiy",),
        "mian": ("CNm", "CNil", "CNeh", "CNn",),
        "miao": ("CNm", "CNil", "CNah", "CNub",),
        "mie": ("CNm", "CNil", "CNee",),
        "min": ("CNm", "CNih", "CNn",),
        "ming": ("CNm", "CNih", "CNng",),
        "miu": ("CNm", "CNil", "CNoh", "CNub",),
        "mo": ("CNm", "CNul", "CNao",),
        "mou": ("CNm", "CNoh", "CNub",),
        "mu": ("CNm", "CNuw",),
        "nv": ("CNn", "CNvw",),
        "nve": ("CNn", "CNvl", "CNee",),
        "nue": ("CNn", "CNvl", "CNee",),
        "na": ("CNn", "CNaa",),
        "nai": ("CNn", "CNay", "CNib",),
        "nan": ("CNn", "CNae", "CNn",),
        "nang": ("CNn", "CNah", "CNng",),
        "nao": ("CNn", "CNah", "CNub",),
        "ne": ("CNn", "CNea",),
        "nei": ("CNn", "CNei", "CNib",),
        "nen": ("CNn", "CNax", "CNn",),
        "neng": ("CNn", "CNoe", "CNng",),
        "ni": ("CNn", "CNiy",),
        "nian": ("CNn", "CNil", "CNeh", "CNn",),
        "niang": ("CNn", "CNil", "CNah", "CNng",),
        "niao": ("CNn", "CNil", "CNah", "CNub",),
        "nie": ("CNn", "CNil", "CNee",),
        "nin": ("CNn", "CNih", "CNn",),
        "ning": ("CNn", "CNih", "CNng",),
        "niu": ("CNn", "CNil", "CNoh", "CNub",),
        "nong": ("CNn", "CNoh", "CNng",),
        "nou": ("CNn", "CNoh", "CNub",),
        "nu": ("CNn", "CNuw",),
        "nuan": ("CNn", "CNul", "CNae", "CNn",),
        "nun": ("CNn", "CNul", "CNax", "CNn",),
        "nuo": ("CNn", "CNul", "CNao",),
        "o": ("CNao",),
        "ou": ("CNoh", "CNub",),
        "pa": ("CNp", "CNaa",),
        "pai": ("CNp", "CNay", "CNib",),
        "pan": ("CNp", "CNae", "CNn",),
        "pang": ("CNp", "CNah", "CNng",),
        "pao": ("CNp", "CNah", "CNub",),
        "pei": ("CNp", "CNei", "CNib",),
        "pen": ("CNp", "CNax", "CNn",),
        "peng": ("CNp", "CNoe", "CNng",),
        "pi": ("CNp", "CNiy",),
        "pian": ("CNp", "CNil", "CNeh", "CNn",),
        "piao": ("CNp", "CNil", "CNah", "CNub",),
        "pie": ("CNp", "CNil", "CNee",),
        "pin": ("CNp", "CNih", "CNn",),
        "ping": ("CNp", "CNih", "CNng",),
        "po": ("CNp", "CNul", "CNao",),
        "pou": ("CNp", "CNoh", "CNub",),
        "pu": ("CNp", "CNuw",),
        "qi": ("CNq", "CNiy",),
        "qia": ("CNq", "CNil", "CNaa",),
        "qian": ("CNq", "CNil", "CNeh", "CNn",),
        "qiang": ("CNq", "CNil", "CNah", "CNng",),
        "qiao": ("CNq", "CNil", "CNah", "CNub",),
        "qie": ("CNq", "CNil", "CNee",),
        "qin": ("CNq", "CNih", "CNn",),
        "qing": ("CNq", "CNih", "CNng",),
        "qiong": ("CNq", "CNil", "CNoh", "CNng",),
        "qiu": ("CNq", "CNil", "CNoh", "CNub",),
        "qu": ("CNq", "CNvw",),
        "quan": ("CNq", "CNvl", "CNeh", "CNn",),
        "que": ("CNq", "CNvl", "CNee",),
        "qun": ("CNq", "CNvl", "CNih", "CNn",),
        "ran": ("CNr", "CNae", "CNn",),
        "rang": ("CNr", "CNah", "CNng",),
        "rao": ("CNr", "CNah", "CNub",),
        "re": ("CNr", "CNea",),
        "ren": ("CNr", "CNax", "CNn",),
        "reng": ("CNr", "CNoe", "CNng",),
        "ri": ("CNr", "CNizh",),
        "rong": ("CNr", "CNoh", "CNng",),
        "rou": ("CNr", "CNoh", "CNub",),
        "ru": ("CNr", "CNuw",),
        "rua": ("CNr", "CNul", "CNaa",),
        "ruan": ("CNr", "CNul", "CNae", "CNn",),
        "rui": ("CNr", "CNul", "CNei", "CNib",),
        "run": ("CNr", "CNul", "CNax", "CNn",),
        "ruo": ("CNr", "CNul", "CNao",),
        "sa": ("CNs", "CNaa",),
        "sai": ("CNs", "CNay", "CNib",),
        "san": ("CNs", "CNae", "CNn",),
        "sang": ("CNs", "CNah", "CNng",),
        "sao": ("CNs", "CNah", "CNub",),
        "se": ("CNs", "CNea",),
        "sei": ("CNs", "CNei", "CNib",),
        "sen": ("CNs", "CNax", "CNn",),
        "seng": ("CNs", "CNoe", "CNng",),
        "sha": ("CNsh", "CNaa",),
        "shai": ("CNsh", "CNay", "CNib",),
        "shan": ("CNsh", "CNae", "CNn",),
        "shang": ("CNsh", "CNah", "CNng",),
        "shao": ("CNsh", "CNah", "CNub",),
        "she": ("CNsh", "CNea",),
        "shei": ("CNsh", "CNei", "CNib",),
        "shen": ("CNsh", "CNax", "CNn",),
        "sheng": ("CNsh", "CNoe", "CNng",),
        "shi": ("CNsh", "CNizh",),
        "shou": ("CNsh", "CNoh", "CNub",),
        "shu": ("CNsh", "CNuw",),
        "shua": ("CNsh", "CNul", "CNaa",),
        "shuai": ("CNsh", "CNul", "CNay", "CNib",),
        "shuan": ("CNsh", "CNul", "CNae", "CNn",),
        "shuang": ("CNsh", "CNul", "CNah", "CNng",),
        "shui": ("CNsh", "CNul", "CNei", "CNib",),
        "shun": ("CNsh", "CNul", "CNax", "CNn",),
        "shuo": ("CNsh", "CNul", "CNao",),
        "si": ("CNs", "CNiz",),
        "song": ("CNs", "CNoh", "CNng",),
        "sou": ("CNs", "CNoh", "CNub",),
        "su": ("CNs", "CNuw",),
        "suan": ("CNs", "CNul", "CNae", "CNn",),
        "sui": ("CNs", "CNul", "CNei", "CNib",),
        "sun": ("CNs", "CNul", "CNax", "CNn",),
        "suo": ("CNs", "CNul", "CNao",),
        "ta": ("CNt", "CNaa",),
        "tai": ("CNt", "CNay", "CNib",),
        "tan": ("CNt", "CNae", "CNn",),
        "tang": ("CNt", "CNah", "CNng",),
        "tao": ("CNt", "CNah", "CNub",),
        "te": ("CNt", "CNea",),
        "tei": ("CNt", "CNei", "CNib",),
        "teng": ("CNt", "CNoe", "CNng",),
        "ti": ("CNt", "CNiy",),
        "tian": ("CNt", "CNil", "CNeh", "CNn",),
        "tiao": ("CNt", "CNil", "CNah", "CNub",),
        "tie": ("CNt", "CNil", "CNee",),
        "tin": ("CNt", "CNih", "CNn",),
        "ting": ("CNt", "CNih", "CNng",),
        "tong": ("CNt", "CNoh", "CNng",),
        "tou": ("CNt", "CNoh", "CNub",),
        "tu": ("CNt", "CNuw",),
        "tuan": ("CNt", "CNul", "CNae", "CNn",),
        "tui": ("CNt", "CNul", "CNei", "CNib",),
        "tun": ("CNt", "CNul", "CNax", "CNn",),
        "tuo": ("CNt", "CNul", "CNao",),
        "wa": ("CNw", "CNaa",),
        "wai": ("CNw", "CNay", "CNib",),
        "wan": ("CNw", "CNae", "CNn",),
        "wang": ("CNw", "CNah", "CNng",),
        "wei": ("CNw", "CNei", "CNib",),
        "wen": ("CNw", "CNax", "CNn",),
        "weng": ("CNw", "CNoe", "CNng",),
        "wo": ("CNw", "CNao",),
        "wu": ("CNw", "CNuw",),
        "xi": ("CNx", "CNiy",),
        "xia": ("CNx", "CNil", "CNaa",),
        "xian": ("CNx", "CNil", "CNeh", "CNn",),
        "xiang": ("CNx", "CNil", "CNah", "CNng",),
        "xiao": ("CNx", "CNil", "CNah", "CNub",),
        "xie": ("CNx", "CNil", "CNee",),
        "xin": ("CNx", "CNih", "CNn",),
        "xing": ("CNx", "CNih", "CNng",),
        "xiong": ("CNx", "CNil", "CNoh", "CNng",),
        "xiu": ("CNx", "CNil", "CNoh", "CNub",),
        "xu": ("CNx", "CNvw",),
        "xuan": ("CNx", "CNvl", "CNeh", "CNn",),
        "xue": ("CNx", "CNvl", "CNee",),
        "xun": ("CNx", "CNvl", "CNih", "CNn",),
        "ya": ("CNy", "CNaa",),
        "yan": ("CNy", "CNae", "CNn",),
        "yang": ("CNy", "CNah", "CNng",),
        "yao": ("CNy", "CNah", "CNub",),
        "ye": ("CNy", "CNee",),
        "yi": ("CNy", "CNiy",),
        "yin": ("CNy", "CNih", "CNn",),
        "ying": ("CNy", "CNih", "CNng",),
        "yo": ("CNy", "CNao",),
        "yong": ("CNy", "CNoh", "CNng",),
        "you": ("CNy", "CNoh", "CNub",),
        "yu": ("CNy", "CNvw",),
        "yuan": ("CNy", "CNvl", "CNeh", "CNn",),
        "yue": ("CNy", "CNvl", "CNee",),
        "yun": ("CNy", "CNvl", "CNih", "CNn",),
        "za": ("CNz", "CNaa",),
        "zai": ("CNz", "CNay", "CNib",),
        "zan": ("CNz", "CNae", "CNn",),
        "zang": ("CNz", "CNah", "CNng",),
        "zao": ("CNz", "CNah", "CNub",),
        "ze": ("CNz", "CNea",),
        "zei": ("CNz", "CNei", "CNib",),
        "zen": ("CNz", "CNax", "CNn",),
        "zeng": ("CNz", "CNoe", "CNng",),
        "zha": ("CNzh", "CNaa",),
        "zhai": ("CNzh", "CNay", "CNib",),
        "zhan": ("CNzh", "CNae", "CNn",),
        "zhang": ("CNzh", "CNah", "CNng",),
        "zhao": ("CNzh", "CNah", "CNub",),
        "zhe": ("CNzh", "CNea",),
        "zhei": ("CNzh", "CNei", "CNib",),
        "zhen": ("CNzh", "CNax", "CNn",),
        "zheng": ("CNzh", "CNoe", "CNng",),
        "zhi": ("CNzh", "CNizh",),
        "zhong": ("CNzh", "CNoh", "CNng",),
        "zhou": ("CNzh", "CNoh", "CNub",),
        "zhu": ("CNzh", "CNuw",),
        "zhua": ("CNzh", "CNul", "CNaa",),
        "zhuai": ("CNzh", "CNul", "CNay", "CNib",),
        "zhuan": ("CNzh", "CNul", "CNae", "CNn",),
        "zhuang": ("CNzh", "CNul", "CNah", "CNng",),
        "zhui": ("CNzh", "CNul", "CNei", "CNib",),
        "zhun": ("CNzh", "CNul", "CNax", "CNn",),
        "zhuo": ("CNzh", "CNul", "CNao",),
        "zi": ("CNz", "CNiz",),
        "zong": ("CNz", "CNoh", "CNng",),
        "zou": ("CNz", "CNoh", "CNub",),
        "zu": ("CNz", "CNuw",),
        "zuan": ("CNz", "CNul", "CNae", "CNn",),
        "zui": ("CNz", "CNul", "CNei", "CNib",),
        "zun": ("CNz", "CNul", "CNax", "CNn",),
        "zuo": ("CNz", "CNul", "CNao",),
        "hm": ("CNh", "CNm",),
        "hng": ("CNh", "CNng",),
        "m": ("CNm",),
        "ng": ("CNng",),
        "kiu": ("CNk", "CNil", "CNoh", "CNub",),
        "fai": ("CNf", "CNay", "CNib",),
        "bia": ("CNb", "CNil", "CNaa",),
        "biu": ("CNb", "CNil", "CNoh", "CNub",),
        "gia": ("CNg", "CNil", "CNaa",),
        "giao": ("CNg", "CNil", "CNah", "CNub",),
        "giu": ("CNg", "CNil", "CNoh", "CNub",),
        "hia": ("CNh", "CNil", "CNaa",),
        "hiao": ("CNh", "CNil", "CNah", "CNub",),
        "hiu": ("CNh", "CNil", "CNoh", "CNub",),
        "kia": ("CNk", "CNil", "CNaa",),
        "kiao": ("CNk", "CNil", "CNah", "CNub",),
        "mia": ("CNm", "CNil", "CNaa",),
        "nia": ("CNn", "CNil", "CNaa",),
        "pia": ("CNp", "CNil", "CNaa",),
        "piu": ("CNp", "CNil", "CNoh", "CNub",),
        "be": ("CNb", "CNea",),
        "pe": ("CNp", "CNea",),
        "bou": ("CNb", "CNoh", "CNub",),
        "bua": ("CNb", "CNul", "CNaa",),
        "cua": ("CNc", "CNul", "CNaa",),
        "dua": ("CNd", "CNul", "CNaa",),
        "lua": ("CNl", "CNul", "CNaa",),
        "mua": ("CNm", "CNul", "CNaa",),
        "nua": ("CNn", "CNul", "CNaa",),
        "pua": ("CNp", "CNul", "CNaa",),
        "sua": ("CNs", "CNul", "CNaa",),
        "tua": ("CNt", "CNul", "CNaa",),
        "bue": ("CNb", "CNvl", "CNee",),
        "due": ("CNd", "CNvl", "CNee",),
        "gue": ("CNg", "CNvl", "CNee",),
        "hue": ("CNh", "CNvl", "CNee",),
        "kue": ("CNk", "CNvl", "CNee",),
        "mue": ("CNm", "CNvl", "CNee",),
        "pue": ("CNp", "CNvl", "CNee",),
        "sue": ("CNs", "CNvl", "CNee",),
        "tue": ("CNt", "CNvl", "CNee",),
        "wue": ("CNw", "CNvl", "CNee",),
        "bui": ("CNb", "CNul", "CNei", "CNib",),
        "jui": ("CNj", "CNul", "CNei", "CNib",),
        "lui": ("CNl", "CNul", "CNei", "CNib",),
        "mui": ("CNm", "CNul", "CNei", "CNib",),
        "nui": ("CNn", "CNul", "CNei", "CNib",),
        "pui": ("CNp", "CNul", "CNei", "CNib",),
        "qui": ("CNq", "CNul", "CNei", "CNib",),
        "xui": ("CNx", "CNul", "CNei", "CNib",),
        "buo": ("CNb", "CNul", "CNao",),
        "juo": ("CNj", "CNul", "CNao",),
        "muo": ("CNm", "CNul", "CNao",),
        "puo": ("CNp", "CNul", "CNao",),
        "quo": ("CNq", "CNul", "CNao",),
        "gie": ("CNg", "CNil", "CNee",),
        "hie": ("CNh", "CNil", "CNee",),
        "kie": ("CNk", "CNil", "CNee",),
        "tia": ("CNt", "CNil", "CNaa",),
        "tiu": ("CNt", "CNil", "CNoh", "CNub",),
    }
    
    @staticmethod
    def s2p(syl):
        if '(' in syl and ')' in syl:
            return tuple([f"EN{i}" for i in syl[1:-1].split('_')]) # (g_uh_d) -> ("ENg", "ENuh", "ENd')
        return Syllable.syl_dict.get(syl, ('sil',))
    
    @staticmethod
    def is_py(syl):
        syl = syl.lower()
        return syl == 'sil' or syl in Syllable.syl_dict
    
    @staticmethod
    def is_sil(syl):
        return syl == 'sil' or syl == 'sil0'


if __name__ == "__main__":
    
    print(Syllable.s2p('sil'))
    