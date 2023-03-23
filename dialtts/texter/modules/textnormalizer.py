# coding: utf-8

import os, sys, re

from opencc import OpenCC

from dialtts.texter import Config
from dialtts.texter.utils import DBC2SBC
from dialtts.texter.utils import Syllable


class Num2Wrd(object):
	def __init__(self):
		self.DIG = ('零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十')
		self.ORD = ('', '十', '百', '千')
		self.MAG = ('', '万', '亿', '兆', '京', '垓', '', '', '', '', '', '', '', '', '', )
		self.DOT = '点'
		self.NEG = '负'
		self.ZEOR = self.DIG[0]

	def __call__(self, nums, cnf="m"):
		nums = DBC2SBC(nums).strip()
		
		if nums == '':
			return ''
		
		if cnf == 'm':
			return self._num2word_m(nums)
		elif cnf == 'i':
			