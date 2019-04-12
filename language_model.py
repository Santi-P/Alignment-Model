# Naive Interpolated Bigram Language Model
# intended for use with translation model decoding
# Santi(chai) Pornavalai 
# 31.3.19
# tested with Python 3.7.2


import re
from collections import Counter
import numpy as np 
from sys import getsizeof

class Ngrams:
	""" Naive Interpolated Bigram Language Model
		contains minimal features and smoothed by interpolating
		with unigrams and small unknown word number. Can take 
		unformatted text files. e.g the brown corpus provided. 
    """
	def __init__(self,fname):
		self.lang_mod, self.uni = self.ngrams(fname)


	def ngrams(self, fname):
		""" main language model builder"""
		text = []
		with open(fname,"r") as f:
			for line in f:
				text.extend(line.split())
		uni = Counter(text)
		bigram = zip(text, text[1:])
		lm = Counter(bigram)
		#lm = Counter(bigram)
		#print(lm["there","is"])
		print("normalizing language model")

		sum_uni = sum(uni.values())
		for key,val in lm.items():
			w1,w2 = key
			lm[w1,w2] = val/uni[w1]
		for w in uni.keys():
			uni[w] /= sum_uni


		return lm, uni



	def get_prob(self,w1, w2, unigram_weight = 0.1):
		""" get interpolated probability """
		# prev word not found
		unknown_weight = 0.00000000000000001

		bigram_weight = 1 - unigram_weight - unknown_weight
		bigram_prob = 0.0
		unigram_prob = 0.0
		if self.lang_mod.get((w1,w2)) != None:
			bigram_prob = self.lang_mod[w1,w2]
			
		if self.uni.get(w2) != None:
			unigram_prob = self.uni[w2]
		
		
		return bigram_weight * bigram_prob + unigram_weight * unigram_prob  + unknown_weight

if __name__ == "__main__":
	lang = Ngrams("small_borwn.txt")
	#print(lang.uni)
	#print(lang.lang_mod)
	
	while True:
		in_string = input("enter two words: ")
		prev, curr = in_string.split()
		print(lang.get_prob(prev,curr))
	