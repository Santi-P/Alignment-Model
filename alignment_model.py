# IBM-1 Alignment Model
# iteratively learns word alignments
# includes a naive noisy channel decoder
# Santi(chai) Pornavalai 
# 31.3.19
# tested with Python 3.7.2

# math stuff
import numpy
# serialized saving
import pickle
# tokenization
import re
# language model 
from language_model import Ngrams

class AlignmentModel:
	""" Implementation of IBM Model 1. Uses EM to learn lexical/alighment
	 probabilities. Language model is included to turn this into a noisy-channel
	 model. A simple decode algorithm is used. External scripts for evaluation
	 Per convention, variables with e mark target language and f for source, or
	 noisy model """
	def __init__(self):

		# Vocabulary indices
		self.__e_vocab = dict()
		self.__f_vocab = dict()
		# inverse mapping of the above
		self.ind2word_f = dict()
		self.ind2word_e = dict()

		# main lexical alignment weights 
		self.trans_prob =[]

		# sentinels for keeping 
		self.__num_e = int()
		self.__num_f = int()

		# 
		self.data_type = numpy.float16
		self.lang_mod = []

		self.punct_filter =re.compile(r'[^\w]')

	def build_save_lm(self,fname):
		""" builds N-gram language model and dumps to binary """
		with open(fname,"wb") as langfile:
			self.lang_mod =  Ngrams("data/brown.txt")
			pickle.dump(self.lang_mod, langfile)
		
	def load_lm(self,fname):
		""" loads N-gram language model from binary file"""
		print("loading language model")
		with open(fname,"rb") as langfile:
			self.lang_mod = pickle.load(langfile) 

	def translate_sentence(self, in_sent):
		""" translates sentence by simply replacing words """
		in_sent = in_sent.split()
		return " ".join([self.translate_word(w) for w in in_sent])



	def decode(self, in_sent):
		""" A naive noisy channel decoder
			fetches a few candidate translations for a target word
			selects the word which has the
			 highest probability given previous word. 
			 This can be highly improved upon by using a variation of the viterbi algorithm
			 e.g viterbi with beamsearch.

			 takes sentence string as argument and returns translated sentence as string 

			  """
		#print("translating :", in_sent)
		in_sent = (self.punct_filter.sub(' ',in_sent)).lower()
		in_sent = in_sent.split()
		if len(in_sent) < 2:
			if len(in_sent) > 0:
				return self.translate_word(in_sent[0])
			else:
				return ""
		results = [self.translate_word(in_sent[0])]
		# go through words in sentence
		for i in range(1,len(in_sent)):
			prev_word = results[i-1]
			curr_word = in_sent[i]
			# get 10 candidate translations
			# this number can be varied

			candidates, scores = self.get_n_best(curr_word,5)
			# candidate with highest probability
			tmp_cand = [self.lang_mod.get_prob(prev_word,w)*score for w,score in zip(candidates,scores)]
			enum = list(enumerate(tmp_cand))

			best = max(enum, key = lambda x: x[1])[0]

			results.append(candidates[best])

		return " ".join(results)

	def lazy_reader(self, fname):
		""" A corpus iterator. Takes a tsv file
			in the form 

			ENGLISH SENTENCE 1 <TAB>  FRENCH SENTENCE 1
			ENGLISH SENTENCE 2 <TAB>  FRENCH SENTENCE 2

			performs minimal tokenization and sets all the words to lowercase
			"""
		# native f to e translation
		with open(fname,"r") as f:
			pattern = re.compile(r'[^\w]')
			for line in f:
				# verify input later
				foreign , native= line.split("\t")

				native = (pattern.sub(' ',native)).lower()
				foreign = (pattern.sub(' ',foreign)).lower() 
				yield  native.strip().split(), foreign.strip().split()


	def trans_prob_getter(self, e_word, f_word):
		""" getter for translation probability"""
		ind_e = self.__e_vocab[e_word]
		ind_f = self.__f_vocab[f_word]
		return self.trans_prob[ind_e][ind_f], ind_e, ind_f
	
	# e to f  

	def get_max(self,word):
		""" get the best translation of word"""
		try:
			ind = self.__e_vocab[word]
		except KeyError:
			print("word not found")
			return 0
		res = numpy.argmax(self.trans_prob[ind])
		return res
	
	def get_n_best(self,word, n = 10):
		""" get k best for a given target word"""
		# argpartition used to avoid sorting the whole array
		# this approach is O(n) worst case
		try: 
			ind = self.__e_vocab[word]
		except KeyError:
			ind = 0
			#raise(ValueError("unknown vocabulary"))
		res = numpy.argpartition(self.trans_prob[ind],-n)
		scores = [self.trans_prob[ind][s] for s in res[-n:]]
		return [self.ind2word_f[word] for word in res[-n:]], scores

	def translate_word(self,word):
		ind_of_best = self.get_max(word)
		return self.ind2word_f[ind_of_best]

	# f to e

	### wrapper functions but for the other way around

	def get_max_e(self,word):
		ind = self.__f_vocab[word]
		res = numpy.argmax(self.trans_prob.T[ind])
		return res
	
	def get_n_best_e(self,word, n = 10):
		# argpartition used to avoid sorting the whole array
		# this approach is O(n) worst case

		ind = self.__f_vocab[word]
		res = numpy.argpartition(self.trans_prob.T[ind],-n)
		return [self.ind2word_f[word] for word in res[-n:]]



	def translate_word_e(self,word):
		ind_of_best = self.get_max_e(word)
		return self.ind2word_e[ind_of_best]


	
	def save_weights(self, tm_fname, idx_fname):
		""" dumps translation probabilities and 
		indices into two files"""
		with open(tm_fname, "wb") as w_file:
			numpy.save(w_file, self.trans_prob)
		with open(idx_fname, "wb") as idx_file:
			pickle.dump((self.ind2word_e, self.ind2word_f, self.__e_vocab, self.__f_vocab),idx_file)

	def load_weights(self, tm_fname, idx_fname):
		""" loads translation probabilities and 
			indices from two files"""

		with open(tm_fname, "rb") as w_file:
			self.trans_prob = numpy.load(w_file)

		with open(idx_fname, "rb") as idx_file:
			self.ind2word_e, self.ind2word_f, self.__e_vocab, self.__f_vocab = pickle.load(idx_file)

	def train(self, fname, iterations = 3, debug = False):
		""" IBM Model 1 alignment model. Trains lexical alignments using
		expectation maxization. Faithful implementation of the pseudo-code as seen 
		in the slides from P. Koehn
		
		This algorithm converges very quickly (2-3 iterations). This can use up a lot of memory. To 
		solve this I recommend using half-precision floats (np.float16). This takes 1/4 less memory compared
		to numpy default doubles. There is a trade off in terms of speed but it is relative negligeble. 

		The debug option prints the sum of the element-wise difference between previous iterations.
		The smaller the better. 
		
		"""
		# train t(e|f ) or corpus
		# collect counts and build array
		count_e_words = 0
		count_f_words = 0

		for e_sent, f_sent in self.lazy_reader(fname):
			for e_token in e_sent:
				if self.__e_vocab.get(e_token) == None:
					self.__e_vocab[e_token] = count_e_words
					self.ind2word_e[count_e_words] = e_token
					count_e_words += 1

			for f_token in f_sent:
				if self.__f_vocab.get(f_token) == None:
					self.__f_vocab[f_token] = count_f_words
					self.ind2word_f[count_f_words] = f_token
					count_f_words += 1

		uniform = 1 / (count_e_words + count_f_words -2 )
		dimensions = (count_e_words, count_f_words)

		print(count_e_words)
		print(count_f_words)
		self.trans_prob = numpy.full(dimensions, uniform, dtype=self.data_type)
		

		for it in range(iterations):
			print("starting iteration: ", it)
			count_ef = numpy.zeros(dimensions, dtype=self.data_type)
			total_f = numpy.zeros((dimensions[1],), dtype=self.data_type)

			for e_sent, f_sent in self.lazy_reader(fname):                
				sent_total = {e : 0 for e in set(e_sent)}
				for e_token in e_sent:
					for f_token in f_sent:
						sent_total[e_token] += self.trans_prob_getter(e_token,f_token)[0]

				for e_token in e_sent:
					for f_token in f_sent:
						#ind_e = self.__e_vocab[e_token]
						#ind_f = self.__f_vocab[f_token]

						tp, ind_e, ind_f =  self.trans_prob_getter(e_token,f_token)
						val = tp / sent_total[e_token]
						count_ef[ind_e][ind_f] += val
						total_f[ind_f] += val 

			print("normalizing")
			if debug:
				print("convergence")
				print(numpy.abs(numpy.sum(self.trans_prob - count_ef / total_f ,dtype = self.data_type)))
			self.trans_prob = count_ef / total_f

			
	def translate_all(self,fname, print_out = False):
		""" translates all sentences in a given file"""
		results = []
		with open(fname,"r") as f:
			for line in f:
				translation = self.decode(line)
				if print_out:
					print(translation)
				results.append(translation)
		return results

	def verify(self):
		""" debugging function for checking translation probabilities
			by summing up the rows and checking if equal to 1
			using half-precision floats often causes
			a lot of rows to not add up to 1 correctly.
			In practice this doesn't make that big of a difference"""
		width, length = self.trans_prob.shape
		print(width,length)
		
		for i in numpy.sum(self.trans_prob,dtype=self.data_type,axis=0):
			#if not numpy.isclose(i,1.0,rtol=1e-2):
			print(i)



if __name__ == "__main__":
	model_1 = AlignmentModel()
	model_1.build_save_lm("langmod_en")
	model_1.train("data/e_f.txt",iterations = 3)

	#print(model_1.get_n_best("large"))
	#print(model_1.get_n_best("je",2))
	model_1.load_weights("trans_weights", "vocab_index")
	model_1.translate_all("french_test.txt",True)
	#model_1.verify()
	#model_1.load_lm("langmod_en")
	#print(model_1.get_n_best("bois", 10))
	#model_1.save_weights("trans_weights_1", "vocab_index_1")
	#print(model_1.decode("elle est une chatte"))
	#print(model_1.decode())
	#print(model_1.decode("je suis une baguette"))
	#print(model_1.decode("je bois du vin"))


	while True:
		in_string = input("enter a sentence in french: ")
		print(model_1.decode(in_string))
		#print(model_1.translate_sentence(in_string))

	#print(model_1.decode("je ne puis le dire"))

	#model_1.save_weights("tf_weights", "word_idx")
	#print(model_1.trans_prob)
