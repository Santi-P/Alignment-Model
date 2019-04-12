# IBM 1 Alignment Model (lexical alignments) with naive noisy channel decoder
Santichai Pornavalai
1.4.19




align_demo.py comes with a few arguments


-i  to specify input file. This should be a tsv file with
 <ENGLISH SENTENCE> TAB <FOREIGN SENTENCE>
-t is the testing file. This can be any file as long as it is unicoded
-o specifies the output file
-s  saves the learned probabilities to binary

--load-weights    loads the lexical alignment probabilites
--load-index       loads word indices 
--load-lm           loads language Model
--interactive       goes into interactive mode

Example Usage

*python align_demo.py -o moliere_english.txt  -t data/moliere.txt  --load-weights trans_weights --load-index vocab_index --load-lm brown.lm --interactive*

load weights, indices, language models from file, translate a play by moliere save it to moliere_english.txt and go into interactive

*python align_demo.py -i e_f.txt - -o moliere_english.txt  -t data/moliere.txt  -s --interactive*

learn alignment probabilities. read and translate moliere. save learned weights to binary and go into interactive mode. 
    
    
    
As this is just a demo program, it isn't really fool proof so it might throw a few
errors here and there. It is meant to learn alignment probabilites and translate 
sentences from french to english. (it can be reversed as well but a language model for french
would be needed)

There are two main classes used: the alignment model and language model. 
The alignment model uses EM training (so called IBM-1. Nothing special,no fertility
or reordering and other fancy stuff). However this algorithm converges at a global maximum. It also
takes very few iterations to converge (2-3).
It also contains a naive noisy channel decoder which only looks at the previous word.
The decoder works by fetching top N candidate translation words and determines the most 
likely one given the previously translated word. This is a rather naive approach. K-best
viterbi would be the ideal candidate for this task. The lack of a syntax model also greatly affects the
fluency of the translated sentences.

Language model is a simple bigram interpolated language model.
If the language model is not explicitly loaded by the Alignment class, it will create and save the language
model to binary. 

The largest limiting factor of this program is the space consumption which is determined by the vocabulary
size of both languages. Since it involves iterating back and forth and repetitive renormalizing, sparse
array solution would have to be very specific and would definitely affect training performance. To make this
some how useable tried experimenting with half-precision floats (numpy.float16). This didn't affect performance
but training speed took a slight hit. I did some research and it may be because on 
the machine leve, they have to be converted to normal floats before arithmetic operations can be performed.
This can save a lot of space however so I set it as a default. 

Testing is done manually by entering a few french words I learned from high school. It seems to work well
for nouns etc but struggles with syntactically functional words. Idioms and such are bad as expected.

There is a bug/unsolved issue which appears when trying to translate an unknown word. I simply just set it
to return the first element in the probability array which is the word "go". 

