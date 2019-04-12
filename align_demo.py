# Demo driver code for alignment project
# based on boilerplate from https://www.tutorialspoint.com/python/python_command_line_arguments.htm
# modified by Santi(chai) Pornavalai
# 31.3.2019
# tested on python 3.7.2

import sys, getopt
from alignment_model import AlignmentModel
from language_model import Ngrams

# commands

#  python align_demo.py -o translated.txt  -t french_test.txt  --load-weights trans_weights --load-index vocab_index --load-lm brown.lm --interactive
#  python align_demo.py -i e_f.txt -o translated.txt -t french_test.txt --interactive

def main(argv):
 
    inputfile = ''
    testpath = ''
    outputfile = ''
    iterations = 5

    interactive = False
    save_to_bin = False

    idx_fname = ""
    weights_fname = ""
    lm_name = ""

    try:
        opts, args = getopt.getopt(argv,"h:o:t:k:i:s",["ifile=","ofile=","iter=","tfile=", \
                                        "save","load-weights=","load-index=", "load-lm=","interactive"])
    except getopt.GetoptError:
        print ("""ERROR!! General USAGE: align_demo.py -i <PATH TO TRAIN FILE> -t <TEST FILE NAME> -o <RESULT FILE NAME>
                   for other options rtm """)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('align_demo.py -i <PATH TO TRAIN FILE> -o <RESULT FILE NAME> -t')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-k", "--iter"):
            iterations = arg
        elif opt in ("-t","--tfile"):
            testpath = arg

        elif opt in ("--save","-s"):
            save_to_bin = True
        elif opt in ("--load-index",):
            idx_fname = arg
        elif opt in ("--load-weights",):
            weights_fname = arg
        elif opt in ("--interactive",):
            interactive = True
        elif opt in ("--load-lm",):
            lm_name = arg

        
    translation_model = AlignmentModel()
    if len(inputfile) == 0:

        if len(weights_fname) > 0 and len(idx_fname) > 0:
            translation_model.load_weights(weights_fname,idx_fname)
            translation_model.load_lm(lm_name)
        else:
            print("No weight/index or train file provided")
            sys.exit(3)

    else:
        translation_model.train(inputfile,iterations, True)
        translation_model.build_save_lm("brown.lm")

    if len(testpath) > 0:
        results = translation_model.translate_all(testpath,print_out = False)
        with open(outputfile, "w") as testf:
            for line in results:
                testf.write(line  + "\n" )
    else:
        print(" no testing")

    if save_to_bin:
        wfile_name = outputfile + ".weights"
        indfile_name = outputfile + ".index"
        translation_model.save_weights(wfile_name, indfile_name)

    if interactive:
        while True:
            in_string = input("enter a sentence in french: ")
            print(translation_model.decode(in_string))


if __name__ == "__main__":
    main(sys.argv[1:])
