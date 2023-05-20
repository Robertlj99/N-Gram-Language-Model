"""N-Gram Language Model.

Usage:
    main.py load [--file <fileset>]
    main.py train [--order <#>]
    main.py generate [--tokens <#>] [--context <string>]
    main.py perplexity
    main.py help

Options:
    --file <fileset>     Dataset file path
    --order <#>          Order of the model you'd like to train (N)
    --context <string>   Space seperated string you'd like to use to prompt text generation
    --tokens <#>         Number of tokens you wish to be generated after the context
    --help               Show this screen
"""

import load_data
from ngram_lm import LanguageModel
import pickle
from docopt import docopt


def load_file(filename):
    data, vocab = load_data.load(filename)
    f = open('content/tokenized_data.pkl', 'wb')
    pickle.dump(data, f)
    pickle.dump(vocab, f)
    f.close()
    print("Data saved to 'content/tokenized_data.pkl'\n")
    load_data.print_data(data, vocab)


def load_pickled_data():
    try:
        f = open('content/tokenized_data.pkl', 'rb')
    except IOError:
        print('No data loaded, please load data first')
        exit(2)
    data = pickle.load(f)
    vocab = pickle.load(f)
    f.close()
    return data, vocab


def create_model(order):
    data, vocab = load_pickled_data()
    lm = LanguageModel(order)
    lm.train(data['train'])
    f = open('content/trained_lm.pkl', 'wb')
    pickle.dump(lm, f)
    print("Model saved to 'content/trained_lm.pkl'\n")
    f.close()


def load_pickled_lm():
    try:
        f = open('content/trained_lm.pkl', 'rb')
    except IOError:
        print("No trained model saved, please train a model first")
        exit(2)
    lm = pickle.load(f)
    f.close()
    return lm


def generate(context="he is the", num_tok=25):
    data, vocab = load_pickled_data()
    lm = load_pickled_lm()
    lm.generate(vocab, context, num_tok)


def perplexity():
    data, vocab = load_pickled_data()
    lm = load_pickled_lm()
    x = lm.compute_perplexity(data['test'], vocab)
    print('perplexity = ', x, '\n')


def main(args):
    if args['load']:
        if not args['--file']:
            filename = input("Please enter fileset you wish to be loaded or leave empty for default ")
            if not filename:
                print('Using default wikitext fileset included in content directory\n')
                filename = 'content/wiki.{}.tokens'
        else:
            filename = args['--file']

        load_file(filename)

    if args['train']:
        if not args['--order']:
            order = input('Please enter the order of the model you would like to train or leave empty for default ')
            if not order:
                print("Using default order of 3")
                order = 3
        else:
            order = args['--order']

        create_model(int(order))

    if args['generate']:
        if not args['--tokens']:
            num_tok = input("Please enter the number of tokens you wish to generate or leave empty for default: ")

            if not num_tok:
                print("Generating default number of tokens (25)\n")
                num_tok = 25
        else:
            num_tok = args['--tokens']

        if not args['--context']:
            context = input("Please enter the context you wish to generate from or leave empty for default: ")
            if not context:
                print("Using default context ('he is the')\n")
                context = 'he is the'
        else:
            context = args['--context']

        generate(context, int(num_tok))

    if args['perplexity']:
        perplexity()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    arguments = docopt(__doc__)
    main(arguments)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
