import time
from collections import defaultdict, Counter
import numpy as np
from math import log, exp


class LanguageModel:
    """
    Class used to represent a language model

    Attributes:
    -order: int, the order of the model (n in ngram model)
    -model: dict, the model itself, a dictionary where the key is the history and the value is a dictionary
    with the next word as keys and their respective probability distributions as values, which is computed
    using the maximum likelihood estimate from the training data

    Notes:
    -Each dictionary stores backoff probabilities also, that is for a bigram (ngram of order 2) model we
    also store the unigram (ngram of order 1) model probabilities as well
    -Each key is a single string where the words that form the history are concantated using spaces. Given
    a key its corresponding value is a dictionary where each word type in the vocabulary is associated with
    its corresponding probability of appearing after the key

    Example:
    For a trigram model (ngram of order 3) the entry for the history (key) of 'w1 w2' (word1 and word2)
    will look like
    lm.model['w1 w2'] = {'w0': 0.001, 'w1': 1e-5, 'w2': 1e-6, 'w3': 2e-7, ...}
    in this example lm.model['w1'] and lm.model[' '] is also stored (the bigram and unigram models
    respectively)
    """
    def __init__(self, order):
        """
        initiates the model

        Parameters:
        -order: int, the order of the model
        """
        self.order = order
        self.model = {}

    def train(self, data):
        """
        Definition to train the model

        Parameters:
        -data: list, the data with which the model is to be trained on
        """
        # Pad based on order size
        order = self.order
        print('Beginning training of language model of order ', order)
        st = time.time()
        order -= 1
        data = ['<S>'] * order + data
        lm = defaultdict(Counter)
        for i in range(len(data) - order):

            # concatenate list of words into string
            words = ' '.join(data[i:i + order])

            # While loop for back off
            while words:

                # Check if words have been seen before, if so update counter
                if words in lm.keys():
                    count = lm[words]
                    count.update(Counter([data[i + order]]))

                # If string has not been seen before create new dictionary entry for it
                else:
                    lm[words] = Counter([data[i + order]])

                # Remove the first word from the string
                x = words.split()
                x.pop(0)
                words = ' '.join(x)

            # Perform same if-else for empty string
            if words in lm.keys():
                count = lm[words]
                count.update(Counter([data[i + order]]))

            else:
                lm[words] = Counter([data[i + order]])

        lm2 = {}
        # Convert counts to probabilities
        for x, y in lm.items():
            i = sum(y.values())
            for k, v in y.items():
                lm2[k] = v / i
            self.model[x] = lm2
            lm2 = {}

        et = time.time()
        ex_time = et - st
        print('Finished training model in: ', ex_time, ' seconds\n')

    def generate(self, vocab, context="he is the", num_tok=25):
        """
        Definition to generate text

        Parameters:
        -vocab: list, the vocab object created by load_data, a list of unique word types in the data set
        computed during load_data
        -context: string, the input context string you wish to condition your language model on, should be
        a space seperated string of tokens, default is "he is the"
        -num_tok: int, the number of tokens you wish to be generated following the input context

        Output:
        -Prints the generated text to the console as a space seperated string

        Notes:
        generate_text works by searching the context in the language model and splitting the words and their
        respective probabilities into two separate lists. It then uses numpy random choice to sample one word
        given the list of words and their probabilities which it then adds to the variable out. After appending
        the generated word it removes the first word and repeats the search for the new set of words.
        """

        # If context has more tokens than the order of lm,
        # generate text that follows the last (order-1) tokens of the context
        # and store it in the variable `history`
        order = self.order
        lm = self.model
        order -= 1
        history = context.split()[-order:]

        # `out` is the list of tokens of context
        # generated tokens are appended to this list
        out = context.split()

        # Add two characters not in vocab but in search keys to the vocab then check
        # that specified search context is in vocab
        spec_char = ['', '<S>']
        spec_vocab = vocab + spec_char
        if not all(item in spec_vocab for item in history):
            print("Specified context not in vocab cannot be searched")
            exit(2)

        for i in range(num_tok):
            # If using a bigram model (order will be reduced to zero in an earlier step for unigrams)
            # then search history will always be blank
            if order == 0:
                history = ['']
            search = ' '.join(history)
            dist = lm[search]
            # Splitting dictionary entries into keys (words) and values (their respective probabilities)
            words = [k for k in dist.keys()]
            probs = [v for v in dist.values()]
            next_word = np.random.choice(words, size=1, p=probs)
            x = ' '.join(next_word)
            out.append(x)
            history.append(x)
            history.pop(0)

        final_string = ' '.join(out)
        print('Generating ', num_tok, ' tokens following given context "' + context + '" with order ', self.order, ':\n')
        print(final_string, '\n')

    def compute_perplexity(self, data, vocab):
        """
        Definition to compute perplexity of the model

        Parameters:
        -data: list, this should be the test data, this is generated in the load_data file, a list object storing
        all the tokens in the test set
        -vocab: list, the vocab object created by load_data, a list of unique word types in the data set computed
        during load_data
        -order: int, the order of the language model

        Output:
        -exp_sum: int, the perplexity of the test data, exp_sum is the exponential sum which is computed from the
        normalized sum (normal_sum) which itself is computed from the log sum (log_sum)

        Notes:
        This function computes the perplexity (inverse probability) of the test data compared to the language
        model trained on train data log and exponent calculations are used to address underflow. Explaining
        the computations entirely would be too much for this section, for more info you can refer to pages
        8 and 9 of this link https://web.stanford.edu/~jurafsky/slp3/3.pdf. For establishing a base case, using
        the datasets provided, the perplexity of the unigram, bigram, trigram, and 4-gram language
        models should be around 795, 203, 141, and 130 respectively for our model.

        If the history is not in the lm object, we back-off to (n-1) order history to check if it is in lm.
        If no history can be found, we just use 1/|V| where |V| is the size of the vocabulary.
        """

        order = self.order
        lm = self.model
        print('Computing perplexity for language model of order ', order, '\n')
        # pad according to order
        order -= 1
        data = ['<S>'] * order + data
        log_sum = 0
        for i in range(len(data) - order):
            h, w = ' '.join(data[i: i + order]), data[i + order]
            x = 1

            # While h is not a key in the model we move our scope over until it is
            while h not in lm.keys():
                h = ' '.join(data[i + x:i + order])
                x += 1

            # While w is not in the keys for h we move our scope over until it is or the order is surpassed
            while w not in lm[h].keys() and x <= order:
                h = ' '.join(data[i + x:i + order])
                x += 1

            # Case where no history is found
            if w not in lm[h].keys():
                logx = log(1 / len(vocab))
                log_sum += logx
                print('exception: ' + w)

            # Case where the history is found
            else:
                logx = log(lm[h][w])
                log_sum += logx

        normal_sum = (-1 / len(data)) * log_sum
        exp_sum = exp(normal_sum)
        return exp_sum
