"""
This file loads the data

Notes:
If you wish to specify a different file to be read you must follow the format of the wikitext files
included in content, that is you should have three data sets to be read a test, train, and valid set.
Ideally it would also follow the name convention name.{}.tokens although this isn't entirely necessary,
also ideally you should store them in the content directory.
"""


def load(fileset):
    """
    Definition that loads the data

    Notes:
    This function splits the data into three sets test, train, and valid. You should ideally provide
    three datafiles for this or otherwise specify in some way how you wish your data to be split

    Parameter:
    -fileset: path to the fileset you wish to load

    Output:
    -data: dict, a dictionary object with three keys: test, train, and valid, where each keys holds the
    respective tokens from the data files located in content, in our base case the wikitext language
    modeling dataset
    -vocab: list, a list of each unique word type in the training set
    """
    data = {'test': '', 'train': '', 'valid': ''}

    try:
        for data_split in data:
            fname = fileset.format(data_split)
            with open(fname, encoding="utf8") as f:
                data[data_split] = f.read().lower().split()
    except IOError:
        print('Fileset does not exist')
        exit(2)

    vocab = list(set(data['train']))
    return data, vocab


def print_data(data, vocab):
    """
    Definition to print the data

    Notes:
    Prints the size and first ten words of each data set and the vocabulary

    Parameters:
    -data: dict, the dictionary created in the above definition
    -vocab: list, the vocab created in the above definition
    """
    print('size of training data: ' + str(len(data['train'])))
    print('first ten words in train : %s ...' % data['train'][:10])
    print('size of test data: ' + str(len(data['test'])))
    print('first ten words in test : %s ...' % data['test'][:10])
    print('size of valid data: ' + str(len(data['valid'])))
    print('first ten words in valid : %s ...' % data['valid'][:10])
    print('size of vocabulary: ' + str(len(vocab)))
    print('first 10 words in vocab: %s' % vocab[:10])
    print('\n')