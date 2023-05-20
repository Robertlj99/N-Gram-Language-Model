<h2>N-Gram Language Model</h2>

Python implementation of an N-Gram Language Model.
With text generation and perplexity computation.

<h3>Data</h3>

- wiki.test.tokens
- wiki.train.tokens
- wiki.valid.tokens
- tokenized_data.pkl
- trained_lm.pkl

All data is contained in the content directory provided.
The train, test, and valid data comes from the wikitext dataset.
The two .pkl files are pickle files used to store and load the tokenized
data and trained language model, you do not need those to start
as the program will automatically make them for you if they are not there.
It is important to have the content directory located exactly as it is
here if you wish to run this exactly as I have it, since the program automatically
loads the .pkl files from content. Otherwise if you wish to load
your own data you can specify the file path during command line interface

<h3>Files</h3>

- load_data.py
- main.py
- ngram_lm.py

The meat of the program is in the ngram_lm file which stores the class and features
for the N-Gram model itself. If you wish to load and or tokenize your data differently you can
alter the load_data file but be sure to also check that when the data is called
in the program it is being correctly called.

<h3>How to use</h3>
Firstly you should load the data you can do this by specifying the load argument in
the command line like this

```commandline
python main.py load [--file <fileset>]
```
The file argument is optional, if you do not specify an argument
the program will prompt you to either enter one or press enter for the default path
which is 'content/wiki.{}.tokens'. This path will load each of the provided
datasets provided in content. The expected output for following the default
option is as follows, note the first ten words of vocab will change each time you run this
command but otherwise everything should be the same.

```commandline
Please enter fileset you wish to be loaded or leave empty for default 
Using default wikitext fileset included in content directory

Data saved to 'content/tokenized_data.pkl'

size of training data: 2051910
first ten words in train : ['=', 'valkyria', 'chronicles', 'iii', '=', 'senjō', 'no', 'valkyria', '3', ':'] ...
size of test data: 241211
first ten words in test : ['=', 'robert', '<unk>', '=', 'robert', '<unk>', 'is', 'an', 'english', 'film'] ...
size of valid data: 213886
first ten words in valid : ['=', 'homarus', 'gammarus', '=', 'homarus', 'gammarus', ',', 'known', 'as', 'the'] ...
size of vocabulary: 28911
first 10 words in vocab: ['transfusions', 'threads', 'rework', 'visited', 'recent', 'ring', 'pérez', 'lexington', 'sbf', 'flaw']

```

Next you should train a model with the following command

```commandline
python main.py train [--order <#>]
```
You can specify the order of the model you wish to train with the optional command
or leave it as default which is order 3.

The expected output is as follows if following the default options

```commandline
Please enter the order of the model you would like to train or leave empty for default 
Beginning training of language model of order  3
Finished training model in:  18.874913215637207  seconds

Model saved to 'content/trained_lm.pkl'
```
After this you can generate text to the command line like so

```commandline
python main.py generate [--tokens <#>] [--context <string>]
```
Tokens is used to select how many tokens you wish to generate and context is used 
to specify what string you would like to use to prompt the model. Note that if you 
enter a context it should not have parentheses around it. The following is an example
without using any flags but specifying the number of tokens and context when prompted.

```commandline
Please enter the number of tokens you wish to generate or leave empty for default 30
Please enter the context you wish to generate from or leave empty for default: he is the
Generating  30  tokens following given context "he is the" with order:  3 

he is the number of people aware of the sensitive political climate of the late 19th century and continued into the early 1970s , he held coaching roles with the new zealand government set
```
Lastly you can compute perplexity by entering the following

```commandline
python main.py perplexity
```
And if following the default the expected response is

```commandline
Computing perplexity for language model of order  3 

141.21882352172221
```
Authored by Robert Johnson
