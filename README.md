# online cognacy identification

This repository accompanies the paper **Fast and unsupervised methods for
multilingual cognate clustering** by Rama, Wahle, Sofroniev, and Jäger.


## setup

```bash
# clone this repository
git clone https://github.com/evolaemp/online_cognacy_ident
cd online_cognacy_ident

# you do not need to create a virtual environment if you know what you are
# doing; remember that the code is written in python3
virtualenv meta/venv
source meta/venv/bin/activate

# install the dependencies
# it is important to use the versions specified in the requirements file
pip install -r requirements.txt

# check the unit tests
python -m unittest discover online_cognacy_ident
```


## usage

```bash
# activate the virtual env if it is not already
source meta/venv/bin/activate

# use run.py to invoke the commands
python run.py --help

# run the pair hidden markov model algorithm
python run.py phmm path/to/dataset.tsv -o output.tsv

# run the pointwise mutual information algorithm
python run.py pmi path/to/dataset.tsv -o output.tsv

# evaluate the output
python eval.py path/to/dataset.tsv output.tsv
```

A dataset should be in csv format. You can specify the csv dialect using the
`--dialect` option, possible values are `excel`, `excel-tab`, and `unix`. If
this is omitted, the script will try to guess the dialect by looking at the file
extension.

A dataset should have a header with at least the following columns: `doculect`
or `language`, `concept` or `gloss`, and `asjp` or `transcription`. Column name
detection is case-insensitive.


## license

The source code is published under the MIT License (see the `LICENSE` file).
