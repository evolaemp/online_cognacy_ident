# online cognacy identification

This repository accompanies the paper **Fast and unsupervised methods for
multilingual cognate clustering** by Rama, Wahle, Sofroniev, and Jäger. The
repository contains both the data and the source code used in the paper's
experiments.


## setup

The code is developed using Python 3.5 but later versions should also do as long
as the dependencies are satisfied.

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

If you run into difficulties, please make sure you have tried the setup in a
fresh virtual environment before opening an issue.

### windows users

Installing `python-igraph` on windows may not work via `pip install python-igraph`.
Please use the appropriate windows binaries from Christoph Gohlkes site:
https://www.lfd.uci.edu/~gohlke/pythonlibs/


## usage

```bash
# activate the virtual env if it is not already
source meta/venv/bin/activate

# ensure the reproducibility of the results
export PYTHONHASHSEED=42

# use run.py to invoke the commands
python run.py --help

# run the pair hidden markov model algorithm
python run.py phmm --ipa datasets/japanese.tsv --output output/japanese.tsv

# run the pointwise mutual information algorithm
python run.py pmi --ipa datasets/japanese.tsv --output output/japanese.tsv

# evaluate the output
python eval.py datasets/japanese.tsv output/japanese.tsv
```

A dataset should be in csv format. You can specify the csv dialect using the
`--dialect-input` option, possible values are `excel`, `excel-tab`, and `unix`.
If this is omitted, the script will try to guess the dialect by looking at the
file extension.

A dataset should have a header with at least the following columns: `doculect`
or `language`, `concept` or `gloss`, and `asjp` or `transcription`. Column name
detection is case-insensitive. If there are two or more words tied to a single
gloss in a given doculect, all but the first are ignored.


## datasets

The datasets used in the paper's experiments can be found in the `datasets`
directory.

| dataset        | language families | transcription | source                |
|----------------|-------------------|---------------|-----------------------|
| `abvd`         | Austronesian      | ipa           | Greenhill et al, 2008 |
| `afrasian`     | Afro-Asiatic      | asjp          | Militarev, 2000       |
| `bai`          | Sino-Tibetan      | ipa           | Wang, 2006            |
| `chinese_1964` | Sino-Tibetan      | ipa           | Běijīng Dàxué, 1964   |
| `chinese_2004` | Sino-Tibetan      | ipa           | Hóu, 2004             |
| `huon`         | Trans-New Guinea  | asjp          | McElhanon, 1967       |
| `ielex`        | Indo-European     | ipa           | Dunn, 2012            |
| `japanese`     | Japonic           | ipa           | Hattori, 1973         |
| `kadai`        | Tai-Kadai         | asjp          | Peiros, 1998          |
| `kamasau`      | Torricelli        | asjp          | Sanders, 1980         |
| `lolo_burmese` | Sino-Tibetan      | asjp          | Peiros, 1998          |
| `mayan`        | Mayan             | asjp          | Brown, 2008           |
| `miao_yao`     | Hmong-Mien        | asjp          | Peiros, 1998          |
| `mixe_zoque`   | Mixe-Zoque        | asjp          | Cysouw et al, 2006    |
| `mon_khmer`    | Austroasiatic     | asjp          | Peiros, 1998          |
| `ob_ugrian`    | Uralic            | ipa           | Zhivlov, 2011         |
| `tujia`        | Sino-Tibetan      | ipa           | Starostin, 2013       |

Please note that you should use the `--ipa` flag when running the algorithms on
any IPA-transcribed dataset, including the ones found in the `datasets` dir.


## license

The datasets are published under a [Creative Commons Attribution-ShareAlike 4.0
International License](https://creativecommons.org/licenses/by-sa/4.0/). The
source code is published under the MIT License (see the `LICENSE` file).
