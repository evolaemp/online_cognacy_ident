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

# use manage.py to invoke the commands
python manage.py --help
```


## license

The source code is published under the MIT License (see the `LICENSE` file).
