import argparse
import csv

from .dataset import Dataset



class Cli:
    """
    Handles the user's input, inits the necessary classes and functions, and
    takes care of exiting the programme.

    Usage:
        if __name__ == '__main__':
            cli = Cli()
            cli.run()
    """

    def __init__(self):
        """
        Init the argparse parser.
        """
        self.parser = argparse.ArgumentParser(description=(
            'run the pmi or phmm online cognacy identification algorithm '
            'on the specified dataset'))

        self.parser.add_argument('algorithm', choices=['pmi', 'phmm'],
                help='which of the two algorithms to use')
        self.parser.add_argument('dataset', help='path to the dataset file')

        self.parser.add_argument('-d', '--dialect', choices=csv.list_dialects(), help=(
            'the csv dialect to use for reading the dataset; '
            'the default is to look at the file extension '
            'and use excel for .csv and excel-tab for .tsv'))


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing).
        """
        args = self.parser.parse_args(raw_args)

        dataset = Dataset(args.dataset, args.dialect)
        [word for word in dataset.get_words()]
