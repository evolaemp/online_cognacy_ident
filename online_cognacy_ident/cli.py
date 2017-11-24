import argparse
import csv
import random

from online_cognacy_ident.clustering import cognate_code_infomap2
from online_cognacy_ident.dataset import Dataset, write_clusters
from online_cognacy_ident.phmm import wrapper
from online_cognacy_ident.pmi import train as train_pmi



def number_in_interval(number, type, interval):
    """
    Typecast the number to an int or a float and ensure that it is within the
    specified interval. Raise an ArgumentTypeError otherwise.

    Helper for Cli's ArgumentParser instance.
    """
    try:
        number = type(number)
    except ValueError:
        raise argparse.ArgumentTypeError(
                'invalid {} value: {!r}'.format(type.__name__, number))

    if number < interval[0] or number > interval[1]:
        raise argparse.ArgumentTypeError(
                '{} not in interval [{}; {}]'.format(number, interval[0], interval[1]))

    return number



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
        self.parser.add_argument('-r', '--random-seed', type=int, default=42, help=(
            'integer to use as seed for python\'s random module; '
            'the default value is 42'))

        self.parser.add_argument('-a', '--alpha',
            type=lambda x: number_in_interval(x, float, [0.5, 1]),
            default=0.75, help=(
                'α, EM hyperparameter; should be within the interval [0.5; 1]; '
                'the default value is 0.75'))
        self.parser.add_argument('-m', '--batch-size',
            type=lambda x: number_in_interval(x, int, [1, float('inf')]),
            default=256, help=(
                'm, EM hyperparameter; should be a positive integer; '
                'the default value is 256'))

        self.parser.add_argument('-o', '--output', help=(
            'path where to write the identified cognate classes; '
            'defaults to stdout'))


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing).
        """
        args = self.parser.parse_args(raw_args)

        random.seed(args.random_seed)

        dataset = Dataset(args.dataset, args.dialect)

        if args.algorithm == 'phmm':
            print(wrapper.training_wrapped(dataset))

        else:
            pmidict = train_pmi(dataset, alpha=args.alpha, max_batch=args.batch_size)
            clusters = cognate_code_infomap2(dataset.get_concepts(), pmidict)
            write_clusters(clusters, args.output)
