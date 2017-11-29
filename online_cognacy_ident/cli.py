import argparse
import csv
import random

from online_cognacy_ident.clustering import cluster
from online_cognacy_ident.dataset import Dataset, DatasetError, write_clusters
from online_cognacy_ident.evaluation import calc_f_score
from online_cognacy_ident.phmm import wrapper
from online_cognacy_ident.pmi import run_pmi



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



class RunCli:
    """
    Handles the user input, invokes the necessary classes and functions, and
    takes care of exiting the programme for running the cognacy identification
    algorithms.

    Usage:
        if __name__ == '__main__':
            cli = RunCli()
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

        self.parser.add_argument('--dialect-input', choices=csv.list_dialects(), help=(
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
        self.parser.add_argument('--dialect-output',
            choices=csv.list_dialects(), default='excel-tab', help=(
                'the csv dialect to use for writing the output; '
                'the default is excel-tab'))

        self.parser.add_argument('-e', '--evaluate', action='store_true', help=(
            'evaluate the output against the input dataset and '
            'print the resulting F-score; this will fail '
            'if the input dataset does not include cognate classes'))


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing).
        """
        args = self.parser.parse_args(raw_args)

        random.seed(args.random_seed)

        try:
            dataset = Dataset(args.dataset, args.dialect_input)
        except DatasetError as err:
            self.parser.error(str(err))

        if args.algorithm == 'phmm':
            print(wrapper.training_wrapped(dataset))
        else:
            scores = run_pmi(dataset, alpha=args.alpha, max_batch=args.batch_size)

        clusters = cluster(dataset, scores)
        write_clusters(clusters, args.output, args.dialect_output)

        if args.evaluate:
            score = calc_f_score(dataset.get_clusters(), clusters)
            print('{:.4f}'.format(score))



class EvalCli:
    """
    Handles the user input, invokes the corresponding code, and takes care of
    exiting the programme for evaluating the output of cognacy identification
    algorithms.

    Usage:
        if __name__ == '__main__':
            cli = EvalCli()
            cli.run()
    """

    def __init__(self):
        """
        Init the argparse parser.
        """
        self.parser = argparse.ArgumentParser(description=(
            'evaluate the cognates clustering of a dataset '
            'against the same data\'s gold-standard cognate classes'))

        self.parser.add_argument('dataset_true', help=(
            'path to the dataset containing the gold-standard cognate classes'))
        self.parser.add_argument('dataset_pred', help=(
            'path to the dataset containing the predicted cognate classes'))

        self.parser.add_argument('--dialect-true',
            choices=csv.list_dialects(), help=(
                'the csv dialect to use for reading the dataset '
                'that contains the gold-standard cognate classes; '
                'the default is to look at the file extension '
                'and use excel for .csv and excel-tab for .tsv'))
        self.parser.add_argument('--dialect-pred',
            choices=csv.list_dialects(), help=(
                'the csv dialect to use for reading the dataset '
                'that contains the predicted cognate classes; '
                'the default is to look at the file extension '
                'and use excel for .csv and excel-tab for .tsv'))


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing), invoke the
        evaluation function and print its output.
        """
        args = self.parser.parse_args(raw_args)

        try:
            dataset_true = Dataset(args.dataset_true, args.dialect_true)
            dataset_pred = Dataset(args.dataset_pred, args.dialect_pred)
        except DatasetError as err:
            self.parser.error(str(err))

        score = calc_f_score(dataset_true.get_clusters(), dataset_pred.get_clusters())
        print('{:.4f}'.format(score))
