import argparse



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


    def run(self, raw_args=None):
        """
        Parse the given args (if these are None, default to parsing sys.argv,
        which is what you would want unless you are unit testing).
        """
        args = self.parser.parse_args(raw_args)
