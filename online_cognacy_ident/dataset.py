from collections import defaultdict, namedtuple

import csv
import itertools
import os.path



"""
Dict mapping the column names that the Dataset class looks for to lists of
possible variants. Used in Dataset._parse_header().
"""
RECOGNISED_COLUMN_NAMES = {
    'doculect': ['doculect', 'language', 'lang'],
    'concept': ['concept', 'gloss'],
    'asjp': ['asjp', 'transcription']
}



"""
The named tuple used in the return values of Dataset.get_*().
"""
Word = namedtuple('Word', 'doculect, concept, asjp')



class DatasetError(ValueError):
    """
    Raised when something goes wrong with reading a dataset.
    """
    pass



class Dataset:
    """
    Handles dataset reading. It is assumed that the dataset would be a csv/tsv
    file that contains at least one of the columns for each list of recognised
    column names.

    Usage:
        try:
            dataset = Dataset(path)
            for concept, words in dataset.concepts():
                print(concept)
        except DatasetError as err:
            print(err)
    """

    def __init__(self, path, dialect=None):
        """
        Set the instance's props. Raise a DatasetError if the given file path
        does not exist. 

        The dialect arg should be either a string identifying one of the csv
        dialects or None, in which case the dialect is inferred based on the
        file extension. Raise a ValueError if the given dialect is specified
        but unrecognised.
        """
        if not os.path.exists(path):
            raise DatasetError('Could not find file: {}'.format(path))

        if dialect is None:
            dialect = 'excel-tab' if path.endswith('.tsv') else 'excel'
        elif dialect not in csv.list_dialects():
            raise ValueError('Unrecognised csv dialect: {!s}'.format(dialect))

        self.path = path
        self.dialect = dialect

        self.alphabet = None


    def _parse_header(self, line):
        """
        Return a {name: number} dict. Raise a DatasetError if not all columns
        can be recovered.
        """
        d = {}

        for index, heading in enumerate(line):
            heading = heading.lower()
            for column, recognised_names in RECOGNISED_COLUMN_NAMES.items():
                if heading in recognised_names:
                    d[column] = index
                    break

        for column in RECOGNISED_COLUMN_NAMES.keys():
            if column not in d:
                raise DatasetError('Could not find the column for {}'.format(column))

        return d


    def _read_words(self):
        """
        Generate the [] of Word entries in the dataset. Raise a DatasetError if
        there is a problem reading the file.
        """
        try:
            with open(self.path, encoding='utf-8', newline='') as f:
                reader = csv.reader(f, dialect=self.dialect)
                header = self._parse_header(next(reader))

                self.alphabet = set()

                for line in reader:
                    self.alphabet |= set(line[header['asjp']])

                    yield Word._make([
                        line[header['doculect']],
                        line[header['concept']],
                        line[header['asjp']] ])

        except OSError as err:
            raise DatasetError('Could not open file: {}'.format(self.path))

        except csv.Error as err:
            raise DatasetError('Could not read file: {}'.format(self.path))


    def get_alphabet(self):
        """
        Return a sorted list of all characters found throughout transcriptions
        in the dataset. Raise a DatasetError if there is a problem.
        """
        if self.alphabet is None:
            self.get_words()

        return sorted(self.alphabet)


    def get_words(self):
        """
        Return the [] of Word named tuple entries comprising the dataset. Raise
        a DatasetError if there is an error reading the file.
        """
        return [word for word in self._read_words()]


    def get_concepts(self):
        """
        Return a {concept: words} dict mapping each concept in the dataset to a
        [] of Word entries that belong to that concept.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        d = defaultdict(list)

        for word in self._read_words():
            d[word.concept].append(word)

        return d


    def generate_pairs(self):
        """
        Generate pairs of transcriptions of words from different languages but
        linked to the same concept.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        for concept, words in self.get_concepts().items():
            for word1, word2 in itertools.combinations(words, 2):
                if word1.doculect != word2.doculect:
                    yield word1.asjp, word2.asjp
