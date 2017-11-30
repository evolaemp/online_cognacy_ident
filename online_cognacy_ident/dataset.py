from collections import defaultdict, namedtuple

import csv
import itertools
import os.path
import sys



"""
Dict mapping the column names that the Dataset class looks for to lists of
possible variants. Used in Dataset._parse_header().
"""
RECOGNISED_COLUMN_NAMES = {
    'doculect': ['doculect', 'language', 'lang'],
    'concept': ['concept', 'gloss'],
    'asjp': ['asjp', 'transcription'],
    'cog_class': ['cog_class', 'cognate_class']
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


    def _parse_header(self, line, exclude=['cog_class']):
        """
        Return a {column name: index} dict, excluding the columns listed in the
        second func arg.

        Raise a DatasetError if not all required columns can be recovered.
        """
        d = {}

        column_names = {column: names
                for column, names in RECOGNISED_COLUMN_NAMES.items()
                if column not in exclude}

        for index, heading in enumerate(line):
            heading = heading.lower()
            for column, recognised_names in column_names.items():
                if heading in recognised_names:
                    d[column] = index
                    break

        for column in column_names.keys():
            if column not in d:
                raise DatasetError('Could not find the column for {}'.format(column))

        return d


    def _read_words(self, cog_sets=False):
        """
        Generate the [] of Word entries in the dataset. Raise a DatasetError if
        there is a problem reading the file.

        If the cog_sets flag is set, then yield (Word, cognate class) tuples.
        """
        try:
            with open(self.path, encoding='utf-8', newline='') as f:
                reader = csv.reader(f, dialect=self.dialect)

                header = self._parse_header(next(reader),
                        exclude=[] if cog_sets else ['cog_class'])

                self.alphabet = set()

                for line in reader:
                    self.alphabet |= set(line[header['asjp']])

                    word = Word._make([
                        line[header['doculect']],
                        line[header['concept']],
                        line[header['asjp']] ])

                    if cog_sets:
                        yield word, line[header['cog_class']]
                    else:
                        yield word

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
        Return the [] of Word named tuples comprising the dataset, excluding
        in-doculect synonyms; i.e. the output should include at most one word
        per doculect per concept.

        Raise a DatasetError if there is an error reading the file.
        """
        words = []
        seen = set()

        for word in self._read_words():
            key = (word.doculect, word.concept,)
            if key not in seen:
                seen.add(key)
                words.append(word)

        return words


    def get_concepts(self):
        """
        Return a {concept: words} dict mapping each concept in the dataset to a
        [] of Word tuples that belong to that concept. In-doculect synonyms are
        excluded.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        d = defaultdict(list)

        for word in self.get_words():
            d[word.concept].append(word)

        return d

    def ldn(self, w1, w2):
        """
        COPIED THIS FUNCTION FROM PMI SCRIPT.
        Leventsthein distance normalized
        :param a: word
        :type a: str
        :param b: word
        :type b: str
        :return: distance score
        :rtype: float
        """
        m = [];
        la = len(w1) + 1;
        lb = len(w2) + 1
        for i in range(0, la):
            m.append([])
            for j in range(0, lb): m[i].append(0)
            m[i][0] = i
        for i in range(0, lb): m[0][i] = i
        for i in range(1, la):
            for j in range(1, lb):
                s = m[i - 1][j - 1]
                if (w1[i - 1] != w2[j - 1]): s = s + 1
                m[i][j] = min(m[i][j - 1] + 1, m[i - 1][j] + 1, s)
        la = la - 1;
        lb = lb - 1
        return float(m[la][lb]) / float(max(la, lb))

    def generate_pairs(self, cutoff=0.5):
        """
        Generate pairs of transcriptions of words from different languages but
        linked to the same concept.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        for concept, words in self.get_concepts().items():
            for word1, word2 in itertools.combinations(words, 2):
                if word1.doculect != word2.doculect:
                    if self.ldn(word1.asjp, word2.asjp) < cutoff:
                        yield word1.asjp, word2.asjp


    def get_clusters(self):
        """
        Return a {concept: cog_sets} dict where the values are frozen sets of
        frozen sets of Word tuples, comprising the set of cognate sets for that
        concept. In-doculect synonyms are excluded.

        Raise a DatasetError if the dataset does not include cognacy info or if
        there is a probelm reading the file.
        """
        d = defaultdict(set)  # {(concept, cog_class): set of words}
        seen = set()  # set of (doculect, concept) tuples
        clusters = defaultdict(list)  # {concept: [frozenset of words, ..]}

        for word, cog_class in self._read_words(cog_sets=True):
            if (word.doculect, word.concept) not in seen:
                seen.add((word.doculect, word.concept))
                d[(word.concept, cog_class)].add(word)

        for (concept, cog_class), cog_set in d.items():
            clusters[concept].append(frozenset(cog_set))

        return {key: frozenset(value) for key, value in clusters.items()}



def write_clusters(clusters, path=None, dialect='excel-tab'):
    """
    Write cognate set clusters to a csv file with columns: concept, doculect,
    transcription, cog_class. The latter comprises automatically generated id
    strings of the type concept:number.

    The clusters arg should be a dict mapping concepts to frozen sets of frozen
    sets of Word named tuples.

    If path is None, use stdout. Raise a DatasetError if the file/stdout cannot
    be written into.
    """
    if path:
        try:
            f = open(path, 'w', encoding='utf-8', newline='')
        except OSError as err:
            raise DatasetError('Could not open file: {}'.format(path))
    else:
        f = sys.stdout

    writer = csv.writer(f, dialect=dialect)
    writer.writerow(['concept', 'doculect', 'transcription', 'cog_class'])

    for concept, cog_sets in sorted(clusters.items()):
        for index, cog_set in enumerate(cog_sets):
            for word in sorted(cog_set):
                writer.writerow([
                    word.concept, word.doculect, word.asjp,
                    '{}:{!s}'.format(concept, index)])

    if path:
        f.close()
