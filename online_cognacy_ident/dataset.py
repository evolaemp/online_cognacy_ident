from collections import defaultdict, namedtuple

import csv
import itertools
import os.path
import sys

from lingpy.sequence.sound_classes import ipa2tokens, tokens2class

from online_cognacy_ident.align import normalized_levenshtein



"""
Dict mapping the column names that the Dataset class looks for to lists of
possible variants. Used in Dataset._read_header().
"""
RECOGNISED_COLUMN_NAMES = {
    'doculect': ['doculect', 'language', 'lang'],
    'concept': ['concept', 'gloss'],
    'asjp': ['asjp', 'transcription'],
    'cog_class': ['cog_class', 'cognate_class', 'cognate class', 'cogid']
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

    def __init__(self, path, dialect=None, is_ipa=False):
        """
        Set the instance's props. Raise a DatasetError if the given file path
        does not exist. 

        The dialect arg should be either a string identifying one of the csv
        dialects or None, in which case the dialect is inferred based on the
        file extension. Raise a ValueError if the given dialect is specified
        but unrecognised.

        If is_ipa is set, assume that the transcriptions are in IPA and convert
        them into ASJP.
        """
        if not os.path.exists(path):
            raise DatasetError('Could not find file: {}'.format(path))

        if dialect is None:
            dialect = 'excel-tab' if path.endswith('.tsv') else 'excel'
        elif dialect not in csv.list_dialects():
            raise ValueError('Unrecognised csv dialect: {!s}'.format(dialect))

        self.path = path
        self.dialect = dialect
        self.is_ipa = is_ipa

        self.alphabet = None


    def _read_header(self, line, exclude=['cog_class']):
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


    def _read_asjp(self, raw_trans):
        """
        Process a raw transcription value into an ASJP transcription:
        (1) if the input string consists of multiple comma-separated entries,
        remove all but the first one;
        (2) remove whitespace chars (the symbols + and _ are also considered
        whitespace and removed);
        (3) if this is an IPA dataset, convert the string to ASJP;
        (4) remove some common non-ASJP offender symbols.

        Helper for the _read_words method.
        """
        trans = raw_trans.strip().split(',')[0].strip()

        for char in '+_ ':
            trans = trans.replace(char, '')

        if self.is_ipa:
            trans = ''.join(tokens2class(ipa2tokens(trans), 'asjp'))

        for char in '"$%*~':
            trans = trans.replace(char, '')

        return trans


    def _read_words(self, cog_sets=False):
        """
        Generate the [] of Word entries in the dataset. Raise a DatasetError if
        there is a problem reading the file.

        If the cog_sets flag is set, then yield (Word, cognate class) tuples.
        """
        try:
            with open(self.path, encoding='utf-8', newline='') as f:
                reader = csv.reader(f, dialect=self.dialect)

                header = self._read_header(next(reader),
                        exclude=[] if cog_sets else ['cog_class'])

                self.equilibrium = defaultdict(float)

                for line in reader:
                    asjp = self._read_asjp(line[header['asjp']])

                    for i in asjp:
                        self.equilibrium[i] += 1.0

                    word = Word._make([
                        line[header['doculect']],
                        line[header['concept']],
                        asjp ])

                    if cog_sets:
                        yield word, line[header['cog_class']]
                    else:
                        yield word

        except OSError as err:
            raise DatasetError('Could not open file: {}'.format(self.path))

        except csv.Error as err:
            raise DatasetError('Could not read file: {}'.format(self.path))


    def get_equilibrium(self):
        """
        Return un-normalized equilibrium counts
        """
        if self.equilibrium is None:
            self.get_words()

        return self.equilibrium


    def get_alphabet(self):
        """
        Return a sorted list of all characters found throughout transcriptions
        in the dataset. Raise a DatasetError if there is a problem.
        """
        if self.alphabet is not None:
            return self.alphabet

        self.alphabet = set()

        for word in self.get_words():
            self.alphabet |= set(word.asjp)

        self.alphabet = sorted(self.alphabet)

        return self.alphabet


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


    def get_asjp_pairs(self, cutoff=1.0, as_int_tuples=False):
        """
        Return the list of the pairs of transcriptions of words from different
        languages but linked to the same concept.

        If the cutoff arg is less than 1.0, pairs with edit distance above that
        threshold are also ignored. If the other keyword arg is set, return the
        transcriptions as tuples of the letters' indices in self.alphabet.

        Raise a DatasetError if there is an error reading the dataset file.
        """
        pairs = []

        if as_int_tuples:
            alphabet = self.get_alphabet()

        for concept, words in self.get_concepts().items():
            for word1, word2 in itertools.combinations(words, 2):
                if word1.doculect == word2.doculect:
                    continue

                if normalized_levenshtein(word1.asjp, word2.asjp) > cutoff:
                    continue

                if as_int_tuples:
                    pair = (
                        tuple([alphabet.index(char) for char in word1.asjp]),
                        tuple([alphabet.index(char) for char in word2.asjp]))
                else:
                    pair = (word1.asjp, word2.asjp)

                pairs.append(pair)

        return pairs


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
