import collections
import csv
import itertools
import os.path
import string
import tempfile

from unittest import TestCase

from hypothesis.strategies import composite, integers, lists, sets, text
from hypothesis import assume, given

from online_cognacy_ident.dataset import (
        Word, DatasetError, Dataset, PairsDataset, write_clusters)



@composite
def clusters(draw):
    clusters = {}

    concepts = draw(sets(text(max_size=10), min_size=1))
    langs = draw(sets(text(max_size=3), min_size=1))

    assume(all(['\0' not in s for s in concepts]))
    assume(all(['\0' not in s for s in langs]))

    num_words = len(concepts) * len(langs)
    asjp = draw(lists(text(alphabet=string.ascii_letters, max_size=5),
            min_size=num_words, max_size=num_words))

    counter = itertools.count()

    for concept in concepts:
        cog_sets = collections.defaultdict(set)

        for lang in langs:
            cog_class = draw(integers(min_value=0, max_value=5))
            word = Word._make([lang, concept, asjp[next(counter)]])
            cog_sets[cog_class].add(word)

        clusters[concept] = frozenset([
            frozenset(cog_set) for cog_set in cog_sets.values() if cog_set])

    return clusters



class DatasetTestCase(TestCase):

    def test_read_asjp(self):
        dataset = Dataset(os.path.abspath(__file__))

        self.assertEqual(dataset._read_asjp('3ne'), '3ne')
        self.assertEqual(dataset._read_asjp('masX~7e'), 'masX7e')
        self.assertEqual(dataset._read_asjp('tX~ur'), 'tXur')
        self.assertEqual(dataset._read_asjp('duC"e'), 'duCe')

        self.assertEqual(dataset._read_asjp('naq k"ari7'), 'naqkari7')
        self.assertEqual(dataset._read_asjp('t"ort"oh'), 'tortoh')

    def test_read_asjp_from_ipa(self):
        dataset = Dataset(os.path.abspath(__file__), is_ipa=True)

        self.assertEqual(dataset._read_asjp('ʔikiʨuri'), '7ikiCuri')
        self.assertEqual(dataset._read_asjp('pizuriːduːɭ'), 'pizuriduL')

    def test_init_with_bad_path(self):
        with self.assertRaises(DatasetError) as cm:
            Dataset('')

        self.assertTrue(str(cm.exception).startswith('Could not find file'))

    def test_init_with_bad_file(self):
        path = os.path.abspath(__file__)
        dataset = Dataset(path)

        with self.assertRaises(DatasetError) as cm:
            dataset.get_concepts()

        self.assertTrue(str(cm.exception).startswith('Could not find the column for'))

    def test_get_alphabet_with_kamasau(self):
        dataset = Dataset('datasets/kamasau.tsv')
        self.assertEqual(dataset.get_alphabet(), [
            '3', '5', '7', 'C', 'N',
            'a', 'b', 'd', 'e', 'g', 'h', 'i', 'j', 'k',
            'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y'])

    def test_get_words_with_kamasau(self):
        dataset = Dataset('datasets/kamasau.tsv')
        words = dataset.get_words()

        self.assertEqual(len(words), 271)
        self.assertEqual(words[0], Word('TRING', 'I', 'Ne'))
        self.assertEqual(words[-9], Word('SAMAP', 'die', 'gureNnand'))
        self.assertEqual(words[-1], Word('SAMAP', 'road (path)', 'N3m'))

    def test_get_concepts_with_kamasau(self):
        dataset = Dataset('datasets/kamasau.tsv')
        concepts = dataset.get_concepts()

        self.assertEqual(len(concepts), 36)
        words = set([word for value in concepts.values() for word in value])
        self.assertEqual(len(words), 271)
        self.assertEqual(words, set(dataset.get_words()))

    def test_get_asjp_pairs_with_kamasau(self):
        dataset = Dataset('datasets/kamasau.tsv')
        pairs = dataset.get_asjp_pairs()

        self.assertEqual(sum([1 for a, b in pairs if a == 'mandi' or b == 'mandi']), 8*7/2)
        self.assertEqual(sum([1 for a, b in pairs if a == 'wiye' and b == 'wiye']), 8*7/2)

    def test_get_clusters_with_kamasau(self):
        dataset = Dataset('datasets/kamasau.tsv')
        words = dataset.get_words()
        clusters = dataset.get_clusters()

        for concept in ['I', 'come', 'water']:
            words_ = [word for word in words if word.concept == concept]
            self.assertEqual(clusters[concept], frozenset([frozenset(words_)]))

        for concept in ['mountain', 'one', 'die']:
            words_ = [word for word in words if word.concept == concept]
            cog_set1 = frozenset([word for word in words_ if word.doculect != 'SAMAP'])
            cog_set2 = frozenset([word for word in words_ if word.doculect == 'SAMAP'])
            self.assertEqual(clusters[concept], frozenset([cog_set1, cog_set2]))

    def test_write_clusters_with_kamasau(self):
        dataset = Dataset('datasets/kamasau.tsv')
        words = dataset.get_words()
        clusters = dataset.get_clusters()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'clusters.tsv')

            for dialect in csv.list_dialects():
                write_clusters(clusters, path, dialect=dialect)

                dataset_ = Dataset(path, dialect=dialect)
                self.assertEqual(sorted(dataset_.get_words()), sorted(words))
                self.assertEqual(dataset_.get_clusters(), clusters)

    @given(clusters())
    def test_get_words(self, clusters):
        words = [word for cog_sets in clusters.values()
                for cog_set in cog_sets for word in cog_set]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'dataset')

            for dialect in csv.list_dialects():
                write_clusters(clusters, path, dialect)

                dataset = Dataset(path, dialect)
                self.assertEqual(set(dataset.get_words()), set(words))

    @given(clusters())
    def test_write_and_get_clusters(self, clusters):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'dataset.tsv')

            write_clusters(clusters, path)
            self.assertTrue(os.path.exists(path))

            dataset = Dataset(path)
            self.assertEqual(dataset.get_clusters(), clusters)



class PairsDatasetTestCase(TestCase):

    MAYAN_DATASET = 'training_data/Mayan_asjp40_word_pairs.txt'

    def test_init_with_bad_path(self):
        with self.assertRaises(DatasetError) as cm:
            PairsDataset('')

        self.assertTrue(str(cm.exception).startswith('Could not find file'))

    def test_init_with_bad_file(self):
        path = os.path.abspath(__file__)
        dataset = PairsDataset(path)

        with self.assertRaises(DatasetError) as cm:
            dataset.get_asjp_pairs()

        self.assertTrue(str(cm.exception).startswith('Could not read file'))

    def test_get_asjp_pairs_with_mayan(self):
        dataset = PairsDataset(self.MAYAN_DATASET)

        pairs = dataset.get_asjp_pairs()
        self.assertEqual(len(pairs), 89787)

        self.assertEqual(pairs[0], ('ta7', 'kata7'))
        self.assertEqual(pairs[-1], ('ha7', 'ya7'))
