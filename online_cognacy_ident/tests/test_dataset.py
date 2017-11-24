import collections
import csv
import itertools
import os.path
import tempfile

from unittest import TestCase

from hypothesis.strategies import composite, integers, lists, sets, text
from hypothesis import assume, given

from online_cognacy_ident.dataset import DatasetError, Dataset, Word, write_clusters



@composite
def clusters(draw):
    clusters = {}

    concepts = draw(sets(text(max_size=10), min_size=1))
    langs = draw(sets(text(max_size=3), min_size=1))

    num_words = len(concepts) * len(langs)
    asjp = draw(lists(text(max_size=5),
            min_size=num_words, max_size=num_words))

    assume(all(['\0' not in s for s in concepts]))
    assume(all(['\0' not in s for s in langs]))
    assume(all(['\0' not in s for s in asjp]))

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

    def test_with_bad_path(self):
        with self.assertRaises(DatasetError) as cm:
            Dataset('')

        self.assertTrue(str(cm.exception).startswith('Could not find file'))

    def test_with_bad_file(self):
        path = os.path.abspath(__file__)
        dataset = Dataset(path)

        with self.assertRaises(DatasetError) as cm:
            dataset.get_concepts()

        self.assertTrue(str(cm.exception).startswith('Could not find the column for'))

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
