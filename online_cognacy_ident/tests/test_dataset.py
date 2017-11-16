import csv
import os.path
import tempfile

from unittest import TestCase

from hypothesis.strategies import lists, text, tuples
from hypothesis import assume, given

from ..dataset import DatasetError, Dataset, Word



def write_dataset(path, header, words, dialect):
    with open(path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, dialect)
        writer.writerow(header)
        for word in words:
            writer.writerow(word)



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

    @given(lists(tuples(text(), text(), text()), min_size=10))
    def test_get_words(self, data):
        assume(all([all(['\0' not in s for s in row]) for row in data]))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'dataset')

            for dialect in csv.list_dialects():
                write_dataset(path, ['doculect', 'concept', 'asjp'], data, dialect)

                dataset = Dataset(path, dialect)
                words = dataset.get_words()

                self.assertEqual(words, [Word._make(i) for i in data])
