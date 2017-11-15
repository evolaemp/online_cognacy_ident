import csv
import os.path
import tempfile

from unittest import TestCase

from hypothesis.strategies import text
from hypothesis import given

from ..dataset import DatasetError, Dataset



def write_dataset(path, header, words, dialect):
    with open(path, 'w', encoding='utf-8') as f:
        writer = csv.writer(f, dialect)
        writer.writerow(header)
        for word in words:
            writer.writerow(word)



class DatasetTestCase(TestCase):

    def test_with_bad_path(self):
        with self.assertRaises(DatasetError) as cm:
            dataset = Dataset('')

        self.assertEqual(str(cm.exception), 'Could not find file: ')

    def test_with_bad_file(self):
        path = os.path.abspath(__file__)
        dataset = Dataset(path)

        with self.assertRaises(DatasetError) as cm:
            dataset.get_concepts()

        self.assertTrue(str(cm.exception).startswith('Could not find the column for'))
