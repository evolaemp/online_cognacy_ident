import os.path
import tempfile

from unittest import TestCase

import numpy as np

from online_cognacy_ident.model import load_model, save_model, ModelError



class ModelTestCase(TestCase):

    def test_load_model_with_bad_path(self):
        with self.assertRaises(ModelError) as cm:
            load_model('')

        self.assertTrue(str(cm.exception).startswith('Could not open model'))

    def test_load_model_with_bad_file(self):
        with self.assertRaises(ModelError) as cm:
            load_model(os.path.abspath(__file__))

        self.assertTrue(str(cm.exception).startswith('Could not read model'))

    def test_save_model_with_bad_path(self):
        with self.assertRaises(ModelError) as cm:
            save_model('', 'pmi', [42])

        self.assertTrue(str(cm.exception).startswith('Could not write model'))

    def test_save_and_load_model_pmi(self):
        pmi = {('ъ', 'ь'): 0.42}

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'model')
            save_model(path, 'pmi', pmi)
            model = load_model(path)

        self.assertEqual(len(model), 2)
        self.assertEqual(model[0], 'pmi')
        self.assertEqual(model[1], pmi)

    def test_save_and_load_model_phmm(self):
        em = np.array([0.1, 0.2])
        gx = np.array([0.3, 0.4])
        gy = np.array([0.5, 0.6])
        trans = np.array([0.7, 0.8])

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'model')
            save_model(path, 'phmm', [em, gx, gy, trans])
            model = load_model(path)

        self.assertEqual(len(model), 5)
        self.assertEqual(model[0], 'phmm')

        for index, param in enumerate([em, gx, gy, trans]):
            self.assertTrue(type(model[index+1]) is np.ndarray)
            np.testing.assert_array_equal(model[index+1], param)
