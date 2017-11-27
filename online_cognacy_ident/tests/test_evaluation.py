from string import ascii_lowercase
from unittest import TestCase

from hypothesis.strategies import lists, sampled_from
from hypothesis import given

from online_cognacy_ident.evaluation import calc_b_cubed, calc_f_score

from online_cognacy_ident.tests.test_dataset import clusters



class EvaluationTestCase(TestCase):

    @given(lists(sampled_from(ascii_lowercase), min_size=1))
    def test_calc_b_cubed_on_identical_lists(self, labels):
        self.assertEqual(calc_b_cubed(labels, labels), (1.0, 1.0, 1.0,))

    @given(lists(sampled_from(ascii_lowercase), min_size=5, max_size=5),
            lists(sampled_from(ascii_lowercase), min_size=5, max_size=5))
    def test_calc_b_cubed_is_in_range(self, labels_a, labels_b):
        res = calc_b_cubed(labels_a, labels_b)
        for item in res:
            self.assertTrue(0.0 <= item <= 1.0)

    @given(clusters())
    def test_calc_f_score_on_identical_clusters(self, clusters):
        self.assertEqual(calc_f_score(clusters, clusters), 1.0)
