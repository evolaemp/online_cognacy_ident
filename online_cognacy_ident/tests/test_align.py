from unittest import TestCase

from online_cognacy_ident.align import needleman_wunsch



class AlignTestCase(TestCase):

    def test_needleman_wunsch(self):
        self.assertEqual(needleman_wunsch("AAAAABBBB", "AACAABBCB"),
            (5.0, (('A', 'A'), ('A', 'A'), ('A', 'C'), ('A', 'A'), ('A', 'A'), ('B', 'B'), ('B', 'B'), ('B', 'C'), ('B', 'B'))))
