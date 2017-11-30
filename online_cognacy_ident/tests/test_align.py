from unittest import TestCase

from online_cognacy_ident.align import needleman_wunsch



class AlignTestCase(TestCase):

    def test_needleman_wunsch(self):
        self.assertEqual(needleman_wunsch("AAAAABBBB", "AACAABBCB"),
            (5.0, [('A', 'A'), ('A', 'A'), ('A', 'C'), ('A', 'A'), ('A', 'A'), ('B', 'B'), ('B', 'B'), ('B', 'C'), ('B', 'B')]))
        self.assertEqual(needleman_wunsch("banana", "mancala", local=True),
            (2.0, [('a', 'a'), ('n', 'n')]))
        self.assertEqual(needleman_wunsch("abc", "t", lodict={('a', ''): 0, ('b', ''): -2, ('c', ''): -0.5}, gop=None),
            (-1.5, [('a', ''), ('b', 't'), ('c', '')]))
