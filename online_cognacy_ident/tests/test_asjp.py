from unittest import TestCase

from online_cognacy_ident.asjp import clean_asjp



class CleanTestCase(TestCase):

    def test_clean_asjp(self):
        self.assertEqual(clean_asjp('3ne'), '3ne')
        self.assertEqual(clean_asjp('masX~7e'), 'masX7e')
        self.assertEqual(clean_asjp('tX~ur'), 'tXur')
        self.assertEqual(clean_asjp('duC"e'), 'duCe')

        self.assertEqual(clean_asjp('naq k"ari7'), 'naqkari7')
        self.assertEqual(clean_asjp('t"ort"oh'), 'tortoh')
