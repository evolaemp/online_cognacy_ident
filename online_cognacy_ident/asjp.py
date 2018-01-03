import unicodedata

from lingpy.sequence.sound_classes import ipa2tokens, tokens2class



"""
String of all valid ASJP symbols.
"""
ASJP_SYMBOLS = 'pbmfv84tdszcnSZCjT5kgxNqGX7hlLwyr!ieE3auo'



def clean_asjp(string, strict=False):
    """
    Clean up an ASJP transcription string. If the latter contains multiple
    entries, all but the first one are stripped.

    If strict=True, all non-ASJP symbols are ignored. Otherwise, only non-ASJP
    symbols that are not letters are ignored.
    """
    string = string.strip().split(',')[0].strip()

    asjp = ''

    for char in string:
        if char in ASJP_SYMBOLS:
            asjp += char
        elif not strict and unicodedata.category(char).startswith('L'):
            asjp += char

    return asjp



def ipa_to_asjp(ipa):
    """
    Convert an IPA transcription into an ASJP one.
    """
    return ''.join(tokens2class(ipa2tokens(ipa), 'asjp'))
