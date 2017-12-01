"""
String of all valid ASJP symbols.
"""
ASJP_SYMBOLS = 'pbmfv84tdszcnSZCjT5kgxNqGX7hlLwyr!ieE3auo'



def clean_asjp(string):
    """
    Return a valid ASJP string out of a dataset field. If the latter contains
    multiple entries, all but the first one are stripped. If it contains
    non-ASJP symbols, these are ignored.
    """
    string = string.strip().split(',')[0].strip()

    asjp = ''

    for char in string:
        if char in ASJP_SYMBOLS:
            asjp += char

    return asjp
