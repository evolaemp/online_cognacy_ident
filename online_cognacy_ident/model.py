import pickle



class ModelError(ValueError):
    """
    Raised when something goes wrong with saving or loading a model.
    """
    pass



def save_model(path, algorithm, params):
    """
    Write a trained model to a pickle file. The algorithm should be either pmi,
    in which case the param should be the PMI dict, or phmm, in which case the
    param should be an [em, gx, gy, trans] sequence.
    """
    if algorithm == 'pmi':
        data = {
            'algorithm': 'pmi',
            'pmi': params}
    else:
        data = {
            'algorithm': 'phmm',
            'em': params[0],
            'gx': params[1],
            'gy': params[2],
            'trans': params[3]}

    try:
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=3)
    except OSError:
        raise ModelError('Could not write model file: {}'.format(path))



def load_model(path):
    """
    Load a saved model and return the model's algorithm and params.
    """
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
    except OSError:
        raise ModelError('Could not open model file: {}'.format(path))
    except pickle.PickleError:
        raise ModelError('Could not read model file: {}'.format(path))

    try:
        assert 'algorithm' in data and data['algorithm'] in ['pmi', 'phmm']
        if data['algorithm'] == 'pmi':
            assert 'pmi' in data
        else:
            for key in ['em', 'gx', 'gy', 'trans']: assert key in data
    except AssertionError:
        raise ModelError('Could not read model file: {}'.format(path))

    if data['algorithm'] == 'pmi':
        return 'pmi', data['pmi']
    else:
        return 'phmm', data['em'], data['gx'], data['gy'], data['trans']
