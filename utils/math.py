import numpy as np

# https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)

def oh_encode(y, num_labels):
    """One hot encoding"""
    num_samples = y.shape[0]
    onehot_matrix = np.zeros((num_labels, num_samples))
    for index, value in enumerate(y):
        onehot_matrix[value, index] = 1.0
    
    return onehot_matrix