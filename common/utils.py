from linalg import as_array

def get_data(filename, delimiter=','):
    with open(filename) as f:
        CDC = [line.strip().split(delimiter)[1] for i, line in enumerate(f)]
    return as_array(CDC)

def check_bounds(x):
    if x < 0: x = 0.0
    if x > 1: x = 1.0
    return x
