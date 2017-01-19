from linalg import as_array

def get_data(filename, delimiter=','):
    with open(filename) as f:
        CDC = [line.strip().split(delimiter)[1] for i, line in enumerate(f)]
    return as_array(CDC)

def check_bounds(x):
    if x < 0: x = 0.0
    if x > 1: x = 1.0
    return x

def read_params(file_path):
    params = {}
    with open(file_path) as f:
        for line in f:
            key, data, data_type = line.rstrip().split('\t')
            if data_type == 'string':
                pass
            elif data_type == 'int':
                data = int(data)
            elif data_type in ['float', 'double']:
                data = float(data)
            elif data_type == 'boolean':
                data = True if data == 'true' else False
            elif data_type == 'list of string':
                data = data.split(',')
            elif data_type == 'list of int': 
                data = map(int, data.split(','))
            else:
                print 'This line is ignored'

            params[key] = data
    
    return params
            
    
