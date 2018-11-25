'''Complexity presentation utilities.

Public interface:
    ComplexityBundle:
        bundle of memory, parameter, and compuational complexities.
    to_string_with_metric_prefixes(): format number with metric prefixes
'''

_METRIC_PREFIXES = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']


class ComplexityBundle:
    '''Bundle of memory, parameter, and compuational complexities.'''

    def __init__(self, memory, parameters, computation):
        '''Contruct ComplexityBundle'''
        self.memory = memory
        self.parameters = parameters
        self.computation = computation
            
    def __str__(self):
        string = (
            f'M={to_string_with_metric_prefixes(self.memory)}B\n'
            f'P={to_string_with_metric_prefixes(self.parameters)}\n'
            f'C={to_string_with_metric_prefixes(self.computation)}MACC\n'
        )
        return string

    def __repr__(self):
        representation = str(self)
        return representation


def to_string_with_metric_prefixes(number):
    number = int(number)
    string = ''
    if number == 0:
        string = '0'
    else:
        for prefix in _METRIC_PREFIXES:
            string = f'{number % 1000:03d}{prefix} {string}'
            number //= 1000
            if number == 0:
                break
    return string