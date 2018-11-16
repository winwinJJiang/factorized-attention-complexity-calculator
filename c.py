class ComplexityBundle:
    
    metric_prefixes = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']

    def __init__(self, memory, parameters, compute):
        self.memory = memory
        self.parameters = parameters
        self.compute = compute

    def to_string_with_metric_prefixes(self, number):
        number = int(number)
        string = ''
        if number == 0:
            string = '0'
        else:
            for prefix in self.metric_prefixes:
                string = f'{number % 1000:03d}{prefix} {string}'
                number //= 1000
                if number == 0:
                    break
        return string
            
    def __str__(self):
        string = (
            f'M={self.to_string_with_metric_prefixes(self.memory)}\n'
            f'P={self.to_string_with_metric_prefixes(self.parameters)}\n'
            f'C={self.to_string_with_metric_prefixes(self.compute)}\n'
        )
        return string

    def __repr__(self):
        representation = str(self)
        return representation


def get_intermediates(bk, m, c):
    bk = bk or c // 2
    m = m or c // 2
    return bk, m


def get_heads(h, bk):
    h = h or max(bk // 64, 1)
    return h


def package_result(memory, parameters, compute):
    result = ComplexityBundle(memory, parameters, compute)
    return result


def conv_dot(k=None, m=None, c=64, s=64 * 64):
    k, m = get_intermediates(k, m, c)
    memory = ((2 * k + 3 * c) * s + s ** 2) * 4
    parameters = 2 * k * c + c ** 2
    compute = (2 * k * c + c ** 2) * s + (2 * k + 2 * c) * s ** 2
    result = package_result(memory, parameters, compute)
    return result


def fa_dot(b=None, m=None, c=64, s=64 * 64):
    b, m = get_intermediates(b, m, c)
    memory = ((2 * b + 3 * c) * s + b * c) * 4
    parameters = 2 * b * c + c ** 2
    compute = (8 * b * c + 2 * c ** 2) * s
    result = package_result(memory, parameters, compute)
    return result


def conv_multi(k=None, m=None, c=64, s=64 * 64, h=None):
    k, m = get_intermediates(k, m, c)
    h = get_heads(h, k)
    memory = ((2 * k + 2 * m + 2 * c) * s + h * s ** 2) * 4
    parameters = 2 * k * c + 2 * m * c
    compute = (4 * k * c + 4 * m * c) * s + (2 * k + 2 * m) * s ** 2
    result = package_result(memory, parameters, compute)
    return result


def fa_multi(b=None, m=None, c=64, s=64 * 64, h=None):
    b, m = get_intermediates(b, m, c)
    h = get_heads(h, b)
    memory = ((2 * b + 2 * m + 2 * c) * s + b * m / h) * 4
    parameters = 2 * b * c + 2 * m * c
    compute = (4 * b * c + 4 * m * c + 4 * b * m) * s
    result = package_result(memory, parameters, compute)
    return result


def conv_trans(k=1024, m=None, c=4096, s=512, h=16, l=24):
    m = m or k
    h = get_heads(h, k)
    memory = l * (((2 * k + 2 * m + 6 * c) * s + h * s ** 2) * 4)
    parameters = l * (2 * k * c + 2 * m * c + 2 * c ** 2)
    compute = l * (
        (4 * k * c + 4 * m * c + 5 * c ** 2) * s + (2 * k + 2 * m) * s ** 2
    )
    result = package_result(memory, parameters, compute)
    return result


def fa_trans(b=1024, m=None, c=4096, s=512, h=16, l=24):
    m = m or b
    h = get_heads(h, b)
    memory = l * (((2 * b + 2 * m + 6 * c) * s + b * m / h) * 4)
    parameters = l * (2 * b * c + 2 * m * c + 2 * c ** 2)
    compute = l * ((4 * b * c + 4 * m * c + 4 * b * m + 5 * c ** 2) * s)
    result = package_result(memory, parameters, compute)
    return result

