'''
Invoke python3 in this dir and `import c` to use the functions.

Example usage:
    import c; c.fa_dot(s=256 * 256)
    import c; c.ca_tfm(s=1024, h=8, c=4096)

Arguments for functions in the module:
    b: basis dimensionality
    k: channel count of key
    bk: b or k
    m: channel count of value
    c: channel count of input
    s: spatiotemporal size of input
    h: number of heads
    l: layers (for Transformers)

Public interface:
    res_block(): compute complexities for ResBlock.
    res_block_bot(): compute complexities for bottleneck ResBlock.
    fa_dot(): compute complexities for FA-dot.
    ca_dot(): compute complexities for CA-dot.
    fa_bot(): compute complexities for FA-bot.
    ca_bot(): compute complexities for CA-bot.
    fa_multi(): compute complexities for FA-multi.
    ca_multi(): compute complexities for CA-multi.
    fa_tfm(): compute complexities for FA-Tfm.
    ca_tfm(): compute complexities for CA-Tfm.
'''


from complexity_bundle import ComplexityBundle


def _get_bk(bk, c):
    bk = bk or c // 2
    return bk


def _get_m(m, c):
    m = m or c // 2
    return m


def _get_h(h, bk):
    h = h or max(bk // 64, 1)
    return h


def help():
    print(__doc__)


def res_block(c=64, s=64 * 64):
    '''Compute complexities for ResBlock.'''
    memory = (5 * c * s) * 4
    parameters = 18 * c ** 2
    computation = 36 * c ** 2 * s
    result = ComplexityBundle(memory, parameters, computation)
    return result


def res_block_bot(m=None, c=64, s=64 * 64):
    '''Compute complexities for bottleneck ResBlock.'''
    m = _get_m(m, c)
    memory = ((4 * m + 3 * c) * s) * 4
    parameters = 2 * m * c + 9 * m ** 2
    computation = (18 * m ** 2 + 4 * m * c) * s
    result = ComplexityBundle(memory, parameters, computation)
    return result


def fa_dot(b=None, c=64, s=64 * 64):
    '''Compute complexities for FA-dot.'''
    b = _get_bk(b, c)
    memory = ((2 * b + 3 * c) * s + b * c) * 4
    parameters = 2 * b * c + c ** 2
    computation = (8 * b * c + 2 * c ** 2) * s
    result = ComplexityBundle(memory, parameters, computation)
    return result


def ca_dot(k=None, c=64, s=64 * 64):
    '''Compute complexities for CA-dot.'''
    k = _get_bk(k, c)
    memory = ((2 * k + 3 * c) * s + s ** 2) * 4
    parameters = 2 * k * c + c ** 2
    computation = (2 * k * c + c ** 2) * s + (2 * k + 2 * c) * s ** 2
    result = ComplexityBundle(memory, parameters, computation)
    return result


def fa_bot(b=None, m=None, c=64, s=64 * 64):
    '''Compute complexities for FA-bot.'''
    b = _get_bk(b, c)
    m = _get_m(m, c)
    memory = ((2 * b + 2 * m + 2 * c) * s + b * m) * 4
    parameters = 2 * b * c + c ** 2
    computation = (4 * b * c + 4 * m * c + 4 * b * m) * s
    result = ComplexityBundle(memory, parameters, computation)
    return result


def ca_bot(k=None, m=None, c=64, s=64 * 64):
    '''Compute complexities for CA-bot.'''
    k = _get_bk(k, c)
    m = _get_m(m, c)
    memory = ((2 * k + 2 * m + 2 * c) * s + s ** 2) * 4
    parameters = 2 * k * c + c ** 2
    computation = (4 * k * c + 4 * m * c) * s + (2 * k + 2 * m) * s ** 2
    result = ComplexityBundle(memory, parameters, computation)
    return result


def fa_multi(b=None, m=None, c=64, s=64 * 64, h=None):
    '''Compute complexities for FA-multi.'''
    b = _get_bk(b, c)
    m = _get_m(m, c)
    h = _get_h(h, b)
    memory = ((2 * b + 2 * m + 2 * c) * s + b * m / h) * 4
    parameters = 2 * b * c + 2 * m * c
    computation = (4 * b * c + 4 * m * c + 4 * b * m) * s
    result = ComplexityBundle(memory, parameters, computation)
    return result


def ca_multi(k=None, m=None, c=64, s=64 * 64, h=None):
    '''Compute complexities for CA-multi.'''
    k = _get_bk(k, c)
    m = _get_m(m, c)
    h = _get_h(h, k)
    memory = ((2 * k + 2 * m + 2 * c) * s + h * s ** 2) * 4
    parameters = 2 * k * c + 2 * m * c
    computation = (4 * k * c + 4 * m * c) * s + (2 * k + 2 * m) * s ** 2
    result = ComplexityBundle(memory, parameters, computation)
    return result


def fa_tfm(b=1024, m=None, c=4096, s=512, h=16, l=1):
    '''Compute complexities for FA-Tfm.'''
    m = m or b
    h = _get_h(h, b)
    memory = l * (((2 * b + 2 * m + 6 * c) * s + b * m / h) * 4)
    parameters = l * (2 * b * c + 2 * m * c + 2 * c ** 2)
    computation = l * ((4 * b * c + 4 * m * c + 4 * b * m + 5 * c ** 2) * s)
    result = ComplexityBundle(memory, parameters, computation)
    return result


def ca_tfm(k=1024, m=None, c=4096, s=512, h=16, l=1):
    '''Compute complexities for CA-Tfm.'''
    m = m or k
    h = _get_h(h, k)
    memory = l * (((2 * k + 2 * m + 6 * c) * s + h * s ** 2) * 4)
    parameters = l * (2 * k * c + 2 * m * c + 2 * c ** 2)
    computation = l * (
        (4 * k * c + 4 * m * c + 5 * c ** 2) * s + (2 * k + 2 * m) * s ** 2
    )
    result = ComplexityBundle(memory, parameters, computation)
    return result
