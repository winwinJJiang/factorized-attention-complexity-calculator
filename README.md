# factorized-attention-complexity-calculator

To use the complexity calculator

``` bash
    python3
```

Example usage:

``` python3
    import c
    c.fa_dot(s=64 * 64, c=32)
    c.ca_tfm(l=24, s=512)
```

Arguments for functions in the module:

- `b`: basis dimensionality
- `k`: channel count of key
- `bk`: b or k
- `m`: channel count of value
- `c`: channel count of input
- `s`: spatiotemporal size of input
- `h`: number of heads
- `l`: layers (for Transformers)

Public interface:

- `res_block()`: compute complexities for ResBlock.
- `res_block_bot()`: compute complexities for bottleneck ResBlock.
- `fa_dot()`: compute complexities for FA-dot.
- `ca_dot()`: compute complexities for CA-dot.
- `fa_bot()`: compute complexities for FA-bot.
- `ca_bot()`: compute complexities for CA-bot.
- `fa_multi()`: compute complexities for FA-multi.
- `ca_multi()`: compute complexities for CA-multi.
- `fa_tfm()`: compute complexities for FA-Tfm.
- `ca_tfm()`: compute complexities for CA-Tfm.
