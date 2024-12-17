# Benchmark

## Multi-evaluations for Univariate Polynomials

Run `cargo bench --bench multi_evals`:

```
Univariate Multi-evaluations/::Fast: Deg=2^10, |Domain|=2^12
                        time:   [1.3931 s 1.4172 s 1.4427 s]
Univariate Multi-evaluations/::Naive: Deg=2^10, |Domain|=2^12
                        time:   [8.4841 s 8.6339 s 8.8074 s]

Univariate Multi-evaluations/::Fast: Deg=2^10, |Domain|=2^15
                        time:   [9.8249 s 10.015 s 10.213 s]
Univariate Multi-evaluations/::Naive: Deg=2^10, |Domain|=2^15
                        time:   [71.133 s 71.767 s 72.791 s]
```
