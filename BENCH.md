# Benchmark

_note_: all following data are run on an Apple M1 with 16GB RAM.

## MultiEval for Univariate Polynomials

Run `cargo bench --bench multi_evals`:

```
Univariate MultiEval/Fast: Deg=2^10, |Domain|=2^12
                        time:   [1.3931 s 1.4172 s 1.4427 s]
Univariate MultiEval/Naive: Deg=2^10, |Domain|=2^12
                        time:   [8.4841 s 8.6339 s 8.8074 s]

Univariate MultiEval/Fast: Deg=2^10, |Domain|=2^15
                        time:   [9.8249 s 10.015 s 10.213 s]
Univariate MultiEval/Naive: Deg=2^10, |Domain|=2^15
                        time:   [71.133 s 71.767 s 72.791 s]
```

## MultiPartialEval for Bivariate Polynomials

```
Bivariate MultiPartialEval/Fast: deg_x=8, deg_y=2^10, |Domain|=2^11
                        time:   [2.2215 s 2.2650 s 2.3067 s]
Bivariate MultiPartialEval/Naive: deg_x=8, deg_y=2^10, |Domain|=2^11
                        time:   [38.202 s 38.432 s 38.683 s]
```

In comparison to ADVZ-style:
```
Start:   AdvzVRS::compute shares (k=1024, n=2048, L=8)
··Start:   encode data
··End:     encode data .............................................................1.469ms
··Start:   commit row_poly
··End:     commit row_poly .........................................................24.137ms
··Start:   multi-eval on agg_poly
··End:     multi-eval on agg_poly ..................................................895.739ms
End:     AdvzVRS::compute shares (k=1024, n=2048, L=8) .............................931.131ms
```

## Bivariate KZG

Run `cargo bench --bench bkzg`, here's the end-to-end result for different sizes

```
bKZG::e2e::deg_x=8,deg_y=16384/commit
                        time:   [216.53 ms 219.02 ms 221.78 ms]
bKZG::e2e::deg_x=8,deg_y=16384/eval
                        time:   [231.26 ms 303.59 ms 402.78 ms]
bKZG::e2e::deg_x=8,deg_y=16384/verify
                        time:   [1.5060 ms 1.5132 ms 1.5355 ms]
bKZG::e2e::deg_x=8,deg_y=16384/parital_eval
                        time:   [228.94 ms 567.71 ms 1.0042 s]
bKZG::e2e::deg_x=8,deg_y=16384/verify_parital
                        time:   [27.192 ms 27.527 ms 27.917 ms]
====
bKZG::e2e::deg_x=8,deg_y=32768/commit
                        time:   [390.47 ms 396.10 ms 401.84 ms]
bKZG::e2e::deg_x=8,deg_y=32768/eval
                        time:   [401.53 ms 403.25 ms 405.07 ms]
bKZG::e2e::deg_x=8,deg_y=32768/verify
                        time:   [1.4603 ms 1.4641 ms 1.4714 ms]
bKZG::e2e::deg_x=8,deg_y=32768/parital_eval
                        time:   [353.69 ms 355.08 ms 356.29 ms]
bKZG::e2e::deg_x=8,deg_y=32768/verify_parital
                        time:   [49.060 ms 49.196 ms 49.431 ms]
====
bKZG::e2e::deg_x=8,deg_y=65536/commit
                        time:   [709.22 ms 719.14 ms 731.79 ms]
bKZG::e2e::deg_x=8,deg_y=65536/eval
                        time:   [760.11 ms 768.93 ms 777.99 ms]
bKZG::e2e::deg_x=8,deg_y=65536/verify
                        time:   [1.4701 ms 1.4738 ms 1.4760 ms]
bKZG::e2e::deg_x=8,deg_y=65536/parital_eval
                        time:   [651.79 ms 658.86 ms 667.48 ms]
bKZG::e2e::deg_x=8,deg_y=65536/verify_parital
                        time:   [90.557 ms 91.575 ms 92.725 ms]
====
bKZG::e2e::deg_x=16,deg_y=16384/commit
                        time:   [385.74 ms 389.55 ms 393.96 ms]
bKZG::e2e::deg_x=16,deg_y=16384/eval
                        time:   [412.07 ms 419.98 ms 429.05 ms]
bKZG::e2e::deg_x=16,deg_y=16384/verify
                        time:   [1.4837 ms 1.4929 ms 1.5003 ms]
bKZG::e2e::deg_x=16,deg_y=16384/parital_eval
                        time:   [381.00 ms 480.21 ms 620.36 ms]
bKZG::e2e::deg_x=16,deg_y=16384/verify_parital
                        time:   [27.211 ms 27.611 ms 27.918 ms]
====
bKZG::e2e::deg_x=16,deg_y=32768/commit
                        time:   [711.31 ms 714.78 ms 718.54 ms]
bKZG::e2e::deg_x=16,deg_y=32768/eval
                        time:   [743.51 ms 748.07 ms 753.09 ms]
bKZG::e2e::deg_x=16,deg_y=32768/verify
                        time:   [1.4724 ms 1.4742 ms 1.4780 ms]
bKZG::e2e::deg_x=16,deg_y=32768/parital_eval
                        time:   [695.88 ms 702.46 ms 710.42 ms]
bKZG::e2e::deg_x=16,deg_y=32768/verify_parital
                        time:   [56.114 ms 66.489 ms 83.541 ms]
====
bKZG::e2e::deg_x=16,deg_y=65536/commit
                        time:   [1.3235 s 1.5084 s 1.8608 s]
bKZG::e2e::deg_x=16,deg_y=65536/eval
                        time:   [1.3923 s 1.4047 s 1.4202 s]
bKZG::e2e::deg_x=16,deg_y=65536/verify
                        time:   [1.4623 ms 1.4701 ms 1.4755 ms]
bKZG::e2e::deg_x=16,deg_y=65536/parital_eval
                        time:   [1.2896 s 1.3902 s 1.5016 s]
bKZG::e2e::deg_x=16,deg_y=65536/verify_parital
                        time:   [89.843 ms 89.954 ms 90.055 ms]
====
bKZG::e2e::deg_x=32,deg_y=16384/commit
                        time:   [747.21 ms 843.83 ms 1.0320 s]
bKZG::e2e::deg_x=32,deg_y=16384/eval
                        time:   [776.74 ms 792.16 ms 816.40 ms]
bKZG::e2e::deg_x=32,deg_y=16384/verify
                        time:   [1.4682 ms 1.4701 ms 1.4725 ms]
bKZG::e2e::deg_x=32,deg_y=16384/parital_eval
                        time:   [755.25 ms 769.45 ms 787.61 ms]
bKZG::e2e::deg_x=32,deg_y=16384/verify_parital
                        time:   [28.224 ms 28.661 ms 29.515 ms]
====
bKZG::e2e::deg_x=32,deg_y=32768/commit
                        time:   [1.4040 s 1.4395 s 1.4783 s]
bKZG::e2e::deg_x=32,deg_y=32768/eval
                        time:   [1.4383 s 1.4982 s 1.6010 s]
bKZG::e2e::deg_x=32,deg_y=32768/verify
                        time:   [1.4870 ms 1.4882 ms 1.4899 ms]
bKZG::e2e::deg_x=32,deg_y=32768/parital_eval
                        time:   [1.3716 s 1.3743 s 1.3772 s]
bKZG::e2e::deg_x=32,deg_y=32768/verify_parital
                        time:   [49.139 ms 49.432 ms 49.588 ms]
====
bKZG::e2e::deg_x=32,deg_y=65536/commit
                        time:   [2.5720 s 2.6744 s 2.8218 s]
bKZG::e2e::deg_x=32,deg_y=65536/eval
                        time:   [2.6466 s 2.6666 s 2.6889 s]
bKZG::e2e::deg_x=32,deg_y=65536/verify
                        time:   [1.4776 ms 1.4786 ms 1.4806 ms]
bKZG::e2e::deg_x=32,deg_y=65536/parital_eval
                        time:   [2.5244 s 2.5357 s 2.5494 s]
bKZG::e2e::deg_x=32,deg_y=65536/verify_parital
                        time:   [89.736 ms 90.113 ms 90.433 ms]
```

### ProofGen for PartialEval along X v.s. Y

```
ℹ️ PartialEval of bivariate poly, deg_x=1024, deg_y=1024
bKZG::PartialEval/at X  time:   [2.4542 s 2.4820 s 2.5110 s]
bKZG::PartialEval/at Y  time:   [2.4231 s 2.4691 s 2.5164 s]
```

## Auxiliary 

The fast version of `fft_rev()` v.s. naive `fft(v.rev())`:
```
fft_rev::deg=8192,domain_size=16384/fast
                        time:   [339.30 µs 340.06 µs 341.19 µs]
fft_rev::deg=8192,domain_size=16384/naive
                        time:   [1.7861 ms 1.7899 ms 1.7931 ms]

fft_rev::deg=8192,domain_size=32768/fast
                        time:   [591.86 µs 594.32 µs 597.31 µs]
fft_rev::deg=8192,domain_size=32768/naive
                        time:   [2.8845 ms 2.8932 ms 2.9038 ms]

fft_rev::deg=8192,domain_size=65536/fast
                        time:   [1.2275 ms 1.2457 ms 1.2746 ms]
fft_rev::deg=8192,domain_size=65536/naive
                        time:   [4.6611 ms 4.7637 ms 4.9163 ms]
```
