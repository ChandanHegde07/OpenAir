# E57 — latent airport operational regime

K=5, silhouette 0.350

K=3 silhouette 0.319
K=4 silhouette 0.335
K=5 silhouette 0.350

## Persistence (mean run length, 5-min bins)
- EDDF: 128.2 bins (0.1 tr/h)
- EDDM: 103.3 bins (0.1 tr/h)
- EGLL: 121.7 bins (0.1 tr/h)
- EHAM: 90.2 bins (0.1 tr/h)
- LEBL: 114.1 bins (0.1 tr/h)
- LEMD: 90.1 bins (0.1 tr/h)
- LFPG: 105.0 bins (0.1 tr/h)
- LIRF: 132.8 bins (0.1 tr/h)
- LSZH: 119.8 bins (0.1 tr/h)
- LTFM: 70.9 bins (0.2 tr/h)

## Taxi distribution by state
- state 0: n=805758 med=905 p90=1451 >30m=3.3%
- state 1: n=7670 med=829 p90=1254 >30m=2.1%
- state 2: n=1059686 med=952 p90=1496 >30m=3.8%
- state 3: n=187239 med=842 p90=1322 >30m=2.7%
- state 4: n=2224 med=902 p90=1930 >30m=10.9%

State-median matched baseline: janjul 482.8 vs v13 ~238.5

## Verdict
Latent airport regimes EXIST (persist 6-11h, silhouette 0.35) but carry almost
no incremental taxi-out information: state-conditioned taxi distributions are
nearly identical (medians 829-952s; >30m rate 2.1-3.8%), except a rare busy
state (0.1% of flights, >30m 10.9%). State-median baseline matched 482.8 vs
v13 238.5 => no predictive value. Per stop-condition 3/4: REJECT. Regime
conditioning cannot beat v13. No v15.
