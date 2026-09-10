# E42 — flight identity residual prior

- selected: key=['CALLSIGN', 'wd_bucket', 'hour_bucket'] lambda=1.0 alpha=1.0 coverage=0.36

| Model | Jan+Jul Overall | Jan+Jul Matched | Top-5% SSE | Dec Matched |
|---|---:|---:|---:|---:|
| E20 | 368.03 | 244.76 | 1.264e+10 | 214.78 |
| Best E42 | - | 249.64 | 1.289e+10 | 223.58 |

JAN+JUL GAIN: -4.88
TOP-5% SSE GAIN: -2.0%
DEC GAIN: -8.81

GO / NO-GO: NO-GO
