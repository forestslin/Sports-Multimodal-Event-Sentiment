# Repeated seed experiment summary

Values are mean ± SD across independent training seeds. The 95% confidence interval uses a t interval over seed-level estimates and is descriptive with n = 5.

| Model | mAP@5 s | mAP@10 s | mAP@20 s | Onset MAE@10 s | Parameters |
|---|---:|---:|---:|---:|---:|
| late | 0.260 ± 0.005 | 0.314 ± 0.003 | 0.348 ± 0.002 | 2.767 ± 0.023 | 159,089 |
| rtgre | 0.251 ± 0.005 | 0.307 ± 0.007 | 0.341 ± 0.007 | 2.775 ± 0.012 | 186,929 |
| rtgre_flow | 0.250 ± 0.005 | 0.302 ± 0.004 | 0.338 ± 0.005 | 2.777 ± 0.029 | 186,929 |
| tcn_fusion | 0.223 ± 0.007 | 0.275 ± 0.006 | 0.313 ± 0.009 | 2.864 ± 0.021 | 130,961 |
| tcn_visual | 0.223 ± 0.005 | 0.271 ± 0.006 | 0.305 ± 0.005 | 2.852 ± 0.015 | 106,385 |
| text | 0.049 ± 0.001 | 0.093 ± 0.004 | 0.144 ± 0.006 | 4.689 ± 0.029 | 82,193 |
| visual | 0.251 ± 0.011 | 0.303 ± 0.011 | 0.336 ± 0.009 | 2.759 ± 0.025 | 106,769 |

## Paired differences in mAP@10 s

| Comparison | Mean difference | 95% CI |
|---|---:|---:|
| late minus visual | +0.011 | [-0.000, +0.022] |
| rtgre minus visual | +0.003 | [-0.006, +0.013] |
| rtgre_flow minus visual | -0.001 | [-0.014, +0.012] |
| tcn_visual minus visual | -0.032 | [-0.045, -0.020] |
| tcn_fusion minus visual | -0.028 | [-0.038, -0.018] |
