# nanoGPT Result Cleaning Summary

Cleaning logic:
- For duplicate rows with the same `model`, keep the most complete row; break ties by preferring a healthier `lambada_openai` band, then higher `avg`.
- `keep`: canonical row, at least 6 populated task metrics, and `lambada_openai <= 100`.
- `review`: canonical row, at least 6 populated task metrics, and `100 < lambada_openai <= 1000`. These are not catastrophic, but they look weaker than the clearly converged rows.
- `exclude`: canonical row with `lambada_openai > 1000`, or canonical row with fewer than 6 populated task metrics, or any non-canonical duplicate row.

Overall counts (canonical rows only):
- Raw rows: 82
- Unique models after de-duplication: 49
- `keep`: 16
- `review`: 4
- `exclude` for instability: 19
- `exclude` for incompleteness: 10

## 0p7b
- Unique models: 21
- `keep`: 6
- `review`: 4
- `exclude` unstable: 7
- `exclude` incomplete: 4
- Best `keep` rows by `avg`:
  - `xsa-paper-0p7b-attn_para_removal-xfrresidual-xsppost_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb4096-lr5e-4-s1337`: avg=54.68, lambada=11.7903, nonempty=8
  - `xsa-paper-0p7b-attn_para_removal-xfrresidual-xsppost_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr5e-4-s1337`: avg=52.71, lambada=15.2432, nonempty=8
  - `xsa-paper-0p7b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fw-ctx2048-gb4096-lr5e-4-min5e-5-s1337-lrlr10`: avg=51.71, lambada=16.7727, nonempty=8
  - `xsa-paper-0p7b-attn_para_removal-xfrresidual-xsppost_o_proj-xftmlp-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr5e-4-s1337`: avg=51.69, lambada=17.2247, nonempty=8
  - `xsa-paper-0p7b-baseline-fineweb100bt-ctx2048-gb4096-lr5e-4-s1337`: avg=46.18, lambada=22.296, nonempty=8
- `review` rows:
  - `xsa-paper-0p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-var-gamma-h1p0-xl0--1-xs1-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10`: avg=45.89, lambada=113.4279, nonempty=8
  - `xsa-paper-0p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-var-gamma-h0p9-xl0--1-xs1-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10`: avg=45.44, lambada=183.6086, nonempty=8
  - `xsa-paper-0p7b-xsa-lgate`: avg=43.88, lambada=302.5456, nonempty=8
  - `xsa-paper-0p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr5e-4-s1337`: avg=44.73, lambada=692.3778, nonempty=8
- Representative unstable rows:
  - `xsa-paper-0p7b-baseline-fineweb100bt-ctx2048-gb2048-lr5e-4-s1337`: avg=41.1, lambada=1170.3582, nonempty=8
  - `xsa-paper-0p7b-baseline-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10`: avg=38.79, lambada=704019717240557.5, nonempty=8
  - `xsa-paper-0p7b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10`: avg=38.87, lambada=706826072815081.0, nonempty=8
  - `xsa-paper-0p7b-xsa-xfrresidual-xsppre_o_proj-xftattn-xl0--1-xs1-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10`: avg=38.83, lambada=706963254002970.0, nonempty=8
  - `xsa-paper-0p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-xlg1p0-xl0--1-xs1-gb4096-lr5e-4-minlr5e-5-s1337-lrlr10`: avg=38.84, lambada=711332626382322.1, nonempty=8

## 1p4b
- Unique models: 10
- `keep`: 3
- `review`: 0
- `exclude` unstable: 7
- `exclude` incomplete: 0
- Best `keep` rows by `avg`:
  - `xsa-paper-1p4b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-gb8192-lr4e-4-minlr4e-5-s1337-lrlr10`: avg=51.33, lambada=15.1088, nonempty=8
  - `xsa-paper-1p4b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-gb4096-lr4e-4-minlr4e-5-s1337-lrlr10`: avg=49.47, lambada=34.0392, nonempty=8
  - `xsa-paper-1p4b-baseline-fineweb100bt-ctx2048-gb256-lr4e-4-s1337`: avg=45.38, lambada=9.815, nonempty=6
- Representative unstable rows:
  - `xsa-paper-1p4b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-bsz4-ga128-gb8192-lr4e-4-minlr4e-5-s1337-lrlr10`: avg=42.47, lambada=1628.4774, nonempty=8
  - `xsa-paper-1p4b-xsa-lgate`: avg=43.99, lambada=1803.5552, nonempty=8
  - `xsa-paper-1p4b-attn_para_removal-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr4e-4-s1337`: avg=43.71, lambada=1946.7583, nonempty=8
  - `xsa-paper-1p4b-xsa-xfrself_value-xsppre_o_proj-xftattn-var-gamma-h1p0-xl0--1-xs1-gb8192-lr4e-4-minlr4e-5-s1337-lrlr10`: avg=44.87, lambada=2633.79, nonempty=8
  - `xsa-paper-1p4b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb8192-lr4e-4-s1337`: avg=43.17, lambada=8440.7312, nonempty=8

## 2p7b
- Unique models: 12
- `keep`: 7
- `review`: 0
- `exclude` unstable: 5
- `exclude` incomplete: 0
- Best `keep` rows by `avg`:
  - `xsa-paper-2p7b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fw-ctx2048-gb8192-lr3e-4-min3e-5-s1337-lrlr10`: avg=52.96, lambada=15.1051, nonempty=8
  - `xsa-paper-2p7b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-gb8192-lr3e-4-minlr3e-5-s1337-lrlr10`: avg=49.29, lambada=29.9633, nonempty=8
  - `xsa-paper-2p7b-baseline-gb8192-lr3e-4-minlr3e-5-s1337-lrlr10`: avg=49.05, lambada=8.5781, nonempty=8
  - `xsa-paper-2p7b-baseline-fineweb100bt-ctx2048-gb8192-lr3e-4-s1337`: avg=49.02, lambada=9.3816, nonempty=8
  - `xsa-paper-2p7b-baseline-fineweb100bt-ctx2048-gb256-lr3e-4-s1337`: avg=46.21, lambada=16.6395, nonempty=8
- Representative unstable rows:
  - `xsa-paper-2p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr3e-4-s1337`: avg=42.81, lambada=51897.1404, nonempty=8
  - `xsa-paper-2p7b-xsa-lgate`: avg=44.05, lambada=111102.4292, nonempty=8
  - `xsa-paper-2p7b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fw-ctx2048-gb4096-lr3e-4-min3e-5-s1337-lrlr10`: avg=42.01, lambada=1.6796286670076563e+20, nonempty=8
  - `xsa-paper-2p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-var-gamma-h1p0-xl0--1-xs1-gb8192-lr3e-4-minlr3e-5-s1337-lrlr10`: avg=41.96, lambada=1.7140799151318178e+20, nonempty=8
  - `xsa-paper-2p7b-xsa-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-gb8192-lr3e-4-minlr3e-5-s1337-lrlr10`: avg=41.96, lambada=1.7140799151318178e+20, nonempty=8

## other
- Unique models: 6
- `keep`: 0
- `review`: 0
- `exclude` unstable: 0
- `exclude` incomplete: 6

