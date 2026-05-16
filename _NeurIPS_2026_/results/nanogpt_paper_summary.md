# nanoGPT Paper Summary

This file summarizes the best kept run for each `(family, method)` pair.

## 0p7b

- `baseline`: avg=46.18 (delta vs baseline=0.00), model=`xsa-paper-0p7b-baseline-fineweb100bt-ctx2048-gb4096-lr5e-4-s1337`
- `residual_attn_control`: avg=54.68 (delta vs baseline=8.50), model=`xsa-paper-0p7b-attn_para_removal-xfrresidual-xsppost_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb4096-lr5e-4-s1337`
- `residual_mlp_control`: avg=51.69 (delta vs baseline=5.51), model=`xsa-paper-0p7b-attn_para_removal-xfrresidual-xsppost_o_proj-xftmlp-xl0--1-xs1-fineweb100bt-ctx2048-gb256-lr5e-4-s1337`
- `xsa_value_control`: avg=43.56 (delta vs baseline=-2.62), model=`xsa-paper-0p7b-xsa-xfrresidual-xsppre_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb4096-lr5e-4-s1337`
- `gated_value_control`: avg=51.71 (delta vs baseline=5.53), model=`xsa-paper-0p7b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fw-ctx2048-gb4096-lr5e-4-min5e-5-s1337-lrlr10`

## 1p4b

- `baseline`: avg=45.38 (delta vs baseline=0.00), model=`xsa-paper-1p4b-baseline-fineweb100bt-ctx2048-gb256-lr4e-4-s1337`
- `residual_attn_control`: missing stable kept run
- `residual_mlp_control`: missing stable kept run
- `xsa_value_control`: missing stable kept run
- `gated_value_control`: avg=51.33 (delta vs baseline=5.95), model=`xsa-paper-1p4b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-gb8192-lr4e-4-minlr4e-5-s1337-lrlr10`

## 2p7b

- `baseline`: avg=49.05 (delta vs baseline=0.00), model=`xsa-paper-2p7b-baseline-gb8192-lr3e-4-minlr3e-5-s1337-lrlr10`
- `residual_attn_control`: avg=45.49 (delta vs baseline=-3.56), model=`xsa-paper-2p7b-attn_para_removal-xfrresidual-xsppost_o_proj-xftattn-xl0--1-xs1-fineweb100bt-ctx2048-gb8192-lr3e-4-s1337`
- `residual_mlp_control`: missing stable kept run
- `xsa_value_control`: missing stable kept run
- `gated_value_control`: avg=52.96 (delta vs baseline=3.91), model=`xsa-paper-2p7b-axon-hard-kftrue-xfrself_value-xsppre_o_proj-xftattn-xl0--1-xs1-fw-ctx2048-gb8192-lr3e-4-min3e-5-s1337-lrlr10`
