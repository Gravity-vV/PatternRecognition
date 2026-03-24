# Multi-seed 稳定性补充实验（best config × 3 seeds）

- seeds: [0, 1, 2]
- embedding 来源 tag（seed=0 的 sweep 最优 logreg）: sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed__clf-logreg
- transfer 来源 tag（seed=0 的 5 epochs 最优）: transfer_partial_e5_aug
- transfer 复现参数（关键项）: mode=partial, epochs=5, lr=0.0005

## 分 seed 结果

|方法|split_seed|acc|precision|recall|f1|train(s)|infer(s)|
|---|---:|---:|---:|---:|---:|---:|---:|
|Embedding+LogReg（best）|0|0.9359|0.9370|0.9356|0.9353|89.10|93.594|
|Embedding+LogReg（best）|1|0.9343|0.9353|0.9340|0.9334|36.11|36.350|
|Embedding+LogReg（best）|2|0.9365|0.9373|0.9362|0.9358|53.90|74.517|
|Transfer（best, 5 epochs）|0|0.9264|0.9310|0.9254|0.9248|250.10|36.522|
|Transfer（best, 5 epochs）|1|0.9278|0.9375|0.9278|0.9267|227.28|40.896|
|Transfer（best, 5 epochs）|2|0.9218|0.9285|0.9211|0.9189|484.39|73.209|

## 汇总（均值±标准差）

|方法|acc|precision|recall|f1|train(s)|infer(s)|
|---|---:|---:|---:|---:|---:|---:|
|Embedding+LogReg（best）|0.9356±0.0011|0.9365±0.0011|0.9353±0.0011|0.9348±0.0013|59.70±26.97|68.153±29.148|
|Transfer（best, 5 epochs）|0.9253±0.0031|0.9323±0.0047|0.9247±0.0034|0.9235±0.0041|320.59±142.31|50.209±20.038|
