# 最优模型 Top-K 易混淆类别统计

最优模型 tag：sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed__clf-logreg（accuracy=0.9359）

说明：混淆矩阵元素 $C_{ij}$ 表示真实类别 $i$ 被预测为 $j$ 的样本数。本文同时给出：
- **方向性混淆**：$i\to j$ 的行归一化比例 $C_{ij}/\sum_j C_{ij}$（更能反映该类被错分到哪一类）。
- **类别对混淆**：$(a,b)$ 的双向混淆比例之和（$a\to b$ 与 $b\to a$）。

## 方向性 Top-5（按行归一化比例）

|#|true|pred|比例(行归一化)|错分数|
|---:|---|---|---:|---:|
|1|American Pit Bull Terrier|Staffordshire Bull Terrier|0.2000|20|
|2|Ragdoll|Birman|0.1300|13|
|3|Egyptian Mau|Bengal|0.1237|12|
|4|Staffordshire Bull Terrier|American Pit Bull Terrier|0.1124|10|
|5|American Pit Bull Terrier|American Bulldog|0.0900|9|

## 类别对 Top-5（双向混淆之和）

|#|类别A|类别B|双向比例和|双向错分数|
|---:|---|---|---:|---:|
|1|American Pit Bull Terrier|Staffordshire Bull Terrier|0.3124|30|
|2|Birman|Ragdoll|0.2200|22|
|3|Bengal|Egyptian Mau|0.1637|16|
|4|Birman|Siamese|0.1200|12|
|5|Abyssinian|Bengal|0.1112|11|

