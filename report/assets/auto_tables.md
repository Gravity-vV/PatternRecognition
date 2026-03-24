# 自动汇总表（由 experiments/report_summarize.py 生成）

来源 CSV：

- experiments/sweep_all_leaderboard.csv
- experiments/transfer_more_epochs_raw.csv

### 总体 Top-10（按 accuracy）

|#|acc|f1|train(s)|infer(s)|preprocess|feature|classifier|tag|
|---:|---:|---:|---:|---:|---|---|---|---|
|1|0.9359|0.9353|0.64|0.004|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding|logreg|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed__clf-logreg|
|2|0.9346|0.9339|0.27|0.704|normalizer_zscore_vec|convnext_tiny_embedding+pca|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_zscore_vec__feat-convnext_embed_pca__clf-svm_rbf|
|3|0.9340|0.9333|1.26|2.654|normalizer_zscore_vec|convnext_tiny_embedding|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_zscore_vec__feat-convnext_embed__clf-svm_rbf|
|4|0.9340|0.9334|0.27|0.701|normalizer_minmax_vec|convnext_tiny_embedding+pca|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_minmax_vec__feat-convnext_embed_pca__clf-svm_rbf|
|5|0.9340|0.9331|1.14|2.466|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding|svm_rbf|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed__clf-svm_rbf|
|6|0.9340|0.9333|0.27|0.688|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding+pca|svm_rbf|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed_pca__clf-svm_rbf|
|7|0.9338|0.9332|0.19|0.005|torch_augment_224+normalizer_zscore_vec|convnext_tiny_embedding|logreg|sweep_torch_embed__img-torch_augment_224__vec-normalizer_zscore_vec__feat-convnext_embed__clf-logreg|
|8|0.9338|0.9330|0.06|0.001|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding+pca|logreg|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed_pca__clf-logreg|
|9|0.9335|0.9328|1.21|2.501|normalizer_minmax_vec|convnext_tiny_embedding|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_minmax_vec__feat-convnext_embed__clf-svm_rbf|
|10|0.9335|0.9326|0.50|0.004|normalizer_minmax_vec|convnext_tiny_embedding|logreg|sweep_torch_embed__img-none__vec-normalizer_minmax_vec__feat-convnext_embed__clf-logreg|

### Embedding 系列 Top-10

|#|acc|f1|train(s)|infer(s)|preprocess|feature|classifier|tag|
|---:|---:|---:|---:|---:|---|---|---|---|
|1|0.9359|0.9353|0.64|0.004|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding|logreg|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed__clf-logreg|
|2|0.9346|0.9339|0.27|0.704|normalizer_zscore_vec|convnext_tiny_embedding+pca|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_zscore_vec__feat-convnext_embed_pca__clf-svm_rbf|
|3|0.9340|0.9333|1.26|2.654|normalizer_zscore_vec|convnext_tiny_embedding|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_zscore_vec__feat-convnext_embed__clf-svm_rbf|
|4|0.9340|0.9334|0.27|0.701|normalizer_minmax_vec|convnext_tiny_embedding+pca|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_minmax_vec__feat-convnext_embed_pca__clf-svm_rbf|
|5|0.9340|0.9331|1.14|2.466|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding|svm_rbf|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed__clf-svm_rbf|
|6|0.9340|0.9333|0.27|0.688|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding+pca|svm_rbf|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed_pca__clf-svm_rbf|
|7|0.9338|0.9332|0.19|0.005|torch_augment_224+normalizer_zscore_vec|convnext_tiny_embedding|logreg|sweep_torch_embed__img-torch_augment_224__vec-normalizer_zscore_vec__feat-convnext_embed__clf-logreg|
|8|0.9338|0.9330|0.06|0.001|torch_augment_224+normalizer_minmax_vec|convnext_tiny_embedding+pca|logreg|sweep_torch_embed__img-torch_augment_224__vec-normalizer_minmax_vec__feat-convnext_embed_pca__clf-logreg|
|9|0.9335|0.9328|1.21|2.501|normalizer_minmax_vec|convnext_tiny_embedding|svm_rbf|sweep_torch_embed__img-none__vec-normalizer_minmax_vec__feat-convnext_embed__clf-svm_rbf|
|10|0.9335|0.9326|0.50|0.004|normalizer_minmax_vec|convnext_tiny_embedding|logreg|sweep_torch_embed__img-none__vec-normalizer_minmax_vec__feat-convnext_embed__clf-logreg|

### 迁移学习（Transfer）Top-10

|#|acc|f1|train(s)|infer(s)|preprocess|feature|classifier|tag|
|---:|---:|---:|---:|---:|---|---|---|---|
|1|0.9300|0.9285|244.50|47.451|torch_augment_224+imagenet_normalize|none|TorchTransferClassifier({"backbone": "convnext_tiny", "batch_size": 32, "device": "mps", "epochs": 5, "lr": 0.0005, "mode": "partial", "pretrained": true, "seed": 0})|transfer_partial_e5_aug|
|2|0.9283|0.9275|142.46|37.037|torch_augment_224+imagenet_normalize|none|TorchTransferClassifier({"backbone": "convnext_tiny", "batch_size": 32, "device": "mps", "epochs": 5, "lr": 0.001, "mode": "head", "pretrained": true, "seed": 0})|transfer_head_e5_aug|
|3|0.9223|0.9209|72.35|65.880|none|none|TorchTransferClassifier({"backbone": "convnext_tiny", "batch_size": 32, "device": "mps", "epochs": 1, "lr": 0.001, "mode": "head", "pretrained": true, "seed": 0})|sweep_torch_transfer__img-none__feat-none__clf-convnext_tiny_transfer|
|4|0.9188|0.9175|67.11|66.170|torch_augment_224|none|TorchTransferClassifier({"backbone": "convnext_tiny", "batch_size": 32, "device": "mps", "epochs": 1, "lr": 0.001, "mode": "head", "pretrained": true, "seed": 0})|sweep_torch_transfer__img-torch_augment_224__feat-none__clf-convnext_tiny_transfer|

### 传统/手工特征（Numpy）Top-10

|#|acc|f1|train(s)|infer(s)|preprocess|feature|classifier|tag|
|---:|---:|---:|---:|---:|---|---|---|---|
|1|0.1235|0.1197|6.62|15.723|normalizer_zscore_image+normalizer_minmax_vec|hog|svm_rbf|sweep_numpy__img-normalizer_zscore_image__vec-normalizer_minmax_vec__feat-hog__clf-svm_rbf|
|2|0.1216|0.1180|3.28|0.955|normalizer_zscore_image|hog+pca|svm_rbf|sweep_numpy__img-normalizer_zscore_image__vec-none__feat-hog_pca__clf-svm_rbf|
|3|0.1213|0.1171|6.54|15.871|normalizer_zscore_image|hog|svm_rbf|sweep_numpy__img-normalizer_zscore_image__vec-none__feat-hog__clf-svm_rbf|
|4|0.1202|0.1165|6.55|15.922|normalizer_zscore_image+normalizer_zscore_vec|hog|svm_rbf|sweep_numpy__img-normalizer_zscore_image__vec-normalizer_zscore_vec__feat-hog__clf-svm_rbf|
|5|0.1202|0.1171|3.28|0.954|normalizer_zscore_image+normalizer_zscore_vec|hog+pca|svm_rbf|sweep_numpy__img-normalizer_zscore_image__vec-normalizer_zscore_vec__feat-hog_pca__clf-svm_rbf|
|6|0.1202|0.1177|3.28|0.983|normalizer_zscore_image+normalizer_minmax_vec|hog+pca|svm_rbf|sweep_numpy__img-normalizer_zscore_image__vec-normalizer_minmax_vec__feat-hog_pca__clf-svm_rbf|
|7|0.1142|0.1100|6.81|15.794|normalizer_minmax_vec|hog|svm_rbf|sweep_numpy__img-none__vec-normalizer_minmax_vec__feat-hog__clf-svm_rbf|
|8|0.1142|0.1100|8.76|14.969|normalizer_minmax_image+normalizer_minmax_vec|hog|svm_rbf|sweep_numpy__img-normalizer_minmax_image__vec-normalizer_minmax_vec__feat-hog__clf-svm_rbf|
|9|0.1131|0.1110|3.52|0.953|normalizer_zscore_vec|hog+pca|svm_rbf|sweep_numpy__img-none__vec-normalizer_zscore_vec__feat-hog_pca__clf-svm_rbf|
|10|0.1131|0.1110|5.52|1.507|normalizer_minmax_image+normalizer_zscore_vec|hog+pca|svm_rbf|sweep_numpy__img-normalizer_minmax_image__vec-normalizer_zscore_vec__feat-hog_pca__clf-svm_rbf|

