# DeepFake TTA

Codebase tach tu 5 notebook:

- tao corruption images: blur, resize, color saturation, color contrast
- trich CLIP features bang `ViT-L-14/openai`
- train linear probe tren FF++
- evaluate linear probe va Tip-Adapter tren FF++ / CelebDFv1 / corruption levels

## Cai dat

```bash
pip install -r requirements.txt
```

Hoac cai editable:

```bash
pip install -e .
```

## Tao anh corruption

Tuong duong notebook `gen-instances-ff-color-saturation.ipynb`:

```bash
python -m deepfake_tta.cli generate-corruptions \
  --input-dir /kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench/Celeb-DF-v1 \
  --output-root /kaggle/working/processed_output \
  --dataset-name Celeb-DF-v1 \
  --methods color_saturation \
  --levels 5
```

## Trich CLIP features cho mot dataset/path

Tuong duong notebook `clip-embeddings-ffpp-train.ipynb`:

```bash
python -m deepfake_tta.cli extract-features \
  --csv-path /kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv \
  --deepfakebench-root /kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench \
  --dataset-name FaceForensics++ \
  --replacement-root /kaggle/input/datasets/vohoanghoavien/ff-color-contrast-5/color_contrast \
  --output-path /kaggle/working/ffpp-color-contrast-5.pt
```

Voi CelebDFv1 transformed path, them `--original-root`:

```bash
python -m deepfake_tta.cli extract-features \
  --csv-path /kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv \
  --deepfakebench-root /kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench \
  --dataset-name Celeb-DF-v1 \
  --original-root /kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench/Celeb-DF-v1 \
  --replacement-root /kaggle/input/datasets/vohoanghoavien/celebdfv1-color-contrast-5/processed_output/color_contrast/level_5 \
  --output-path /kaggle/working/celebdfv1-color-contrast-5.pt \
  --transform-name color_contrast \
  --transform-level 5
```

## Trich features cho tat ca CelebDFv1 corruptions cung mot level

Tuong duong notebook `clip-embeddings-deepfake-celebdf.ipynb`:

```bash
python -m deepfake_tta.cli extract-celebdf-corruptions \
  --csv-path /kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv \
  --deepfakebench-root /kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench \
  --processed-root /kaggle/input/datasets/elisevo/celebdfv1-level-1/processed_output \
  --output-dir /kaggle/working/celebdfv1_level1_features \
  --level 1
```

Trich features cho tat ca 5 corruption levels cua CelebDFv1:

```bash
python -m deepfake_tta.cli extract-celebdf-corruption-levels \
  --csv-path /kaggle/input/datasets/jamestashvik/deepfakebench/deepfakebench_dataset.csv \
  --deepfakebench-root /kaggle/input/datasets/jamestashvik/deepfakebench/DeepFakeBench \
  --processed-root /kaggle/input/datasets/elisevo/celebdfv1-level-1/processed_output \
  --output-dir /kaggle/working/celebdfv1_all_level_features \
  --levels 1 2 3 4 5
```

## Train va evaluate

Tuong duong notebook `df-tip-adapter.ipynb`:

```bash
python -m deepfake_tta.cli train-eval \
  --train-features /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp_train_features.pt \
  --test-features \
    /kaggle/input/datasets/vhonghoavin/deepfakebench-features/celebdfv1-color-constrast-5.pt \
    /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp-color-constrast-5.pt \
  --model-output /kaggle/working/ufd_linear_probe.pt \
  --results-output /kaggle/working/train_eval_results.csv
```

Chon TTA method bang `--tta-methods`. Cac method hien co:

- `tip_adapter`: source cache adapter tu notebook goc
- `boost_adapter`: source cache + confident online target cache
- `crg`: cache + residual prototype alignment + Gaussian modeling
- `dmn`: static source memory + dynamic target memory
- `dpe`: dual prototype evolving tren feature space
- `bca`: feature-space Bayesian class prior/prototype adaptation
- `dota`: online diagonal Gaussian distribution adaptation
- `freetta`: source-free online EM adaptation tren test stream
- `dynaprompt`: prompt-free proxy bang online logit calibration
- `compact_cache_adapter`: compact source cache, khong random
- `online_confident_cache_adapter`: source cache confident + online dynamic target cache
- `prototype_linear_tta`: prototype anchor + update linear head

```bash
python -m deepfake_tta.cli train-eval \
  --train-features /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp_train_features.pt \
  --test-features /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp-color-constrast-5.pt \
  --tta-methods tip_adapter boost_adapter crg dmn dpe dota freetta bca dynaprompt
```

## Kaggle: chay tat ca phuong phap bang bash

Clone branch `dev` va cai editable:

```bash
git clone -b dev https://github.com/hoavien0110/training-free-tta-for-deepfake-detection.git
cd training-free-tta-for-deepfake-detection
pip install -q -e . --no-deps
```

Kiem tra GPU va feature files:

```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY

ls /kaggle/input/datasets/vhonghoavin/deepfakebench-features | head
```

Neu da co linear probe checkpoint, chay evaluate tat ca method tren 5 levels:

```bash
python -m deepfake_tta.cli eval-corruption-levels \
  --feature-dir /kaggle/input/datasets/vhonghoavin/deepfakebench-features \
  --train-features /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp_train_features.pt \
  --load-model /kaggle/working/ufd_linear_probe_ffpp_all_levels.pt \
  --levels 1 2 3 4 5 \
  --corruptions color_contrast color_saturation gaussian_blur resize \
  --tta-methods \
    tip_adapter \
    boost_adapter \
    compact_cache_adapter \
    online_confident_cache_adapter \
    crg \
    dmn \
    dpe \
    dota \
    freetta \
    bca \
    dynaprompt \
    prototype_linear_tta \
  --method-cache-dir /kaggle/working/tta_method_cache \
  --continue-on-error \
  --device cuda \
  --results-output /kaggle/working/all_methods_results.csv
```

Neu chua co checkpoint, bo `--load-model`; CLI se train linear probe mot lan roi save vao `--model-output`:

```bash
python -m deepfake_tta.cli eval-corruption-levels \
  --feature-dir /kaggle/input/datasets/vhonghoavin/deepfakebench-features \
  --train-features /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp_train_features.pt \
  --model-output /kaggle/working/ufd_linear_probe_ffpp_all_levels.pt \
  --levels 1 2 3 4 5 \
  --corruptions color_contrast color_saturation gaussian_blur resize \
  --tta-methods tip_adapter boost_adapter compact_cache_adapter online_confident_cache_adapter bca dpe \
  --method-cache-dir /kaggle/working/tta_method_cache \
  --continue-on-error \
  --device cuda \
  --results-output /kaggle/working/all_methods_results.csv
```

Doc nhanh ket qua:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("/kaggle/working/all_methods_results.csv")
print(df.sort_values(["level", "corruption", "method"]).head(30))
print(df.groupby("method")[["acc", "f1", "auc", "ap", "eer"]].mean(numeric_only=True).sort_values("f1", ascending=False))
PY
```

## Evaluate CelebDFv1 corruption theo level

Tuong duong notebook `df-tip-adapter-celebdfv1.ipynb`:

```bash
python -m deepfake_tta.cli eval-corruptions \
  --feature-dir /kaggle/input/datasets/vhonghoavin/deepfakebench-features \
  --train-features /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp_train_features.pt \
  --level 3 \
  --model-output /kaggle/working/ufd_linear_probe_ffpp_level3.pt \
  --results-output /kaggle/working/celebdfv1_level3_corruption_results.csv
```

Evaluate CelebDFv1 tren ca 5 corruption levels, gom ket qua vao mot CSV:

```bash
python -m deepfake_tta.cli eval-corruption-levels \
  --feature-dir /kaggle/input/datasets/vhonghoavin/deepfakebench-features \
  --train-features /kaggle/input/datasets/vhonghoavin/deepfakebench-features/ffpp_train_features.pt \
  --levels 1 2 3 4 5 \
  --tta-methods tip_adapter boost_adapter online_confident_cache_adapter crg dmn dpe dota freetta bca dynaprompt \
  --load-model /kaggle/working/ufd_linear_probe_ffpp_all_levels.pt \
  --method-cache-dir /kaggle/working/tta_method_cache \
  --continue-on-error \
  --model-output /kaggle/working/ufd_linear_probe_ffpp_all_levels.pt \
  --results-output /kaggle/working/celebdfv1_all_level_corruption_results.csv
```
