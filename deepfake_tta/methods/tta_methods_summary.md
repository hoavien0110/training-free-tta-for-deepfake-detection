# TTA Methods Summary

File này tóm tắt 24 phương pháp đang chạy trong codebase, gồm `linear_probe` baseline và 23 TTA methods trong `registry.py`.

Quy ước label:

- `0 = REAL`
- `1 = FAKE`

Các setting dưới đây là default hiện tại khi gọi qua `create_tta_method(...)` trong `deepfake_tta/methods/registry.py`, với batch size test lấy từ CLI (`--tip-test-batch-size` trong `cli.py` hoặc `--test-batch-size` trong `cli_2.py`, mặc định thường là `512`).

| # | Method | Nhóm | Cache/storage | Online update | Update model | Setting chính | Mô tả ngắn |
|---:|---|---|---|---|---|---|---|
| 1 | `linear_probe` | Baseline | Không cache | Không | Không | `Linear(768 -> 1)`, `threshold=0.5` | Classifier tuyến tính trên CLIP feature. Baseline luôn chạy trước các TTA methods. |
| 2 | `tip_adapter` | Cache-based | Full train cache | Không | Không | `alpha=0.5`, `beta=5.5`, `cache_batch_size=8192` | Query test feature vào toàn bộ train cache rồi blend với linear probe. |
| 3 | `boost_adapter` | Cache-based approximation | Full train cache | Không | Không | `alpha=0.5`, `beta=5.5`, `cache_batch_size=8192` | Bản approximation trong repo: dùng source/train cache để boost probability. Không phải BoostAdapter paper-exact test-cache version. |
| 4 | `compact_cache_adapter` | Compact cache | Cache vừa, hiện `50% train` | Không | Không | `cache_ratio=0.5`, `alpha=0.5`, `beta=5.5`, `top_k=64`, `prior_correction=True` | Chọn subset cache đại diện theo centroid từng class, rồi query top-k cache. |
| 5 | `online_confident_cache_adapter` | Online cache | `10% train cache` + dynamic test cache | Có. Sau mỗi batch, thêm test samples tự tin/entropy thấp vào dynamic cache theo pseudo-label. | Không | `source_cache_ratio=0.1`, `alpha_source=0.2`, `alpha_dynamic=0.25`, `add_threshold=0.9`, `max_entropy=0.35`, `use_threshold=0.85`, `max_dynamic_per_class=2048` | Chọn source cache tự tin/entropy thấp, rồi thêm test samples tự tin vào dynamic cache. |
| 6 | `online_cache_10_balanced` | Online cache + balance | `10% train cache` + bounded test cache | Có. Update dynamic test cache giống online cache, rồi wrapper chỉnh batch prior về `50/50`. | Không | `source_cache_ratio=0.1`, `alpha_source=0.25`, `alpha_dynamic=0.2`, `add_threshold=0.92`, `max_entropy=0.3`, `use_threshold=0.88`, `max_dynamic_per_class=1024`, balance strength `0.6` | Bản nhẹ hơn của online cache, có chỉnh prior batch về gần `50/50`. |
| 7 | `crg` | Cache + Gaussian | Full train cache + Gaussian stats | Không | Không | `beta=5.5`, `batch_size=512` | Full cache giống Tip-Adapter, thêm Gaussian mean/variance per class để hiệu chỉnh residual. |
| 8 | `dmn` | Dual memory | Full train cache + dynamic test cache | Có. Thêm confident test samples vào dynamic memory cho các batch sau. | Không | `beta=5.5`, `batch_size=512` | Static memory từ train và dynamic memory từ test stream. Mạnh nhưng memory lớn. |
| 9 | `dota` | Distribution-based | Không full cache | Có. Update Gaussian `means`, `vars`, `priors` theo test batches bằng momentum. | Không | `base_weight=0.35`, `momentum=0.97`, `min_var=1e-4`, `confidence_threshold=0.0` | Fit Gaussian mean/variance/prior từng class từ train, rồi update online theo test batches. |
| 10 | `dota_balanced` | Distribution + balance | Không full cache | Có. Update Gaussian stats như DOTA, sau đó wrapper chỉnh output prior về `50/50`. | Không | `base_weight=0.55`, `momentum=0.99`, balance strength `0.8`, target prior `(0.5, 0.5)` | DOTA variant có correction để output batch bớt lệch class. |
| 11 | `dota_linear_ensemble_balanced` | Distribution ensemble | Không full cache | Có. Update DOTA Gaussian stats, blend với linear probe, rồi balance prior. | Không | DOTA `base_weight=0.55`, `momentum=0.99`; ensemble `linear_weight=0.4`, `inner_weight=0.6`; balance strength `0.7` | Blend linear probe với DOTA, sau đó chỉnh prior về gần `50/50`. |
| 12 | `freetta` | Source-free distribution | Không train cache | Có. Online EM: từ posterior batch tính `means`, `vars`, `priors`, rồi momentum update. | Không | `base_weight=0.4`, `momentum=0.95`, `prior_power=1.0`, `warmup_batches=1`, `min_var=1e-4` | Online EM Gaussian trên test stream, seed bằng linear probe. Không dùng train cache lúc adaptation. |
| 13 | `freetta_linear_ensemble` | Source-free ensemble | Không train cache | Có. FreeTTA update Gaussian stats; kết quả được blend với linear probe cố định. | Không | `linear_weight=0.3`, `freetta_weight=0.7`; FreeTTA default | Blend xác suất linear probe với FreeTTA. |
| 14 | `freetta_balanced` | Source-free + balance | Không train cache | Có. FreeTTA update Gaussian stats, rồi wrapper chỉnh batch prior về `50/50`. | Không | FreeTTA `base_weight=0.35`; balance strength `0.8`, target prior `(0.5, 0.5)` | FreeTTA rồi chỉnh output prior theo batch về gần `50/50`. |
| 15 | `freetta_linear_ensemble_balanced` | Source-free ensemble + balance | Không train cache | Có. FreeTTA update stats, blend linear, rồi balance prior. | Không | `linear_weight=0.25`, `freetta_weight=0.75`; balance strength `0.7` | Linear probe + FreeTTA, sau đó balance prior. Đây là lightweight candidate chính để tăng AUC. |
| 16 | `bca` | Bayesian/prototype | 2 prototypes + prior | Có. Update class prototypes và class priors online bằng soft prediction. | Không | `temperature=0.07`, `base_weight=0.5`, `prior_momentum=0.95`, `prototype_momentum=0.98` | Lưu 1 prototype REAL, 1 prototype FAKE và prior; update prototype/prior online. |
| 17 | `bca_balanced` | Bayesian/prototype + balance | 2 prototypes + prior | Có. Update BCA prototypes/priors, rồi wrapper chỉnh prior về `50/50`. | Không | BCA `base_weight=0.6`, `prior_momentum=0.98`, `prototype_momentum=0.99`; balance strength `0.7` | BCA chậm update hơn và có prior correction về `50/50`. |
| 18 | `bca_linear_ensemble_balanced` | Prototype ensemble + balance | 2 prototypes + prior | Có. Update BCA prototypes/priors, blend linear, rồi balance prior. | Không | BCA `base_weight=0.55`, `prior_momentum=0.98`, `prototype_momentum=0.99`; ensemble `linear_weight=0.35`, `inner_weight=0.65`; balance strength `0.7` | Blend linear probe với BCA rồi balance prior. |
| 19 | `dpe` | Prototype evolving | 2 visual + 2 text-like prototypes | Có. Update visual/text-like prototypes bằng weighted mean từ soft prediction batch. | Không | `temperature=0.07`, `base_weight=0.4`, `visual_momentum=0.95`, `text_momentum=0.995` | Dual Prototype Evolving trong feature space: prototypes update online bằng soft prediction. |
| 20 | `dpe_balanced` | Prototype evolving + balance | Prototype nhỏ | Có. Update DPE prototypes, rồi wrapper balance output prior. | Không | DPE `base_weight=0.55`, `visual_momentum=0.98`; balance strength `0.7` | DPE với update chậm hơn và prior correction. |
| 21 | `dpe_linear_ensemble_balanced` | Prototype ensemble + balance | Prototype nhỏ | Có. Update DPE prototypes, blend linear, rồi balance prior. | Không | DPE `base_weight=0.55`, `visual_momentum=0.98`; ensemble `linear_weight=0.35`, `inner_weight=0.65`; balance strength `0.7` | Blend linear probe với DPE rồi balance prior. |
| 22 | `dynaprompt` | Prompt-style approximation | Không full cache | Có. Update feature-space/prediction state theo entropy/confidence approximation. | Không | `batch_size=512` | Feature-space approximation của dynamic prompt/entropy adaptation. Không phải prompt CLIP paper-exact. |
| 23 | `prototype_linear_tta` | Prototype + linear-head update | K-means prototypes nhỏ | Có. Mỗi test batch tính anchor probability từ prototypes để tạo loss adaptation. | Có. Copy linear probe rồi update `fc.weight`/`fc.bias` bằng `loss.backward()` + `optimizer.step()`. | `prototypes_by_class={0:32,1:128}`, `lr=1e-5`, `steps_per_batch=1`, `anchor_weight=0.3`, `reg_weight=0.1`, `balance_weight=0.0` | K-means prototypes làm anchor probability, rồi update linear head online mỗi test batch. |
| 24 | `prototype_linear_tta_balanced` | Prototype + linear-head update + balance | K-means prototypes nhỏ | Có. Tính anchor/prototype loss, thêm target prior `50/50`, rồi wrapper balance output. | Có. Update copied `fc.weight`/`fc.bias`; model gốc không đổi. | `lr=5e-6`, `anchor_weight=0.2`, `reg_weight=0.2`, `balance_weight=0.1`, `target_prior=(0.5,0.5)`, wrapper balance strength `0.6` | PrototypeLinearTTA bản ổn định hơn, có target prior 50/50 và output balance. |

## Cache Groups

### Không cache train / storage rất nhẹ

```text
freetta
freetta_linear_ensemble
freetta_balanced
freetta_linear_ensemble_balanced
dota
dota_balanced
dota_linear_ensemble_balanced
bca
bca_balanced
bca_linear_ensemble_balanced
dpe
dpe_balanced
dpe_linear_ensemble_balanced
dynaprompt
prototype_linear_tta
prototype_linear_tta_balanced
```

### Cache ít hoặc bounded

```text
compact_cache_adapter              # hiện 50% train cache
online_confident_cache_adapter     # 10% train cache + dynamic cache <= 2048/class
online_cache_10_balanced           # 10% train cache + dynamic cache <= 1024/class
```

### Cache nhiều

```text
tip_adapter     # full train cache
boost_adapter   # full train cache trong repo implementation hiện tại
crg             # full train cache + Gaussian stats
dmn             # full train cache + dynamic test cache
```

## Recommended Lightweight AUC Candidates

Nếu mục tiêu là tăng AUC nhưng hạn chế storage, ưu tiên test các method này trước:

```text
freetta_linear_ensemble_balanced
freetta_balanced
dota_linear_ensemble_balanced
prototype_linear_tta_balanced
online_cache_10_balanced
```

Trong nhóm này, `online_cache_10_balanced` dùng cache 10%; các method còn lại không lưu full train cache.

---
- test thêm:
    - test ff++ (ko corruption)
    - test ff++ (có corruption)
    - test celebdfv1 (ko corruption)
    
- sau khi có source model:
    - chạy thử với method tta:
    - setting:
        - balanced
        - imbalanced