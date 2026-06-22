# Improved Faster R-CNN untuk Deteksi Cacat Permukaan Baja

Re-implementasi **PyTorch + torchvision** dari arsitektur "Improved Faster R-CNN" pada jurnal:

> Wang et al., *"Automatic Detection and Classification of Steel Surface Defect Using Deep CNNs"*, **Metals**, 2021.

Jurnal asli mengimplementasikan model di atas **PaddlePaddle/PaddleDetection**. Repositori ini me-port enam peningkatan arsitektural utama dari jurnal tersebut ke atas `torchvision.models.detection.FasterRCNN`, dengan tetap mempertahankan blok-blok bangunan torchvision (`AnchorGenerator`, `MultiScaleRoIAlign`, `RoIHeads`, `torchvision.ops.DeformConv2d`) jika memungkinkan.

---

## 1. Spesifikasi Target

| Spesifikasi | Nilai |
|---|---|
| Input citra | RGB, ukuran tetap **480 x 480 x 3** |
| Jumlah kelas objek (foreground) | **7** |
| Jumlah kelas internal detektor | **7 + 1 = 8** (background ditambahkan otomatis) |
| Framework | PyTorch + torchvision (build CPU) |
| Python | 3.10 |

### Penanganan kelas background

Jurnal mendefinisikan **7 kelas cacat foreground**. Konvensi torchvision menempatkan **background pada indeks 0**, sehingga detektor dibangun secara internal dengan `num_classes + 1 = 8`. Karena itu:

- `build_model(num_classes=7, ...)` menerima **jumlah kelas foreground** (bukan termasuk background).
- Di dalam `build_model`, torchvision `FasterRCNN` dipanggil dengan `num_classes=num_classes + 1`.
- Label foreground yang valid pada target adalah `1..7`; label `0` dicadangkan untuk background dan tidak boleh muncul pada anotasi.
- Pada post-processing, kelas background (kolom indeks 0) dibuang sebelum NMS.

Citra di-resize ke 480x480 dengan menyetel `min_size = max_size = 480` pada torchvision, dan dinormalisasi memakai statistik ImageNet RGB (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`).

---

## 2. Pemetaan Arsitektur ke Jurnal

Keenam peningkatan jurnal dan lokasi implementasinya:

| # | Komponen (peningkatan jurnal) | File | Kelas / Fungsi publik | Bagian jurnal |
|---|---|---|---|---|
| 1 | **Backbone ResNet50-vd** (stem tiga konv 3x3; blok downsample memakai `AvgPool(2)+1x1 conv` pada shortcut) **+ DCNv2** (modulated deformable conv) pada konv 3x3 di stage 2, 3, 4 | `models/resnet50_vd.py`, `models/deform_conv.py` | `ResNet50vd`, `DCNv2` | Backbone (improved feature extraction) |
| 2 | **SPP** (Spatial Pyramid Pooling gaya YOLO, mempertahankan H, W) diterapkan pada feature terdalam, **di dalam FPN** | `models/spp.py` (dipakai oleh `models/fpn.py`) | `SPP` | Multi-scale context |
| 3 | **Enhanced FPN** dengan konvolusi lateral 1x1 diganti **CoordConv** | `models/fpn.py`, `models/coordconv.py` | `EnhancedFPN`, `CoordConv` | Enhanced FPN |
| 4 | **Anchor kustom**: aspect ratio dari `[0.5, 1, 2]` menjadi `[0.25, 0.5, 1, 2, 5]` | `models/anchors.py` | `build_anchor_generator` | Anchor design |
| 5 | **Matrix NMS** (dari SOLOv2, diadaptasi ke bounding box) menggantikan NMS standar pada post-processing | `models/matrix_nms.py` | `matrix_nms` | Post-processing / NMS |
| 6 | **Loss imbalance kelas**: Focal Loss (`gamma=2`, `alpha=0.75`) dan Weighted Cross-Entropy | `models/losses.py` | `FocalLoss`, `WeightedCrossEntropyLoss`, `fastrcnn_focal_loss` | Class imbalance |
| - | **Perakitan** semua komponen menjadi detektor torchvision | `models/backbone.py`, `models/faster_rcnn.py` | `ResNet50vdFPN`, `build_model` | Keseluruhan model |

### Detail tiap komponen (sesuai implementasi nyata)

**#1 ResNet50-vd + DCNv2 (`resnet50_vd.py`, `deform_conv.py`).**
Stem "deep" mengganti konv 7x7 tunggal dengan tiga konv 3x3 (channel 32->32->64; konv pertama stride 2), diikuti max-pool 3x3 stride 2 sehingga keluaran stem memiliki stride 4. Setiap blok downsampling memakai trik ResNet-D: shortcut = `AvgPool2d(2, stride=2)` lalu konv 1x1 stride 1 (bukan konv 1x1 stride 2). Konv 3x3 di dalam bottleneck pada stage `2,3,4` (default `dcn_stages=(2,3,4)`) diganti `DCNv2`. `DCNv2` membungkus `torchvision.ops.DeformConv2d` dan sebuah `conv_offset_mask` (di-init nol) yang memprediksi offset (2*kh*kw) dan mask modulasi (kh*kw, melalui sigmoid). Keluaran backbone: dict `'0'..'3'` (C2..C5) dengan channel `[256, 512, 1024, 2048]` pada stride `4, 8, 16, 32`.

**#2 SPP (`spp.py`).**
Blok SPP gaya YOLO yang **mempertahankan H, W**: konv 1x1 reduksi -> beberapa cabang `MaxPool2d` stride-1 berpadding (kernel `(5, 9, 13)`) -> concat -> konv 1x1 fusi. SPP dipasang **di dalam `EnhancedFPN`** pada lateral terdalam (proyeksi C5) sebelum jalur top-down dimulai, bukan di dalam `ResNet50vd`.

**#3 Enhanced FPN + CoordConv (`fpn.py`, `coordconv.py`).**
FPN top-down klasik (Lin et al., CVPR 2017) tetapi setiap konv lateral 1x1 diganti `CoordConv` (Liu et al., NeurIPS 2018) yang menambahkan dua channel koordinat ternormalisasi `(x, y)` (opsional channel radius) sebelum konv 1x1. Dengan `extra_block=True`, FPN menambah satu level kasar ekstra (P6) lewat max-pool, menghasilkan **5 feature map** berkunci `'0'..'4'`.

**#4 Anchor kustom (`anchors.py`).**
`build_anchor_generator` membangun `torchvision ... AnchorGenerator` dengan 5 ukuran (satu per level FPN) dan aspect ratio `(0.25, 0.5, 1.0, 2.0, 5.0)` yang di-broadcast ke semua level. Rasio ekstrem `0.25` dan `5` ditujukan untuk cacat memanjang/tipis (crazing, scratch).

**#5 Matrix NMS (`matrix_nms.py`).**
`matrix_nms(boxes, scores, labels, sigma, method, score_threshold, max_detections)` mengganti loop supresi greedy dengan satu lintasan ter-vektorisasi penuh: hitung IoU berpasangan, ambil hanya pengaruh box berskor lebih tinggi (upper-triangular), masking per-kelas, lalu faktor peluruhan Gaussian/linear; skor dikalikan faktor peluruhan dan disaring `score_threshold`. Mengembalikan `(kept_indices, decayed_scores)` urut menurun.

**#6 Loss imbalance (`losses.py`).**
`FocalLoss(alpha=0.75, gamma=2.0)` multi-kelas (softmax): `FL = alpha * (1 - pt)^gamma * CE`. `WeightedCrossEntropyLoss(class_weights)` adalah CE dengan bobot per-kelas (buffer, ikut `.to(device)` dan `state_dict`). `fastrcnn_focal_loss(...)` adalah pengganti drop-in untuk `fastrcnn_loss` torchvision: klasifikasi memakai focal loss, regresi box tetap smooth-L1 (Huber, `beta=1/9`) hanya atas sampel foreground.

---

## 3. API Publik Utama

Titik masuk model:

```python
from models.faster_rcnn import build_model

def build_model(
    num_classes=7,                                   # jumlah kelas FOREGROUND
    input_size=480,                                  # min_size == max_size == 480
    anchor_sizes=((16,), (32,), (64,), (128,), (256,)),  # WAJIB 5 entri (5 level FPN); default generik.
    anchor_ratios=(0.25, 0.5, 1.0, 2.0, 5.0),            # train.py menimpa dgn nilai tuned di config.py:
                                                         #   sizes (12,24,40,64,104), ratios (0.2,.5,1,2,5)
    score_thresh=0.05,
    matrix_nms_sigma=0.5,
    detections_per_img=100,
    pretrained_backbone=False,   # disediakan untuk simetri API; backbone dilatih dari awal
    use_focal_loss=False,        # True -> RoIHeads memakai Focal Loss (improvement #6)
) -> nn.Module                   # torchvision FasterRCNN; jalan di mode train & eval
```

Perilaku model yang dikembalikan:

- **Mode train:** `model(images, targets) -> dict` berisi tensor loss skalar (`loss_classifier`, `loss_box_reg`, `loss_objectness`, `loss_rpn_box_reg`); `.backward()` berfungsi.
- **Mode eval:** `model(images) -> list[dict]` dengan kunci `boxes` (`[N,4]` float32), `labels` (`[N]` int64, nilai `1..7`), `scores` (`[N]` float32). Post-processing memakai Matrix NMS.

Komponen yang di-bind ke `model.roi_heads` (lihat `faster_rcnn.py`):

```python
# Pengganti RoIHeads.postprocess_detections (Matrix NMS), di-bind via types.MethodType.
def matrix_nms_postprocess(self, class_logits, box_regression, proposals, image_shapes)
    # -> (all_boxes, all_scores, all_labels)
# sigma disimpan di model.roi_heads.matrix_nms_sigma

# Varian RoIHeads.forward dengan Focal Loss; di-bind HANYA saat use_focal_loss=True.
def _focal_roi_heads_forward(self, features, proposals, image_shapes, targets=None)
    # -> (result, losses)
```

Building block lain (semua bebas dari `config.py`, impor antar-sibling memakai relative import):

| File | API publik |
|---|---|
| `models/backbone.py` | `ResNet50vdFPN(dcn_stages=(2,3,4), out_channels=256, extra_block=True)` -> dict 5 feature map `'0'..'4'`; punya atribut `out_channels` |
| `models/resnet50_vd.py` | `ResNet50vd(dcn_stages=(2,3,4), norm_layer=nn.BatchNorm2d)`; atribut `out_channels_list = [256,512,1024,2048]` |
| `models/fpn.py` | `EnhancedFPN(in_channels_list, out_channels=256, extra_block=True)` |
| `models/deform_conv.py` | `DCNv2(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, deformable_groups=1)` |
| `models/coordconv.py` | `CoordConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, with_r=False, bias=True)` |
| `models/spp.py` | `SPP(in_channels, out_channels, pool_sizes=(5,9,13))` |
| `models/anchors.py` | `build_anchor_generator(sizes=..., aspect_ratios=...) -> AnchorGenerator` |
| `models/matrix_nms.py` | `matrix_nms(boxes, scores, labels, sigma=0.5, method='gaussian', score_threshold=0.05, max_detections=100)`; juga `box_iou(boxes)` |
| `models/losses.py` | `FocalLoss`, `WeightedCrossEntropyLoss`, `fastrcnn_focal_loss(...)` |

> Catatan desain: hanya `config.py`, `train.py`, dan `inference.py` yang membaca konfigurasi. Modul di dalam `models/` sepenuhnya ter-parameterisasi dan **tidak** mengimpor `config.py`, sehingga bisa dipakai ulang di luar proyek ini.

---

## 4. Instalasi

Butuh Python 3.10. PyTorch dan torchvision dipasang dari index wheel **CPU** agar pip tidak menarik wheel CUDA default.

```bash
# 1) PyTorch + torchvision (build CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) Dependensi sisanya (numpy, Pillow, tqdm, pycocotools opsional)
pip install -r requirements.txt
```

Dependensi (lihat `requirements.txt`): `torch`, `torchvision`, `numpy`, `Pillow`, `tqdm`, dan `pycocotools` (**opsional**, hanya untuk evaluasi mAP gaya COCO). Di Windows `pycocotools` butuh Microsoft C++ Build Tools dan sering gagal dipasang; sisa proyek berjalan tanpanya. Alternatif Windows: `pip install pycocotools-windows`.

---

## 5. Cara Menjalankan

### Smoke test (tersedia, sudah berfungsi)

`tests/smoke_test.py` berjalan end-to-end di CPU memakai data acak kecil dan seed tetap. Diuji: pembangunan model, mode train (dict loss + `backward()` menghasilkan gradien non-nol), dan mode eval (list dict deteksi dengan kunci/shape/dtype benar).

```bash
# dari dalam direktori python_alternatif
python tests/smoke_test.py
```

Output yang diharapkan diakhiri baris `SMOKE TEST PASSED`. Script memiliki shim `sys.path` sehingga bisa dijalankan dari root repo maupun dari dalam `python_alternatif`.

### Verifikasi sintaks

```bash
python -m py_compile models/*.py config.py tests/smoke_test.py
```

### Membangun model langsung (cek jumlah parameter)

```bash
python -m models.faster_rcnn
# Mencetak Total parameters: 42,442,752
```

### Dataset (CVAT + letterbox)

Dataset nyata ada di `../dataset/` (`images_aug/` + `annotations_aug.xml`, format **CVAT v1.1**): 1588 frame **956x720**, 10.929 box, **7 kelas** plastik dengan label `"NAMA (idx)"`:

| idx | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|-----|-----|------|-----|------|-----|-----|-------|
| nama | PET | HDPE | PVC | LDPE | PP | PS | Other |

Setiap frame di-**letterbox** ke 480x480 (`data/transforms.py` → `letterbox_resize`): skala tunggal `s = min(480/956, 480/720) = 0.502`, gambar jadi 480x362, lalu di-pad abu-abu 59px atas/bawah. Bounding box ditransformasi seragam `(x,y) → (x·s + pad_left, y·s + pad_top)` sehingga tetap presisi (tanpa distorsi aspek rasio).

Verifikasi visual 6 sampel (hasil tersimpan ke `samples_letterbox.png`):

```bash
python visualize_samples.py --num 6 --out samples_letterbox.png
```

`data/cvat_dataset.py` mem-parsing XML, menerapkan letterbox, dan men-split train/val (default 80/20) **per base-frame** — semua augmentasi (`_aug1`/`_aug2`) ikut frame aslinya agar tidak ada kebocoran train↔val.

### Training dan inferensi

```bash
# Training pada dataset CVAT (default): batch 16, progress bar, log metrik & grafik
python train.py                                    # config defaults (from-scratch: 200 epoch, batch 16, lr 0.005)
python train.py --batch-size 16 --eval-every 5     # lebih cepat: validasi tiap 5 epoch
python train.py --batch-size 8                     # turunkan jika OOM
python train.py --use-focal-loss true              # aktifkan Focal Loss (imp. #6)
python train.py --dataset synthetic --epochs 1     # uji loop tanpa dataset
python train.py --max-train-batches 4 --eval-max-batches 3   # uji cepat

# Inferensi pada satu frame (otomatis di-letterbox; box sejajar kanvas 480x480)
python inference.py --image ../dataset/images_aug/frame_000081.png \
    --checkpoint outputs/model_best.pth --score-thresh 0.3 --output annotated.png
```

`train.py` melatih dengan SGD + warm-up + multi-step LR drop (sesuai jurnal) + **gradient clipping** (`Config.GRAD_CLIP_NORM`, mencegah loss divergen → NaN/segfault), menampilkan **progress bar tqdm** tiap epoch, lalu menyimpan `model_best.pth` (mAP validasi tertinggi) & `model_latest.pth` ke `OUTPUT_DIR`.

**Logging metrik & grafik (`metrics.py`).** Setiap epoch dicatat untuk **train DAN validation**:

| Metrik | Definisi |
|---|---|
| **Loss** | loss deteksi torchvision (cls + box_reg + objectness + rpn_box_reg); val-loss dihitung dengan BatchNorm dibekukan |
| **Accuracy** | `TP/(TP+FP+FN)` pada IoU≥0.5 & score≥0.5 (matching prediksi↔GT, class-aware) |
| **RMSE** | RMSE koordinat box (px) pada pasangan true-positive — kualitas regresi lokasi |
| **mAP** | COCO mAP@[.50:.95] & @.50 (torchmetrics) — metrik utama jurnal |
| **Precision / Recall / F1** | dari TP/FP/FN yang sama (metrik Tabel 4 jurnal) |

Output ke `OUTPUT_DIR/`:
- `metrics_log.csv` + `metrics_log.json` — satu baris per epoch (semua metrik di atas)
- **`training_curves.png`** — 6 panel (Loss, Accuracy, RMSE, mAP, P/R/F1, LR), **diperbarui tiap epoch** (buka & refresh untuk pantau langsung)

> Metrik train dihitung pada sampel `TRAIN_EVAL_BATCHES` batch (default 20) agar cepat; validasi penuh. *Catatan:* "accuracy" & "RMSE" bukan metrik native deteksi — didefinisikan seperti tabel di atas (didokumentasikan di `metrics.py`).

### Export ONNX (otomatis di akhir training)

Saat training selesai, checkpoint **terbaik** (`model_best.pth`, mAP validasi tertinggi) otomatis di-export ke **`outputs/model_best.onnx`** dan diverifikasi (onnx.checker + dijalankan via onnxruntime, dibandingkan dengan output PyTorch). Nonaktifkan dengan `--no-export-onnx`. Manual:

```bash
python export_onnx.py                                  # outputs/model_best.pth -> outputs/model_best.onnx
python export_onnx.py --checkpoint outputs/model_best.pth --output model.onnx
python export_onnx.py --nms matrix                     # pakai Matrix NMS di graph (default: standard)
```

Dua kendala export model deteksi ini ditangani otomatis:
- **DCNv2** — operator native `torchvision::deform_conv2d` tidak punya implementasi ONNX yang bisa dijalankan onnxruntime. `export_onnx.py` mengaktifkan mode `onnx_export` pada tiap DCNv2 → DCN didekomposisi ke **`grid_sample` + operasi standar** (cocok dengan torchvision ~1e-5), sehingga ONNX **portabel tanpa custom op**.
- **NMS** — default memakai **NMS standar** torchvision (→ ONNX `NonMaxSuppression`, aman untuk deployment & robust di berbagai input). Matrix NMS punya kontrol-alur data-dependent yang akan "dibekukan" saat tracing; pakai `--nms matrix` hanya bila perlu fidelitas penuh.

Graph ONNX: input `images` `[3,480,480]` → output `boxes [N,4]`, `labels [N]`, `scores [N]` (N dinamis), opset 16. Terverifikasi: `onnx.checker` OK, onnxruntime jalan, selisih box vs PyTorch ~6e-5.

Hyperparameter (LR, batch size, epoch, anchor, sigma Matrix NMS, alpha/gamma focal, grad clip, bobot kelas, dll.) terpusat di `config.py` (`Config` / instance `CONFIG`). `Config` memvalidasi invarian: panjang `CLASS_NAMES` dan `CLASS_WEIGHTS` harus sama dengan `NUM_CLASSES`, dan jumlah `ANCHOR_SIZES` harus sama dengan jumlah level FPN.

**Hyperparameter from-scratch (RTX 4080 Super 16 GB).** Karena backbone dilatih dari nol (tanpa bobot ImageNet), default di `config.py` disetel:

| Param | Nilai | Alasan |
|---|---|---|
| `BATCH_SIZE` | 16 | terukur: batch 8 pakai ~6,7 GB, jadi 16 (~13 GB) muat & ~2× lebih cepat; turunkan ke 8 bila OOM |
| `LR` | 0.005 | konservatif untuk from-scratch (linear-scaling penuh = 0.01 @ batch 16, tapi 0.005 lebih stabil); turunkan ke 0.0025 bila tak stabil |
| `EPOCHS` | 200 | from-scratch butuh skedul panjang; `model_best.pth` disimpan otomatis |
| `WARMUP_ITERS` | 2000 | ~12 epoch warm-up, penting untuk init acak |
| `LR_DROP_EPOCHS` | (130, 175) | LR ×0.1 di tiap milestone (multi-step) |
| `GRAD_CLIP_NORM` | 10.0 | jaring pengaman terhadap divergensi awal |

**Anchor disesuaikan ke dataset.** `ANCHOR_SIZES = (12, 24, 40, 64, 104)` dan `ANCHOR_RATIOS = (0.2, 0.5, 1, 2, 5)` dipilih dari analisis statistik box (k-means + anchor-recall) di ruang letterbox 480: objek berukuran ~11–153px (median 47), sehingga level 256 lama mubazir. Hasil **anchor-recall@IoU0.5 = 99.9%** dan **@IoU0.6 = 98.8%** atas seluruh 10.929 box (vs 74% @0.6 untuk anchor lama) — anchor kini mengcover seluruh dataset dengan IoU ketat.

> **Catatan kecepatan:** di RTX 4080 Super dengan batch 16 (~80 iter/epoch) satu epoch hanya beberapa puluh detik, jadi 200 epoch sangat layak (~1–2 jam). Pakai `--eval-every 5` untuk mempercepat (validasi lebih jarang). Pantau `training_curves.png` / mAP validasi; hentikan (Ctrl+C) saat plateau — `model_best.pth` selalu tersimpan.

---

## 6. Struktur File

```
python_alternatif/
├── config.py                 # [ada] Hyperparameter terpusat (Config / CONFIG). Hanya train/inference yang membacanya.
├── requirements.txt          # [ada] Dependensi + instruksi index CPU.
├── README.md                 # [ada] Dokumen ini.
├── models/
│   ├── __init__.py           # [ada] Re-export simbol publik (impor guarded try/except).
│   ├── deform_conv.py        # [ada] DCNv2  (improvement #1)
│   ├── coordconv.py          # [ada] CoordConv (improvement #3)
│   ├── spp.py                # [ada] SPP (improvement #2)
│   ├── matrix_nms.py         # [ada] matrix_nms, box_iou (improvement #5)
│   ├── anchors.py            # [ada] build_anchor_generator (improvement #4)
│   ├── losses.py             # [ada] FocalLoss, WeightedCrossEntropyLoss, fastrcnn_focal_loss (improvement #6)
│   ├── resnet50_vd.py        # [ada] ResNet50vd (+ DCNv2) (improvement #1)
│   ├── fpn.py                # [ada] EnhancedFPN (improvement #2 + #3)
│   ├── backbone.py           # [ada] ResNet50vdFPN (backbone gabungan, .out_channels, dict feature)
│   └── faster_rcnn.py        # [ada] build_model(...) -> torchvision FasterRCNN
├── data/
│   ├── __init__.py           # [ada] Penanda paket.
│   ├── transforms.py         # [ada] letterbox_resize / undo_letterbox_boxes (956x720 → 480x480)
│   ├── cvat_dataset.py       # [ada] CvatDefectDataset + parse_cvat_xml + split_records + collate_fn
│   └── dataset.py            # [ada] SyntheticDefectDataset + stub COCO + collate_fn (uji loop)
├── metrics.py                # [ada] Metrik (loss/acc/RMSE/mAP/P/R/F1) + TrainingLogger + plot
├── visualize_samples.py      # [ada] Render 6 sampel letterbox + box untuk verifikasi
├── export_onnx.py            # [ada] Export checkpoint terbaik -> ONNX (DCN->grid_sample) + verifikasi
├── train.py                  # [ada] Loop pelatihan + validasi + progress + logging + grafik + auto-export ONNX.
├── inference.py              # [ada] Inferensi (letterbox otomatis) + anotasi.
└── tests/
    └── smoke_test.py         # [ada] Smoke test end-to-end CPU.
```

Semua file sudah ada, terkompilasi, dan terverifikasi berjalan (smoke test, training CVAT + eval mAP, inferensi).

---

## 7. Catatan Penyederhanaan vs. Jurnal

Re-implementasi ini setia pada arsitektur jurnal, tetapi ada beberapa perbedaan praktis yang perlu disadari:

1. **Framework: PaddlePaddle -> PyTorch/torchvision.** Jurnal memakai PaddleDetection. Di sini DCNv2 dibangun di atas `torchvision.ops.DeformConv2d`, dan keseluruhan detektor di atas `torchvision ... FasterRCNN`. Detail kuantitatif (resep init bobot, skema sampling RPN/RoI, konstanta loss) mengikuti **konvensi torchvision**, yang bisa berbeda halus dari Paddle.

2. **Bobot pretrained.** `pretrained_backbone` disediakan untuk simetri API tetapi saat ini **tidak memuat bobot ImageNet**; backbone ResNet50-vd dilatih dari awal. Tidak ada checkpoint ResNet50-vd torchvision resmi yang tersedia, jadi pretraining harus disediakan terpisah jika diinginkan.

3. **Penempatan SPP.** SPP diterapkan pada lateral terdalam (proyeksi C5) **di dalam `EnhancedFPN`**, mempertahankan H x W (gaya YOLOv3-SPP). Jurnal menyebut SPP "before/within the FPN"; penempatan pasti dapat berbeda dari konfigurasi Paddle asli, namun semantik (memperkaya konteks multi-skala pada feature terdalam tanpa mengubah resolusi) dipertahankan.

4. **Matrix NMS.** Diadaptasi dari SOLOv2 (yang aslinya bekerja atas mask) ke **bounding box xyxy** dengan IoU berpasangan. Faktor peluruhan Gaussian (`sigma=0.5`) menjadi default; skor yang dikembalikan adalah skor **setelah peluruhan** (bukan sekadar hasil seleksi), sehingga distribusi skor akhir berbeda dari NMS greedy standar.

5. **Pemakaian level FPN.** RPN memakai kelima level (`'0'..'4'`) untuk proposal, sedangkan RoI head melakukan pooling pada empat level terhalus (`'0'..'3'`) lewat `MultiScaleRoIAlign` — mengikuti konvensi FPN Faster R-CNN torchvision, bukan tentu konfigurasi eksak jurnal.

6. **Focal loss bersifat opsional dan default mati.** Jalur default (`use_focal_loss=False`) memakai cross-entropy torchvision dan terbukti berfungsi penuh (train + eval). Jalur focal (`use_focal_loss=True`) mengganti `RoIHeads.forward` agar memakai `fastrcnn_focal_loss`; jalur eval-nya tetap melewati post-process Matrix NMS. `WeightedCrossEntropyLoss` tersedia sebagai utilitas namun belum di-wire ke RoIHeads (disediakan untuk dipasang manual bila perlu).

7. **Dataset & letterbox.** Spesifikasi target RGB 480x480x3, 7 kelas. Dataset nyata berformat **CVAT** (`../dataset/`, frame 956x720) dimuat oleh `data/cvat_dataset.py` dengan **letterbox** (jaga aspek rasio + pad abu-abu) — bukan resize-stretch — dan box ditransformasi seragam. Split train/val dikelompokkan per base-frame agar augmentasi tidak bocor. `data/dataset.py` tetap menyediakan `SyntheticDefectDataset` untuk uji loop tanpa data. Nama kelas di `config.py` = `[PET, HDPE, PVC, LDPE, PP, PS, Other]` (indeks = idx-1 dari label CVAT). Inferensi pada gambar non-480 juga otomatis di-letterbox agar konsisten dengan training.

8. **Evaluasi mAP.** Validasi memakai `torchmetrics.detection.MeanAveragePrecision` (xyxy, IoU bbox) — metrik yang dilaporkan jurnal. Bila `torchmetrics` tidak terpasang, evaluasi dilewati dengan peringatan.

9. **Stabilitas training.** Ditambahkan **gradient clipping** (`Config.GRAD_CLIP_NORM=10.0`) dan guard skip-step saat loss non-finite. Tanpa ini, divergensi (umum pada backbone-from-scratch + data acak) mengirim box inf ke NMS native torchvision dan menyebabkan hard-crash (segfault). Bukan bagian eksplisit jurnal, tetapi praktik standar pelatihan deteksi.

10. **Build CPU.** Seluruh kode diverifikasi pada build CPU PyTorch/torchvision. `config.py` mendeteksi CUDA secara otomatis (fallback aman ke `cpu`), namun benchmark performa/kecepatan jurnal (yang dilakukan di GPU) tidak direplikasi di sini.
