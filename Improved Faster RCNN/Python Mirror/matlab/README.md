# Deteksi di MATLAB dengan model ONNX

## Apakah `model_best.onnx` kompatibel dengan MATLAB? — **Tidak secara langsung.**

Importer ONNX MATLAB (`importNetworkFromONNX`) **tidak** mendukung operator inti yang membuat model ini jadi detektor. Dari enumerasi operator graph hasil export:

| Operator | Jumlah | Status di MATLAB |
|---|---|---|
| `GridSample` | 117 | dari DCN; baru jadi *custom layer* sejak **R2024b**, belum bersih |
| `RoiAlign` | 4 | **tidak didukung** → placeholder kosong |
| `NonMaxSuppression` | 1 | **tidak didukung** → placeholder |
| `NonZero`, `TopK`, `If`, `Range` | banyak | **tidak didukung** + output N dinamis tak bisa jadi `dlnetwork` statis |

> Opset 16 **bukan** masalah (MATLAB mendukung opset 6–20). Masalahnya **operator**, jadi menurunkan opset tidak membantu. Mengimpor graph penuh = praktis menulis ulang detektor di MATLAB. Tidak praktis.

## Solusi: jalankan ONNX lewat ONNX Runtime DARI MATLAB ✅

Menjalankan graph apa adanya via ONNX Runtime (Python) — semua operator didukung, bobot & post-processing Anda utuh, N dinamis ditangani natural. MATLAB hanya memanggil dan menggambar hasil.

**File:**
- `onnx_detect.py` — fungsi `detect(image, onnx, score_thresh)`: letterbox 480×480 (sama dengan training) → ONNX Runtime → box dikembalikan ke koordinat **gambar asli** + nama kelas.
- `run_detection.m` — skrip MATLAB: panggil helper, gambar bounding box.

**Setup (sekali):**
1. Pastikan ada `outputs/model_best.onnx` (otomatis dari `python train.py`, atau `python export_onnx.py`).
2. Python (yang sama dengan training) punya: `onnxruntime numpy pillow torch`.
3. Edit `PYEXE`, `ONNXFILE`, `IMAGEFILE` di `run_detection.m`, lalu jalankan.

Uji helper tanpa MATLAB:
```bash
python onnx_detect.py --image ../../dataset/images_aug/frame_000081.png --onnx ../outputs/model_best.onnx --score-thresh 0.5
```

## Alternatif (jika butuh pipeline MATLAB-native penuh)
Export ulang **backbone+RPN saja** (berhenti sebelum RoiAlign/NMS) → hanya operator built-in (Conv/BN/Relu/Resize) → bisa `importNetworkFromONNX`, lalu tulis RoiAlign/decode/NMS di MATLAB (`selectStrongestBboxMulticlass`). Kerja jauh lebih banyak, dan DCN (GridSample) di backbone tetap perlu R2024b+ atau di-fold ke conv biasa. Hubungi bila ingin mode export ini ditambahkan.

> Catatan: `trainFasterRCNNObjectDetector` MATLAB **tidak** bisa mereproduksi model ini (tak mendukung deformable conv / backbone ResNet50-vd custom), dan ditandai *not recommended* sejak R2024b.
