# Hasil Transfer Learning Kucing vs Anjing

Eksekusi selesai pada 22 Oktober 2026 menggunakan TensorFlow 2.21.0 pada CPU.
Target accuracy 90% tercapai pada validation dan test yang terpisah.

| Data | Jumlah gambar unik | Accuracy |
| --- | ---: | ---: |
| Train bersih, mode inference | 22.463 | 98,78% |
| Validation | 1.248 | 98,16% |
| Test akhir | 1.248 | **97,92%** |

Gap clean train–validation adalah **0,62 poin persentase**, di bawah batas
5 poin persentase yang ditetapkan sebelum training. Kurva validation stabil
dan hasil test mendekati validation, sehingga tidak terlihat overfitting besar
pada split ini. Ini merupakan hasil satu seed dan satu pembagian data,
bukan bukti mutlak bebas overfitting pada semua distribusi gambar.

## Hasil test

- Prediksi benar: **1.222 dari 1.248**; prediksi salah: **26**.
- Macro F1: **0,9792**.
- ROC-AUC, kelas positif Anjing: **0,9973**.
- Log loss tanpa penalti regularisasi: **0,05818**.

| Label sebenarnya | Prediksi Kucing | Prediksi Anjing |
| --- | ---: | ---: |
| Kucing | 610 | 14 |
| Anjing | 12 | 612 |

## Cara target dicapai

MobileNetV2 berbobot ImageNet dipakai sebagai ekstraktor fitur yang sepenuhnya
dibekukan. Hanya classifier sigmoid dengan 1.281 parameter yang dilatih.
Kandidat terpilih memakai dropout 0,30 dan L2 0,0001. Training menggunakan
fitur gambar train asli serta satu versi augmentasi per gambar, Adam,
pengurangan learning rate saat validation loss stagnan, dan early stopping.
Kandidat terpilih berhenti setelah 13 epoch dan memakai checkpoint terbaik.

Dua kandidat regularisasi ditentukan di awal. Keduanya mencapai validation
98,16%; kandidat dipilih dengan validation log loss tanpa penalti L2
terendah di antara model yang memenuhi batas accuracy dan gap. Test baru
dievaluasi setelah pemilihan selesai, dengan threshold tetap 0,5.

Seluruh 24.959 gambar bersih digunakan; tidak ada pembatasan sampel.
Audit dataset awal mengeluarkan 9 file rusak/kosong, 6 file dalam kelompok
label bertentangan, dan 26 salinan identik. Split sebelumnya dipertahankan,
SHA-256 gambar diverifikasi ulang, dan tidak ada hash identik melintasi split.
Foto mirip dengan encoding berbeda atau foto hewan yang sama belum
dipisahkan berdasarkan identitas, sehingga hasil tetap memiliki batas generalisasi.

## Artefak dan pemeriksaan

- Notebook lengkap dengan output dan grafik: `cats_vs_dogs_transfer_learning.executed.ipynb`.
- Notebook sumber: `cats_vs_dogs_transfer_learning.ipynb`.
- Model: `outputs/transfer_20261022T031434_475254Z/best_model.keras`.
- Ringkasan mesin: `outputs/transfer_20261022T031434_475254Z/run_summary.json`.
- Metrik: `outputs/transfer_20261022T031434_475254Z/test_metrics.json`.
- Perbandingan kandidat: `outputs/transfer_20261022T031434_475254Z/validation_comparison.csv`.
- Prediksi per gambar: `outputs/transfer_20261022T031434_475254Z/test_predictions.csv`.

Seluruh 17 sel kode selesai tanpa error. Metrik accuracy diperiksa ulang dari
CSV prediksi. Model tersimpan berhasil dimuat ulang dan skornya cocok dengan
classifier pada fitur validation yang tersimpan. Kurva training dan confusion
matrix juga telah diperiksa secara visual.

Untuk inferensi, gunakan gambar **RGB float32 ukuran 160 × 160, rentang
0–255**. Normalisasi sudah berada di dalam model. Keluaran sigmoid adalah
skor Anjing; skor Kucing adalah `1 - skor_anjing`.
