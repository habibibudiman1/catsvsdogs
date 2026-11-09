# Klasifikasi Kucing dan Anjing

Hasil eksekusi transfer learning: **test accuracy 97,92%**, validation
98,16%, dan clean train 98,78%. Gap train–validation **0,62 poin persentase**.
Lihat [hasil dan pemeriksaan overfitting](RESULTS.md) serta
`cats_vs_dogs_transfer_learning.executed.ipynb` untuk output sebenarnya.

## Versi transfer learning: target 90%

Notebook pengembangan utama: `cats_vs_dogs_transfer_learning.ipynb`. Versi ini memakai MobileNetV2 berbobot ImageNet dengan backbone beku dan classifier kecil (1.281 parameter) yang dilatih memakai dropout, L2, early stopping, serta pengurangan learning rate. Pendekatan mengikuti [panduan transfer learning Keras](https://keras.io/guides/transfer_learning/). Notebook CNN awal tetap tersedia di `cats_vs_dogs_indonesia.ipynb`.

Target penerimaan ditetapkan sebelum training: accuracy validation minimal 90%, gap clean train–validation maksimal 5 poin persentase, dan konfirmasi accuracy test minimal 90% setelah pemilihan model. Ini adalah target, bukan jaminan atau bukti mutlak bebas overfitting. Dua kandidat regularisasi dipilih menggunakan validation log loss tanpa penalti L2; test tidak dipakai untuk memilih kandidat, epoch, atau threshold.

Versi ini memakai manifest hasil audit `outputs/20261022T024529_916223Z/split_manifest.csv`: 22.463 train, 1.248 validation, dan 1.248 test. File gambar diverifikasi ulang dengan SHA-256. Ekstraksi fitur dibekukan dan disimpan di `data/feature_cache/`; train memakai fitur gambar asli serta satu versi augmentasi per gambar. Pemakaian cache membuat training classifier jauh lebih ringan pada CPU. Jika memindahkan proyek, salin manifest audit dan atur `source_run` serta `data_dir` pada notebook.

```powershell
.\.venv\Scripts\python.exe run_notebook.py cats_vs_dogs_transfer_learning.ipynb
```

Hasil lengkap ada di `cats_vs_dogs_transfer_learning.executed.ipynb` dan folder `outputs/transfer_<waktu_run>/`. `run_summary.json` mencatat train accuracy, validation accuracy, gap, test accuracy, dan apakah target benar-benar tercapai. Model terbaik adalah `best_model.keras`, dengan input RGB float32 **160 × 160 pada rentang 0–255**, keluaran sigmoid skor Anjing, dan threshold 0,5. Normalisasi MobileNetV2 sudah di dalam model; jangan membagi piksel lagi. Bobot ImageNet perlu diunduh satu kali bila belum ada di cache Keras. Notebook ini tidak mengubah train/validation/test untuk mengejar angka target.

Notebook pembelajaran berbahasa Indonesia yang dikembangkan dari [Cats vs Dogs Classification oleh Fares Elmenshawii](https://www.kaggle.com/code/fareselmenshawii/cats-vs-dogs-classification). Dataset mengikuti sumber aslinya: [Microsoft Cats vs Dogs di Kaggle](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset).

## Isi proyek

- `cats_vs_dogs_indonesia.ipynb`: notebook utama, dijalankan berurutan dari atas.
- `reference/cats-vs-dogs-classification.original.ipynb`: unduhan sumber asli tanpa perubahan.
- `reference/kaggle-source-metadata.json`: metadata sumber, termasuk pemilik dan versi.
- `requirements.txt`: dependensi untuk menjalankan notebook secara lokal.

Sumber asli versi 39 diambil pada 22 Oktober 2026 melalui API publik Kaggle, setara pengambilan sumber menggunakan `kaggle kernels pull`. CLI Kaggle belum tersedia pada lingkungan awal. Perhatikan ketentuan penggunaan dan lisensi pada halaman notebook/dataset sebelum mendistribusikan ulang.

## Cara termudah: Kaggle

1. Buat notebook baru di Kaggle lalu impor `cats_vs_dogs_indonesia.ipynb`.
2. Pilih **Add Input**, tambahkan dataset `shaunthesheep/microsoft-catsvsdogs-dataset`.
3. Aktifkan akselerator GPU jika tersedia. Notebook tidak memerlukan bobot pretrained atau akses internet saat training.
4. Jalankan semua sel dari atas ke bawah. Hasil disimpan di `/kaggle/working/cats_vs_dogs/<waktu_run>/`.
5. Untuk percobaan singkat, ubah `max_per_class = 200` dan `epochs = 1`. Percobaan ini hanya memeriksa alur; gunakan `max_per_class = None` untuk eksperimen penuh.

## Google Colab

Unggah notebook utama, pilih runtime GPU, lalu unggah/ekstrak dataset hingga tersedia folder `PetImages/Cat` dan `PetImages/Dog`. Atur `data_dir = Path("/content/PetImages")` pada sel konfigurasi. Colab biasanya menyediakan dependensi analisis; bila import gagal, instal paket pada `requirements.txt`, lalu restart runtime. Jangan mengunduh dataset di setiap sel atau memasukkan token Kaggle ke notebook.

## Menjalankan lokal di Windows

Pasang Python 3.11 64-bit untuk lingkungan lokal. Python portabel di `.tools/`, jika ada, digunakan untuk pemeriksaan pengembangan dan tidak menjadi prasyarat notebook.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Untuk menjalankan semua sel secara otomatis dari terminal:

```powershell
.\.venv\Scripts\python.exe run_notebook.py
```

Eksekutor membuat `cats_vs_dogs_indonesia.executed.ipynb` berisi output dan grafik, serta menyimpan kemajuan berkala di `.tools/jupyter/execution_status.json`. Log training ada di `.tools/jupyter/execution.log`. Notebook sumber tetap tersedia untuk diedit. Setiap pemanggilan memulai run baru menggunakan parameter pada notebook sumber; proses ini tidak melanjutkan epoch dari run sebelumnya.

Unduh dataset dari halaman Kaggle lalu ekstrak sehingga strukturnya:

```text
data/
  PetImages/
    Cat/
      0.jpg
      ...
    Dog/
      0.jpg
      ...
```

Alternatif unduhan dengan CLI Kaggle setelah autentikasi akun sesuai dokumentasi Kaggle:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
.\.venv\Scripts\kaggle.exe datasets download -d shaunthesheep/microsoft-catsvsdogs-dataset -p data --unzip
.\.venv\Scripts\kaggle.exe kernels pull fareselmenshawii/cats-vs-dogs-classification -p reference --metadata
```

Jika hasil ekstraksi memiliki folder tambahan, arahkan `data_dir` langsung ke `PetImages`. [TensorFlow setelah versi 2.10 tidak mendukung GPU pada Windows native](https://www.tensorflow.org/install/pip); gunakan Kaggle, Colab, atau WSL2 untuk training GPU. Notebook juga mendukung CPU, dengan waktu training lebih lama.

## Alur dan perbaikan metode

1. Periksa gambar dengan decoder TensorFlow yang juga dipakai saat training; catat file rusak/kosong tanpa mengubah dataset sumber.
2. Keluarkan seluruh kelompok file identik yang memiliki label bertentangan, lalu hapus salinan identik berlabel sama berdasarkan SHA-256 sebelum pembagian data. Semua pengecualian dicatat dalam CSV. Ini tidak mendeteksi gambar yang mirip atau hasil encoding ulang.
3. Bagi data secara stratified dengan seed tetap: 90% train, 5% validation, 5% test, dengan pembulatan jumlah sampel. Simpan manifest dan periksa tidak ada hash yang melintasi split.
4. Pertahankan CNN dasar sumber: enam convolution, dua max pooling, global average pooling, dense 1024, dan keluaran dua kelas. Bandingkan tanpa/dengan augmentasi menggunakan arsitektur, inisialisasi bobot, split, optimizer, dan batas epoch yang sama.
5. Pilih checkpoint dan eksperimen hanya berdasarkan validation loss. Test digunakan untuk evaluasi akhir model terpilih, bukan untuk tuning.
6. Tampilkan accuracy, precision/recall/F1, ROC-AUC, confusion matrix, contoh prediksi, dan Grad-CAM. Simpan model serta metadata preprocessing.

Pada sumber asli, test generator menunjuk folder validation, parameter `split_size` tidak dipakai, pemisahan tidak memiliki seed, visualisasi prediksi memakai indeks yang bergeser, dan CAM memakai bobot dense yang tidak langsung terhubung ke kelas. Versi ini memperbaikinya. Eksperimen augmentasi sumber juga mengubah arsitektur, sehingga versi ini menyamakan arsitektur agar pengaruh augmentasi lebih mudah dinilai. Augmentasi modern dan decoder dapat memberi hasil berbeda dari sumber; tidak ada jaminan peningkatan akurasi.

## Artefak setelah training

Setiap run mempunyai folder tersendiri agar checkpoint lama tidak tercampur. Isinya mencakup konfigurasi, versi paket, audit file rusak/duplikat, manifest split, riwayat training, model `.keras`, perbandingan validation, prediksi test, classification report, metrik test, dan ringkasan run. Gambar grafik tampil di notebook. Untuk memuat ulang model, gunakan `tf.keras.models.load_model(...)` dengan path model pada `run_summary.json`; kirim gambar RGB float32 berukuran 150 × 150 pada rentang 0–255. Normalisasi 1/255 sudah berada di model.

Notebook sumber adalah template; output training sebenarnya tersimpan pada `cats_vs_dogs_indonesia.executed.ipynb` dan folder run setelah eksekusi. Selama proses masih berjalan, notebook hasil diperbarui berkala dan metrik akhir belum tersedia. Nilai softmax bukan jaminan probabilitas terkalibrasi; input selain kucing/anjing tetap akan dipaksa ke salah satu kelas.

Validasi pengembangan menggunakan Python 3.11.9 dan TensorFlow 2.21.0 pada CPU: format notebook dan sintaks valid; seluruh 21 sel kode berhasil dijalankan memakai 40 gambar sintetis ukuran 48 × 48, satu epoch per eksperimen. Uji mencakup deteksi dua file rusak/kosong dan satu duplikat, kesamaan bobot awal, training kedua model, evaluasi, Grad-CAM, prediksi gambar, serta kesamaan prediksi setelah model dimuat ulang. Forward pass ukuran asli 150 × 150 juga diperiksa. Uji ini memverifikasi alur perangkat lunak, bukan akurasi pada dataset kucing/anjing.

## Referensi implementasi

- [Decoder gambar TensorFlow](https://www.tensorflow.org/api_docs/python/tf/io/decode_image)
- [Augmentasi RandomRotation Keras](https://keras.io/api/layers/preprocessing_layers/image_augmentation/random_rotation/)
- [Grad-CAM Keras](https://keras.io/examples/vision/grad_cam/)
