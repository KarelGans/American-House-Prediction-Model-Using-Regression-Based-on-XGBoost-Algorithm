# Laporan Proyek Machine Learning Prediksi Harga Rumah Bedasarkan Data Harga Rumah Amerika - Carolus Christadi Cahyono

## Domain Proyek

Proyek ini berada dalam domain Real Estate dan Data Science, dengan fokus pada prediksi harga rumah di berbagai kota besar di Amerika Serikat. Dengan meningkatnya kebutuhan akan properti dan fluktuasi harga rumah, penting bagi pembeli, penjual, maupun investor untuk memahami faktor-faktor yang memengaruhi harga rumah.

### Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan
Seperti yang dibahas pada tulisan "Determinants of Real House Price Dynamics" [Determinants of Real House Price Dynamics](https://www.nber.org/papers/w9262), 
Prediksi harga rumah yang akurat membantu:
- Investor membuat keputusan berdasarkan data.
- Calon pembeli harga rumah bedasarkan lokasi.
- Pemerintah dan pengembang merancang kebijakan dan proyek pembangunan.
  
Terutama pada peningkatan dasn perubahan harga yang secara dinamika berubah dari beberapa faktor. Model yang dibuat ingin menjadi baseline untuk menentukan harga rumah dari beberapa faktor yang terlihat.
## Business Understanding

Pada bagian ini, kamu perlu menjelaskan proses klarifikasi masalah.

Bagian laporan ini mencakup:

### Problem Statements

- Apa saja fitur-fitur demografis dan lingkungan yang paling memengaruhi harga rumah?
- Bagaimana memprediksi harga rumah berdasarkan fitur yang tersedia?

### Goals

- Mengidentifikasi fitur-fitur penting yang memiliki korelasi tinggi terhadap harga rumah.

- Membangun model machine learning untuk memprediksi harga rumah dengan akurasi tinggi.
Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements

- Mencoba mencari algoritma yang dapat memenuhi problem statement seperti linear regression, XGBoost, dan Neural Network
- Melakukan hyperparameter tuning secara manual atau dengan algoritma seperti Optuna.
- Mengukur performa model dengan metrik Mean Squared Error (MSE) dan Mean Absolute Percentage Error (MAPE) untuk gambaran hasil yang lebih jelas.

## Data Understanding
Dataset diambil dari Kaggle: https://www.kaggle.com/datasets/jeremylarcher/american-house-prices-and-demographics-of-top-cities/data

Dataset ini menggabungkan data harga rumah dan data demografi dari kota-kota besar di Amerika Serikat.
Dataset memuat variabel-variabel :

- Price: Harga properti yang tercantum dalam listing.

- Beds: Jumlah kamar tidur yang disebutkan dalam listing.

- Baths: Jumlah kamar mandi yang disebutkan dalam listing.

- Living Space: Total luas ruang hidup (dalam kaki persegi / square feet) yang tercantum dalam listing.

- Address: Alamat jalan properti.

- City: Kota lokasi properti.

- State: Negara bagian lokasi properti.

- Zip Code Population: Perkiraan jumlah penduduk dalam kode pos tersebut (Sumber: Simplemaps.com).

- Zip Code Density: Kepadatan penduduk per mil persegi dalam kode pos (Sumber: Simplemaps.com).

- County: Kabupaten tempat properti berada.

- Median Household Income: Pendapatan rumah tangga median (Sumber: U.S. Census Bureau).

- Latitude: Garis lintang kode pos (Sumber: Simplemaps.com).

- Longitude: Garis bujur kode pos (Sumber: Simplemaps.com).

## Visualisai Data

Dalam tahap EDA, dibutuhkan visualisasi untuk mengetahui korelasi antara variabel, maka digunakan function pair plot dari library seaborn untuk melihar korelasi bivariate secara visual.


## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
