# Tugas-Tutorial-Computer-Vision

Name &emsp;: Iffa Hesti Adlik Putri <br>
NIM &emsp;&ensp;&nbsp;: 23/514098/PA/21977 <br>

Cara Run Program <br>
1. Siapkan 1 folder untuk menyimpan requirements.txt, images, training.py, dan testing.py: <br>
   ├── Face Detection/ &emsp;&emsp;&emsp;&emsp;&emsp;&ensp;# Folder untuk menyimpan semua file yang dibutuhkan <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── requirements.txt &emsp;&emsp;&emsp;# Requirements yang diperlukan untuk menjalankan program <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── images/  &emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;&nbsp;&nbsp; # Folder dataset gambar (per orang di dalam subfolder) <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── training.py &emsp;&emsp;&emsp;&emsp;&emsp;&ensp;# Script untuk training model <br>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── testing.py &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp; # Script untuk real-time face recognition <br>
   
2. Sebelum menjalankan program, pastikan sudah menginstall Python 3 dan library berikut di local environment: <br>
   **pip install -r requirements.txt** <br>
   
4. Jalankan script training.py untuk melatih model menggunakan dataset yang sudah disiapkan: <br>
   **python training.py** <br>
   - Model hasil training (eigenface_model.joblib) dan label (labels.npy) akan disimpan ke folder models/. <br>
   
5. Setelah training selesai, jalankan script testing.py untuk mendeteksi dan mengenali wajah secara real-time menggunakan webcam: <br>
   **python testing.py** <br>
   - Pastikan webcam komputer aktif. <br>
   - Tekan tombol q untuk keluar dari program. <br>
