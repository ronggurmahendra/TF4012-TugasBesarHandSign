# TF4012-TugasBesarHandSign
Document : https://docs.google.com/document/d/1I39GIGiEm-a0SNGL3kj-bkTzG65nw37QxyB8-fPnffk/edit?usp=sharing


## Cara eksekusi Program 
1. Clone repository ini menggunakan perintah pada terminal "git clone https://github.com/ronggurmahendra/TF4012-TugasBesarHandSign.git"
2. Pergi ke dalam directory menggunakan perintah "cd .\TF4012-TugasBesarHandSign\"
3. Pastikan library cv2, tensorflow,tensorflow_datasets, keras, matplotlib, os, numpy, scipy, skimage, random sudah terinstall 
4. Untuk mengeksekusi program lakukan perintah "py VideoIndentifier.py" *pastikan komputer sudah terhubung ke kamera dan terminal memiliki akses ke kamera
5. secara default model yang akan digunakan adalah model yang tersave pada directory saved_model/Model_3

## Cara membuat model yang baru
1. Clone repository ini menggunakan perintah pada terminal "git clone https://github.com/ronggurmahendra/TF4012-TugasBesarHandSign.git"
2. Pergi ke dalam directory menggunakan perintah "cd .\TF4012-TugasBesarHandSign\"
3. Copy folder Data pada tautan https://drive.google.com/drive/u/2/folders/1a2KeAgAA_yl6i7O70CVS5kvLCyGhmcI3 pada root directory
4. Lewati langkah 5 sampai 7 jika ingin menggunakan training data yang disediakan 
5. Taruh video data training pada directory ./Data/Video/
6. Lakukan perintah "py .\videoToImages.py" dan input diretory video (i.e. ./Data/Video/Data.mp4) pada terminal
7. Pada direcotry Images akan terdapat banyak gambar klasifikasi gambar tersebut pada directory ./Data/ClassifiedData/train sesuai dengan kelasnya. (direcotry akan terlihat kurang lebih seperti ini)
TF4012-TugasBesarHandSign/

.Data/

..ClassifiedData/

...Train/

....A/

.....*.jpg/

....B/

.....*.jpg/

....C/

.....*.jpg/

....D/

.....*.jpg/

 dan seterusnya
8. Pastikan library cv2, tensorflow,tensorflow_datasets, keras, matplotlib, os, numpy, scipy, skimage, random sudah terinstall 
9. buka CreateModelHandSignToAlphabet.ipynb dan execute semua cell
10. model akan ada pada direcory saved_model (model yang sebelumnya di buat akan di overwrite)

## Author
Ronggur Mahendra Widya Putra