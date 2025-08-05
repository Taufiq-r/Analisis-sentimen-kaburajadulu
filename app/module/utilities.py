# File: app/module/utilities.py
# Deskripsi: Berisi fungsi-fungsi pembantu seperti split data dan kalkulasi metrik.

import numpy as np 
import random # Modul untuk operasi acak
from collections import Counter # # Modul untuk menghitung frekuensi custom_metrics, meskipun sudah ada di naive_bayes



def custom_metrics(y_true, y_pred, labels):
    """
    Menghitung metrik evaluasi (Akurasi, Presisi, Recall, F1-Score, Confusion Matrix) dari nol.
    
    :param y_true: Array label asli (ground truth).
    :param y_pred: Array label hasil prediksi model.
    :param labels: Daftar unik dari semua kemungkinan label (misal: ['positif', 'netral', 'negatif']).
    :return: Dictionary berisi laporan metrik.
    """
    report = {}
    
    # Convert y_true dan y_pred ke numpy array untuk operasi yang lebih efisien jika belum
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Pastikan label di confusion matrix memiliki urutan yang konsisten
    # Jika labels input tidak ada atau kosong, gunakan urutan default ini.
    # Jika ada, pastikan semua label yang ada di y_true/y_pred tercakup.
    consistent_labels_order = sorted(list(set(labels) | set(np.unique(y_true)) | set(np.unique(y_pred))))
    
    # Hitung metrik untuk setiap kelas/label
    for label in consistent_labels_order:
        # True Positive: Prediksi benar untuk kelas saat ini
        tp = np.sum((y_true == label) & (y_pred == label))
        # False Positive: Prediksi salah (kelas lain diprediksi sebagai kelas ini)
        fp = np.sum((y_true != label) & (y_pred == label))
        # False Negative: Prediksi salah (kelas ini diprediksi sebagai kelas lain)
        fn = np.sum((y_true == label) & (y_pred != label))
        
        # Hitung Presisi: Dari semua yang diprediksi 'positif', berapa yang benar?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        # Hitung Recall: Dari semua yang seharusnya 'positif', berapa yang berhasil terdeteksi?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Hitung F1-Score: Rata-rata harmonik dari presisi dan recall
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        report[str(label)] = {"precision": precision, "recall": recall, "f1-score": f1}
        
    # Hitung Akurasi Keseluruhan: (Total prediksi benar) / (Total semua data)
    accuracy = np.sum(y_true == y_pred) / len(y_true) if len(y_true) > 0 else 0.0 # Hitung akurasi
    report['accuracy'] = accuracy # Tambah akurasi ke laporan
    
    # Pastikan label_order mencakup semua label yang ada di y_true dan y_pred
    all_unique_labels = sorted(list(set(y_true) | set(y_pred))) # Ambil semua label unik
    
    final_label_order = sorted(list(set(labels) | set(y_true) | set(y_pred))) # Urutkan label akhir
    
    n = len(final_label_order) # Jumlah label
    confusion = np.zeros((n, n), dtype=int) # Inisialisasi matriks konfusi
    
    # Buat mapping label ke indeks yang konsisten
    label_to_index = {label: i for i, label in enumerate(final_label_order)}
    
    # Isi confusion matrix
    for yt, yp in zip(y_true, y_pred):
        # Pastikan label ada di mapping, jika tidak, abaikan atau log warning
        i = label_to_index.get(str(yt).lower())# Ambil indeks y_true
        j = label_to_index.get(str(yp).lower()) # Ambil indeks y_pred
        if i is not None and j is not None: # Periksa apakah indeks valid
            confusion[i, j] += 1  # Tambah ke matriks konfusi
            
    report['confusion_matrix'] = confusion.tolist()# Tambah matriks konfusi ke laporan
    
    return report # Kembalikan laporan metrik

# Fungsi random_oversample telah dipindahkan ke app/module/naive_bayes.py
# untuk menghindari duplikasi dan mengkonsolidasikan utilitas oversampling
# di dekat tempat ia paling sering digunakan (klasifikasi).
# Oleh karena itu, fungsi ini dihapus dari utilities.py.

