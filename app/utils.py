import math
import re 
import random 
import json
import numpy as np
from collections import Counter
from flask import current_app

# Konstanta untuk urutan label yang konsisten di seluruh aplikasi
SENTIMENT_LABELS = ['positif', 'negatif', 'netral']

def get_sampled_tfidf(full_tfidf_data, num_docs=3, num_tokens=5):
    """
    Mengambil sampel TF-IDF dari data yang sudah diproses untuk ditampilkan.
    Fungsi ini mengacak dokumen, lalu mengambil token dengan skor tertinggi dari setiap dokumen sampel.
    
    Parameters:
    -----------
    full_tfidf_data : list of dict
        Data TF-IDF lengkap, di mana setiap dict mewakili satu dokumen.
    num_docs : int, opsional
        Jumlah dokumen yang akan dijadikan sampel.
    num_tokens : int, opsional
        Jumlah token teratas (berdasarkan skor) yang akan diambil dari setiap dokumen sampel.
        
    Returns:
    --------
    list of dict
        Sampel data TF-IDF yang siap ditampilkan.
    """
    if not isinstance(full_tfidf_data, list) or not full_tfidf_data:
        return []
    
    logger = current_app.logger if current_app else None
    
    # Acak salinan data untuk mendapatkan sampel yang representatif tanpa mengubah data asli
    sampled_indices = random.sample(range(len(full_tfidf_data)), min(num_docs, len(full_tfidf_data)))
    
    sampled_data = []
    for i in sampled_indices:
        doc_data = full_tfidf_data[i]
        
        if not isinstance(doc_data, dict):
            if logger:
                logger.warning(f"Format data TF-IDF tidak dikenal: {type(doc_data)}. Data: {doc_data}")
            sampled_data.append({"error": "Format data tidak valid"})
            continue

        # Ambil hanya item yang merupakan token (bukan metadata seperti 'info' atau 'error')
        valid_tokens = {k: v for k, v in doc_data.items() if k not in ['info', 'error']}
        
        # Urutkan token berdasarkan skor TF-IDF secara menurun
        sorted_tokens = sorted(valid_tokens.items(), key=lambda item: item[1], reverse=True)
        
        # Ambil N token teratas
        top_tokens = sorted_tokens[:num_tokens]
        
        # Buat dictionary hasil dengan skor yang dibulatkan
        result_dict = {token: round(score, 4) for token, score in top_tokens}
        
        # Tambahkan kembali metadata jika ada
        if 'info' in doc_data: result_dict['info'] = doc_data['info']
        if 'error' in doc_data: result_dict['error'] = doc_data['error']
        
        # Jika tidak ada token sama sekali setelah diproses
        if not top_tokens and 'info' not in result_dict and 'error' not in result_dict:
            result_dict['info'] = "Tidak ada token TF-IDF signifikan dalam sampel ini."
            
        sampled_data.append(result_dict)
            
    return sampled_data


def generate_classification_report(y_true, y_pred, zero_division=0.0):
    """
    Menghasilkan laporan klasifikasi (presisi, recall, F1, support, akurasi, confusion matrix)
    secara manual dengan urutan label yang konsisten.
    
    Parameters:
    -----------
    y_true : list or np.array
        Label sebenarnya (ground truth).
    y_pred : list or np.array
        Label yang diprediksi oleh model.
    zero_division : float, opsional
        Nilai yang akan dikembalikan jika terjadi pembagian dengan nol (misal, saat support=0).
        
    Returns:
    --------
    dict
        Sebuah dictionary yang berisi semua metrik evaluasi.
    """
    logger = current_app.logger if current_app else None
    
    # Gunakan urutan label yang sudah ditentukan secara global untuk konsistensi
    labels = SENTIMENT_LABELS
    label_to_idx = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)
    
    # Normalisasi input untuk memastikan konsistensi (lower case, strip whitespace)
    y_true_norm = [str(l).strip().lower() for l in y_true]
    y_pred_norm = [str(l).strip().lower() for l in y_pred]

    # Inisialisasi Confusion Matrix dengan ukuran yang pasti (3x3)
    cm = np.zeros((num_labels, num_labels), dtype=int)
    
    # Isi confusion matrix
    for true_label, pred_label in zip(y_true_norm, y_pred_norm):
        true_idx = label_to_idx.get(true_label)
        pred_idx = label_to_idx.get(pred_label)
        
        if true_idx is not None and pred_idx is not None:
            cm[true_idx, pred_idx] += 1
        # elif logger: # Opsi untuk debugging label yang tidak dikenal
        #     logger.warning(f"Label tidak dikenal ditemukan: true='{true_label}', pred='{pred_label}'")

    # Hitung metrik per kelas
    report_metrics = {}
    true_positives = np.diag(cm)
    support = np.sum(cm, axis=1) # Jumlah sampel sebenarnya per kelas (TP + FN)
    predicted_positives = np.sum(cm, axis=0) # Jumlah prediksi per kelas (TP + FP)
    
    # Kalkulasi presisi, recall, dan f1-score
    precision = np.divide(true_positives, predicted_positives, out=np.full(num_labels, zero_division), where=predicted_positives!=0)
    recall = np.divide(true_positives, support, out=np.full(num_labels, zero_division), where=support!=0)
    
    f1_denominator = precision + recall
    f1_score = np.divide(2 * precision * recall, f1_denominator, out=np.full(num_labels, zero_division), where=f1_denominator!=0)
    
    # Susun laporan per kelas
    per_class_metrics = {}
    for i, label in enumerate(labels):
        per_class_metrics[label] = {
            'precision': round(precision[i], 4),
            'recall': round(recall[i], 4),
            'f1-score': round(f1_score[i], 4),
            'support': int(support[i])
        }
    
    # Hitung metrik agregat (accuracy, macro avg, weighted avg)
    total_samples = np.sum(support)
    accuracy = np.sum(true_positives) / total_samples if total_samples > 0 else 0.0
    
    macro_avg = {
        'precision': np.mean(precision),
        'recall': np.mean(recall),
        'f1-score': np.mean(f1_score)
    }
    
    weighted_avg = {
        'precision': np.average(precision, weights=support),
        'recall': np.average(recall, weights=support),
        'f1-score': np.average(f1_score, weights=support)
    }
    
    # Format akhir laporan
    final_report = {
        'accuracy': round(accuracy, 4),
        'confusion_matrix': cm.tolist(),
        'labels': labels,
        'per_class': per_class_metrics,
        'macro_avg': {k: round(v, 4) for k, v in macro_avg.items()},
        'weighted_avg': {k: round(v, 4) for k, v in weighted_avg.items()},
        'total_support': int(total_samples)
    }
    
    if logger:
        logger.info(f"Generated classification report. Accuracy: {final_report['accuracy']}")
        
    return final_report
