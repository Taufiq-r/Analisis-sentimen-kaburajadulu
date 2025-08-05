from flask import Blueprint, render_template, flash  # Impor modul Flask
from app.models import ComparisonHistory, KlasifikasiNB, KlasifikasiSVM, DataSplit  # Impor model database
from app import db  # Impor objek database
import math  # Modul untuk operasi matematika

kesimpulan_bp = Blueprint(
    'kesimpulan_tasks', 
    __name__, 
    template_folder='../templates'
)  # Buat blueprint Flask

def get_best_model_analysis(label_source_value: str):
    # Analisis model terbaik berdasarkan sumber label
    history = ComparisonHistory.query.filter_by(label_source=label_source_value).order_by(ComparisonHistory.id.desc()).first()  # Ambil riwayat terbaru
    if not history:  # Periksa apakah riwayat ada
        return None  # Kembalikan None jika tidak ada

    ModelTabelNB = KlasifikasiNB  # Tabel untuk Naive Bayes
    ModelTabelSVM = KlasifikasiSVM  # Tabel untuk SVM

    nb_data_exists = True  # Inisialisasi status data NB
    if history.accuracy_nb is not None:  # Periksa akurasi NB
        nb_data_exists = ModelTabelNB.query.filter_by(split_id=history.split_id).first() is not None  # Verifikasi data NB

    svm_data_exists = True  # Inisialisasi status data SVM
    if history.accuracy_svm is not None:  # Periksa akurasi SVM
        svm_data_exists = ModelTabelSVM.query.filter_by(split_id=history.split_id).first() is not None  # Verifikasi data SVM

    if not nb_data_exists and not svm_data_exists:  # Periksa apakah kedua data tidak ada
        return None  # Kembalikan None jika tidak ada data

    accuracy_nb = history.accuracy_nb or 0  # Ambil akurasi NB, default 0
    total_test_nb = history.nb_total_test or 0  # Ambil total tes NB, default 0
    correct_nb = round(accuracy_nb * total_test_nb) if nb_data_exists else 0  # Hitung prediksi benar NB

    accuracy_svm = history.accuracy_svm or 0  # Ambil akurasi SVM, default 0
    total_test_svm = history.svm_total_test or 0  # Ambil total tes SVM, default 0
    correct_svm = round(accuracy_svm * total_test_svm) if svm_data_exists else 0  # Hitung prediksi benar SVM
    
    if correct_nb == 0 and correct_svm == 0:  # Periksa apakah tidak ada prediksi benar
        return None  # Kembalikan None jika tidak ada

    best_model_name = "Naive Bayes" if correct_nb >= correct_svm else "SVM"  # Pilih model terbaik
    best_model_correct_count = max(correct_nb, correct_svm)  # Ambil jumlah prediksi benar terbaik
    
    BestModelTabel = ModelTabelNB if best_model_name == "Naive Bayes" else ModelTabelSVM  # Pilih tabel model terbaik
    label_column = 'label_pakar' if label_source_value == 'pakar' else 'label_otomatis'  # Tentukan kolom label
    
    correctly_predicted_pos = BestModelTabel.query.filter(
        BestModelTabel.split_id == history.split_id, getattr(BestModelTabel, label_column) == 'positif', BestModelTabel.label_prediksi == 'positif'
    ).count()  # Hitung prediksi benar positif
    correctly_predicted_neg = BestModelTabel.query.filter(
        BestModelTabel.split_id == history.split_id, getattr(BestModelTabel, label_column) == 'negatif', BestModelTabel.label_prediksi == 'negatif'
    ).count()  # Hitung prediksi benar negatif
    correctly_predicted_neu = BestModelTabel.query.filter(
        BestModelTabel.split_id == history.split_id, getattr(BestModelTabel, label_column) == 'netral', BestModelTabel.label_prediksi == 'netral'
    ).count()  # Hitung prediksi benar netral
    
    correct_counts = {
        'Positif': correctly_predicted_pos, 'Negatif': correctly_predicted_neg, 'Netral': correctly_predicted_neu
    }  # Simpan jumlah prediksi benar per sentimen

    dominant_sentiment_by_correctness = "Tidak Ada Hasil"  # Default jika tidak ada hasil
    if any(v > 0 for v in correct_counts.values()):  # Periksa apakah ada prediksi benar
        dominant_sentiment_by_correctness = max(correct_counts, key=correct_counts.get)  # Pilih sentimen dominan
    
    return {
        "correct_nb": correct_nb,  # Jumlah prediksi benar NB
        "correct_svm": correct_svm,  # Jumlah prediksi benar SVM
        "best_model_name": best_model_name,  # Nama model terbaik
        "best_model_correct_count": best_model_correct_count,  # Jumlah prediksi benar terbaik
        "dominant_sentiment": dominant_sentiment_by_correctness,  # Sentimen dominan
        "correctly_predicted_pos": correctly_predicted_pos,  # Prediksi benar positif
        "correctly_predicted_neg": correctly_predicted_neg,  # Prediksi benar negatif
        "correctly_predicted_neu": correctly_predicted_neu  # Prediksi benar netral
    }  # Kembalikan hasil analisis

@kesimpulan_bp.route('/perbandingan-kesimpulan-prediksi-benar')  # Rute untuk halaman kesimpulan
def comparison_conclusion_by_correct_prediction():
    try:
        pakar_results = get_best_model_analysis('pakar')  # Analisis model untuk label pakar
        otomatis_results = get_best_model_analysis('otomatis')  # Analisis model untuk label otomatis

        if not pakar_results and not otomatis_results:  # Periksa apakah tidak ada hasil
            flash('Jalankan proses klasifikasi untuk "Label Pakar" dan/atau "Label Otomatis" terlebih dahulu untuk melihat kesimpulan.', 'info')  # Tampilkan peringatan

        return render_template(
            'kesimpulan_akhir.html',  # Render template
            title='Kesimpulan Prediksi Benar Terbanyak',  # Judul halaman
            pakar=pakar_results,  # Hasil analisis pakar
            otomatis=otomatis_results  # Hasil analisis otomatis
        )
                             
    except Exception as e:  # Tangani error
        flash(f'Terjadi kesalahan saat memuat halaman perbandingan: {e}', 'danger')  # Tampilkan pesan error
        return render_template('kesimpulan_akhir.html', title='Kesimpulan Prediksi Benar Terbanyak', error=str(e))  # Render template dengan error