from flask import Blueprint, render_template, session, current_app, redirect, url_for, flash  # Impor modul Flask
import os  # Modul untuk operasi sistem file
import pickle  # Modul untuk memuat file pickle
from sqlalchemy import and_, func  # Impor fungsi SQLAlchemy
from app.models import KlasifikasiNB, KlasifikasiSVM, DataSplit, DataPakar, ComparisonHistory, Preprocessing  # Impor model database
from app import db  # Impor objek database

comparison_bp = Blueprint('comparison_tasks', __name__, template_folder='../templates')  # Buat blueprint Flask

@comparison_bp.route('/perbandingan', methods=['GET'])  # Rute untuk halaman perbandingan
def comparison_classification_results():
    model_folder = current_app.config['MODEL_FOLDER_PATH']  # Ambil path folder model
    
    def load_report(model_type, source):  # Fungsi untuk memuat laporan klasifikasi
        report_id = session.get(f'{model_type}_report_id_{source}')  # Ambil ID laporan dari sesi
        if not report_id:  # Periksa apakah ID laporan ada
            return None  # Kembalikan None jika tidak ada
        
        filename = f"last_classification_report_{model_type}_{report_id}.pkl"  # Nama file laporan
        report_path = os.path.join(model_folder, model_type, filename)  # Path lengkap file laporan
        
        if os.path.exists(report_path):  # Periksa apakah file ada
            try:
                with open(report_path, 'rb') as f:  # Buka file pickle
                    return pickle.load(f)  # Muat laporan
            except Exception as e:  # Tangani error
                current_app.logger.error(f"Error loading {model_type} report for {source} from {report_path}: {e}", exc_info=True)  # Log error
                session.pop(f'{model_type}_report_id_{source}', None)  # Hapus ID dari sesi
        else:
            current_app.logger.warning(f"Report file not found for {model_type} ({source}): {report_path}")  # Log peringatan
        return None  # Kembalikan None jika gagal

    report_nb_otomatis = load_report('nb', 'otomatis')  # Muat laporan Naive Bayes otomatis
    report_nb_pakar = load_report('nb', 'pakar')  # Muat laporan Naive Bayes pakar
    report_svm_otomatis = load_report('svm', 'otomatis')  # Muat laporan SVM otomatis
    report_svm_pakar = load_report('svm', 'pakar')  # Muat laporan SVM pakar
    
    active_split = DataSplit.query.filter_by(is_active=True).first()  # Ambil data split aktif
    test_ratio = active_split.test_ratio if active_split else None  # Ambil rasio tes

    comparison_data = []  # Inisialisasi list untuk data perbandingan
    from collections import Counter  # Impor Counter untuk menghitung label
    pakar_counter = Counter()  # Counter untuk label pakar
    otomatis_counter = Counter()  # Counter untuk label otomatis
    nb_counter = Counter()  # Counter untuk label Naive Bayes
    svm_counter = Counter()  # Counter untuk label SVM
    if test_ratio:  # Periksa apakah rasio tes ada
        nb_results = KlasifikasiNB.query.filter(func.abs(KlasifikasiNB.test_ratio - test_ratio) < 1e-6).all()  # Ambil hasil NB
        svm_results = KlasifikasiSVM.query.filter(func.abs(KlasifikasiSVM.test_ratio - test_ratio) < 1e-6).all()  # Ambil hasil SVM

        nb_dict = {item.full_text: item.label_prediksi for item in nb_results}  # Buat kamus NB: teks -> prediksi
        svm_dict = {item.full_text: item.label_prediksi for item in svm_results}  # Buat kamus SVM: teks -> prediksi

        data_pakar_all = DataPakar.query.all()  # Ambil semua data pakar
        data_pakar_map = { (dp.full_text or '').strip().casefold(): dp.label for dp in data_pakar_all }  # Buat kamus pakar

        all_texts = set(nb_dict.keys()) | set(svm_dict.keys())  # Gabungkan semua teks unik
        for text in all_texts:  # Iterasi setiap teks
            label_otomatis = None  # Inisialisasi label otomatis
            label_pakar = None  # Inisialisasi label pakar
            label_otomatis = next((item.label_otomatis for item in nb_results if item.full_text == text), None)  # Ambil label otomatis dari NB
            if not label_otomatis:  # Fallback ke SVM jika tidak ada di NB
                label_otomatis = next((item.label_otomatis for item in svm_results if item.full_text == text), None)
            label_pakar = data_pakar_map.get((text or '').strip().casefold())  # Ambil label pakar

            comparison_data.append({  # Tambah data perbandingan
                'full_text': text,  # Teks asli
                'label_otomatis': label_otomatis,  # Label otomatis
                'label_pakar': label_pakar,  # Label pakar
                'Naive Bayes': nb_dict.get(text, '-'),  # Prediksi NB
                'SVM': svm_dict.get(text, '-')  # Prediksi SVM
            })

        from collections import Counter  # Impor ulang Counter (redundan)
        pakar_counter = Counter()  # Reset counter pakar
        otomatis_counter = Counter()  # Reset counter otomatis
        nb_counter = Counter()  # Reset counter NB
        svm_counter = Counter()  # Reset counter SVM
        for row in comparison_data:  # Iterasi data perbandingan
            if row['label_pakar']:  # Jika ada label pakar
                pakar_counter[row['label_pakar']] += 1  # Tambah hitungan
            if row['label_otomatis']:  # Jika ada label otomatis
                otomatis_counter[row['label_otomatis']] += 1  # Tambah hitungan
            if row['Naive Bayes']:  # Jika ada prediksi NB
                nb_counter[row['Naive Bayes']] += 1  # Tambah hitungan
            if row['SVM']:  # Jika ada prediksi SVM
                svm_counter[row['SVM']] += 1  # Tambah hitungan

    history = ComparisonHistory.query.order_by(ComparisonHistory.created_at.desc()).all()  # Ambil 10 riwayat terbaru.limit(10)

    if not history:  # Periksa apakah riwayat kosong
        flash('Tidak ada riwayat perbandingan yang ditemukan. Pastikan proses klasifikasi dan penyimpanan berjalan dengan benar.', 'warning')  # Tampilkan peringatan

    return render_template('perbandingan_klasifikasi.html',  # Render template
                           title='Perbandingan',  # Judul halaman
                           report_nb_otomatis=report_nb_otomatis,  # Laporan NB otomatis
                           report_nb_pakar=report_nb_pakar,  # Laporan NB pakar
                           report_svm_otomatis=report_svm_otomatis,  # Laporan SVM otomatis
                           report_svm_pakar=report_svm_pakar,  # Laporan SVM pakar
                           comparison_data=comparison_data,  # Data perbandingan
                           pakar_counter=pakar_counter,  # Counter label pakar
                           otomatis_counter=otomatis_counter,  # Counter label otomatis
                           nb_counter=nb_counter,  # Counter prediksi NB
                           svm_counter=svm_counter,  # Counter prediksi SVM
                           history=history)  # Riwayat perbandingan

@comparison_bp.route('/reset_comparison_history', methods=['POST'])  # Rute untuk reset riwayat
def reset_comparison_history():
    try:
        num_deleted = ComparisonHistory.query.delete()  # Hapus semua riwayat
        db.session.commit()  # Simpan perubahan
        flash(f'Berhasil menghapus {num_deleted} riwayat perbandingan.', 'success')  # Tampilkan pesan sukses
    except Exception as e:  # Tangani error
        db.session.rollback()  # Batalkan perubahan
        flash('Gagal menghapus riwayat perbandingan.', 'danger')  # Tampilkan pesan error
    return redirect(url_for('comparison_tasks.comparison_classification_results'))  # Redirect ke halaman perbandingan