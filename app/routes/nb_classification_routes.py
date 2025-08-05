from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session
import os
import pickle
import numpy as np
from sqlalchemy import or_, cast, String
import json
import math
import uuid
import glob
from scipy import sparse
# Impor pustaka untuk Flask, file handling, serialisasi, array, SQLAlchemy, JSON, matematika, UUID, dan matriks sparse.

from app import db
from app.models import Preprocessing, KlasifikasiNB, DataSplit, ComparisonHistory, DataPakar
# Impor model database dari aplikasi.
from app.module.naive_bayes import MultinomialNaiveBayesClassifier, oversample_minority
# Impor kelas Naive Bayes dan fungsi oversampling.
from app.module.tfidf_vectorizer import CustomTfidf, preprocess_text_for_vectorizers, format_tfidf_for_display
# Impor kelas dan fungsi untuk TF-IDF vectorizer.
from app.utils import generate_classification_report, get_sampled_tfidf, SENTIMENT_LABELS
# Impor utilitas untuk laporan klasifikasi dan label sentimen.

nb_classification_bp = Blueprint('nb_classification_tasks', __name__, template_folder='../templates')
# Membuat blueprint Flask untuk rute klasifikasi Naive Bayes.

class ListPagination:
    def __init__(self, items_list_for_page, page, per_page, total_items_overall):
        # Inisialisasi objek paginasi dengan daftar item, halaman, item per halaman, dan total item.
        self.items = items_list_for_page
        self.page = page
        self.per_page = per_page
        self.total = total_items_overall
    
    @property
    def pages(self):
        # Menghitung jumlah total halaman.
        if self.per_page == 0 or self.total == 0: return 0
        return math.ceil(self.total / self.per_page)
    
    @property
    def has_prev(self): 
        # Memeriksa apakah ada halaman sebelumnya.
        return self.page > 1
    
    @property
    def prev_num(self): 
        # Mengembalikan nomor halaman sebelumnya jika ada.
        return self.page - 1 if self.has_prev else None
    
    @property
    def has_next(self): 
        # Memeriksa apakah ada halaman berikutnya.
        return self.page < self.pages
    
    @property
    def next_num(self): 
        # Mengembalikan nomor halaman berikutnya jika ada.
        return self.page + 1 if self.has_next else None

@nb_classification_bp.route('/naive_bayes', methods=['GET', 'POST'])
def classification_nb():
    # Rute untuk halaman klasifikasi Naive Bayes.
    current_app.logger.info(f"Route classification_nb dipanggil dengan metode: {request.method}")
    # Log metode HTTP yang digunakan.
    
    page = int(request.args.get("page", 1))
    # Mendapatkan nomor halaman dari query string, default 1.
    per_page = int(request.args.get("per_page", current_app.config.get('PER_PAGE', 10)))
    # Mendapatkan jumlah item per halaman dari konfigurasi, default 10.
    
    search_query = request.args.get("search", session.get('nb_search_query', ''))
    # Mendapatkan query pencarian dari parameter atau sesi.
    session['nb_search_query'] = search_query
    # Menyimpan query pencarian ke sesi.

    active_splits = DataSplit.query.filter_by(is_active=True).all()
    # Mengambil semua pembagian data aktif dari database.
    active_split_otomatis = next((s for s in active_splits if s.label_source == 'otomatis'), None)
    # Mengambil pembagian data aktif untuk sumber 'otomatis'.
    active_split_pakar = next((s for s in active_splits if s.label_source == 'pakar'), None)
    # Mengambil pembagian data aktif untuk sumber 'pakar'.
    
    test_ratio_display = active_split_otomatis.test_ratio if active_split_otomatis else active_split_pakar.test_ratio if active_split_pakar else None
    # Menentukan rasio tes untuk ditampilkan, prioritas 'otomatis'.
    
    model_folder = current_app.config['MODEL_FOLDER_PATH']
    # Mendapatkan path folder model dari konfigurasi aplikasi.

    should_run_classification = request.method == 'POST' and request.form.get('start_classification') == 'true'
    # Memeriksa apakah klasifikasi harus dijalankan (POST dengan flag tertentu).
    
    if should_run_classification:
        # Jika klasifikasi diminta:
        USE_OVERSAMPLING = False
        # Menetapkan status oversampling (default nonaktif).

        if not active_splits:
            # Jika tidak ada pembagian data aktif:
            flash('Silakan lakukan pembagian data terlebih dahulu di halaman Pembagian Data.', 'warning')
            # Tampilkan peringatan.
            return redirect(url_for('nb_classification_tasks.classification_nb'))
            # Redirect ke halaman klasifikasi.

        try:
            for source in ['otomatis', 'pakar']:
                # Iterasi untuk sumber label 'otomatis' dan 'pakar'.
                active_split = next((s for s in active_splits if s.label_source == source), None)
                # Mengambil pembagian data aktif untuk sumber saat ini.
                if not active_split:
                    flash(f'Pembagian data untuk sumber label "{source}" tidak ditemukan. Harap lakukan pembagian data lagi.', 'warning')
                    # Peringatan jika tidak ada pembagian data.
                    continue

                X_train_stem = pickle.loads(active_split.x_train_data)
                # Memuat data pelatihan (X_train) dari database.
                X_test_stem = pickle.loads(active_split.x_test_data)
                # Memuat data pengujian (X_test) dari database.
                y_train = pickle.loads(active_split.y_train_data)
                # Memuat label pelatihan (y_train) dari database.
                y_test = pickle.loads(active_split.y_test_data)
                # Memuat label pengujian (y_test) dari database.

                vectorizer = CustomTfidf(
                    max_features=current_app.config.get('TFIDF_MAX_FEATURES', 1000),
                    min_df=current_app.config.get('TFIDF_MIN_DF', 3),
                    max_df_ratio=current_app.config.get('TFIDF_MAX_DF_RATIO', 0.9),
                    ngram_range=current_app.config.get('TFIDF_NGRAM_RANGE', (1, 2))
                )
                # Inisialisasi TF-IDF vectorizer dengan parameter dari konfigurasi.
                X_train_tfidf = vectorizer.fit_transform(X_train_stem, normalize=False)
                # Melatih dan mengubah data pelatihan menjadi matriks TF-IDF.
                
                if USE_OVERSAMPLING:
                    current_app.logger.info(f"Oversampling AKTIF untuk sumber: {source}")
                    # Log jika oversampling aktif.
                    X_train_to_fit, y_train_to_fit = oversample_minority(X_train_tfidf, y_train)
                    # Melakukan oversampling pada data pelatihan.
                else:
                    current_app.logger.info(f"Oversampling NONAKTIF untuk sumber: {source}")
                    # Log jika oversampling nonaktif.
                    X_train_to_fit, y_train_to_fit = X_train_tfidf, y_train
                    # Gunakan data asli tanpa oversampling.
                
                nb_classifier = MultinomialNaiveBayesClassifier(alpha=1.0)
                # Inisialisasi model Naive Bayes dengan alpha=1.0.
                nb_classifier.fit(X_train_to_fit, y_train_to_fit)
                # Melatih model Naive Bayes.
                
                model_path = os.path.join(model_folder, 'nb', f'naive_bayes_model_{source}.pkl')
                # Menentukan path untuk menyimpan model.
                vectorizer_path = os.path.join(model_folder, f'tfidf_vectorizer_{source}.pkl')
                # Menentukan path untuk menyimpan vectorizer.
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                # Membuat folder jika belum ada.
                with open(model_path, 'wb') as f_model:
                    pickle.dump(nb_classifier, f_model)
                # Menyimpan model ke file.
                with open(vectorizer_path, 'wb') as f_vec:
                    pickle.dump(vectorizer, f_vec)
                # Menyimpan vectorizer ke file.
                
                X_test_tfidf = vectorizer.transform(X_test_stem, normalize=False)
                # Mengubah data pengujian menjadi matriks TF-IDF.
                y_pred = nb_classifier.predict(X_test_tfidf)
                # Memprediksi labelasi pada data pengujian.

                report = generate_classification_report(y_test, y_pred)
                # Membuat laporan klasifikasi (akurasi, presisi, dll.).
                report['model_params'] = {'alpha': nb_classifier.alpha}
                # Menyimpan parameter model (alpha).
                report['vectorizer_params'] = {
                    'max_features': vectorizer.max_features, 'min_df': vectorizer.min_df,
                    'max_df_ratio': vectorizer.max_df_ratio, 'ngram_range': vectorizer.ngram_range,
                    'vocab_size': len(vectorizer.vocabulary_)
                }
                # Menyimpan parameter vectorizer.
              
                report['data_stats'] = {
                    'total_train': len(y_train),
                    'total_train': len(y_train_to_fit),
                    'total_test': len(y_test),
                    'test_ratio': active_split.test_ratio,
                    'oversampling_status': 'Aktif' if USE_OVERSAMPLING else 'Nonaktif'
                }
                # Menyimpan statistik data (jumlah data, rasio, status oversampling).

                report['is_cross_validation'] = False
                # Menandakan bahwa ini bukan validasi silang.
                classification_report = {}
                for label in report['labels']:
                    classification_report[label] = report['per_class'][label]
                classification_report['macro avg'] = report['macro_avg']
                classification_report['weighted avg'] = report['weighted_avg']
                report['classification_report'] = classification_report
                # Menyusun laporan klasifikasi per kelas dan rata-rata.
                report['vectorizer_stats'] = report.get('vectorizer_params', {})
                # Menyimpan statistik vectorizer.

                KlasifikasiNB.query.filter_by(split_id=active_split.id).delete()
                # Menghapus hasil klasifikasi sebelumnya untuk split ini.
                
                pakar_data = {d.text_stem.strip().lower(): d for d in DataPakar.query.all() if d.text_stem}
                # Membuat kamus data pakar berdasarkan text_stem.
                preproc_data = {d.text_stem.strip().lower(): d for d in Preprocessing.query.all() if d.text_stem}
                # Membuat kamus data preprocessing berdasarkan text_stem.

                for ft_stem, y_true, y_pred_val in zip(X_test_stem, y_test, y_pred):
                    # Iterasi untuk setiap data pengujian.
                    test_text = ft_stem.strip().lower()
                    # Normalisasi teks pengujian.
                    
                    pakar_entry = pakar_data.get(test_text)
                    # Mengambil entri pakar berdasarkan teks.
                    preproc_entry = preproc_data.get(test_text)
                    # Mengambil entri preprocessing berdasarkan teks.

                    full_text_val, username_val = None, None
                    label_pakar_val, data_pakar_id_val = None, None
                    label_auto_val, preproc_id_val = None, None
                    # Inisialisasi variabel untuk menyimpan data.

                    if pakar_entry:
                        label_pakar_val = pakar_entry.label
                        data_pakar_id_val = pakar_entry.id
                        full_text_val = pakar_entry.full_text
                        username_val = getattr(pakar_entry, 'username', None)
                        # Mengisi data dari entri pakar jika ada.

                    if preproc_entry:
                        label_auto_val = preproc_entry.label_otomatis
                        preproc_id_val = preproc_entry.id
                        if not full_text_val:
                            full_text_val = preproc_entry.full_text
                        if not username_val:
                            username_val = getattr(preproc_entry, 'username', None)
                        # Mengisi data dari entri preprocessing jika ada.
                    
                    klas_entry = KlasifikasiNB(
                        full_text=full_text_val or ft_stem,
                        text_stem=ft_stem,
                        label_otomatis=label_auto_val,
                        label_pakar=label_pakar_val,
                        label_prediksi=y_pred_val,
                        username=username_val,
                        model_name='Naive Bayes',
                        test_ratio=active_split.test_ratio,
                        preprocessing_id=preproc_id_val,
                        split_id=active_split.id,
                        data_pakar_id=data_pakar_id_val
                    )
                    # Membuat entri klasifikasi baru.
                    db.session.add(klas_entry)
                    # Menambahkan entri ke sesi database.
                
                report_id = str(uuid.uuid4())
                # Membuat ID unik untuk laporan.
                report_path = os.path.join(model_folder, 'nb', f"last_classification_report_nb_{report_id}.pkl")
                # Menentukan path untuk menyimpan laporan.
                with open(report_path, 'wb') as f:
                    pickle.dump(report, f)
                # Menyimpan laporan ke file.
                session[f'nb_report_id_{source}'] = report_id
                # Menyimpan ID laporan ke sesi.

                history = ComparisonHistory.query.filter_by(split_id=active_split.id, label_source=source).first()
                # Mengambil riwayat perbandingan untuk split dan sumber ini.
                if not history:
                    history = ComparisonHistory(split_id=active_split.id, label_source=source)
                    db.session.add(history)
                    # Membuat riwayat baru jika belum ada.
                
                history.accuracy_nb = report.get('accuracy')
                history.nb_total_train = report['data_stats'].get('total_train')
                history.nb_total_test = report['data_stats'].get('total_test')
                history.nb_test_ratio = report['data_stats'].get('test_ratio')
                history.nb_vocab_size = report['vectorizer_params'].get('vocab_size')
                # Memperbarui riwayat dengan metrik klasifikasi.

            db.session.commit()
            # Menyimpan semua perubahan ke database.
            flash('Klasifikasi Naive Bayes untuk kedua sumber label berhasil dijalankan.', 'success')
            # Menampilkan pesan sukses.
            return redirect(url_for('nb_classification_tasks.classification_nb'))
            # Redirect ke halaman klasifikasi.

        except Exception as e:
            db.session.rollback()
            # Membatalkan perubahan jika terjadi error.
            current_app.logger.error(f"NB Classification error: {e}", exc_info=True)
            # Log error.
            flash(f"Error saat klasifikasi Naive Bayes: {str(e)}", "danger")
            # Menampilkan pesan error.
            return redirect(url_for('nb_classification_tasks.classification_nb'))
            # Redirect ke halaman klasifikasi.

    report_otomatis, report_pakar = None, None
    # Inisialisasi variabel untuk laporan otomatis dan pakar.
    report_id_otomatis = session.get('nb_report_id_otomatis')
    # Mengambil ID laporan otomatis dari sesi.
    report_id_pakar = session.get('nb_report_id_pakar')
    # Mengambil ID laporan pakar dari sesi.

    if report_id_otomatis:
        path = os.path.join(model_folder, 'nb', f"last_classification_report_nb_{report_id_otomatis}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f: report_otomatis = pickle.load(f)
            # Memuat laporan otomatis jika file ada.
    
    if report_id_pakar:
        path = os.path.join(model_folder, 'nb', f"last_classification_report_nb_{report_id_pakar}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f: report_pakar = pickle.load(f)
            # Memuat laporan pakar jika file ada.

    results_query = KlasifikasiNB.query.order_by(KlasifikasiNB.id.desc())
    # Mengambil hasil klasifikasi, diurutkan dari terbaru.
    if search_query:
        search_term = f"%{search_query}%"
        results_query = results_query.filter(
            or_(
                KlasifikasiNB.full_text.ilike(search_term),
                KlasifikasiNB.label_prediksi.ilike(search_term),
                cast(KlasifikasiNB.id, String).ilike(search_term)
            )
        )
        # Memfilter hasil berdasarkan query pencarian (teks, prediksi, atau ID).
    
    pagination = results_query.paginate(page=page, per_page=per_page, error_out=False)
    # Membuat paginasi untuk hasil query.

    return render_template('classification.html',
                           title='Klasifikasi Naive Bayes',
                           results=pagination.items,
                           report_otomatis=report_otomatis,
                           report_pakar=report_pakar,
                           data=pagination,
                           search_query=search_query,
                           per_page=per_page,
                           test_ratio=test_ratio_display)
    # Merender template dengan data hasil, laporan, dan paginasi.

@nb_classification_bp.route('/reset_nb_classification', methods=['POST'])
def reset_nb_classification():
    # Rute untuk mereset hasil klasifikasi Naive Bayes.
    try:
        KlasifikasiNB.query.delete()
        # Menghapus semua entri klasifikasi dari database.
        db.session.commit()
        # Menyimpan perubahan.

        session.pop('nb_report_id_otomatis', None)
        session.pop('nb_report_id_pakar', None)
        session.pop('nb_search_query', None)
        # Menghapus data sesi terkait klasifikasi.

        model_folder = current_app.config['MODEL_FOLDER_PATH']
        # Mendapatkan path folder model.
        
        nb_specific_patterns = ['last_classification_report_nb_*.pkl', 'naive_bayes_model.pkl', 'tfidf_vectorizer_*.pkl']
        # Pola file yang akan dihapus.
        
        for pattern in nb_specific_patterns:
            search_path = os.path.join(model_folder, 'nb', pattern)
            if 'tfidf' in pattern:
                search_path = os.path.join(model_folder, pattern)
            # Menyesuaikan path untuk file TF-IDF.
            
            for file_path in glob.glob(search_path):
                try:
                    os.remove(file_path)
                    current_app.logger.info(f"File berhasil dihapus: {file_path}")
                    # Menghapus file dan log keberhasilan.
                except Exception as e:
                    current_app.logger.error(f"Gagal menghapus file {file_path}: {e}")
                    # Log jika gagal menghapus file.

        flash('Hasil klasifikasi Naive Bayes berhasil direset.', 'success')
        # Menampilkan pesan sukses.
        
    except Exception as e:
        db.session.rollback()
        # Membatalkan perubahan jika terjadi error.
        current_app.logger.error(f"Gagal mereset klasifikasi NB: {e}", exc_info=True)
        # Log error.
        flash(f'Gagal mereset hasil klasifikasi: {str(e)}', 'danger')
        # Menampilkan pesan error.
        
    return redirect(url_for('nb_classification_tasks.classification_nb'))
    # Redirect ke halaman klasifikasi.