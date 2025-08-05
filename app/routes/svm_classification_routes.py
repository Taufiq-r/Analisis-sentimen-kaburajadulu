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
from app.models import Preprocessing, KlasifikasiSVM, DataSplit, ComparisonHistory, DataPakar
# Impor model database dari aplikasi.
from app.module.svm import SupportVectorMachine
# Impor kelas Support Vector Machine.
from app.module.naive_bayes import oversample_minority
# Impor fungsi oversampling dari modul Naive Bayes.
from app.module.tfidf_vectorizer import CustomTfidf, preprocess_text_for_vectorizers
# Impor kelas dan fungsi untuk TF-IDF vectorizer.
from app.utils import generate_classification_report, get_sampled_tfidf, SENTIMENT_LABELS
# Impor utilitas untuk laporan klasifikasi dan label sentimen.

svm_classification_bp = Blueprint('svm_classification_tasks', __name__, template_folder='../templates')
# Membuat blueprint Flask untuk rute klasifikasi SVM.

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
    
    def iter_pages(self, left_edge=1, right_edge=1, left_current=2, right_current=2):
        # Menghasilkan daftar nomor halaman untuk navigasi paginasi.
        if self.pages <= 1: return []
        page_numbers = []
        if 1 not in page_numbers: page_numbers.append(1)
        # Tambahkan halaman pertama.
        start_curr = max(2, self.page - left_current)
        end_curr = min(self.pages - 1, self.page + right_current)
        # Tentukan rentang halaman sekitar halaman saat ini.
        for i in range(start_curr, end_curr + 1): page_numbers.append(i)
        # Tambahkan halaman dalam rentang.
        if self.pages not in page_numbers: page_numbers.append(self.pages)
        # Tambahkan halaman terakhir.
        page_numbers = sorted(list(set(page_numbers))) 
        # Urutkan dan hapus duplikat.
        final_pages = []
        last_p = 0
        for p in page_numbers:
            if p > last_p + 1: 
                if not (p == 2 and last_p == 1) and not (len(final_pages) > 0 and final_pages[-1] is None):
                    final_pages.append(None)
                    # Tambahkan None untuk jeda antar halaman.
            final_pages.append(p)
            last_p = p
        if self.pages <= (left_edge + left_current + right_current + right_edge + 1): 
            return [p for p in final_pages if p is not None]
            # Kembalikan hanya nomor halaman jika total halaman kecil.
        return final_pages
        # Kembalikan daftar halaman dengan jeda.

@svm_classification_bp.route('/svm', methods=['GET', 'POST'])
def classification_svm():
    # Rute untuk halaman klasifikasi SVM.
    current_app.logger.info(f"Route classification_svm called with method: {request.method}")
    # Log metode HTTP yang digunakan.
    
    page = int(request.args.get("page", 1))
    # Dapatkan nomor halaman dari query string, default 1.
    per_page = int(request.args.get("per_page", current_app.config.get('PER_PAGE', 10)))
    # Dapatkan jumlah item per halaman dari konfigurasi, default 10.

    search_query = request.args.get("search", session.get('svm_search_query', ''))
    # Dapatkan query pencarian dari parameter atau sesi.
    session['svm_search_query'] = search_query
    # Simpan query pencarian ke sesi.

    active_splits = DataSplit.query.filter_by(is_active=True).all()
    # Ambil semua pembagian data aktif dari database.
    active_split_otomatis = next((s for s in active_splits if s.label_source == 'otomatis'), None)
    # Ambil pembagian data aktif untuk sumber 'otomatis'.
    active_split_pakar = next((s for s in active_splits if s.label_source == 'pakar'), None)
    # Ambil pembagian data aktif untuk sumber 'pakar'.
    
    test_ratio_display = active_split_otomatis.test_ratio if active_split_otomatis else active_split_pakar.test_ratio if active_split_pakar else None
    # Tentukan rasio tes untuk ditampilkan, prioritas 'otomatis'.
    
    model_folder = current_app.config['MODEL_FOLDER_PATH']
    # Dapatkan path folder model dari konfigurasi aplikasi.

    should_run_classification = request.method == 'POST' and request.form.get('start_classification') == 'true'
    # Periksa apakah klasifikasi harus dijalankan (POST dengan flag tertentu).

    if should_run_classification:
        # Jika klasifikasi diminta:
        USE_OVERSAMPLING = False
        # Set status oversampling (default nonaktif).
       
        if not active_splits:
            flash('Silakan lakukan pembagian data terlebih dahulu di halaman Pembagian Data.', 'warning')
            # Tampilkan peringatan jika tidak ada pembagian data.
            return redirect(url_for('svm_classification_tasks.classification_svm'))
            # Redirect ke halaman klasifikasi.

        try:
            for source in ['otomatis', 'pakar']:
                # Iterasi untuk sumber label 'otomatis' dan 'pakar'.
                active_split = next((s for s in active_splits if s.label_source == source), None)
                # Ambil pembagian data aktif untuk sumber saat ini.
                if not active_split:
                    flash(f'Pembagian data untuk sumber label "{source}" tidak ditemukan. Harap lakukan pembagian data lagi.', 'warning')
                    # Peringatkan jika tidak ada pembagian data.
                    continue

                X_train = pickle.loads(active_split.x_train_data)
                # Muat data pelatihan (X_train) dari database.
                y_train = pickle.loads(active_split.y_train_data)
                # Muat label pelatihan (y_train) dari database.
                X_test = pickle.loads(active_split.x_test_data)
                # Muat data pengujian (X_test) dari database.
                y_test = pickle.loads(active_split.y_test_data)
                # Muat label pengujian (y_test) dari database.

                vectorizer_path = os.path.join(model_folder, f'tfidf_vectorizer_{source}.pkl')
                # Tentukan path untuk file vectorizer.
                if not os.path.exists(vectorizer_path):
                    flash(f'File TF-IDF Vectorizer untuk sumber "{source}" tidak ditemukan. Harap jalankan klasifikasi Naive Bayes terlebih dahulu.', 'danger')
                    # Peringatkan jika file vectorizer tidak ada.
                    continue
                
                with open(vectorizer_path, 'rb') as f_vec:
                    vectorizer = pickle.load(f_vec)
                # Muat vectorizer dari file.

                X_train_vectorized = vectorizer.transform(X_train, normalize=True)
                # Ubah data pelatihan menjadi matriks TF-IDF.
                X_test_vectorized = vectorizer.transform(X_test, normalize=True)
                # Ubah data pengujian menjadi matriks TF-IDF.

                if X_train_vectorized.shape[0] == 0:
                    flash(f'Data latih untuk sumber "{source}" menjadi kosong setelah proses vectorisasi.', 'danger')
                    # Peringatkan jika data latih kosong setelah vectorisasi.
                    continue

                svm_classifier = SupportVectorMachine(
                    label_source=source,
                    random_state=42,
                    use_oversampling=USE_OVERSAMPLING
                )
                # Inisialisasi model SVM dengan sumber label dan status oversampling.
              
                svm_classifier.fit(X_train_vectorized, y_train)
                # Latih model SVM.
                y_pred = svm_classifier.predict(X_test_vectorized)
                # Prediksi label pada data pengujian.

                svm_model_path = os.path.join(model_folder, 'svm', f'svm_model_{source}.pkl')
                # Tentukan path untuk menyimpan model SVM.
                os.makedirs(os.path.dirname(svm_model_path), exist_ok=True)
                # Buat folder jika belum ada.
                with open(svm_model_path, 'wb') as f:
                    pickle.dump(svm_classifier, f)
                # Simpan model SVM ke file.
                
                report = generate_classification_report(y_test, y_pred)
                # Buat laporan klasifikasi (akurasi, presisi, dll.).
                report['model_params'] = svm_classifier.get_params()
                # Simpan parameter model SVM.
                report['vectorizer_params'] = {
                    'max_features': vectorizer.max_features, 'min_df': vectorizer.min_df,
                    'max_df_ratio': vectorizer.max_df_ratio, 'ngram_range': vectorizer.ngram_range,
                    'vocab_size': len(vectorizer.vocabulary_)
                }
                # Simpan parameter vectorizer.
                
                y_train_fit_count = svm_classifier.X_train_.shape[0] if hasattr(svm_classifier, 'X_train_') else len(y_train)
                # Hitung jumlah data latih setelah oversampling (jika ada).
              
                report['data_stats'] = {
                    'total_train': y_train_fit_count, 
                    'total_test': len(y_test), 
                    'test_ratio': active_split.test_ratio,
                    'oversampling_status': 'Aktif' if USE_OVERSAMPLING else 'Nonaktif'
                }
                # Simpan statistik data (jumlah data, rasio, status oversampling).

                report['vectorizer_stats'] = report.get('vectorizer_params', {})
                # Simpan statistik vectorizer.
                classification_report = {}
                for label in report['labels']:
                    classification_report[label] = report['per_class'][label]
                classification_report['macro avg'] = report['macro_avg']
                classification_report['weighted avg'] = report['weighted_avg']
                report['classification_report'] = classification_report
                # Susun laporan klasifikasi per kelas dan rata-rata.

                KlasifikasiSVM.query.filter_by(split_id=active_split.id).delete()
                # Hapus hasil klasifikasi sebelumnya untuk split ini.
                
                pakar_data = {d.text_stem.strip().lower(): d for d in DataPakar.query.all() if d.text_stem}
                # Buat kamus data pakar berdasarkan text_stem.
                preproc_data = {d.text_stem.strip().lower(): d for d in Preprocessing.query.all() if d.text_stem}
                # Buat kamus data preprocessing berdasarkan text_stem.

                for ft_stem, lm, lp in zip(X_test, y_test, y_pred):
                    # Iterasi untuk setiap data pengujian.
                    test_text = ft_stem.strip().lower()
                    # Normalisasi teks pengujian.
                    
                    pakar_entry = pakar_data.get(test_text)
                    # Ambil entri pakar berdasarkan teks.
                    preproc_entry = preproc_data.get(test_text)
                    # Ambil entri preprocessing berdasarkan teks.

                    full_text_val, username_val = None, None
                    label_pakar_val, data_pakar_id_val = None, None
                    label_auto_val, preproc_id_val = None, None
                    # Inisialisasi variabel untuk menyimpan data.

                    if pakar_entry:
                        label_pakar_val = pakar_entry.label
                        data_pakar_id_val = pakar_entry.id
                        full_text_val = pakar_entry.full_text
                        username_val = getattr(pakar_entry, 'username', None)
                        # Isi data dari entri pakar jika ada.

                    if preproc_entry:
                        label_auto_val = preproc_entry.label_otomatis
                        preproc_id_val = preproc_entry.id
                        if not full_text_val:
                            full_text_val = preproc_entry.full_text
                        if not username_val:
                            username_val = getattr(preproc_entry, 'username', None)
                        # Isi data dari entri preprocessing jika ada.

                    klas_entry = KlasifikasiSVM(
                        full_text=full_text_val or ft_stem,
                        text_stem=ft_stem,
                        label_otomatis=label_auto_val,
                        label_pakar=label_pakar_val,
                        label_prediksi=lp,
                        username=username_val,
                        model_name='SVM',
                        test_ratio=active_split.test_ratio,
                        preprocessing_id=preproc_id_val,
                        split_id=active_split.id,
                        data_pakar_id=data_pakar_id_val
                    )
                    # Buat entri klasifikasi baru.
                    db.session.add(klas_entry)
                    # Tambahkan entri ke sesi database.

                report_id = str(uuid.uuid4())
                # Buat ID unik untuk laporan.
                report_path = os.path.join(model_folder, 'svm', f"last_classification_report_svm_{report_id}.pkl")
                # Tentukan path untuk menyimpan laporan.
                with open(report_path, 'wb') as f:
                    pickle.dump(report, f)
                # Simpan laporan ke file.
                session[f'svm_report_id_{source}'] = report_id
                # Simpan ID laporan ke sesi.

                history = ComparisonHistory.query.filter_by(split_id=active_split.id, label_source=source).first()
                # Ambil riwayat perbandingan untuk split dan sumber ini.
                if not history:
                    history = ComparisonHistory(split_id=active_split.id, label_source=source)
                    db.session.add(history)
                    # Buat riwayat baru jika belum ada.
                
                if history:
                    history.accuracy_svm = report.get('accuracy')
                    history.svm_total_train = report['data_stats'].get('total_train')
                    history.svm_total_test = report['data_stats'].get('total_test')
                    history.svm_test_ratio = report['data_stats'].get('test_ratio')
                    history.svm_learning_rate = report.get('model_params', {}).get('learning_rate')
                    history.svm_lambda = report.get('model_params', {}).get('lambda_param')
                    history.svm_vocab_size = report.get('vectorizer_params', {}).get('vocab_size')
                    # Perbarui riwayat dengan metrik klasifikasi.

            db.session.commit()
            # Simpan semua perubahan ke database.
            flash('Klasifikasi SVM untuk kedua sumber label berhasil dijalankan.', 'success')
            # Tampilkan pesan sukses.
            return redirect(url_for('svm_classification_tasks.classification_svm'))
            # Redirect ke halaman klasifikasi.

        except Exception as e:
            db.session.rollback()
            # Batalkan perubahan jika terjadi error.
            current_app.logger.error(f"SVM Classification error: {str(e)}", exc_info=True)
            # Log error.
            flash(f'Terjadi kesalahan saat klasifikasi: {str(e)}', 'danger')
            # Tampilkan pesan error.
            return redirect(url_for('svm_classification_tasks.classification_svm'))
            # Redirect ke halaman klasifikasi.

    report_otomatis, report_pakar = None, None
    # Inisialisasi variabel untuk laporan otomatis dan pakar.
    report_id_otomatis = session.get('svm_report_id_otomatis')
    # Ambil ID laporan otomatis dari sesi.
    report_id_pakar = session.get('svm_report_id_pakar')
    # Ambil ID laporan pakar dari sesi.

    if report_id_otomatis:
        path = os.path.join(model_folder, 'svm', f"last_classification_report_svm_{report_id_otomatis}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f: report_otomatis = pickle.load(f)
            # Muat laporan otomatis jika file ada.
    
    if report_id_pakar:
        path = os.path.join(model_folder, 'svm', f"last_classification_report_svm_{report_id_pakar}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f: report_pakar = pickle.load(f)
            # Muat laporan pakar jika file ada.
    
    results_query = KlasifikasiSVM.query
    # Inisialisasi query untuk hasil klasifikasi.
    if search_query:
        search_term_db = f"%{search_query}%"
        results_query = results_query.filter(
            or_(
                cast(KlasifikasiSVM.id, String).ilike(search_term_db),
                KlasifikasiSVM.full_text.ilike(search_term_db),
                KlasifikasiSVM.label_prediksi.ilike(search_term_db)
            )
        )
        # Filter hasil berdasarkan query pencarian (ID, teks, atau prediksi).

    data_pagination_from_db = results_query.order_by(KlasifikasiSVM.id.desc()).paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    # Buat paginasi untuk hasil query, urutkan dari terbaru.

    return render_template('classification_svm.html',
                           title='Klasifikasi SVM',
                           results=data_pagination_from_db.items if data_pagination_from_db else [],
                           report_otomatis=report_otomatis,
                           report_pakar=report_pakar,
                           data=data_pagination_from_db,
                           search_query=search_query,
                           per_page=per_page,
                           test_ratio=test_ratio_display)
    # Render template dengan data hasil, laporan, dan paginasi.

@svm_classification_bp.route('/reset_svm_classification', methods=['POST'])
def reset_svm_classification():
    # Rute untuk mereset hasil klasifikasi SVM.
    try:
        KlasifikasiSVM.query.delete()
        # Hapus semua entri klasifikasi dari database.
        db.session.commit()
        # Simpan perubahan.
        
        session.pop('svm_report_id_otomatis', None)
        session.pop('svm_report_id_pakar', None)
        session.pop('svm_search_query', None)
        # Hapus data sesi terkait klasifikasi.

        model_folder = current_app.config['MODEL_FOLDER_PATH']
        # Dapatkan path folder model.
       
        for file_pattern in ['svm_model_*.pkl', 'last_classification_report_svm_*.pkl']:
            # Pola file yang akan dihapus.
            for file_path_to_delete in glob.glob(os.path.join(model_folder, 'svm', file_pattern)):
                if os.path.exists(file_path_to_delete):
                    try:
                        os.remove(file_path_to_delete)
                        # Hapus file.
                    except Exception as e:
                        current_app.logger.error(f"Gagal menghapus file {file_path_to_delete}: {e}")
                        # Log jika gagal menghapus file.

        flash('Hasil klasifikasi SVM berhasil direset.', 'success')
        # Tampilkan pesan sukses.
        
    except Exception as e:
        db.session.rollback()
        # Batalkan perubahan jika terjadi error.
        current_app.logger.error(f"Gagal reset klasifikasi: {e}", exc_info=True)
        # Log error.
        flash(f'Gagal reset hasil klasifikasi: {str(e)}', 'danger')
        # Tampilkan pesan error.
    return redirect(url_for('svm_classification_tasks.classification_svm'))
    # Redirect ke halaman klasifikasi.