from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session  # Impor modul Flask
import os  # Modul untuk operasi sistem file
import pickle  # Modul untuk serialisasi data
import numpy as np  # Modul untuk operasi numerik
from sqlalchemy import or_  # Impor fungsi SQLAlchemy
import json  # Modul untuk memproses JSON
import math  # Modul untuk operasi matematika
import uuid  # Modul untuk menghasilkan UUID
import random  # Modul untuk operasi acak

from app import db  # Impor objek database
from app.models import Preprocessing, DataSplit, DataPakar  # Impor model database
from app.utils import SENTIMENT_LABELS  # Impor label sentimen

split_data_bp = Blueprint('classification_tasks', __name__, template_folder='../templates')  # Buat blueprint Flask

@split_data_bp.route('/select_test_ratio', methods=['GET', 'POST'])  # Rute untuk pembagian data
def select_test_ratio():
    """
    Menangani logika untuk membagi data menjadi set pelatihan dan pengujian.
    Mendukung sumber data dari 'label otomatis' dan 'label pakar'.
    Menyimpan hasil pembagian ke database untuk digunakan oleh model klasifikasi.
    """
    # Hitung jumlah data untuk UI
    total_otomatis = Preprocessing.query.filter(Preprocessing.label_otomatis.isnot(None), Preprocessing.label_otomatis != '').count()  # Jumlah data berlabel otomatis
    total_pakar = DataPakar.query.filter(DataPakar.label.isnot(None), DataPakar.label != '').count()  # Jumlah data berlabel pakar
    total_labeled = total_otomatis + total_pakar  # Total data berlabel
    total_data = Preprocessing.query.count()  # Total semua data
    total_unlabeled = total_data - total_labeled  # Total data tanpa label

    if request.method == 'POST':  # Tangani permintaan POST
        test_ratio_str = request.form.get('test_ratio')  # Ambil rasio tes dari form
        if not test_ratio_str:  # Periksa apakah rasio tes kosong
            flash('Silakan pilih rasio data uji terlebih dahulu.', 'warning')  # Tampilkan peringatan
            return redirect(url_for('classification_tasks.select_test_ratio'))  # Redirect ke halaman

        try:
            test_ratio = float(test_ratio_str)  # Konversi rasio ke float
            if not (0.0 < test_ratio < 1.0):  # Validasi rasio
                flash('Rasio data uji harus antara 0.0 dan 1.0 (eksklusif).', 'warning')  # Tampilkan peringatan
                return redirect(url_for('classification_tasks.select_test_ratio'))  # Redirect ke halaman

            # Nonaktifkan semua split sebelumnya
            DataSplit.query.update({DataSplit.is_active: False})  # Set semua split ke non-aktif
            db.session.commit()  # Simpan perubahan

            success_messages = []  # List untuk pesan sukses
            error_messages = []  # List untuk pesan error

            for source in ['otomatis', 'pakar']:  # Iterasi sumber data
                source_data = []  # Inisialisasi data sumber
                if source == 'otomatis':  # Jika sumber otomatis
                    source_data = Preprocessing.query.filter(
                        Preprocessing.text_stem.isnot(None), Preprocessing.text_stem != '',
                        Preprocessing.label_otomatis.isnot(None), Preprocessing.label_otomatis != ''
                    ).all()  # Ambil data berlabel otomatis
                else:  # Jika sumber pakar
                    source_data = DataPakar.query.filter(
                        DataPakar.text_stem.isnot(None), DataPakar.text_stem != '',
                        DataPakar.label.isnot(None), DataPakar.label != ''
                    ).all()  # Ambil data berlabel pakar

                if not source_data:  # Periksa apakah data kosong
                    error_messages.append(f'Tidak ada data berlabel {source} yang valid untuk dibagi.')  # Tambah pesan error
                    continue

                all_ids = np.array([d.id for d in source_data])  # Ambil ID data
                all_texts = np.array([d.text_stem for d in source_data])  # Ambil teks stemmed
                all_labels = np.array([d.label if source == 'pakar' else d.label_otomatis for d in source_data])  # Ambil label

                train_indices, test_indices = [], []  # Inisialisasi indeks latih dan uji
                unique_labels = np.unique(all_labels)  # Ambil label unik
                
                for label in unique_labels:  # Iterasi setiap label
                    label_indices = np.where(all_labels == label)[0]  # Ambil indeks label
                    np.random.shuffle(label_indices)  # Acak indeks
                    n_test_for_label = math.ceil(len(label_indices) * test_ratio)  # Hitung jumlah data uji
                    if len(label_indices) == 1:  # Jika hanya satu data
                        n_test_for_label = 0  # Tidak ada data uji
                    elif len(label_indices) == n_test_for_label:  # Jika semua data jadi uji
                        n_test_for_label -= 1  # Kurangi satu
                    test_indices.extend(label_indices[:n_test_for_label])  # Tambah indeks uji
                    train_indices.extend(label_indices[n_test_for_label:])  # Tambah indeks latih

                random.shuffle(train_indices)  # Acak indeks latih
                random.shuffle(test_indices)  # Acak indeks uji

                X_train, y_train = all_texts[train_indices], all_labels[train_indices]  # Data latih
                X_test, y_test = all_texts[test_indices], all_labels[test_indices]  # Data uji

                new_split = DataSplit(  # Buat entri DataSplit baru
                    test_ratio=test_ratio,  # Simpan rasio tes
                    train_size=len(X_train),  # Jumlah data latih
                    test_size=len(X_test),  # Jumlah data uji
                    train_indices=json.dumps(np.array(train_indices).tolist()),  # Simpan indeks latih
                    test_indices=json.dumps(np.array(test_indices).tolist()),  # Simpan indeks uji
                    is_active=True,  # Tandai split aktif
                    label_source=source,  # Sumber label
                    x_train_data=pickle.dumps(X_train),  # Simpan data latih X
                    x_test_data=pickle.dumps(X_test),  # Simpan data uji X
                    y_train_data=pickle.dumps(y_train),  # Simpan data latih y
                    y_test_data=pickle.dumps(y_test),  # Simpan data uji y
                    preprocessing_ids=json.dumps(all_ids.tolist()) if source == 'otomatis' else None,  # Simpan ID otomatis
                    data_pakar_ids=json.dumps(all_ids.tolist()) if source == 'pakar' else None  # Simpan ID pakar
                )
                db.session.add(new_split)  # Tambah split ke sesi
                success_messages.append(f'Data {source} berhasil dibagi: {len(X_train)} latih, {len(X_test)} uji.')  # Tambah pesan sukses

            if error_messages:  # Tampilkan pesan error
                for msg in error_messages:
                    flash(msg, 'warning')
            
            if success_messages:  # Jika ada keberhasilan
                db.session.commit()  # Simpan perubahan
                for msg in success_messages:
                    flash(msg, 'success')  # Tampilkan pesan sukses
                session.pop('nb_report_id_otomatis', None)  # Hapus ID laporan NB otomatis
                session.pop('nb_report_id_pakar', None)  # Hapus ID laporan NB pakar
                session.pop('svm_report_id_otomatis', None)  # Hapus ID laporan SVM otomatis
                session.pop('svm_report_id_pakar', None)  # Hapus ID laporan SVM pakar
            else:
                db.session.rollback()  # Batalkan perubahan jika gagal

            return redirect(url_for('classification_tasks.select_test_ratio'))  # Redirect ke halaman

        except Exception as e:  # Tangani error
            db.session.rollback()  # Batalkan perubahan
            current_app.logger.error(f"Error during data split: {e}", exc_info=True)  # Log error
            flash(f'Terjadi kesalahan saat membagi data: {str(e)}', 'danger')  # Tampilkan pesan error
            return redirect(url_for('classification_tasks.select_test_ratio'))  # Redirect ke halaman

    # Tangani permintaan GET
    active_splits_list = DataSplit.query.filter_by(is_active=True).all()  # Ambil split aktif
    active_splits = {s.label_source: s for s in active_splits_list}
    return render_template('split_data.html',  # Render template
                           title='Pembagian Data',  # Judul halaman
                           total_labeled=total_labeled,  # Total data berlabel
                           total_unlabeled=total_unlabeled,  # Total data tanpa label
                           total_otomatis=total_otomatis,  # Total label otomatis
                           total_pakar=total_pakar,  # Total label pakar
                           active_splits=active_splits)  # Split aktif

@split_data_bp.route('/reset_all_test_ratios', methods=['POST'])  # Rute untuk reset split
def reset_all_test_ratios():
    """Menghapus semua record pembagian data dari database."""
    try:
        DataSplit.query.delete()  # Hapus semua data split
        db.session.commit()  # Simpan perubahan
        session.pop('active_split_id', None)  # Hapus ID split aktif dari sesi
        session.pop('label_source', None)  # Hapus sumber label dari sesi
        flash('Semua riwayat pembagian data berhasil direset.', 'success')  # Tampilkan pesan sukses
    except Exception as e:  # Tangani error
        db.session.rollback()  # Batalkan perubahan
        current_app.logger.error(f"Error resetting all splits: {str(e)}", exc_info=True)  # Log error
        flash(f'Error saat mereset semua pembagian data: {str(e)}', 'danger')  # Tampilkan pesan error
    
    return redirect(url_for('classification_tasks.select_test_ratio'))  # Redirect ke halaman