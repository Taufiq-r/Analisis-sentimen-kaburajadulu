from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session, make_response, send_file  # Impor modul Flask
from sqlalchemy import or_, text  # Impor fungsi SQLAlchemy
import pandas as pd  # Modul untuk memproses data tabular
import io  # Modul untuk menangani I/O di memori

from .. import db  # Impor instance SQLAlchemy
from ..models import Preprocessing, Dataset  # Impor model database
from ..module.preprocessing import preprocess_workflow  # Impor fungsi preprocessing

preprocessing_bp = Blueprint('preprocessing', __name__)  # Buat blueprint Flask

@preprocessing_bp.route('/', methods=['GET'])  # Rute untuk menampilkan hasil preprocessing
def show_preprocessing_results(): 
    page = int(request.args.get("page", 1))  # Ambil nomor halaman
    search_query = request.args.get("search", "").strip()  # Ambil kueri pencarian
    per_page = int(request.args.get("per_page", current_app.config.get('PER_PAGE', 10)))  # Ambil jumlah data per halaman

    query = Preprocessing.query  # Kueri dasar untuk tabel Preprocessing
    if search_query:  # Jika ada kueri pencarian
        search_term = f"%{search_query}%"  # Format kueri pencarian
        filter_conditions = [
            Preprocessing.full_text.ilike(search_term),
            Preprocessing.text_clean.ilike(search_term),
            Preprocessing.text_stopwords.ilike(search_term),
            Preprocessing.text_stem.ilike(search_term)
        ]  # Kondisi filter pencarian
        if search_query.isdigit():  # Jika kueri adalah angka
            filter_conditions.append(Preprocessing.id == int(search_query))  # Tambah filter ID
        query = query.filter(or_(*filter_conditions))  # Terapkan filter
        
    data_pagination = query.order_by(Preprocessing.id.asc()).paginate(page=page, per_page=per_page, error_out=False)  # Paginate data
    
    total_dataset_source_rows = Dataset.query.count()  # Hitung jumlah data di tabel Dataset

    return render_template('preprocessing.html', 
                         title='Hasil Preprocessing Data',  # Judul halaman
                         data=data_pagination,  # Data pagination
                         total_dataset_source=total_dataset_source_rows,  # Total data sumber
                         search_query=search_query,  # Kueri pencarian
                         per_page=per_page)  # Jumlah data per halaman

@preprocessing_bp.route('/run', methods=['POST'])  # Rute untuk menjalankan preprocessing
def run_preprocessing_pipeline():
    dataset_entries = Dataset.query.all()  # Ambil semua data dari tabel Dataset
    if not dataset_entries:  # Periksa apakah data kosong
        flash("Tidak ada data di tabel Dataset untuk diproses.", "warning")  # Tampilkan peringatan
        return redirect(url_for('preprocessing.show_preprocessing_results'))  # Redirect ke halaman
    
    try:
        slangwords_path = current_app.config['SLANGWORDS_JSON_PATH']  # Ambil path file slangwords
        processed_data_dicts = preprocess_workflow(dataset_entries, slangwords_path)  # Jalankan preprocessing
        
        if processed_data_dicts:  # Jika ada data yang diproses
            for data_dict in processed_data_dicts:  # Iterasi hasil preprocessing
                dataset = Dataset.query.filter_by(full_text=data_dict.get('full_text')).first()  # Cari data di Dataset
                existing = Preprocessing.query.filter_by(
                    full_text=data_dict.get('full_text')
                ).first()  # Cari data di Preprocessing
                
                if existing:  # Jika data sudah ada
                    for k, v in data_dict.items():  # Perbarui atribut
                        if k not in ['label_otomatis', 'id']:  # Kecualikan label_otomatis dan id
                            setattr(existing, k, v)
                    if dataset:  # Jika data Dataset ada
                        existing.dataset_id = dataset.id  # Set dataset_id
                else:  # Jika data baru
                    new_data = {
                        'label_otomatis': None
                    }  # Inisialisasi data baru
                    new_data.update(data_dict)  # Tambah hasil preprocessing
                    if dataset:  # Jika data Dataset ada
                        new_data['dataset_id'] = dataset.id  # Set dataset_id
                    db.session.add(Preprocessing(**new_data))  # Tambah ke sesi
        
        db.session.commit()  # Simpan perubahan
        flash("Preprocessing selesai dan data disimpan.", "success")  # Tampilkan pesan sukses
        
    except Exception as e:  # Tangani error
        db.session.rollback()  # Batalkan perubahan
        flash(f"Error saat menjalankan proses preprocessing: {str(e)}", "danger")  # Tampilkan error
        current_app.logger.error(f"Preprocessing pipeline error: {e}", exc_info=True)  # Log error

    return redirect(url_for('preprocessing.show_preprocessing_results'))  # Redirect ke halaman

@preprocessing_bp.route('/delete_all', methods=['POST'])  # Rute untuk menghapus semua data preprocessing
def delete_all_preprocessing_data():
    try:
        deleted = Preprocessing.query.delete()  # Hapus semua data dari tabel Preprocessing
        db.session.commit()  # Simpan perubahan sementara
        
        try:
            db.session.execute(text('ALTER TABLE preprocessing AUTO_INCREMENT = 1;'))  # Reset auto-increment
        except Exception as e_alter:  # Tangani error reset auto-increment
            current_app.logger.warning(f"Gagal mereset auto_increment: {str(e_alter)}")  # Log peringatan
        
        db.session.commit()  # Simpan perubahan akhir
        
        session.pop('labeling_show_stats', None)  # Hapus status statistik dari sesi
        
        flash(f"Data preprocessing berhasil dihapus ({deleted} baris).", "success")  # Tampilkan pesan sukses
        
    except Exception as e:  # Tangani error
        db.session.rollback()  # Batalkan perubahan
        current_app.logger.error(f"Error saat menghapus data preprocessing: {str(e)}")  # Log error
        flash(f"Error saat menghapus data: {str(e)}", "danger")  # Tampilkan error
    
    return redirect(url_for('preprocessing.show_preprocessing_results'))  # Redirect ke halaman

@preprocessing_bp.route('/download_preprocessing_csv')  # Rute untuk unduh hasil preprocessing sebagai CSV
def download_preprocessing_csv():
    data = Preprocessing.query.order_by(Preprocessing.id.asc()).all()  # Ambil semua data Preprocessing
    if not data:  # Periksa apakah data kosong
        flash('Tidak ada data preprocessing untuk diunduh.', 'warning')  # Tampilkan peringatan
        return redirect(url_for('preprocessing.show_preprocessing_results'))  # Redirect ke halaman
    df = pd.DataFrame([row.__dict__ for row in data])  # Konversi ke DataFrame
    df = df.drop(columns=['_sa_instance_state'], errors='ignore')  # Hapus kolom internal SQLAlchemy
    output = io.StringIO()  # Buat buffer string
    df.to_csv(output, index=False, encoding='utf-8-sig')  # Tulis ke CSV
    output.seek(0)  # Kembali ke awal buffer
    response = make_response(output.getvalue())  # Buat respons
    response.headers["Content-Disposition"] = "attachment; filename=preprocessing_results.csv"  # Set header nama file
    response.headers["Content-type"] = "text/csv"  # Set tipe konten
    return response  # Kembalikan file CSV

@preprocessing_bp.route('/download_preprocessing_excel')  # Rute untuk unduh hasil preprocessing sebagai Excel
def download_preprocessing_excel():
    data = Preprocessing.query.order_by(Preprocessing.id.asc()).all()  # Ambil semua data Preprocessing
    if not data:  # Periksa apakah data kosong
        flash('Tidak ada data preprocessing untuk diunduh.', 'warning')  # Tampilkan peringatan
        return redirect(url_for('preprocessing.show_preprocessing_results'))  # Redirect ke halaman
    df = pd.DataFrame([row.__dict__ for row in data])  # Konversi ke DataFrame
    df = df.drop(columns=['_sa_instance_state'], errors='ignore')  # Hapus kolom internal SQLAlchemy
    output = io.BytesIO()  # Buat buffer bytes
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:  # Tulis ke Excel
        df.to_excel(writer, index=False, sheet_name='Preprocessing')  # Simpan ke sheet
    output.seek(0)  # Kembali ke awal buffer
    response = make_response(output.getvalue())  # Buat respons
    response.headers["Content-Disposition"] = "attachment; filename=preprocessing_results.xlsx"  # Set header nama file
    response.headers["Content-type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # Set tipe konten
    return response  # Kembalikan file Excel