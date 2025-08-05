from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
import os
from werkzeug.utils import secure_filename
import pandas as pd
from sqlalchemy import or_, text
from .. import db # Impor db dari app.__init__
from ..models import Dataset # Impor model dari app.models

dataset_bp = Blueprint('dataset', __name__)

@dataset_bp.route('/', methods=['GET', 'POST']) 
def input_data():
    if request.method == 'POST':
        current_app.logger.info("POST request received for dataset upload")
        file = request.files.get('file')
        current_app.logger.info(f"File received: {file.filename if file else 'No file'}")
        if file and file.filename and file.filename.endswith('.csv'):
            try:                # Langsung baca file ke DataFrame tanpa simpan ke disk
                current_app.logger.info("Membaca file CSV...")
                df = pd.read_csv(file)
                current_app.logger.info(f"File CSV dibaca. Jumlah baris: {len(df)}")
                current_app.logger.info(f"Kolom yang tersedia: {df.columns.tolist()}")
                
                # Hapus duplikat berdasarkan full_text jika kolom tersebut ada
                if 'full_text' in df.columns:
                    before_drop = len(df)
                    df = df.drop_duplicates(subset=['full_text'])
                    after_drop = len(df)
                    current_app.logger.info(f"Menghapus duplikat: {before_drop - after_drop} baris dihapus")
                else:
                    current_app.logger.warning("Kolom 'full_text' tidak ditemukan di CSV")
                    flash("Peringatan: Kolom 'full_text' tidak ditemukan di CSV untuk penghapusan duplikat.", "warning")

                # Hapus semua data lama dari tabel Dataset sebelum memasukkan yang baru
                db.session.query(Dataset).delete()
                # Reset auto_increment (MySQL specific)
                db.session.execute(text('ALTER TABLE dataset AUTO_INCREMENT = 1'))

                entries = []
                for _, row in df.iterrows():
                    entry = Dataset(
                        username=row.get('username'),
                        full_text=row.get('full_text'),
                        created_at=str(row.get('created_at', ''))
                    )
                    entries.append(entry)
                if entries:
                    current_app.logger.info(f"Menyimpan {len(entries)} entries ke database...")
                    try:
                        db.session.bulk_save_objects(entries)
                        db.session.commit()
                        current_app.logger.info("Data berhasil disimpan ke database")
                        flash(f'{len(entries)} baris data berhasil dimasukkan ke database.', 'info')
                    except Exception as db_error:
                        db.session.rollback()
                        current_app.logger.error(f"Error saat menyimpan ke database: {str(db_error)}")
                        flash(f'Error saat menyimpan ke database: {str(db_error)}', 'danger')
                else:
                    current_app.logger.warning("Tidak ada data valid untuk dimasukkan")
                    flash('Tidak ada data valid untuk dimasukkan dari file CSV.', 'info')

            except Exception as e:
                db.session.rollback()
                flash(f'Error saat memproses file: {str(e)}', 'danger')
        elif file and file.filename:
            flash('Format file tidak valid. Harus .csv', 'danger')
        else:
            flash('Tidak ada file yang dipilih untuk diunggah.', 'warning')
        return redirect(url_for('dataset.input_data'))

    page = int(request.args.get("page", 1))
    search_query = request.args.get("search", "")
    per_page = int(request.args.get("per_page", current_app.config['PER_PAGE']))
    
    query = Dataset.query
    if search_query:
        search_term = f"%{search_query}%"
        filter_conditions = [
            Dataset.full_text.ilike(search_term),
            Dataset.username.ilike(search_term),
            Dataset.created_at.ilike(search_term)
        ]
        if search_query.isdigit():
            filter_conditions.append(Dataset.id == int(search_query))
        query = query.filter(or_(*filter_conditions))
        
    data_pagination = query.order_by(Dataset.id.asc()).paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('input_data.html',
                           title='Dataset',
                           data=data_pagination, # Objek pagination
                           search_query=search_query,
                           per_page=per_page)

@dataset_bp.route('/delete_all_dataset_entries', methods=['POST']) # Menjadi /dataset/delete_all_db_and_file
def delete_all_dataset_entries():
    
   
    
    try:
        num_deleted_db = db.session.query(Dataset).delete()
        db.session.commit() # Commit delete DB
        if num_deleted_db > 0: # Hanya reset jika ada data yang dihapus
            db.session.execute(text('ALTER TABLE dataset AUTO_INCREMENT = 1')) # MySQL specific
            db.session.commit() # Commit alter table
      
    except Exception as e:
        db.session.rollback()
        flash(f"Error saat menghapus dataset: {str(e)}", "danger")
    return redirect(url_for('dataset.input_data'))
