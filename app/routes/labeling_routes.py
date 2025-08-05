import csv  # Modul untuk membaca file CSV
import os  # Modul untuk operasi sistem file
import pandas as pd  # Modul untuk memproses data tabular
from urllib.parse import urlparse, parse_qs, urlencode  # Modul untuk manipulasi URL
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session  # Impor modul Flask
from sqlalchemy import or_, text  # Impor fungsi SQLAlchemy
from werkzeug.utils import secure_filename  # Modul untuk keamanan nama file
from .. import db  # Impor objek database
from ..models import Preprocessing, DataPakar  # Impor model database
from ..module.labelling import load_weighted_lexicon, sentiment_analysis_lexicon, count_labels, reset_labels  # Impor fungsi pelabelan

labeling_bp = Blueprint('labeling', __name__)  # Buat blueprint Flask

def load_sentiment_lexicon(kamus_folder_path):
    # Memuat kamus sentimen positif dan negatif
    pos_lex = load_weighted_lexicon(os.path.join(kamus_folder_path, 'positive.csv'))  # Muat kamus positif
    neg_lex = load_weighted_lexicon(os.path.join(kamus_folder_path, 'negative.csv'))  # Muat kamus negatif
    
    if not pos_lex and not neg_lex:  # Periksa apakah kedua kamus kosong
        current_app.logger.warning(f"Satu atau kedua file kamus sentimen (positive.csv/negative.csv) tidak ditemukan di folder kamus: {kamus_folder_path}")  # Log peringatan
    
    return pos_lex, neg_lex  # Kembalikan kamus positif dan negatif

@labeling_bp.route('/', methods=['GET', 'POST'])  # Rute untuk menampilkan atau melakukan pelabelan
def show_or_perform_label():
    kamus_folder = current_app.config['KAMUS_FOLDER_PATH']  # Ambil path folder kamus
    pos_lex, neg_lex = load_sentiment_lexicon(kamus_folder)  # Muat kamus sentimen

    page = int(request.args.get("page", 1))  # Ambil nomor halaman
    search_query = request.args.get("search", "")  # Ambil kueri pencarian
    per_page = int(request.args.get("per_page", current_app.config['PER_PAGE']))  # Ambil jumlah data per halaman
    
    current_show_stats = request.args.get('show_stats', session.get('label_show_stats', '0'))  # Status tampilan statistik
    session['label_show_stats'] = current_show_stats  # Simpan status di sesi

    base_query_for_data = Preprocessing.query.filter(
        Preprocessing.text_stem.isnot(None),
        Preprocessing.text_stem != ''
    )  # Kueri dasar untuk data dengan teks stemmed
    
    if request.method == 'POST':  # Tangani permintaan POST
        data_to_label_auto = base_query_for_data.filter(
            Preprocessing.label_otomatis == None
        ).all()  # Ambil data tanpa label otomatis

        if not data_to_label_auto:  # Periksa apakah tidak ada data untuk dilabeli
            flash("Tidak ada data hasil preprocessing yang siap untuk dilabeli (atau semua sudah memiliki label otomatis).", "info")  # Tampilkan info
        elif not pos_lex and not neg_lex:  # Periksa apakah kamus kosong
            flash("Kamus sentimen kosong atau tidak dapat dimuat. Pelabelan otomatis tidak dapat dilakukan.", "warning")  # Tampilkan peringatan
        else:
            updated_count = 0  # Inisialisasi jumlah data yang diperbarui
            try:
                for row_obj in data_to_label_auto:  # Iterasi data tanpa label
                    score, label_lex = sentiment_analysis_lexicon(row_obj.text_stem)  # Analisis sentimen
                    if label_lex:  # Jika label valid
                        row_obj.label_otomatis = label_lex  # Set label otomatis
                        updated_count += 1  # Tambah hitungan
                if updated_count > 0:  # Jika ada data yang diperbarui
                    db.session.commit()  # Simpan perubahan
                    flash(f"{updated_count} data berhasil dilabeli secara otomatis (data yang belum memiliki label telah diisi).", "success")  # Tampilkan sukses
                else:
                    flash("Tidak ada label otomatis yang kosong yang berhasil diisi oleh lexicon, atau lexicon tidak menghasilkan label baru untuk data yang ada.", "info")  # Tampilkan info
            except Exception as e:  # Tangani error
                db.session.rollback()  # Batalkan perubahan
                flash(f"Error saat pelabelan otomatis: {str(e)}", "danger")  # Tampilkan error
                current_app.logger.error(f"Auto-label error: {e}", exc_info=True)  # Log error
        session['label_show_stats'] = '1'  # Tampilkan statistik setelah pelabelan
        session['lexicon_stats_ready'] = True  # Tandai statistik siap
        return redirect(url_for('labeling.show_or_perform_label', show_stats='1', page=1, per_page=per_page))  # Redirect ke halaman
    
    total_data_for_labeling = base_query_for_data.count()  # Hitung total data untuk pelabelan

    total_data, total_positif_otomatis, total_negatif_otomatis, total_netral_otomatis = count_labels()  # Hitung statistik label otomatis

    total_positif_pakar = DataPakar.query.filter(DataPakar.label == 'positif').count()  # Hitung label positif pakar
    total_negatif_pakar = DataPakar.query.filter(DataPakar.label == 'negatif').count()  # Hitung label negatif pakar
    total_netral_pakar = DataPakar.query.filter(DataPakar.label == 'netral').count()  # Hitung label netral pakar
    total_pakar_data = DataPakar.query.filter(DataPakar.label.isnot(None)).count()  # Hitung total data pakar berlabel

    total_positif_lexicon = total_positif_otomatis  # Set statistik lexicon positif
    total_negatif_lexicon = total_negatif_otomatis  # Set statistik lexicon negatif
    total_netral_lexicon = total_netral_otomatis  # Set statistik lexicon netral

    total_unlabeled = base_query_for_data.filter(Preprocessing.label_otomatis.is_(None)).count()  # Hitung data tanpa label

    if search_query:  # Jika ada kueri pencarian
        search_term = f"%{search_query}%"  # Format kueri pencarian
        filter_conditions = [
            Preprocessing.full_text.ilike(search_term),
            Preprocessing.text_clean.ilike(search_term),
            Preprocessing.text_baku.ilike(search_term),
            Preprocessing.text_stopwords.ilike(search_term),
            Preprocessing.text_stem.ilike(search_term),
            Preprocessing.label_otomatis.ilike(search_term)  # Pencarian berdasarkan label otomatis
        ]
        if search_query.isdigit():  # Jika kueri adalah angka
            filter_conditions.append(Preprocessing.id == int(search_query))  # Tambah filter ID
        base_query_for_data = base_query_for_data.filter(or_(*filter_conditions))  # Terapkan filter pencarian
            
    data_pagination = base_query_for_data.order_by(Preprocessing.id.asc()).paginate(page=page, per_page=per_page, error_out=False)  # Paginate data

    enhanced_items = []  # Inisialisasi list untuk data yang diperkaya
    data_pakar_map = { (dp.full_text or '').strip().casefold(): dp.label for dp in DataPakar.query.all() }  # Buat kamus label pakar
    
    for item in data_pagination.items:  # Iterasi item pagination
        score, polarity = sentiment_analysis_lexicon(item.text_stem)  # Analisis sentimen
        label_pakar_for_display = data_pakar_map.get((item.full_text or '').strip().casefold())  # Ambil label pakar
        enhanced_item = {
            'id': item.id,
            'full_text': item.full_text,
            'text_clean': item.text_clean,
            'text_stem': item.text_stem,
            'label_otomatis': item.label_otomatis,
            'label_pakar': label_pakar_for_display,  # Label pakar untuk tampilan
            'polarity': polarity,
        }  # Buat item yang diperkaya
        enhanced_items.append(enhanced_item)  # Tambah ke list
    
    return render_template('label.html',  # Render template
                       title='Label',  # Judul halaman
                       results_for_table=enhanced_items,  # Data untuk tabel
                       total_data_for_labeling=total_data_for_labeling,  # Total data untuk pelabelan
                       total_unlabeled=total_unlabeled,  # Total data tanpa label
                       total_positif_pakar=total_positif_pakar,  # Total label positif pakar
                       total_negatif_pakar=total_negatif_pakar,  # Total label negatif pakar
                       total_netral_pakar=total_netral_pakar,  # Total label netral pakar
                       total_pakar_data=total_pakar_data,  # Total data pakar
                       total_positif_otomatis=total_positif_otomatis,  # Total label positif otomatis
                       total_negatif_otomatis=total_negatif_otomatis,  # Total label negatif otomatis
                       total_netral_otomatis=total_netral_otomatis,  # Total label netral otomatis
                       total_positif_lexicon=total_positif_lexicon,  # Total positif lexicon
                       total_negatif_lexicon=total_negatif_lexicon,  # Total negatif lexicon
                       total_netral_lexicon=total_netral_lexicon,  # Total netral lexicon
                       data_pagination=data_pagination,  # Data pagination
                       search_query=search_query,  # Kueri pencarian
                       per_page=per_page,  # Jumlah data per halaman
                       show_stats=current_show_stats)  # Status tampilan statistik

@labeling_bp.route('/upload_expert_labels', methods=['POST'])  # Rute untuk unggah label pakar
def upload_expert_labels():
    if 'file' not in request.files:  # Periksa apakah file ada
        flash('Tidak ada file yang dipilih', 'danger')  # Tampilkan error
        return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman
    
    file = request.files['file']  # Ambil file dari permintaan
    if file.filename == '':  # Periksa apakah nama file kosong
        flash('Tidak ada file yang dipilih', 'danger')  # Tampilkan error
        return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman
    
    ALLOWED_EXTENSIONS = {'csv'}  # Ekstensi file yang diizinkan
    def allowed_file(filename):  # Fungsi untuk validasi ekstensi
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS  # Periksa ekstensi
    
    if not allowed_file(file.filename):  # Validasi format file
        flash('Format file tidak diizinkan. Gunakan CSV ', 'danger')  # Tampilkan error
        return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman
    
    try:
        if file.filename.endswith('.csv'):  # Jika file CSV
            df = pd.read_csv(file, encoding='utf-8')  # Baca file CSV
        else:
            df = pd.read_excel(file)  # Baca file Excel (tidak digunakan karena hanya CSV diizinkan)
        
        required_columns = ['id', 'username', 'full_text', 'text_clean', 'text_baku', 'text_stopwords', 'text_stem', 'created_at', 'label']  # Kolom wajib
        missing_columns = [col for col in required_columns if col not in df.columns]  # Periksa kolom yang hilang
        if missing_columns:  # Jika ada kolom yang hilang
            flash(f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_columns)}", 'danger')  # Tampilkan error
            return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman
        
        df['label'] = df['label'].str.lower()  # Konversi label ke huruf kecil
        valid_labels = ['positif', 'negatif', 'netral']  # Label yang valid
        invalid_labels = df[~df['label'].isin(valid_labels)]['label'].unique()  # Periksa label tidak valid
        if len(invalid_labels) > 0:  # Jika ada label tidak valid
            flash(f"Ditemukan label tidak valid: {', '.join(invalid_labels)}. Gunakan: positif, negatif, atau netral", 'danger')  # Tampilkan error
            return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman

        df_with_labels = df[df['label'].notna() & (df['label'] != '')]  # Filter data dengan label valid
        
        if len(df_with_labels) == 0:  # Periksa apakah tidak ada data valid
            flash('Tidak ada data dengan label yang valid ditemukan dalam file', 'warning')  # Tampilkan peringatan
            return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman

        DataPakar.query.delete()  # Hapus semua data pakar
        db.session.commit()  # Simpan perubahan
        
        engine = db.session.bind or db.engine  # Ambil engine database
        engine_name = engine.dialect.name  # Ambil nama engine
        if engine_name in ('mysql', 'mariadb'):  # Jika engine MySQL/MariaDB
            db.session.execute(text("ALTER TABLE data_pakar AUTO_INCREMENT = 1;"))  # Reset auto-increment
            db.session.commit()  # Simpan perubahan

        success_count = 0  # Inisialisasi jumlah data sukses
        skipped_count = 0  # Inisialisasi jumlah data yang dilewati
        
        for _, row in df_with_labels.iterrows():  # Iterasi baris data
            if pd.isna(row['username']) or pd.isna(row['full_text']) or pd.isna(row['created_at']):  # Periksa field wajib
                skipped_count += 1  # Tambah hitungan data yang dilewati
                continue
                
            data_pakar = DataPakar(  # Buat entri DataPakar baru
                username=row['username'],
                full_text=row['full_text'],
                created_at=row['created_at'],
                label=row['label'],
                text_clean=row['text_clean'] if pd.notna(row['text_clean']) else '',
                text_stem=row['text_stem'] if pd.notna(row['text_stem']) else '',
                text_baku=row['text_baku'] if pd.notna(row['text_baku']) else '',
                text_stopwords=row['text_stopwords'] if pd.notna(row['text_stopwords']) else ''
            )
            db.session.add(data_pakar)  # Tambah ke sesi
            success_count += 1  # Tambah hitungan sukses

        db.session.commit()  # Simpan perubahan
        
        message = f'Berhasil mengunggah {success_count} data pakar ke database.'  # Pesan sukses
        if skipped_count > 0:  # Jika ada data yang dilewati
            message += f' {skipped_count} data dilewati karena field penting kosong.'  # Tambah info
        
        flash(message, 'success')  # Tampilkan pesan sukses
        
    except Exception as e:  # Tangani error
        db.session.rollback()  # Batalkan perubahan
        current_app.logger.error(f"Upload error: {str(e)}", exc_info=True)  # Log error
        flash(f"Terjadi kesalahan saat mengupload file: {str(e)}", 'danger')  # Tampilkan error
    
    return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman

@labeling_bp.route('/reset_all_labels', methods=['POST'])  # Rute untuk reset label otomatis
def reset_all_otomatis_labels():  # Nama fungsi bisa membingungkan, lebih baik 'reset_all_labels_from_db'
    try:
        updated = Preprocessing.query.update({Preprocessing.label_otomatis: None})  # Reset label otomatis
        db.session.commit()  # Simpan perubahan
        flash(f"Berhasil mereset {updated} label otomatis.", "success")  # Tampilkan pesan sukses
    except Exception as e:  # Tangani error
        db.session.rollback()  # Batalkan perubahan
        flash(f"Error saat mereset label: {str(e)}", "danger")  # Tampilkan error
    return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman

@labeling_bp.route('/reset_data_pakar', methods=['POST'])  # Rute untuk reset data pakar
def reset_data_pakar():
    try:
        deleted = DataPakar.query.delete()  # Hapus semua data pakar
        db.session.commit()  # Simpan perubahan
        flash(f"Seluruh data pakar berhasil dihapus ({deleted} baris) dari tabel DataPakar.", "info")  # Tampilkan info
    except Exception as e:  # Tangani error
        db.session.rollback()  # Batalkan perubahan
        flash(f"Error saat mereset data pakar: {str(e)}", "danger")  # Tampilkan error
        current_app.logger.error(f"Reset data pakar error: {e}", exc_info=True)  # Log error
    return redirect(url_for('labeling.show_or_perform_label'))  # Redirect ke halaman

@labeling_bp.route('/edit_manual_label', methods=['POST'])  # Rute untuk edit label manual
def edit_manual_label():
    row_id = request.form.get('row_id')  # Ambil ID baris
    new_label = request.form.get('new_label')  # Ambil label baru
    
    ref = request.referrer or url_for('labeling.show_or_perform_label')  # Ambil URL referrer
    parsed_url = urlparse(ref)  # Parse URL
    query_params_dict = parse_qs(parsed_url.query)  # Ambil parameter kueri
    
    redirect_args = {
        'page': query_params_dict.get('page', ['1'])[0],  # Ambil halaman
        'per_page': query_params_dict.get('per_page', [str(current_app.config['PER_PAGE'])])[0],  # Ambil jumlah per halaman
        'search': query_params_dict.get('search', [''])[0],  # Ambil kueri pencarian
        'show_stats': query_params_dict.get('show_stats', [session.get('label_show_stats', '0')])[0]  # Ambil status statistik
    }

    if row_id and row_id.isdigit():  # Validasi ID
        row_to_edit = Preprocessing.query.get(int(row_id))  # Ambil data berdasarkan ID
        if row_to_edit:  # Periksa apakah data ada
            try:
                row_to_edit.label_otomatis = new_label if new_label else None  # Set label baru
                db.session.commit()  # Simpan perubahan
                flash(f"Label untuk ID {row_id} berhasil diubah menjadi '{new_label if new_label else 'Kosong'}'.", "success")  # Tampilkan sukses
            except Exception as e:  # Tangani error
                db.session.rollback()  # Batalkan perubahan
                flash(f"Error saat mengubah label: {str(e)}", "danger")  # Tampilkan error
        else:
            flash(f"Data dengan ID {row_id} tidak ditemukan.", "error")  # Tampilkan error
    else:
        flash("ID data tidak valid atau label baru tidak disediakan.", "error")  # Tampilkan error

    cleaned_redirect_args = {k: v for k, v in redirect_args.items() if v}  # Bersihkan argumen redirect
    return redirect(url_for('labeling.show_or_perform_label', **cleaned_redirect_args))  # Redirect ke halaman

