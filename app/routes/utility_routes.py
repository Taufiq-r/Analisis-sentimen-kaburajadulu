from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app, session  # Impor modul Flask
import os  # Modul untuk operasi sistem file
import pickle  # Modul untuk serialisasi data
import glob  # Modul untuk pencarian file
from wordcloud import WordCloud, STOPWORDS  # Modul untuk membuat wordcloud
import numpy as np  # Modul untuk operasi numerik

from app.models import KlasifikasiNB, KlasifikasiSVM  # Impor model database
from app.module.tfidf_vectorizer import CustomTfidf  # Impor vectorizer TF-IDF

utility_bp = Blueprint('utils', __name__, template_folder='../templates')  # Buat blueprint Flask

def calculate_label_counts_from_db(model_name_for_chart):
    # Menghitung jumlah label dari database
    if model_name_for_chart == 'Naive Bayes':  # Jika model Naive Bayes
        classification_data = KlasifikasiNB.query.all()  # Ambil semua data NB
    elif model_name_for_chart == 'SVM':  # Jika model SVM
        classification_data = KlasifikasiSVM.query.all()  # Ambil semua data SVM
    else:
        classification_data = []  # Data kosong jika model tidak valid
    
    counts = {'positif': 0, 'negatif': 0, 'netral': 0}  # Inisialisasi counter label
    for item in classification_data:  # Iterasi data klasifikasi
        label = item.label_prediksi  # Ambil label prediksi
        if label in counts: 
            counts[label] += 1  # Tambah hitungan
    return [counts['positif'], counts['negatif'], counts['netral']]  # Kembalikan list hitungan

@utility_bp.route('/visualization', methods=['GET', 'POST'])  # Rute untuk visualisasi sentimen
def sentiment_visualization():
    static_images_folder = os.path.join(current_app.static_folder, 'images')  # Path folder gambar statis
    os.makedirs(static_images_folder, exist_ok=True)  # Buat folder jika belum ada

    if request.method == 'POST':  # Tangani permintaan POST
        action = request.form.get('action')  # Ambil aksi dari form
        if action == 'reset_visuals':  # Jika aksi reset visualisasi
            files_removed_count = 0  # Inisialisasi jumlah file yang dihapus
            for f_path in glob.glob(os.path.join(static_images_folder, 'wordcloud_*.png')):  # Cari file wordcloud
                if os.path.isfile(f_path):  # Periksa apakah file ada
                    try:
                        os.remove(f_path)  # Hapus file
                        files_removed_count += 1  # Tambah hitungan
                    except Exception as e:  # Tangani error
                        current_app.logger.error(f"Error removing wordcloud file {f_path}: {e}")  # Log error
            
            for model in ['nb', 'svm']:  # Iterasi model
                for source in ['otomatis', 'pakar']:  # Iterasi sumber label
                    session.pop(f'{model}_{source}_label_counts_js', None)  # Hapus hitungan label dari sesi
                    session.pop(f'{model}_{source}_chart_generated', None)  # Hapus status chart dari sesi

            flash('Semua visualisasi berhasil direset.', "info")  # Tampilkan pesan info
            return redirect(url_for('utils.sentiment_visualization'))  # Redirect ke halaman

        elif action == 'generate_visuals':  # Jika aksi generate visualisasi
            combinations = [
                ('Naive Bayes', 'otomatis', KlasifikasiNB),
                ('Naive Bayes', 'pakar', KlasifikasiNB),
                ('SVM', 'otomatis', KlasifikasiSVM),
                ('SVM', 'pakar', KlasifikasiSVM)
            ]  # Kombinasi model dan sumber
            generation_messages = []  # List untuk pesan generasi
            any_visuals_generated = False  # Status visualisasi dibuat

            for model_name, source, model_class in combinations:  # Iterasi kombinasi
                if source == 'pakar':  # Jika sumber pakar
                    results = model_class.query.filter(model_class.data_pakar_id.isnot(None)).all()  # Ambil data pakar
                else:  # Jika sumber otomatis
                    results = model_class.query.filter(model_class.preprocessing_id.isnot(None)).all()  # Ambil data otomatis

                if not results:  # Periksa apakah data kosong
                    generation_messages.append(f'Tidak ada data klasifikasi untuk {model_name} ({source}).')  # Tambah pesan
                    continue

                texts_by_label = {'positif': '', 'negatif': '', 'netral': ''}  # Inisialisasi teks per label
                for item in results:  # Iterasi hasil
                    text = item.full_text or item.text_stem or ''  # Ambil teks
                    if item.label_prediksi in texts_by_label:  # Jika label valid
                        texts_by_label[item.label_prediksi] += ' ' + text  # Tambah teks ke label
                
                custom_stopwords = set(STOPWORDS).union(set(current_app.config.get('CUSTOM_STOPWORDS', [])))  # Gabung stopwords
                
                for label, text_content in texts_by_label.items():  # Iterasi teks per label
                    if text_content.strip():  # Periksa apakah teks tidak kosong
                        try:
                            wc = WordCloud(width=450, height=250, background_color='#1e1e1e', colormap='Pastel1',
                                           stopwords=custom_stopwords, collocations=False).generate(text_content)  # Buat wordcloud
                            
                            model_key_for_file = 'nb' if model_name == 'Naive Bayes' else 'svm'  # Kunci model untuk file
                            img_filename = f'wordcloud_{model_key_for_file}_{source}_{label}.png'  # Nama file wordcloud
                            wc.to_file(os.path.join(static_images_folder, img_filename))  # Simpan wordcloud
                            any_visuals_generated = True  # Tandai visualisasi dibuat
                        except Exception as e:  # Tangani error
                            current_app.logger.error(f"Gagal membuat wordcloud {model_name}/{source}/{label}: {e}")  # Log error

                counts = {'positif': 0, 'negatif': 0, 'netral': 0}  # Inisialisasi counter label
                for item in results:  # Iterasi hasil
                    if item.label_prediksi in counts:  # Jika label valid
                        counts[item.label_prediksi] += 1  # Tambah hitungan
                
                label_counts_list = [counts['positif'], counts['negatif'], counts['netral']]  # List hitungan label
                
                model_key_for_session = 'nb' if model_name == 'Naive Bayes' else 'svm'  # Kunci model untuk sesi
                session_key_counts = f'{model_key_for_session}_{source}_label_counts_js'  # Kunci sesi untuk hitungan
                session_key_generated = f'{model_key_for_session}_{source}_chart_generated'  # Kunci sesi untuk status
                session[session_key_counts] = label_counts_list  # Simpan hitungan ke sesi
                session[session_key_generated] = sum(label_counts_list) > 0  # Simpan status chart
                any_visuals_generated = True  # Tandai visualisasi dibuat

            if any_visuals_generated:  # Jika ada visualisasi yang dibuat
                flash("Visualisasi telah berhasil dibuat/diperbarui.", "success")  # Tampilkan pesan sukses
            else:
                flash("Tidak ada data yang cukup untuk membuat visualisasi.", "warning")  # Tampilkan peringatan
            
            return redirect(url_for('utils.sentiment_visualization'))  # Redirect ke halaman

    contexts = {}  # Inisialisasi konteks visualisasi
    can_reset_any_visuals = False  # Status apakah visualisasi bisa direset

    for model_name in ['Naive Bayes', 'SVM']:  # Iterasi model
        for source in ['otomatis', 'pakar']:  # Iterasi sumber
            model_key = 'nb' if model_name == 'Naive Bayes' else 'svm'  # Kunci model
            context_key = f"{model_key}_{source}"  # Kunci konteks
            
            chart_counts = session.get(f'{context_key}_label_counts_js', [0,0,0])  # Ambil hitungan dari sesi
            chart_generated = session.get(f'{context_key}_chart_generated', False)  # Ambil status chart

            wc_paths = {}  # Inisialisasi path wordcloud
            has_wordclouds = False  # Status keberadaan wordcloud
            for label in ['positif', 'negatif', 'netral']:  # Iterasi label
                filename = f'wordcloud_{model_key}_{source}_{label}.png'  # Nama file wordcloud
                relative_path = os.path.join('images', filename).replace('\\', '/')  # Path relatif
                if os.path.exists(os.path.join(current_app.static_folder, relative_path)):  # Periksa keberadaan file
                    wc_paths[label] = relative_path  # Simpan path
                    has_wordclouds = True  # Tandai wordcloud ada
            
            contexts[context_key] = {
                'has_data': chart_generated,  # Status data chart
                'label_counts_js': chart_counts,  # Hitungan label
                'has_wordclouds': has_wordclouds,  # Status wordcloud
                'wordclouds_relative_paths': wc_paths  # Path wordcloud
            }
            if chart_generated or has_wordclouds:  # Jika ada chart atau wordcloud
                can_reset_any_visuals = True  # Tandai bisa reset

    return render_template('visualization.html',
                            title='Visualisasi Sentimen',  # Judul halaman
                            visuals=contexts,  # Data visualisasi
                            can_reset=can_reset_any_visuals)  # Status reset

@utility_bp.route('/word_prediction', methods=['GET', 'POST'])  # Rute untuk prediksi sentimen teks
def word_sentiment_prediction():
    input_text = ''  # Inisialisasi teks input
    selected_model_choice = 'Naive Bayes'  # Default model
    prediction = None  # Inisialisasi prediksi
    prediction_score = None  # Inisialisasi skor prediksi
    error = None  # Inisialisasi error
    if request.method == 'POST':  # Tangani permintaan POST
        input_text = request.form.get('input_text', '')  # Ambil teks input
        selected_model_choice = request.form.get('model_choice', 'Naive Bayes')  # Ambil pilihan model
        if not input_text.strip():  # Periksa apakah teks kosong
            error = 'Teks tidak boleh kosong.'  # Set error
        else:
            try:
                model_folder = os.path.join(current_app.root_path, '..', 'model')  # Path folder model
                vectorizer_path = os.path.join(model_folder, 'tfidf_vectorizer.pkl')  # Path vectorizer

                if selected_model_choice == 'Naive Bayes':  # Jika model Naive Bayes
                    model_path = os.path.join(model_folder, 'nb', 'naive_bayes_model.pkl')  # Path model NB
                    
                    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):  # Periksa keberadaan file
                        error = 'Model Naive Bayes atau Vectorizer bersama belum tersedia. Silakan lakukan klasifikasi terlebih dahulu.'  # Set error
                    else:
                        with open(model_path, 'rb') as f_model:  # Buka model NB
                            nb_model = pickle.load(f_model)  # Muat model
                        with open(vectorizer_path, 'rb') as f_vec:  # Buka vectorizer
                            vectorizer = pickle.load(f_vec)  # Muat vectorizer
                        
                        X_input_tfidf = vectorizer.transform([input_text])  # Transformasi teks ke TF-IDF
                        
                        if hasattr(nb_model, 'predict_with_score'):  # Jika model punya predict_with_score
                            label_score_list = nb_model.predict_with_score(X_input_tfidf)  # Prediksi dengan skor
                            if label_score_list:  # Jika ada hasil
                                prediction = label_score_list[0][0]  # Ambil label
                                prediction_score = label_score_list[0][1]  # Ambil skor
                            else:
                                prediction = None  # Set prediksi kosong
                                prediction_score = None  # Set skor kosong
                        else:
                            prediction = nb_model.predict(X_input_tfidf)[0]  # Prediksi label
                            if hasattr(nb_model, 'predict_proba'):  # Jika model punya predict_proba
                                proba = nb_model.predict_proba(X_input_tfidf)[0]  # Ambil probabilitas
                                class_idx = list(nb_model.classes_).index(prediction)  # Ambil indeks kelas
                                prediction_score = proba[class_idx]  # Ambil skor
                            else:
                                prediction_score = 'N/A'  # Skor tidak tersedia

                elif selected_model_choice == 'SVM':  # Jika model SVM
                    model_path = os.path.join(model_folder, 'svm', 'svm_model.pkl')  # Path model SVM

                    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):  # Periksa keberadaan file
                        error = 'Model SVM atau Vectorizer bersama belum tersedia. Silakan lakukan klasifikasi terlebih dahulu.'  # Set error
                    else:
                        with open(model_path, 'rb') as f_model:  # Buka model SVM
                            svm_model = pickle.load(f_model)  # Muat model
                        with open(vectorizer_path, 'rb') as f_vec:  # Buka vectorizer
                            vectorizer = pickle.load(f_vec)  # Muat vectorizer
                        
                        X_input_tfidf = vectorizer.transform([input_text])  # Transformasi teks ke TF-IDF
                        
                        if hasattr(svm_model, 'predict_proba'):  # Jika model punya predict_proba
                            prediction_proba = svm_model.predict_proba(X_input_tfidf)[0]  # Ambil probabilitas
                            predicted_class_index = np.argmax(prediction_proba)  # Ambil indeks kelas
                            prediction = svm_model.classes_[predicted_class_index]  # Ambil label
                            prediction_score = prediction_proba[predicted_class_index]  # Ambil skor
                        else:
                            prediction = svm_model.predict(X_input_tfidf)[0]  # Prediksi label
                            prediction_score = 'N/A'  # Skor tidak tersedia

            except ValueError as ve:  # Tangani error nilai
                error = f"Terjadi kesalahan saat prediksi: {ve}"  # Set error
                current_app.logger.error(f"Prediction ValueError: {ve}", exc_info=True)  # Log error
            except Exception as e:  # Tangani error lain
                error = f"Terjadi kesalahan tak terduga: {e}"  # Set error
                current_app.logger.error(f"Prediction error: {e}", exc_info=True)  # Log error

    return render_template('word_prediction.html', 
                            title='Uji Model',  # Judul halaman
                            input_text=input_text,  # Teks input
                            selected_model_choice=selected_model_choice,  # Pilihan model
                            prediction=prediction,  # Hasil prediksi
                            prediction_score=prediction_score,  # Skor prediksi
                            error=error)  # Pesan error