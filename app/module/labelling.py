import csv  # Modul untuk membaca file CSV
from app.models import db, Preprocessing  # Impor database dan model Preprocessing

def load_weighted_lexicon(file_path):
    # Memuat kamus berbobot dari file CSV
    lexicon = {}  # Inisialisasi kamus kosong
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:  # Buka file CSV
            reader = csv.reader(file, delimiter=';')  # Baca dengan pemisah ;
            next(reader, None)  # Lewati header
            for row in reader:  # Iterasi baris CSV
                word = row[0].strip()  # Ambil kata
                weight = int(row[1])  # Ambil bobot sebagai integer
                lexicon[word] = weight  # Tambah ke kamus
    except Exception as e:  # Tangani error
        print(f"Error loading lexicon from {file_path}: {str(e)}")  # Tampilkan pesan error
    return lexicon  # Kembalikan kamus

try:
    positive_lexicon = load_weighted_lexicon("kamus/positive.csv")  # Muat kamus positif
    negative_lexicon = load_weighted_lexicon("kamus/negative.csv")  # Muat kamus negatif
except FileNotFoundError as e:  # Tangani file tidak ditemukan
    print(f"Warning: Lexicon file not found: {e}")  # Tampilkan peringatan
    positive_lexicon = {}  # Inisialisasi kamus positif kosong
    negative_lexicon = {}  # Inisialisasi kamus negatif kosong

def sentiment_analysis_lexicon(text):
    # Analisis sentimen berbasis kamus
    score = 0  # Inisialisasi skor
    text = text.lower().split()  # Konversi ke huruf kecil dan pisah menjadi token
    for word in text:  # Iterasi setiap token
        if word in positive_lexicon:  # Periksa kamus positif
            score += positive_lexicon[word]  # Tambah skor positif
    for word in text:  # Iterasi setiap token
        if word in negative_lexicon:  # Periksa kamus negatif
            score += negative_lexicon[word]  # Tambah skor negatif
    if score > 0:  # Tentukan polaritas berdasarkan skor
        polarity = 'positif'  # Skor positif
    elif score < 0:
        polarity = 'negatif'  # Skor negatif
    else:
        polarity = 'netral'  # Skor nol
    return score, polarity  # Kembalikan skor dan polaritas

def count_labels():
    # Menghitung jumlah data per label
    total_data = Preprocessing.query.count()  # Jumlah total data
    total_positif = Preprocessing.query.filter_by(label_otomatis="positif").count()  # Jumlah label positif
    total_negatif = Preprocessing.query.filter_by(label_otomatis="negatif").count()  # Jumlah label negatif
    total_netral = Preprocessing.query.filter_by(label_otomatis="netral").count()  # Jumlah label netral
    return total_data, total_positif, total_negatif, total_netral  # Kembalikan jumlah

def reset_labels():
    # Mengatur ulang label di database
    entries = Preprocessing.query.all()  # Ambil semua entri
    for entry in entries:  # Iterasi setiap entri
        entry.label = None  # Set label ke None
    db.session.commit()  # Simpan perubahan ke database
    return entries  # Kembalikan entri yang direset