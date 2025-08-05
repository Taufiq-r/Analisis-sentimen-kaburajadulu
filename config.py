import os

# Path dasar proyek (root folder /Tugas Akhir 1.3 NB + svm/)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    # GANTI ' ' 
    # Untuk production, ini HARUS di-override oleh environment variable.
    SECRET_KEY = os.getenv('SECRET_KEY', ' ') 
    
    SQLALCHEMY_DATABASE_URI = os.getenv(
        'DATABASE_URL', 
        'mysql+mysqlconnector://root@localhost/analisissentimenkad' # Pastikan DB ini ada dan user/pass sesuai
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- TF-IDF Vectorizer Parameters ---
    # Mengatur jumlah fitur (kata unik) maksimum yang akan dibuat.
    TFIDF_MAX_FEATURES = 2000
    # Mengabaikan kata yang muncul di kurang dari X dokumen.
    TFIDF_MIN_DF = 3
    # Mengabaikan kata yang muncul di lebih dari X% dokumen (mengatasi kata umum).
    TFIDF_MAX_DF_RATIO = 0.85
    # Menggunakan unigram, bigram, dan trigram untuk menangkap frasa yang lebih panjang.
    TFIDF_NGRAM_RANGE = (1, 2)
    # Definisikan nama folder relatif terhadap BASE_DIR (ada di root proyek)
   
    MODEL_FOLDER_NAME = 'model'
    KAMUS_FOLDER_NAME = 'kamus'
    # STATIC_FOLDER_NAME digunakan oleh Flask untuk menunjuk ke app.static_folder
    # yang mana adalah 'app/static' dalam struktur baru kita
    
    # Path absolut ke folder-folder tersebut (ada di root proyek)
   
    MODEL_FOLDER_PATH = os.path.join(BASE_DIR, MODEL_FOLDER_NAME)
    KAMUS_FOLDER_PATH = os.path.join(BASE_DIR, KAMUS_FOLDER_NAME)
    
    # Untuk file slangwords.json di dalam app/module/
    # BASE_DIR menunjuk ke root, jadi pathnya adalah BASE_DIR + app + module + namafile
    SLANGWORDS_JSON_PATH = os.path.join(BASE_DIR, 'app', 'module', 'slangwords.json') # Pastikan nama file ini benar
    PREPROCESSING_JSON_PATH = SLANGWORDS_JSON_PATH

    # Default items per page untuk pagination
    PER_PAGE = 10

    # Contoh custom stopwords untuk WordCloud, bisa Anda perluas
    CUSTOM_STOPWORDS = [
        'https', 'co', 'lu', 'di', 'ya',
        'itu', 'bro', 'yg', 'ga', 'gak', 'nya', 'aja', 'sih', 'gue', 'gw', 'gua', 'saya', 'ini',
        'sangat', 'sekali', 'yang', 'dan', 'dari', 'ke', 'tidak',
        'tapi', 'ada', 'untuk', 'dengan', 'juga', 'sudah',
        'belum', 'akan', 'lagi', 'bisa', 'harus', 'karena', 'seperti',
        'hanya', 'atau', 'banget', 'nih', 'kok', 'biar', 'iya', 'dong', 'kak', 'kakak', 'jgn', 'emg',
        'kalo', 'pas', 'bgt', 'tdk', 'tp', 'sy', 'trs', 'jd', 'utk', 'kl', 'org', 'org2', 'sm', 'kyk',
        'dll', 'dsb', 'dst', 'dgn', 'gk', 'gaes', 'guys', 'jg', 'jdi', 'sdh', 'enggak', 'nggak', 'tak',
        'kaga', 'engga', 'ngga', 't', 'amp',
    ]

    

    TESTING = False
    DEBUG = False # Default DEBUG adalah False untuk Config dasar


class ProductionConfig(Config):
    DEBUG = False # Eksplisit set False
    # SECRET_KEY akan diwarisi dari Config.
    # Saat deployment ke production, pastikan environment variable SECRET_KEY
    # diatur dengan nilai yang aman dan unik untuk meng-override default dari Config.


class DevelopmentConfig(Config):
    DEBUG = True
    # Untuk development, default SECRET_KEY dari Config sudah cukup.
    # Contoh penggunaan SQLite untuk development jika diinginkan:
    # SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(BASE_DIR, 'instance', 'dev.db') # Pastikan folder instance ada


# Pilih konfigurasi berdasarkan lingkungan
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig # Default jika FLASK_CONFIG tidak diset
}
