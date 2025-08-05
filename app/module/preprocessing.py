import re  # Modul untuk regex
import json  # Modul untuk memproses JSON
import multiprocessing  # Modul untuk pemrosesan paralel
import time  # Modul untuk mengukur waktu
import os  # Modul untuk operasi sistem file
import html  # Modul untuk menangani entitas HTML
from concurrent.futures import ProcessPoolExecutor  # Modul untuk eksekusi paralel
from functools import lru_cache  # Modul untuk caching fungsi

# Dictionary pola regex untuk pembersihan teks
REGEX_PATTERNS = {
    'links': re.compile(r'(?:https?://)?(?:www\.)?(?:t\.co|bit\.ly|tinyurl\.com|goo\.gl|instagram\.com|facebook\.com|twitter\.com|youtube\.com)/\S+|https?://\S+', re.IGNORECASE),  # Hapus URL
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Hapus alamat email
    'numbers': re.compile(r'\d+'),  # Hapus angka
    'non_ascii': re.compile(r'[^\x00-\x7F]+'),  # Hapus karakter non-ASCII
    'hashtags': re.compile(r'#\w+'),  # Hapus hashtag
    'repeated_words': re.compile(r'\b(\w+)(\s+\1)+\b', re.IGNORECASE),  # Hapus kata berulang
    'multiple_symbols': re.compile(r'[^\w\s]{2,}'),  # Hapus simbol berulang
    'punctuation': re.compile(r'[^\w\s]'),  # Hapus tanda baca
    'repeated_letters': re.compile(r'(\w)\1{2,}'),  # Normalisasi huruf berulang
    'multiple_spaces': re.compile(r'\s+'),  # Normalisasi spasi berlebih
    'username_mentions': re.compile(r'@[A-Za-z0-9_]+'),  # Hapus mention pengguna
    'rt_pattern': re.compile(r'\bRT\b', re.IGNORECASE),  # Hapus kata "RT"
    'extra_whitespace': re.compile(r'^\s+|\s+$|\s+(?=\s)')  # Hapus spasi awal/akhir
}

# Daftar frasa penting untuk diganti dengan token khusus
IMPORTANT_PHRASES = [
    (re.compile(r'#kaburajadulu', re.IGNORECASE), 'kabur_aja_dulu'),  # Ganti hashtag kaburajadulu
    (re.compile(r'kabur\s+aja\s+dulu', re.IGNORECASE), 'kabur_saja_dulu'),  # Ganti frasa kabur aja dulu
    (re.compile(r'#kaburselamanya', re.IGNORECASE), 'kabur_selamanya'),  # Ganti hashtag kaburselamanya
    (re.compile(r'#KaburSajaDulu', re.IGNORECASE), 'kabur_saja_dulu'),  # Ganti hashtag KaburSajaDulu
    (re.compile(r'#KaburinDuitDulu', re.IGNORECASE), 'kaburin_duit_dulu'),  # Ganti hashtag KaburinDuitDulu
    (re.compile(r'gen\s+z', re.IGNORECASE), 'generasi_z'),  # Ganti frasa gen z
]

def load_json_dict(file_path):
    # Memuat file JSON ke dictionary
    if not file_path or not os.path.exists(file_path):  # Periksa keberadaan file
        print(f"Warning: File not found at {file_path}")  # Tampilkan peringatan jika file tidak ada
        return {}  # Kembalikan dictionary kosong
    try:
        with open(file_path, 'r', encoding='utf-8') as f:  # Buka file dengan encoding UTF-8
            return json.load(f)  # Muat JSON ke dictionary
    except json.JSONDecodeError:  # Tangani error format JSON
        print(f"Warning: Invalid JSON format in {file_path}")  # Tampilkan peringatan
        return {}  # Kembalikan dictionary kosong
    except Exception as e:  # Tangani error lainnya
        print(f"Warning: Error loading JSON from {file_path}: {e}")  # Tampilkan peringatan
        return {}  # Kembalikan dictionary kosong

def remove_emoji(text):
    # Menghapus emoji dari teks
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    emoji_pattern = re.compile(
        "["  # Pola Unicode untuk emoji
        "\U0001F600-\U0001F64F"  # Emotikon
        "\U0001F300-\U0001F5FF"  # Simbol & piktogram
        "\U0001F680-\U0001F6FF"  # Transportasi & peta
        "\U0001F1E0-\U0001F1FF"  # Bendera
        "\U0001F700-\U0001F77F"  # Alkemia
        "\U0001F780-\U0001F7FF"  # Bentuk geometris
        "\U0001F800-\U0001F8FF"  # Panah tambahan
        "\U0001F900-\U0001F9FF"  # Simbol tambahan
        "\U0001FA00-\U0001FA6F"  # Simbol catur
        "\U0001FA70-\U0001FAFF"  # Simbol diperpanjang
        "\U00002702-\U000027B0"  # Dingbat
        "\U000024C2-\U0001F251"  # Simbol lainnya
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)  # Ganti emoji dengan string kosong

def clean_html_entities(text):
    # Membersihkan entitas HTML dan karakter khusus
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    text = html.unescape(text)  # Konversi entitas HTML seperti & ke &
    html_entities = {
        ' ': ' ',  # Ganti spasi non-breaking dengan spasi
        '–': '-',  # Ganti en-dash dengan tanda hubung
        '—': '-',  # Ganti em-dash dengan tanda hubung
        '…': '...'  # Ganti elipsis dengan tiga titik
    }
    for entity, replacement in html_entities.items():  # Iterasi entitas HTML
        text = text.replace(entity, replacement)  # Ganti entitas dengan pengganti
    return text  # Kembalikan teks yang telah dibersihkan

def remove_urls_and_mentions(text):
    # Menghapus URL, email, mention, dan RT
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    text = REGEX_PATTERNS['links'].sub('', text)  # Hapus URL
    text = REGEX_PATTERNS['email'].sub('', text)  # Hapus alamat email
    text = REGEX_PATTERNS['username_mentions'].sub('', text)  # Hapus mention pengguna
    text = REGEX_PATTERNS['rt_pattern'].sub('', text)  # Hapus kata "RT"
    return text  # Kembalikan teks yang telah dibersihkan

def normalize_text_patterns(text):
    # Mengganti frasa penting dan menghapus hashtag
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    for pattern, replacement in IMPORTANT_PHRASES:  # Iterasi frasa penting
        text = pattern.sub(replacement, text)  # Ganti frasa dengan token
    text = REGEX_PATTERNS['hashtags'].sub('', text)  # Hapus hashtag lainnya
    text = re.sub(r'(\b\w+)\s*/\s*(\b\w+)', r'\1 atau \2', text)  # Ganti "A/B" dengan "A atau B"
    return text  # Kembalikan teks yang telah dinormalisasi

def normalize_numeric_comparisons(text):
    # Normalisasi ekspresi perbandingan numerik
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    text = re.sub(r'>=\s*(\d+)', r'lebih dari sama dengan \1', text)  # Ganti >= dengan teks
    text = re.sub(r'<=\s*(\d+)', r'kurang dari sama dengan \1', text)  # Ganti <= dengan teks
    text = re.sub(r'>\s*(\d+)', r'lebih dari \1', text)  # Ganti > dengan teks
    text = re.sub(r'<\s*(\d+)', r'kurang dari \1', text)  # Ganti < dengan teks
    return text  # Kembalikan teks yang telah dinormalisasi

def clean_symbols_and_punctuation(text):
    # Membersihkan simbol berulang dan tanda baca
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    text = REGEX_PATTERNS['repeated_letters'].sub(r'\1\1', text)  # Batasi huruf berulang
    text = REGEX_PATTERNS['multiple_symbols'].sub(' ', text)  # Ganti simbol berulang dengan spasi
    text = REGEX_PATTERNS['punctuation'].sub(' ', text)  # Ganti tanda baca dengan spasi
    text = re.sub(r'!{2,}', '!', text)  # Batasi tanda seru menjadi satu
    text = re.sub(r'\?{2,}', '?', text)  # Batasi tanda tanya menjadi satu
    return text  # Kembalikan teks yang telah dibersihkan

def handle_special_numbers(text):
    # Menghapus angka dari teks
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    return REGEX_PATTERNS['numbers'].sub('', text)  # Ganti angka dengan string kosong

def normalize_whitespace(text):
    # Normalisasi spasi dan hapus kata berulang
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return text  # Kembalikan teks asli jika tidak valid
    text = REGEX_PATTERNS['multiple_spaces'].sub(' ', text)  # Ganti spasi berlebih dengan satu spasi
    text = REGEX_PATTERNS['repeated_words'].sub(r'\1', text)  # Hapus kata berulang
    return text.strip()  # Hapus spasi awal/akhir dan kembalikan teks

def clean_text_pipeline(text):
    # Pipeline pembersihan teks
    if not isinstance(text, str) or not text.strip():  # Periksa apakah input string valid
        return ""  # Kembalikan string kosong jika tidak valid
    text = clean_html_entities(text)  # Bersihkan entitas HTML
    text = remove_urls_and_mentions(text)  # Hapus URL, email, mention, dan RT
    text = remove_emoji(text)  # Hapus emoji
    text = normalize_text_patterns(text)  # Normalisasi frasa dan hashtag
    text = normalize_numeric_comparisons(text)  # Normalisasi perbandingan numerik
    text = text.lower()  # Konversi ke huruf kecil
    text = clean_symbols_and_punctuation(text)  # Bersihkan simbol dan tanda baca
    text = handle_special_numbers(text)  # Hapus angka
    text = REGEX_PATTERNS['non_ascii'].sub('', text)  # Hapus karakter non-ASCII
    text = normalize_whitespace(text)  # Normalisasi spasi
    return text.strip()  # Kembalikan teks yang telah dibersihkan

def replace_taboo_words(text, kamus_tidak_baku):
    # Mengganti kata slang dengan kata baku
    if not isinstance(text, str) or not text or not kamus_tidak_baku:  # Periksa input valid
        return text  # Kembalikan teks asli jika tidak valid
    sorted_slang_items = sorted(kamus_tidak_baku.items(), key=lambda item: len(item[0]), reverse=True)  # Urutkan berdasarkan panjang frasa
    modified_text = text
    for slang_phrase, baku_word in sorted_slang_items:  # Iterasi kamus slang
        pattern = r'\b' + re.escape(slang_phrase) + r'\b'  # Pola untuk kata utuh
        modified_text = re.sub(pattern, baku_word, modified_text, flags=re.IGNORECASE)  # Ganti slang dengan baku
    return modified_text  # Kembalikan teks yang telah diganti

def remove_stopwords(tokens, stopwords_set):
    # Menghapus stopwords, kecuali frasa dengan underscore
    if not isinstance(tokens, list) or not tokens:  # Periksa apakah input list valid
        return []  # Kembalikan list kosong jika tidak valid
    return [word for word in tokens if word and ('_' in word or word not in stopwords_set)]  # Filter token

def apply_stemming(tokens):
    # Menerapkan stemming pada token, kecuali frasa dengan underscore
    if not isinstance(tokens, list) or not tokens:  # Periksa apakah input list valid
        return []  # Kembalikan list kosong jika tidak valid
    global cached_stem  # Akses fungsi stemming yang di-cache
    if 'cached_stem' not in globals():  # Periksa apakah stemmer tersedia
        return tokens  # Kembalikan token asli jika stemmer tidak ada
    return [cached_stem(word) if '_' not in word else word for word in tokens if word]  # Stem token tanpa underscore

def tokenize(text):
    # Memecah teks menjadi token
    if not isinstance(text, str) or not text:  # Periksa apakah input string valid
        return []  # Kembalikan list kosong jika tidak valid
    tokens = [token.strip() for token in text.split() if token.strip()]  # Pisah teks berdasarkan spasi
    return [token for token in tokens if '_' in token or (len(token) >= 2 and not token.isdigit())]  # Filter token minimal 2 huruf

def init_worker(slang_words_path):
    # Inisialisasi worker untuk pemrosesan paralel
    global stopwords_worker, slang_words_worker, cached_stem  # Deklarasi variabel global
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Impor Sastrawi stopwords
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # Impor Sastrawi stemmer
        stopword_factory = StopWordRemoverFactory()  # Buat factory stopwords
        stopwords_worker = set(stopword_factory.get_stop_words())  # Muat stopwords default
        custom_stopwords = [
            'https', 'http', 'www', 'co', 'com', 'rt', 'amp', 'gt', 'lt', 'quot', 
            'apos', 'nbsp', 'ya', 'yah', 'iya', 'sih', 'deh', 'dong', 'kok', 'lah',
            'wkwk', 'wkwkwk', 'haha', 'hehe', 'hehehe', 'hahaha', 'hihi', 'huhu',
            'wow', 'wah', 'aduh', 'astaga', 'alamak', "awkwkwowk", "awkwokwowk", "wkwowk",
            'prof', 'anjay', 'cuy', 'sa', 'an', "l", "n", "toh", "wkwkwkkw", "si", "s", "bos"
        ]  # Daftar stopwords kustom
        stopwords_worker.update(custom_stopwords)  # Tambah stopwords kustom
        stemmer_factory = StemmerFactory()  # Buat factory stemmer
        stemmer_worker = stemmer_factory.create_stemmer()  # Buat stemmer
        slang_words_worker = load_json_dict(slang_words_path)  # Muat kamus slang
        if slang_words_worker:  # Periksa apakah kamus slang berhasil dimuat
            print(f"Worker PID {os.getpid()}: Successfully loaded {len(slang_words_worker)} slang words.")  # Log jumlah slang
        @lru_cache(maxsize=10000)  # Cache hingga 10000 kata
        def stem_word(word):  # Fungsi stemming dengan caching
            if not word or not isinstance(word, str):  # Periksa input valid
                return word  # Kembalikan kata asli jika tidak valid
            return stemmer_worker.stem(word)  # Stem kata
        globals()['cached_stem'] = stem_word  # Simpan fungsi stemming ke global
    except ImportError as e:  # Tangani error impor Sastrawi
        print(f"Error: Failed to import Sastrawi: {e}. Preprocessing will be limited.")  # Tampilkan peringatan
        stopwords_worker, slang_words_worker = set(), {}  # Inisialisasi kosong
        globals()['cached_stem'] = lambda word: word  # Fallback stemmer
    except Exception as e:  # Tangani error lainnya
        print(f"Error during worker initialization: {e}")  # Tampilkan peringatan
        stopwords_worker, slang_words_worker = set(), {}  # Inisialisasi kosong
        globals()['cached_stem'] = lambda word: word  # Fallback stemmer

def preprocess_single_text_worker(text_info):
    # Memproses satu teks dengan pipeline penuh
    try:
        text = text_info['full_text']  # Ambil teks dari input
        if not isinstance(text, str) or not text.strip():  # Periksa apakah teks valid
            return None  # Kembalikan None jika tidak valid
        cleaned = clean_text_pipeline(text)  # Bersihkan teks
        if not cleaned.strip():  # Periksa apakah teks bersih kosong
            return None  # Kembalikan None jika kosong
        baku = replace_taboo_words(cleaned, slang_words_worker)  # Ganti kata slang dengan baku
        tokens = tokenize(baku)  # Tokenisasi teks
        if not tokens:  # Periksa apakah token kosong
            return None  # Kembalikan None jika kosong
        filtered = remove_stopwords(tokens, stopwords_worker)  # Hapus stopwords
        stemmed = apply_stemming(filtered)  # Terapkan stemming
        if not stemmed:  # Periksa apakah hasil stemming kosong
            return None  # Kembalikan None jika kosong
        return {
            'username': text_info.get('username', ''),  # Ambil username
            'full_text': text,  # Simpan teks asli
            'text_clean': cleaned,  # Simpan teks yang telah dibersihkan
            'text_baku': baku,  # Simpan teks dengan kata baku
            'text_stopwords': " ".join(filtered),  # Simpan teks tanpa stopwords
            'text_stem': " ".join(stemmed),  # Simpan teks yang telah distem
            'created_at': text_info.get('created_at', '')  # Ambil waktu pembuatan
        }  # Kembalikan dictionary hasil preprocessing
    except Exception as e:  # Tangani error selama pemrosesan
        print(f"Error processing text in worker: {e}")  # Tampilkan peringatan
        return None  # Kembalikan None jika gagal

def preprocess_texts_batch(texts_data, slang_words_path, max_workers=None):
    # Memproses teks secara paralel
    if not texts_data:  # Periksa apakah data teks kosong
        return []  # Kembalikan list kosong jika kosong
    if max_workers is None:  # Tentukan jumlah worker jika tidak ditentukan
        max_workers = min(4, os.cpu_count() or 1)  # Gunakan hingga 4 worker
    print(f"Starting batch preprocessing with {max_workers} workers for {len(texts_data)} texts...")  # Log mulai proses
    t0 = time.time()  # Catat waktu mulai
    results = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(slang_words_path,)) as executor:  # Buat executor paralel
            results = list(filter(None, executor.map(preprocess_single_text_worker, texts_data)))  # Proses teks paralel
    except Exception as e:  # Tangani error pemrosesan paralel
        print(f"Error in batch processing: {e}. Falling back to single process mode.")  # Tampilkan peringatan
        init_worker(slang_words_path)  # Inisialisasi worker untuk mode tunggal
        results = [res for text_data in texts_data if (res := preprocess_single_text_worker(text_data)) is not None]  # Proses tunggal
    t1 = time.time()  # Catat waktu selesai
    success_rate = (len(results) / len(texts_data) * 100) if texts_data else 0  # Hitung tingkat keberhasilan
    print(f"Batch processing completed in {t1-t0:.2f} seconds.")  # Log waktu proses
    print(f"- Successfully processed: {len(results)}/{len(texts_data)} texts ({success_rate:.2f}%)")  # Log hasil proses
    return results  # Kembalikan hasil preprocessing

def preprocess_single_text(text, slang_words_path=None, return_all_steps=False):
    # Memproses satu teks tanpa paralel
    if not isinstance(text, str) or not text.strip():  # Periksa apakah teks valid
        return {} if return_all_steps else None  # Kembalikan hasil kosong
    if slang_words_path and 'slang_words_worker' not in globals():  # Periksa kamus slang
        init_worker(slang_words_path)  # Inisialisasi worker jika diperlukan
    result = preprocess_single_text_worker({'full_text': text})  # Proses teks
    if not result:  # Periksa apakah hasil kosong
        return {} if return_all_steps else None  # Kembalikan hasil kosong
    return result if return_all_steps else result['text_stem']  # Kembalikan semua langkah atau hanya teks stemmed

def extract_important_phrases(text):
    # Mengekstrak frasa penting dari teks
    if not isinstance(text, str) or not text:  # Periksa apakah teks valid
        return []  # Kembalikan list kosong jika tidak valid
    detected_phrases = []
    for pattern, replacement in IMPORTANT_PHRASES:  # Iterasi frasa penting
        if pattern.search(text):  # Periksa kecocokan pola
            detected_phrases.append(replacement)  # Tambah frasa yang cocok
    return list(set(detected_phrases))  # Kembalikan daftar frasa unik

def preprocess_workflow(dataset_entries, slangwords_path, max_workers=None):
    # Workflow utama untuk memproses dataset
    if not dataset_entries:  # Periksa apakah dataset kosong
        print("Warning: No dataset entries provided.")  # Tampilkan peringatan
        return []  # Kembalikan list kosong
    texts_data = [
        {
            'full_text': getattr(entry, 'full_text', str(entry)),  # Ambil teks dari entri
            'username': getattr(entry, 'username', ''),  # Ambil username
            'created_at': getattr(entry, 'created_at', '')  # Ambil waktu pembuatan
        }
        for entry in dataset_entries  # Iterasi entri dataset
    ]
    if not texts_data:  # Periksa apakah data teks kosong
        print("Error: No valid texts to process after parsing entries.")  # Tampilkan peringatan
        return []  # Kembalikan list kosong
    return preprocess_texts_batch(texts_data, slangwords_path, max_workers)  # Proses batch dan kembalikan hasil