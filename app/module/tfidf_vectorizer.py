import numpy as np  # Modul untuk operasi numerik
import scipy.sparse as sp  # Modul untuk matriks sparse
from collections import Counter  # Modul untuk menghitung frekuensi
import re  # Modul untuk regex

class CustomTfidf:
    def __init__(self, max_features=1000, ngram_range=(1, 2), min_df=3, max_df_ratio=0.9):
        # Inisialisasi parameter TF-IDF
        self.max_features = max_features  # Batas maksimum fitur
        self.ngram_range = ngram_range  # Rentang n-gram (uni-gram dan bi-gram)
        self.min_df = min_df  # Minimum frekuensi dokumen
        self.max_df_ratio = max_df_ratio  # Rasio maksimum frekuensi dokumen
        self.vocabulary_ = {}  # Kamus untuk menyimpan term dan indeks
        self.idf_ = None  # Array IDF
        self.document_count_ = 0  # Jumlah dokumen
        
    def _generate_ngrams(self, tokens):
        # Menghasilkan n-gram dari token
        ngrams = []  # List untuk menyimpan n-gram
        min_n, max_n = self.ngram_range  # Ambil rentang n-gram
        for n in range(min_n, max_n + 1):  # Iterasi untuk setiap n
            for i in range(len(tokens) - n + 1):  # Iterasi posisi token
                ngrams.append(' '.join(tokens[i:i+n]))  # Tambah n-gram ke list
        return ngrams  # Kembalikan list n-gram

    def fit(self, raw_documents):
        # Melatih vectorizer untuk membangun kamus dan IDF
        self.vocabulary_ = {}  # Reset kamus
        document_frequency = Counter()  # Hitung frekuensi dokumen
        term_frequency = Counter()  # Hitung frekuensi term
        self.document_count_ = len(raw_documents)  # Simpan jumlah dokumen
        for doc in raw_documents:  # Iterasi setiap dokumen
            tokens = doc.split() if isinstance(doc, str) else doc  # Tokenisasi dokumen
            ngrams = self._generate_ngrams(tokens)  # Buat n-gram
            term_frequency.update(ngrams)  # Tambah frekuensi term
            document_frequency.update(set(ngrams))  # Tambah frekuensi dokumen
        max_df = self.document_count_ * self.max_df_ratio  # Hitung batas maksimum DF
        valid_terms = {
            term: freq for term, freq in document_frequency.items()
            if self.min_df <= freq <= max_df  # Filter term berdasarkan min_df dan max_df
        }
        top_terms = sorted(
            valid_terms.keys(),
            key=lambda term: (-term_frequency[term], term)  # Urutkan berdasarkan frekuensi
        )[:self.max_features]  # Ambil hingga max_features
        self.vocabulary_ = {term: idx for idx, term in enumerate(top_terms)}  # Buat kamus term-indeks
        self.idf_ = np.zeros(len(self.vocabulary_))  # Inisialisasi array IDF
        for term, idx in self.vocabulary_.items():  # Iterasi term dalam kamus
            df = document_frequency[term]  # Ambil frekuensi dokumen
            self.idf_[idx] = np.log((1 + self.document_count_) / (1 + df)) + 1  # Hitung IDF
        return self  # Kembalikan instance

    def transform(self, raw_documents, normalize=True):
        # Mengubah dokumen menjadi matriks TF-IDF
        if not self.vocabulary_:  # Periksa apakah kamus sudah ada
            raise ValueError("Vocabulary not learned. Call fit() first.")  # Lempar error jika belum fit
        n_samples = len(raw_documents)  # Jumlah dokumen
        n_features = len(self.vocabulary_)  # Jumlah fitur
        rows, cols, data = [], [], []  # List untuk matriks sparse
        for doc_idx, doc in enumerate(raw_documents):  # Iterasi dokumen
            tokens = doc.split() if isinstance(doc, str) else doc  # Tokenisasi dokumen
            term_counts = Counter(self._generate_ngrams(tokens))  # Hitung frekuensi n-gram
            for term, count in term_counts.items():  # Iterasi term dan frekuensi
                if term in self.vocabulary_:  # Periksa apakah term ada di kamus
                    term_idx = self.vocabulary_[term]  # Ambil indeks term
                    tf = 1 + np.log(count)  # Hitung TF
                    tfidf_score = tf * self.idf_[term_idx]  # Hitung skor TF-IDF
                    rows.append(doc_idx)  # Tambah indeks baris
                    cols.append(term_idx)  # Tambah indeks kolom
                    data.append(tfidf_score)  # Tambah skor TF-IDF
        X = sp.csr_matrix((data, (rows, cols)), shape=(n_samples, n_features))  # Buat matriks sparse
        if not normalize:  # Periksa apakah normalisasi diperlukan
            return X  # Kembalikan matriks tanpa normalisasi
        norms = np.sqrt(X.power(2).sum(axis=1))  # Hitung norma L2
        norms[norms == 0] = 1.0  # Hindari pembagian nol
        inv_norms_diag = sp.diags(1.0 / np.array(norms).ravel())  # Buat matriks diagonal invers norma
        X_normalized = inv_norms_diag @ X  # Normalisasi matriks
        return X_normalized  # Kembalikan matriks ternormalisasi

    def fit_transform(self, raw_documents, normalize=True):
        # Melatih dan mengubah dokumen menjadi matriks TF-IDF
        self.fit(raw_documents)  # Panggil fit untuk membangun kamus dan IDF
        return self.transform(raw_documents, normalize=normalize)  # Kembalikan matriks TF-IDF

    def get_feature_names_out(self):
        # Mengembalikan nama fitur dari kamus
        feature_names = np.empty(len(self.vocabulary_), dtype=object)  # Buat array untuk nama fitur
        for term, idx in self.vocabulary_.items():  # Iterasi kamus
            feature_names[idx] = term  # Isi array dengan term
        return feature_names  # Kembalikan array nama fitur

def preprocess_text_for_vectorizers(text):
    # Pra-pemrosesan teks untuk vectorizer
    if not isinstance(text, str):  # Periksa apakah input bukan string
        text = str(text)  # Konversi ke string
    text = text.lower()  # Konversi ke huruf kecil
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Hapus URL
    text = re.sub(r'\@\w+|\#\w+', '', text)  # Hapus mention dan hashtag
    text = re.sub(r'[^\w\s]', '', text)  # Hapus tanda baca
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'\s+', ' ', text).strip()  # Normalisasi spasi
    return text  # Kembalikan teks yang telah dibersihkan

def format_tfidf_for_display(texts, vectorizer):
    # Memformat hasil TF-IDF untuk tampilan
    if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:  # Periksa apakah vectorizer sudah di-fit
        return [{"info": "Vectorizer belum di-fit."}] * len(texts)  # Kembalikan pesan error
    processed_texts = [preprocess_text_for_vectorizers(text) for text in texts]  # Pra-proses teks
    tfidf_matrix = vectorizer.transform(processed_texts)  # Transformasi ke matriks TF-IDF
    feature_names = vectorizer.get_feature_names_out()  # Ambil nama fitur
    display_list = []  # List untuk hasil tampilan
    for i in range(tfidf_matrix.shape[0]):  # Iterasi setiap dokumen
        doc_vector = tfidf_matrix[i]  # Ambil vektor dokumen
        doc_scores = {
            feature_names[col]: float(doc_vector[0, col])  # Ambil skor TF-IDF per term
            for col in doc_vector.indices  # Iterasi indeks fitur aktif
        }
        if not doc_scores:  # Periksa apakah tidak ada skor
            display_list.append({"info": "Tidak ada token yang relevan dalam vocabulary."})  # Tambah pesan error
        else:
            sorted_scores = dict(sorted(doc_scores.items(), key=lambda item: item[1], reverse=True))  # Urutkan skor
            display_list.append(sorted_scores)  # Tambah skor ke list
    return display_list  # Kembalikan list hasil tampilan