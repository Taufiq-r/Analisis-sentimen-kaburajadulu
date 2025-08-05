import math
import numpy as np
from collections import Counter
import scipy.sparse as sp
import random

# Import pustaka untuk perhitungan matematis, array, penghitungan kelas, matriks sparse, dan pengacakan.

class MultinomialNaiveBayesClassifier:
    # Deklarasi kelas untuk model Naive Bayes Multinomial.

    def __init__(self, alpha=1.0):
        # Inisialisasi model dengan parameter alpha (smoothing).
        self.alpha = alpha  # Parameter untuk Laplace smoothing.
        self.class_priors_ = None  # Probabilitas prior kelas (akan diisi saat pelatihan).
        self.feature_log_prob_ = None  # Log probabilitas fitur per kelas.
        self.classes_ = None  # Daftar kelas unik.
        self.n_features_ = None  # Jumlah fitur.

    def fit(self, X, y):
        # Melatih model dengan data input X dan label y.
        if not sp.issparse(X):
            raise TypeError("Input X harus berupa scipy sparse matrix (misal, csr_matrix).")
            # Memastikan X adalah matriks sparse.

        n_samples, n_features = X.shape
        # Mendapatkan jumlah sampel dan fitur dari X.
        self.classes_ = np.unique(y)
        # Menyimpan kelas unik dari y.
        n_classes = len(self.classes_)
        # Menghitung jumlah kelas.
        self.n_features_ = n_features
        # Menyimpan jumlah fitur.

        self.class_priors_ = np.zeros(n_classes)
        # Inisialisasi array untuk prior kelas.
        feature_counts_per_class = np.zeros((n_classes, n_features))
        # Inisialisasi array untuk jumlah fitur per kelas.

        for i, current_class in enumerate(self.classes_):
            # Iterasi untuk setiap kelas.
            X_class = X[y == current_class]
            # Mengambil data X untuk kelas tertentu.
            self.class_priors_[i] = math.log(X_class.shape[0] / n_samples)
            # Menghitung log probabilitas prior kelas (frekuensi kelas / total sampel).
            feature_counts_per_class[i, :] = X_class.sum(axis=0)
            # Menghitung jumlah kemunculan fitur untuk kelas ini.

        total_counts_per_class = feature_counts_per_class.sum(axis=1)
        # Menghitung total jumlah fitur per kelas.
        numerator = feature_counts_per_class + self.alpha
        # Menambahkan alpha (smoothing) ke jumlah fitur.
        denominator = total_counts_per_class[:, np.newaxis] + self.alpha * self.n_features_
        # Menghitung penyebut untuk probabilitas fitur (total + smoothing).
        self.feature_log_prob_ = np.log(numerator) - np.log(denominator)
        # Menghitung log probabilitas fitur per kelas.
        
        return self
        # Mengembalikan objek model yang sudah dilatih.

    def predict(self, X):
        # Memprediksi kelas untuk data input X.
        if self.feature_log_prob_ is None:
            raise RuntimeError("Model harus dilatih terlebih dahulu dengan metode .fit()")
            # Memastikan model sudah dilatih.
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Jumlah fitur data input ({X.shape[1]}) tidak sesuai dengan data latih ({self.n_features_}).")
            # Memastikan jumlah fitur sesuai.

        log_likelihoods = X.dot(self.feature_log_prob_.T)
        # Menghitung log likelihood (X * log probabilitas fitur).
        log_posteriors = log_likelihoods + self.class_priors_
        # Menambahkan log prior untuk mendapatkan log posterior.
        predicted_indices = np.argmax(log_posteriors, axis=1)
        # Memilih kelas dengan probabilitas tertinggi.
        return self.classes_[predicted_indices]
        # Mengembalikan kelas yang diprediksi.

    def predict_proba(self, X):
        # Memprediksi probabilitas kelas untuk data input X.
        if self.feature_log_prob_ is None:
            raise RuntimeError("Model harus dilatih terlebih dahulu dengan metode .fit()")
            # Memastikan model sudah dilatih.
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Jumlah fitur data input ({X.shape[1]}) tidak sesuai dengan data latih ({self.n_features_}).")
            # Memastikan jumlah fitur sesuai.

        log_likelihoods = X.dot(self.feature_log_prob_.T)
        # Menghitung log likelihood.
        log_posteriors = log_likelihoods + self.class_priors_
        # Menghitung log posterior.
        max_log_probs = np.max(log_posteriors, axis=1, keepdims=True)
        # Mendapatkan nilai maksimum untuk normalisasi.
        exp_scores = np.exp(log_posteriors - max_log_probs)
        # Mengubah log posterior ke skala eksponensial.
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Menghitung probabilitas dengan normalisasi.
        return probas
        # Mengembalikan probabilitas kelas.

    def score(self, X, y):
        # Menghitung akurasi model.
        if len(y) == 0: return 0.0
        # Mengembalikan 0 jika tidak ada data.
        preds = self.predict(X)
        # Memprediksi kelas untuk X.
        return np.mean(preds == y)
        # Mengembalikan rata-rata prediksi yang benar.

def oversample_minority(X, y, random_state=42):
    # Fungsi untuk oversampling kelas minoritas.
    random.seed(random_state)
    np.random.seed(random_state)
    # Mengatur seed untuk pengacakan yang konsisten.
    
    if X.shape[0] == 0:
        return X, y
        # Mengembalikan input jika kosong.

    class_counts = Counter(y)
    # Menghitung jumlah sampel per kelas.
    if len(class_counts) <= 1:
        return X, y
        # Mengembalikan input jika hanya ada satu kelas.
        
    majority_size = max(class_counts.values())
    # Mendapatkan jumlah sampel kelas mayoritas.
    resampled_indices = []
    # Inisialisasi daftar indeks untuk resampling.

    for cls in class_counts:
        # Iterasi untuk setiap kelas.
        class_indices = np.where(y == cls)[0]
        # Mendapatkan indeks sampel untuk kelas tertentu.
        resampled_indices.extend(class_indices)
        # Menambahkan indeks asli ke daftar.
        n_to_add = majority_size - len(class_indices)
        # Menghitung jumlah sampel yang perlu ditambahkan.
        if n_to_add > 0:
            new_indices = np.random.choice(class_indices, size=n_to_add, replace=True)
            # Memilih indeks secara acak untuk oversampling.
            resampled_indices.extend(new_indices)
            # Menambahkan indeks baru ke daftar.
    
    random.shuffle(resampled_indices)
    # Mengacak urutan indeks.
    X_resampled = X[resampled_indices]
    # Membuat X baru berdasarkan indeks yang diacak.
    y_resampled = y[resampled_indices]
    # Membuat y baru berdasarkan indeks yang diacak.
    
    return X_resampled, y_resampled
    # Mengembalikan data yang sudah di-oversample.