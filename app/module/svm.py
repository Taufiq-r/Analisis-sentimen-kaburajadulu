import numpy as np
import random
from collections import Counter
# Impor pustaka untuk operasi array, pengacakan, dan penghitungan kelas.

# Parameter default untuk label otomatis
LABEL_OTOMATIS_PARAMS = {
    'learning_rate': 0.005,  # Laju belajar untuk optimasi.
    'lambda_param': 0.01,    # Parameter regularisasi untuk mencegah overfitting.
    'n_iters': 2000,         # Jumlah iterasi pelatihan.
    'batch_size': 16,        # Ukuran batch untuk pembaruan gradien.
    'lr_decay': 0.01         # Faktor penurunan laju belajar.
}

# Parameter default untuk label pakar (lebih stabil)
LABEL_PAKAR_PARAMS = {
    'learning_rate': 0.005,  # Laju belajar lebih kecil untuk konvergensi stabil.
    'lambda_param': 0.01,    # Nilai regularisasi yang optimal.
    'n_iters': 2000,         # Iterasi lebih banyak untuk pembelajaran menyeluruh.
    'batch_size': 16,        # Batch kecil untuk pembaruan gradien halus.
    'lr_decay': 0.01         # Faktor penurunan laju belajar yang optimal.
}

class SupportVectorMachine:
    # Kelas untuk model Support Vector Machine (SVM).
    
    def __init__(self, label_source='otomatis', learning_rate=None, lambda_param=None, 
                 n_iters=None, batch_size=None, lr_decay=None, random_state=42, 
                 class_weight='balanced', use_oversampling=False):
        # Inisialisasi model SVM dengan parameter opsional.
        default_params = LABEL_PAKAR_PARAMS if label_source.lower() == 'pakar' else LABEL_OTOMATIS_PARAMS
        # Pilih parameter default berdasarkan sumber label.
        
        self.lr = learning_rate if learning_rate is not None else default_params['learning_rate']
        # Set laju belajar, gunakan default jika tidak ditentukan.
        self.lambda_param = lambda_param if lambda_param is not None else default_params['lambda_param']
        # Set parameter regularisasi.
        self.n_iters = n_iters if n_iters is not None else default_params['n_iters']
        # Set jumlah iterasi.
        self.batch_size = batch_size if batch_size is not None else default_params['batch_size']
        # Set ukuran batch.
        self.lr_decay = lr_decay if lr_decay is not None else default_params['lr_decay']
        # Set faktor penurunan laju belajar.
        
        self.random_state = random_state
        # Set seed untuk pengacakan.
        self.label_source = label_source.lower()
        # Simpan sumber label (otomatis/pakar).
        self.class_weight = class_weight
        # Set mode bobot kelas (balanced atau tidak).
        self.use_oversampling = use_oversampling
        # Set status oversampling.
        
        self.classes_ = None
        # Daftar kelas (diisi saat pelatihan).
        self.class_weights_ = None
        # Bobot kelas (diisi saat pelatihan).
        self.w = {}
        # Bobot untuk setiap kelas.
        self.b = {}
        # Bias untuk setiap kelas.
        
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        # Atur seed untuk konsistensi pengacakan.

    def _init_weights_bias(self, n_features, class_label):
        # Inisialisasi bobot dan bias untuk kelas tertentu.
        scale = 1 / np.sqrt(n_features)
        # Skala inisialisasi berdasarkan jumlah fitur.
        self.w[class_label] = np.random.normal(0, scale, n_features)
        # Inisialisasi bobot dengan distribusi normal.
        self.b[class_label] = 0.0
        # Inisialisasi bias dengan nol.

    def _hinge_loss(self, w, b, X, y, sample_weights):
        # Menghitung hinge loss dan gradien untuk optimasi.
        margin = y * (X.dot(w) + b)
        # Hitung margin (y * (w*X + b)).
        
        loss_per_sample = np.maximum(0, 1 - margin)
        # Hitung hinge loss per sampel (max(0, 1 - margin)).
        weighted_loss = np.mean(loss_per_sample * sample_weights)
        # Hitung rata-rata loss tertimbang.
        
        total_loss = weighted_loss + 0.5 * self.lambda_param * np.dot(w, w)
        # Total loss = loss data + regularisasi.
        
        violated_mask = margin < 1
        # Identifikasi sampel yang melanggar margin (< 1).
        
        if not np.any(violated_mask):
            # Jika tidak ada pelanggaran:
            dw = self.lambda_param * w
            # Gradien bobot hanya dari regularisasi.
            db = 0.0
            # Gradien bias nol.
            return total_loss, dw, db
        
        X_violated = X[violated_mask]
        # Ambil data yang melanggar margin.
        y_violated = y[violated_mask]
        # Ambil label yang melanggar margin.
        weights_violated = sample_weights[violated_mask]
        # Ambil bobot sampel yang melanggar margin.
        
        weighted_y = (y_violated * weights_violated).reshape(-1, 1)
        # Bobot label untuk gradien.
        
        grad_w_data = X_violated.multiply(weighted_y)
        # Hitung gradien bobot dari data.
        
        dw = -np.mean(grad_w_data, axis=0)
        # Rata-rata gradien bobot dari data.
        
        if hasattr(dw, 'A1'):
            dw = dw.A1
            # Konversi ke array jika sparse.
            
        db = -np.mean(y_violated * weights_violated)
        # Hitung gradien bias.
        
        dw += self.lambda_param * w
        # Tambahkan gradien regularisasi ke bobot.
        
        return total_loss, dw, db
        # Kembalikan total loss, gradien bobot, dan gradien bias.

    def _get_learning_rate(self, epoch):
        # Menghitung laju belajar dengan penurunan (decay).
        return self.lr / (1 + self.lr_decay * epoch)
        # Laju belajar menurun seiring epoch.

    def fit(self, X, y):
        # Melatih model SVM.
        if X is None or y is None or X.shape[0] == 0 or len(y) == 0:
            raise ValueError("Input data X and y cannot be empty.")
            # Validasi input tidak kosong.
        if X.shape[0] != len(y):
            raise ValueError("Shape mismatch between X and y.")
            # Validasi jumlah sampel sesuai.
            
        n_samples, n_features = X.shape
        # Dapatkan jumlah sampel dan fitur.
        y_input = np.asarray(y)
        # Konversi label ke array.
        
        self.classes_ = np.unique(y_input)
        # Dapatkan kelas unik dari label.
        if len(self.classes_) < 2:
            raise ValueError("Training data must have at least 2 classes.")
            # Pastikan ada minimal 2 kelas.
            
        priority_order = {'positif': 0, 'negatif': 1, 'netral': 2}
        self.classes_ = np.array(sorted(
            self.classes_,
            key=lambda x: priority_order.get(str(x).lower(), 999)
        ))
        # Urutkan kelas dengan prioritas (positif, negatif, netral).
        
        print(f"[SVM] Starting training with {n_samples} samples, {n_features} features.")
        print(f"[SVM] Params: lr={self.lr}, lambda={self.lambda_param}, iters={self.n_iters}, batch={self.batch_size}")
        print(f"[SVM] Found classes: {self.classes_}")
        # Log informasi pelatihan.

        if self.use_oversampling:
            print("[SVM] Applying oversampling...")
            X, y_input = self.oversample_minority_classes(X, y_input)
            n_samples = X.shape[0]
            print(f"[SVM] Data size after oversampling: {n_samples} samples.")
            # Terapkan oversampling jika diaktifkan.

        if self.class_weight == 'balanced':
            class_counts = Counter(y_input)
            total_samples = len(y_input)
            n_classes = len(self.classes_)
            self.class_weights_ = {
                cls: total_samples / (n_classes * count)
                for cls, count in class_counts.items()
            }
            print(f"[SVM] Calculated class weights: {self.class_weights_}")
            # Hitung bobot kelas untuk menangani ketidakseimbangan.
        else:
            self.class_weights_ = {cls: 1.0 for cls in self.classes_}
            # Gunakan bobot seragam jika tidak 'balanced'.

        for class_label in self.classes_:
            print(f"\n[SVM] Training classifier for class: '{class_label}'")
            # Log pelatihan untuk setiap kelas.
            self._init_weights_bias(n_features, class_label)
            # Inisialisasi bobot dan bias untuk kelas.
            
            y_binary = np.where(y_input == class_label, 1, -1)
            #one versus rest
            # Ubah label menjadi biner (1 untuk kelas ini, -1 untuk lainnya).
            
            sample_weights = np.ones(n_samples)
            # Inisialisasi bobot sampel.
            pos_weight = self.class_weights_[class_label]
            # Bobot untuk kelas positif.
            neg_weights_sum = sum(w for c, w in self.class_weights_.items() if c != class_label)
            neg_count = len(self.classes_) - 1
            neg_weight = neg_weights_sum / max(1, neg_count)
            # Hitung bobot untuk kelas negatif.
            
            sample_weights[y_binary == 1] = pos_weight
            sample_weights[y_binary == -1] = neg_weight
            # Terapkan bobot sampel.
            print(f"[SVM] Using pos_weight={pos_weight:.2f}, neg_weight={neg_weight:.2f}")
            # Log bobot kelas.

            for epoch in range(self.n_iters):
                # Iterasi pelatihan.
                current_lr = self._get_learning_rate(epoch)
                # Dapatkan laju belajar untuk epoch ini.
                
                if epoch == 0 or (epoch + 1) % 200 == 0 or epoch == self.n_iters - 1:
                    total_loss, _, _ = self._hinge_loss(
                        self.w[class_label], self.b[class_label],
                        X, y_binary, sample_weights
                    )
                    print(f"[SVM][{class_label}] Epoch {epoch+1}/{self.n_iters} - Loss: {total_loss:.4f}, LR: {current_lr:.6f}")
                    # Log loss pada epoch tertentu.

                indices = np.random.permutation(n_samples)
                # Acak indeks sampel.
                X_shuffled = X[indices]
                y_binary_shuffled = y_binary[indices]
                weights_shuffled = sample_weights[indices]
                # Acak data untuk pelatihan.
                
                for i in range(0, n_samples, self.batch_size):
                    # Iterasi per batch.
                    batch_end = i + self.batch_size
                    X_batch = X_shuffled[i:batch_end]
                    y_batch = y_binary_shuffled[i:batch_end]
                    weights_batch = weights_shuffled[i:batch_end]
                    # Ambil batch data.

                    _, grad_w, grad_b = self._hinge_loss(
                        self.w[class_label], self.b[class_label],
                        X_batch, y_batch, weights_batch
                    )
                    # Hitung gradien untuk batch.
                    
                    self.w[class_label] -= current_lr * grad_w
                    self.b[class_label] -= current_lr * grad_b
                    # Perbarui bobot dan bias.

        print("\n[SVM] Training complete.")
        # Log selesai pelatihan.
        return self
        # Kembalikan model yang sudah dilatih.

    def _predict_class(self, X, class_label):
        # Prediksi skor untuk kelas tertentu.
        if class_label not in self.w:
            raise ValueError(f"Model not trained for class '{class_label}'.")
            # Pastikan model dilatih untuk kelas ini.
        
        return X.dot(self.w[class_label]) + self.b[class_label]
        # Hitung skor (w*X + b).

    def predict(self, X):
        # Prediksi kelas untuk data input.
        if not self.w or not self.classes_.size:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
            # Pastikan model sudah dilatih.

        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))
        # Inisialisasi matriks skor untuk semua kelas.

        for i, class_label in enumerate(self.classes_):
            scores[:, i] = self._predict_class(X, class_label)
            # Hitung skor untuk setiap kelas.
        
        highest_score_indices = np.argmax(scores, axis=1)
        # Pilih kelas dengan skor tertinggi.
        return self.classes_[highest_score_indices]
        # Kembalikan kelas yang diprediksi.

    def predict_proba(self, X):
        # Prediksi probabilitas kelas.
        if not self.w or not self.classes_.size:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
            # Pastikan model sudah dilatih.
        
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, len(self.classes_)))
        # Inisialisasi matriks skor.

        for i, class_label in enumerate(self.classes_):
            scores[:, i] = self._predict_class(X, class_label)
            # Hitung skor untuk setiap kelas.
        
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        # Ubah skor ke probabilitas dengan softmax.
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Kembalikanそうな。ikan probabilitas.

    def get_params(self, deep=True):
        # Mengembalikan parameter model.
        return {
            'label_source': self.label_source,
            'learning_rate': self.lr,
            'lambda_param': self.lambda_param,
            'n_iters': self.n_iters,
            'batch_size': self.batch_size,
            'lr_decay': self.lr_decay,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'use_oversampling': self.use_oversampling
        }
    
    def oversample_minority_classes(self, X, y):
        # Fungsi untuk oversampling kelas minoritas.
        class_counts = Counter(y)
        # Hitung jumlah sampel per kelas.
        max_count = max(class_counts.values())
        # Dapatkan jumlah sampel kelas mayoritas.

        X_resampled, y_resampled = [], []
        # Inisialisasi daftar untuk data yang di-oversample.
        
        is_sparse = hasattr(X, "tocsr")
        # Periksa apakah data dalam format sparse.

        for cls, count in class_counts.items():
            class_indices = np.where(y == cls)[0]
            # Dapatkan indeks untuk kelas ini.
            if is_sparse:
                X_resampled.append(X[class_indices])
            else:
                X_resampled.extend(X[class_indices])
            # Tambahkan data kelas ke daftar.
            y_resampled.extend([cls] * count)
            # Tambahkan label kelas ke daftar.

            n_samples_to_add = max_count - count
            # Hitung jumlah sampel yang perlu ditambahkan.
            if n_samples_to_add > 0:
                original_class_indices = np.where(y == cls)[0]
                oversampled_indices = np.random.choice(original_class_indices, size=n_samples_to_add, replace=True)
                # Pilih indeks secara acak untuk oversampling.
                if is_sparse:
                    X_resampled.append(X[oversampled_indices])
                else:
                    X_resampled.extend(X[oversampled_indices])
                y_resampled.extend([cls] * n_samples_to_add)
                # Tambahkan data dan label yang di-oversample.

        if is_sparse:
            from scipy.sparse import vstack
            X_resampled = vstack(X_resampled)
            # Gabungkan data sparse.
        else:
            X_resampled = np.array(X_resampled)
            # Konversi ke array jika tidak sparse.
            
        y_resampled = np.array(y_resampled)
        # Konversi label ke array.
        
        shuffle_indices = np.random.permutation(len(y_resampled))
        # Acak indeks data.
        return X_resampled[shuffle_indices], y_resampled[shuffle_indices]
        # Kembalikan data dan label yang di-oversample dan diacak.