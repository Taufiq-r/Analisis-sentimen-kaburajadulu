# /Tugas Akhir 1.3 NB + svm/app/models.py

from . import db # Impor db dari __init__.py di paket 'app' (folder ini)
from datetime import datetime, timedelta
import numpy as np
from scipy import sparse
import os
import json

def get_wib_time():
    """Get current time in WIB (UTC+7)"""
    return datetime.utcnow() + timedelta(hours=7)

class Dataset(db.Model):
    __tablename__ = 'dataset'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=True)
    full_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.String(255), nullable=True) # Pertimbangkan menggunakan db.DateTime

    preprocessings = db.relationship('Preprocessing', backref='dataset', lazy=True)
    # Hapus relasi ke DataPakar karena tidak ada FK

    def __repr__(self):
        return f'<Dataset {self.id}'

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

class Preprocessing(db.Model):
    __tablename__ = 'preprocessing'
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=True)
    username = db.Column(db.String(255), nullable=True) 
    full_text = db.Column(db.Text, nullable=True)
    text_clean = db.Column(db.Text, nullable=True)
    text_baku = db.Column(db.Text, nullable=True)  # Hasil pembakuan kata (baru)
    text_stopwords = db.Column(db.Text, nullable=True)
    text_stem = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.String(255), nullable=True)
    label_otomatis = db.Column(db.String(50), nullable=True)
    klasifikasi_nbs = db.relationship('KlasifikasiNB', backref='preprocessing', lazy=True)
    klasifikasi_svms = db.relationship('KlasifikasiSVM', backref='preprocessing', lazy=True)

    def __repr__(self):
        return f'<Preprocessing {self.id}>'

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

class DataPakar(db.Model):
    __tablename__ = 'data_pakar'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=True)
    full_text = db.Column(db.Text, nullable=False)
    text_clean = db.Column(db.Text, nullable=True)
    text_baku = db.Column(db.Text, nullable=True)
    text_stopwords = db.Column(db.Text, nullable=True)
    text_stem = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.String(255), nullable=True)
    label = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f'<DataPakar {self.id}: {self.label}>'

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

# --- PERBAIKAN ---
# 1. Hapus baris 'from app.models import ...' yang menyebabkan circular import.
# 2. Perbaiki variabel __all__ agar mencakup semua model yang relevan dan hapus 'Klasifikasi' yang tidak ada.
__all__ = [
    'DataPakar', 'Dataset', 'Preprocessing', 
    'KlasifikasiNB', 'KlasifikasiSVM', 'DataSplit', 'ComparisonHistory'
]
# --- AKHIR PERBAIKAN ---

class KlasifikasiNB(db.Model):
    __tablename__ = 'klasifikasi_nb'
    id = db.Column(db.Integer, primary_key=True)
    preprocessing_id = db.Column(db.Integer, db.ForeignKey('preprocessing.id'), nullable=True)
    split_id = db.Column(db.Integer, db.ForeignKey('data_split.id'), nullable=True)
    data_pakar_id = db.Column(db.Integer, db.ForeignKey('data_pakar.id'), nullable=True)  # FK baru
    data_pakar = db.relationship('DataPakar', backref='klasifikasi_nbs', lazy=True)
    username = db.Column(db.String(255), nullable=True)
    full_text = db.Column(db.Text)
    text_stem = db.Column(db.Text)
    label_otomatis = db.Column(db.String(50))
    label_pakar = db.Column(db.String(50), nullable=True)
    label_prediksi = db.Column(db.String(50))
    model_name = db.Column(db.String(50), nullable=False, index=True)
    test_ratio = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<KlasifikasiNB {self.id} - Model: {self.model_name}>'

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

class KlasifikasiSVM(db.Model):
    __tablename__ = 'klasifikasi_svm'
    id = db.Column(db.Integer, primary_key=True)
    preprocessing_id = db.Column(db.Integer, db.ForeignKey('preprocessing.id'), nullable=True)
    split_id = db.Column(db.Integer, db.ForeignKey('data_split.id'), nullable=True)
    data_pakar_id = db.Column(db.Integer, db.ForeignKey('data_pakar.id'), nullable=True)  # FK baru
    data_pakar = db.relationship('DataPakar', backref='klasifikasi_svms', lazy=True)
    username = db.Column(db.String(255), nullable=True)
    full_text = db.Column(db.Text)
    text_stem = db.Column(db.Text)
    label_otomatis = db.Column(db.String(50))
    label_pakar = db.Column(db.String(50), nullable=True)
    label_prediksi = db.Column(db.String(50))
    model_name = db.Column(db.String(50), nullable=False, index=True)
    test_ratio = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<KlasifikasiSVM {self.id} - Model: {self.model_name}>'

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

class DataSplit(db.Model):
    __tablename__ = 'data_split'
    id = db.Column(db.Integer, primary_key=True)
    test_ratio = db.Column(db.Float, nullable=False)
    test_size = db.Column(db.Integer, nullable=False)
    train_size = db.Column(db.Integer, nullable=False)
    test_indices = db.Column(db.Text, nullable=False)
    train_indices = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    label_source = db.Column(db.String(50), default='otomatis')
    x_train_data = db.Column(db.PickleType, nullable=True)
    x_test_data = db.Column(db.PickleType, nullable=True)
    y_train_data = db.Column(db.PickleType, nullable=True)
    y_test_data = db.Column(db.PickleType, nullable=True)
    data_pakar_ids = db.Column(db.Text, nullable=True)  # Menggantikan data_pakar_id
    preprocessing_ids = db.Column(db.Text, nullable=True)  # Simpan list ID preprocessing (JSON)
    klasifikasi_nbs = db.relationship('KlasifikasiNB', backref='split', lazy=True)
    klasifikasi_svms = db.relationship('KlasifikasiSVM', backref='split', lazy=True)
    comparison_histories = db.relationship('ComparisonHistory', backref='split', lazy=True)

    def save_split_data(self, X_train, X_test, y_train, y_test):
        """Menyimpan data split langsung ke database"""
        import pickle
        
        self.x_train_data = pickle.dumps(X_train)
        self.x_test_data = pickle.dumps(X_test)
        self.y_train_data = pickle.dumps(y_train)
        self.y_test_data = pickle.dumps(y_test)
        
        from app import db
        db.session.commit()

    def get_split_data(self):
        """Mengambil data split dari database"""
        if not all([self.x_train_data, self.x_test_data, 
                    self.y_train_data, self.y_test_data]):
            return None, None, None, None
            
        import pickle
        
        try:
            X_train = pickle.loads(self.x_train_data)
            X_test = pickle.loads(self.x_test_data)
            y_train = pickle.loads(self.y_train_data)
            y_test = pickle.loads(self.y_test_data)
            return X_train, X_test, y_train, y_test
        except:
            return None, None, None, None

    def reset_all(self):
        """Reset pembagian data."""
        try:
            # Hapus instance dari database
            from app import db
            db.session.delete(self)
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            return False

    def __repr__(self):
        return f'<DataSplit {self.test_ratio}>'

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()

class ComparisonHistory(db.Model):
    __tablename__ = 'comparison_history'
    id = db.Column(db.Integer, primary_key=True)
    split_id = db.Column(db.Integer, db.ForeignKey('data_split.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=get_wib_time)
    accuracy_nb = db.Column(db.Float)
    nb_total_train = db.Column(db.Integer)
    nb_total_test = db.Column(db.Integer)
    nb_test_ratio = db.Column(db.Float)
    nb_vocab_size = db.Column(db.Integer)
    label_source = db.Column(db.String(50), nullable=True)
    accuracy_svm = db.Column(db.Float)
    svm_total_train = db.Column(db.Integer)
    svm_total_test = db.Column(db.Integer)
    svm_test_ratio = db.Column(db.Float)
    svm_learning_rate = db.Column(db.Float)
    svm_lambda = db.Column(db.Float)
    svm_vocab_size = db.Column(db.Integer) 

    def __repr__(self):
        return f'<ComparisonHistory {self.id} - NB Acc: {self.accuracy_nb}, SVM Acc: {self.accuracy_svm}>'

    def save(self):
        """Simpan instance ke database."""
        db.session.add(self)
        db.session.commit()

    def delete(self):
        """Hapus instance dari database."""
        db.session.delete(self)
        db.session.commit()

#SELECT * FROM preprocessing WHERE id='30'
#SELECT * FROM preprocessing WHERE text_clean LIKE '%kabur%' AND label_otomatis = 'negatif';

