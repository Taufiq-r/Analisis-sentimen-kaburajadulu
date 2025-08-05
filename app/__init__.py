# /Tugas Akhir 1.3 NB + svm/app/__init__.py

from flask import Flask, current_app # Tambahkan current_app untuk logging di __init__
from flask_sqlalchemy import SQLAlchemy
import os
import sys

# Tambahkan path root proyek ke sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from config import config, DevelopmentConfig
except ImportError as e:
    raise ImportError(f"Gagal mengimpor config.py. Pastikan Anda menjalankan aplikasi dari root proyek dan file config.py ada di root.\nDetail: {e}")


db = SQLAlchemy() 

def create_app(config_name='default'):
    app = Flask(__name__, 
                static_folder='static', 
                template_folder='templates',
                instance_relative_config=False) # Biasanya False jika config dari objek/file

    # Atur konfigurasi
    app.config.from_object(config.get(config_name, DevelopmentConfig))

    # Pastikan SECRET_KEY diatur untuk session
    if not app.config.get('SECRET_KEY') and not app.config.get('TESTING', False):
        app.secret_key = os.urandom(24) 
        app.logger.warning("PERINGATAN: SECRET_KEY tidak diatur di config, menggunakan nilai random sementara. Ini tidak aman untuk produksi!")
    elif app.config.get('SECRET_KEY'):
        app.secret_key = app.config['SECRET_KEY']


    # Inisialisasi direktori (UPLOAD, MODEL, KAMUS) dari config
    # Pastikan kunci ini ada di objek config Anda
   
    os.makedirs(app.config.get('MODEL_FOLDER_PATH', os.path.join(project_root, 'models')), exist_ok=True)
    os.makedirs(app.config.get('KAMUS_FOLDER_PATH', os.path.join(project_root, 'kamus')), exist_ok=True)


    db.init_app(app)

    # Impor dan daftarkan Blueprints
    from .routes.main_routes import main_bp
    from .routes.dataset_routes import dataset_bp
    from .routes.preprocessing_routes import preprocessing_bp
    from .routes.labeling_routes import labeling_bp
    from app.routes.kesimpulan_routes import kesimpulan_bp

    
    # Impor blueprint baru untuk klasifikasi
    from .routes.split_data import split_data_bp
    from .routes.nb_classification_routes import nb_classification_bp
    from .routes.svm_classification_routes import svm_classification_bp
    
    from .routes.utility_routes import utility_bp
    from .routes.comparison_routes import comparison_bp

    app.register_blueprint(main_bp) 
    app.register_blueprint(dataset_bp, url_prefix='/dataset')
    app.register_blueprint(preprocessing_bp, url_prefix='/preprocessing')
    app.register_blueprint(labeling_bp, url_prefix='/label')
    
    # Daftarkan blueprint baru
    app.register_blueprint(split_data_bp, url_prefix='/classify')
    app.register_blueprint(nb_classification_bp, url_prefix='/classify/naive_bayes') 
    app.register_blueprint(svm_classification_bp, url_prefix='/classify/svm')
    app.register_blueprint(comparison_bp, url_prefix='/compare')
    
    app.register_blueprint(utility_bp, url_prefix='/utils')
    app.register_blueprint(kesimpulan_bp)
    with app.app_context():
        db.create_all()

    return app