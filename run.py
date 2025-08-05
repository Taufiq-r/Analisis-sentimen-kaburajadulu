# /Tugas Akhir 1.3 NB + svm/run.py

import os
import sys

# Tambahkan direktori root proyek (C:\TEST) ke sys.path
# Ini memungkinkan kita mengimpor 'app' sebagai top-level package
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Jika 'app' ada di dalam direktori lain di dalam project_root, sesuaikan
# Misalnya, jika struktur adalah project_root/src/app, maka sys.path.insert(0, os.path.join(project_root, 'src'))

# Sekarang impor create_app dari paket app
from app import create_app

# Logika konfigurasi Anda
config_name = os.getenv('FLASK_CONFIG') or 'default'
# Jika Anda ingin selalu development saat menjalankan run.py secara langsung:
# config_name = 'development' 

app = create_app(config_name)

if __name__ == '__main__':
    # Ambil debug flag dari konfigurasi jika ada, default ke True untuk pengembangan
    app.run(debug=app.config.get('DEBUG', True), host='0.0.0.0', port=5000)

#C:\Analisis Sentimen>"C:/Analisis Sentimen/.venv/Scripts/python.exe" "c:/Analisis Sentimen/run.py" RUN PY
#C:\Analisis Sentimen V3.3>"C:/Analisis Sentimen V3.3/.venv/Scripts/python.exe" "c:/Analisis Sentimen V3.3/run.py #
#"C:/Tugas Akhir Taufiqu Rahman/App/Analisis Sentimen Final/.venv/Scripts/python.exe" "c:/Tugas Akhir Taufiqu Rahman/App/Analisis Sentimen Final/run.py"