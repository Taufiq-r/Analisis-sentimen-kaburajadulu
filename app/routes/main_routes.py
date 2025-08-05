from flask import Blueprint, render_template, current_app
from app.models import Dataset 

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
@main_bp.route('/dashboard')
def dashboard():
    total_data = 0
    try:
        total_data = Dataset.query.count()
    except Exception as e:
        if hasattr(current_app, 'logger') and current_app.logger is not None:
            current_app.logger.error(f"Error di dashboard saat query Dataset: {e}")
        else:
            print(f"Error di dashboard saat query Dataset: {e}") 
    return render_template('dashboard.html', title='Dashboard', total_data=total_data)
