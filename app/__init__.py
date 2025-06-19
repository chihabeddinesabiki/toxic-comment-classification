import os
from flask import Flask
from app.routes import main

def create_app():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    template_dir = os.path.join(base_dir, '..', 'templates')
    static_dir = os.path.join(base_dir, '..', 'static')

    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
    app.config['UPLOAD_FOLDER'] = os.path.join(base_dir, '..', 'data', 'uploads')

    app.register_blueprint(main)

    return app
