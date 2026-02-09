import os

from flask import Flask
from flask_cors import CORS

from webApp.api.routes import api_bp

def create_app():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    template_dir = os.path.join(root_dir, "frontend", "templates")
    static_dir = os.path.join(root_dir, "frontend", "static")

    app = Flask(
        __name__,
        template_folder=template_dir,
        static_folder=static_dir,
        static_url_path="/static",
    )
    CORS(app)
    app.register_blueprint(api_bp)
    return app
