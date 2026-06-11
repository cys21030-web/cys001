"""Flask app factory and initialization."""
import pathlib
from flask import Flask
from flask_cors import CORS


def create_app(config_path: str | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    
    # Configuration
    app.config["SECRET_KEY"] = "dev-secret-key-change-in-production"
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB max upload
    
    # Enable CORS for API endpoints
    CORS(app)
    
    # Create necessary directories
    data_dir = pathlib.Path(__file__).parent.parent / "data"
    snapshot_dir = data_dir / "snapshot"
    model_dir = data_dir / "models"
    
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    app.config["SNAPSHOT_DIR"] = str(snapshot_dir)
    app.config["MODEL_DIR"] = str(model_dir)
    
    # Register blueprints
    from app.routes import api_bp, ui_bp
    app.register_blueprint(ui_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    
    return app
