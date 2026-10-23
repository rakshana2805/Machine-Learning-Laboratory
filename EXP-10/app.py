
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from backend.routes.predict import predict_bp

# ==========================================
# Project Directories
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
TEMPLATE_DIR = BACKEND_DIR / "templates"
STATIC_DIR = TEMPLATE_DIR / "static"

# ==========================================
# Flask Application
# ==========================================
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static"
)

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# ==========================================
# Register API Blueprint
# ==========================================
app.register_blueprint(predict_bp, url_prefix="/api")


# ==========================================
# Web Pages
# ==========================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/result")
def result():
    return render_template("result.html")


@app.route("/history")
def history():
    return render_template("history.html")


# ==========================================
# Error Handlers
# ==========================================
@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "success": False,
        "error": "Image size must not exceed 10 MB."
    }), 413


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "API endpoint not found."
        }), 404

    return "Page not found", 404


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.exception("Internal server error")

    if request.path.startswith("/api/"):
        return jsonify({
            "success": False,
            "error": "Internal server error. Check the Flask terminal for details."
        }), 500

    return "Internal server error. Check the Flask terminal for details.", 500


# ==========================================
# Run Application
# ==========================================
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )