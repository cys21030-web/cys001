"""Flask routes for UI and API."""
import json
import pathlib
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, send_file, current_app
from threading import Thread

from ai.ToFTrainer import ToFTrainer, ToFSample, ToFLabels
from common.ToFData import ToFData
from common.ViewAngle import ViewAngle
from common.WorldCoord import WorldCoord


ui_bp = Blueprint("ui", __name__)
api_bp = Blueprint("api", __name__)

# Global state
sensor_data_buffer = None
is_recording = False
inference_results = []


# ==================== UI Routes ====================


@ui_bp.route("/")
def index():
    """Dashboard home page."""
    return render_template("index.html")


@ui_bp.route("/collect")
def collect():
    """Data collection interface."""
    return render_template("collect.html")


@ui_bp.route("/train")
def train():
    """Model training interface."""
    return render_template("train.html")


@ui_bp.route("/inference")
def inference():
    """Inference interface."""
    return render_template("inference.html")


# ==================== API Routes: Data Collection ====================


@api_bp.route("/sensor/stream", methods=["GET"])
def sensor_stream():
    """Stream sensor data to web client (Server-Sent Events)."""
    def generate():
        global sensor_data_buffer
        # In real implementation, would connect to actual sensor
        # For now, returns placeholder data
        yield f"data: {json.dumps({'status': 'streaming'})}\n\n"
    
    return generate(), 200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
    }


@api_bp.route("/snapshot/take", methods=["POST"])
def take_snapshot():
    """
    Capture a snapshot of current ToF data and save it.
    
    Expected JSON: {"label": "Normal" or "Upstairs" or "Downstairs"}
    """
    try:
        data = request.get_json()
        label_name = data.get("label", "").strip()
        
        # Validate label
        label_map = {"Normal": ToFLabels.Normal, "Upstairs": ToFLabels.Upstairs, "Downstairs": ToFLabels.Downstairs}
        if label_name not in label_map:
            return jsonify({"error": "Invalid label"}), 400
        
        label = label_map[label_name]
        
        # Create snapshot with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
        snapshot_dir = pathlib.Path(current_app.config["SNAPSHOT_DIR"]) / label_name
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw data file
        raw_file = snapshot_dir / f"tof_{timestamp}.dat"
        
        # TODO: In real implementation, get actual sensor data
        # For now, save placeholder
        raw_file.write_text("")
        
        return jsonify({
            "success": True,
            "timestamp": timestamp,
            "label": label_name,
            "file": str(raw_file)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/snapshot/list", methods=["GET"])
def list_snapshots():
    """List all collected snapshots by label."""
    try:
        snapshot_base = pathlib.Path(current_app.config["SNAPSHOT_DIR"])
        result = {}
        
        for label_dir in snapshot_base.iterdir():
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            files = sorted([f.name for f in label_dir.glob("tof_*.dat")])
            result[label_name] = files
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/snapshot/view/<label>/<filename>", methods=["GET"])
def view_snapshot(label: str, filename: str):
    """View snapshot details (raw matrix and point cloud)."""
    try:
        snapshot_dir = pathlib.Path(current_app.config["SNAPSHOT_DIR"]) / label
        dat_file = snapshot_dir / filename
        
        if not dat_file.exists():
            return jsonify({"error": "File not found"}), 404
        
        # In real implementation, load ToF data and compute visualizations
        return jsonify({
            "label": label,
            "filename": filename,
            "raw_matrix": None,  # TODO: Load and normalize to 0-1 for display
            "point_cloud": None  # TODO: Generate 3D points
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/snapshot/export", methods=["POST"])
def export_snapshots():
    """
    Export selected snapshots as zip.
    
    Expected JSON: {"snapshots": [{"label": "...", "file": "..."}]}
    """
    try:
        data = request.get_json()
        # TODO: Implement zip export
        return jsonify({"success": True, "message": "Export in progress"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== API Routes: Training ====================


@api_bp.route("/train/start", methods=["POST"])
def start_training():
    """
    Start model training.
    
    Expected JSON: {
        "snapshots": [{"label": "...", "file": "..."}],
        "epochs": 20,
        "batch_size": 16,
        "learning_rate": 0.001,
        "val_split": 0.2
    }
    """
    try:
        data = request.get_json()
        
        # Load snapshots
        snapshots = data.get("snapshots", [])
        epochs = data.get("epochs", 20)
        batch_size = data.get("batch_size", 16)
        learning_rate = data.get("learning_rate", 1e-3)
        val_split = data.get("val_split", 0.2)
        
        # TODO: Load ToF data from files and create ToFSample objects
        samples = []
        
        if not samples:
            return jsonify({"error": "No valid samples loaded"}), 400
        
        # Train in background thread
        def train_model():
            try:
                trainer = ToFTrainer()
                train_samples, val_samples = trainer.train_test_split(
                    samples, test_ratio=val_split, seed=42
                )
                history = trainer.fit(
                    train_samples,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    validation_samples=val_samples,
                    verbose=True
                )
                
                # Evaluate
                eval_result = trainer.evaluate(val_samples)
                
                # Save model
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                model_path = pathlib.Path(current_app.config["MODEL_DIR"]) / f"model_{timestamp}.pth"
                trainer.save_model(model_path)
                
                # Save metadata
                metadata = {
                    "timestamp": timestamp,
                    "model_file": str(model_path),
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "val_split": val_split,
                    "num_samples": len(samples),
                    "history": history,
                    "evaluation": eval_result
                }
                meta_path = model_path.with_suffix(".json")
                meta_path.write_text(json.dumps(metadata, indent=2))
                
            except Exception as e:
                print(f"Training error: {e}")
        
        thread = Thread(target=train_model, daemon=True)
        thread.start()
        
        return jsonify({"success": True, "message": "Training started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/train/list", methods=["GET"])
def list_models():
    """List all trained models."""
    try:
        model_dir = pathlib.Path(current_app.config["MODEL_DIR"])
        models = []
        
        for meta_file in sorted(model_dir.glob("model_*.json")):
            try:
                metadata = json.loads(meta_file.read_text())
                models.append(metadata)
            except:
                pass
        
        return jsonify({"models": models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/train/export/<model_id>", methods=["GET"])
def export_model(model_id: str):
    """Export trained model as zip with metadata."""
    try:
        # TODO: Implement
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== API Routes: Inference ====================


@api_bp.route("/inference/start", methods=["POST"])
def start_inference():
    """
    Start inference session with a selected model.
    
    Expected JSON: {"model_file": "path/to/model.pth"}
    """
    global is_recording, inference_results
    try:
        data = request.get_json()
        model_file = data.get("model_file")
        
        if not model_file:
            return jsonify({"error": "Model file not specified"}), 400
        
        # Load model
        model_path = pathlib.Path(model_file)
        if not model_path.exists():
            return jsonify({"error": "Model not found"}), 404
        
        # TODO: Load trainer from model file
        # trainer = ToFTrainer.load_model(model_path)
        
        is_recording = True
        inference_results = []
        
        return jsonify({"success": True, "message": "Inference started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/inference/stop", methods=["POST"])
def stop_inference():
    """Stop inference session."""
    global is_recording
    is_recording = False
    return jsonify({"success": True})


@api_bp.route("/inference/predict", methods=["POST"])
def predict():
    """
    Get prediction for next recorded data point.
    
    Returns: {"label": "Normal", "confidence": 0.95, "data": {...}}
    """
    global is_recording
    try:
        if not is_recording:
            return jsonify({"error": "Inference not active"}), 400
        
        # TODO: Get latest sensor data and predict
        # prediction = trainer.predict(...)
        
        return jsonify({
            "label": "Normal",
            "confidence": 0.95,
            "raw_matrix": None,
            "point_cloud": None
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/inference/feedback", methods=["POST"])
def feedback():
    """
    Accept or reject inference result.
    
    Expected JSON: {"result_id": "...", "correct": true/false}
    """
    global inference_results
    try:
        data = request.get_json()
        result_id = data.get("result_id")
        is_correct = data.get("correct", True)
        
        # Update results tracking
        inference_results.append({
            "id": result_id,
            "correct": is_correct,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update confusion matrix / accuracy
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/inference/report", methods=["GET"])
def generate_report():
    """Generate inference report as PDF."""
    global inference_results
    try:
        # Calculate metrics
        if not inference_results:
            return jsonify({"error": "No inference results"}), 400
        
        total = len(inference_results)
        correct = sum(1 for r in inference_results if r["correct"])
        accuracy = correct / total if total else 0.0
        
        # TODO: Generate PDF report
        report_path = pathlib.Path(current_app.config["MODEL_DIR"]) / "inference_report.pdf"
        
        return jsonify({
            "success": True,
            "total_inferences": total,
            "correct": correct,
            "accuracy": accuracy,
            "report_file": str(report_path)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
