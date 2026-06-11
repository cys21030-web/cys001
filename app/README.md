# Flask App Framework for ToF Gesture Classification

## Overview

This is a complete Flask web application framework for the ToF (Time-of-Flight) sensor-based gesture classification system. It provides a web interface for:

1. **Data Collection** - Capture and save ToF sensor snapshots with gesture labels
2. **Model Training** - Train neural network classifiers on collected data
3. **Real-time Inference** - Use trained models for live gesture recognition
4. **Performance Monitoring** - Track accuracy and generate reports

## Project Structure

```
app/
├── __init__.py              # Flask app factory
├── routes.py                # API and UI routes
├── utils.py                 # Helper functions
├── templates/               # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Dashboard
│   ├── collect.html        # Data collection interface
│   ├── train.html          # Model training interface
│   └── inference.html      # Inference interface
└── static/                  # Static assets
    ├── style.css           # Styling
    └── script.js           # Client-side JavaScript

data/                        # Data storage (auto-created)
├── snapshot/               # Collected data organized by label
│   ├── Normal/
│   ├── Upstairs/
│   └── Downstairs/
└── models/                 # Trained models and metadata
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Flask app:
```bash
python run_app.py
```

3. Open your browser to: `http://localhost:5000`

## Features

### 1. Dashboard (`/`)
- Overview of collected samples and trained models
- Quick access to main features
- Recent model performance display

### 2. Data Collection (`/collect`)
- Live 8x8 ToF sensor heatmap visualization
- 3D point cloud visualization (toggle on/off)
- Snapshot capture with label selection (Normal, Upstairs, Downstairs)
- Sample management and export
- **API Endpoints:**
  - `POST /api/snapshot/take` - Capture and save snapshot
  - `GET /api/snapshot/list` - List all collected snapshots
  - `GET /api/snapshot/view/<label>/<filename>` - View snapshot details
  - `POST /api/snapshot/export` - Export snapshots as zip

### 3. Model Training (`/train`)
- Data selection interface for training samples
- Hyperparameter configuration:
  - Epochs (default: 20)
  - Batch size (default: 16)
  - Learning rate (default: 0.001)
  - Validation split (default: 20%)
- Training progress monitoring with history visualization
- Model list and performance metrics
- **API Endpoints:**
  - `POST /api/train/start` - Start training
  - `GET /api/train/list` - List trained models
  - `GET /api/train/export/<model_id>` - Export model

### 4. Inference (`/inference`)
- Model selection dropdown
- Real-time prediction display with confidence
- Feedback mechanism (correct/incorrect)
- Live accuracy and confusion matrix updates
- Report generation
- **API Endpoints:**
  - `POST /api/inference/start` - Start inference session
  - `POST /api/inference/stop` - Stop inference
  - `POST /api/inference/predict` - Get next prediction
  - `POST /api/inference/feedback` - Submit feedback
  - `GET /api/inference/report` - Generate report

## File Format Specifications

### ToF Data Files (.dat)
- Format: Space-separated decimal values, 8 per line
- Contains 64 distance values (8x8 matrix)
- Each value represents distance in millimeters (0-30000)

### Snapshot Structure
```
data/snapshot/
├── Normal/
│   ├── tof_2026-06-04-10-15-41.dat
│   ├── tof_2026-06-04-10-16-26.dat
│   └── ...
├── Upstairs/
│   ├── tof_2026-06-04-10-20-15.dat
│   └── ...
└── Downstairs/
    ├── tof_2026-06-04-10-25-30.dat
    └── ...
```

### Model Storage
```
data/models/
├── model_2026-06-04-11-30-45.pth      # PyTorch model weights
└── model_2026-06-04-11-30-45.json     # Metadata
```

Model metadata JSON structure:
```json
{
  "timestamp": "2026-06-04-11-30-45",
  "model_file": "data/models/model_2026-06-04-11-30-45.pth",
  "epochs": 20,
  "batch_size": 16,
  "learning_rate": 0.001,
  "val_split": 0.2,
  "num_samples": 120,
  "history": {
    "loss": [0.5, 0.4, ...],
    "val_loss": [0.6, 0.5, ...]
  },
  "evaluation": {
    "accuracy": 0.95,
    "confusion_matrix": {
      "0": {"0": 30, "1": 1, "2": 0},
      "1": {"0": 0, "1": 29, "2": 1},
      "2": {"0": 1, "1": 0, "2": 29}
    }
  }
}
```

## API Response Formats

All API responses use JSON format:

### Success Response
```json
{
  "success": true,
  "data": {},
  "message": "Optional message"
}
```

### Error Response
```json
{
  "error": "Error description",
  "code": "ERROR_CODE"
}
```

## Client-Side Utilities (static/script.js)

Helpful functions for template implementations:

```javascript
// Show notification
showNotification("Message", "success", 3000)

// Make API calls
const data = await apiCall("/api/endpoint", {
  method: "POST",
  body: JSON.stringify({...})
})

// Draw 8x8 heatmap
drawHeatMap(data, "heat-map-container-id")

// Draw 3D point cloud
drawPointCloud(points, "canvas-id")
```

## Integration with ToFTrainer

The Flask app integrates with the PyTorch-based classifier:

```python
from ai.ToFTrainer import ToFTrainer, ToFSample, ToFLabels

# Load trainer
trainer = ToFTrainer()

# Fit with samples
trainer.fit(train_samples, epochs=20, batch_size=16, validation_samples=val_samples)

# Predict
predictions = trainer.predict(test_samples)

# Evaluate
metrics = trainer.evaluate(test_samples)

# Save/load
trainer.save_model(pathlib.Path("model.pth"))
trainer = ToFTrainer.load_model(pathlib.Path("model.pth"))
```

## Usage Workflow

### 1. Data Collection Phase
- Start the app
- Navigate to "Collect Data"
- Perform a gesture (Normal, Upstairs, or Downstairs)
- Click "Take Snapshot" to save data
- Repeat for all three gesture types (collect ~30-50 samples each)
- Export collected data for backup

### 2. Training Phase
- Go to "Train Model"
- Select samples to use for training (typically 80-90%)
- Adjust hyperparameters if needed
- Click "Start Training"
- Monitor training progress and history
- Model automatically saved with metadata

### 3. Inference Phase
- Go to "Inference"
- Select a trained model
- Click "Start Inference"
- Perform gestures in front of sensor
- System predicts gesture in real-time
- Provide feedback (correct/incorrect)
- Watch accuracy improve over time
- Generate report when done

## Configuration

### Environment Variables
- `FLASK_ENV`: Set to "development" or "production"
- `FLASK_DEBUG`: Set to 1 for debug mode

### Flask Config (app/__init__.py)
- `SECRET_KEY`: Session encryption key
- `MAX_CONTENT_LENGTH`: Maximum upload size
- `SNAPSHOT_DIR`: Directory for data snapshots
- `MODEL_DIR`: Directory for trained models

## Performance Considerations

- **Data Volume**: Each 8x8 sensor reading = 64 floats
- **Training**: 100 samples × 20 epochs ≈ 2-5 minutes on CPU
- **Inference**: Real-time (~50-100 predictions/sec)
- **Storage**: Model (~2MB), Snapshot (~100 bytes)

## Future Enhancements

- [ ] WebSocket support for real-time streaming
- [ ] Database backend for persistent data storage
- [ ] User authentication and multi-user support
- [ ] Advanced visualizations (3D plots, real-time charts)
- [ ] Model versioning and comparison
- [ ] Automated hyperparameter tuning
- [ ] Export to different formats (ONNX, TensorFlow Lite)
- [ ] Mobile app integration
- [ ] Cloud deployment support

## Troubleshooting

### Port Already in Use
```bash
# Use different port
python run_app.py --port 5001
```

### Debug Mode Issues
```bash
# Run with full debug output
python run_app.py --debug
```

### Data Not Saving
Check that `data/snapshot/` and `data/models/` directories exist and are writable.

## License

This project is part of CYS001 - ToF Gesture Classification System
