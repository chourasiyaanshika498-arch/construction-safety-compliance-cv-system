# 🏗️ Construction Safety Compliance CV System

An AI-powered computer vision application that helps monitor construction-site safety by detecting personal protective equipment (PPE), identifying potential safety violations, and generating risk assessments through an interactive Streamlit dashboard.

## 🚀 Overview

Construction sites require consistent safety monitoring to reduce workplace hazards and improve compliance with personal protective equipment requirements.

This project uses **YOLOv8, Python, OpenCV, and Streamlit** to analyze images and videos, detect safety-related objects, apply rule-based compliance checks, calculate risk scores, and generate readable safety reports.

## ✨ Key Features

- **PPE Detection:** Detect hardhats, safety vests, masks, gloves, and goggles.
- **Violation Detection:** Identify missing PPE using object-detection results and configurable safety rules.
- **Risk Assessment:** Calculate risk scores and categorize safety conditions.
- **Fall Detection:** Support fall-event detection and critical safety alerts.
- **Zone-Based Monitoring:** Evaluate safety conditions in configured work zones.
- **Worker Tracking:** Support worker-related tracking functionality.
- **Multiple Input Modes:** Analyze images, uploaded videos, and camera snapshots; live-camera functionality is also provided.
- **Interactive Dashboard:** Review detection results, compliance status, risk scores, and safety reports.
- **Model Training and Evaluation:** Includes scripts for training, evaluation, and project health checks.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| YOLOv8 (Ultralytics) | Object detection |
| PyTorch | Deep-learning inference and training |
| OpenCV | Image and video processing |
| Streamlit | Interactive web dashboard |
| Pandas and PyYAML | Data and configuration handling |
| Pytest | Automated testing |

## 🧠 Detection Classes

The model configuration defines 14 classes:

- Fall-Detected
- Gloves
- Goggles
- Hardhat
- Ladder
- Mask
- NO-Gloves
- NO-Goggles
- NO-Hardhat
- NO-Mask
- NO-Safety Vest
- Person
- Safety Cone
- Safety Vest

## 📊 Model Evaluation

The existing training summary reports the following results:

| Metric | Validation | Test |
|---|---:|---:|
| mAP@50 | 76.64% | 77.63% |
| mAP@50–95 | 49.49% | 49.93% |
| mAP@75 | 52.73% | 52.62% |

**Model configuration:** YOLOv8n, 512-pixel image size, 100 epochs.

These are the reported model evaluation results, not a guarantee of real-world safety performance. Detection quality may vary with lighting, camera angles, occlusion, and construction-site conditions.

## 📁 Project Structure

```text
construction-safety-compliance-cv-system/
├── app/
│   └── streamlit_app.py
├── assets/
│   ├── sample_images/
│   └── sample_predictions/
├── configs/
│   ├── ppe_yolov8.yaml
│   ├── risk_rules.yaml
│   └── zones.example.json
├── docs/
├── models/
├── notebooks/
├── scripts/
├── src/
│   ├── compliance_rules.py
│   ├── contextual_risk.py
│   ├── detection.py
│   ├── fall_alerts.py
│   ├── predict.py
│   ├── reporting.py
│   ├── risk_scoring.py
│   ├── video_processor.py
│   ├── visualization.py
│   ├── worker_tracking.py
│   └── zone_safety.py
├── tests/
├── requirements.txt
├── TRAINING_SUMMARY.md
├── README.md
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rajanish-8/construction-safety-compliance-cv-system.git
cd construction-safety-compliance-cv-system
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the appropriate execution-policy guidance for your environment, or activate the environment through your preferred Python tool.

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a compatible Python and PyTorch environment for your installed Ultralytics version.

### 4. Configure the trained model

The application requires compatible trained YOLO weights. The `.pt` model files are excluded from Git, so obtain the trained weights separately or train the model yourself.

Place the model at the path expected by the application, for example:

```text
models/ppe_yolov8n_100e_reasonable_best.pt
```

If your model is stored elsewhere, update the **Weights path** in the dashboard or the relevant model configuration.

### 5. Run the application

From the repository root:

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed in your terminal, normally:

```text
http://localhost:8501
```

## 🧪 Run Tests

Run the automated test suite:

```bash
python -m pytest
```

Run the project health check:

```bash
python scripts/project_healthcheck.py
```

The test suite includes coverage for compliance rules, risk scoring, contextual risk, and zone safety.

## 🏋️ Model Training

To train using the project's GPU training script in a compatible Bash environment:

```bash
bash scripts/train_reasonable_gpu.sh 0
```

Training requires the configured YOLO-format dataset and a compatible GPU environment. Review the dataset configuration and training script before running it.

## 🔒 Safety Disclaimer

This project is a **safety-assistance prototype**, not a certified safety system. It does not replace qualified safety officers, site inspections, official procedures, or emergency-response protocols. All detected hazards should be verified by responsible personnel.

## 🔮 Future Improvements

- Improve detection of underrepresented PPE classes.
- Evaluate performance across more construction environments.
- Enhance fall-event confirmation and worker tracking.
- Add database-backed safety history.
- Integrate notification and alert services.
- Support Docker-based deployment and an API backend.


## Screenshots

### Application Dashboard
![Application Dashboard](assets/screenshots/dashboard.png)

### PPE Detection Results
![PPE Detection Results](assets/screenshots/detection-result.png)

### Compliance Report
![Compliance Report](assets/screenshots/compliance-report.png)


## 👨‍💻 Author

**Rajanish Kumrawat**

GitHub: [@rajanish-8](https://github.com/rajanish-8)

---

*Built as an AI and computer vision project to support construction-site safety monitoring and compliance analysis.*
