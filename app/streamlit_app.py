from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import av  # type: ignore
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer  # type: ignore
    LIVE_AVAILABLE = True
except Exception:
    av = None  # type: ignore
    VideoProcessorBase = object  # type: ignore
    WebRtcMode = None  # type: ignore
    webrtc_streamer = None  # type: ignore
    LIVE_AVAILABLE = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.compliance_rules import detect_violations
from src.detection import get_safe_device, load_yolo_model, parse_yolo_results, run_detection
from src.reporting import export_json, summarize_logs
from src.risk_scoring import calculate_risk_score, get_risk_level, load_risk_config
from src.visualization import draw_detections

APP_VERSION = "FORMAL_PRE_TRAINING_READY_V3"


DEFAULT_WEIGHTS_CANDIDATES = [
    "models/ppe_yolov8n_100e_reasonable_best.pt",
    "runs/train/ppe_yolov8n_100e_reasonable/weights/best.pt",
    "runs/train/ppe_yolov8n_100e_auto_batch/weights/best.pt",
    "runs/train/ppe_yolov8n_100e_fast/weights/best.pt",
    "runs/train/ppe_yolov8s_100e/weights/best.pt",
    "runs/detect/runs/train/test_ppe_yolov8n/weights/best.pt",
    "yolov8n.pt",
]



def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_default_weights() -> str:
    for candidate in DEFAULT_WEIGHTS_CANDIDATES:
        if resolve_path(candidate).exists():
            return candidate
    return DEFAULT_WEIGHTS_CANDIDATES[0]


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model_cached(weights_path: str):
    return load_yolo_model(weights_path)


@st.cache_resource
def load_risk_config_cached() -> dict[str, Any]:
    return load_risk_config()


def decode_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image.")

    return image


def to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_demo_face_boxes(image: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw face boxes for demo mode only. This is not a PPE detection."""
    annotated = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        return annotated, []

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    demo_detections: list[dict[str, Any]] = []

    for x, y, w, h in faces:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 40), 2)
        cv2.rectangle(
            annotated,
            (x1, max(0, y1 - 24)),
            (x1 + 115, y1),
            (255, 180, 40),
            -1,
        )
        cv2.putText(
            annotated,
            "Demo Face",
            (x1 + 4, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        demo_detections.append(
            {
                "class_id": None,
                "class_name": "Demo-Face",
                "confidence": None,
                "bbox_xyxy": [x1, y1, x2, y2],
                "note": "Demo face box only. Not a PPE safety detection.",
            }
        )

    return annotated, demo_detections


def add_manual_height_risk(
    violations: list[dict[str, Any]],
    risk_result: dict[str, Any],
    risk_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    violations = list(violations)
    risk_result = dict(risk_result)

    violations.append(
        {
            "type": "working_at_height",
            "severity": "high",
            "class_name": "Manual-Context",
            "message": "Scene marked as working at height or exposed edge. Fall protection review required.",
            "risk_weight": 80,
            "source": "manual_context_override",
        }
    )

    score = int(risk_result.get("risk_score", 0)) + 80
    risk_result["risk_score"] = score
    risk_result["risk_level"] = get_risk_level(score, risk_config)
    risk_result["contextual_factors"] = ["working_at_height"]

    return violations, risk_result


def analyze_image(
    image: np.ndarray,
    model,
    risk_config: dict[str, Any],
    conf: float,
    imgsz: int,
    device: str,
    enforce_ppe_rules: bool,
    manual_height_risk: bool,
    add_demo_face_if_empty: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    results = run_detection(
        model=model,
        source=image,
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=False,
        project=str(PROJECT_ROOT / "results"),
        name="streamlit_tmp",
    )

    detections = parse_yolo_results(results)

    if enforce_ppe_rules:
        compliance = detect_violations(detections)
        violations = compliance["violations"]
        risk_result = calculate_risk_score(detections, risk_config)

        if manual_height_risk:
            violations, risk_result = add_manual_height_risk(
                violations,
                risk_result,
                risk_config,
            )

        is_compliant = len(violations) == 0
        mode = "construction"
        annotated = draw_detections(image, detections)

    else:
        violations = []
        risk_result = {
            "risk_score": 0,
            "risk_level": "safe",
            "contributing_classes": [],
            "note": "PPE rules disabled because this is a demo/non-construction scene.",
        }
        is_compliant = True
        mode = "demo_non_construction"

        if add_demo_face_if_empty and not detections:
            annotated, detections = draw_demo_face_boxes(image)
        else:
            annotated = draw_detections(image, detections)

    payload = {
        "mode": mode,
        "detections": detections,
        "is_compliant": is_compliant,
        "violations": violations,
        "risk": risk_result,
    }

    return annotated, payload

# ---------------------------------------------------------------------
# UI HELPERS AND WEBSITE TABS
# ---------------------------------------------------------------------

if "LIVE_AVAILABLE" not in globals():
    LIVE_AVAILABLE = False


def build_prediction_human_report(payload: dict[str, Any]) -> str:
    mode = payload.get("mode", "unknown")
    detections = payload.get("detections", [])
    violations = payload.get("violations", [])
    risk = payload.get("risk", {})

    risk_score = risk.get("risk_score", 0)
    risk_level = str(risk.get("risk_level", "unknown")).upper()
    is_compliant = payload.get("is_compliant", False)

    mode_label = (
        "Construction Safety Inspection"
        if mode == "construction"
        else "Demo / Non-Construction Scene"
    )

    class_counts = Counter(
        str(item.get("class_name", "unknown"))
        for item in detections
        if isinstance(item, dict)
    )

    detection_lines = (
        "\n".join(f"- {name}: {count}" for name, count in class_counts.most_common())
        if class_counts
        else "- No objects were detected by the model."
    )

    if mode != "construction":
        interpretation = (
            "This image was analyzed in demo/non-construction mode. "
            "Detections may be shown, but PPE safety rules were not enforced "
            "to avoid false violations in normal rooms or non-site scenes."
        )
    elif violations:
        interpretation = (
            f"The system detected {len(violations)} safety violation(s). "
            f"The current risk level is **{risk_level}** with a score of **{risk_score}**."
        )
    else:
        interpretation = (
            "No safety violations were detected by the current model and rules. "
            "This does not guarantee full safety, especially for hazards outside the trained classes."
        )

    if violations:
        violation_text = "\n".join(
            f"- **{str(v.get('severity', 'unknown')).upper()}** — "
            f"{v.get('type', 'unknown')}: {v.get('message', 'Safety issue detected.')}"
            for v in violations
        )
    else:
        violation_text = "- No safety violations were recorded."

    if mode != "construction":
        recommendations = [
            "Use Construction Safety Inspection mode only for real construction site images.",
            "Use Demo mode for webcam tests in a normal room.",
        ]
    elif violations:
        recommendations = [
            "Review the annotated image and verify each detected violation.",
            "Check whether the worker is missing required PPE.",
            "For height, edge, scaffold, or ladder hazards, enable the contextual high-risk option.",
        ]
    else:
        recommendations = [
            "Continue monitoring and collect more site images/videos for safety records.",
            "For high-risk scenes, manually enable the working-at-height/contextual risk option if needed.",
        ]

    recommendation_text = "\n".join(f"- {item}" for item in recommendations)

    return f"""# Safety Analysis Report

## Summary

- Analysis mode: **{mode_label}**
- Risk score: **{risk_score}**
- Risk level: **{risk_level}**
- Compliance status: **{'Compliant' if is_compliant else 'Not compliant'}**
- Total detections: **{len(detections)}**
- Total violations: **{len(violations)}**

## Interpretation

{interpretation}

## Detected Objects

{detection_lines}

## Safety Violations

{violation_text}

## Recommended Actions

{recommendation_text}

## Important Note

This report is generated automatically from computer vision predictions. It should support safety review, but it should not replace human inspection or official safety supervision.
"""


def build_video_human_report(summary: dict[str, Any]) -> str:
    processed_frames = summary.get("processed_frames", 0)
    total_detections = summary.get("total_detections", 0)
    total_violations = summary.get("total_violations", 0)
    average_risk = summary.get("average_risk_score", 0)
    output_video = summary.get("output_video", "N/A")

    interpretation = (
        f"The processed video contains {total_violations} safety violation(s). "
        "Review the output video and sampled frames."
        if total_violations > 0
        else "No safety violations were detected in the analyzed frames."
    )

    return f"""# Video Safety Analysis Report

## Summary

- Processed frames: **{processed_frames}**
- Total detections: **{total_detections}**
- Total violations: **{total_violations}**
- Average risk score: **{average_risk}**
- Output video: `{output_video}`

## Interpretation

{interpretation}

## Recommended Actions

- Review the processed output video.
- Reduce frame stride for more detailed analysis.
- Use construction mode only for real construction scenes.

## Important Note

Video analysis may skip frames depending on the selected frame stride. Lower stride means slower but more detailed analysis.
"""


def show_readable_report(payload: dict[str, Any], filename: str) -> None:
    if "records" in payload and "processed_frames" in payload:
        report_text = build_video_human_report(payload)
    else:
        report_text = build_prediction_human_report(payload)

    st.markdown(report_text)

    st.download_button(
        "Download readable report (.md)",
        data=report_text,
        file_name=filename.replace(".json", ".md"),
        mime="text/markdown",
    )

    with st.expander("Show raw JSON for developers"):
        raw_json = json.dumps(payload, indent=2, ensure_ascii=False)
        st.code(raw_json, language="json")
        st.download_button(
            "Download raw JSON",
            data=raw_json,
            file_name=filename,
            mime="application/json",
        )


def show_summary(payload: dict[str, Any]) -> None:
    risk = payload["risk"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Score", risk.get("risk_score", 0))
    c2.metric("Risk Level", str(risk.get("risk_level", "unknown")).upper())
    c3.metric("Compliant", "Yes" if payload["is_compliant"] else "No")
    c4.metric("Mode", "Construction" if payload["mode"] == "construction" else "Demo")


def show_detections(detections: list[dict[str, Any]]) -> None:
    if not detections:
        st.info("No detections found.")
        return

    rows = [
        {
            "class_id": det.get("class_id"),
            "class_name": det.get("class_name"),
            "confidence": det.get("confidence"),
            "bbox_xyxy": det.get("bbox_xyxy"),
            "worker_id": det.get("worker_id"),
        }
        for det in detections
    ]

    st.dataframe(pd.DataFrame(rows), width='stretch')


def show_violations(violations: list[dict[str, Any]]) -> None:
    if not violations:
        st.success("No safety violations detected.")
        return

    for violation in violations:
        severity = str(violation.get("severity", "unknown")).upper()
        vtype = violation.get("type", "unknown")
        message = violation.get("message", "")

        if severity in {"HIGH", "CRITICAL"}:
            st.warning(f"{severity} — {vtype}: {message}")
        else:
            st.info(f"{severity} — {vtype}: {message}")


def sidebar_settings() -> dict[str, Any]:
    st.sidebar.header("Settings")

    weights_path = st.sidebar.text_input("Weights path", find_default_weights())
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.45, 0.05)
    imgsz = st.sidebar.selectbox("Image size", [320, 416, 512, 640, 768], index=2)
    device = st.sidebar.text_input("Device", "auto")
    resolved_device = get_safe_device(device)
    st.sidebar.caption(f"Resolved device: {resolved_device}")

    st.sidebar.divider()
    st.sidebar.header("Analysis Mode")

    mode = st.sidebar.radio(
        "Mode",
        [
            "Construction safety inspection",
            "Demo / normal room / not a construction site",
        ],
        index=0,
    )

    enforce_ppe_rules = mode == "Construction safety inspection"

    manual_height_risk = False
    if enforce_ppe_rules:
        manual_height_risk = st.sidebar.checkbox(
            "Working at height / exposed edge risk",
            value=False,
        )
    else:
        st.sidebar.info("PPE violations are disabled in demo mode.")

    return {
        "weights_path": resolve_path(weights_path),
        "conf": conf,
        "imgsz": imgsz,
        "device": resolved_device,
        "enforce_ppe_rules": enforce_ppe_rules,
        "manual_height_risk": manual_height_risk,
    }


def image_tab(settings: dict[str, Any], model, risk_config: dict[str, Any]) -> None:
    st.subheader("Image Upload Analysis")

    uploaded = st.file_uploader(
        "Upload construction image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_upload",
    )

    if not uploaded:
        st.info("Upload an image to analyze.")
        return

    image = decode_uploaded_image(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(to_rgb(image), caption="Original", width='stretch')

    annotated, payload = analyze_image(
        image=image,
        model=model,
        risk_config=risk_config,
        conf=settings["conf"],
        imgsz=settings["imgsz"],
        device=settings["device"],
        enforce_ppe_rules=settings["enforce_ppe_rules"],
        manual_height_risk=settings["manual_height_risk"],
    )

    with col2:
        st.image(to_rgb(annotated), caption="Annotated", width='stretch')

    st.subheader("Risk Summary")
    show_summary(payload)

    st.subheader("Detections")
    show_detections(payload["detections"])

    st.subheader("Violations")
    show_violations(payload["violations"])

    st.subheader("Readable Safety Report")
    show_readable_report(payload, "image_safety_report.json")


def camera_tab(settings: dict[str, Any], model, risk_config: dict[str, Any]) -> None:
    st.subheader("Camera Snapshot Analysis")

    st.info(
        "Camera Snapshot is demo/non-construction by default. "
        "Enable construction rules only if the camera is looking at a real site."
    )

    apply_construction_rules = st.checkbox(
        "Apply construction safety rules to this camera snapshot",
        value=False,
        key="camera_apply_construction_rules",
    )

    manual_height_risk = False
    if apply_construction_rules:
        manual_height_risk = st.checkbox(
            "Working at height / exposed edge risk for this snapshot",
            value=False,
            key="camera_height_risk",
        )

    captured = st.camera_input("Take one camera snapshot")

    if not captured:
        st.info("Take a snapshot to analyze.")
        return

    image = decode_uploaded_image(captured)

    col1, col2 = st.columns(2)

    with col1:
        st.image(to_rgb(image), caption="Snapshot", width='stretch')

    annotated, payload = analyze_image(
        image=image,
        model=model,
        risk_config=risk_config,
        conf=settings["conf"],
        imgsz=settings["imgsz"],
        device=settings["device"],
        enforce_ppe_rules=apply_construction_rules,
        manual_height_risk=manual_height_risk,
        add_demo_face_if_empty=not apply_construction_rules,
    )

    with col2:
        st.image(to_rgb(annotated), caption="Annotated", width='stretch')

    st.subheader("Risk Summary")
    show_summary(payload)

    if not apply_construction_rules:
        st.success(
            "Demo mode active: detections may appear, but PPE safety violations are disabled for this camera snapshot."
        )

    st.subheader("Detections")
    show_detections(payload["detections"])

    st.subheader("Violations")
    show_violations(payload["violations"])

    st.subheader("Readable Safety Report")
    show_readable_report(payload, "camera_snapshot_safety_report.json")

def save_uploaded_video(uploaded_file) -> Path:
    output_dir = PROJECT_ROOT / "results" / "uploaded_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(uploaded_file.name).suffix or ".mp4"
    path = output_dir / f"uploaded_video_{int(time.time())}{suffix}"
    path.write_bytes(uploaded_file.getvalue())

    return path


def process_uploaded_video(
    video_path: Path,
    model,
    risk_config: dict[str, Any],
    conf: float,
    imgsz: int,
    device: str,
    enforce_ppe_rules: bool,
    manual_height_risk: bool,
    frame_stride: int,
    max_frames: int,
) -> tuple[Path, dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir = PROJECT_ROOT / "results" / "website_videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"processed_video_{int(time.time())}.mp4"

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_index = 0
    processed_count = 0
    records: list[dict[str, Any]] = []
    last_annotated = None

    progress = st.progress(0)
    status = st.empty()

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_index += 1

        if frame_index % max(1, frame_stride) == 0:
            annotated, payload = analyze_image(
                image=frame,
                model=model,
                risk_config=risk_config,
                conf=conf,
                imgsz=imgsz,
                device=device,
                enforce_ppe_rules=enforce_ppe_rules,
                manual_height_risk=manual_height_risk,
            )
            last_annotated = annotated
            processed_count += 1
            records.append({"frame_index": frame_index, **payload})
        else:
            annotated = last_annotated if last_annotated is not None else frame

        writer.write(annotated)

        if max_frames > 0 and frame_index >= max_frames:
            break

        if max_frames > 0:
            progress.progress(min(1.0, frame_index / max_frames))

        status.write(f"Processed sampled frames: {processed_count} | Current frame: {frame_index}")

    cap.release()
    writer.release()
    progress.progress(1.0)

    total_detections = sum(len(r.get("detections", [])) for r in records)
    total_violations = sum(len(r.get("violations", [])) for r in records)

    avg_risk = (
        sum(float(r.get("risk", {}).get("risk_score", 0)) for r in records) / len(records)
        if records
        else 0
    )

    return output_path, {
        "video": str(video_path),
        "output_video": str(output_path),
        "processed_frames": processed_count,
        "total_detections": total_detections,
        "total_violations": total_violations,
        "average_risk_score": round(avg_risk, 2),
        "records": records,
    }


def video_tab(settings: dict[str, Any], model, risk_config: dict[str, Any]) -> None:
    st.subheader("Video Upload Analysis")

    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    frame_stride = st.slider("Analyze every N frames", 1, 30, 10)
    max_frames = st.number_input(
        "Maximum frames to read",
        min_value=0,
        max_value=5000,
        value=300,
        step=50,
    )

    if not uploaded_video:
        st.info("Upload a video to process it inside the website.")
        return

    st.video(uploaded_video)

    if not st.button("Process video"):
        return

    with st.spinner("Processing video..."):
        video_path = save_uploaded_video(uploaded_video)
        output_video, summary = process_uploaded_video(
            video_path=video_path,
            model=model,
            risk_config=risk_config,
            conf=settings["conf"],
            imgsz=settings["imgsz"],
            device=settings["device"],
            enforce_ppe_rules=settings["enforce_ppe_rules"],
            manual_height_risk=settings["manual_height_risk"],
            frame_stride=frame_stride,
            max_frames=int(max_frames),
        )

    st.success("Video processed successfully.")

    st.subheader("Processed Video")
    st.video(str(output_video))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Processed Frames", summary["processed_frames"])
    c2.metric("Total Detections", summary["total_detections"])
    c3.metric("Total Violations", summary["total_violations"])
    c4.metric("Average Risk", summary["average_risk_score"])

    st.subheader("Readable Video Report")
    show_readable_report(summary, "video_safety_report.json")


if LIVE_AVAILABLE:
    class WebsiteLiveProcessor(VideoProcessorBase):  # type: ignore[misc]
        def __init__(
            self,
            model,
            risk_config: dict[str, Any],
            conf: float,
            imgsz: int,
            device: str,
            enforce_ppe_rules: bool,
            manual_height_risk: bool,
            process_every_n_frames: int,
        ) -> None:
            self.model = model
            self.risk_config = risk_config
            self.conf = conf
            self.imgsz = imgsz
            self.device = device
            self.enforce_ppe_rules = enforce_ppe_rules
            self.manual_height_risk = manual_height_risk
            self.process_every_n_frames = max(1, int(process_every_n_frames))
            self.frame_count = 0
            self.last_annotated = None
            self.last_payload: dict[str, Any] = {
                "risk": {"risk_score": 0, "risk_level": "safe"},
                "detections": [],
                "violations": [],
                "is_compliant": True,
            }

        def recv(self, frame: Any) -> Any:
            image_rgb = frame.to_ndarray(format="rgb24")
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            self.frame_count += 1

            if self.frame_count % self.process_every_n_frames == 0:
                annotated, payload = analyze_image(
                    image=image_bgr,
                    model=self.model,
                    risk_config=self.risk_config,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    enforce_ppe_rules=self.enforce_ppe_rules,
                    manual_height_risk=self.manual_height_risk,
                    add_demo_face_if_empty=not self.enforce_ppe_rules,
                )
                self.last_annotated = annotated
                self.last_payload = payload

            annotated = self.last_annotated.copy() if self.last_annotated is not None else image_bgr.copy()

            risk = self.last_payload.get("risk", {})
            overlay = (
                f"Mode: {'Construction' if self.enforce_ppe_rules else 'Demo'} | "
                f"Risk: {str(risk.get('risk_level', 'safe')).upper()} | "
                f"Score: {risk.get('risk_score', 0)}"
            )

            cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 40), (30, 30, 30), -1)
            cv2.putText(
                annotated,
                overlay,
                (12, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                format="rgb24",
            )

def live_tab(settings: dict[str, Any], model, risk_config: dict[str, Any]) -> None:
    st.subheader("Live Camera Analysis")

    if not LIVE_AVAILABLE:
        st.error(
            "Live Camera needs streamlit-webrtc and av. "
            "Install them with: python -m pip install streamlit-webrtc av"
        )
        return

    st.info("Live camera runs inside the same website. Demo mode is recommended for normal rooms.")

    apply_construction_rules = st.checkbox(
        "Apply construction safety rules to live camera",
        value=False,
        key="live_apply_construction_rules",
    )

    manual_height_risk = False

    if apply_construction_rules:
        manual_height_risk = st.checkbox(
            "Working at height / exposed edge risk for live camera",
            value=False,
            key="live_height_risk",
        )
    else:
        st.success("Live demo mode active: PPE violations are disabled.")

    process_every_n_frames = st.slider(
        "Run YOLO every N frames",
        1,
        20,
        5,
    )

    st.warning("Click START below and allow camera permission. Increase frame interval if it is slow.")

    webrtc_streamer(
        key="website-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: WebsiteLiveProcessor(
            model=model,
            risk_config=risk_config,
            conf=settings["conf"],
            imgsz=settings["imgsz"],
            device=settings["device"],
            enforce_ppe_rules=apply_construction_rules,
            manual_height_risk=manual_height_risk,
            process_every_n_frames=process_every_n_frames,
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )


def build_daily_report_text(summary: dict[str, Any]) -> str:
    total_items = summary.get("total_processed_items", 0)
    total_detections = summary.get("total_detections", 0)
    total_violations = summary.get("total_violations", 0)
    compliance = summary.get("compliance_percentage", 0)
    avg_risk = summary.get("average_risk_score", 0)
    most_common = summary.get("most_common_violation") or "No repeated violation detected"
    critical = summary.get("critical_alerts_count", 0)
    generated = summary.get("generated_at", "Unknown time")
    breakdown = summary.get("violation_breakdown", {})

    breakdown_text = (
        "\n".join(f"- {k}: {v}" for k, v in breakdown.items())
        if breakdown
        else "- No violations recorded."
    )

    return f"""# Daily Safety Report

## Executive Summary

Generated at: **{generated}**

## Key Results

- Total processed items: **{total_items}**
- Total detections: **{total_detections}**
- Total violations: **{total_violations}**
- Compliance percentage: **{compliance}%**
- Average risk score: **{avg_risk}**
- Critical alerts: **{critical}**
- Most common violation: **{most_common}**

## Violation Breakdown

{breakdown_text}

## Recommended Actions

- Review frames/images with violations.
- Confirm repeated issues manually.
- Use this report as a monitoring aid, not a replacement for human inspection.
"""


def reports_tab() -> None:
    st.subheader("Daily Safety Reports")

    log_dir = st.text_input("Logs directory", "results/logs")
    output_report = st.text_input("Report output path", "reports/daily_report_from_website.json")

    if st.button("Generate daily report"):
        try:
            summary = summarize_logs(log_dir)
            path = export_json(summary, output_report)
            st.success(f"Report generated and saved to: {path}")
            st.markdown(build_daily_report_text(summary))

            with st.expander("Show raw JSON for developers"):
                st.json(summary)

        except Exception as error:
            st.error(f"Failed to generate report: {error}")

    report_files = sorted((PROJECT_ROOT / "reports").glob("*.json"))

    if report_files:
        st.divider()
        selected = st.selectbox(
            "Select existing report",
            report_files,
            format_func=lambda p: p.name,
        )

        try:
            summary = json.loads(selected.read_text(encoding="utf-8"))
            st.markdown(build_daily_report_text(summary))

            with st.expander("Show raw JSON for developers"):
                st.json(summary)

        except Exception as error:
            st.error(f"Could not read report: {error}")


def samples_tab() -> None:
    st.subheader("Sample Predictions")

    sample_dir = PROJECT_ROOT / "assets" / "sample_predictions"
    images = sorted(sample_dir.glob("sample_*.jpg"))

    if not images:
        st.info("No sample predictions found. Run scripts/create_sample_predictions.py first.")
        return

    cols = st.columns(3)

    for index, image_path in enumerate(images[:12]):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        with cols[index % 3]:
            st.image(to_rgb(image), caption=image_path.name, width='stretch')

    json_path = sample_dir / "sample_predictions.json"

    if json_path.exists():
        with st.expander("Sample predictions raw JSON"):
            st.json(json.loads(json_path.read_text(encoding="utf-8")))


def main() -> None:
    st.set_page_config(
        page_title="Construction Safety Compliance CV System",
        page_icon="🦺",
        layout="wide",
    )

    st.title("Construction Safety Compliance CV System")
    st.caption(f"Stable unified website — {APP_VERSION}")
    st.caption("Image, camera snapshot, video upload, live camera, readable reports, and samples in one app.")

    settings = sidebar_settings()

    if not settings["weights_path"].exists():
        st.error(f"Weights file not found: {settings['weights_path']}")
        st.info("Train a model first or set the correct weights path in the sidebar.")
        return

    st.sidebar.success("Weights file found.")

    page = st.sidebar.radio(
        "Page",
        [
            "Image",
            "Camera Snapshot",
            "Video Upload",
            "Live Camera",
            "Reports",
            "Samples",
        ],
        index=0,
    )

    model = load_model_cached(str(settings["weights_path"]))
    risk_config = load_risk_config_cached()

    st.divider()

    if page == "Image":
        image_tab(settings, model, risk_config)

    elif page == "Camera Snapshot":
        camera_tab(settings, model, risk_config)

    elif page == "Video Upload":
        video_tab(settings, model, risk_config)

    elif page == "Live Camera":
        live_tab(settings, model, risk_config)

    elif page == "Reports":
        reports_tab()

    elif page == "Samples":
        samples_tab()


if __name__ == "__main__":
    main()
