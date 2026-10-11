"""
LSTM Handwriting Recognition - Localhost Interactive Web Application Server.

Features:
- Multi-threaded HTTP server using Python built-in http.server.
- Pre-loads trained LSTM checkpoint models/lstm_handwriting_model.keras.
- Real-time handwriting canvas classification endpoint (/api/predict).
- Official EMNIST test split sample explorer (/api/random_test_sample, /api/test_sample).
- Comprehensive metrics endpoint (/api/metrics).
- Serves static assets from web/ and figures from outputs/figures/.
"""

import base64
import io
import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image, ImageOps

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    TEST_SAMPLES_TOTAL,
)
from src.predict import HandwritingPredictor


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server to prevent request blocking."""
    daemon_threads = True
    allow_reuse_address = True


# Global predictor and dataset cache
PREDICTOR: HandwritingPredictor = None
TEST_IMAGES = None
TEST_LABELS = None
CLASS_MAPPING = None


def initialize_backend():
    """Initializes model predictor and caches test data for fast interactive browsing."""
    global PREDICTOR, TEST_IMAGES, TEST_LABELS, CLASS_MAPPING
    print("[BACKEND] Initializing LSTM Handwriting Recognition Engine...")
    PREDICTOR = HandwritingPredictor()
    PREDICTOR._ensure_test_data_loaded()
    TEST_IMAGES = PREDICTOR._x_test
    TEST_LABELS = PREDICTOR._y_test
    CLASS_MAPPING = PREDICTOR.class_mapping
    print(f"[BACKEND] Ready: Model loaded ({PREDICTOR.model.count_params():,} params).")
    print(f"[BACKEND] Test dataset ready: {len(TEST_IMAGES):,} samples across {len(CLASS_MAPPING)} classes.")


def preprocess_canvas_image(image_bytes: bytes) -> Tuple[np.ndarray, str]:
    """
    Processes an image uploaded or drawn on the web canvas to match EMNIST format:
    1. Converts to grayscale.
    2. Identifies strokes (inverting if drawn dark-on-light).
    3. Crops tightly to the bounding box of the drawn character.
    4. Resizes to 20x20 maintaining aspect ratio.
    5. Centers inside a 28x28 black canvas with 4-pixel padding.
    6. Normalizes to float32 [0.0, 1.0].
    
    Returns:
        Tuple[np.ndarray, str]: Preprocessed (1, 28, 28) array and base64 PNG data URL of the 28x28 image.
    """
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")

    # Composite over black background to handle alpha channels
    bg = Image.new("RGBA", pil_img.size, (0, 0, 0, 255))
    composite = Image.alpha_composite(bg, pil_img).convert("L")
    img_arr = np.array(composite)

    # Check if image is drawn dark-on-light or light-on-dark
    # In EMNIST: background is black (0), stroke is white (255)
    mean_val = np.mean(img_arr)
    if mean_val > 127:  # White background with dark strokes
        img_arr = 255 - img_arr

    # Threshold to eliminate faint anti-aliasing noise
    img_arr[img_arr < 30] = 0

    # Find bounding box of non-zero pixels
    coords = np.argwhere(img_arr > 0)
    if coords.size == 0:
        # Blank canvas
        blank = np.zeros((1, 28, 28), dtype=np.float32)
        blank_pil = Image.fromarray(np.zeros((28, 28), dtype=np.uint8))
        buf = io.BytesIO()
        blank_pil.save(buf, format="PNG")
        b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        return blank, b64

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = Image.fromarray(img_arr[y0:y1, x0:x1])

    # Resize preserving aspect ratio into 20x20
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cropped.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Place centered into a 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)
    paste_x = (28 - new_w) // 2
    paste_y = (28 - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # Convert to float32 [0.0, 1.0] and shape (1, 28, 28)
    final_arr = np.array(canvas, dtype=np.float32) / 255.0
    tensor = np.expand_dims(final_arr, axis=0)

    # Export base64 preview of the exact 28x28 input fed into the LSTM
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    b64_preview = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    return tensor, b64_preview


def array_to_base64_png(img_arr: np.ndarray) -> str:
    """Converts a (28, 28) float32 array in [0.0, 1.0] to a base64 PNG data URL."""
    uint8_img = np.clip(img_arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8_img)
    # Scale up for clean visual inspection in browser
    scaled = pil_img.resize((140, 140), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    scaled.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class HandwritingAppHandler(SimpleHTTPRequestHandler):
    """HTTP request handler for the interactive web application."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PROJECT_ROOT / "web"), **kwargs)

    def _send_json(self, data: Dict, status_code: int = 200):
        """Helper to send JSON response."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        """CORS preflight handling."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        """Route GET requests for static files, figures, and REST API endpoints."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. API: Metrics and Model Summary
        if path == "/api/metrics":
            try:
                with open(METRICS_DIR / "test_evaluation.json", "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                with open(METRICS_DIR / "top_confusions.json", "r", encoding="utf-8") as f:
                    confusions_data = json.load(f)
                with open(METRICS_DIR / "class_performance_summary.json", "r", encoding="utf-8") as f:
                    perf_data = json.load(f)

                response = {
                    "model": {
                        "name": "LSTM Handwriting Classifier",
                        "total_params": PREDICTOR.model.count_params(),
                        "classes_count": NUM_CLASSES,
                        "time_steps": 28,
                        "features_per_step": 28,
                    },
                    "test_evaluation": eval_data.get("metrics", {}),
                    "strongest_classes": perf_data.get("strongest_classes", [])[:5],
                    "weakest_classes": perf_data.get("weakest_classes", [])[:5],
                    "top_confusions": confusions_data.get("top_confusions", [])[:5],
                }
                self._send_json(response)
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
            return

        # 2. API: Random Test Sample
        if path == "/api/random_test_sample":
            try:
                sample_idx = int(np.random.randint(0, TEST_SAMPLES_TOTAL))
                sample_img = TEST_IMAGES[sample_idx]
                b64_img = array_to_base64_png(sample_img)
                pred_result = PREDICTOR.predict_test_sample(sample_idx, top_k=5)

                response = {
                    "sample_index": sample_idx,
                    "ground_truth_index": pred_result["true_class"],
                    "ground_truth_character": pred_result["true_character"],
                    "predicted_index": pred_result["predicted_class"],
                    "predicted_character": pred_result["predicted_character"],
                    "confidence": round(pred_result["confidence"] * 100, 2),
                    "is_correct": pred_result["is_correct"],
                    "top_5": pred_result["top_k"],
                    "image_data": b64_img,
                    "scanline_intensities": [float(v) for v in np.mean(sample_img, axis=-1)],
                }
                self._send_json(response)
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
            return

        # 3. API: Specific Test Sample by Index
        if path == "/api/test_sample":
            try:
                idx_str = query.get("index", ["0"])[0]
                sample_idx = max(0, min(TEST_SAMPLES_TOTAL - 1, int(idx_str)))
                sample_img = TEST_IMAGES[sample_idx]
                b64_img = array_to_base64_png(sample_img)
                pred_result = PREDICTOR.predict_test_sample(sample_idx, top_k=5)

                response = {
                    "sample_index": sample_idx,
                    "ground_truth_index": pred_result["true_class"],
                    "ground_truth_character": pred_result["true_character"],
                    "predicted_index": pred_result["predicted_class"],
                    "predicted_character": pred_result["predicted_character"],
                    "confidence": round(pred_result["confidence"] * 100, 2),
                    "is_correct": pred_result["is_correct"],
                    "top_5": pred_result["top_k"],
                    "image_data": b64_img,
                    "scanline_intensities": [float(v) for v in np.mean(sample_img, axis=-1)],
                }
                self._send_json(response)
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
            return

        # 4. Static Figure Serving: /figures/<filename>
        if path.startswith("/figures/"):
            fig_filename = path.replace("/figures/", "")
            fig_path = FIGURES_DIR / fig_filename
            if fig_path.exists() and fig_path.is_file():
                with open(fig_path, "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_error(404, f"Figure {fig_filename} not found.")
                return

        # 5. Fallback to default static file handler for web/
        super().do_GET()

    def do_POST(self):
        """Route POST requests for drawing canvas inference."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                payload = json.loads(body.decode("utf-8"))

                data_url = payload.get("image", "")
                if "," in data_url:
                    header, b64_data = data_url.split(",", 1)
                else:
                    b64_data = data_url

                raw_bytes = base64.b64decode(b64_data)
                tensor_input, b64_preview = preprocess_canvas_image(raw_bytes)

                # Check if blank
                if np.max(tensor_input) == 0.0:
                    self._send_json({
                        "error": "Canvas is blank. Please draw a character before submitting.",
                        "is_blank": True,
                    })
                    return

                # Perform prediction
                pred_dict = PREDICTOR.predict_image(tensor_input[0], is_raw=False, top_k=5)
                time_steps_mean = [float(np.mean(row)) for row in tensor_input[0]]

                # Known orthographic confusion alert
                predicted_char = pred_dict["predicted_character"]
                ambiguity_note = None
                if predicted_char in ["1", "L", "I"]:
                    ambiguity_note = "Vertical stroke ambiguity: '1', 'L', and 'I' share similar single-stroke profiles."
                elif predicted_char in ["0", "O"]:
                    ambiguity_note = "Continuous oval loop ambiguity: '0' and 'O' are geometrically near-identical."
                elif predicted_char in ["F", "f"]:
                    ambiguity_note = "Case ambiguity: 'f' and 'F' share crossbar and ascender geometry under normalization."

                response = {
                    "predicted_character": predicted_char,
                    "class_index": pred_dict["predicted_class"],
                    "confidence": round(pred_dict["confidence"] * 100, 2),
                    "top_5": pred_dict["top_k"],
                    "preview_image": b64_preview,
                    "ambiguity_note": ambiguity_note,
                    "scanline_intensities": time_steps_mean,
                    "is_blank": False,
                }
                self._send_json(response)
            except Exception as e:
                self._send_json({"error": str(e)}, status_code=500)
            return

        self.send_error(404, "Endpoint not found.")


def run_server(port: int = 8000):
    """Starts the multi-threaded HTTP application server."""
    initialize_backend()
    web_dir = PROJECT_ROOT / "web"
    web_dir.mkdir(parents=True, exist_ok=True)

    server_address = ("127.0.0.1", port)
    httpd = ThreadedHTTPServer(server_address, HandwritingAppHandler)
    url = f"http://localhost:{port}/"
    print("=" * 70)
    print("LSTM HANDWRITING RECOGNITION - LOCALHOST WEB SERVER ACTIVE")
    print(f"URL: {url}")
    print("Serving interactive canvas, test sample explorer, and metrics dashboard.")
    print("Press Ctrl+C to terminate.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping server...")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LSTM Handwriting Web App on Localhost")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()
    run_server(port=args.port)
