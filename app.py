"""Project 4: OCR and MobileNet-SSD object-detection pipeline.

Example:
    python app.py --mode both --image test1.png
"""
from __future__ import annotations
import argparse
import shutil
import sys
from pathlib import Path
import cv2
import pytesseract
from pytesseract import Output

DEFAULT_TESSERACT_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
DEFAULT_PROTOTXT_PATH = Path("models/deploy.prototxt")
DEFAULT_MODEL_PATH = Path("models/mobilenet_iter_73000.caffemodel")
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

def configure_tesseract() -> None:
    """Use Tesseract from PATH, or its usual Windows installation location."""
    if shutil.which("tesseract"):
        return
    if DEFAULT_TESSERACT_PATH.exists():
        pytesseract.pytesseract.tesseract_cmd = str(DEFAULT_TESSERACT_PATH)
        return
    raise RuntimeError(
        "Tesseract was not found. Install it from https://github.com/UB-Mannheim/"
        "tesseract/wiki and add it to PATH, or update DEFAULT_TESSERACT_PATH."
    )

def preprocess(image: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """Convert an image to a clean binary image suitable for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

def read_text(
    binary_image: cv2.typing.MatLike, min_confidence: float, psm: int
) -> tuple[list[dict[str, int | float | str]], str]:
    """Return accepted words and formatted text, filtering weak OCR predictions."""
    data = pytesseract.image_to_data(
        binary_image,
        output_type=Output.DICT,
        config=f"--oem 3 --psm {psm}",
    )
    words: list[dict[str, int | float | str]] = []
    lines: dict[tuple[int, int, int], list[str]] = {}
    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0

        if not text or confidence < min_confidence:
            continue

        word = {
            "text": text,
            "confidence": confidence,
            "left": int(data["left"][index]),
            "top": int(data["top"][index]),
            "width": int(data["width"][index]),
            "height": int(data["height"][index]),
        }
        words.append(word)
        line_key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        lines.setdefault(line_key, []).append(text)
    return words, "\n".join(" ".join(line) for line in lines.values())

def annotate(image: cv2.typing.MatLike, words: list[dict[str, int | float | str]]) -> cv2.typing.MatLike:
    """Draw boxes and confidence scores to visually confirm OCR detections."""
    result = image.copy()
    for word in words:
        x, y = int(word["left"]), int(word["top"])
        width, height = int(word["width"]), int(word["height"])
        label = f'{word["text"]} ({float(word["confidence"]):.0f}%)'
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 180, 0), 2)
        cv2.putText(
            result,
            label,
            (x, max(20, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 120, 0),
            1,
            cv2.LINE_AA,
        )
    return result


def detect_objects(
    image: cv2.typing.MatLike,
    min_confidence: float,
    prototxt_path: Path,
    model_path: Path,
) -> list[dict[str, int | float | str]]:
    """Detect Pascal VOC objects with the pre-trained MobileNet-SSD model."""
    if not prototxt_path.is_file() or not model_path.is_file():
        raise RuntimeError(
            "MobileNet-SSD model files are missing. Expected "
            f"{prototxt_path} and {model_path}."
        )

    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=127.5,
    )
    net.setInput(blob)
    detections = net.forward()
    height, width = image.shape[:2]
    objects: list[dict[str, int | float | str]] = []

    for index in range(detections.shape[2]):
        confidence = float(detections[0, 0, index, 2]) * 100
        class_index = int(detections[0, 0, index, 1])
        if confidence < min_confidence or not 0 < class_index < len(VOC_CLASSES):
            continue

        x1, y1, x2, y2 = (
            detections[0, 0, index, 3:7] * [width, height, width, height]
        ).astype(int)
        objects.append(
            {
                "label": VOC_CLASSES[class_index],
                "confidence": confidence,
                "left": max(0, x1),
                "top": max(0, y1),
                "right": min(width - 1, x2),
                "bottom": min(height - 1, y2),
            }
        )
    return objects


def annotate_objects(
    image: cv2.typing.MatLike, objects: list[dict[str, int | float | str]]
) -> cv2.typing.MatLike:
    """Draw labelled bounding boxes around accepted object detections."""
    result = image.copy()
    for detected_object in objects:
        x1, y1 = int(detected_object["left"]), int(detected_object["top"])
        x2, y2 = int(detected_object["right"]), int(detected_object["bottom"])
        label = f'{detected_object["label"]} ({float(detected_object["confidence"]):.1f}%)'
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(
            result, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 100, 0), 2, cv2.LINE_AA,
        )
    return result

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR and MobileNet-SSD object detection.")
    parser.add_argument("--image", default="test.png", help="Input image path (default: test.png).")
    parser.add_argument(
        "--mode", choices=("ocr", "detect", "both"), default="both",
        help="Recognition mode: OCR, object detection, or both (default: both).",
    )
    parser.add_argument("--output", default="ocr_result.png", help="Annotated OCR output image path.")
    parser.add_argument(
        "--detection-output", default="detection_result.png",
        help="Annotated object-detection output image path.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=80.0,
        help="Minimum accepted confidence percentage (default: 80).",
    )
    parser.add_argument("--psm", type=int, default=11, help="Tesseract page segmentation mode.")
    parser.add_argument("--prototxt", default=str(DEFAULT_PROTOTXT_PATH), help="MobileNet-SSD prototxt path.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="MobileNet-SSD weights path.")
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    image_path = Path(args.image)
    if not image_path.is_file():
        sys.exit(f"Error: input image not found: {image_path}")

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            sys.exit(f"Error: unable to read image: {image_path}")
        if args.mode in ("ocr", "both"):
            configure_tesseract()
    except pytesseract.TesseractNotFoundError:
        sys.exit("Error: Tesseract is not installed or its path is incorrect.")
    except RuntimeError as error:
        sys.exit(f"Error: {error}")
    if args.mode in ("ocr", "both"):
        binary_image = preprocess(image)
        words, text = read_text(binary_image, args.min_confidence, args.psm)
        if not words:
            print(f"No OCR text met the {args.min_confidence:.0f}% confidence threshold.")
        else:
            if not cv2.imwrite(args.output, annotate(image, words)):
                sys.exit(f"Error: unable to save OCR output image: {args.output}")
            average_confidence = sum(float(word["confidence"]) for word in words) / len(words)
            print("===== VALIDATED OCR TEXT =====")
            print(text)
            print(f"Accepted words: {len(words)}")
            print(f"Average accepted confidence: {average_confidence:.2f}%")
            print(f"OCR visual confirmation saved to: {args.output}\n")

    if args.mode in ("detect", "both"):
        try:
            objects = detect_objects(
                image, args.min_confidence, Path(args.prototxt), Path(args.model)
            )
        except RuntimeError as error:
            sys.exit(f"Error: {error}")

        if not objects:
            print(f"No supported objects met the {args.min_confidence:.0f}% detection threshold.")
        else:
            if not cv2.imwrite(args.detection_output, annotate_objects(image, objects)):
                sys.exit(f"Error: unable to save detection output image: {args.detection_output}")
            average_confidence = sum(float(item["confidence"]) for item in objects) / len(objects)
            print("===== VALIDATED OBJECT DETECTIONS =====")
            for item in objects:
                print(f'{item["label"]}: {float(item["confidence"]):.2f}%')
            print(f"Accepted objects: {len(objects)}")
            print(f"Average accepted confidence: {average_confidence:.2f}%")
            print(f"Detection visual confirmation saved to: {args.detection_output}")
if __name__ == "__main__":
    main()
