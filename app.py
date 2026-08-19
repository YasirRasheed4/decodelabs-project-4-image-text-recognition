"""Project 4: OCR pipeline with preprocessing and confidence validation.

Example:
    python app.py --image test.png --output ocr_result.png
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

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR with preprocessing and confidence filtering.")
    parser.add_argument("--image", default="test.png", help="Input image path (default: test.png).")
    parser.add_argument("--output", default="ocr_result.png", help="Annotated output image path.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=95.0,
        help="Minimum accepted OCR confidence percentage (default: 95).",
    )
    parser.add_argument("--psm", type=int, default=11, help="Tesseract page segmentation mode.")
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    image_path = Path(args.image)
    if not image_path.is_file():
        sys.exit(f"Error: input image not found: {image_path}")

    try:
        configure_tesseract()
        image = cv2.imread(str(image_path))
        if image is None:
            sys.exit(f"Error: unable to read image: {image_path}")
        binary_image = preprocess(image)
        words, text = read_text(binary_image, args.min_confidence, args.psm)
    except pytesseract.TesseractNotFoundError:
        sys.exit("Error: Tesseract is not installed or its path is incorrect.")
    except RuntimeError as error:
        sys.exit(f"Error: {error}")
    if not words:
        sys.exit(
            f"No text met the {args.min_confidence:.0f}% confidence threshold. "
            "Try a clearer image or a different --psm value."
        )
    result = annotate(image, words)
    if not cv2.imwrite(args.output, result):
        sys.exit(f"Error: unable to save output image: {args.output}")
    average_confidence = sum(float(word["confidence"]) for word in words) / len(words)
    print("===== VALIDATED OCR TEXT =====")
    print(text)
    print("\n===== VALIDATION SUMMARY =====")
    print(f"Accepted words: {len(words)}")
    print(f"Minimum confidence: {args.min_confidence:.0f}%")
    print(f"Average accepted confidence: {average_confidence:.2f}%")
    print(f"Visual confirmation saved to: {args.output}")
if __name__ == "__main__":
    main()
