# Artificial Intelligence Project 4

## Title

Image Text Recognition and Object Detection Using OCR and MobileNet-SSD

## Objective

Build an AI recognition pipeline that extracts text and detects physical objects in an input image, removes low-confidence predictions, and visually confirms validated detections.

## Tools and Libraries

- Python
- Tesseract OCR through `pytesseract`
- OpenCV (`cv2`) for image preprocessing and visual annotations
- MobileNet-SSD, a pre-trained object-detection model loaded through OpenCV DNN
- NumPy (installed as an OpenCV dependency)

## Methodology

1. The program reads the input image with OpenCV.
2. It converts the RGB/BGR image to grayscale, reducing the image from three colour channels to one intensity channel.
3. Gaussian blur reduces small noise and minor visual artifacts.
4. Adaptive thresholding converts the grayscale image to a high-contrast black-and-white image. This improves character separation under uneven lighting.
5. Tesseract OCR reads the processed image using page segmentation mode 11, which is appropriate for sparse or separately positioned text.
6. The program obtains word-level OCR confidence scores with `pytesseract.image_to_data`.
7. Only words with a confidence score of at least 80% are accepted.
8. Green bounding boxes and confidence labels are drawn around accepted words, creating visual evidence of the final output.
9. In object-detection mode, the image is converted into a 300 × 300 DNN blob and passed to MobileNet-SSD.
10. The model returns a class label, confidence score, and normalized bounding-box coordinates for each detected object. Detections below 80% are rejected and accepted detections are drawn in blue.

## Input and Output

Input images: `test.png` (an EDUCARE ACADEMY poster) and `test1.png` (a quote image).

Validated output images: `ocr_result.png` and `ocr_result_test1.png`.

Validated text extracted from `test.png`:

```text
EDUCARE
LEARN
GROW e SUCCEED
PREP
```

Validated text extracted from `test1.png`:

```text
NO
“You have no Enemies”
```

## Results

The program was tested on two input images using the configured minimum confidence of 80%.

| Test image | Accepted words | Average accepted confidence | Visual output |
| --- | ---: | ---: | --- |
| `test.png` | 6 | 93.00% | `ocr_result.png` |
| `test1.png` | 5 | 89.40% | `ocr_result_test1.png` |

Both OCR images contain bounding boxes and confidence values, satisfying the required visual confirmation.

### Object-detection result

`test1.png` was also processed with MobileNet-SSD. The illustrated person was detected with **96.54%** confidence and saved as `detection_result_test1.png`.

## Validation Against Requirements

| Requirement | Evidence |
| --- | --- |
| Library integration | `pytesseract` is used for OCR and OpenCV DNN loads the pre-trained MobileNet-SSD model. |
| Preprocessing integrity | The pipeline performs grayscale conversion, Gaussian blur, and adaptive thresholding. |
| Accuracy benchmark | Predictions below 80% are filtered out. The tested output averaged 93.00%. |
| Visual confirmation | OCR outputs and `detection_result_test1.png` display labelled bounding boxes with confidence values. |

## How to Run

```powershell
.\venv\Scripts\python.exe app.py --image test.png --output ocr_result.png --min-confidence 80 --psm 11
```

Run both OCR and object detection together:

```powershell
.\venv\Scripts\python.exe app.py --mode both --image test1.png --output ocr_result_test1.png --detection-output detection_result_test1.png --min-confidence 80 --psm 11
```

Run only object detection:

```powershell
.\venv\Scripts\python.exe app.py --mode detect --image test1.png --detection-output detection_result_test1.png --min-confidence 80
```

Tesseract OCR must be installed. On Windows, the program automatically checks the usual installation location: `C:\Program Files\Tesseract-OCR\tesseract.exe`.

## Limitations and Future Work

Decorative fonts, low contrast, tilted images, and complicated backgrounds can lower OCR accuracy. Future improvements could include deskewing, text-region detection, image upscaling, and language selection.

## Conclusion

This project demonstrates a practical pre-trained AI recognition workflow. It converts an unstructured visual input into machine-readable text, validates predictions with an 80% confidence threshold, and produces an annotated visual result.
