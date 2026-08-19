# DecodeLabs Project 4: Image and Text Recognition

An Artificial Intelligence project that demonstrates two recognition tasks:

- **Optical Character Recognition (OCR):** extracts readable text from images with Tesseract.
- **Object Detection:** detects Pascal VOC objects using a pre-trained MobileNet-SSD model.

The project follows the DecodeLabs Project 4 requirement to use pre-trained AI libraries/models, process sample inputs, validate predictions using an 80% confidence threshold, and generate visual confirmation of the output.

## Features

- Grayscale conversion, Gaussian blur, and adaptive thresholding for OCR preprocessing.
- Word-level OCR confidence filtering.
- Pre-trained MobileNet-SSD object detection through OpenCV DNN.
- 80% default minimum confidence threshold for both modes.
- Annotated output images with labelled bounding boxes and confidence scores.

## Requirements

- Python 3
- Tesseract OCR installed on Windows at:

  ```text
  C:\Program Files\Tesseract-OCR\tesseract.exe
  ```

- Python packages from `requirements.txt`

Install the Python packages:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Project

### Run OCR only

```powershell
.\venv\Scripts\python.exe app.py --mode ocr --image test.png --output ocr_result.png --min-confidence 80 --psm 11
```

### Run object detection only

```powershell
.\venv\Scripts\python.exe app.py --mode detect --image test1.png --detection-output detection_result_test1.png --min-confidence 80
```

### Run both OCR and object detection

```powershell
.\venv\Scripts\python.exe app.py --mode both --image test1.png --output ocr_result_test1.png --detection-output detection_result_test1.png --min-confidence 80 --psm 11
```

## Sample Results

| Test | Result |
| --- | --- |
| OCR on `test.png` | 6 validated words, 93.00% average confidence |
| OCR on `test1.png` | 5 validated words, 89.40% average confidence |
| Object detection on `test1.png` | Person detected with 96.54% confidence |

## Project Structure

```text
├── app.py                         # OCR and object-detection application
├── requirements.txt               # Python dependencies
├── models/
│   ├── deploy.prototxt            # MobileNet-SSD network definition
│   └── mobilenet_iter_73000.caffemodel  # Pre-trained model weights
├── test.png                       # OCR test image
├── test1.png                      # OCR and object-detection test image
├── ocr_result.png                 # OCR visual output
├── ocr_result_test1.png           # OCR visual output for test1
├── detection_result_test1.png     # Object-detection visual output
├── REPORT.md                      # Project report
└── VIVA.md                        # Viva questions and answers
```

## Output

The application prints validated predictions in the terminal and saves annotated images. Green boxes indicate OCR detections; blue boxes indicate detected objects.

## Model

Object detection uses the pre-trained [MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD) model, loaded locally from the `models` folder.
