# Project 4 Viva Questions and Answers

1. **What is the purpose of this project?**  
   To extract text with OCR and detect physical objects with MobileNet-SSD, accepting only predictions with at least 80% confidence.

2. **Which AI libraries and models are used?**  
   `pytesseract`, a Python wrapper around Google’s Tesseract OCR engine, and the pre-trained MobileNet-SSD model through OpenCV DNN.

3. **What is OCR?**  
   Optical Character Recognition is the process of converting text in images or scanned documents into machine-readable text.

4. **Why do you convert the image to grayscale?**  
   OCR primarily needs intensity differences between text and background. Grayscale removes unnecessary colour information and simplifies processing.

5. **Why use Gaussian blur?**  
   It reduces small noise and artifacts before thresholding, which can improve the separation of characters from the background.

6. **What is adaptive thresholding?**  
   It converts an image to black and white using a threshold calculated from each local area. It works well when illumination is uneven.

7. **What does the 80% confidence threshold mean?**  
   A word is included only when Tesseract estimates that it is at least 80% reliable. Lower-confidence predictions are rejected.

8. **What is PSM 11?**  
   Tesseract Page Segmentation Mode 11 is designed for sparse text, where text may appear in different parts of an image.

9. **How is the output visually confirmed?**  
   The program draws a green bounding box and a confidence label around every accepted word in `ocr_result.png`.

10. **What were the measured results on the test images?**  
    `test.png` produced 6 accepted words with a 93.00% average confidence. `test1.png` produced 5 accepted words with an 89.40% average confidence. Both passed the 80% threshold.

11. **What can reduce OCR accuracy?**  
    Blurry text, decorative fonts, low contrast, shadows, image rotation, and busy backgrounds.

12. **How could the project be improved?**  
    Add deskewing, image upscaling, language support, more advanced text detection, and better preprocessing chosen for each image type.

13. **What is MobileNet-SSD?**  
    It is a pre-trained deep-learning object detector. MobileNet is the efficient feature-extraction network, and SSD (Single Shot Detector) predicts object classes and bounding boxes in one forward pass.

14. **What was the object-detection result?**  
    The person illustration in `test1.png` was detected with 96.54% confidence, above the required 80% threshold.
