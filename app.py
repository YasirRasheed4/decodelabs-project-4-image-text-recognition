import pytesseract
from PIL import Image

# Tell pytesseract where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image_path = "test.png"

try:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    print("\n===== EXTRACTED TEXT =====\n")
    print(text)

except FileNotFoundError:
    print("Error: test.png was not found.")
    print("Place an image named 'test.png' in the project folder.")
    import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image_path = "test.png"

try:
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)

    print("\n===== EXTRACTED TEXT =====\n")
    print(text)

except FileNotFoundError:
    print("Error: test.png was not found.")
    print("Place an image named 'test.png' in the project folder.")