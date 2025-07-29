import cv2
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def save_to_txt(text, filename="output.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Text saved to {filename}")

def process_image(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding (Otsu's method)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR
    text = pytesseract.image_to_string(thresh, config='--oem 3 --psm 6')
    
    print("Extracted Text:\n", text)

    save_to_txt(text)

    return text