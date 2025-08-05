from paddleocr import PaddleOCR

# Step 1: Use Hindi model directly
ocr = PaddleOCR(lang='hi', det=True, rec=True, cls=False)

# Step 2: Test the Hindi image
result = ocr.ocr('test_images/hindi.jpg', det=True, rec=True, cls=False)

# Step 3: Print all detected lines
for line in result[0]:
    text, conf = line[1]
    print(f"{text}  (confidence: {conf:.2f})")
