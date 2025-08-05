import cv2
from paddleocr import PaddleOCR, draw_ocr

ocr_detector = PaddleOCR(det=True, rec=False, cls=False)  # only detection

def detect_text(img_path: str):
    result = ocr_detector.ocr(img_path, det=True, rec=False, cls=False)

    image = cv2.imread(img_path)
    boxes = [line[0] for line in result[0]]

    # Optionally visualize detection
    image_with_boxes = draw_ocr(image, boxes=boxes, txts=None, scores=None)
    cv2.imshow("Detected Text", image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxes

if __name__ == "__main__":
    img_path = r"dataset/IIIT5K/test/3_2.png"
    boxes = detect_text(img_path)
    print("Detected text regions:", boxes)
