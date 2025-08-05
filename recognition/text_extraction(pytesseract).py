import cv2
import pytesseract
import matplotlib.pyplot as plt
#set your image path

image_path=r"C:\Users\Gajanan\Desktop\Shrutes\longtext.jpg"

#load the image and check if it exists
img= cv2.imread(image_path)
if img is None:
  print(f"Error:image not foundor unable to load at path :{image_path}")
  exit()

#preprocessing
def preprocess_image(image_path):
  img =cv2.imread(image_path)
  gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  blur= cv2.GaussianBlur(gray,(3,3),0)
  _, thresh= cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return thresh

preprocessed_img=preprocess_image(image_path)
plt.imshow(preprocessed_img, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()

#list the languages you want
languages='eng+hin+guj+fra+chi_sim+kor+tam'

#draw ocr bounding boxes
boxes= pytesseract.image_to_boxes(preprocessed_img, lang=languages)
img_copy =img.copy()
h,w,_ =img_copy.shape
for b  in boxes.splitlines():
  b=b.split(' ')
  x1,y1,x2,y2=int(b[1]),int(b[2]),int(b[3]),int(b[4])
  cv2.rectangle(img_copy,(x1, h-y1),(x2, h-y2),(0,255,0),2)

plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
plt.title("OCR Bounding Boxes")
plt.axis("off")
plt.show()

#text extraction
def ocr_image(preprocesses_image, languages ='eng+hin+guj+fra+chi_sim+kor+tam'):
  text=pytesseract.image_to_string(preprocessed_img, lang=languages, config='--psm 6')
  return text
languages = 'eng+hin+guj+fra+chi_sim+kor+tam'
text_result=ocr_image(preprocessed_img,languages=languages)
print("Extracted Text in multiple languages:\n")
print(text_result)