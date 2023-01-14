import cv2
import numpy as np
import torch
import imutils
from imutils.contours import sort_contours
import torchvision.transforms as transforms
from PIL import Image
import sys
sys.path.insert(1, "C:/Users/karan/Software Projects/WARG_CV_Bootcamp")
from model_training import CNN
import os

model = CNN()
model.load_state_dict(torch.load("C:/Users/karan/Software Projects/OCR Project/MNIST-model.pth"))
model.eval()

def deNoiseIMG(image_path):
    img = cv2.imread(image_path)
    
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)

    edges = cv2.Canny(img, 30, 200)
    contour = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour = imutils.grab_contours(contour)
    contour = sort_contours(contour, method = "left-to-right")[0]

    img_chars, n = processingContours(img, contour)
    print(img_chars)
    boxes = [b[1] for b in img_chars]
    images = [im[0] for im in img_chars]
    img_chars = np.array([c[0] for c in img_chars], dtype="float32")

    print(boxes)
    for b in boxes:
        (x, y, w, h) = b
        print(x,y,w,h)
        img_copy = cv2.rectangle(img_copy, (x,y), (x+w, y+h), (255,0,0), 2) 
        
    
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]
    transform = transforms.ToTensor()
    for i in range(n):
        fname = "C:/Users/karan/Software Projects/OCR Project/cells/cell"+ str(i) + ".png"
        image = Image.open(fname)
        image = transform(image)
        image = torch.stack([image])
        prediction = model(image)
        pred_y = torch.max(prediction, 1)[1].data.numpy().squeeze()
        print(pred_y.item())
        
    cv2.imshow("ORIGINAL IMAGE", img_copy)
    cv2.imshow("PROCESSED IMAGE", img)
    cv2.waitKey(0)

def processingContours(img, contour):
    img_chars = []
    img_copy = img.copy()
    n = 0
    for c in contour:
        (x, y, w, h) = cv2.boundingRect(c)
        print(x, y, w, h)
        if (w >= 5 and w <=150) and (h >= 15 and h <= 150):
            roi = img_copy[y:y+h, x:x+w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            if tH > tW:
                imutils.resize(thresh, height = 16)
            if tW > tH:
                imutils.resize(thresh, width = 16)
            
            (tH, tW) = thresh.shape

            dX = int(max(0, 16 - tW) / 2.0)
            dY = int(max(0, 16 - tH) / 2.0)

            img = cv2.copyMakeBorder(img, top = dY, bottom = dY, left = dX, right = dX, borderType = cv2.BORDER_CONSTANT, value = (0, 0, 0))
            img = cv2.resize(img, (16, 16))
            
            os.chdir("C:/Users/karan/Software Projects/OCR Project/cells/")
            
            fname = "cell"+ str(n) + ".png"
            cv2.imwrite(fname, img)

            img_chars.append((img, (x, y, w, h)))
            n=n+1
            roi = None

    return img_chars, n


if __name__ == "__main__":
    image_path = input("Enter the name of the image: ")
    image_path = "C:/Users/karan/Software Projects/OCR Project/images/" +  image_path
    deNoiseIMG(image_path)