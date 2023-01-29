import cv2
import numpy as np
import torch
import imutils
from imutils.contours import sort_contours
import torchvision.transforms as transforms
from PIL import Image
from model_training import CNN
import os

model = CNN()
model.load_state_dict(torch.load("./AZMNIST.pth"))
model.eval()

def predictText(img_copy, img, img_chars):
    boxes = [b[1] for b in img_chars]
    images = [im[0] for im in img_chars]

    labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames] 
    index = 0

    for image in images:
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        prediction = model(image)
        pred_index = torch.argmax(prediction, dim = 1)
        (x, y, w, h) = boxes[index]
        img_copy = cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img_copy, labelNames[pred_index], (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        index+=1


    return img_copy  

def processContours(img):
    edges = cv2.Canny(img, 30, 200)
    contour = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = imutils.grab_contours(contour)
    contour = sort_contours(contour, method = "left-to-right")[0]    
    img_chars = []
    img_copy = img.copy()
    for c in contour:
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= 5 and w <=150) and (h >= 15 and h <= 150):
            roi = img_copy[y:y+h, x:x+w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape
            if tH > tW:
                imutils.resize(thresh, height = 32)
            if tW > tH:
                imutils.resize(thresh, width = 32)
            
            (tH, tW) = thresh.shape

            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)
            
            img = cv2.copyMakeBorder(thresh, top = dY, bottom = dY, left = dX, right = dX, borderType = cv2.BORDER_CONSTANT, value = (0, 0, 0))
            img = cv2.resize(img, (32, 32))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=-1)

            img_chars.append((img, (x, y, w, h)))
            roi = None

    return img_chars

def processImage(path):
    img = cv2.imread(path)
    img_copy = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.GaussianBlur(img, (5,5), 0)
    return img, img_copy

def main(path):
    img, img_copy = processImage(path)
    img_chars = processContours(img)
    img_copy = predictText(img_copy, img, img_chars)
    return img_copy

if __name__ == '__main__':
    path = input("Enter the Image Path: ")
    main(path)