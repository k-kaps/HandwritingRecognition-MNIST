from tkinter import *
from tkinter import filedialog
import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from ttkthemes import themed_tk as theme
from PIL import ImageTk,Image
import time
from text_prediction_v2 import main
import cv2

def processfn():
    global file_name
    global img_file3
    global img_file4
    #label5 = ttk.Label(root, text = "Your Image is processing", font=("Segoe UI Semilight",10)).grid(row=3, column=7)
    img_copy = main(file_name)
    img = Image.fromarray(img_copy)
    img = img.resize((350,350),Image.ANTIALIAS)
    img_file3 = ImageTk.PhotoImage(img)
    label6 = Label(root,image = img_file3).grid(row=3, column=7)
    #label7 = Label(root, text = x() , font=("Segoe UI Semilight",10)).grid(row=3, column=7)

def openfn():
     print("Hello")
     global file_name
     global img_file
     global img_file2 
     file_name = filedialog.askopenfilename()
     image_handwriting = Image.open(file_name)
     image_handwriting2 = image_handwriting.resize((350,350),Image.ANTIALIAS)
     img_file = ImageTk.PhotoImage(image_handwriting2)
     label5 = Label(root,image = img_file).grid(row=3, column=2)
     
     
     

root = theme.ThemedTk()
root.get_themes()
root.set_theme("vista")
root.title("HANDRWRITING RECOGNIZER")
root.geometry("1000x650")
label1 = ttk.Label(root, text="Handwriting", font=("Segoe UI Semilight", 30), anchor="e").grid(sticky="E", row=0, column=2)
label1 = ttk.Label(root, text="Recognizer", font=("Segoe UI Semilight", 30), anchor ="w").grid(sticky="W", row=0, column=7)
label2 = ttk.Label(root, text="  ", font=("Segoe UI",30)).grid(row=1, column=3)
label3 = ttk.Label(root, text="     ", font=("Segoe UI",20)).grid(row=1, column=1)
button1 = ttk.Button(root, text="OPEN IMAGE", width=60, command=openfn).grid(row=2, column=2)
button2 = ttk.Button(root, text="CONVERT!", width=60, command=processfn).grid(row=2, column=7)
#label4 = ttk.Label(root, text="      ", font=("Segoe UI",20)).grid(row=2, column=5)
label4 = ttk.Label(root, text=" ", font=("Segoe UI",20)).grid(row=2, column=5)
root.mainloop()