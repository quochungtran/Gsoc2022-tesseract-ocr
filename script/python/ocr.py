import argparse
import matplotlib.pyplot as plt
import cv2 
import pytesseract
import numpy as np
import os
from pytesseract import Output
from pre_process import *
from utils import *

ap = argparse.ArgumentParser()
ap.add_argument('--oem'            , '--oem'               , help = "OCR Engine mode")
ap.add_argument('--psm'            , '--psm'               , help = "Page Segmentation mode")
ap.add_argument('--dpi'            , '--dpi'               , help = "the resolution N in ")
ap.add_argument('--l'              , '--l'                 , help = "the language or script to use")
ap.add_argument('--img'            , '--img'               , help = "name image")
args = vars(ap.parse_args())

oem = str(args["oem"])
psm = str(args["psm"])
dpi = str(args["dpi"])
l = str(args["l"])
name_image = str(args["img"])

DATA_PATH = "../../data"
img_test_path = os.path.join(DATA_PATH, name_image)
img = {'image'    : cv2.imread(img_test_path),
       'name_img' : name_image}


# Preprocessing
is_processing  = True

if (is_processing  == True ):
        pre_process = PreProcessing_engine(img['image'])
        # Tesseract does this internally (Otsu algorithm)        
        output = Preprocessing_img_pipeline(img['image'], ['remove_shadow', 'gray'])
else : 
        output = img['image'].copy()


visualize(output)

# configuring parameters for tesseract
### oem : OCR Engine mode.
### psm : Page Segmentation mode 
### dpi : the resolution N in DPI 
### l   : the language or script to use 

custom_config = "--oem " + oem + "--psm " + psm + "--dpi " + dpi + " --l " + l
print(custom_config)


# feeding image to tesseract
results = pytesseract.image_to_data(output, 
                                    output_type=Output.DICT,
                                    config=custom_config,
                                    )

text = pytesseract.image_to_string(output, 
                                  config=custom_config)
print(text)

img_box(results, 
        img['image'], 
        img['name_img'])

# save image 
final_text = save_text_file(results)