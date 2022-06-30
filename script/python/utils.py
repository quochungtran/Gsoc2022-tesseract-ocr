import cv2
import pre_process
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def img_box(results, 
            img,
            name_image):
    """
    This function takes three argument as
    input. it draw boxes on text area detected
    by Tesseract. it also writes resulted image to
    your local disk so that you can view it.
    :param img: image
    :param results: dictionary
    :param name_image: name_image
    :return: None
    """

    im = img.copy()
    for i in range(0, len(results["text"])):

        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]

        text = results["text"][i]
        conf = int(results["conf"][i])

        if conf > 20:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(im, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        output_ocr_path = os.path.join('../../results-ocr', 'ocr_'+ name_image)
        cv2.imwrite(output_ocr_path, im)
        

    
def save_text_file(results) :

    parse_text = []
    word_list  = []
    last_word  = []

    for word in results['text']:
        if word != '': 
            word_list.append(word)
            last_word = word

    if (last_word != '' and word == '') or (word == results['text'][-1]): 
        parse_text.append(word_list)
        word_list = []
    
    
    with open('result_text.txt', 'w', newline="") as file:
        csv.writer(file, delimiter=" ").writerows(parse_text)

    return parse_text


def visualize(img):
    figure(figsize=(10, 30), 
           dpi=80)
    plt.imshow(img,cmap = 'gray')
    plt.show()
