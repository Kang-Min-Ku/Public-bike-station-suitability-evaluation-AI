import matplotlib.pyplot as plt
#import keras_ocr #I ran keras_ocr in docker env
import cv2
import math
import numpy as np

#delete text from image
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
#delete text from image
def delete_text(img_path, pipeline):

    img = keras_ocr.tools.read(img_path) 
    
    prediction_groups = pipeline.recognize([img])
    
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        #Define the line and inpaint
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        inpainted_img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
                 
    return(inpainted_img)
#왼쪽 상단에 국토지리정보원 마크 없애기 + 이미지 사이즈 맞추기
#현재 html 이미지 최대 사이즈는 (1024, 1024)
def crop(img, crop_size):
    assert len(crop_size) == 2
    crop_img = None
    center = [len_//2 for len_ in img.shape[:2]]
    crop_size_half = [len_//2 for len_ in crop_size]
    pad_row = 0
    pad_col = 0
    if crop_size[0] % 2 == 1:
        pad_row = 1
    if crop_size[1] % 2 == 1:
        pad_col = 1
    crop_img = img[center[0]-crop_size_half[0]:center[0]+crop_size_half[0]+pad_row,
                    center[1]-crop_size_half[1]:center[1]+crop_size_half[1]+pad_col]
    return crop_img       

def blur(img, kernel_size=(3,3), sigma=0.5):
    #kernel size가 클수록 이미지가 
    #sigma가 클수록 이미지가 흐려집니다
    img = cv2.GaussianBlur(img, kernel_size, sigma)
    return img

def circle_masking(img, loc, radius=20, color=(255,0,0), thickness=-1):
    #thickness가 -1이면 원 내부를 color로 채웁니다
    assert len(loc) == 2
    img = cv2.circle(img, loc, radius=radius, color=color, thickness=thickness)
    return img