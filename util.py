import cv2
import os

"""
Draws a rectangle around a ROI
"""
def draw_rectangle(img, x, y, w, h, rgb_color = (255, 0, 0), stroke_size = 2):
    if (x == 0 and y == 0):
        return
    
    rectangle_width = x + w
    rectangle_height = y + h
    cv2.rectangle(img, (x, y), (rectangle_width, rectangle_height), rgb_color, stroke_size)

"""
Renames all files in a folder as their numberic index in the folder
returns new directory structure
"""
def numberic_reorder_dir(dir_path):
    files = os.listdir(dir_path)
    
    if files[0].startswith('1') and files[-1].startswith(str(len(files))):
        return files
    
    for k, file in enumerate(files):
        abs_path = os.path.abspath(dir_path)
        file_extension = ''.join(('.',file.split('.')[-1]))
        new_file_name = str(k + 1) + file_extension
        os.rename(abs_path + os.sep + file,  abs_path + os.sep + new_file_name)
    
    return os.listdir(dir_path)
        
