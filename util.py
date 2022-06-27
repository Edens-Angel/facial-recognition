import cv2

def draw_rectangle(img, x, y, w, h, rgb_color = (255, 0, 0)):
    if (x == 0 and y == 0):
        return
    
    rectangle_width = x + w
    rectangle_height = y + h
    cv2.rectangle(img, (x, y), (rectangle_width, rectangle_height), rgb_color, 2)
