import cv2
from PIL import Image
import numpy


def OpenCV2PIL(name):
    img = cv2.imread(name)
    cv2.imshow("OpenCV", img)
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image.show()
    cv2.waitKey()
    return image


def PIL2OpenCV(name):
    # PIL.Image转换成OpenCV格式：
    image = Image.open(name)
    image.show()
    img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
    cv2.imshow("OpenCV", img)
    cv2.waitKey()
    return cv2

