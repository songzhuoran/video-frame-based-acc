import csv
import os,sys
import re
from PIL import Image, ImageDraw, ImageColor
import cv2
import numpy as np

DATA_DIR="/home/songzhuoran/benchmark/imagenet-detection/Data/VID/val/"

interval = 3

videofiles= os.listdir(DATA_DIR)
for videofile in videofiles:
    imagefiles = os.listdir(DATA_DIR+videofile)
    for imagefile in imagefiles:
