#! /bin/python2

import numpy
import cv2
import os
import struct

def get_tf(angle, center, offset):
    a_radian = numpy.radians(angle)
    c = numpy.cos(a_radian)
    s = numpy.sin(a_radian)
    tl = numpy.matrix([[1.0, 0.0, -center[0]], [0.0, 1.0, -center[1]], [0.0, 0.0, 1.0]])
    rot = numpy.matrix([[c, -s, 0.0 ], [s, c, 0.0], [0.0, 0.0, 1.0]])
    retl = numpy.matrix([[1.0, 0.0, (center[0] + offset[0])], [0.0, 1.0, (center[1] + offset[1])], [0.0, 0.0, 1.0]])
    return retl * rot * tl

BLACK = (0,)
WHITE = (255,)
DIR_OUT = "./img/"
SIZE_CANVAS = 50
SIZE_FEATURE = 28
SIZE_BLOCK = 32
NUM_IMG = 10000
DIGITS = tuple([chr(ord("0") + i) for i in range(10)] + [""])
FONTS = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, 
        cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)

if not os.path.isdir(DIR_OUT):
    os.mkdir(DIR_OUT)
os.system("rm -rf " + DIR_OUT + "*")

fl = open(DIR_OUT + "label", "wb")
ff = open(DIR_OUT + "feature", "wb")

# headers
fl.write(struct.pack(">i", 2049))
fl.write(struct.pack(">i", NUM_IMG))
ff.write(struct.pack(">i", 2051))
ff.write(struct.pack(">i", NUM_IMG))
ff.write(struct.pack(">i", SIZE_FEATURE))
ff.write(struct.pack(">i", SIZE_FEATURE))

canvas = numpy.ones((SIZE_CANVAS, SIZE_CANVAS), dtype = numpy.uint8) * 255
# cv2.imwrite(dir_img + "canvas.png", canvas)
for id_img in range(NUM_IMG):
    copy = numpy.copy(canvas)
    id_digit = numpy.random.randint(0, len(DIGITS))
    id_font = numpy.random.randint(0, len(FONTS))
    thickness = numpy.random.randint(1, 3)
    base_line = cv2.getTextSize(DIGITS[id_digit], FONTS[id_font], 1.0, thickness)[1] + 1
    scale_font = float(numpy.random.randint(40, 80)) / 100.0
    scale = float(SIZE_BLOCK) * 0.5 * scale_font / float(base_line)
    offset = float(SIZE_CANVAS) / 2.0 -  float(base_line) * scale
    cv2.putText(copy, DIGITS[id_digit], (0, 2 * base_line + 1), 
            FONTS[id_font], 1.0, BLACK, thickness)
    copy = cv2.warpAffine(copy, numpy.matrix([[scale, 0.0, offset], [0.0, scale, offset]]), 
            copy.shape, borderValue = WHITE) 
    # draw lines
    thickness_line = numpy.random.randint(1, 3)
    cv2.line(copy, (0, (SIZE_CANVAS - SIZE_BLOCK) / 2 - thickness_line), 
            (SIZE_CANVAS - 1, (SIZE_CANVAS - SIZE_BLOCK) / 2 - thickness_line), 
            BLACK, thickness_line)
    cv2.line(copy, (0, (SIZE_CANVAS + SIZE_BLOCK) / 2 + thickness_line), 
            (SIZE_CANVAS - 1, (SIZE_CANVAS + SIZE_BLOCK) / 2 + thickness_line), 
            BLACK, thickness_line)
    cv2.line(copy, ((SIZE_CANVAS - SIZE_BLOCK) / 2 - thickness_line, 0), 
            ((SIZE_CANVAS - SIZE_BLOCK) / 2 - thickness_line, SIZE_CANVAS - 1), 
            BLACK, thickness_line)
    cv2.line(copy, ((SIZE_CANVAS + SIZE_BLOCK) / 2 + thickness_line, 0), 
            ((SIZE_CANVAS + SIZE_BLOCK) / 2 + thickness_line, SIZE_CANVAS - 1), 
            BLACK, thickness_line)
    # rotation
    copy = cv2.warpAffine(copy, get_tf(float(numpy.random.randint(0,10)), (float(SIZE_CANVAS) / 2.0, float(SIZE_CANVAS) / 2.0),
        (numpy.random.randint(-5, 6), numpy.random.randint(-5, 6)))[0:2, :], 
        copy.shape, borderValue = WHITE) 

    copy = copy[(SIZE_CANVAS - SIZE_FEATURE) / 2:(SIZE_CANVAS + SIZE_FEATURE) / 2,
            (SIZE_CANVAS - SIZE_FEATURE) / 2:(SIZE_CANVAS + SIZE_FEATURE) / 2]

    copy[copy < 192] = 0
    copy[copy >= 192] = 255
    copy = copy.astype(numpy.uint8)
    ff.write(copy.data)
    fl.write(numpy.uint8(id_digit))
    if id_img % 1000 == 0:
        print id_img, NUM_IMG

fl.close()
ff.close()
