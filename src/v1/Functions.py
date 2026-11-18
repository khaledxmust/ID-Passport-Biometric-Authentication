"""Perspective correction and fixed-layout ID crops from the original project."""

import os
import cv2
import numpy as np
import pytesseract

if os.getenv('TESSERACT_CMD'):
    pytesseract.pytesseract.tesseract_cmd = os.environ['TESSERACT_CMD']


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth < 2 or maxHeight < 2:
        raise ValueError('Document mask produced a degenerate rectangle.')

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def imgMasking(image, polygon, normalized=True):
    """Rectify a segmented document while retaining the original color pixels."""
    points = np.asarray(polygon, dtype=np.float32).copy()
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError('Document mask has insufficient polygon points.')
    height, width = image.shape[:2]
    if normalized:
        points *= np.array([width, height], dtype=np.float32)
    rectangle = cv2.minAreaRect(points)
    warped = four_point_transform(image, cv2.boxPoints(rectangle))
    if warped.shape[0] > warped.shape[1]:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped


def frontID(image, langs):
    dim = image.shape
    mapping = {'fname': [int(dim[0]*0.24),int(dim[0]*0.36), int(dim[1]*0.80),dim[1]],
            'lname':[int(dim[0]*0.34),int(dim[0]*0.48), int(dim[1]*0.50),dim[1]],
            'fadrs':[int(dim[0]*0.46),int(dim[0]*0.58), int(dim[1]*0.50),dim[1]],
            'ladrs':[int(dim[0]*0.56),int(dim[0]*0.68), int(dim[1]*0.50),dim[1]],
            'numId':[int(dim[0]*0.72),int(dim[0]*0.92), int(dim[1]*0.40),int(dim[1]*1)]}
    # cv2_imshow(image)
    fname = ' '.join(pytesseract.image_to_string(image[mapping['fname'][0]:mapping['fname'][1], mapping['fname'][2]:mapping['fname'][3]], lang=langs[2], config='--oem 3 --psm 12').split())
    lname = ' '.join(pytesseract.image_to_string(image[mapping['lname'][0]:mapping['lname'][1], mapping['lname'][2]:mapping['lname'][3]], lang=langs[2], config='--oem 3 --psm 12').split())
    fadrs = ' '.join(pytesseract.image_to_string(image[mapping['fadrs'][0]:mapping['fadrs'][1], mapping['fadrs'][2]:mapping['fadrs'][3]], lang=langs[2], config='--oem 3 --psm 12').split()).replace('\u200f','').replace('\u200e','')
    ladrs = ' '.join(pytesseract.image_to_string(image[mapping['ladrs'][0]:mapping['ladrs'][1], mapping['ladrs'][2]:mapping['ladrs'][3]], lang=langs[2], config='--oem 3 --psm 12').split()).replace('\u200f','').replace('\u200e','')
    numId = ' '.join(pytesseract.image_to_string(image[mapping['numId'][0]:mapping['numId'][1], mapping['numId'][2]:mapping['numId'][3]], lang=langs[0], config='--oem 3 --psm 12').split()).replace(' ','').replace('\u200e','')
    if len(numId) == 14:
        return {'fname': fname, 'lname':lname, 'fadrs':fadrs, 'ladrs':ladrs, 'numId':numId, 'bdate':numId[1:7]}
    return {'fname': fname, 'lname':lname, 'fadrs':fadrs, 'ladrs':ladrs, 'numId':numId}

def backID(image, langs):
    dim = image.shape
    mapping = {'titl1': [int(dim[0]*0.1),int(dim[0]*0.22), int(dim[1]*0.35),int(dim[1]*0.83)],
            'titl2':[int(dim[0]*0.19),int(dim[0]*0.30), int(dim[1]*0.35),int(dim[1]*0.83)],
            'sexty':[int(dim[0]*0.28),int(dim[0]*0.38), int(dim[1]*0.70),int(dim[1]*0.83)],
            'relig':[int(dim[0]*0.28),int(dim[0]*0.38), int(dim[1]*0.50),int(dim[1]*0.7)],
            'marti':[int(dim[0]*0.28),int(dim[0]*0.38), int(dim[1]*0.30),int(dim[1]*0.45)],
            'exprd':[int(dim[0]*0.44),int(dim[0]*0.54), int(dim[1]*0.28),int(dim[1]*0.53)],
            'numId':[int(dim[0]*0.02),int(dim[0]*0.12), int(dim[1]*0.44),int(dim[1]*0.82)]}
    # cv2_imshow(image)
    titl1 = ' '.join(pytesseract.image_to_string(image[mapping['titl1'][0]:mapping['titl1'][1], mapping['titl1'][2]:mapping['titl1'][3]], lang=langs[2], config='--oem 3 --psm 12').split())
    titl2 = ' '.join(pytesseract.image_to_string(image[mapping['titl2'][0]:mapping['titl2'][1], mapping['titl2'][2]:mapping['titl2'][3]], lang=langs[2], config='--oem 3 --psm 12').split())
    sexty = ' '.join(pytesseract.image_to_string(image[mapping['sexty'][0]:mapping['sexty'][1], mapping['sexty'][2]:mapping['sexty'][3]], lang=langs[2], config='--oem 3 --psm 12').split()).replace('\u200f','-')
    relig = ' '.join(pytesseract.image_to_string(image[mapping['relig'][0]:mapping['relig'][1], mapping['relig'][2]:mapping['relig'][3]], lang=langs[2], config='--oem 3 --psm 12').split())
    marti = ' '.join(pytesseract.image_to_string(image[mapping['marti'][0]:mapping['marti'][1], mapping['marti'][2]:mapping['marti'][3]], lang=langs[2], config='--oem 3 --psm 12').split())
    exprd = ' '.join(pytesseract.image_to_string(image[mapping['exprd'][0]:mapping['exprd'][1], mapping['exprd'][2]:mapping['exprd'][3]], lang=langs[2], config='--oem 3 --psm 12').split())
    numId = ' '.join(pytesseract.image_to_string(image[mapping['numId'][0]:mapping['numId'][1], mapping['numId'][2]:mapping['numId'][3]], lang=langs[0], config='--oem 3 --psm 12').split()).replace(' ','')
    if len(numId) == 14:
        return {'titl1': titl1, 'titl2':titl2, 'sexty':sexty, 'relig':relig, 'marti':marti, 'exprd':exprd, 'numId':numId, 'bdate':numId[1:7]}
    return {'titl1': titl1, 'titl2':titl2, 'sexty':sexty, 'relig':relig, 'marti':marti, 'exprd':exprd, 'numId':numId}


