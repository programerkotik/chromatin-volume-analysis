from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
from sklearn.cluster import DBSCAN
import matplotlib.cm as cm
import time


def open_image(path):
    image = Image.open(path)
    return np.array(image)


def threshold(image, threshold_value):
    preimage = cv2.GaussianBlur(image, (15, 15), 0)
    ret, thresh = cv2.threshold(
        preimage, threshold_value, 255, cv2.THRESH_BINARY)
    return np.invert(thresh)


def find_chromatin(nuclei, chromatin, chromocenters):
    chromatin[nuclei == 0] = 0
    chromatin[chromocenters == 255] = 0
    return chromatin


def find_chromocentra(image, threshold_value, a):
    preimage = cv2.GaussianBlur(image, (15, 15), 0)
    ret, thresh = cv2.threshold(
        preimage, threshold_value, 255, cv2.THRESH_BINARY)
    thresh = np.invert(thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((15, 15)))
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area_sum = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        area_sum = area_sum + area

    filtered = []
    for contour in contours:
        if cv2.contourArea(contour) > (a*area_sum)/(len(contours)):
            filtered.append(contour)

    chromocenters = np.zeros(image.shape)
    cv2.drawContours(chromocenters, filtered, -1, 255, -1)
    return chromocenters


def find_nuclei(chromatin, nuclei_region):
    # apply closing morphology operation to get filled chromatin
    nuclei = cv2.morphologyEx(binorized, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200)))

    # find all objects contours
    contours, hierarchy = cv2.findContours(
        nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter out only objects within mean range of area (maybe better filter by circularity)
    area_sum = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        area_sum = area_sum + area
    filtered = []
    for contour in contours:
        if cv2.contourArea(contour) > area_sum/len(contours):
            filtered.append(contour)

    # get amount of nuclei
    NucleiNumber = len(filtered)

    # draw nuclei and their locations
    nuclei = np.zeros(nuclei.shape)
    cv2.drawContours(nuclei, filtered, -1, 255, -1)
    nuclei_contour = np.zeros(nuclei.shape)
    cv2.drawContours(nuclei_contour, filtered, -1, (255, 0, 0), nuclei_region)
    return nuclei, nuclei_contour, NucleiNumber


def find_nucleoli_using_dbscan(image, nuclei):
    nuclei = np.uint8(nuclei)
    # find all nuclei
    num_labels, labels = cv2.connectedComponents(nuclei)

    h, w = image.shape
    nucleoli = np.zeros((h, w), dtype=np.uint8)

    # find nucleoli inside each nuclei
    for i in range(1, num_labels):
        mask = np.where(labels == i, image, 0)
        preclustering = np.where((mask > 130) & (mask < 175), 1, 0)

        X = np.where(preclustering == 1)
        X = np.array(X).T

        clustering = DBSCAN(eps=14, min_samples=300).fit(X)
        labels1 = clustering.labels_
        unique_labels, unique_counts = np.unique(labels1, return_counts=True)
        L = len(unique_labels)

        unique_labels = unique_labels[1:]
        unique_counts = unique_counts[1:]

        if len(unique_labels) >= 1:
            max_cluster_id = np.argmax(unique_counts)
            biggest_cluster = unique_labels[max_cluster_id]

            mask = labels1 == biggest_cluster

            for x in X[mask]:
                x1, x2 = x
                nucleoli[x1, x2] = 255
    return nucleoli


def create_regions_map(nuclei, nuclei_contour, nucleoli_location):
    # make separate regions
    kernel = np.ones((11, 11))
    nuclei_location_dilated = cv2.dilate(nuclei_contour, kernel, iterations=5)

    # translate them to be separate r,g,b arrays
    r = nuclei_location_dilated == 255
    g = nucleoli_location == 255
    b = (nuclei == 255) & ~(r | g)

    map = np.dstack((r, g, b))
    return map


def discriminate_chromatin(image, map, chromatin, chromocenters):
    num_labels, labels = cv2.connectedComponents(chromatin)

    r = map[:, :, 0]
    g = map[:, :, 1]

    membrane = np.full(np.shape(r), False)
    nucleoli = np.full(np.shape(g), False)

    membrane_labels = np.unique(labels[r])
    nucleoli_labels = np.unique(labels[g])

    intrsc_lbls = np.intersect1d(membrane_labels, nucleoli_labels)

    mask = np.full(np.shape(r), False)
    for label in membrane_labels:
        if label == 0:
            continue
        mask |= labels == label
    membrane[mask] = True

    mask = np.full(np.shape(r), False)
    for label in nucleoli_labels:
        if label in intrsc_lbls:
            continue
        mask |= labels == label
    nucleoli[mask] = True

    nucleoplasm = (chromatin == 255) & ~(membrane | nucleoli)

    blank = np.zeros_like(membrane, np.uint8)
    blank[chromocenters == 255] = 255
    blank[membrane] = 125
    blank[nucleoli] = 80
    blank[nucleoplasm] = 20
    blank[0:120, :] = 0

    return blank


def create_color_map(image, stack, chromacenters):
    image = np.repeat(image[:, :, np.newaxis], 3, 2)
    if stack.dtype == bool:
        image[stack] = 255
    else:
        image[stack == 255] = 200

    image[chromocenters.astype(bool), 1] = 150
    image[chromocenters.astype(bool), 2] = 150
    return image


# workflow
paths = glob.glob('data\*.tif')

for i, path in enumerate(tqdm(paths)):

    image_name = path.split('\\')[1]
    n = image_name.split('-')[3].split('.')[0]
    n = int(n)
	
	image = open_image(path)

	binorized = threshold(image, 150)

	chromocenters = find_chromocentra(image, 30, 3)

	nuclei, nuclei_contour, number_of_nuclei = find_nuclei(binorized, 60)

	chromatin = find_chromatin(nuclei, binorized, chromocenters)

	nucleoli_location = find_nucleoli_using_dbscan(image, nuclei)

	map = create_regions_map(nuclei, nuclei_contour, nucleoli_location)

	complex_result = discriminate_chromatin(
		image, map, chromatin, chromocenters)

	Image.fromarray(complex_result).save(
		f'with_intensity_code\{image_name}')
