import random
import sys
from math import sqrt
from PIL import Image
from collections import defaultdict

def extract_feature(img):
    imageW = img.size[0]
    imageH = img.size[1]
    feature_vector = [0] * imageW * imageH
    for x in range(0, imageH):
        for y in range(0, imageW):
            offset = x * imageW + y
            rgb = img.getpixel((y, x))
            feature_vector[offset] = [x, y, rgb[0], rgb[1], rgb[2]]    
    return feature_vector

def output_image(img, new_centers, assignments):
    output = img
    pixels = output.load() 
    imageW = img.size[0]
    imageH = img.size[1]
    feature_vector = [0] * imageW * imageH
    for x in range(0, imageH):
        for y in range(0, imageW):
            offset = x * imageW + y
            center = new_centers[assignments[offset]]
            new_color = center[2:5]
            new_color_int = [int(n) for n in new_color]
            pixels[y, x] = tuple(new_color_int)

    output.show()
    outputImageFilename = str(sys.argv[3])
    output.save(outputImageFilename)

def update_centers(data_set, assignments):

    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.itervalues():
        centers.append(map(mean, zip(*points)))

    return centers

def mean(a):
    return sum(a) / len(a)

def update_assignments(points, centers):

    assignments = []
    for point in points:
        shortest = () 
        shortest_index = 0
        for i in xrange(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    dimensions = len(a)
    
    dim_sum = 0
    for dimension in xrange(dimensions):
        dim_sum += (a[dimension] - b[dimension]) ** 2

    return sqrt(dim_sum)

def imageSegmentation(dataset, k):
    random_points = random.sample(dataset, k)
    assignments = update_assignments(dataset, random_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = update_assignments(dataset, new_centers)

    output_image(img, new_centers, assignments)

# main
inputImageFilename = str(sys.argv[2])
K = int(sys.argv[1])
img = Image.open(inputImageFilename)
points = extract_feature(img)
imageSegmentation(points, K)