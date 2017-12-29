import cv2
import numpy as np

image_show = True

# Finding contours with thresholding
def give_contours_with_th(threshold, original):

    # default thresholding I usualy fall back to
    # ret, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # trying with adaptive threshold and checking results. Did not get good results on few.
    # threshold = cv2.adaptiveThreshold(hue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # dilate and erode to reduce noise and rogue contours
    # kernel = np.ones((2, 2), np.uint8)
    # threshold = cv2.dilate(img, kernel, iterations=1)
    # threshold = cv2.erode(threshold, kernel, iterations=1)
    # threshold = cv2.medianBlur(threshold, 3)
    # threshold = cv2.bilateralFilter(threshold, 9, 75, 75)

    # Display threshold only
    if image_show:
        cv2.imshow('th', threshold)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image, contours, heirarchy = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    print(len(contours))
    cv2.drawContours(original, contours, -1, (0, 0, 255), 2)
    if image_show:
        cv2.namedWindow('contours', cv2.WINDOW_NORMAL)
        cv2.imshow('contours', original)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    return [threshold, contours]


# Hue separation and returning
def give_hue(image):

    #separating into hue, saturation and value channels
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    #Display hue only
    if image_show:
        cv2.imshow('hue', h)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return h

def canny(image):
    edged = cv2.Canny(image, 30, 200)
    # Display hue only
    if image_show:
        cv2.imshow('canny', edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# K-means color quantization
def give_kmeans_cq(image, K):
    #conertinng to euclidean
    z = image.reshape((-1, 3))
    z = np.float32(z)

    #setting kmeans criteria and findinf k means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10,1.0)
    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    #applying color quantization using kmeans data
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)

    #display kmeans cq result
    if image_show:
        cv2.imshow('res2', res2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return res2

def green_mask(image):
    # dilate and erode
    kernel = np.ones((2, 2), np.uint8)
    threshold = cv2.dilate(image, kernel, iterations=1)
    threshold = cv2.erode(image, kernel, iterations=1)
    threshold = cv2.medianBlur(image, 3)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = 30;
    lower_green = np.array([60 - sensitivity, 35, 10])
    upper_green = np.array([60 + sensitivity, 255, 255])
    mask = cv2.inRange(image, lower_green, upper_green)
    mask = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    # display kmeans cq result
    if image_show:
        cv2.imshow('after green mask', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image


def get_features(original):
    image = original.copy()

    if image_show:
        cv2.imshow('original', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Bilaterall filter for slight blurring and noise cancelling
    image = cv2.bilateralFilter(image, 9, 75, 75)
    if image_show:
        cv2.imshow('bf', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image = canny(image)
    result = give_contours_with_th(image, original)
    # Get Kmeans color quantized image
    image = give_kmeans_cq(image, 12)


    # Apply a green mask
    image = green_mask(image)

    # image = give_hue(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = give_contours_with_th(image, original)
    image = result[0]
    contours = result[1]
    return contours

# original = cv2.imread('Corn/Common Rust/3.jpg')
# original = cv2.imread('Corn/Eyespot/3.jpg')
# original = cv2.imread('Corn/SouthernCornLeafBlight/5.jpg')
#
# original = cv2.imread('Corn/Goss Wilt/1.jpg')
# contours = get_features(original)
#







