import cv2


def get_descriptor_and_label(image_path):
    sift = cv2.SIFT_create()
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)  # SIFT only applied to gray variables
    _, descriptors = sift.detectAndCompute(img, None)
    return descriptors
