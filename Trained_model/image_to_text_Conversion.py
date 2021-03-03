from tqdm import tqdm
import imutils as imutils
import numpy as np
import cv2
import pytesseract

vidcam = False
file = "test/img_2.png"
min_thresh = 0
resize_factor = 2
rotation = 6
rotation_off = False

if vidcam:
    rotation_off = True
    resize_flag = True
    non_processed_scan = True
    show_progress = False
    resize_factor = 2
    processed_scan = False

else:
    resize_flag = False
    non_processed_scan = True
    show_progress = True
    processed_scan = True
    rotation_off = False


def nothing(x):
    pass


# not used for now.. only used in debuging
def initializeTrackbars(max=100, min=0, initial_value=50, name="trackbar", windoname="contorl"):
    cv2.namedWindow(windoname)
    cv2.createTrackbar(name, windoname, min, max, nothing)
    cv2.setTrackbarPos(name, windoname, initial_value)


camera = cv2.VideoCapture(0)

while 1:

    if vidcam:
        _, images = camera.read(0)

    else:
        images = cv2.imread(file)

    #
    # processing the image
    #

    if resize_flag:
        images = cv2.resize(images, (int(images.shape[0] / resize_factor), int(images.shape[0] / resize_factor)))

    # convert to grayscale image
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # convert the image into binary using Otsu’s Binarization
    binary_img = cv2.threshold(gray, min_thresh, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    if min_thresh == 255:
        min_thresh = 0

    # performing some median blur
    cv2.medianBlur(binary_img, 3)

    #
    # Pre- processing is done on the image
    #

    # taking the original copy of image
    original_image = images
    cv2.copyTo(images, original_image)
    images = binary_img

    # taking the text in multiple orientations
    final_text = " "
    final_angle = 0
    text_from_original = ""
    text = ""

    if rotation_off:

        if processed_scan:
            text = pytesseract.image_to_string(images, lang='eng')

        if non_processed_scan:
            text_from_original = pytesseract.image_to_string(original_image, lang='eng')

            # taking the maximum predicted text
        if len(text_from_original) > len(text):
            final_text = text_from_original

    else:

        for angle in tqdm(np.arange(0, 360, rotation), desc="Looking into various orientations"):
            rotated = imutils.rotate_bound(images, angle)

            if show_progress:
                cv2.imshow("Looking into various orientation", rotated)
                cv2.waitKey(25)

            if processed_scan:
                text = pytesseract.image_to_string(images, lang='eng')

            if non_processed_scan:
                text_from_original = pytesseract.image_to_string(original_image, lang='eng')

            # taking the maximum predicted text
            if len(text_from_original) > len(text):
                text = text_from_original

            if len(text) > len(final_text):
                final_text = text
                final_angle = angle

    # closing the temporary windo which was showing the rotaion
    if show_progress:
        cv2.destroyWindow("Looking into various orientation")

    # showing the original image
    cv2.imshow("original", original_image)

    # correcting the orientation which gave maximum text
    original_image = imutils.rotate(original_image, final_angle)

    # showing the corrected orientation of the original image
    if show_progress:
        cv2.imshow("More appropriate image orientation show", original_image)

    # printing the text recognized from the image
    print(final_text)

    if vidcam:
        cv2.waitKey(30)

    if not vidcam:
        cv2.waitKey(0)
        break

cv2.destroyAllWindows()
