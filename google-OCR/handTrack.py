import numpy as np
import cv2
from collections import deque

# tesseract
import os
from PIL import Image
import PIL.ImageDraw
import PIL.ImageOps
import PIL.ImageFont
import pytesseract


# process image libraries and functions
import math

_DEFAULT_FONT = os.path.join(os.path.dirname(__file__), 'NotoSansCJK-Bold.ttc')  # high coverage font
_PIXEL_ON = 0  # PIL color value to indicate a shape should be used (black)
_PIXEL_OFF = 255  # PIL color value to indicate a shape is off (white)


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Function to apply the Canny transform


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Function to apply the gaussian blur


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def draw_lines(img, lines, color=[0, 0, 255], thickness=3):

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = ((y2 - y1) / (x2 - x1))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # print(type(lines))
    # print(lines.shape[::])
    # print(lines[0])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# convert string to image:


def string_image(string, font_path=None):
    """Return a grayscale image with black characters on a white background.
    arguments:
    string - this string will be converted to an image
             if string has "\n" token in it, interpret it as a newline
    font_path - path to a font file (for example impact.ttf)
               if font path is provided, it might work in three ways
               1) path completely defines location of a font
               2) just a file name works for a font in the current working directory
               3) just a file name works for a font somewhere in the system path
               4) on windows, PILLOW may search the windows fonts directory.
                  on linux, it does not as of 2015-August
    """
    grayscale = 'L'
    # parse any literal '\n' into newlines
    lines = string.split('\\n')
    # choose a font
    large_font = 1000
    font_path = font_path or _DEFAULT_FONT
    try:
        font = PIL.ImageFont.truetype(font_path, size=large_font)
    except IOError:
        font = None
    if font is None:
        if font_path == _DEFAULT_FONT:
            raise RuntimeError('Unable to load built-in font ({})'.format(_DEFAULT_FONT))
        else:
            raise ValueError('Unable to load provided font ({})'.format(font_path))

    # make the background image based on the combination of font and lines
    def pt2px(pt): return int(round(pt * 96.0 / 72))  # convert points to pixels
    max_width_line = max(lines, key=lambda s: font.getsize(s)[0])
    # max height is adjusted down because it's too large visually for spacing
    test_string = 'abcdefghijklmnopqrstuvwxyz'  # some bug with single chars
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    height = max_height * len(lines)  # perfect or a little oversized
    width = int(round(max_width + 40))  # a little oversized
    image = PIL.Image.new(grayscale, (width, height), color=_PIXEL_OFF)
    draw = PIL.ImageDraw.Draw(image)

    # draw each line of text
    vertical_position = 5
    horizontal_position = 5
    line_spacing = int(round(max_height * 0.65))  # reduced spacing seems better
    for line in lines:
        draw.text((horizontal_position, vertical_position),
                  line, fill=_PIXEL_ON, font=font)
        vertical_position += line_spacing
    # crop the text
    c_box = PIL.ImageOps.invert(image).getbbox()
    image = image.crop(c_box)
    return image


"""
Initialize variables
"""

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([100, 60, 60])
blueUpper = np.array([140, 255, 255])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

# Initialize deques to store different colors in different arrays
bpoints = [deque(maxlen=512)]

# Initialize an index variable for each of the colors
bindex = 0

# Blue
color = (255, 0, 0)


# Load the video
camera = cv2.VideoCapture(0)
width = (int)(camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = (int)(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

"""
Set up a paint interface
"""

# Create a blank white image
#
#(471, 636, 3)
paintWindow = np.zeros((width, height, 3), np.uint8) + 255


# Create a window to display the above image (later)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Keep looping
while True:
    # Grab the current paintWindow or current video image frame
    (check, frame) = camera.read()
    # flip frame in vertical direction
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Check to see if we have reached the end of the video (useful when input is a video file not a live video stream)
    if not check:
        break

    # Add the same paint interface to the camera feed captured through the webcam (for ease of usage)
    frame = cv2.rectangle(frame, (40, 1), (140, 65), (122, 122, 122), -1)
    frame = cv2.rectangle(frame, (160, 1), (255, 65), color, -1)

    cv2.putText(frame, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

 # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours in the image
    cnts, hierrachy = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

 # Check to see if any contours (blue stuff) were found
    if len(cnts) > 0:
        # Sort the contours and find the largest one -- we assume this contour correspondes to the area of the bottle cap
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        # Draw the circle around the contour
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        # Get the moments to calculate the center of the contour (in this case a circle)
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear All
                # Empty all the point holders
                bpoints = [deque(maxlen=512)]
                # Reset the indices
                bindex = 0
                # Make the frame all white again
                paintWindow[67:, :, :] = 255

        # Store the center (point) in its assigned color deque
        else:
            bpoints[bindex].appendleft(center)

    # Draw blue line
    for i in range(len(bpoints)):
        for k in range(1, len(bpoints[i])):
            if bpoints[i][k - 1] is None or bpoints[i][k] is None:  # x,y
                continue
            cv2.line(frame, bpoints[i][k - 1], bpoints[i][k], color, 2)
            thickness = 5
            cv2.line(paintWindow, bpoints[i][k - 1], bpoints[i][k], color, thickness)

    # Show the frame and the paintWindow image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
         # Rescale the image, if needed.
        #img = cv2.resize(paintWindow, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

        gray = grayscale(paintWindow)
        gaus = gaussian_blur(gray, 35)
        #img = cv.bilateralFilter(img,9,75,75)
        # cv.threshold(img,127,255,cv.THRESH_BINARY)
        c = cv2.Canny(gaus, 2, 10)
        rho = 3     # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 40     # minimum number of votes (intersections in Hough grid cell)
        min_line_len = 70  # minimum number of pixels making up a line
        max_line_gap = 100
        h = hough_lines(c, rho, theta, threshold, min_line_len, max_line_gap)

        # Apply dilation and erosion to remove some noise
        # kernel = np.ones((1, 1), np.uint8)
        # img = cv2.dilate(h, kernel, iterations=1)
        # img = cv2.erode(img, kernel, iterations=1)

        # Apply threshold to get image with only black and white
        #img = apply_threshold(img, method)
        #cv2.threshold(h, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cv2.imwrite("trackingImage.png", img)
    
        cv2.imwrite("paintImage.png", paintWindow)
        
        break

# gausian blur
# canny
# edge detection
# Cleanup code
camera.release()
cv2.destroyAllWindows()
