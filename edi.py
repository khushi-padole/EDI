import cv2
import numpy as np

# --------------------------------
# Fixed scale (from your notebook)
# --------------------------------
# PIXELS_PER_CM = 86.65

# 1st correct for image 1
# PIXELS_PER_CM = 77.732



# image 5

PIXELS_PER_CM =   73.14244151685797


# --------------------------------
# Load image
# --------------------------------
image = cv2.imread("I_4.png")

if image is None:
    print("Image not found")
    exit()

orig = image.copy()

# --------------------------------
# Preprocessing
# --------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

edges = cv2.Canny(blur,50,150)
edges = cv2.dilate(edges,None,iterations=2)
edges = cv2.erode(edges,None,iterations=1)

# --------------------------------
# Find contours
# --------------------------------
contours,_ = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

for c in contours:

    area = cv2.contourArea(c)

    if area < 2000:
        continue

    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = box.astype(int)

    width_pixels = rect[1][0]
    height_pixels = rect[1][1]

    if width_pixels == 0 or height_pixels == 0:
        continue

    width_cm = width_pixels / PIXELS_PER_CM
    height_cm = height_pixels / PIXELS_PER_CM

    label = f"{width_cm:.2f}cm x {height_cm:.2f}cm"

    cv2.drawContours(orig,[box],0,(0,255,0),2)

    x,y = box[0]

    cv2.putText(orig,label,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(255,0,0),2)

# --------------------------------
# Show result
# --------------------------------

cv2.namedWindow("Object Measurement", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Object Measurement", 400, 400)


cv2.imshow("Object Measurement",orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
