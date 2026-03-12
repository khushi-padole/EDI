import cv2
import numpy as np

# Real object dimensions (cm)
# REAL_LENGTH = 17
# REAL_WIDTH = 4


# IMAGE 5
REAL_LENGTH = 5.7
REAL_WIDTH = 3.52

# Load image
image = cv2.imread("I_5.png")

if image is None:
    print("Image not found")
    exit()

orig = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to remove noise
blur = cv2.GaussianBlur(gray, (7,7), 0)

# Edge detection
edges = cv2.Canny(blur, 50, 150)

# Close gaps in edges
kernel = np.ones((5,5), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)
edges = cv2.erode(edges, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort contours by area
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Largest contour = object
c = contours[0]

rect = cv2.minAreaRect(c)

width_pixels = rect[1][0]
height_pixels = rect[1][1]

# Ensure length is larger side
length_pixels = max(width_pixels, height_pixels)
width_pixels = min(width_pixels, height_pixels)

print("Length in pixels:", length_pixels)
print("Width in pixels:", width_pixels)

# Calculate pixels per cm
pixels_per_cm_length = length_pixels / REAL_LENGTH
pixels_per_cm_width = width_pixels / REAL_WIDTH

pixels_per_cm = (pixels_per_cm_length + pixels_per_cm_width) / 2

print("\nPixels per cm:", pixels_per_cm)

cm_per_pixel = 1 / pixels_per_cm

print("1 pixel =", cm_per_pixel, "cm")

# Draw rectangle for verification
box = cv2.boxPoints(rect)
box = box.astype(int)

cv2.drawContours(orig, [box], 0, (0,255,0), 2)


cv2.imshow("Detected Object", orig)
cv2.waitKey(0)
cv2.destroyAllWindows()
