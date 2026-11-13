import cv2

def detect_object(frame):
    h, w = frame.shape[:2]

    x1, x2 = int(w * 0.3), int(w * 0.65)
    y1, y2 = int(h * 0.1), int(h * 0.7)

    roi = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_blue = (95,70,50)
    upper_blue = (135,255,255)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #find centroid of object in mask
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < 300:
        return None, None

    # Centroid
    M = cv2.moments(contour)

    if M["m00"] == 0:
        return None, None

    u = int(M["m10"] / M["m00"]) + x1
    v = int(M["m01"] / M["m00"]) + y1

    # Bounding box
    x, y, width, height = cv2.boundingRect(contour)

    # Convert ROI-relative bounds to full image coordinates
    bounds = (x + x1, y + y1, width, height)

    return (u,v), bounds