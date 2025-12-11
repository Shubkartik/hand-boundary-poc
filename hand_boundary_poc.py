import cv2
import numpy as np
import time

# ----------- CONFIGURATION -------------
# Virtual boundary size
RECT_WIDTH = 200
RECT_HEIGHT = 150

# Distance thresholds (in pixels)
WARNING_THRESHOLD = 120
DANGER_THRESHOLD = 60

# ---------------------------------------

def detect_hand(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None, None

    # Largest contour = hand
    hand = max(contours, key=cv2.contourArea)
    if cv2.contourArea(hand) < 2000:
        return None, None

    hull = cv2.convexHull(hand)
    return hand, hull

def min_distance_to_rect(contour, rect):
    (x1, y1, x2, y2) = rect
    min_dist = 999999
    closest_point = None

    for point in contour:
        px, py = point[0]

        # compute distance to rectangle edges
        dx = max(x1 - px, 0, px - x2)
        dy = max(y1 - py, 0, py - y2)
        dist = (dx*dx + dy*dy) ** 0.5

        if dist < min_dist:
            min_dist = dist
            closest_point = (px, py)

    return min_dist, closest_point

cap = cv2.VideoCapture(0)

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # ----------------- Skin Segmentation -----------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60])
    upper = np.array([25, 200, 255])
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.medianBlur(mask, 7)

    # ----------------- Find hand -----------------
    hand_contour, hull = detect_hand(mask)

    # ----------------- Virtual Rectangle -----------------
    cx, cy = w // 2, h // 2
    rect = (cx - RECT_WIDTH//2, cy - RECT_HEIGHT//2,
            cx + RECT_WIDTH//2, cy + RECT_HEIGHT//2)

    (rx1, ry1, rx2, ry2) = rect

    state = "SAFE"
    color = (0, 255, 0)  # green

    if hand_contour is not None:
        # Distance from hand to rectangle
        dist, pt = min_distance_to_rect(hand_contour, rect)

        # State logic
        if dist <= DANGER_THRESHOLD:
            state = "DANGER"
            color = (0, 0, 255)  # red
        elif dist <= WARNING_THRESHOLD:
            state = "WARNING"
            color = (0, 165, 255)  # yellow/orange

        # Draw hand contour
        cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)
        cv2.drawContours(frame, [hull], -1, (0, 255, 255), 2)

        # Closest point marker
        if pt is not None:
            cv2.circle(frame, pt, 6, (255, 255, 255), -1)
            cv2.putText(frame, f"Dist: {int(dist)} px", (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw rectangle
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, 3)

    # State indicator
    cv2.putText(frame, f"State: {state}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # Danger alert
    if state == "DANGER":
        cv2.putText(frame, "DANGER  DANGER", (100, cy),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

    # FPS calculation
    curr = time.time()
    fps = 1 / (curr - prev_time)
    prev_time = curr

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show windows
    cv2.imshow("Hand Boundary POC", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
