import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="CV Techniques", layout="wide")
st.title("Explore Computer Vision!")

#converts BGR to RGB for display
def convert_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#sidebar for technique selection
technique = st.sidebar.selectbox("Choose CV Technique", [
    "Image Manipulation", 
    "Object Tracking", 
    "Shape Detection", 
    "Dense Optical Flow"
])

# IMAGE MANIPULATION
if technique == "Image Manipulation":
    st.subheader("🖼️ Image Manipulation")
    
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Sharpening
        kernel_sharpening = np.array([[-1, -1, -1],
                                      [-1,  9, -1],
                                      [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # Dilation
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(image, kernel, iterations=1)

        # Erosion
        eroded = cv2.erode(image, kernel, iterations=1)

        # Combined
        combined = cv2.erode(cv2.dilate(sharpened, kernel, iterations=1), kernel, iterations=1)

        # Edges
        edges = cv2.Canny(dilated, 30, 60) # lowered thresholds to detect more edges
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        st.image([convert_to_rgb(image), convert_to_rgb(sharpened), convert_to_rgb(dilated), convert_to_rgb(eroded), convert_to_rgb(combined), convert_to_rgb(edges)], 
                 caption=["Original", "Sharpened", "Dilated", "Eroded", "Combined", "Edges"], width=250)

# OBJECT TRACKING
elif technique == "Object Tracking":
    st.subheader("🎯 Object Tracking")

    run = st.checkbox("Start Tracking")
    if run:
        FRAME_WINDOW = st.image([])

        cap = cv2.VideoCapture(0)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        points = []
        frame_count = 0

        while run:
            ret, frame = cap.read()
            if not ret:
                break

            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            Height, Width = frame.shape[:2]
            center = (int(Width / 2), int(Height / 2))
            radius = 0

            if contours:
                c = max(contours, key=cv2.contourArea)
                (x, y), radius = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                try:
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                except:
                    center = (int(Width / 2), int(Height / 2))

                if radius > 25:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)

            points.append(center)

            if radius > 25:
                for i in range(1, len(points)):
                    try:
                        cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
                    except:
                        pass
                frame_count = 0
            else:
                frame_count += 1
                if frame_count == 10:
                    points = []
                    frame_count = 0

            frame = cv2.flip(frame, 1)
            FRAME_WINDOW.image(convert_to_rgb(frame))
            if cv2.waitKey(1) == 13:
                break

        cap.release()
        cv2.destroyAllWindows()

# SHAPE DETECTION
elif technique == "Shape Detection":
    st.subheader("🔷 Shape Detection via Webcam")

    run = st.checkbox("Start Shape Detection")
    if run:
        FRAME_WINDOW = st.image([])

        def detect_shapes(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 1000:
                    continue

                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                shape_name, color = None, (0, 0, 0)
                if len(approx) == 3:
                    shape_name, color = "Triangle", (0, 255, 0)
                elif len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    ratio = w / float(h)
                    if 0.95 < ratio < 1.05:
                        shape_name, color = "Square", (0, 125, 255)
                    else:
                        shape_name, color = "Rectangle", (0, 0, 255)
                elif len(approx) == 10:
                    shape_name, color = "Star", (255, 255, 0)
                else:
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.7:
                        shape_name, color = "Circle", (0, 255, 255)

                if shape_name:
                    cv2.drawContours(frame, [approx], 0, color, 3)
                    cv2.putText(frame, shape_name, (cx - 50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            return frame

        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            output = detect_shapes(frame)
            FRAME_WINDOW.image(convert_to_rgb(output))
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

# DENSE OPTICAL FLOW
elif technique == "Dense Optical Flow":
    st.subheader("💨 Dense Optical Flow")

    video_file = st.file_uploader("Upload a video (e.g., .avi)", type=["mp4", "avi"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        ret, first_frame = cap.read()
        previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(first_frame)
        hsv[..., 1] = 255

        FRAME_WINDOW = st.image([])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(previous_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            FRAME_WINDOW.image(convert_to_rgb(final))
            previous_gray = next_gray
        cap.release()
