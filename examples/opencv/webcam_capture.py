import os

import cv2
import requests


def download_haarcascade(filename: str) -> None:
    """Download the Haar cascade file from the OpenCV repository."""
    url = f"https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/{filename}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()  # Check if the request was successful
    with open(filename, "wb") as f:
        f.write(response.content)


def capture_and_grayscale() -> None:
    """Capture video from the webcam and convert it to grayscale."""
    filename = "haarcascade_frontalface_default.xml"

    if not os.path.isfile(filename):
        print(f"{filename} not found, downloading...")
        download_haarcascade(filename)

    # Load the cascade for face detection
    face_cascade = cv2.CascadeClassifier(filename)

    # Search for available webcams
    working_cam = None
    for index in range(8):
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            print(f"Webcam {index} not found")
            cap.release()
            continue
        else:
            print(f"Webcam {index} found")
            working_cam = cap
            break

    assert working_cam is not None, "No webcam found"

    # Check if the webcam is opened correctly
    if not working_cam.isOpened():
        raise OSError("Cannot open webcam")

    captured_frame = None
    while True:
        # Read the current frame from the webcam
        ret, frame = working_cam.read()
        if not ret:
            break

        if captured_frame is None:
            captured_frame = frame
            print("Frame captured")
            print(
                "Frame width/height: %d/%d",
                working_cam.get(cv2.CAP_PROP_FRAME_WIDTH),
                working_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Convert grayscale image back to BGR
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Replace the grayscale face area with the original colored face
        for x, y, w, h in faces:
            gray_bgr[y : y + h, x : x + w] = frame[y : y + h, x : x + w]
            cv2.rectangle(gray_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the image
        cv2.imshow("Input", gray_bgr)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the VideoCapture object
    working_cam.release()

    # Destroy all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_grayscale()
