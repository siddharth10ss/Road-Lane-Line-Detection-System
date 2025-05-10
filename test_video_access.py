import cv2

def test_video_access(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video source")
    else:
        print("Success: Video source opened")
    cap.release()

if __name__ == "__main__":
    video_path = r"C:\\Users\\Admin\\next24_intern_projects\\Road_lane\\test_video.mp4"
    test_video_access(video_path)
