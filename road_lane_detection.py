import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def warp_perspective(img, src_points, dst_points):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M, Minv

def process_frame(frame):
    """
    Process a single frame for lane detection with perspective transform and polynomial fit
    """
    height, width = frame.shape[:2]

    # Step 1: Preprocess (Grayscale + Blur)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Step 3: Region of Interest Mask
    vertices = np.array([[
        (100, height),
        (width//2 - 50, height//2 + 50),
        (width//2 + 50, height//2 + 50),
        (width - 100, height)
    ]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Step 4: Perspective Transform (Bird's Eye View)
    src = np.float32([
        [100, height],
        [width//2 - 50, height//2 + 50],
        [width//2 + 50, height//2 + 50],
        [width - 100, height]
    ])
    dst = np.float32([
        [200, height],
        [200, 0],
        [width - 200, 0],
        [width - 200, height]
    ])
    warped, M, Minv = warp_perspective(masked_edges, src, dst)

    # Step 5: Detect lane pixels and fit polynomial
    histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set parameters for sliding windows
    nwindows = 9
    window_height = int(warped.shape[0]//nwindows)
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Initialize good_left_inds and good_right_inds to empty arrays to avoid UnboundLocalError
    good_left_inds = np.array([], dtype=int)
    good_right_inds = np.array([], dtype=int)
    leftx_current = int(np.mean(nonzerox[good_left_inds])) if good_left_inds.size > 0 else leftx_base
    rightx_current = int(np.mean(nonzerox[good_right_inds])) if good_right_inds.size > 0 else rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each lane line
    left_fit = None
    right_fit = None
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = None
    right_fitx = None
    if left_fit is not None:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if right_fit is not None:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast x and y for cv2.fillPoly()
    if left_fitx is not None and right_fitx is not None:
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space
    newwarp = cv2.warpPerspective(color_warp, Minv, (frame.shape[1], frame.shape[0]))
    result = cv2.addWeighted(frame, 1, newwarp, 0.3, 0)

    # Calculate lane center and vehicle center offset
    if left_fitx is not None and right_fitx is not None:
        lane_center = (left_fitx[-1] + right_fitx[-1]) / 2
        vehicle_center = width / 2
        center_offset = (vehicle_center - lane_center) * 3.7 / 700  # meters, assuming lane width 3.7m and 700 pixels width

        # Add text feedback
        font = cv2.FONT_HERSHEY_SIMPLEX
        direction = ""
        if center_offset > 0.1:
            direction = "Move Right"
        elif center_offset < -0.1:
            direction = "Move Left"
        else:
            direction = "On Lane"

        cv2.putText(result, f'Offset: {center_offset:.2f} m', (50, 50), font, 1, (255, 255, 255), 2)
        cv2.putText(result, f'Direction: {direction}', (50, 100), font, 1, (255, 255, 255), 2)
    else:
        # If lane lines are not detected, just return the original frame
        return frame

    return result

def main():
    # For video file input (replace with your video path)
    cap = cv2.VideoCapture(r'C:\\Users\\Admin\\next24_intern_projects\\Road_lane\\test_video.mp4')

    # For webcam input (uncomment below)
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        cv2.imshow('Lane Detection', processed_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
