import cv2
import numpy as np
import time

# Global variables for tracking
prev_time = time.time()
prev_tvec = None
marker_physical_width_m = 0.075  # Marker width = 5 cm

# Camera calibration parameters (replace with your actual calibration)
camera_matrix = np.array([[2.67316869e+03, 0.00000000e+00, 1.50382273e+03],
                        [0.00000000e+00, 2.65505218e+03, 1.94810216e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
dist_coeffs = np.array([[ 2.67592540e-01, -1.66000383e+00, -4.45377936e-04, -1.29048332e-03, 4.11844897e+00]]
, dtype=np.float32)  # Assuming no distortion for now

def detect_aruco(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids

def visualize_aruco(frame, corners, ids):
    global prev_time, prev_tvec
    current_time = time.time()

    if ids is not None:
        ids = ids.flatten()

        print("\033c")  # Clear terminal

        for (corner, ID) in zip(corners, ids):
            reshaped_corners = corner.reshape((4, 2))
            topLeft, topRight, bottomRight, bottomLeft = reshaped_corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            area = cv2.contourArea(np.array([topLeft, topRight, bottomRight, bottomLeft]))

            centerX = int((topLeft[0] + topRight[0] + bottomRight[0] + bottomLeft[0]) / 4)
            centerY = int((topLeft[1] + topRight[1] + bottomRight[1] + bottomLeft[1]) / 4)

            xaxis = (int((topRight[0] + topLeft[0]) / 2), int((topRight[1] + topLeft[1]) / 2))
            yaxis = (int((bottomRight[0] + topRight[0]) / 2), int((bottomRight[1] + topRight[1]) / 2))

            # Estimate pose of the marker
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, marker_physical_width_m, camera_matrix, dist_coeffs)
            # if prev_tvec is not None:
            #     delta = tvec[0][0] - prev_tvec[0]  # x, y, z
            #     real_distance_m = np.linalg.norm(delta)


            # Distance moved calculation
            
            # if current_time - prev_time >= 0.1:  # 10 Hz
                # if prev_tvec is not None:
                #     delta = tvec[0][0] - prev_tvec[0]  # x, y, z
                #     real_distance_m = np.linalg.norm(delta)
                #     # print(f"Moved: {real_distance_m:.4f} meters in last 0.1s")
                # prev_tvec = tvec
                # prev_time = current_time

            # Debug prints
            print('ID:', ID)
            print('topRight:', topRight)
            print('topLeft:', topLeft)
            print('bottomRight:', bottomRight)
            print('bottomLeft:', bottomLeft)
            print('Xaxis:', xaxis)
            print('Area:', area)
            print('Center:', centerX, centerY)
            print('tvec (position in meters):', tvec[0][0])
            print('rvec (rotation):', rvec[0][0])
            # if real_distance_m is not None: 
                # print('delta: ', real_distance_m)
            print('--------------------------------------')

            # Drawing
            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
            cv2.line(frame, (centerX, centerY), xaxis, (0, 0, 255), 2)
            cv2.putText(frame, str(ID), topLeft, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Draw axis
            # cv2.aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)  # axis length = 3 cm

    cv2.imshow("Detected ArUco markers", frame)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        corners, ids = detect_aruco(frame)
        visualize_aruco(frame, corners, ids)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
