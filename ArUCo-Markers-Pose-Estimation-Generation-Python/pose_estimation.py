import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time

rvec_old = None
tvec_old = None
last_time = time.time()

def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    global rvec_old, tvec_old, last_time

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and any(i[0] in (1, 2, 3) for i in ids):
        for i, corner in enumerate(corners):
            marker_length = 0.02  # meters
            obj_points = np.array([
                [-marker_length / 2, marker_length / 2, 0],
                [ marker_length / 2, marker_length / 2, 0],
                [ marker_length / 2,-marker_length / 2, 0],
                [-marker_length / 2,-marker_length / 2, 0]
            ], dtype=np.float32)

            img_points = corner[0].astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(obj_points, img_points, matrix_coefficients, distortion_coefficients)

            if success:
                current_time = time.time()
                if current_time - last_time >= 0.5:  # Every 100 ms
                    if rvec_old is not None:
                        rvec_delta = rvec - rvec_old
                    else:
                        rvec_delta = np.zeros_like(rvec)

                    if tvec_old is not None:
                        tvec_delta = tvec - tvec_old
                    else:
                        tvec_delta = np.zeros_like(tvec)

                    print("\033c", end="")  # Clear terminal
                    print(f"tvec ID {ids[i][0]}")
                    print(np.round(tvec.reshape(-1), 3))
                    print("\ndelta:")
                    print(np.round(tvec_delta.reshape(-1), 3))

                    rvec_old = rvec
                    tvec_old = tvec
                    last_time = current_time

                cv2.aruco.drawDetectedMarkers(frame, [corner])
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    k = np.load(args["K_Matrix"])
    d = np.load(args["D_Coeff"])

    video = cv2.VideoCapture(0)
    time.sleep(2.0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        output = pose_esitmation(frame, aruco_dict_type, k, d)
        cv2.imshow('Estimated Pose', output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
