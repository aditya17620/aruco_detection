#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState

# ==============================================================================
# === ROS & CONTROL PARAMETERS (TUNE THESE) ===
# ==============================================================================

CONTROL_TOPIC = '/cartesian_impedance_example_controller/equilibrium_pose'
PIXEL_TO_METER_GAIN = 0.0005
AREA_TO_METER_GAIN = 0.000002
SMOOTHING_FACTOR_XY = 0.1
SMOOTHING_FACTOR_Z = 0.07

# ==============================================================================
# === PHYSICAL & CAMERA PARAMETERS (SET THESE ONCE) ===
# ==============================================================================

CAMERA_MATRIX = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
                          [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
DIST_COEFFS = np.array([[0.09617148, -0.1634111, 0.00022246, -0.00056651, -0.00507073]], dtype=np.float32)
CAMERA_INDEX = 0

# ==============================================================================
# === CORE LOGIC (GENERALLY DO NOT NEED TO EDIT BELOW THIS LINE) ===
# ==============================================================================

target_pose_msg = PoseStamped()
initial_pose_position = None
initial_pose_orientation = None
target_pixel_area = None
smoothed_command_xy_m = np.array([0.0, 0.0])
smoothed_command_z_m = 0.0

def detect_aruco_properties(image_to_detect):
    """
    Detects an ArUco marker from a given image.
    MODIFIED: This function now ONLY performs detection and returns raw data. It does not draw.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(image_to_detect)

    if ids is not None:
        marker_corners = corners[0].reshape((4, 2))
        centerX = int(np.mean(marker_corners[:, 0]))
        centerY = int(np.mean(marker_corners[:, 1]))
        area = cv2.contourArea(marker_corners)
        # Return all detected properties, including corners for drawing later
        return marker_corners, centerX, centerY, area

    return None, None, None, None

def init_ros_and_robot_state(cap):
    """
    Initializes ROS, robot state, and target area.
    MODIFIED: Uses the new preprocessing steps to detect the initial marker.
    """
    global target_pose_msg, initial_pose_position, initial_pose_orientation, target_pixel_area

    rospy.init_node('aruco_3d_pixel_controller', anonymous=True)
    pub = rospy.Publisher(CONTROL_TOPIC, PoseStamped, queue_size=1)

    print("Waiting for initial Franka state...")
    current_state = rospy.wait_for_message("franka_state_controller/franka_states", FrankaState)
    print("Franka state received.")

    initial_pose_position = np.array([current_state.O_T_EE[12], current_state.O_T_EE[13], current_state.O_T_EE[14]])
    initial_pose_orientation = tf.transformations.quaternion_from_matrix(
        np.transpose(np.reshape(current_state.O_T_EE, (4, 4))))
    initial_pose_orientation /= np.linalg.norm(initial_pose_orientation)

    print("Detecting ArUco marker to set target Z-distance...")
    ret, frame = cap.read()
    if not ret:
        rospy.logerr("Cannot read from camera during initialization. Exiting.")
        exit()

    # --- NEW: Apply same preprocessing for initialization ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed_frame = cv2.equalizeHist(gray_frame)
    _, _, _, initial_area = detect_aruco_properties(processed_frame)

    if initial_area is None:
        rospy.logerr("Could not find ArUco marker during initialization. Make sure it is visible. Exiting.")
        exit()

    target_pixel_area = initial_area
    print(f"Marker detected. Target area for Z-control set to: {target_pixel_area:.2f} pixels^2")

    target_pose_msg.header.frame_id = "panda_link0"
    target_pose_msg.pose.orientation.x = initial_pose_orientation[0]
    target_pose_msg.pose.orientation.y = initial_pose_orientation[1]
    target_pose_msg.pose.orientation.z = initial_pose_orientation[2]
    target_pose_msg.pose.orientation.w = initial_pose_orientation[3]

    print(f"Initialization complete. Robot starting at: {initial_pose_position}")
    return pub

def process_frame(detection_frame, visualization_frame, pub):
    """
    Main processing pipeline.
    MODIFIED: Takes two frames - one for detection, one for visualization.
    """
    global smoothed_command_xy_m, smoothed_command_z_m

    height, width, _ = visualization_frame.shape
    image_center_x, image_center_y = width // 2, height // 2

    # --- Use the processed `detection_frame` for finding the marker ---
    marker_corners, marker_center_x, marker_center_y, marker_area = detect_aruco_properties(detection_frame)

    if marker_center_x is not None:
        # --- 1. Control Calculation (No changes here) ---
        error_x = marker_center_x - image_center_x
        error_y = marker_center_y - image_center_y
        raw_command_xy_m = np.array([-error_y * PIXEL_TO_METER_GAIN, -error_x * PIXEL_TO_METER_GAIN])
        smoothed_command_xy_m = SMOOTHING_FACTOR_XY * raw_command_xy_m + (1 - SMOOTHING_FACTOR_XY) * smoothed_command_xy_m

        area_error = marker_area - target_pixel_area
        raw_command_z_m = -area_error * AREA_TO_METER_GAIN
        smoothed_command_z_m = SMOOTHING_FACTOR_Z * raw_command_z_m + (1 - SMOOTHING_FACTOR_Z) * smoothed_command_z_m

        # --- 2. Publishing (No changes here) ---
        target_pose_msg.pose.position.x = initial_pose_position[0] + smoothed_command_xy_m[0]
        target_pose_msg.pose.position.y = initial_pose_position[1] + smoothed_command_xy_m[1]
        target_pose_msg.pose.position.z = initial_pose_position[2] + smoothed_command_z_m
        target_pose_msg.header.stamp = rospy.Time.now()
        pub.publish(target_pose_msg)

        # --- 3. Visualization (NEW: Drawing is now done here on the color frame) ---
        # Draw marker outline and center on the original color image
        cv2.polylines(visualization_frame, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(visualization_frame, (marker_center_x, marker_center_y), 4, (0, 0, 255), -1)

        # Print status
        print("\033c", end="") # Clears console
        # ... (print statements are the same) ...
        print(f"Final Target Pos (m):  [{target_pose_msg.pose.position.x:.3f}, {target_pose_msg.pose.position.y:.3f}, {target_pose_msg.pose.position.z:.3f}]")

    # Draw the target image center on the visualization frame
    cv2.circle(visualization_frame, (image_center_x, image_center_y), 5, (255, 0, 0), -1)
    
    # Display both the final visualization and the processed frame for debugging
    cv2.imshow("ArUco Control - Visualization", visualization_frame)
    cv2.imshow("Processed Frame for Detection", detection_frame)


if __name__ == '__main__':
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            rospy.logerr(f"Cannot open camera at index {CAMERA_INDEX}. Exiting.")
            exit()

        control_pub = init_ros_and_robot_state(cap)

        print("\nStarting control loop. Press 'q' in any OpenCV window to quit.")

        while not rospy.is_shutdown():
            ret, color_frame = cap.read()
            if not ret:
                rospy.logwarn("Failed to grab frame from camera.")
                continue

            # --- NEW: IMAGE PREPROCESSING STEP ---
            # 1. Convert the original color frame to grayscale.
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            # 2. Apply histogram equalization to improve contrast.
            processed_frame = cv2.equalizeHist(gray_frame)
            
            # Pass BOTH the processed frame (for detection) and the original color frame (for visualization)
            process_frame(processed_frame, color_frame, control_pub)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"An unexpected error occurred: {e}")
    finally:
        print("\nShutting down.")
        cap.release()
        cv2.destroyAllWindows()
