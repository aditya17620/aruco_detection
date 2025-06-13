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

# The topic to publish control commands to.
CONTROL_TOPIC = '/cartesian_impedance_example_controller/equilibrium_pose'

# --- XY Control ---
# Converts pixel error in X/Y to a robot movement command in meters.
# A larger value means the robot moves more for the same pixel error.
PIXEL_TO_METER_GAIN = 0.0005  # meters per pixel

# --- Z Control (NEW) ---
# Converts pixel area error to a robot Z-movement command in meters.
# A larger value means the robot moves more for the same area difference.
AREA_TO_METER_GAIN = 0.000002  # meters per pixel^2

# --- Smoothing ---
# Low-pass filter factor for smoothing the robot's motion (0.0 to 1.0).
# Smaller value = more smoothing, more lag. Larger value = more responsive, more jitter.
SMOOTHING_FACTOR_XY = 0.1
SMOOTHING_FACTOR_Z = 0.07 # Z might benefit from slightly more smoothing

# ==============================================================================
# === PHYSICAL & CAMERA PARAMETERS (SET THESE ONCE) ===
# ==============================================================================

# Your camera's calibration matrix.
CAMERA_MATRIX = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
                          [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
DIST_COEFFS = np.array([[0.09617148, -0.1634111, 0.00022246, -0.00056651, -0.00507073]], dtype=np.float32)

# Which camera to use.
CAMERA_INDEX = 0

# ==============================================================================
# === CORE LOGIC (GENERALLY DO NOT NEED TO EDIT BELOW THIS LINE) ===
# ==============================================================================

# Global variable to hold the target pose message.
target_pose_msg = PoseStamped()

# Global variables for storing robot and control state.
initial_pose_position = None
initial_pose_orientation = None
target_pixel_area = None # The area we want the marker to have.

# Global variables for smoothing the command.
smoothed_command_xy_m = np.array([0.0, 0.0])
smoothed_command_z_m = 0.0

def detect_aruco_properties(frame):
    """
    Detects an ArUco marker and returns its center (x,y) and its area in pixels.
    MODIFIED: Now also returns area.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        # Take the first detected marker
        marker_corners = corners[0].reshape((4, 2))

        # Calculate the center of the marker
        centerX = int(np.mean(marker_corners[:, 0]))
        centerY = int(np.mean(marker_corners[:, 1]))

        # Calculate the area of the marker
        area = cv2.contourArea(marker_corners)

        # For visualization
        cv2.polylines(frame, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(frame, (centerX, centerY), 4, (0, 0, 255), -1)

        return centerX, centerY, area

    return None, None, None

def init_ros_and_robot_state(cap):
    """
    Initializes ROS, gets the robot's starting pose, and determines the
    target ArUco marker area for Z-axis control.
    MODIFIED: Now requires the camera capture object to find the initial area.
    """
    global target_pose_msg, initial_pose_position, initial_pose_orientation, target_pixel_area

    rospy.init_node('aruco_3d_pixel_controller', anonymous=True)
    pub = rospy.Publisher(CONTROL_TOPIC, PoseStamped, queue_size=1)

    print("Waiting for initial Franka state...")
    current_state = rospy.wait_for_message("franka_state_controller/franka_states", FrankaState)
    print("Franka state received.")

    # Store the initial position [x, y, z]
    initial_pose_position = np.array([
        current_state.O_T_EE[12],
        current_state.O_T_EE[13],
        current_state.O_T_EE[14]
    ])

    # Store the initial orientation as a quaternion [x, y, z, w]
    initial_pose_orientation = tf.transformations.quaternion_from_matrix(
        np.transpose(np.reshape(current_state.O_T_EE, (4, 4)))
    )
    initial_pose_orientation /= np.linalg.norm(initial_pose_orientation)

    # --- NEW: Set the target area for Z-control ---
    print("Attempting to detect ArUco marker to set target Z-distance...")
    ret, frame = cap.read()
    if not ret:
        rospy.logerr("Cannot read from camera during initialization. Exiting.")
        exit()

    _, _, initial_area = detect_aruco_properties(frame) # We only need the area for init
    if initial_area is None:
        rospy.logerr("Could not find ArUco marker during initialization. Make sure it is visible to the camera. Exiting.")
        exit()

    target_pixel_area = initial_area
    print(f"Marker detected. Target area for Z-control set to: {target_pixel_area:.2f} pixels^2")

    # Prepare the constant parts of our target pose message
    target_pose_msg.header.frame_id = "panda_link0"
    target_pose_msg.pose.orientation.x = initial_pose_orientation[0]
    target_pose_msg.pose.orientation.y = initial_pose_orientation[1]
    target_pose_msg.pose.orientation.z = initial_pose_orientation[2]
    target_pose_msg.pose.orientation.w = initial_pose_orientation[3]

    print(f"Initialization complete. Robot starting at:\n"
          f"Position: {initial_pose_position}\n"
          f"Orientation: {initial_pose_orientation}")

    return pub

def process_frame(frame, pub):
    """
    Main processing pipeline for each frame. Detects the marker, calculates
    the XYZ control command, and publishes it.
    """
    global smoothed_command_xy_m, smoothed_command_z_m

    height, width, _ = frame.shape
    image_center_x, image_center_y = width // 2, height // 2
    cv2.circle(frame, (image_center_x, image_center_y), 5, (255, 0, 0), -1)

    marker_center_x, marker_center_y, marker_area = detect_aruco_properties(frame)

    if marker_center_x is not None:
        # --- 1. XY Control (same as before) ---
        error_x = marker_center_x - image_center_x
        error_y = marker_center_y - image_center_y
        raw_command_xy_m = np.array([
           -error_y * PIXEL_TO_METER_GAIN, # Maps pixel Y-error to robot X-motion
           -error_x * PIXEL_TO_METER_GAIN  # Maps pixel X-error to robot Y-motion
        ])
        smoothed_command_xy_m = SMOOTHING_FACTOR_XY * raw_command_xy_m + (1 - SMOOTHING_FACTOR_XY) * smoothed_command_xy_m

        # --- 2. Z Control (NEW) ---
        # If the marker is bigger than the target (closer), move up (positive Z).
        # If the marker is smaller than the target (further), move down (negative Z).
        area_error = marker_area - target_pixel_area
        raw_command_z_m = -area_error * AREA_TO_METER_GAIN # Negative sign moves robot to *reduce* the error
        smoothed_command_z_m = SMOOTHING_FACTOR_Z * raw_command_z_m + (1 - SMOOTHING_FACTOR_Z) * smoothed_command_z_m

        # --- 3. PUBLISHING ---
        # The command is an offset from the initial pose
        target_pose_msg.pose.position.x = initial_pose_position[0] + smoothed_command_xy_m[0]
        target_pose_msg.pose.position.y = initial_pose_position[1] + smoothed_command_xy_m[1]
        target_pose_msg.pose.position.z = initial_pose_position[2] + smoothed_command_z_m

        target_pose_msg.header.stamp = rospy.Time.now()
        pub.publish(target_pose_msg)

        # --- 4. VISUALIZATION ---
        print("\033c", end="") # Clears console
        print(f"--- XY CONTROL ---")
        print(f"Pixel Error (px):   ({error_x}, {error_y})")
        print(f"Command Offset (m):    {np.round(smoothed_command_xy_m, 3)}")
        print(f"--- Z CONTROL ---")
        print(f"Area (Tgt | Cur):   ({target_pixel_area:.0f} | {marker_area:.0f})")
        print(f"Area Error (px^2):   {area_error:.0f}")
        print(f"Command Offset (m):    {smoothed_command_z_m:.3f}")
        print("---------------------------------------------")
        final_pos = [target_pose_msg.pose.position.x, target_pose_msg.pose.position.y, target_pose_msg.pose.position.z]
        print(f"Final Target Pos (m):  {np.round(final_pos, 3)}")

    # Display the final image
    cv2.imshow("ArUco 3D Pixel Control", frame)

if __name__ == '__main__':
    try:
        # Start video capture first, as it's needed for initialization
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            rospy.logerr(f"Cannot open camera at index {CAMERA_INDEX}. Exiting.")
            exit()

        # Initialize ROS, robot state, and target area
        control_pub = init_ros_and_robot_state(cap)

        print("\nStarting control loop. Press 'q' in the OpenCV window to quit.")

        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn("Failed to grab frame from camera.")
                continue

            process_frame(frame, control_pub)

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
