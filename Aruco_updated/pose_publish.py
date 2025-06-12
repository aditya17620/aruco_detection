import cv2
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from collections import deque
from std_msgs.msg import Header

# ==============================================================================
# === TUNABLE PARAMETERS (EDIT THESE VALUES TO CHANGE BEHAVIOR) ===
# ==============================================================================

# --- Robot Control Parameters ---
# The ROS topic your robot listens to for Pose commands.
CONTROL_TOPIC = '/equilibrium_pose'

# The gain applied to the X and Y axes.
# A higher value will make the robot react more aggressively to the marker's movement.
# A value of 1.0 means no gain is applied.
# A value of 3.0 means a 10cm movement of the marker will result in a 30cm target command.
XY_GAIN = 3.0

# --- Filtering and Smoothing Parameters ---
# The smoothing factor (alpha) for the low-pass filter.
# It determines how much weight is given to the new measurement vs. the old, smoothed value.
# A smaller value means MORE smoothing but more lag (e.g., 0.1).
# A larger value means LESS smoothing but more jitter (e.g., 0.8).
SMOOTHING_FACTOR = 0.15

# The maximum distance (in meters) a marker can "jump" between two consecutive frames.
# If a new detection is further than this from the last known position, it's considered
# a noisy measurement and is ignored. This prevents wild, erroneous commands.
# Set this to a realistic value based on how fast your marker can possibly move.
JUMP_REJECTION_THRESHOLD = 0.5 # meters

# --- Physical and Camera Parameters ---
# The exact size of your ArUco marker's black square in meters.
MARKER_SIZE = 0.075 # meters

# Your camera's calibration matrix.
CAMERA_MATRIX = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
                          [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

# Your camera's distortion coefficients.
DIST_COEFFS = np.array([[0.09617148, -0.1634111, 0.00022246, -0.00056651, -0.00507073]], dtype=np.float32)

# ==============================================================================
# === CORE LOGIC (GENERALLY DO NOT NEED TO EDIT BELOW THIS LINE) ===
# ==============================================================================

# Real-world 3D coordinates of the marker's corners.
# The Z-coordinate is 0 because the marker is flat.
OBJECT_POINTS = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

# Global variables to store the smoothed pose between frames.
# They act as the "memory" for the filtering process.
smoothed_tvec = None
smoothed_rvec = None

def init_ros_publisher():
    """Initializes the ROS node and a single publisher for the control command."""
    rospy.init_node('aruco_to_pose_controller', anonymous=True)
    pub = rospy.Publisher(CONTROL_TOPIC, PoseStamped, queue_size=10)
    return pub

def publish_control_pose(pub, tvec, rvec):
    """Creates and publishes a PoseStamped message from tvec and rvec."""
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "camera_link" # Use a standard frame_id if you have one

    # The tvec (translation vector) directly maps to the position.
    msg.pose.position.x = tvec[0]
    msg.pose.position.y = tvec[1]
    msg.pose.position.z = tvec[2]

    # The rvec (rotation vector) must be converted to a quaternion for the pose message.
    # 'sxyz' is a common convention for the order of axes.
    quat = tf.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2], 'sxyz')
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]

    pub.publish(msg)

def detect_aruco(frame):
    """Detects ArUco markers in a given camera frame."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids

def process_frame(frame, pub):
    """The main processing pipeline for each frame."""
    global smoothed_tvec, smoothed_rvec
    corners, ids = detect_aruco(frame)

    if ids is not None:
        # --- 1. PERCEPTION: Get the raw physical pose of the first detected marker ---
        # This pose is in real-world units (meters) relative to the camera.
        image_points = corners[0].reshape((4, 2)).astype(np.float32)
        success, rvec_raw, tvec_raw = cv2.solvePnP(OBJECT_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS)
        
        if not success:
            return # If pose estimation fails, do nothing for this frame.

        # Reshape for easier math
        tvec_raw = tvec_raw.flatten()
        rvec_raw = rvec_raw.flatten()

        # --- 2. FILTERING (Part A): Reject large, noisy jumps ---
        # This step is crucial and works correctly because we are comparing real-world distances.
        if smoothed_tvec is not None:
            delta = np.linalg.norm(tvec_raw - smoothed_tvec)
            if delta > JUMP_REJECTION_THRESHOLD:
                print(f"⚠️  Rejected noisy detection (jump of {delta:.2f} meters)")
                return # Ignore this frame's data as it's likely noise.

        # --- 3. FILTERING (Part B): Smooth the pose using a low-pass filter ---
        # This reduces jitter and creates a more stable output.
        if smoothed_tvec is None:
            # On the very first detection, initialize the smoothed values directly.
            smoothed_tvec = tvec_raw
            smoothed_rvec = rvec_raw
        else:
            # Apply exponential moving average for smoothing.
            smoothed_tvec = SMOOTHING_FACTOR * tvec_raw + (1 - SMOOTHING_FACTOR) * smoothed_tvec
            smoothed_rvec = SMOOTHING_FACTOR * rvec_raw + (1 - SMOOTHING_FACTOR) * smoothed_rvec

        # --- 4. CONTROL: Apply gain to the clean, smoothed pose to create the final command ---
        # We work on a copy so the `smoothed_tvec` remains the clean, real-world value.
        control_tvec = smoothed_tvec.copy()
        control_rvec = smoothed_rvec.copy()
        
        # Apply the gain to the X and Y axes.
        control_tvec[0] *= XY_GAIN
        control_tvec[1] *= XY_GAIN

        control_rvec[0] *= XY_GAIN
        control_rvec[1] *= XY_GAIN
        control_rvec[2] *= XY_GAIN
        # NOTE: Z-axis (distance) is not scaled here, but you could add a Z_GAIN if needed.

        # --- 5. PUBLISH: Send the final, gained control pose to the robot ---
        publish_control_pose(pub, control_tvec, smoothed_rvec)

        # --- 6. VISUALIZATION & DEBUGGING ---
        # Draw the axes on the video feed based on the RAW detection for instant feedback.
        cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec_raw, tvec_raw.reshape(3, 1), 0.05)
        # Print useful info to the console.
        print("\033c", end="") # Clears the console for clean output
        # print(f"REAL Smoothed Pose (m): {np.round(smoothed_tvec, 3)}")
        print(f"CONTROL Cmd Pose (Gained): {np.round(control_tvec, 3)}")
        print(f"XY Gain: {XY_GAIN}, Smoothing: {SMOOTHING_FACTOR}")

    # Display the final image
    cv2.imshow("ArUco Pose Control", frame)


if __name__ == '__main__':
    # Initialize ROS publisher
    control_pub = init_ros_publisher()
    # Start video capture
    cap = cv2.VideoCapture(0)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
        
        process_frame(frame, control_pub)

        # Allow exiting with the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
