#!/usr/bin/env python3

import rospy
import tf
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped
from threading import Lock

class SpaceNavController:
    def __init__(self):
        rospy.init_node('spacenav_teleop_controller')

        self.lock = Lock()
        self.current_pose = None

        # Subscribers
        rospy.Subscriber('/cartesian_pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/spacenav/twist', Twist, self.twist_callback)

        # Publisher
        self.pose_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=10)

        # Parameters
        self.linear_scale = rospy.get_param('~linear_scale', 0.01)
        self.angular_scale = rospy.get_param('~angular_scale', 0.01)
        self.deadband = rospy.get_param('~deadband', 0.01)
        self.publish_rate = rospy.get_param('~publish_rate', 30.0)

        self.last_pub_time = rospy.Time.now()

        rospy.loginfo("SpaceNav Teleop Controller Initialized")

    def pose_callback(self, msg):
        with self.lock:
            self.current_pose = msg

    def twist_callback(self, twist):
        now = rospy.Time.now()
        if (now - self.last_pub_time).to_sec() < 1.0 / self.publish_rate:
            return

        with self.lock:
            if self.current_pose is None:
                return

            # Deadband filter
            if all(abs(v) < self.deadband for v in [
                twist.linear.x, twist.linear.y, twist.linear.z,
                twist.angular.x, twist.angular.y, twist.angular.z]):
                return

            # Get current pose
            new_pose = PoseStamped()
            new_pose.header.stamp = now
            new_pose.header.frame_id = self.current_pose.header.frame_id
            new_pose.pose.position = self.current_pose.pose.position
            new_pose.pose.orientation = self.current_pose.pose.orientation

            # Transform linear velocity into current orientation frame
            orientation_q = [
                self.current_pose.pose.orientation.x,
                self.current_pose.pose.orientation.y,
                self.current_pose.pose.orientation.z,
                self.current_pose.pose.orientation.w
            ]
            rotation_matrix = tf.transformations.quaternion_matrix(orientation_q)[:3, :3]
            input_linear = np.array([twist.linear.x, twist.linear.y, twist.linear.z]) * self.linear_scale
            rotated_linear = rotation_matrix.dot(input_linear)

            new_pose.pose.position.x += rotated_linear[0]
            new_pose.pose.position.y += rotated_linear[1]
            new_pose.pose.position.z += rotated_linear[2]

            # Update orientation using relative angular input
            current_euler = tf.transformations.euler_from_quaternion(orientation_q)
            delta_euler = (
                twist.angular.x * self.angular_scale,
                twist.angular.y * self.angular_scale,
                twist.angular.z * self.angular_scale
            )
            new_euler = [sum(x) for x in zip(current_euler, delta_euler)]
            new_quaternion = tf.transformations.quaternion_from_euler(*new_euler)

            # Normalize quaternion
            new_quaternion /= np.linalg.norm(new_quaternion)

            new_pose.pose.orientation.x = new_quaternion[0]
            new_pose.pose.orientation.y = new_quaternion[1]
            new_pose.pose.orientation.z = new_quaternion[2]
            new_pose.pose.orientation.w = new_quaternion[3]

            self.pose_pub.publish(new_pose)
            self.last_pub_time = now

if __name__ == '__main__':
    try:
        controller = SpaceNavController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
