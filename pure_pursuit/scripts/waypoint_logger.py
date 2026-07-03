#!/usr/bin/env python3
import csv
import math
import os

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def quaternion_to_yaw(q):
  siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
  cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
  return math.atan2(siny_cosp, cosy_cosp)


class WaypointLogger(Node):
  def __init__(self):
    super().__init__('waypoint_logger')

    self.declare_parameter('output_file', 'waypoints.csv')
    self.declare_parameter('pose_topic', '/ego_racecar/odom')
    self.declare_parameter('pose_type', 'odom')
    self.declare_parameter('min_distance', 0.2)

    self.output_file = self.get_parameter('output_file').get_parameter_value().string_value
    self.min_distance = self.get_parameter('min_distance').get_parameter_value().double_value
    pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
    pose_type = self.get_parameter('pose_type').get_parameter_value().string_value

    self.last_x = None
    self.last_y = None
    self.points = []

    if os.path.exists(self.output_file):
      os.remove(self.output_file)

    with open(self.output_file, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(['x', 'y', 'theta'])

    if pose_type == 'pose_stamped':
      self.create_subscription(PoseStamped, pose_topic, self.pose_stamped_callback, 10)
    else:
      self.create_subscription(Odometry, pose_topic, self.odom_callback, 10)

    self.get_logger().info(f'Logging waypoints to {self.output_file}')

  def maybe_log(self, x, y, yaw):
    if self.last_x is not None:
      dist = math.hypot(x - self.last_x, y - self.last_y)
      if dist < self.min_distance:
        return

    self.last_x = x
    self.last_y = y
    self.points.append((x, y, yaw))

    with open(self.output_file, 'a', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow([f'{x:.4f}', f'{y:.4f}', f'{yaw:.4f}'])

  def odom_callback(self, msg):
    pose = msg.pose.pose
    yaw = quaternion_to_yaw(pose.orientation)
    self.maybe_log(pose.position.x, pose.position.y, yaw)

  def pose_stamped_callback(self, msg):
    pose = msg.pose
    yaw = quaternion_to_yaw(pose.orientation)
    self.maybe_log(pose.position.x, pose.position.y, yaw)


def main(args=None):
  rclpy.init(args=args)
  node = WaypointLogger()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  node.get_logger().info('Waypoint logging stopped')
  node.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
