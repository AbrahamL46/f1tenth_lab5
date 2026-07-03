#!/usr/bin/env python3
import csv
import math
import os

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from visualization_msgs.msg import Marker


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def load_waypoints(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Waypoint file not found: {path}')

    points = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() in ('x', '#x'):
                continue
            if row[0].strip().startswith('#'):
                continue
            points.append([float(row[0]), float(row[1])])

    if len(points) < 2:
        raise ValueError('Waypoint file must contain at least two points')

    return np.asarray(points, dtype=np.float64)


class PurePursuit(Node):
  """Pure pursuit path tracking for F1TENTH."""

  def __init__(self):
    super().__init__('pure_pursuit_node')

    self.declare_parameter('waypoint_file', '')
    self.declare_parameter('lookahead_distance', 1.0)
    self.declare_parameter('wheelbase', 0.33)
    self.declare_parameter('max_steering', 0.4189)
    self.declare_parameter('velocity', 2.0)
    self.declare_parameter('pose_topic', '/ego_racecar/odom')
    self.declare_parameter('pose_type', 'odom')
    self.declare_parameter('drive_topic', '/drive')
    self.declare_parameter('frame_id', 'odom')

    waypoint_file = self.get_parameter('waypoint_file').get_parameter_value().string_value
    if not waypoint_file:
      raise RuntimeError('Set parameter waypoint_file to a CSV with x,y columns')

    self.lookahead = self.get_parameter('lookahead_distance').get_parameter_value().double_value
    self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
    self.max_steering = self.get_parameter('max_steering').get_parameter_value().double_value
    self.velocity = self.get_parameter('velocity').get_parameter_value().double_value
    pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
    pose_type = self.get_parameter('pose_type').get_parameter_value().string_value
    drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value
    self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

    self.waypoints = load_waypoints(waypoint_file)
    self.get_logger().info(f'Loaded {len(self.waypoints)} waypoints from {waypoint_file}')

    self.drive_pub = self.create_publisher(AckermannDriveStamped, drive_topic, 10)
    self.path_marker_pub = self.create_publisher(Marker, '/pure_pursuit/waypoints', 1)
    self.lookahead_marker_pub = self.create_publisher(Marker, '/pure_pursuit/lookahead', 1)

    if pose_type == 'pose_stamped':
      self.pose_sub = self.create_subscription(
        PoseStamped, pose_topic, self.pose_stamped_callback, 10)
    else:
      self.pose_sub = self.create_subscription(
        Odometry, pose_topic, self.odom_callback, 10)

    self.publish_path_marker()
    # Republish path so RViz still shows it if displays are added after startup
    self.create_timer(2.0, self.publish_path_marker)

  def publish_path_marker(self):
    marker = Marker()
    marker.header.frame_id = self.frame_id
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = 'pure_pursuit'
    marker.id = 0
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.15
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.pose.orientation.w = 1.0

    for x, y in self.waypoints:
      p = Point()
      p.x = float(x)
      p.y = float(y)
      p.z = 0.0
      marker.points.append(p)

    self.path_marker_pub.publish(marker)

  def publish_lookahead_marker(self, gx, gy):
    marker = Marker()
    marker.header.frame_id = self.frame_id
    marker.header.stamp = self.get_clock().now().to_msg()
    marker.ns = 'pure_pursuit'
    marker.id = 1
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD
    marker.pose.position.x = float(gx)
    marker.pose.position.y = float(gy)
    marker.pose.position.z = 0.0
    marker.pose.orientation.w = 1.0
    marker.scale.x = 0.25
    marker.scale.y = 0.25
    marker.scale.z = 0.25
    marker.color.r = 1.0
    marker.color.g = 0.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    self.lookahead_marker_pub.publish(marker)

  def get_lookahead_point(self, x, y):
    distances = np.hypot(self.waypoints[:, 0] - x, self.waypoints[:, 1] - y)
    closest_idx = int(np.argmin(distances))

    n = len(self.waypoints)
    for i in range(n):
      idx = (closest_idx + i) % n
      if distances[idx] >= self.lookahead:
        if i == 0:
          return self.waypoints[idx, 0], self.waypoints[idx, 1], distances[idx]

        prev_idx = (closest_idx + i - 1) % n
        d_prev = distances[prev_idx]
        d_curr = distances[idx]
        if d_curr <= d_prev:
          return self.waypoints[idx, 0], self.waypoints[idx, 1], d_curr

        t = (self.lookahead - d_prev) / (d_curr - d_prev)
        t = float(np.clip(t, 0.0, 1.0))
        gx = self.waypoints[prev_idx, 0] + t * (
          self.waypoints[idx, 0] - self.waypoints[prev_idx, 0])
        gy = self.waypoints[prev_idx, 1] + t * (
          self.waypoints[idx, 1] - self.waypoints[prev_idx, 1])
        actual_l = math.hypot(gx - x, gy - y)
        return gx, gy, max(actual_l, 1e-3)

    furthest_idx = int(np.argmax(distances))
    return (
      self.waypoints[furthest_idx, 0],
      self.waypoints[furthest_idx, 1],
      distances[furthest_idx],
    )

  def transform_to_vehicle_frame(self, gx, gy, x, y, yaw):
    dx = gx - x
    dy = gy - y
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return local_x, local_y

  def compute_steering(self, local_x, local_y, lookahead_dist):
    if local_x <= 0.0:
      return 0.0

    # gamma = 2*y / L^2 from lecture; steering = atan(wheelbase * gamma)
    gamma = 2.0 * local_y / (lookahead_dist * lookahead_dist)
    steering = math.atan(self.wheelbase * gamma)
    return float(np.clip(steering, -self.max_steering, self.max_steering))

  def pose_callback(self, x, y, yaw):
    gx, gy, lookahead_dist = self.get_lookahead_point(x, y)
    local_x, local_y = self.transform_to_vehicle_frame(gx, gy, x, y, yaw)
    steering = self.compute_steering(local_x, local_y, lookahead_dist)

    drive_msg = AckermannDriveStamped()
    drive_msg.header.stamp = self.get_clock().now().to_msg()
    drive_msg.header.frame_id = self.frame_id
    drive_msg.drive.steering_angle = steering
    drive_msg.drive.speed = self.velocity
    self.drive_pub.publish(drive_msg)
    self.publish_lookahead_marker(gx, gy)

  def odom_callback(self, msg):
    pose = msg.pose.pose
    yaw = quaternion_to_yaw(pose.orientation)
    self.pose_callback(pose.position.x, pose.position.y, yaw)

  def pose_stamped_callback(self, msg):
    pose = msg.pose
    yaw = quaternion_to_yaw(pose.orientation)
    self.pose_callback(pose.position.x, pose.position.y, yaw)


def main(args=None):
  rclpy.init(args=args)
  node = PurePursuit()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  node.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
