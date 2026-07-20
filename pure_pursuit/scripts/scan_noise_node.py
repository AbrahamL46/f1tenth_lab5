#!/usr/bin/env python3
"""
Add Gaussian noise to LaserScan ranges.

Subscribes to /scan (configurable) and publishes /scan_noisy (configurable).
Use the same --ros-args -p style as pure_pursuit_node.py.

Example:
  ros2 run pure_pursuit scan_noise_node.py --ros-args \
    -p range_std:=0.05 \
    -p input_topic:=/scan \
    -p output_topic:=/scan_noisy
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanNoise(Node):
  """Add Gaussian noise to laser scan ranges for research experiments."""

  def __init__(self):
    super().__init__('scan_noise_node')

    # Parameters (same pattern as pure_pursuit_node.py)
    self.declare_parameter('range_std', 0.05)
    self.declare_parameter('input_topic', '/scan')
    self.declare_parameter('output_topic', '/scan_noisy')

    self.range_std = self.get_parameter('range_std').get_parameter_value().double_value
    input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
    output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

    self.rng = np.random.default_rng()

    self.subscription = self.create_subscription(
      LaserScan,
      input_topic,
      self.scan_callback,
      10
    )
    self.publisher = self.create_publisher(LaserScan, output_topic, 10)

    self.get_logger().info(
      f'Scan noise started: {input_topic} -> {output_topic}, range_std={self.range_std}'
    )

  def scan_callback(self, msg):
    noisy_msg = LaserScan()

    # Copy metadata
    noisy_msg.header = msg.header
    noisy_msg.angle_min = msg.angle_min
    noisy_msg.angle_max = msg.angle_max
    noisy_msg.angle_increment = msg.angle_increment
    noisy_msg.time_increment = msg.time_increment
    noisy_msg.scan_time = msg.scan_time
    noisy_msg.range_min = msg.range_min
    noisy_msg.range_max = msg.range_max

    ranges = np.array(msg.ranges, dtype=np.float64)
    noise = self.rng.normal(0.0, self.range_std, size=ranges.shape)

    # Only add noise to finite ranges (leave inf/nan alone)
    valid = np.isfinite(ranges)
    ranges[valid] = ranges[valid] + noise[valid]
    ranges[valid] = np.clip(ranges[valid], msg.range_min, msg.range_max)

    noisy_msg.ranges = ranges.tolist()
    noisy_msg.intensities = msg.intensities

    self.publisher.publish(noisy_msg)


def main(args=None):
  rclpy.init(args=args)
  node = ScanNoise()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  node.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
