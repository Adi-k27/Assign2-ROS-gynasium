import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import irobot_create_msgs.msg
from irobot_create_msgs.msg import StopStatus
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CreateRedBallEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 0}
    
    def __init__(self, render_mode=None):
        # Define observation space
        self.observation_space = spaces.Discrete(641)  # 0 to 640 pixels for x-axis

        # Define action space
        self.action_space = spaces.Discrete(641)  # 0 to 640 pixels for rotation
        
        # Initialize ROS node
        rclpy.init()

        # Instantiate RedBall node
        self.redball = RedBall()

        # Initialize episode step count
        self.step_count = 0

        # Flag to indicate if the ball has been initially detected
        self.ball_initially_detected = False

    def reset(self, seed=None, options=None):
        # Reset step count
        self.step_count = 0
        
        # Return the most recent observation (x-axis pixel value)
        return self.redball.redball_position, {}

    def step(self, action):
        # Convert action to rotation angle (in radians)
        rotation_angle = (action - 320) / 320 * (3.14 / 2)
        
        # Publish Twist message for rotation
        self.redball.set_twist_from_action(rotation_angle)
        
        # Reset stop flag
        self.redball.create3_is_stopped = False
        
        # Spin ROS until Create 3 stops moving or a certain time has elapsed
        max_rotation_steps = 400
        rotation_steps = 0
        while not self.redball.create3_is_stopped and rotation_steps < max_rotation_steps:
            rclpy.spin_once(self.redball)
            rotation_steps += 1
        
        # Check if the ball is detected
        ball_detected = self.redball.ball_detected
        
        # If the ball was initially undetected, rotate the robot to search for it
        if not self.ball_initially_detected and not ball_detected:
            rotation_angle = np.random.uniform(-1, 1) * (3.14 / 2)
            self.redball.set_twist_from_action(rotation_angle)
            # Reset step count
            self.step_count = 0
            # Increment rotation steps
            rotation_steps = 0
            while not self.redball.create3_is_stopped and rotation_steps < max_rotation_steps:
                rclpy.spin_once(self.redball)
                rotation_steps += 1
            ball_detected = self.redball.ball_detected
        
        # Increment step count
        self.step_count += 1
        
        # Return observation, reward, done, and info
        return self.redball.redball_position, self.reward(ball_detected), self.step_count == 100, False, {"info": None}

    def render(self):
        # Do nothing for now
        pass

    def close(self):
        # Shutdown ROS node
        self.redball.destroy_node()
        rclpy.shutdown()

    def reward(self, ball_detected):
        if ball_detected:
            # Positive reward when the ball is detected
            return 100
        else:
            # Negative reward when the ball is not detected
            return -1


class RedBall(Node):
    """
    A Node to analyze red balls in images and publish the results
    """
    def __init__(self):
        super().__init__('redball')
        self.subscription = self.create_subscription(
            Image,
            'custom_ns/camera1/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # A converter between ROS and OpenCV images
        self.br = CvBridge()
        self.target_publisher = self.create_publisher(Image, 'target_redball', 10)
        self.twist_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        # Initialize red ball position
        self.redball_position = 320  # Start from the middle

        # Flag to track if Create 3 is stopped
        self.create3_is_stopped = False
        
        # Flag to track if the ball is detected
        self.ball_detected = False
        
        # Subscriber to stop status
        self.stop_subscriber = self.create_subscription(
            irobot_create_msgs.msg.StopStatus,
            'stop_status',
            self.stop_callback,
            10)

    def stop_callback(self, msg):
        # Check if Create 3 is stopped
        if msg.is_stopped:
            self.create3_is_stopped = True

    def listener_callback(self, msg):
        frame = self.br.imgmsg_to_cv2(msg)

        # convert image to BGR format (red ball becomes blue)
        hsv_conv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        bright_red_lower_bounds = (110, 100, 100)
        bright_red_upper_bounds = (130, 255, 255)
        bright_red_mask = cv2.inRange(hsv_conv_img, bright_red_lower_bounds, bright_red_upper_bounds)

        blurred_mask = cv2.GaussianBlur(bright_red_mask,(9,9),3,3)
        # some morphological operations (closing) to remove small blobs
        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        eroded_mask = cv2.erode(blurred_mask,erode_element)
        dilated_mask = cv2.dilate(eroded_mask,dilate_element)

        # on the color-masked, blurred and morphed image I apply the cv2.HoughCircles-method to detect circle-shaped objects
        detected_circles = cv2.HoughCircles(dilated_mask, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=20, minRadius=2, maxRadius=2000)
        the_circle = None
        if detected_circles is not None:
            for circle in detected_circles[0, :]:
                circled_orig = cv2.circle(frame, (int(circle[0]), int(circle[1])), int(circle[2]), (0,255,0),thickness=3)
                the_circle = (int(circle[0]), int(circle[1]))
                self.ball_detected = True
            self.target_publisher.publish(self.br.cv2_to_imgmsg(circled_orig))
            self.ball_initially_detected = True  # Set the flag to True when the ball is first detected
            self.get_logger().info('ball detected')
        else:
            self.ball_detected = False
            self.get_logger().info('no ball detected')

    def set_twist_from_action(self, action):
        # Translate action to Twist message
        twist_msg = Twist()
        twist_msg.angular.z = action # Convert to radians
        self.twist_publisher.publish(twist_msg)

