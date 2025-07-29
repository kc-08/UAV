import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import time

class WebCamNode(Node):
    def __init__(self):
        super().__init__("web_cam_node")
        self.bridge = CvBridge()
        self.cam_pub = self.create_publisher(Image, "web_camera/image_raw", 10)
        self.frame_count = 0
        self.last_successful_frame = None

        self.get_logger().info("Connecting to webcam stream...")
        self.video = cv2.VideoCapture("udp://@:1234", cv2.CAP_FFMPEG)
        
        # Set buffer size to reduce latency
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Check if the video capture is opened
        if not self.video.isOpened():
            self.get_logger().error("Failed to open UDP stream")
            return

        self.get_logger().info("Video capture opened, waiting for valid frames...")
        
        # Wait for a valid frame (with longer timeout)
        valid_frame_found = False
        for attempt in range(50):  # Increased attempts
            ret, frame = self.video.read()
            self.get_logger().info(f"Attempt {attempt + 1}: ret={ret}, frame={'None' if frame is None else f'shape={frame.shape}'}")
            
            if ret and frame is not None and len(frame.shape) == 3 and frame.shape[0] > 0 and frame.shape[1] > 0:
                self.get_logger().info(f"Webcam stream locked in! Frame shape: {frame.shape}")
                self.last_successful_frame = frame.copy()
                valid_frame_found = True
                break
            time.sleep(0.1)
        
        if not valid_frame_found:
            self.get_logger().error("Failed to lock onto webcam stream after 50 attempts.")
            self.get_logger().info("Will continue and try to use test pattern...")

        # Use a slower timer to avoid overwhelming the system
        self.timer = self.create_timer(0.016, self.timer_callback)  # ~60 Hz

    def timer_callback(self):
        ret, frame = self.video.read()
        
        if ret and frame is not None and len(frame.shape) == 3 and frame.shape[0] > 0 and frame.shape[1] > 0:
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Log every 30 frames (~0.5 second)
                self.get_logger().info(f"Got frame #{self.frame_count}: {frame.shape}")
            
            self.last_successful_frame = frame.copy()
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.cam_pub.publish(img_msg)
        else:
            # If no valid frame, try to use the last successful frame or create a test pattern
            if self.last_successful_frame is not None:
                if self.frame_count % 30 == 0:
                    self.get_logger().warn("Using last successful frame")
                img_msg = self.bridge.cv2_to_imgmsg(self.last_successful_frame, "bgr8")
                self.cam_pub.publish(img_msg)
            else:
                # Create a test pattern
                self.publish_test_pattern()
    
    def publish_test_pattern(self):
        """Publish a test pattern when no valid stream is available"""
        import numpy as np
        
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Simple color pattern
        frame[:, :width//3] = [255, 0, 0]      # Blue
        frame[:, width//3:2*width//3] = [0, 255, 0]  # Green  
        frame[:, 2*width//3:] = [0, 0, 255]    # Red
        
        # Add text
        cv2.putText(frame, "NO UDP STREAM", (width//2 - 120, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Check stream source", (width//2 - 140, height//2 + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if self.frame_count % 60 == 0:  # Log every 60 calls (~2 seconds)
            self.get_logger().warn("Publishing test pattern - no valid UDP stream")
        
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.cam_pub.publish(img_msg)

    def destroy_node(self):
        if hasattr(self, 'video') and self.video is not None:
            self.video.release()
        super().destroy_node()

def main():
    rclpy.init()
    node = WebCamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()