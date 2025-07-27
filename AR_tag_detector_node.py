import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, Point, Quaternion
import numpy as np
import cv2

class ARTagDetector_Node(Node):
    def __init__(self):
        super().__init__('AR_Tag_detector')
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pose_pub = self.create_publisher(Pose, '/mavros/local_position/pose', 10)
        self.tag_pub = self.create_publisher(Image, 'camera/detected_tag', 10)

        self.length = 0.266
        self.object_points = np.array([[-self.length/2, self.length/2, 0], [self.length/2, self.length/2, 0], [self.length/2, -self.length/2, 0], [-self.length/2, -self.length/2, 0]])
        f_x, f_y = 1182.644, 1182.828
        c_x, c_y = 251.1132, 324.621
        self.dist_coeff = np.array([[-0.0476501462, 1.46445018, 0.0119584430 , -0.00576903121, -6.08971241]])
        
        self.camera_matrix = np.array([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        # parameters = cv2.aruco.DetectorParameters()
        self.parameters = cv2.aruco.DetectorParameters_create()
        # self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    def image_callback(self, msg):
        img = msg.data
        aruco_dict = self.aruco_dict
        parameters = self.parameters
        camera_matrix = self.camera_matrix
        object_points = self.object_points
        dist_coeff = self.dist_coeff
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray_img, aruco_dict, parameters = parameters)
        # corners, ids, _ = detector.detectMarkers(gray_img)
        if ids is not None:
            self.get_logger().info(f'{len(ids)} tags Detected')
            for id, cnr in zip(ids, corners):
                cv2.aruco.drawDetectedMarkers(img, cnr, id)
                _, rvec, tvec = cv2.solvePnP(object_points, cnr.reshape((4,2)), camera_matrix, dist_coeff)
                P_m_c__c = np.hstack([tvec.T,np.array([[1.0]])])
                P_m_c__c = P_m_c__c.reshape(4,1)
                R_m2c = cv2.Rodrigues(rvec)[0]
                R_c2m = np.hstack([R_m2c.T, (np.array([0.0, 0.0, 0.0]).reshape(3,1))])
                R_c2m = np.vstack([R_c2m, np.array([0.0, 0.0, 0.0, 1.0])])
                P_c_m__m = -1 * (R_c2m @ P_m_c__c)
                P_c_m__m = P_c_m__m[:3]
                pose_msg = Pose()
                pose_msg.position = Point(x= P_c_m__m[0], y = P_c_m__m[1], z = P_c_m__m[2] )
                tr = R_m2c.T[0,0] + R_m2c.T[1,1] + R_m2c.T[2,2]
                if tr > 0:
                    w = 0.5 * np.sqrt(1+tr)
                    x = (R_m2c.T[2,1] + R_m2c.T[1,2])/(4*w)
                    y = (R_m2c.T[0,2] - R_m2c.T[2,0])/(4*w)
                    z = (R_m2c.T[1,0] - R_m2c.T[0,1])/(4*w)
                elif tr < 0 and R_m2c.T[0,0] >= R_m2c.T[1,1] and R_m2c.T[0,0] >= R_m2c.T[2,2]:
                    S = 2* np.sqrt(1 + R_m2c.T[0,0] - R_m2c.T[1,1] - R_m2c.T[2,2])
                    w = (R_m2c.T[0,2]- R_m2c.T[2,0])/S
                    x = 0.25 * S
                    y = (R_m2c.T[0,1]+ R_m2c.T[1,0])/S
                    z = (R_m2c.T[0,2] + R_m2c.T[2,0])/S
                elif tr < 0 and R_m2c.T[1,1] >= R_m2c.T[0,0] and R_m2c.T[1,1]>= R_m2c.T[2,2]:
                    S = 2 * np.sqrt(1 + R_m2c.T[1,1] - R_m2c.T[0,0] - R_m2c.T[2,2])
                    w = (R_m2c.T[0,2]-R_m2c.T[2,0])/S
                    x = (R_m2c.T[0,1]+ R_m2c.T[1,0])/S
                    y = 0.25 * S
                    z = (R_m2c.T[1,2]+ R_m2c.T[2,1])/S
                elif tr < 0 and R_m2c.T[2,2] >= R_m2c.T[1,1] and R_m2c.T[2,2] >= R_m2c.T[0,0]:
                    S = 2 * np.sqrt(1+ R_m2c.T[2,2] - R_m2c.T[0,0] - R_m2c.T[1,1])
                    w = (R_m2c.T[1,0] - R_m2c.T[0,1])/S
                    x = (R_m2c.T[0,2] + R_m2c.T[2,0])/S
                    y = (R_m2c.T[1,2] + R_m2c.T[2,1])/S
                    z = 0.25 * S
                pose_msg.orientation = Quaternion(x = x, y = y, z = z, w = w)
                self.pose_pub.publish(pose_msg)
                self.get_logger().info(f'Pose relative to ID {id}: {pose_msg}')
                self.tag_pub.publish(img)

        else:
            self.get_logger().info('No tags Detected')

def main(args = None):
    rclpy.init()
    node = ARTagDetector_Node()
    rclpy.spin()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


