#!/home/autonav-linux/catkin_ws/src/yolov5_ROS/scripts/yolov5/bin/python3

import rospy
from std_msgs.msg import String

def printer(data):
    print(data.data)

def listener():
    
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('classes', String, printer)
    rospy.spin()

if __name__ == "__main__":
    listener()