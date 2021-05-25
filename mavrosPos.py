import rospy

from sensor_msgs.msg import NavSatFix


def callback(data):
	latitude = data.latitude
	longitude = data.longitude
	altitude = data.altitude






def listener():
	rospy.init_node('listener', anonymous=True)
	rospy.Subscriber("/mavros/global_position/global", NavSatFix, callback)

	rospy.spin()


if __name__ == '__main__':
    listener()



