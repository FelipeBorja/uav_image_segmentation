from picamera import PiCamera
from time import sleep

camera = PiCamera()

image_num = 0

camera.start_preview()
while(True):
    image_num += 1
    camera.capture('/home/pi/Desktop/image%s.jpg' % image_num)
    sleep(10)

