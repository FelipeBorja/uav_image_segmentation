from GPSPhoto import gpsphoto

photo = gpsphoto.GPSPhoto("RenderImageCropped.jpg")


info = gpsphoto.GPSInfo((35.104860, -106.628915), alt=10)

photo.modGPSData(info, 'RenderImageCropped3.jpg')



