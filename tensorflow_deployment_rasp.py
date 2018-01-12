from picamera import PiCamera
from time import sleep
import picamera.array
import time
import picamera
import io
import tensorflow as tf
from tensorflow import GraphDef
import numpy
import sys


with picamera.PiCamera() as camera:
	
	camera.start_preview()
	camera.resolution=(299, 299)
	rawcapture=picamera.array.PiRGBArray(camera, size=(299, 299))
	time.sleep(2)
	try:
		with open('/home/pi/Desktop/pi-camera_file/cyrrup_googlenet/cyrrup_data/cyrrup_labels.txt', 'r') as fin:
			labels=[line.rstrip('\n') for line in fin]
			
		with tf.gfile.FastGFile("/home/pi/Desktop/pi-camera_file/cyrrup_googlenet/cyrrup_data/cyrrup_inception.pb",'rb') as f:
			graph_def=tf.GraphDef()
			graph_def.ParseFromString(f.read())
			_=tf.import_graph_def(graph_def, name='')	
			    
		with tf.Session()as sess:
			softmax_tensor=sess.graph.get_tensor_by_name('final_result:0')
			print("software tensor started..")

			for i, filename in enumerate(camera.capture_continuous(rawcapture, format='bgr', use_video_port=True)):	
				decoded_image=filename.array
				print("decoded image..")
				predictions=sess.run(softmax_tensor,{'DecodeJpeg:0': decoded_image})
				start=time.time()
				print("session run...")
				prediction=predictions[0]
				prediction=prediction.tolist()
				max_value=max(prediction)
				max_index=prediction.index(max_value)
				predicted_label=labels[max_index]
				end=time.time()
				print("%s (%.2f%%)" %(predicted_label, max_value*100))
				print("The predicted time is:.3f%", (end-start))
				rawcapture.truncate(0)

	except Exception as e:
		print(str(e))
	finally:
		camera.stop_preview()
		print("Camera closed..")
