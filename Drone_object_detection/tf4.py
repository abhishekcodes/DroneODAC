# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:48:54 2019

@author: Durgesh
"""

prediction_class = ObjectDetectionPredict(model_name=MODEL_NAME)
all_test_images = glob.glob("%s/*.jpg"%(test_image_dir.rstrip('/')))
print("Number of Test Images : ", len(all_test_images))

for image in all_test_images:
  #### boxes are in [ymin. xmin. ymax, xmax] format
  scores, classes, img, boxes = prediction_class.predict_single_image(image)
  image_name, ext = image.rsplit('.', 1)
  new_image_name = image_name + "_prediction." + ext
  img.save(new_image_name)
prediction_class.sess.close()