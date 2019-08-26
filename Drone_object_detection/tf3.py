# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:48:26 2019

@author: Durgesh
"""

def predict_single_image(self, image_file_path):
        """make object detection prediction on single image
        Args:
        image_file_path: Full path of image file to predict
        """
        image = Image.open(image_file_path)
        im_width, im_height = image.size
        image_np = np.array(image.getdata())[:,0:3].reshape((im_height, im_width, 3)).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        (boxes, scores, classes, num_detections) = self.sess.run(
            [self.boxes.outputs[0], self.scores.outputs[0], self.classes.outputs[0], self.num_detections.outputs[0]],
            feed_dict={self.image_tensor.outputs[0]: image_np_expanded})

        # Discard detections that do not meet the threshold score
        correct_prediction = [(s, np.multiply(b, [im_height, im_width, im_height, im_width]), c) 
                                                    for c, s, b in zip(classes[0], scores[0], boxes[0]) if (s > threshold and c in self.category_index)]
        if correct_prediction:
            scores, boxes, classes = zip(*correct_prediction)   
            draw = ImageDraw.Draw(image)
            for s, b, c in correct_prediction:
                draw_rectangle(draw, b, 'red', 5)
        else:
            scores, boxes, classes = [], [], []
        
        print("Number of detections: {}".format(len(scores)))


        print("\n".join("{0:<20s}: {1:.1f}%".format(self.category_index[c]['name'], s*100.) for (c, s, box) in zip(classes, scores, boxes)))

        return scores, classes, image, boxes