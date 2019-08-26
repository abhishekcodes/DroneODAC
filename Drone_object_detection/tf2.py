# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:47:14 2019

@author: Durgesh
"""

def draw_rectangle(draw, coordinates, color, width=1):
    for i in range(width):
        rect_start = (coordinates[1] - i, coordinates[0] - i)
        rect_end = (coordinates[3] + i, coordinates[2] + i)
        draw.rectangle((rect_start, rect_end), outline = color)
        
        
class ObjectDetectionPredict():
    """class method to Load tf graph and 
    make prediction on test images using predict function
    """
    
    def __init__(self, model_name):
        """ Downloads, initialize the tf model graph and stores in Memory
        for prediction
        """
        download_model(model_name)
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, 
                                                                    max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.load_graph(model_name)
    
    def load_graph(self, model_name):
        """ Loads the model into RAM
        Args:
        model_name: name of model to load
        """
        model_file = model_name + '/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with self.detection_graph.as_default():
            tf.import_graph_def(graph_def)

        self.sess = tf.Session(graph=self.detection_graph)
        
        self.image_tensor = self.detection_graph.get_operation_by_name('import/image_tensor')
        self.boxes = self.detection_graph.get_operation_by_name('import/detection_boxes')
        self.scores = self.detection_graph.get_operation_by_name('import/detection_scores')
        self.classes = self.detection_graph.get_operation_by_name('import/detection_classes')
        self.num_detections = self.detection_graph.get_operation_by_name('import/num_detections')
        return 0