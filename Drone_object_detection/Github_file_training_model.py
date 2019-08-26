# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:51:08 2019

@author: Durgesh
"""


## change the folder path if needed
image_folder = "../data/Images"
annotation_folder = "../data/Annotations"
object_list = ['person'] ## will refer as categories
extentions = ['.jpeg'] ## list of image extension

AUTH_KEY = "" ## put your api key here

### imports

import requests, json, os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from multiprocessing import Pool

#### CONSTANTS
BASE_URL = "https://app.nanonets.com/api/v2/ObjectDetection/Model/"
MODEL_URL = "https://app.nanonets.com/api/v2/ObjectDetection/Models/"

def print_result(result):
    model_id, model_type, categories, state = (result["model_id"], result["model_type"], result["categories"], result["state"])
    
    print ("MODEL ID : %s"%model_id)
    for cat in categories:
        print ("Label: %s :: Count: %d"%(cat['name'], cat['count']))
    
    print ("CURRENT STATE :: %s"%(result['status']))
    print ("Model Accuracy : %f"%(result['accuracy']))

def create_new_model(categories):
    """ function will create a new model architecture for training
    
    Args:
    categories: List of objects you want to predict
    
    return:
    model_id: a unique reference to new created model
    """
    
    
    payload = json.dumps({"categories" : categories})
    headers = {
        'Content-Type': "application/json",
        }

    response = requests.request("POST", BASE_URL, headers=headers, auth=requests.auth.HTTPBasicAuth(AUTH_KEY, ''), data=payload)

    result = json.loads(response.text)
    print_result(result)
    model_id, model_type, categories = (result["model_id"], result["model_type"], result["categories"])
    return model_id

model_id = create_new_model(object_list)
print model_id


ef get_model_info(model_id):
    """function to get information about model at any time
    Args:
    model_id: unique model_id generated at model creation time
    """
    
    response = requests.request('GET', '%s%s'%(BASE_URL, model_id), auth= requests.auth.HTTPBasicAuth(AUTH_KEY, ''))
    result = json.loads(response.text)
    print_result(result)
    model_id, model_type, categories, state = (result["model_id"], result["model_type"], result["categories"], result["state"])
    return model_id, model_type, categories, state


def get_label(s):
    return s.lower().replace(' ', '')

def get_annotaion_file_name(image_file):
    return os.path.join(annotation_folder, "%s.gt"%(image_file))

def valid_bb(bb):
    xmin, ymin, xmax, ymax = bb
    
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    
    if xmin >= xmax: return False
    if ymin >= ymax: return False
    
    return [xmin, ymin, xmax, ymax]

def create_annotation_to_object(image_file):
    annotation_file_path = get_annotaion_file_name(image_file)
    if not os.path.isfile(annotation_file_path): return False
    
    bb_list = filter(None, open(annotation_file_path).read().split('\n'))
    object_list = []
    for bb_info in bb_list:
        try:
            xmin, ymin, xmax, ymax, label_text = bb_info.split(' ')
            label_text = get_label(label_text)
            bb = valid_bb(map(int, [xmin, ymin, xmax, ymax]))
            if bb:
                bndbox = {}
                bndbox['xmin'] = bb[0]
                bndbox['ymin'] = bb[1]
                bndbox['xmax'] = bb[2]
                bndbox['ymax'] = bb[3]
                object_list.append({'name': label_text, 'bndbox': bndbox})
        except:
            print ("Error in BB : %s for Image : %s"%(bb_info, image_file))
    return object_list

def upload_file_data(f):
    if not f.endswith(tuple(extentions)): return 0
    print (f)
    filename = os.path.join(image_folder, f)
    file = open(filename, 'rb')
    
    object_data = create_annotation_to_object(f)
    if not object_data: return 0
    object_data = json.dumps(object_data)
    
    data = {'file' : file,
                'data' :('', '[{"filename":"%s", "object": %s}]'%(f, object_data)),
                'modelId' :('', '%s'%model_id)}
    response = requests.post('%s%s/UploadFile/'%(BASE_URL, model_id), auth=requests.auth.HTTPBasicAuth(AUTH_KEY, ''), files=data)
    return 1
    
def upload_objects_by_file(model_id):
    pool = Pool(4)
    
    image_count = 0
    print ("uploading images....")
    results = pool.map(upload_file_data, os.listdir(image_folder))
    return get_model_info(model_id)

_ = upload_objects_by_file(model_id)


def train_model(model_id):

    headers = {'authorization': 'Basic %s'%AUTH_KEY}
    querystring = {'modelId': model_id}
    response = requests.request('POST', '%s%s/Train/'%(BASE_URL, model_id), headers=headers, auth=requests.auth.HTTPBasicAuth(AUTH_KEY, ''), params=querystring)
    print ("training started .... ")
    print_result(json.loads(response.text))
    
train_model(model_id)

_ = get_model_info(model_id)

test_image_path = "/Users/atish/Downloads/blog_data/test_images/image1.jpeg"

from PIL import Image, ImageFont, ImageDraw

def predict_single_image(model_id, filepath):
    
    url = '%s%s/LabelFile/'%(BASE_URL, model_id)
    if not test_image_path.endswith(tuple(extentions)):
        print ("provide image with correct extentions")
        return 0
    data = {'file': open(filepath, 'rb'),
            'modelId': ('', '%s'%model_id)}
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(AUTH_KEY, ''), files=data)
    result = json.loads(response.text)
    return result

def visualize_result(result, image_file):
    if result['message'] != "Success":
        print ("Error in Prediction")
        return
    
    img = Image.open(image_file)
    draw = ImageDraw.Draw(img)
    for res in result['result']:
        for pred in res['prediction']:
            score = pred['score']
            label = pred['label']
            bb = [pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax']]
            draw.rectangle(bb, outline='red')
            draw.text((pred['xmin']+10,  pred['ymin']+10), "%s:%0.2f"%(label[:3], score), (0,0,0))
    return img


###### need to set model id if you want to predict on previously trained model

# model_id = 'aea81d42-4839-46f7-b0bc-900bf6f4fc36'    ##example mode_id


result = predict_single_image(model_id, test_image_path)
visualize_result(result, test_image_path)

