from base64 import b64encode
from os import makedirs
from os.path import join, basename
import os
import json
import requests



def make_image_data_list(image_filenames):
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    img_requests = []
    for imgname in image_filenames:
        with open(imgname, 'rb') as f:
            ctxt = b64encode(f.read()).decode()
            img_requests.append({
                    'image': {'content': ctxt},
                    'features': [{
                        'type': 'TEXT_DETECTION',
                        'maxResults': 1
                    }]
            })
    return img_requests

def make_image_data(image_filenames):
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(image_filenames)
    return json.dumps({"requests": imgdict }).encode()


def request_ocr(api_key, image_filenames):
    response = requests.post(ENDPOINT_URL,
                             data=make_image_data(image_filenames),
                             params={'key': api_key},
                             headers={'Content-Type': 'application/json'})
    return response

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
RESULTS_DIR = '../../../datasets/HateSPic/HateSPic/img_text/txt/'
# makedirs(RESULTS_DIR, exist_ok=True)

api_key = 'AIzaSyB9TigeOiqzneipm-LQJUkHs_6xGd04oiM'
image_filenames = []

generated_base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']
base_path = '../../../datasets/HateSPic/'

for i,dataset in enumerate(datasets):
    for file in os.listdir(generated_base_path + dataset):
        id = json.load(open(generated_base_path + dataset + file, 'r'))['id']
        image_filenames.append(base_path + 'HateSPic/img/' + str(id) + '.jpg')

response = request_ocr(api_key, image_filenames)
if response.status_code != 200 or response.json().get('error'):
    print(response.text)
else:
    for idx, resp in enumerate(response.json()['responses']):
        # save to JSON file
        imgname = image_filenames[idx]
        jpath = join(RESULTS_DIR, basename(imgname) + '.json')
        with open(jpath, 'w') as f:
            datatxt = json.dumps(resp, indent=2)
            print("Wrote", len(datatxt), "bytes to", jpath)
            f.write(datatxt)

        # print the plaintext to screen for convenience
        print("---------------------------------------------")
        t = resp['textAnnotations'][0]
        print("    Bounding Polygon:")
        print(t['boundingPoly'])
        print("    Text:")
        print(t['description'])