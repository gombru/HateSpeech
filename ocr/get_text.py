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
        try:
            with open(imgname, 'rb') as f:
                ctxt = b64encode(f.read()).decode()
                img_requests.append({
                        'image': {'content': ctxt},
                        'features': [{
                            'type': 'TEXT_DETECTION',
                            'maxResults': 1
                        }]
                })

        except:
            print("Image not found")


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
RESULTS_DIR = '../../../datasets/HateSPic/HateSPic/img_text/'
results = {}
# makedirs(RESULTS_DIR, exist_ok=True)

api_key = 'AIzaSyB9TigeOiqzneipm-LQJUkHs_6xGd04oiM'
image_filenames = []

generated_base_path = '../../../datasets/HateSPic/HateSPicLabeler/generated_json/'
datasets = ['HateSPic/','SemiSupervised/','WZ-LS/']
base_path = '../../../datasets/HateSPic/'

for i,dataset in enumerate(datasets):
    for num_files, file in enumerate(os.listdir(generated_base_path + dataset)):
        id = json.load(open(generated_base_path + dataset + file, 'r'))['id']
        # If img_txt file already exists, skip
        if not os.isfile(RESULTS_DIR + str(id) + '.json'):
            image_filenames.append(base_path + 'HateSPic/img/' + str(id) + '.jpg')


# I do it image by image to don't fuck indices
for count, cur_image_filename in enumerate(image_filenames):
    print count
    try:
        response = request_ocr(api_key, [cur_image_filename])
        if response.status_code != 200 or response.json().get('error'):
            print(response.text)
        else:
            for idx, resp in enumerate(response.json()['responses']):
                if len(resp) == 0: continue
                img_txt = ""
                for detected_element in resp['textAnnotations']: # Loop for each text detected
                    img_txt += detected_element['description']

            if len(img_txt) > 3:
                # save to JSON file
                save_dict = {'img_text': img_txt}
                jpath = join(RESULTS_DIR, basename(cur_image_filename)[:-4] + '.json')
                with open(jpath, 'w') as f:
                    json.dump(save_dict, f)
    except:
        print("Error with image: " + cur_image_filename)

print("DONE")