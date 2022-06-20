import os
import sys
import datetime
import cv2
import numpy as np
from matplotlib import pyplot as plt
import cbir


def find_keypoints_descriptors(image):
    ''' This function should take an image as input and return a list
    of keypoints as output
    '''
    # --- Add your code here ---
    # Creating the detector and setting some properties
    # orb = cv2.ORB.create()
    img = cv2.imread(image)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # Detecting the keypoints
    # keypoints = orb.detect(image)
    # --- End of custom code ---
    return keypoints, descriptors


def extract_descriptors(image, keypoints):
    ''' This function should take an image and keypoints as input and return a list
    of  ORB features as output
    '''
    # --- Add your code here ---
    orb = cv2.ORB.create(1500, nlevels=32)
    keypoints, features = orb.compute(image, keypoints)
    # --- End of custom code ---
    return features


'''
image = "/Users/liukelian-kelian/Desktop/test_pic/0.jpg"
kp, desc = find_keypoints_descriptors(image)
print(len(kp), desc.shape)
'''

# plt.figure(figsize=(15, 10))

start = datetime.datetime.now()

''' Train '''
jpg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/jpg")
dataset = cbir.Dataset(jpg_dir)
sift = cbir.descriptors.Sift()
descriptor = cbir.descriptors.Sift()
voc = cbir.encoders.VocabularyTree(n_branches=4, depth=4, descriptor=sift)
features = voc.extract_features(dataset)
voc.fit(features)

end = datetime.datetime.now()
print("build voc tree: {}".format(end - start))

'''
image_id_1 = "0"
image_id_2 = "3"

# read the images
image1 = dataset.read_image(image_id_1)
image2 = dataset.read_image(image_id_2)

em_1 = voc.embedding(image1)
em_2 = voc.embedding(image2)
print("em_1......")
print(em_1[0])
voc.subgraph(image_id_1)
print("Image embedding as graph: ")
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(20, 5))
ax[0].bar(np.arange(len(em_1)), em_1)
ax[0].set_title("Image 1 TF-IDF")
ax[1].bar(np.arange(len(em_2)), em_2)
_ = ax[1].set_title("Image 2 TF-IDF")
plt.tight_layout()
'''

''' Index '''
db = cbir.Database(dataset, encoder=voc)
db.index()
db.save()

end1 = datetime.datetime.now()
print("index and save: {}".format(end1 - end))

''' Query '''
image_names = dataset.image_paths
for img_name in image_names:
    print("query : {}".format(img_name))
    scores = db.retrieve(img_name)
    top4pairs = {k: scores[k] for k in list(scores)[:4]}
    print(top4pairs)

end2 = datetime.datetime.now()
print("total query: {}".format(end2 - end1))
# db.show_results(query, scores, figsize=(150, 50))
# plt.show()
