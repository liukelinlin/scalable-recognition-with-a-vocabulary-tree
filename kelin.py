import os
import sys
import datetime
import json
import argparse
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


# db.show_results(query, scores, figsize=(150, 50))
# plt.show()


def main(args):
    data_dir = args.input_data
    gt_file = args.gt_file

    '''Train'''
    start = datetime.datetime.now()
    dataset = cbir.Dataset(data_dir)
    sift = cbir.descriptors.Sift()
    voc = cbir.encoders.VocabularyTree(n_branches=args.branch, depth=args.depth, descriptor=sift)
    features = voc.extract_features(dataset)
    voc.fit(features)

    end = datetime.datetime.now()
    print("build voc tree: {}".format(end - start))

    ''' Index '''
    db = cbir.Database(dataset, encoder=voc)
    db.index()
    db.save()

    end1 = datetime.datetime.now()
    print("index and save: {}".format(end1 - end))

    ''' Query '''
    func = lambda file_name: os.path.splitext(file_name)[0]
    results = {}
    image_names = dataset.image_paths
    for img_name in image_names:
        print("query : {}".format(img_name))
        scores = db.retrieve(img_name)
        top4pairs = {k: scores[k] for k in list(scores)[:4] if scores[k] < args.score_threshold}
        print(top4pairs)
        results[func(img_name)] = [func(k) for k, v in top4pairs.items() if k != img_name]
    end2 = datetime.datetime.now()
    print("total query: {}".format(end2 - end1))
    print(results)

    if gt_file:
        with open(gt_file, 'r') as fh:
            gt = json.loads(fh.read())
            top1_correct = 0
            top3_correct = 0
            for key, values in results.items():
                if not values: continue
                if key in gt and values[0] in gt[key]:
                    top1_correct += 1
                    top3_correct += 1
                    continue
                for val in values:
                    if key in gt and val in gt[key]:
                        top3_correct += 1

            print("total queries: {}, top1 hit: {}, top3 hit: {}".format(len(results), top1_correct, top3_correct))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo')
    parser.add_argument('-i', '--input_data', required=True, help='Input data directory')
    parser.add_argument('-gt', '--gt_file', required=False, help='gt json file')
    parser.add_argument('-br', '--branch', type=int, required=False, default=4, help='branch number')
    parser.add_argument('-dep', '--depth', type=int, required=False, default=4, help='depth number')
    parser.add_argument('-st', '--score_threshold', type=float, required=False, default=0.4, help='score threshold')

    args = parser.parse_args()
    main(args)
