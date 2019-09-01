import json
import os
import warnings

import numpy as np
import scipy
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

splits = '/mnt/creeper/grad/kumarkm/first_year/crowd_incremental/dataset/Penguin/utils/Splits_2016_07_11/imdb.json'
# folders = os.listdir('/mnt/creeper/grad/kumarkm/first_year/crowd_incremental/dataset/Penguin/images/')

images = {}
splits = '/mnt/creeper/grad/kumarkm/first_year/crowd_incremental/dataset/Penguin/dataset/'

split = [os.path.join(splits, i) for i in os.listdir(splits)]

for sp in split:
    print(sp)
    folder = [os.path.join(sp, f) for f in os.listdir(sp)]
    for f in folder:
        imgs = os.listdir(f)
        for img in imgs:
            images[img.split('.')[0]] = os.path.join(f, img)


# this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
def gaussian_filter_density(gt):
    #     print gt.shape
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(zip(np.nonzero(gt)[1], np.nonzero(gt)[0]))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    #     print 'generate density...'
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    #     print 'done.'
    return density


from glob import glob

all_json = '../../dataset/Penguin/utils/CompleteAnnotations_2016-07-11/*.json'
ajsons = glob(all_json)
current_jsons = ajsons[12:16]
skipped_files = []
for js in current_jsons:
    print("Processing: {}".format(js))
    with open(js) as fp:
        data = json.load(fp)
    for idx, d in enumerate(data['dots']):
        try:
            path = os.path.join('../../dataset/Penguin/csvs', d['imName'] + '.csv')
            if os.path.exists(path):
                print("Path: {}, exists!".format(path))
                continue
            if (idx + 1) % 10 == 0:
                print("Progress: {}/{}".format(idx + 1, len(data['dots'])))
            if d['xy'] is not None:
                img = plt.imread(images[d['imName']])
                k = np.zeros((img.shape[0], img.shape[1]))
                gt = d['xy'][0]
                if gt is not None:
                    for i in range(0, len(gt)):
                        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                            k[int(gt[i][1]), int(gt[i][0])] = 1
                    k = gaussian_filter_density(k)
                    np.savetxt(path, k, delimiter=",")
        except:
            print("Skipped: {}".format(d['imName']))
            skipped_files.append(d['imName'])

with open('skipped_four.txt', 'w') as fp:
    fp.write('\n'.join(skipped_files))
