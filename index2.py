import os
from sklearn.preprocessing import LabelBinarizer
import numpy as np
if __name__ == '__main__':
    label_binarizer = LabelBinarizer()
    path = '五分钟/'
    files = os.listdir(path)
    hotspot_max_staying_time = {}
    hotspots = []
    file = files[0]
    print(file)
    with open(path + file) as f:
        for line in f:
            data = line.strip().split(',')
            hotspots.append(data[0])
            hotspot_max_staying_time[data[0]] = data[1]
    hotspot_one_hot_encoded = label_binarizer.fit_transform(hotspots)

    rows, cols = hotspot_one_hot_encoded.shape
    hotspot_max_staying_time_one_hot_encoded = np.zeros(shape=[rows * 12, cols + 1], dtype=np.int32)
    index = 0
    for row in range(rows):
        hotspot = label_binarizer.inverse_transform(hotspot_one_hot_encoded[[row]])
        for key, value in hotspot_max_staying_time.items():
            if hotspot == key:
                max_staying_time = int(hotspot_max_staying_time[key])
                for i in range(1, max_staying_time + 1):
                    # a = hotspot_one_hot_encoded[row]
                    # a = np.r_[a, i]
                    # print(a)
                    a = hotspot_one_hot_encoded[row]
                    a = np.r_[a, i]
                    hotspot_max_staying_time_one_hot_encoded[index] = a
                    index += 1
    hotspot_max_staying_time_one_hot_encoded = hotspot_max_staying_time_one_hot_encoded[:index]
