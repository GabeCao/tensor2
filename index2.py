import os
from sklearn.preprocessing import LabelBinarizer
if __name__ == '__main__':
    path = '五分钟/'
    files = os.listdir(path)
    max_staying_time = {}
    for file in files:
        name = file.split('.')[0]
        lines = []
        with open(path + file) as f:
            for line in f:
                lines.append(line.strip())
        max_staying_time[name] = lines
    print(max_staying_time[str(8)])
    label_binarizer = LabelBinarizer()
    one = label_binarizer.fit_transform(max_staying_time[str(8)])
    row, col = one.shape
    for i in range(row):
        print('......')
        print(one[i])