from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random
import os

def str_to_seconds(input_str):
    data = input_str.split(':')
    hour = int(data[0]) - 8
    minute = int(data[1])
    second = int(data[2])
    return hour * 3600 + minute * 60 + second


if __name__ == '__main__':
    for i in range(8, 22):
        print(i)
    # res = {}
    # files = os.listdir('hotspot中sensor的访问情况/' + str(8) + '时间段')
    # for file in files:
    #     hotspot_num = file.split('.')[0]
    #     with open('hotspot中sensor的访问情况/' + str(8) + '时间段/' + file) as f:
    #         lines = f.readlines()
    #         res[hotspot_num] = lines
    # print(res)
    # a = np.array([[1, 2, 3, 6, 4],
    #               [1, 3, 12, 42, 44]
    #               ])
    # print(a.shape)
    # a = np.array([1, 2, 3, 6, 4]
    #               )
    # b = a[-1:]
    # print(b)
    # b = str(b[0]) + ',' + str(b[1])
    # print(b)
    # e = np.array([0, 51, 31])
    # f = np.array([1, 3, 4])
    # e = e[np.newaxis, :]
    # f = f[np.newaxis, :]
    #
    # b = np.array([4, 5, 1])
    # d = np.array([11, 33, 44])
    # b = b[np.newaxis, :]
    # d = d[np.newaxis, :]
    # c = np.c_[b, d]
    # c = np.vstack((c, np.c_[e, f]))
    # print(c)
    # rows, cols = a.shape
    # for row in rows:
    #     a[row]
    # a = [str(i) for i in range(1, 43)]
    # print(a)
    # label_binarizer = LabelBinarizer()
    # a_encoded = label_binarizer.fit_transform(a)
    # print(label_binarizer.transform(['3']))
    # a = np.array([[2]])
    # print(a[0][0])
    # a = list('23,4'.split(','))
    # a = list(map(int, a))
    # a = np.array(a)
    # print(a[0])
    # print(a[1])
    # print(a)
    # a = np.array([[1, 2, 3],
    #               [2, 4, 9],
    #               [1, 22, 33],
    #               [2, 44, 55]
    #               ])
    # b = np.random.choice(4, size=2)
    # print(b)
    # c = a[b, :]
    # print(c)
    # d = c[:, -2:]
    # print(d)

    # random.seed(1)
    # np.random.seed(1)
    # action_one_hot_encoded_8 = {}
    # # 存放每个文件中的所有hotspot
    # hotspots = []
    # # 存放每个hotspot 对应的最大等待时间
    # hotspot_max_staying_time = {}
    #
    # with open('五分钟/8.txt') as f:
    #     for line in f:
    #         data = line.strip().split(',')
    #         hotspots.append(data[0])
    #         hotspot_max_staying_time[data[0]] = data[1]
    # # 对hotpsots 进行独热编码
    # label_binarizer = LabelBinarizer()
    # hotspot_one_hot_encoded = label_binarizer.fit_transform(hotspots)
    # # 获取独热编码后的行和列
    # rows, cols = hotspot_one_hot_encoded.shape
    #
    # for row in range(rows):
    #     # 获得独热编码前的hotspot 编号，用这个编号在hotspot_max_staying_time 找到其最大等待时间
    #     hotspot = label_binarizer.inverse_transform(hotspot_one_hot_encoded[[row]])[0]
    #     for key, value in hotspot_max_staying_time.items():
    #         if hotspot == key:
    #             max_staying_time = int(hotspot_max_staying_time[key])
    #             # 找到最大等待时间以后，对独热编码后的当前行后面添加一个等待时间(从1到最大等待时间)
    #             for i in range(1, max_staying_time + 1):
    #                 tmp = hotspot_one_hot_encoded[row]
    #                 tmp = np.r_[tmp, i]
    #                 action_one_hot_encoded_8[hotspot + ',' + str(i)] = tmp
    #
    # action_list = list(action_one_hot_encoded_8.items())
    # for i in range(3):
    #     print(random.choice(action_list))
