from sklearn.preprocessing import LabelBinarizer
import numpy as np
import random

def str_to_seconds(input_str):
    data = input_str.split(':')
    hour = int(data[0]) - 8
    minute = int(data[1])
    second = int(data[2])
    return hour * 3600 + minute * 60 + second


if __name__ == '__main__':
    a = np.array([[1, 2, 3],
                  [2, 4, 9]
                  ])
    print(a)
    print(a[:, 2])

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
