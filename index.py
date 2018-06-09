from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
if __name__ == '__main__':
    data_1 = ['北京', '上海', '广州']
    data_2 = ['成都', '杭州']
    def label_to_binary_list(label_binarizer, label):
        return label_binarizer.transform([label])[0].tolist()
    label_binarizer = LabelBinarizer()
    one_hot_encoded_2 = label_binarizer.fit_transform(data_2)
    print(one_hot_encoded_2)

    # print(one_hot_encoded)
    # print(label_binarizer.transform(['北京']))
    # a = np.array([[1, 2, 2],
    #             [ 2, 3, 2]
    #              ])
    # print(a[[0, 1]])
    # sensor_name = []
    # state = []
    # files = os.listdir('sensor数据四秒')
    # for i in range(48):
    #     state.append(i)
    #
    # for file in files:
    #     sensor = file.split('.')[0]
    #     sensor_name.append(sensor)
    # label_binarizer = LabelBinarizer()
    # sensor_name_one_encoded = label_binarizer.fit_transform(sensor_name)

    # for sensor in sensor_name:
    #     sensor_one_encoded = label_binarizer.transform([sensor])
    #     state.append(sensor_one_encoded)
    #     state.append(1)
    #     break
    def contains(small, big):
        for i in range(len(big) - len(small) + 1):
            for j in range(len(small)):
                if big[i + j] != small[j]:
                    break
            else:
                return i, i + len(small)
        return False


    # print(state)
    # sensor_one_encoded = label_binarizer.transform([sensor_name[0]])[0].tolist()
    # for i in sensor_one_encoded:
    #     state.append(i)
    #
    # start, end = contains(sensor_one_encoded, state)
    # print('start ', start)
    # print('end ', end)
    # print(len(state))