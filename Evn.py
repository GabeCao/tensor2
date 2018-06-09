from sklearn.preprocessing import LabelBinarizer
import numpy as np
import os
class Evn:
    def __init__(self):
        self.state = []
        # sensor 和 mc的能量信息
        self.sensors_mobile_charger = {}
        # 保存所有的sensor 编号
        self.sensors = []
        # 初始化self.sensors_mobile_charger 和 self.sensors
        self.set_sensors_mobile_charger()
        # 对剩余寿命进行独热编码
        self.rl = ['Greater than the threshold value, 0', 'Smaller than the threshold value, 1', 'dead, -1']
        self.rl_label_binarizer = LabelBinarizer()
        self.rl_one_hot_encoded = self.rl_label_binarizer.fit_transform(self.rl)
        # 对是否属于hotspot 独热编码
        self.belong = ['1', '0']
        self.belong_label_binarizer = LabelBinarizer()
        self.belong_one_hot_encoded = self.belong_label_binarizer.fit_transform(self.belong)
        # 对action 独热编码,用字典表示,key: action, value: action的编码结果
        self.action_one_hot_encoded_8 = {}
        self.action_one_hot_encoded_9 = {}
        self.action_one_hot_encoded_10 = {}
        self.action_one_hot_encoded_11 = {}
        self.action_one_hot_encoded_12 = {}
        self.action_one_hot_encoded_13 = {}
        self.action_one_hot_encoded_14 = {}
        self.action_one_hot_encoded_15 = {}
        self.action_one_hot_encoded_16 = {}
        self.action_one_hot_encoded_17 = {}
        self.action_one_hot_encoded_18 = {}
        self.action_one_hot_encoded_19 = {}
        self.action_one_hot_encoded_20 = {}
        self.action_one_hot_encoded_21 = {}

    def action_to_one_hot_encoded(self, path, file, action_one_hot_encoded):
        # 存放每个文件中的所有hotspot
        hotspots = []
        # 存放每个hotspot 对应的最大等待时间
        hotspot_max_staying_time = {}
        with open(path + file) as f:
            for line in f:
                data = line.strip().split(',')
                hotspots.append(data[0])
                hotspot_max_staying_time[data[0]] = data[1]
        # 对hotpsots 进行独热编码
        label_binarizer = LabelBinarizer()
        hotspot_one_hot_encoded = label_binarizer.fit_transform(hotspots)
        # 获取独热编码后的行和列
        rows, cols = hotspot_one_hot_encoded.shape

        for row in range(rows):
            # 获得独热编码前的hotspot 编号，用这个编号在hotspot_max_staying_time 找到其最大等待时间
            hotspot = label_binarizer.inverse_transform(hotspot_one_hot_encoded[[row]])
            for key, value in hotspot_max_staying_time.items():
                if hotspot == key:
                    max_staying_time = int(hotspot_max_staying_time[key])
                    # 找到最大等待时间以后，对独热编码后的当前行后面添加一个等待时间(从1到最大等待时间)
                    for i in range(1, max_staying_time + 1):
                        tmp = hotspot_one_hot_encoded[row]
                        tmp = np.r_[tmp, i]
                        action_one_hot_encoded[hotspot + ',' + str(i)] = tmp

    def set_action_one_hot_encoded(self):
        # 保存最大等待时间的文件夹
        path = '五分钟/'
        files = os.listdir(path)
        # 对于每个文件,读取文件的内容, 进行热编码放入相应的字典中。例如8.txt代表8时间段的内容，
        # 放入到action_one_hot_encoded_8字典中
        for file in files:
            if file == '8.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_8)
            elif file == '9.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_9)
            elif file == '10.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_10)
            elif file == '11.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_11)
            elif file == '12.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_12)
            elif file == '13.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_13)
            elif file == '14.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_14)
            elif file == '15.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_15)
            elif file == '16.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_16)
            elif file == '17.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_17)
            elif file == '18.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_18)
            elif file == '19.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_19)
            elif file == '20.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_20)
            elif file == '21.txt':
                self.action_to_one_hot_encoded(path, file, self.action_one_hot_encoded_21)

    def set_sensors_mobile_charger(self):
        self.sensors_mobile_charger['000'] = [0.7 * 10.8 * 1000, 0.6, '08:00:00']
        self.sensors.append('000')
        self.sensors_mobile_charger['001'] = [0.3 * 10.8 * 1000, 0.8, '08:00:00']
        self.sensors.append('001')
        self.sensors_mobile_charger['003'] = [0.9 * 10.8 * 1000, 1, '08:00:00']
        self.sensors.append('003')
        self.sensors_mobile_charger['004'] = [0.5 * 10.8 * 1000, 0.5, '08:00:00']
        self.sensors.append('004')
        self.sensors_mobile_charger['005'] = [0.1 * 10.8 * 1000, 0.8, '08:00:00']
        self.sensors.append('005')
        self.sensors_mobile_charger['015'] = [0.4 * 10.8 * 1000, 0.6, '08:00:00']
        self.sensors.append('015')
        self.sensors_mobile_charger['030'] = [1 * 10.8 * 1000, 0.9, '08:00:00']
        self.sensors.append('030')
        self.sensors_mobile_charger['042'] = [0.2 * 10.8 * 1000, 0.8, '08:00:00']
        self.sensors.append('042')
        self.sensors_mobile_charger['065'] = [1 * 10.8 * 1000, 1, '08:00:00']
        self.sensors.append('065')
        self.sensors_mobile_charger['081'] = [0.9 * 10.8 * 1000, 0.7, '08:00:00']
        self.sensors.append('081')
        self.sensors_mobile_charger['082'] = [0.8 * 10.8 * 1000, 0.5, '08:00:00']
        self.sensors.append('082')
        self.sensors_mobile_charger['085'] = [0.3 * 10.8 * 1000, 0.7, '08:00:00']
        self.sensors.append('085')
        self.sensors_mobile_charger['096'] = [0.4 * 10.8 * 1000, 1, '08:00:00']
        self.sensors.append('096')
        self.sensors_mobile_charger['125'] = [0.6 * 10.8 * 1000, 0.6, '08:00:00']
        self.sensors.append('123')
        self.sensors_mobile_charger['126'] = [0.3 * 10.8 * 1000, 0.5, '08:00:00']
        self.sensors.append('126')
        self.sensors_mobile_charger['167'] = [0.5 * 10.8 * 1000, 0.8, '08:00:00']
        self.sensors.append('167')
        self.sensors_mobile_charger['179'] = [0.8 * 10.8 * 1000, 0.9, '08:00:00']
        self.sensors.append('179')
        self.sensors_mobile_charger['MC'] = [2000 * 1000, 50]

    def reset(self):
        # 前面0~47 都初始化为 0
        for i in range(48):
            self.state.append(i)


if __name__ == '__main__':
    evn = Evn()
    evn.set_action_one_hot_encoded()

