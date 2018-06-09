from sklearn.preprocessing import LabelBinarizer
class Evn:
    def __init__(self):
        self.state = []
        # sensor 和 mc的能量信息
        self.sensors_mobile_charger = {}
        # 保存所有的sensor 编号
        self.sensors = []
        # 初始化self.sensors_mobile_charger 和 self.sensors
        self.set_sensors_mobile_charger()

        self


    def sensors_one_hot_encoded(self):


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
    evn.reset()
    print(evn.state)