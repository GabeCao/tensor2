from Hotspot import Hotspot
import math
from Point import Point
import copy
class Greedy:
    def __init__(self):
        # sensor 和 mc的能量信息
        self.sensors_mobile_charger = {}
        self.set_sensors_mobile_charger()
        # 获得所有的hotspot
        self.hotspots = []
        self.set_hotspots()
        # charging tour
        self.CS = []
        # 一个回合最大的时间，用秒来表示，早上八点到晚上10点，十四个小时，总共 14 * 3600 秒的时间
        # 如果self.get_evn_time() 得到的时间大于这个时间，则表示该回合结束
        self.one_episode_time = 14 * 3600
        # mc移动花费的时间
        self.move_time = 0
        # 当前时刻所在的hotspot，初始化为base_station
        self.current_hotspot = self.hotspots[0]


    def set_sensors_mobile_charger(self):
        # [0.7 * 6 * 1000, 0.6, 0, True]  依次代表：上一次充电后的剩余能量，能量消耗的速率，上一次充电的时间，
        # 是否已经死掉(计算reward的惩罚值时候使用，避免将一个sensor计算死掉了多次)
        self.sensors_mobile_charger['0'] = [0.7 * 6 * 1000, 0.6, 0, True]
        self.sensors_mobile_charger['1'] = [0.3 * 6 * 1000, 0.4, 0, True]
        self.sensors_mobile_charger['2'] = [0.9 * 6 * 1000, 0.6, 0, True]
        self.sensors_mobile_charger['3'] = [0.5 * 6 * 1000, 0.4, 0, True]
        self.sensors_mobile_charger['4'] = [0.2 * 6 * 1000, 0.3, 0, True]
        self.sensors_mobile_charger['5'] = [0.4 * 6 * 1000, 0.4, 0, True]
        self.sensors_mobile_charger['6'] = [1 * 6 * 1000, 0.8, 0, True]
        self.sensors_mobile_charger['7'] = [0.3 * 6 * 1000, 0.3, 0, True]
        self.sensors_mobile_charger['8'] = [1 * 6 * 1000, 0.4, 0, True]
        self.sensors_mobile_charger['9'] = [0.9 * 6 * 1000, 0.3, 0, True]
        self.sensors_mobile_charger['10'] = [0.8 * 6 * 1000, 0.3, 0, True]
        self.sensors_mobile_charger['11'] = [0.5 * 6 * 1000, 0.5, 0, True]
        self.sensors_mobile_charger['12'] = [0.4 * 6 * 1000, 0.3, 0, True]
        self.sensors_mobile_charger['13'] = [0.6 * 6 * 1000, 0.3, 0, True]
        self.sensors_mobile_charger['14'] = [0.3 * 6 * 1000, 0.3, 0, True]
        self.sensors_mobile_charger['15'] = [0.9 * 6 * 1000, 0.7, 0, True]
        self.sensors_mobile_charger['16'] = [0.8 * 6 * 1000, 0.5, 0, True]
        self.sensors_mobile_charger['MC'] = [2000 * 1000, 50]

    def set_hotspots(self):
        # 这是编号为0 的hotspot，也就是base_stattion,位于整个充电范围中心
        base_station = Hotspot((116.333 - 116.318) * 85000 / 2, (40.012 - 39.997) * 110000 / 2, 0)
        self.hotspots.append(base_station)
        # 读取hotspot.txt 的文件，获取所有的hotspot，放入self.hotspots中
        path = 'hotspot.txt'
        with open(path) as file:
            for line in file:
                data = line.strip().split(',')
                hotspot = Hotspot(float(data[0]), float(data[1]), int(data[2]))
                self.hotspots.append(hotspot)

    # 根据hotspot 的编号，在self.hotspots 中找到对应的hotpot
    def find_hotspot_by_num(self, num):
        for hotspot in self.hotspots:
            if hotspot.get_num() == num:
                return hotspot

    def str_to_seconds(self, input_str):
        data = input_str.split(':')
        hour = int(data[0]) - 8
        minute = int(data[1])
        second = int(data[2])
        return hour * 3600 + minute * 60 + second

    # 获得当前环境的秒
    def get_evn_time(self):
        total_t = 0
        # 获得CS中的所有等待时间
        for action in self.CS:
            staying_time = int(action.split(',')[1])
            total_t += staying_time
        #  CS中的时间加上移动的时间得到总共当前环境的时间
        return total_t * 5 * 60 + self.move_time

    def get_result(self):
        # 如果当前环境时间小于一个回合时间，并且 mc能量大于0
        while self.get_evn_time() < self.one_episode_time and self.sensors_mobile_charger['MC'][0] > 0:
            # 获取当前时间段
            current_slot = int(self.get_evn_time() / 3600) + 8
            path = '五分钟/' + str(current_slot) + '.txt'
            with open(path) as f:
                max_chose_reward = 0
                max_chose_action = None
                for line in f:
                    # 对于每一行就是一个action，我们依次迭代计算每一个action带来的reward，
                    # 每一次计算都要使用相同的初始环境，所以我们复制self.sensors_mobile_charger = {}
                    sensor_mc = copy.deepcopy(self.sensors_mobile_charger)
                    chose_reward = 0
                    chose_action = line.strip()
                    hotspot_num_max_staying_time = line.strip().split(',')
                    # 选择的hotspot
                    hotspot = self.find_hotspot_by_num(int(hotspot_num_max_staying_time[0]))
                    # 最大等待时间
                    max_staying_time = int(hotspot_num_max_staying_time[1])
                    # 距离当前hotspot的距离
                    distance = hotspot.get_distance_between_hotspot(self.current_hotspot)
                    move_time = self.move_time + distance / 5
                    # 到达hotspot后，开始等待
                    start_seconds = self.get_evn_time()
                    # 结束等待的时间
                    end_seconds = start_seconds + max_staying_time * 5 * 60
                    # 获得所有的sensor 轨迹点
                    for i in range(17):
                        sensor_path = 'sensor数据/' + str(i) + '.txt'
                        with open(sensor_path) as sensor_file:
                            for sensor_line in sensor_file:
                                sensor_line = sensor_line.strip().split(',')
                                point = Point(float(sensor_line[0]), float(sensor_line[1]), sensor_line[2])
                                point_time = self.str_to_seconds(point.get_time())

                                if start_seconds <= point_time <= end_seconds and point.get_distance_between_point_and_hotspot(hotspot) < 60:
                                    # 取出sensor
                                    sensor = self.sensors_mobile_charger[str(i)]
                                    # 上一次充电后的电量
                                    sensor_energy_after_last_time_charging = sensor[0]
                                    # 当前sensor 电量消耗的速率
                                    sensor_consumption_ratio = sensor[1]
                                    # 上一次的充电时间
                                    previous_charging_time = sensor[2]
                                    # 当前sensor 的剩余电量
                                    sensor_reserved_energy = sensor_energy_after_last_time_charging - \
                                                                (point_time - previous_charging_time) * sensor_consumption_ratio
                                    # 当前sensor 的剩余寿命
                                    rl = sensor_reserved_energy / sensor_consumption_ratio
                                    # 如果剩余寿命大于两个小时
                                    if rl >= 2 * 3600:
                                        break
                                    # 如果剩余寿命在0 到 两个小时
                                    elif 0 < rl < 2 * 3600:
                                        # mc 给该sensor充电， 充电后更新剩余能量
                                        self.sensors_mobile_charger['MC'][0] = self.sensors_mobile_charger['MC'][0] \
                                                                               - (6 * 1000 - sensor_reserved_energy)
                                        # 设置sensor 充电后的剩余能量 是满能量
                                        sensor[0] = 6 * 1000
                                        # 更新被充电的时间
                                        sensor[2] = point_time
                                        # 加上得到的奖励,需要先将 rl 的单位先转化成小时
                                        rl = rl / 3600
                                        chose_reward += math.exp(-rl)
                                    else:
                                        if sensor[3] is True:
                                            chose_reward += -0.5
                                            sensor[3] = False

                    if chose_reward > max_chose_reward:
                        max_chose_reward = chose_reward
                        max_chose_action = chose_action


