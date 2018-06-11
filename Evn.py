from sklearn.preprocessing import LabelBinarizer
import math
from RL_brain_modified import DeepQNetwork
from Hotspot import Hotspot
from Point import Point
class Evn:
    def __init__(self):
        # 当前环境state
        self.state = []
        # mc移动花费的时间
        self.move_time = 0
        # 一个回合最大的时间，用秒来表示，早上八点到晚上10点，十四个小时，总共 14 * 3600 秒的时间
        # 如果self.get_evn_time() 得到的时间大于这个时间，则表示该回合结束
        self.one_episode_time = 14 * 3600
        # sensor 和 mc的能量信息
        self.sensors_mobile_charger = {}
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
        # 获得所有的hotspot
        self.hotspots = []
        self.set_hotspots()
        # 记录当前时刻所在的hotspot，在环境初始化的时候设置为base_station
        self.current_hotspot = None

    def set_sensors_mobile_charger(self):
        # [0.7 * 10.8 * 1000, 0.6, 0, True]  依次代表：上一次充电后的剩余能量，能量消耗的速率，上一次充电的时间，
        # 是否已经死掉(计算reward的惩罚值时候使用，避免将一个sensor计算死掉了多次)
        self.sensors_mobile_charger['0'] = [0.7 * 10.8 * 1000, 0.6, 0, True]
        self.sensors_mobile_charger['1'] = [0.3 * 10.8 * 1000, 0.8, 0, True]
        self.sensors_mobile_charger['2'] = [0.9 * 10.8 * 1000, 1, 0, True]
        self.sensors_mobile_charger['3'] = [0.5 * 10.8 * 1000, 0.5, 0, True]
        self.sensors_mobile_charger['4'] = [0.1 * 10.8 * 1000, 0.8, 0, True]
        self.sensors_mobile_charger['5'] = [0.4 * 10.8 * 1000, 0.6, 0, True]
        self.sensors_mobile_charger['6'] = [1 * 10.8 * 1000, 0.9, 0, True]
        self.sensors_mobile_charger['7'] = [0.2 * 10.8 * 1000, 0.8, 0, True]
        self.sensors_mobile_charger['8'] = [1 * 10.8 * 1000, 1, 0, True]
        self.sensors_mobile_charger['9'] = [0.9 * 10.8 * 1000, 0.7, 0, True]
        self.sensors_mobile_charger['10'] = [0.8 * 10.8 * 1000, 0.5, 0, True]
        self.sensors_mobile_charger['11'] = [0.3 * 10.8 * 1000, 0.7, 0, True]
        self.sensors_mobile_charger['12'] = [0.4 * 10.8 * 1000, 1, 0, True]
        self.sensors_mobile_charger['13'] = [0.6 * 10.8 * 1000, 0.6, 0, True]
        self.sensors_mobile_charger['14'] = [0.3 * 10.8 * 1000, 0.5, 0, True]
        self.sensors_mobile_charger['15'] = [0.5 * 10.8 * 1000, 0.8, 0, True]
        self.sensors_mobile_charger['16'] = [0.8 * 10.8 * 1000, 0.9, 0, True]
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

    # 传入一个action, 得到下一个state，reward，和 done(是否回合结束)的信息
    def step(self, action):
        # action 的 表示形如 43,1 表示到43 号hotspot 等待1个t
        action = action.split(',')
        # 得到hotspot 的编号
        hotspot_num = int(action[0])
        # 得到等待时间
        staying_time = int(action[1])

        # 得到下一个hotspot
        hotspot = self.find_hotspot_by_num(hotspot_num)
        # 当前hotspot 和 下一个hotspot间的距离,得到移动花费的时间，添加到self.move_time 里
        distance = hotspot.get_distance_between_hotspot(self.current_hotspot)
        time = distance / 5
        self.move_time += time
        # 更新self.current_hotspot 为 action 中选择的 hotspot
        self.current_hotspot = hotspot

        # 获取当前时间段,加8是因为从8点开始
        start_wait_seconds = self.get_evn_time()
        hour = int(start_wait_seconds / 3600) + 8
        # 将在hotspot_num 等待的时间 添加到state中的CS
        self.state[hotspot_num] += staying_time
        # mc 结束等待后环境的时间
        end_wait_seconds = self.get_evn_time()
        path = 'hotspot中sensor的访问情况/' + str(hour) + '时间段/' + str(hotspot_num) + '.txt'
        # 读取文件，得到在当前时间段，hotspot_num 的访问情况，用字典保存。key: sensor 编号；value: 访问次数
        hotspot_num_sensor_arrived_times = {}
        with open(path) as f:
            for line in f:
                data = line.strip().split(',')
                hotspot_num_sensor_arrived_times[data[0]] = data[1]
        # 一共17个sensor，现在更新每个sensor 的信息
        reward = 0
        for i in range(17):
            # 取出当前sensor 访问 hotspot_num 的次数,如果大于 0 belong = 1, 表示属于，sensor 可能会到达这个hotspot 。
            # 否则0，表示不属于，sensor 不可能会到达这个hotspot
            times = int(hotspot_num_sensor_arrived_times[str(i)])
            # times == 0，表示该sensor不属于该hotspot，不可能到达该hotspot充电
            if times == 0:
                belong = str(0)
                # 取出sensor
                sensor = self.sensors_mobile_charger[str(i)]
                # 上一次充电后的电量
                sensor_energy_after_last_time_charging = sensor[0]
                # 当前sensor 电量消耗的速率
                sensor_consumption_ratio = sensor[1]
                # 上一次的充电时间
                previous_charging_time = sensor[2]
                # 在mc等待了action 中的等待时间以后，sensor 的剩余电量
                sensor_reserved_energy = sensor_energy_after_last_time_charging - \
                                         (end_wait_seconds - previous_charging_time) * sensor_consumption_ratio
                # 当前sensor 的剩余寿命
                rl = sensor_reserved_energy / sensor_consumption_ratio

                # 如果剩余寿命大于两个小时
                if rl >= 2 * 3600:
                    # 得到 大于阈值的 独热编码,转换成list,然后更新state 中的能量状态
                    rl_one_hot_encoded = \
                    self.rl_label_binarizer.transform(['Greater than the threshold value, 0']).tolist()[0]
                    start = 48 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        self.state[start] = rl_one_hot_encoded[rl_i]
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                elif 0 < rl < 2 * 3600:
                    # 更新state中 的剩余寿命信息的状态
                    # 得到 小于阈值的 独热编码,转换成list,然后更新state 中的状态
                    rl_one_hot_encoded = self.rl_label_binarizer.transform(['Smaller than the threshold value, 1']) \
                        .tolist()[0]
                    start = 48 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        a = rl_one_hot_encoded[rl_i]
                        self.state[start] = a
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                else:
                    # 更新state中 的剩余寿命信息的状态
                    # 得到 死掉的 独热编码,转换成list,然后更新state 中的状态
                    rl_one_hot_encoded = self.rl_label_binarizer.transform(['dead, -1']) \
                        .tolist()[0]
                    start = 48 + i * 4
                    end = start + 2
                    rl_i = 0
                    while start <= end:
                        a = rl_one_hot_encoded[rl_i]
                        self.state[start] = a
                        rl_i += 1
                        start += 1
                    # 更新是否属于 state中 的 是否属于hotspot 的信息
                    # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                    belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                    self.state[start] = belong_one_hot_encoded
                    # 如果这个sensor是第一次死掉，那么reward减去 0.5
                    if sensor[3] is True:
                        reward += -0.5
                        sensor[3] = False
            else:
                # times不等于0 的情况下，sensor可能会过来
                belong = str(1)
                # 读取第i 个sensor 的轨迹点信息
                sensor_path = 'sensor数据/' + str(i) + '.txt'
                with open(sensor_path) as f:
                    for point_line in f:
                        data = point_line.strip().split(',')
                        point_time = self.str_to_seconds(data[2])
                        point = Point(float(data[0]), float(data[1]), data[2])

                        # 如果第 i 个sensor的轨迹点的时间 小于end_wait_seconds且大于start_wait_seconds，
                        # 同时轨迹点和hotspot 的距离小于60，则到达该hotspot
                        if (start_wait_seconds <= point_time <= end_wait_seconds) and (point.get_distance_between_point_and_hotspot(self.current_hotspot) < 60):
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
                                reward += 0
                                # 得到 大于阈值的 独热编码,转换成list,然后更新state 中的能量状态
                                rl_one_hot_encoded = \
                                    self.rl_label_binarizer.transform(['Greater than the threshold value, 0']).tolist()[
                                        0]
                                start = 48 + i * 4
                                end = start + 2
                                rl_i = 0
                                while start <= end:
                                    self.state[start] = rl_one_hot_encoded[rl_i]
                                    rl_i += 1
                                    start += 1
                                # 更新是否属于 state中 的 是否属于hotspot 的信息
                                # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                                belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                                self.state[start] = belong_one_hot_encoded
                            # 如果剩余寿命在0 到 两个小时
                            elif 0 < rl < 2 * 3600:
                                # mc 给该sensor充电， 充电后更新剩余能量
                                self.sensors_mobile_charger['MC'][0] = self.sensors_mobile_charger['MC'][0] \
                                                                       - (10.8 * 1000 - sensor_reserved_energy)
                                # 设置sensor 充电后的剩余能量 是满能量
                                sensor[0] = 10.8 * 1000
                                # 更新被充电的时间
                                sensor[2] = point_time
                                # 更新state中 的剩余寿命信息的状态
                                # 得到 小于阈值的 独热编码,转换成list,然后更新state 中的状态
                                rl_one_hot_encoded = \
                                self.rl_label_binarizer.transform(['Greater than the threshold value, 0']) \
                                    .tolist()[0]
                                start = 48 + i * 4
                                end = start + 2
                                rl_i = 0
                                while start <= end:
                                    a = rl_one_hot_encoded[rl_i]
                                    self.state[start] = a
                                    rl_i += 1
                                    start += 1
                                # 更新是否属于 state中 的 是否属于hotspot 的信息
                                # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                                belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                                self.state[start] = belong_one_hot_encoded
                                # 加上得到的奖励,需要先将 rl 的单位先转化成小时
                                rl = rl / 3600
                                reward += math.exp(-rl)
                            else:
                                # sensor 能量小于0，在到达hotspot前就死掉了
                                # 更新state中 的剩余寿命信息的状态
                                # 得到 死掉的 独热编码,转换成list,然后更新state 中的状态
                                rl_one_hot_encoded = self.rl_label_binarizer.transform(['dead, -1']) \
                                    .tolist()[0]
                                start = 48 + i * 4
                                end = start + 2
                                rl_i = 0
                                while start <= end:
                                    a = rl_one_hot_encoded[rl_i]
                                    self.state[start] = a
                                    rl_i += 1
                                    start += 1
                                # 更新是否属于 state中 的 是否属于hotspot 的信息
                                # 通过belong 找到对应的 独热编码，转换成list,因为这个独热编码只有一位，所以取第一位得到结果
                                belong_one_hot_encoded = self.belong_label_binarizer.transform([belong]).tolist()[0][0]
                                self.state[start] = belong_one_hot_encoded
                                # 如果这个sensor是第一次死掉，那么reward减去 0.5
                                if sensor[3] is True:
                                    reward += -0.5
                                    sensor[3] = False

        # mc 给到达的sensor 充电后，如果能量为负或者 self.get_evn_time() > self.one_episode_time，则回合结束，反之继续
        if self.sensors_mobile_charger['MC'][0] <= 0 or self.get_evn_time() > self.one_episode_time:
            done = True
        else:
            done = False

        return self.state, reward, done

    # 初始化整个环境
    def reset(self, RL):
        # 前面0~47 都初始化为 0。记录CS的信息
        for i in range(48):
            self.state.append(0)
        # 48位开始记录sensor的信息,每一个sensor需要4位，17个sensor，共68位
        for i in range(48, 48 + 68 + 1):
                self.state.append(0)
        # 得到一个随机的8点时间段的action,例如 43,1 表示到43 号hotspot 等待1个t
        action = RL.random_action()
        self.current_hotspot = self.hotspots[0]

        state__, reward_, done_ = self.step(action)
        return state__, reward_, done_

    # 传入时间字符串，如：09：00：00，转化成与 08:00:00 间的秒数差
    def str_to_seconds(self, input_str):
        data = input_str.split(':')
        hour = int(data[0]) - 8
        minute = int(data[1])
        second = int(data[2])
        return hour * 3600 + minute * 60 + second

    # 获得当前环境的秒
    def get_evn_time(self):
        total_t = 0
        for i in range(48):
            total_t += 0
        total_time = total_t * 5 * 60 + self.move_time
        return total_time

if __name__ == '__main__':
    evn = Evn()
    RL = DeepQNetwork()
    state_, reward, done = evn.reset(RL)
    print(state_)
    print(reward)
    print(done)