from Hotspot import Hotspot
from Point import Point
import copy

def str_to_seconds(input_str):
    data = input_str.split(':')
    hour = int(data[0]) - 8
    minute = int(data[1])
    second = int(data[2])
    return hour * 3600 + minute * 60 + second


if __name__ == '__main__':
    sensors_mobile_charger = {}

    sensors_mobile_charger['0'] = [0.7 * 6 * 1000, 0.6, 0, True]
    sensors_mobile_charger['1'] = [0.3 * 6 * 1000, 0.4, 0, True]


    mc = copy.deepcopy(sensors_mobile_charger)
    print(sensors_mobile_charger)
    sensor = mc['1']
    sensor[3] = False
    print(mc)