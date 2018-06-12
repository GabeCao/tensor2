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
    hospot = Hotspot(1,2, 1)
    print(str(hospot.get_num()) + ',1')