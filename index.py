from Hotspot import Hotspot
from Point import Point


def str_to_seconds(input_str):
    data = input_str.split(':')
    hour = int(data[0]) - 8
    minute = int(data[1])
    second = int(data[2])
    return hour * 3600 + minute * 60 + second


if __name__ == '__main__':
    hotpot = Hotspot(129.9038105676658, 1169.1342951089923, 10)
    count = 0
    with open('sensor数据/12.txt') as f:
        for line in f:
            data = line.strip().split(',')
            point = Point(float(data[0]), float(data[1]), data[2])
            times = str_to_seconds(point.get_time())
            if 0 <= times <= 3600 and point.get_distance_between_point_and_hotspot(hotpot) < 60:
                count += 1

    print(count)