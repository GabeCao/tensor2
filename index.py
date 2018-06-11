from Hotspot import Hotspot
from Point import Point
if __name__ == '__main__':
    hotpot = Hotspot(129.9038105676658,1169.1342951089923,10)
    count = 0
    with open('sensor数据/12.txt') as f:
        for line in f:
            data = line.strip().split(',')
            point = Point(float(data[0]), float(data[1]), data[2])
            if point.get_distance_between_point_and_hotspot(hotpot) < 60:
                count += 1

    print(count)