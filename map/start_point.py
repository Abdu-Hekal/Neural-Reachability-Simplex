import random


def get_rand_start_point(raceline, num):
    lines = open(raceline).read().splitlines()
    line = random.sample(list(enumerate(lines[0:len(lines)])), num)

    return line


if __name__ == '__main__':
    line = get_rand_start_point('map/Spielberg_raceline.csv', 2)
    for start_point in line:
        index, point = start_point
        point_array = point.split(";")
