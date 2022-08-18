import random


def get_rand_start_point(raceline, num, l=1, h=0):
    lines = open(raceline).read().splitlines()
    h = len(lines) if h==0 else h
    line = random.sample(list(enumerate(lines[l:h])), num)

    return line


if __name__ == '__main__':
    line = get_rand_start_point('map/Spielberg_raceline.csv', 2)
    for start_point in line:
        index, point = start_point
        point_array = point.split(";")
