import random


def get_rand_start_point(raceline, num):
    lines = open(raceline).read().splitlines()
    line = random.sample(lines[3:50], num)

    return line


if __name__ == '__main__':
    line = get_rand_start_point('Spielberg_raceline.csv', 1)
    print(line[0].split(";"))
