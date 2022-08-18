# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""
Generates random tracks.
Adapted from https://gym.openai.com/envs/CarRacing-v0
Author: Hongrui Zheng
"""

import cv2
import os
import numpy as np
import shapely.geometry as shp
import matplotlib.pyplot as plt
import math
from operator import itemgetter

if not os.path.exists('maps'):
    print('Creating maps/ directory.')
    os.makedirs('maps')

WIDTH = 10.0
def create_track():

    # post processing, converting to numpy, finding exterior and interior walls
    track_1 = [(i/10, 0) for i in range(3000)]
    track_2 = [(300-i/10,0) for i in range(6000)]
    track_3 = [(i/10-300,0) for i in range(3000)]
    track_4 = [(0,i/10) for i in range(3000)]
    track_5 = [(0,300-i/10) for i in range(6000)]
    track_xy = track_1 + track_2 + track_3 + track_4 + track_5
    track_xy = np.asarray(track_xy)
    track_poly = shp.Polygon(track_xy)
    track_xy_offset_in = track_poly.buffer(WIDTH)
    track_xy_offset_out = track_poly.buffer(-WIDTH)
    track_xy_offset_in_np = np.array(track_xy_offset_in.exterior)
    track_xy_offset_out_np = np.array(track_xy_offset_out.exterior)
    return track_xy, track_xy_offset_in_np, track_xy_offset_out_np


def convert_track(track, track_int, track_ext):

    # converts track to image and saves the centerline as waypoints
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 20)
    ax.plot(*track_int.T, color='black', linewidth=3)
    ax.plot(*track_ext.T, color='black', linewidth=3)
    plt.tight_layout()
    ax.set_aspect('equal')
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    plt.axis('off')
    plt.savefig('custom_maps/maps/intersection.png', dpi=80)

    map_width, map_height = fig.canvas.get_width_height()
    print('map size: ', map_width, map_height)

    # transform the track center line into pixel coordinates
    xy_pixels = ax.transData.transform(track)
    origin_x_pix = xy_pixels[0, 0]
    origin_y_pix = xy_pixels[0, 1]

    print(origin_x_pix, origin_y_pix)

    xy_pixels = xy_pixels - np.array([[origin_x_pix, origin_y_pix]])

    map_origin_x = -origin_x_pix*0.05
    map_origin_y = -origin_y_pix*0.05

    # convert image using cv2
    cv_img = cv2.imread('custom_maps/maps/intersection.png', -1)
    # convert to bw
    cv_img_bw = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    # saving to img
    cv2.imwrite('custom_maps/maps/intersection.png', cv_img_bw)
    cv2.imwrite('custom_maps/maps/intersection.pgm', cv_img_bw)

    # create yaml file
    yaml = open('custom_maps/maps/intersection.yaml', 'w')
    yaml.write('image: intersection.pgm\n')
    yaml.write('resolution: 0.062500\n') #why is it 0.0625? and not 0.05796 like other maps
    yaml.write('origin: [' + str(map_origin_x) + ',' + str(map_origin_y) + ', 0.000000]\n')
    yaml.write('negate: 0\noccupied_thresh: 0.45\nfree_thresh: 0.196')
    yaml.close()
    plt.close()

    # saving track centerline as a csv in ros coords
    waypoints_csv = open('custom_maps/maps/intersection_centerline.csv', 'w')
    for row in xy_pixels:
        waypoints_csv.write(str(0.05*row[0]) + ', ' + str(0.05*row[1]) + '\n')
    waypoints_csv.close()

    return xy_pixels

def create_raceline(xy_pixels):
    raceline_1 = []
    raceline_2 = []
    for x in xy_pixels:
        if x[0] == 0 and x.tolist() not in raceline_1:
            raceline_1.append(x.tolist())
        if x[1] == 0 and x.tolist() not in raceline_2:
            raceline_2.append(x.tolist())

    raceline_1 = sorted(raceline_1, key=itemgetter(1))
    raceline_2 = sorted(raceline_2, key=itemgetter(0))


    raceline_1_csv = open('custom_maps/maps/intersection_raceline_1.csv', 'w')
    raceline_2_csv = open('custom_maps/maps/intersection_raceline_2.csv', 'w')
    raceline_3_csv = open('custom_maps/maps/intersection_raceline_3.csv', 'w')
    raceline_4_csv = open('custom_maps/maps/intersection_raceline_4.csv', 'w')

    raceline_1_csv.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2")
    raceline_2_csv.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2")
    raceline_3_csv.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2")
    raceline_4_csv.write("# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2")

    prev_x1, prev_y1, prev_x2, prev_y2, prev_x3, prev_y3, prev_x4, prev_y4, prev_sm1, prev_sm2, prev_sm3, prev_sm4 = 0,0,0,0,0,0,0,0,0,0,0,0
    for i, (row_1, row_2, row_3, row_4) in enumerate(zip(raceline_1,raceline_2, reversed(raceline_1), reversed(raceline_2))):
        x1, y1, x2, y2, x3, y3, x4, y4 = (row_1[0]-15)*0.05, row_1[1]*0.05, row_2[0]*0.05, (row_2[1]+15)*0.05, (row_3[0]+15)*0.05, row_3[1]*0.05, row_4[0]*0.05, (row_4[1]-15)*0.05
        if i == 1:
            sm1, sm2, sm3, sm4 = 0.0,0.0, 0.0, 0.0
        else:
            sm1 = math.sqrt((x1 - prev_x1)** 2 + (y1 - prev_y1)** 2) + prev_sm1
            sm2 = math.sqrt((x2 - prev_x2) ** 2 + (y2 - prev_y2) ** 2) + prev_sm2
            sm3 = math.sqrt((x3 - prev_x3) ** 2 + (y3 - prev_y3) ** 2) + prev_sm3
            sm4 = math.sqrt((x4 - prev_x4) ** 2 + (y4 - prev_y4) ** 2) + prev_sm4

        raceline_1_csv.write(f"{sm1}; {x1}; {y1}; 1.5708; 0.0; 10.0; 0.0 \n")
        raceline_2_csv.write(f"{sm2}; {x2}; {y2}; 0.0; 0.0; 10.0; 0.0 \n")
        raceline_3_csv.write(f"{sm3}; {x3}; {y3}; -1.5708; 0.0; 10.0; 0.0 \n")
        raceline_4_csv.write(f"{sm4}; {x4}; {y4}; 3.14159; 0.0; 10.0; 0.0 \n")


        prev_x1, prev_y1, prev_x2, prev_y2, prev_sm1, prev_sm2 = x1, y1, x2, y2, sm1, sm2

    raceline_1_csv.close()
    raceline_2_csv.close()



if __name__ == '__main__':
    track, track_int, track_ext = create_track()
    xy_pixels = convert_track(track, track_int, track_ext)
    create_raceline(xy_pixels)