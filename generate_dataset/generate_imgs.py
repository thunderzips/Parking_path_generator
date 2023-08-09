import cv2
import numpy as np
from time import sleep, time
import argparse
import random
import copy

from environment import Environment, Parking1
from pathplanning import PathPlanning, ParkPathPlanning, interpolate_path
from control import Car_Dynamics, MPC_Controller, Linear_MPC_Controller
from utils import angle_of_line, make_square, DataLogger


"""
TODO: Create dataset such that only the final path is displayed in the output dataset.
Train on this dataset and then test.
"""


def around(num,err=5,uni=False):
    """
    returns a number with a certain error.
    err: max error in the number.
    uni: if true, only increases hte number by the error
    """

    if not uni:
        return int(num + err*random.random()*list([-1,1])[int(random.random()*2)])
    else:
        return int(num + err*random.random())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x_start', type=int, default=0, help='X of start')
    parser.add_argument('--y_start', type=int, default=90, help='Y of start')
    parser.add_argument('--psi_start', type=int, default=0, help='psi of start')
    parser.add_argument('--x_end', type=int, default=90, help='X of end')
    parser.add_argument('--y_end', type=int, default=80, help='Y of end')
    parser.add_argument('--parking', type=int, default=1, help='park position in parking1 out of 24')

    args = parser.parse_args()
    logger = DataLogger()

    avg_time = 0

    for iter in range(1,350):

        print("#########################")
        print(f"iteration number:{iter}")


        '''
        default variables
        '''

        start = np.array([around(args.x_start,err=15,uni=True), around(args.y_start,err=10,uni=True)]) 
        end   = np.array([args.x_end, args.y_end])
        psi_start = 90

        empty_spot = iter%23 + 1
        
        parking1 = Parking1(empty_spot)

        end, obs = parking1.generate_obstacles()


        # add squares
        
        rand_thresh = 0.6

        if random.random() > rand_thresh:
            square1 = make_square(around(10),around(65),around(10))
            obs = np.vstack([obs,square1])

        if random.random() > rand_thresh:
            square1 = make_square(around(15),around(30),around(15))
            obs = np.vstack([obs,square1])

        if random.random() > rand_thresh:
            square1 = make_square(50,around(50),around(7))
            obs = np.vstack([obs,square1])
        
        if random.random() > rand_thresh:
            square1 = make_square(50,around(70),around(7))
            obs = np.vstack([obs,square1])

        ########################### initialization ##################################################
        env = Environment(obs)
        my_car = Car_Dynamics(start[0], start[1], 0, np.deg2rad(psi_start), length=4, dt=0.2)
        # MPC_HORIZON = 5
        # controller = MPC_Controller()
        # controller = Linear_MPC_Controller()

        # res = env.render(my_car.x, my_car.y, my_car.psi, 0)

        ############################# path planning #################################################
        park_path_planner = ParkPathPlanning(obs)
        path_planner = PathPlanning(obs)

        start_time = time()

        print('     planning park scenario ...')
        new_end, park_path, ensure_path1, ensure_path2 = park_path_planner.generate_park_scenario(int(start[0]),int(start[1]),int(end[0]),int(end[1]))
        
        print('     routing to destination ...')
        path = path_planner.plan_path(int(start[0]),int(start[1]),int(new_end[0]),int(new_end[1]))
        path = np.vstack([path, ensure_path1])

        print('     interpolating ...')
        interpolated_path = interpolate_path(path, sample_rate=5)
        interpolated_park_path = interpolate_path(park_path, sample_rate=2)
        interpolated_park_path = np.vstack([ensure_path1[::-1], interpolated_park_path, ensure_path2[::-1]])

        end_time = time()

        avg_time += (end_time - start_time)

        print(f"time taken = {end_time - start_time}")
        print(f"average time taken = {avg_time/iter}")

        final_path = np.vstack([interpolated_path, interpolated_park_path, ensure_path2])
        
        env_path = copy.deepcopy(env)

        env.place_obstacles(obs)
        env.draw_goal(interpolated_park_path)

        res = env.render(my_car.x, my_car.y, my_car.psi, 0)
        cv2.imwrite(f"generate_dataset/input_images/environment{iter}.png",res)

        env_path.draw_path(interpolated_path)
        env_path.draw_path(interpolated_park_path)
        
        # res = env_path.render(my_car.x, my_car.y, my_car.psi, 0)
        # res = env.render()

        rendered = cv2.resize(np.flip(env_path.background, axis=0), (700,700))

        alpha = 1
        beta = 500

        rendered = cv2.convertScaleAbs(rendered, alpha, beta)

        cv2.imwrite(f"generate_dataset/output_images/environment{iter}.png",rendered)

        cv2.destroyAllWindows()


