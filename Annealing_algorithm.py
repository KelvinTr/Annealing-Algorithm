# Kelvin Tran
# Develop simulated annealing algorithm to solve traveling salesman problem with 20 cities that
# are uniformly distributed within a unit square in a 2-dimensional plane. 

import numpy as np
import matplotlib.pyplot as plt
import math
import copy



def total_energy(cities):
    distance = 0
    for i in range((len(cities[0]) - 1)):
        x_square = math.pow(cities[0][i+1] - cities[0][i],2)
        y_square = math.pow(cities[1][i+1] - cities[1][i],2)
        distance += math.sqrt(x_square + y_square)
    x_square = math.pow(cities[0][0] - cities[0][-1], 2)
    y_square = math.pow(cities[1][0] - cities[1][-1], 2)
    distance += math.sqrt(x_square + y_square)
    return distance


def plot(cities, axis):

    for i in range((len(cities[0]) - 1)):
        plt.plot([cities[0][i], cities[0][i+1]], [cities[1][i], cities[1][i+1]], 'r')

    plt.plot([cities[0][-1], cities[0][0]], [cities[1][-1], cities[1][0]], 'r')
    plt.plot(cities[0], cities[1], 'o', color='black')
    plt.axis(axis)
    plt.show()


def annealing(To, factor, iterations, coords):
    curr_Energy = total_energy(coords)
    for i in range(iterations):
        print(i, 'cost = ', curr_Energy)
        
        city1, city2, city3, city4 = np.random.randint(0, len(coords[0]), size=4)

        hold_coords = copy.deepcopy(coords)

        temp1 = [hold_coords[0][city1], hold_coords[1][city1]]
        hold_coords[0][city1] = hold_coords[0][city2]
        hold_coords[1][city1] = hold_coords[1][city2]
        hold_coords[0][city2] = temp1[0]
        hold_coords[1][city2] = temp1[1]

        temp2 = [hold_coords[0][city3], hold_coords[1][city3]]
        hold_coords[0][city3] = hold_coords[0][city4]
        hold_coords[1][city3] = hold_coords[1][city4]
        hold_coords[0][city4] = temp2[0]
        hold_coords[1][city4] = temp2[1]

        new_Energy = total_energy(hold_coords)
        delta_Energy = new_Energy - curr_Energy

        if delta_Energy < 0:
            curr_Energy = new_Energy
            coords = hold_coords
        else:
            random_val = np.random.uniform()
            prob = math.exp((curr_Energy - new_Energy)/ To)
            if random_val < prob:
                curr_Energy = new_Energy
                coords = hold_coords

        To = To*factor

    return coords

def annealing101(To, factor, iterations, coords):
    curr_Energy = total_energy(coords)
    for i in range(iterations):
        print(i, 'cost = ', curr_Energy)
        
        edge1, edge2 = np.random.randint(0, len(coords[0]), size=2)

        hold_coords = coords.copy()


        hold_coords[0][edge1:edge2] = np.flip(coords[0][edge1:edge2])
        hold_coords[1][edge1:edge2] = np.flip(coords[1][edge1:edge2])


        new_Energy = total_energy(hold_coords)
        delta_Energy = new_Energy - curr_Energy

        if delta_Energy < 0:
            curr_Energy = new_Energy
            coords = hold_coords
        else:
            random_val = np.random.uniform()
            prob = math.exp((curr_Energy - new_Energy)/ To)
            if random_val < prob:
                curr_Energy = new_Energy
                coords = hold_coords

        To = To*factor

    return coords


if __name__ == '__main__':
    cities20_x = np.array([0.6606, 0.9695, 0.5906, 0.2124, 0.0398, 0.1367, 0.9536, 
            0.6091, 0.8767, 0.8148, 0.3876, 0.7041, 0.0213, 0.3429, 0.7471, 
            0.5449, 0.9464, 0.1247, 0.1636, 0.8668])

    cities20_y = np.array([0.9500, 0.6740, 0.5029, 0.8274, 0.9697, 0.5979, 0.2184, 
                0.7148, 0.2395, 0.2867, 0.8200, 0.3296, 0.1649, 0.3025, 0.8192, 
                0.9392, 0.8191, 0.4351, 0.8646, 0.6768])

    cities20 = np.array([cities20_x, cities20_y])


    cities101_x = np.array([41,35,55,55,15,25,20,10,55,30,20,50,30,15,30,10,5,20,
                    15,45,45,45,55,65,65,45,35,41,64,40,31,35,53,65,63,2,20,5,60,
                    40,42,24,23,11,6,2,8,13,6,47,49,27,37,57,63,53,32,36,21,17,12,
                    24,27,15,62,49,67,56,37,37,57,47,44,46,49,49,53,61,57,56,55,15,
                    14,11,16,4,28,26,26,31,15,22,18,26,25,22,25,19,20,18,35])

    cities101_y = np.array([49,17,45,20,30,30,50,43,60,60,65,35,25,10,5,20,30,40,
                    60,65,20,10,5,35,20,30,40,37,42,60,52,69,52,55,65,60,20,5,12,
                    25,7,12,3,14,38,48,56,52,68,47,58,43,31,29,23,12,12,26,24,34,
                    24,58,69,77,77,73,5,39,47,56,68,16,17,13,11,42,43,52,48,37,54,
                    47,37,31,22,18,18,52,35,67,19,22,24,27,24,27,21,21,26,18,35])

    cities101 = np.array([cities101_x, cities101_y])


    it = 100000
    curr_temp = 300
    factor = 0.99
    curr_temp101 = 200
    factor101 = 0.999

    new20 = annealing(curr_temp, factor, it, cities20)
    new101 = annealing101(curr_temp101, factor101, it, cities101)

    plot(new20, [0, 1, 0 ,1])
    plot(new101, [0, 70 ,0, 80])







    
