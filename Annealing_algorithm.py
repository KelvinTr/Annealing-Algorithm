# Kelvin Tran
# Develop simulated annealing algorithm to solve traveling salesman problem with 20 cities that
# are uniformly distributed within a unit square in a 2-dimensional plane. 

import numpy as np
import matplotlib.pyplot as plt
import math




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


def plot(cities):

    for i in range((len(cities[0]) - 1)):
        plt.plot([cities[0][i], cities[0][i+1]], [cities[1][i], cities[1][i+1]], 'r')

    plt.plot([cities[0][-1], cities[0][0]], [cities[1][-1], cities[1][0]], 'r')
    plt.plot(cities[0], cities[1], 'o', color='black')
    plt.axis([0, 1, 0 ,1])
    plt.show()



def annealing(To, factor, iterations, coords):
    curr_Energy = total_energy(coords)
    for i in range(iterations):
        print(i, 'cost = ', curr_Energy)

        for j in range(100):
            city1, city2 = np.random.randint(0, len(coords[0]), size=2)

            hold_coords = coords

            temp = [hold_coords[0][city1], hold_coords[1][city1]]
            hold_coords[0][city1] = hold_coords[0][city2]
            hold_coords[1][city1] = hold_coords[1][city2]

            hold_coords[0][city2] = temp[0]
            hold_coords[1][city2] = temp[1]

            new_Energy = total_energy(hold_coords)
            delta_Energy = new_Energy - curr_Energy

            if delta_Energy < 0:
                curr_Energy = new_Energy
                coords = hold_coords
                continue
            else:
                random_val = np.random.uniform()
                prob = math.exp((curr_Energy - new_Energy)/ To)
                if random_val < prob:
                    curr_Energy = new_Energy
                    coords = hold_coords

        To = To*factor







if __name__ == '__main__':
    cities_x = [0.6606, 0.9695, 0.5906, 0.2124, 0.0398, 0.1367, 0.9536, 
            0.6091, 0.8767, 0.8148, 0.3876, 0.7041, 0.0213, 0.3429, 0.7471, 
            0.5449, 0.9464, 0.1247, 0.1636, 0.8668]

    cities_y = [0.9500, 0.6740, 0.5029, 0.8274, 0.9697, 0.5979, 0.2184, 
                0.7148, 0.2395, 0.2867, 0.8200, 0.3296, 0.1649, 0.3025, 0.8192, 
                0.9392, 0.8191, 0.4351, 0.8646, 0.6768]

    cities = [cities_x, cities_y]

    it = 000
    curr_temp = 200
    factor = 0.9

    annealing(curr_temp, factor, it, cities)

    plot(cities)



    
