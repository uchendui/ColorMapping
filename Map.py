import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


class Map(object):
    """
    Class that implements a Kohenen self-organizing map
    :param data: Input data
    :param num_rows: Number of rows of the map
    :param num_cols: Number of columns of the map
    :param learning_rate: Learning rate
    """

    class MapNeuron(object):
        def __init__(self, dimension):
            self.weights = np.random.rand(dimension, 1)

    def __init__(self, data, num_rows, num_cols, learning_rate):
        fig = plt.figure()
        ax1 = fig.add_subplot()
        self.data = data
        x = plt.imshow(self.data)
        plt.show()
        self.learning_rate = learning_rate
        self.neurons = np.ones((num_rows, num_cols, data.shape[2])) * 256 / 2
        # self.neurons = np.random.rand(num_rows, num_cols, data.shape[2]) * 256
        assert (self.neurons.shape == (num_rows, num_cols, 3))

    def train(self, num_iterations):
        """
        Train the self organizing map for num_iterations
        :return:
        """
        sigma_0 = self.neurons.shape[0] / 2
        sigma = sigma_0
        learning_rate = self.learning_rate
        lambd = 10
        for t in range(1, num_iterations + 1):
            x = self.data[np.random.randint(0, self.data.shape[0])][0]
            r, c = self.find_winning_neuron(x)

            # Update the BMU and its neighborhood
            self.update_neighborhood(x, sigma, learning_rate, np.asarray((r, c)))

            # Decay the size of the neighborhood and learning rate with exponential decay
            sigma = sigma_0 * np.exp(-t / lambd)

            # TODO: Let us try not decaying the learning rate
            # learning_rate = self.learning_rate * np.exp(-t / lambd)

            assert (sigma < sigma_0)
            # assert (learning_rate < self.learning_rate)

            # Display the current SOM
            # self.display_map()

    def update_neighborhood(self, x, sigma, learning_rate, bmu):
        for r, row in enumerate(self.neurons):
            for c, neuron in enumerate(row):
                distance = np.linalg.norm(bmu - np.asarray((r, c)))
                neighborhood = np.exp(-(distance ** 2) / (2 * sigma ** 2))
                row[c] -= learning_rate * neighborhood * (row[c] - x)

    def find_winning_neuron(self, input):
        """
        Find the winning neuron in the lattice
        :param input: Randomly chosen feature vector from the input data
        :return: Row and column of the winning neuron
        """
        minr, minc, mindist = 0, 0, None
        for r, row in enumerate(self.neurons):
            for c, neuron in enumerate(row):
                norm = np.linalg.norm(input - neuron)
                if mindist is None or norm < mindist:
                    mindist = norm
                    minr = r
                    minc = c

        return minr, minc

    def display_map(self):
        x = plt.imshow(self.neurons)
        plt.show()
