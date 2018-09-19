import numpy as np
import matplotlib.pyplot as plt


class Map(object):
    """
    Class that implements a Kohenen self-organizing map
    :param data: Input data
    :param num_rows: Number of rows of the map
    :param num_cols: Number of columns of the map
    :param learning_rate: Learning rate
    """

    class MapNeuron(object):
        def __init__(self, dimension, x, y):
            self.x = x
            self.y = y
            self.weights = np.random.normal(loc=1, size=(1, dimension))
            self.weights = np.random.rand(1, dimension)
            x = 2

        def get_distance(self, x):
            return np.linalg.norm(x - self.weights)

        def get_lattice_distance(self, neuron):
            return np.linalg.norm(np.asarray((self.x, self.y)) - np.asarray((neuron.x, neuron.y)))

        def adjust_weights(self, x, learning_rate, influence):
            dist = np.linalg.norm(x - self.weights)
            self.weights = self.weights + learning_rate * influence * (x - self.weights)
            # assert (dist >= np.linalg.norm(x - self.weights))

    def __init__(self, data, num_rows, num_cols):
        plt.ion()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.data = data
        self.neurons = []
        for r in range(num_rows):
            for c in range(num_cols):
                self.neurons.append(self.MapNeuron(data.shape[2], r, c))

        self.colors = np.array([neuron.weights for neuron in self.neurons])
        self.colors = np.reshape(self.colors, (num_rows, num_cols, self.data.shape[2]))
        self.map_radius = max(num_rows, num_cols) / 2
        self.color_map = plt.imshow(self.colors)

    def train(self, initial_learning_rate=0.1, nb_iterations=100, show_matrix=False):
        """
        Train the self organizing map for num_iterations
        :return:
        """

        learning_rate = initial_learning_rate
        time_constant = nb_iterations / np.log(self.map_radius)

        for t in range(nb_iterations):
            # Calculate the radius of the bmu's neighborhood
            neighborhood_radius = self.map_radius * np.exp(-t / time_constant)

            np.random.shuffle(self.data)
            for x in self.data:

                # Choose a vector at random from the training set
                # x = self.data[np.random.randint(0, self.data.shape[2])]

                # Find the Best-matching unit
                bmu = self.find_winning_neuron(x)

                # Adjust the weight of the BMU and its neighbors
                for neuron in self.neurons:
                    lattice_distance = neuron.get_lattice_distance(bmu)
                    # if lattice_distance < neighborhood_radius:
                    influence = np.exp(-lattice_distance ** 2 / (2 * neighborhood_radius ** 2))
                    neuron.adjust_weights(x, learning_rate, influence)

            # TODO: Experiment with the other constant here
            # Adjust the learning rate
            learning_rate = initial_learning_rate * np.exp(-t / nb_iterations)

            if show_matrix:
                self.display_map()

    def find_winning_neuron(self, x):
        return min(self.neurons, key=lambda neuron: neuron.get_distance(x))

    def display_map(self):
        colors = np.array([neuron.weights for neuron in self.neurons])
        self.colors = np.reshape(colors, (self.num_rows, self.num_cols, self.data.shape[2]))
        self.color_map.set_data(self.colors)
        plt.pause(0.000005)
