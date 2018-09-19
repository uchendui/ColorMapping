import numpy as np
from Map import Map


def main():
    data = np.random.rand(4, 1, 3)
    map = Map(data, 20, 30)
    map.train(nb_iterations=3000, initial_learning_rate=1, show_matrix=True)
    map.display_map()
    input("press any key...")


main()
