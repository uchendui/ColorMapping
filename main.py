import numpy as np
from Map import Map


def main():
    data = np.random.rand(15, 1, 3) * 256
    map = Map(data, 20, 30, 0.01)
    map.display_map()
    map.train(2000)
    map.display_map()


main()
