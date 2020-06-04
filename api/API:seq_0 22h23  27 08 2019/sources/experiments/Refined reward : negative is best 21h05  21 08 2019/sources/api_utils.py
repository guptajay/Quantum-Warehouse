import os
from abc import ABC, abstractclassmethod
from warehouse.envs.constants.basic_constants import initial_filled_spots_ratio, packages
from warehouse.envs.utils import Counter
import numpy as np


class APIInputOuput(ABC):

    def __init__(self, root='./api', init_path="init", output_path="sequence", tag="test"):
        self.root = root
        self.init_path = os.path.join(self.root, init_path)
        self.output_path = os.path.join(self.root, output_path)
        self.sequence_tag = tag

    @abstractclassmethod
    def from_csv(self):
        '''
        Takes csv file as input, with two first columns being spot, package. Returns dictionnay
        {'spot_ids': [id1, ..],
        'package_ids : [id1, ..]}
        '''
        pass

    def to_csv(self, *args):
        pass

class APIWarehouseState(APIInputOuput):
    """
    Class use descrive warehouse state to the hardware side. Takes csv, keep relevant information (spot index,
    package ids), removing irrelevant information
    """
    def __init__(self, fname="warehouse"):
        super().__init__()
        self.fname = fname
        self.sequence_counter = Counter()

    def from_csv(self, init=False):

        path = self.output_path if not init else self.init_path
        warehouse = {}

        fname = self.fname if init else "{}_{}_{}".format(self.fname, self.sequence_tag, self.sequence_counter.value)

        with open(os.path.join(path, "{}.csv".format(fname)), "r") as f:
            lines = f.readlines()

            warehouse.update({"spot_ids": [int(x.split(',')[0]) for x in lines[1:]]})
            warehouse.update({"package_ids": [int(x.split(',')[1]) for x in lines[1:]]})

        return warehouse

    def to_csv(self, spot_ids, package_ids):

        self.sequence_counter()
        fname = "{}_{}_{}.csv".format(self.fname, self.sequence_tag, self.sequence_counter.value)

        lines = ["Spot, Package, S/N\n"]+["{}, {}, 999\n".format(spot_id, package_id) for spot_id, package_id in zip(spot_ids, package_ids)]

        with open(os.path.join(self.output_path, fname), 'w') as f:
            f.writelines(lines)


class APIQueue(APIInputOuput):

    def __init__(self, fname="queue"):
        super().__init__()
        self.fname = fname
        self.sequence_counter = Counter()

    def from_csv(self, init=False):
        path = self.output_path if not init else self.init_path
        queue = {}

        with open(os.path.join(path, "{}.csv".format(self.fname)), "r") as f:
            lines = f.readlines()

            queue.update({"package_ids": [int(x.split(',')[0]) for x in lines[1:]]})

        return queue

    def to_csv(self, package_ids):

        self.sequence_counter()
        fname = "{}_{}_{}.csv".format(self.fname, self.sequence_tag, self.sequence_counter.value)

        lines = ["Package, S/N\n"]+["{}, 999\n".format(package_id) for package_id in package_ids]

        with open(os.path.join(self.output_path, fname), 'w') as f:
            f.writelines(lines)


class WarehouseFileGenerator(object):

    def __init__(self, n_spots=50, root="api", path="init", fname="warehouse.csv"):
        self.root = root
        self.n_spots = n_spots
        self.path = os.path.join(root, path)
        self.fname = fname

    def generate(self):

        packages_ = np.zeros(self.n_spots)
        mask = np.random.choice(range(self.n_spots), size=int(self.n_spots*initial_filled_spots_ratio))
        packages_[mask] = np.random.choice([package_id for package_id in packages],
                                     int(self.n_spots*initial_filled_spots_ratio),
                     p=[packages[package_id]['frequency'] for package_id in packages])

        lines =["Spot, Package, S/N\n"]+["{}, {}, 999\n".format(spot_id, int(package_id)) for spot_id, package_id in
                                         zip(range(self.n_spots), packages_)]

        with open(os.path.join(self.path, self.fname), 'w') as f:
            f.writelines(lines)


if __name__ == "__main__":

    w = WarehouseFileGenerator()
    w.generate()



