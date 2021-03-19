
import numpy as np
from deap import benchmarks


class GroundTruth(object):

    """ Ground Truth generator class.
        It creates a ground truth within the specified navigation map.
        The ground truth is generated randomly following some realistic rules of the enviornment
        and using a Shekel function.
    """

    def __init__(self, grid, resolution, max_number_of_peaks=None):

        """ Maximum number of peaks encountered in the scenario. """
        self.max_number_of_peaks = 6 if max_number_of_peaks is None else max_number_of_peaks

        """ random map features creation """
        self.grid = grid
        self.resolution = resolution
        self.xy_size = np.array([self.grid.shape[1]/self.grid.shape[0]*10, 10])

        # Peaks positions bounded from 1 to 9 in every axis
        self.number_of_peaks = np.random.randint(1, self.max_number_of_peaks+1)
        self.A = np.random.rand(self.number_of_peaks, 2) * self.xy_size * 0.9 + self.xy_size*0.1
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = np.random.rand(self.number_of_peaks, 1) * 4 + 1

        """ Creation of the map field """
        self._x = np.arange(0, self.grid.shape[1],
                            self.resolution)
        self._y = np.arange(0, self.grid.shape[0],
                            self.resolution)

        self._x, self._y = np.meshgrid(self._x, self._y)

        self._z, self.meanz, self.stdz, self.normalized_z = None, None, None, None # To instantiate attr after assigning in __init__
        self.create_field()  # This method creates the normalized_z values

    def shekel_arg0(self, sol):

        return np.nan if self.grid[sol[1]][sol[0]] == 1 else \
            benchmarks.shekel(sol[:2]/np.array(self.grid.shape)*10, self.A, self.C)[0]

    def create_field(self):

        """ Creation of the normalized z field """
        self._z = np.fromiter(map(self.shekel_arg0, zip(self._x.flat, self._y.flat)), dtype=np.float32,
                              count=self._x.shape[0] * self._x.shape[1]).reshape(self._x.shape)

        self.meanz = np.nanmean(self._z)
        self.stdz = np.nanstd(self._z)

        if self.stdz > 0.001:
            self.normalized_z = (self._z - self.meanz) / self.stdz
        else:
            self.normalized_z = self._z

    def reset(self):
        """ Reset ground Truth """
        # Peaks positions bounded from 1 to 9 in every axis
        self.number_of_peaks = np.random.randint(1,self.max_number_of_peaks+1)
        self.A = np.random.rand(self.number_of_peaks, 2) * self.xy_size * 0.9 + self.xy_size*0.1
        # Peaks size bounded from a minimum 2.5 to 5
        self.C = np.random.rand(self.number_of_peaks, 1) * 4 + 1
        # Reconstruct the field #
        self.create_field()

    def read(self, position=None):

        """ Read the complete ground truth or a certain position """

        if position is None:
            return self.normalized_z
        else:
            return self.normalized_z[position[0]][position[1]]

    def render(self):

        """ Show the ground truth """
        plt.imshow(self.read(), cmap='inferno', interpolation='none')
        cs = plt.contour(self.read(), colors='royalblue', alpha=1, linewidths=1)
        plt.clabel(cs, inline=1, fontsize=7)
        plt.title("Nº of peaks: {}".format(gt.number_of_peaks), color='black', fontsize=10)
        im = plt.plot(self.A[:, 0]*self.grid.shape[0]/self.resolution/10,
                      self.A[:, 1]*self.grid.shape[1]/self.resolution/10, 'hk', )
        plt.show()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ypacarai_map = np.genfromtxt('occupation_map.csv',delimiter=',',dtype=float)
    gt = GroundTruth(ypacarai_map, 1, max_number_of_peaks=5)

    for i in range(10):
        gt.reset()
        gt.render()






