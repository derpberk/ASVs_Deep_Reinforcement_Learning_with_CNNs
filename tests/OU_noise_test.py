from exploration_strategies.OUNoise import OrnsteinUhlenbeckActionNoise
import numpy as np
import matplotlib.pyplot as plt

mu = np.array([0.0,0.0])
sigma = 0.03
theta = 0.2
dt = 1
seed = 0

explorative_noise = OrnsteinUhlenbeckActionNoise(mu=mu, sigma=sigma, theta=theta, dt=dt, seed=seed)

N = 1000

v = np.array([[0,0]])
explorative_noise.reset()
for i in range(N):
    v = np.vstack((v,explorative_noise()))


plt.plot(v[:,0],'ro')
plt.plot(v[:,1],'b-o')


plt.show()
