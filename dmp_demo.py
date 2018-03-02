import dmp
import numpy as np
import matplotlib.pyplot as plt

# w = np.loadtxt('weights.dat')

target_trajectory = np.loadtxt('hello.dat')
target_trajectory_x = target_trajectory[:, 0]
target_trajectory_y = target_trajectory[:, 1]

lr = 0.5    # learning rate
kernel_num = 10  # number of kernels
kernel_width = 0.001    # kernel width
epoch_num = 10  # number of learning epochs

Wx = [0 for x in range(epoch_num)]  # x target trajectory
Wy = [0 for x in range(epoch_num)]  # y target trajectory

for Ci in range(epoch_num):
    for Cwxi in range(Wx):
        Wx[Cwxi] += lr * (target_trajectory_x[Cwxi])


Sx = 0.95  # start X
Sy = 0.75  # start Y
Gx = 1.35  # goal X
Gy = 0.10  # goal Y
T = 70  # movement duration (in steps)
s0 = 0.03  # kernel width

# generate X trajectory
X = dmp.generate_trajectory(Sx, Gx, T, Wx, s0)
# generate Y trajectory
Y = dmp.generate_trajectory(Sy, Gy, T, Wy, s0)

# save trajectory to a file
data = np.array([X, Y])
np.savetxt('trajectory.dat', data.T, fmt='%12.6f %12.6f')

# plot trajectories
plt.plot(X, Y, ' b')
plt.xlabel('Trajectory X')
plt.ylabel('Trajectory Y')
plt.show()

# compute squared velocity
# V=np.sqrt( np.array(pow(np.array(np.diff(X)),2) + pow(np.array(np.diff(Y)),2)) )

# signal upsampling via interpolation
# t=np.arange(0,T)
# t_new=np.arange(0,T-1,0.05)
# fx = intrp.interp1d(t, X, 'cubic')
# fy = intrp.interp1d(t, Y, 'cubic')
# X = fx(t_new)
# Y = fy(t_new)
