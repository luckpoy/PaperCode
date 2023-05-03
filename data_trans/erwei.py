import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# 生成二维高斯分布的数据
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
mu = np.array([0, 0])
sigma = np.array([[1, 0], [0, 1]])
rv = multivariate_normal(mu, sigma)
z = rv.pdf(pos)

# 绘制三维高斯分布的密度图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis')

# 设置坐标轴样式
ax.tick_params(top=False, bottom=False, left=False, right=False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Probability Density')


# 显示图形
plt.savefig("erwei.png")
