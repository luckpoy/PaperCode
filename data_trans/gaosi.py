import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-4, 4, 100)
y = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2)

# 绘制图形
fig, ax = plt.subplots()
ax.plot(x, y, color='black', label='standard normal distribution')

# 设置坐标轴样式
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(top=False, bottom=False, left=False, right=False)

# 添加标签和图例
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.legend()
plt.axis('off')

ax.axhline(0, color='black', lw=1)
ax.axvline(0, color='black', lw=1)

# 显示图形
plt.savefig("gaosi.png")
