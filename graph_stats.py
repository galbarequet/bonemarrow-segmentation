import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

with open('stats/stats.pkl', 'rb') as f:
    stats = pickle.load(f)

bone_csd = stats['bone_dsc']
fat_csd = stats['fat_dsc']
train_loss = stats['train_loss']
valid_loss = stats['valid_loss']

bone_csd = [np.mean(bone_csd[i*10: i*10 + 10]) for i in range(int(len(bone_csd)/10))]
fat_csd = [np.mean(fat_csd[i*10: i*10 + 10]) for i in range(int(len(fat_csd)/10))]
train_loss = [np.mean(train_loss[i*10: i*10 + 10]) for i in range(int(len(train_loss)/10))]
valid_loss = [np.mean(valid_loss[i*10: i*10 + 10]) for i in range(int(len(valid_loss)/10))]

x_axis = [i for i in range(len(bone_csd))]

"""
bone_csd = np.array([np.mean(x) for x in bone_csd])
fat_csd = np.array([np.mean(x) for x in fat_csd])

x_smooth = np.linspace(x_axis.min(), x_axis.max(), 200)

# spline - always goes through all the data points x/y
spl_bone = interpolate.UnivariateSpline(x_axis, bone_csd)
spl_fat = interpolate.UnivariateSpline(x_axis, fat_csd)

bone_csd = interpolate.spline(x_axis, bone_csd, x_smooth)
fat_csd = interpolate.spline(x_axis, fat_csd, x_smooth)
"""

plt.plot(x_axis, bone_csd, label='bone csd')
plt.plot(x_axis, fat_csd, label='fat csd')
plt.legend()
plt.show()

x_axis = [i for i in range(len(train_loss))]

plt.plot(x_axis, train_loss, label='train loss')
plt.plot(x_axis, valid_loss, label='valid loss')
plt.legend()
plt.show()
