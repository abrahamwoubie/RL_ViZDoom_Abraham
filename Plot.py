import matplotlib.pyplot as plt
import numpy as np
list_Reward=[
[0.41, 0.57, 0.6, 0.58, 0.41, 0.5, 0.59, 0.32, 0.35, 0.53, 0.46, 0.56, 0.58, 0.5, 0.54, 0.46, 0.48, 0.35, 0.43, 0.46, 0.56, 0.35, 0.41, 0.29, 0.42, 0.46, 0.38, 0.57, 0.4, 0.42, 0.46, 0.38, 0.35, 0.56, 0.5, 0.32, 0.42, 0.36, 0.27, 0.35, 0.25, 0.3, 0.3, 0.5, 0.52, 0.4, 0.52, 0.42, 0.38, 0.41, 0.41, 0.44, 0.42, 0.42, 0.35, 0.18, 0.44, 0.3, 0.5, 0.41, 0.46, 0.41, 0.48, 0.47, 0.56, 0.44, 0.5, 0.56, 0.55, 0.53, 0.4, 0.41, 0.48, 0.53, 0.28, 0.62, 0.56, 0.53, 0.58, 0.66, 0.55, 0.66, 0.69, 0.73, 0.64, 0.7, 0.59, 0.64, 0.74, 0.75, 0.76, 0.7, 0.71, 0.72, 0.66, 0.71, 0.76, 0.78, 0.55, 0.69, 0.78, 0.67, 0.69, 0.68, 0.64, 0.93, 0.64, 0.72, 0.66, 0.7, 0.83, 0.81, 0.76, 0.67, 0.73, 0.73, 0.69, 0.78, 0.84, 0.81, 0.65, 0.72, 0.66, 0.7, 0.67, 0.69, 0.73, 0.72, 0.73, 0.68, 0.68, 0.74, 0.75, 0.64, 0.79, 0.77, 0.68, 0.87, 0.66, 0.69, 0.76, 0.67, 0.57, 0.81, 0.68, 0.7, 0.79, 0.66, 0.67, 0.71]
]
mu_reward = np.mean(list_Reward, axis=0)
std_reward = np.std(list_Reward, axis=0)
number_of_steps=len(list_Reward[0])
time = np.arange(1, number_of_steps + 1, 1.0)
time=np.arange(1, number_of_steps + 1, 1.0)
plt.plot(time, mu_reward, color='green')#, label='Mean Reward')
plt.grid()
plt.show()

'''
plt.fill_between(time, mu_reward-std_reward, mu_reward+std_reward, facecolor='blue', alpha=0.3)
#plt.legend(loc='upper right')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title('Spectrogram')
file_name = 'Spectrogram.png'
plt.savefig('./Learning_Curves/'+file_name)
plt.show()
'''