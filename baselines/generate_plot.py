import matplotlib.pyplot as plt
import json

loss_dict = json.load(open('loss.json', 'r'))
episode = [key for key, _ in loss_dict.items()]
v0_loss = [value["v0"] for _, value in loss_dict.items()]
v1_loss = [value["v1"] for _, value in loss_dict.items()]
v2_loss = [value["v2"] for _, value in loss_dict.items()]

plt.plot(episode[2:], v0_loss[2:], color='aquamarine')
plt.xlabel('episode')
plt.ylabel('loss')

plt.plot(episode[2:], v1_loss[2:], color='cornflowerblue')
plt.xlabel('episode')
plt.ylabel('loss')

plt.plot(episode[2:], v2_loss[2:], color='wheat')
plt.xlabel('episode')
plt.ylabel('loss')

plt.show()