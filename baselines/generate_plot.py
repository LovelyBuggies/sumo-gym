import matplotlib.pyplot as plt
import json

loss_dict = json.load(open('loss.json', 'r'))
episode = [key for key, _ in loss_dict.items()]
v0_loss = [value["v0"] for _, value in loss_dict.items()]
plt.grid(True)

plt.plot(episode, v0_loss, color='maroon', marker='o')
plt.xlabel('episode')
plt.ylabel('loss')

plt.show()