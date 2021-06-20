import numpy as np
from matplotlib import pyplot as plt
import re

with open('../logs/log.txt', 'r') as f:
	logs = f.readlines()

acc1 = []
val_acc1 = []
for line in logs:
	if 'val' in line and line.startswith('['):
		res = re.findall(r'.*acc@1(.*).*val_acc@1(.*).*acc@3.*val_acc@3.*', line)
		acc1.append(float(res[0][0]))
		val_acc1.append(float(res[0][1]))

# val_acc1 = val_acc1[180:210]
x = [i for i in range(1, len(val_acc1)+1)]
plt.plot(x, val_acc1, c='green', alpha=0.5)
plt.scatter(x, val_acc1, c='blue', alpha=0.5)
plt.grid()
plt.savefig('2.jpg')
plt.show()
