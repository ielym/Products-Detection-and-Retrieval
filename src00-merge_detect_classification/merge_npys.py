import os
from glob import glob
import numpy as np

npy_paths = glob(os.path.join('./', '*.npy'))
print(npy_paths)

all_feature_classes = []
for path in npy_paths:
	feature_classes = np.load(path)
	print(path, feature_classes.shape)
	all_feature_classes.append(feature_classes)

all_feature_classes = np.vstack(all_feature_classes)
print(all_feature_classes.shape)

np.save('./merge.npy', all_feature_classes)

