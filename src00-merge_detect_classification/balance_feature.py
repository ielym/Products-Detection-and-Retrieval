import os
from glob import glob
# import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# feature_path = r'./extrace_feature-Resize.npy'
feature_path = r'./merge.npy'

feature_classes = np.load(feature_path)

classes = feature_classes[..., -1].astype(np.int).tolist()
print(classes)

record = {}
for clss in classes:
	if not clss in record.keys():
		record[clss] = 1
	else:
		record[clss] += 1

max_length = max(record.values())

balance_features = []
for clss_id, clss_cnt in record.items():
	clss_feature = feature_classes[feature_classes[..., -1] == clss_id]

	cnt = clss_feature.shape[0]
	while cnt < max_length:

		choise_feature_idxs = np.random.choice(range(clss_feature.shape[0]), 5)
		new_feature = []
		for idx in choise_feature_idxs:
			new_feature.append(clss_feature[idx, :].reshape(1, -1))
		new_feature = np.vstack(new_feature)

		new_feature = np.mean(new_feature, axis=0).reshape(1, -1)
		clss_feature = np.vstack([clss_feature, new_feature])
		cnt += 1
	balance_features.append(clss_feature)

balance_features = np.vstack(balance_features)
print(balance_features.shape)
np.save('./balance.npy', balance_features)
