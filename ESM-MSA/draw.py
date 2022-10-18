import matplotlib.pyplot as plt
import copy
import pandas as pd
import numpy as np
import matplotlib as mpl


def generate_coordinates(path, db_path):
	res = pd.read_csv(path, sep='\t')
	db = pd.read_csv(db_path, sep='\t')

	info_dict = {}
	for _, row in db.iterrows():
		cls, seq = row
		info_dict[seq] = cls
		if cls in info_dict.keys():
			info_dict[cls] += 1
		else:
			info_dict[cls] = 1

	# sort results by E value in ascending order
	res = res.sort_values('E-value')
	coordinates = np.zeros((res.shape[0], 2))

	i = 0
	for _, row in res.iterrows():
		query, target, _ = row
		cls = info_dict[query]
		if info_dict[query] == info_dict[target]:
			coordinates[i, 1] = 1

		else:
			coordinates[i, 0] = 1

		i += 1

	coordinates = np.cumsum(coordinates, axis=0)
	print(coordinates)
	return coordinates


def draw(coordinate_list):
	# 高效作图第一步：创建figure和axes
	fig, ax = plt.subplots()  # 添加图和坐标系

	for coordinates in coordinate_list:
		coordinates = copy.copy(coordinates)
		# 对坐标进行放缩变化
		for i, v in enumerate(coordinates[:, 0]):
			if 1 < v <= 10:
				coordinates[i, 0] = (v - 1) / 9 + 1

			elif 10 < v <= 100:
				coordinates[i, 0] = (v - 10) / 90 + 2

			elif 100 < v <= 1000:
				coordinates[i, 0] = (v - 100) / 900 + 3

		coordinates = coordinates[coordinates[:, 0] > 0]
		# 高效作图第二步：添加基础类对象
		ax.plot(coordinates[:, 0], coordinates[:, 1])  # 坐标系-线
		props = {'xlabel': 'Superfamily-weighted false positive pairs',  # 坐标系-坐标轴-标签
		         'ylabel': 'Superfamily-weighted true positive pairs'}  # 坐标系-坐标轴-标签
		ax.set(**props)
	plt.show()