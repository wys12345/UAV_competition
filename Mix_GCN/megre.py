import numpy as np

# 文件名列表
# file_names = ["M/joint.npy", "M/bone.npy", "M/jm.npy", "M/bm.npy", "M/jb.npy","G/joint.npy", "G/bone.npy", "G/jm.npy", "G/bm.npy", "G/jb.npy", "B/joint.npy", "B/bone.npy", "B/jm.npy", "B/bm.npy", "B/jb.npy", "MS/joint.npy", "MS/bone.npy", "MS/jm.npy", "MS/bm.npy", "MS/jb.npy"]

file_names = ["M/joint.npy", "M/bone.npy", "M/jm.npy", "M/bm.npy", "M/jb.npy","G/joint.npy", "G/bone.npy", "G/jm.npy", "G/bm.npy", "G/jb.npy", "B/joint.npy", "B/bone.npy", "B/jm.npy", "B/bm.npy", "B/jb.npy", "T/joint.npy", "T/bone.npy", "T/jm.npy", "T/bm.npy", "T/jb.npy"]

# file_names = ["M/joint.npy", "M/bone.npy", "M/jm.npy", "M/bm.npy", "M/jb.npy","G/joint.npy", "G/bone.npy", "G/jm.npy", "G/bm.npy", "G/jb.npy", "B/joint.npy", "B/bone.npy", "B/jm.npy", "B/bm.npy", "B/jb.npy"]

# 权重列表
weights = [1.2, 1.2, 1.2, 0.2, 1.2, 1.2, 0.8086936792522563, 0.2, 0.2, 0.2, 1.2, 1.2, 0.2, 0.2, 0.5231386815356827, 0.44442874390598386, 0.8679991412973587, 0.2, 0.2, 1.2]
# weights = [1.2, 1.2, 0.706054431859225, 0.2, 1.2, 1.2, 1.2, 1.2, 0.2, 1.2, 0.8122018542344545, 1.2, 0.6832454237053911, 0.2, 1.2, 0.2, 0.2, 0.2, 0.2, 0.2]
# weights = [1.2, 1.2, 0.2, 0.2, 0.6453971863004777, 1.2, 1.2, 1.2, 0.2, 0.2, 0.2, 1.2, 0.2, 0.2, 0.7884769319871978, 0.2, 0.2, 0.2, 0.2, 0.8559374490478839]
# 加载每个文件并乘以对应的权重
weighted_data = []
for file, weight in zip(file_names, weights[:-1]):  # 最后一项权重为0.2不对应任何文件
    data = np.load(file)
    weighted_data.append(data * weight)

# 合并所有加权后的数据
final_data = np.sum(weighted_data, axis=0)

# 保存结果到pred.npy文件
np.save("result/pred.npy", final_data)

# 返回最终数据形状以确认
print("Shape of final combined data:", final_data.shape)
