import numpy as np

# 读取 npz 文件
data = np.load('/data_new2/sz_zzz/Data/Teeth/RD_1/crown_occ/0000001.npz')  # 替换 'your_file.npz' 为你的文件名

# 打印所有的 keys
print("Keys in the npz file:")
for key in data.keys():
    print(key)
