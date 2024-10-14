import os
import shutil
import random
project_name = "biojava"

positive_folder = './' + project_name+'/POSITIVE'
negative_folder = './' + project_name+'/NEGATIVE'
train_folder = './' + project_name+'/TRAIN'
test_folder = './' + project_name+'/TEST'

train_positive_folder = os.path.join(train_folder, 'POSITIVE')
train_negative_folder = os.path.join(train_folder, 'NEGATIVE')
train_all_folder = os.path.join(train_folder, 'ALL')
test_positive_folder = os.path.join(test_folder, 'POSITIVE')
test_negative_folder = os.path.join(test_folder, 'NEGATIVE')
test_all_folder = os.path.join(test_folder, 'ALL')

os.makedirs(train_positive_folder, exist_ok=True)
os.makedirs(train_negative_folder, exist_ok=True)
os.makedirs(train_all_folder, exist_ok=True)
os.makedirs(test_all_folder, exist_ok=True)
os.makedirs(test_positive_folder, exist_ok=True)
os.makedirs(test_negative_folder, exist_ok=True)


positive_files = [f for f in os.listdir(positive_folder) if f.endswith('.json')]
negative_files = [f for f in os.listdir(negative_folder) if f.endswith('.json')]


random.shuffle(positive_files)
random.shuffle(negative_files)

# 计算9:1的划分比例
positive_train_size = int(0.9 * len(positive_files))
negative_train_size = int(0.9 * len(negative_files))

#据此分割训练集和测试集
positive_train_files = positive_files[:positive_train_size]
positive_test_files = positive_files[positive_train_size:]

negative_train_files = negative_files[:negative_train_size]
negative_test_files = negative_files[negative_train_size:]

# 将文件移动到训练集文件夹
for file in positive_train_files:
    shutil.copy(os.path.join(positive_folder, file), os.path.join(train_positive_folder, file))
    shutil.copy(os.path.join(positive_folder, file), os.path.join(train_all_folder, file))

for file in negative_train_files:
    shutil.copy(os.path.join(negative_folder, file), os.path.join(train_negative_folder, file))
    shutil.copy(os.path.join(negative_folder, file), os.path.join(train_all_folder, file))

# 将文件移动到测试集文件夹
for file in positive_test_files:
    shutil.copy(os.path.join(positive_folder, file), os.path.join(test_positive_folder, file))
    shutil.copy(os.path.join(positive_folder, file), os.path.join(test_all_folder, file))

for file in negative_test_files:
    shutil.copy(os.path.join(negative_folder, file), os.path.join(test_negative_folder, file))
    shutil.copy(os.path.join(negative_folder, file), os.path.join(test_all_folder, file))




print(f"训练集: {len(positive_train_files)} positive, {len(negative_train_files)} negative")
print(f"测试集: {len(positive_test_files)} positive, {len(negative_test_files)} negative")
