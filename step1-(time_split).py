import os
import json
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor,as_completed


project = './' + 'zeppelin'
train_folder = os.path.join(project, 'train','ALL')
test_folder = os.path.join(project, 'test','ALL')


os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)


json_files = []
for filename in os.listdir(project):
    if filename.endswith('.json'):
        file_path = os.path.join(project, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
            # 提取 prod_time 的 $date 值
            date_value = data['prod_time']['$date']
            json_files.append((date_value, filename))


json_files.sort(key=lambda x: x[0])


split_index = int(len(json_files) * 0.9)


def move_file(file_info, destination):
    date_value, filename = file_info
    shutil.move(os.path.join(project, filename), os.path.join(destination, filename))

# 使用多线程移动文件，并添加进度条
with ThreadPoolExecutor(max_workers=40) as executor:
    # 移动到训练集
    futures_train = [executor.submit(move_file, file_info, train_folder) for file_info in json_files[:split_index]]

    for future in tqdm(as_completed(futures_train), total=len(futures_train), desc="Moving to Train Set"):
        future.result()  # 阻塞等待每个任务完成

    # 移动到测试集
    futures_test = [executor.submit(move_file, file_info, test_folder) for file_info in json_files[split_index:]]

    for future in tqdm(as_completed(futures_test), total=len(futures_test), desc="Moving to Test Set"):
        future.result()  # 阻塞等待每个任务完成

print("文件已成功移动到训练集和测试集。")




