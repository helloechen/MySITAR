import os
import json
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
project_name = "zeppelin"

test_folder = './' + project_name + './test/ALL'
train_folder = './' + project_name + './train/ALL'

target = train_folder

json_files = [f for f in os.listdir(target) if f.endswith('.json')]

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的JSON文件，跳过...")
    except PermissionError:
        print(f"文件 {file_path} 被占用，跳过...")
    return None


positive = 0
negative = 0

for filename in tqdm(json_files, desc="Reading JSON files"):
    file_path = os.path.join(target, filename)
    data = read_json_file(file_path)

    if data and 'sample_type' in data:
        if data['sample_type'] == 'POSITIVE':
            positive += 1
        else:
            negative += 1

print(f"训练集POSITIVE：{positive}")
print(f"训练集NEGATIVE：{negative}")