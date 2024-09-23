import os
import json
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
project_name = "logging-log4j2"

source_folder = './' + project_name
positive_folder = './' + project_name+'/POSITIVE'
negative_folder = './' + project_name+'/NEGATIVE'

os.makedirs(positive_folder, exist_ok=True)
os.makedirs(negative_folder, exist_ok=True)

json_files = [f for f in os.listdir(source_folder) if f.endswith('.json')]

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as json_file:
            return json.load(json_file)
    except json.JSONDecodeError:
        print(f"文件 {file_path} 不是有效的JSON文件，跳过...")
    except PermissionError:
        print(f"文件 {file_path} 被占用，跳过...")
    return None


def move_file(file_path, target_folder):
    try:
        target_path = os.path.join(target_folder, os.path.basename(file_path))
        shutil.move(file_path, target_path)
    except PermissionError:
        print(f"文件 {file_path} 被占用，无法移动...")


tasks = []

for filename in tqdm(json_files, desc="Reading JSON files"):
    file_path = os.path.join(source_folder, filename)
    data = read_json_file(file_path)

    if data and 'sample_type' in data:
        target_folder = positive_folder if data['sample_type'] == 'POSITIVE' else negative_folder
        tasks.append((file_path, target_folder))

# 多线程移动文件，避免文件占用问题
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(move_file, task[0], task[1]) for task in tasks]
    for _ in tqdm(as_completed(futures), total=len(futures), desc="Moving Files"):
        pass
