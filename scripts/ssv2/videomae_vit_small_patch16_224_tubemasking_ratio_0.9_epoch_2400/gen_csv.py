import os
import random


def findAllFile(base, ext=None):
    file_path = []
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            if not ext is None and not fullname.endswith(ext):
                continue
            file_path.append(fullname)
    return file_path


def gen(paths):
    all_csv = {
        'train': [],
        'val': [],
        'test': []
    }
    labels = ['hy', 'slight', 'mild']
    for path in paths:
        mode, label, name = path.split('/')[-3:]
        all_csv[mode].append(f'{path} {labels.index(label)}\n')
    # print(all_csv)
    for key in all_csv:
        with open(f'{key}.csv', 'w') as file:
            random.shuffle(all_csv[key])
            file.writelines(all_csv[key])
        

if __name__ == '__main__':        
    root = '/root/proj/PD/MAEVideo/videos_all'
    paths = findAllFile(root)
    gen(paths)