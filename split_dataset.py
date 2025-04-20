import os
import random
import shutil

original_dataset_dir = 'D:/coding/project/custom_dataset'

base_dir = 'D:/coding/project/data_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

split_ratio = 0.8

for root in [train_dir, val_dir]:
    os.makedirs(root, exist_ok=True)

for class_name in os.listdir(original_dataset_dir):
    class_path = os.path.join(original_dataset_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_point = int(len(images) * split_ratio)
    train_images = images[:split_point]
    val_images = images[split_point:]


    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))

    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print("Veri seti başarıyla ayrıldı.")
