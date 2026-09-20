import os
import shutil
import math
import random
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def copy_folder_structure_and_files(src_root, dst_root):
    for src_dir, _, files in os.walk(src_root):
        dst_dir = src_dir.replace(src_root, dst_root, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file in files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied file {src_file} to {dst_file}")

def copy_folder_structure(src_root, dst_root):
    for src_dir, _, _ in os.walk(src_root):
        dst_dir = src_dir.replace(src_root, dst_root, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
            print(f"Created folder structure {dst_dir}")

def copy_new_images_to_train(new_images_root, train_root):
    for src_dir, _, files in os.walk(new_images_root):
        dst_dir = src_dir.replace(new_images_root, train_root, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file in files:
            src_file = os.path.join(src_dir, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied new image {src_file} to {dst_file}")

def count_images_in_subfolders(root_folder):
    counts = {}
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            counts[subfolder] = len([
                f for f in os.listdir(subfolder_path)
                if os.path.isfile(os.path.join(subfolder_path, f))
            ])
            print(f"Subfolder {subfolder} has {counts[subfolder]} images")
    return counts

def augment_images(source_folder, augment_count, save_folder):
    aug_transforms = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4),
        A.Blur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 15.0)),
        A.RandomGamma(gamma_limit=(80, 120)),
    ])

    image_files = os.listdir(source_folder)
    num_original_images = len(image_files)
    num_augmentations_per_image = math.ceil(augment_count / num_original_images)

    os.makedirs(save_folder, exist_ok=True)

    for i in tqdm(image_files, total=num_original_images):
        image_path = os.path.join(source_folder, i)
        image = Image.open(image_path).convert('RGB')
        
        for j in range(1, num_augmentations_per_image + 1):
            augmented = aug_transforms(image=np.array(image))
            new_filename = f'{i.replace(".jpg", "")}-{j}.jpg'
            output_path = os.path.join(save_folder, new_filename)
            augmented_image = Image.fromarray(augmented['image'])
            augmented_image.convert('RGB').save(output_path)
            if len(os.listdir(save_folder)) >= augment_count:
                break
        if len(os.listdir(save_folder)) >= augment_count:
            break

def move_augmented_images_to_subfolder(augmented_folder, target_subfolder):
    for file in os.listdir(augmented_folder):
        src_file = os.path.join(augmented_folder, file)
        dst_file = os.path.join(target_subfolder, file)
        print(f"Moving augmented image from {src_file} to {dst_file}")
        shutil.move(src_file, dst_file)
    shutil.rmtree(augmented_folder)
    print(f"Deleted temporary augmented folder {augmented_folder}")

def split_dataset(destination_root, validation_size=400):
    validation_root = os.path.join(destination_root, 'valid')
    train_root = os.path.join(destination_root, 'train')
    os.makedirs(validation_root, exist_ok=True)

    for subfolder in os.listdir(destination_root):
        subfolder_path = os.path.join(destination_root, subfolder)
        if os.path.isdir(subfolder_path) and subfolder not in ['valid', 'train']:
            images = os.listdir(subfolder_path)
            random.shuffle(images)

            validation_images = images[:validation_size] if len(images) >= validation_size else images
            train_images = images[len(validation_images):]

            os.makedirs(os.path.join(validation_root, subfolder), exist_ok=True)
            os.makedirs(os.path.join(train_root, subfolder), exist_ok=True)

            for img in validation_images:
                shutil.move(os.path.join(subfolder_path, img), os.path.join(validation_root, subfolder, img))
            for img in train_images:
                shutil.move(os.path.join(subfolder_path, img), os.path.join(train_root, subfolder, img))

            print(f"Split {subfolder}: {len(validation_images)} valid, {len(train_images)} train")

def split_data(root_folder, valid_count):
    train_data = []
    valid_data = []

    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            image_files = [
                f for f in os.listdir(subfolder_path)
                if os.path.isfile(os.path.join(subfolder_path, f))
            ]

            valid_split = image_files[:valid_count]
            train_split = image_files[valid_count:]

            train_data.append((subfolder_path, train_split))
            valid_data.append((subfolder_path, valid_split))
            print(f"Split {len(valid_split)} valid, {len(train_split)} train in {subfolder}")

    return train_data, valid_data

def option_1(old_data_root, new_images_root):
    destination_root ='/home/nihit/dhanush/Style_Bazar_1'
    print("Starting Option 1: Merge and Balance")

    copy_folder_structure_and_files(old_data_root, destination_root)
    copy_new_images_to_train(new_images_root, destination_root)

    train_counts = count_images_in_subfolders(destination_root)
    max_images_subfolder = max(train_counts, key=train_counts.get)
    max_image_count = train_counts[max_images_subfolder]

    for subfolder, image_count in train_counts.items():
        if subfolder != max_images_subfolder:
            augment_count = max_image_count - image_count
            if augment_count > 0:
                augmented_folder = os.path.join(destination_root, f'{subfolder}_augmented')
                augment_images(os.path.join(destination_root, subfolder), augment_count, augmented_folder)
                move_augmented_images_to_subfolder(augmented_folder, os.path.join(destination_root, subfolder))
                print(f"Augmented {augment_count} images for {subfolder}")

    split_dataset(destination_root, validation_size=100)
    print("Option 1 completed")

def option_2(old_data_root, new_images_root):
    destination_root = '/home/ubuntu/dhanush/max_sea_2'
    train_folder = os.path.join(destination_root, 'train/')
    valid_folder = os.path.join(destination_root, 'valid/')

    print("Starting Option 2: Split and Augment")

    copy_folder_structure(old_data_root, train_folder)
    copy_folder_structure(old_data_root, valid_folder)

    valid_count = 100
    train_data, valid_data = split_data(old_data_root, valid_count)

    for subfolder_path, train_files in train_data:
        for file_name in train_files:
            src_file = os.path.join(subfolder_path, file_name)
            dst_dir = os.path.join(train_folder, os.path.relpath(subfolder_path, old_data_root))
            os.makedirs(dst_dir, exist_ok=True)
            dst_file = os.path.join(dst_dir, file_name)
            shutil.copy2(src_file, dst_file)
            print(f"Copied training image {src_file} to {dst_file}")

    for subfolder_path, valid_files in valid_data:
        for file_name in valid_files:
            src_file = os.path.join(subfolder_path, file_name)
            dst_dir = os.path.join(valid_folder, os.path.relpath(subfolder_path, old_data_root))
            os.makedirs(dst_dir, exist_ok=True)
            dst_file = os.path.join(dst_dir, file_name)
            shutil.copy2(src_file, dst_file)
            print(f"Copied validation image {src_file} to {dst_file}")

    copy_new_images_to_train(new_images_root, train_folder)

    train_counts = count_images_in_subfolders(train_folder)
    max_images_subfolder = max(train_counts, key=train_counts.get)
    max_image_count = train_counts[max_images_subfolder]

    for subfolder, image_count in train_counts.items():
        if subfolder != max_images_subfolder:
            augment_count = max_image_count - image_count
            if augment_count > 0:
                src_folder = os.path.join(train_folder, subfolder)
                augmented_folder = f"{src_folder}_augmented"
                augment_images(src_folder, augment_count, augmented_folder)
                move_augmented_images_to_subfolder(augmented_folder, src_folder)
                print(f"Augmented {augment_count} images for {subfolder}")

    print("Option 2 completed")
def main():
    old_data_root = "/home/nihit/dhanush/Stylebazzar"
    new_images_root = '/home/nihit/dhanush/just'

    option = input("Enter option (1: merge and balance, 2: split and augment): ")

    if option == '1':
        option_1(old_data_root, new_images_root)
    elif option == '2':
        option_2(old_data_root, new_images_root)
    else:
        print("Invalid option entered!")

if __name__ == "__main__":
    main()
