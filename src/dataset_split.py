import os
import shutil
import random

def create_train_val_test_split(base_path, output_base_path, test_ratio=0.3, val_ratio=0.2):
    movements = [f'Movimiento_{i}' for i in range(1, 13)]
    phases = ['Train', 'Validation', 'Test']

    for phase in phases:
        for movement in movements:
            os.makedirs(os.path.join(output_base_path, phase, movement), exist_ok=True)

    for movement in movements:
        movement_path = os.path.join(base_path, movement)
        images = os.listdir(movement_path)
        random.shuffle(images)

        num_images = len(images)
        num_test = int(num_images * test_ratio)
        num_val = int(num_images * (1 - test_ratio) * val_ratio)

        test_images = images[:num_test]
        val_images = images[num_test:num_test + num_val]
        train_images = images[num_test + num_val:]

        for img in test_images:
            shutil.copy(os.path.join(movement_path, img), os.path.join(output_base_path, 'Test', movement, img))

        for img in val_images:
            shutil.copy(os.path.join(movement_path, img), os.path.join(output_base_path, 'Validation', movement, img))

        for img in train_images:
            shutil.copy(os.path.join(movement_path, img), os.path.join(output_base_path, 'Train', movement, img))

# Ejemplo de uso
base_path = '/content/drive/My Drive/PAPER/'
output_base_path = '/content/drive/My Drive/PAPER/'
create_train_val_test_split(base_path, output_base_path)
