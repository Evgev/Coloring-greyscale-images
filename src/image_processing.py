import cv2
import numpy as np
import os

def list_files_recursively(directory):
      file_list = []
      for root, dirs, files in os.walk(directory):
          for file in files:
              file_list.append(os.path.join(root, file))
      return file_list


def get_dataset(path_to_images_folder: str = None, desired_size = (400, 400), max_samples: int = 100):
  """
  Функия рекурсивно загружает изображения из заданной папки.

  params:
    path_to_images_folder: путь докорневой папки с изображениями
    desired_size: размер изображений на выходе
    max_samples: максимальное количество изображений

  return:
    np.array(dtype = uint8) shape = (количество изображений, desired_size, 3)
  """
  
  if path_to_images_folder is None: return None
  
  images_list = []

  # Проходим по всем файлам в директории
  for filename in list_files_recursively(path_to_images_folder):

      if len(images_list) >= max_samples: break
      
      if filename.endswith(".jpg"):  # Проверяем, что файл является изображением в формате JPG

          # Загружаем изображение в BGR
          image = cv2.imread(filename, cv2.IMREAD_COLOR)

          if (image is not None) and (image.shape[:2] > desired_size):  # Проверяем, что изображение успешно загружено
              # Преобразуем BGR в RGB
              image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              image_resized = cv2.resize(image_rgb, desired_size)
              images_list.append(image_resized)
              

  images_array = np.array(images_list, dtype=np.uint8)

  # Проверяем размерности и тип массива
  print("Shape of images_array:", images_array.shape)  # (количество изображений, высота, ширина, количество каналов)
  print("Data type of images_array:", images_array.dtype)  # dtype будет uint8

  return images_array

# def get_dataset(path_to_images_folder: str = None, dataset_name: str = None , desired_size = (400, 400), max_samples: int = None ):
#   """
#   Функия загружает изображения либо из заданной папки либо из Open Images Dataset, приводит их к заданому размеру.

#   params:
#     path_to_images_folder: путь до папки с изображениями
#     dataset_name: название датасета в Open Images
#     desired_size: размер изображений на выходе

#   return:
#     np.array(dtype = uint8) shape = (количество изображений, desired_size, 3)
#   """
#   import cv2


#   def list_files_recursively(directory):
#       file_list = []
#       for root, dirs, files in os.walk(directory):
#           for file in files:
#               file_list.append(os.path.join(root, file))
#       return file_list

#   if dataset_name is not None:

#     import fiftyone as fo
#     import fiftyone.zoo as foz

#     # Загружаем Open Images Dataset
#     dataset = foz.load_zoo_dataset(dataset_name, max_samples=max_samples)

#     # Папка, куда будут сохраняться изображения
#     images_dir = os.path.join(os.getcwd(), path_to_images_folder)  # Путь к директории
#     os.makedirs(images_dir, exist_ok=True)  # Создаем директорию, если она не существует

#     # Перебираем образцы и копируем их изображения
#     for sample in dataset:
#         image_path = sample.filepath  # Получаем путь к изображению
#         output_path = os.path.join(images_dir, os.path.basename(image_path))  # Путь для сохранения

#         # Копируем изображение в выходную директорию
#         shutil.copy(image_path, output_path)  # Копирование изображения

#   if path_to_images_folder is not None:
#     # Путь к папке с изображениями
#     images_dir = path_to_images_folder


#   images_list = []

#   # Проходим по всем файлам в директории
#   for filename in list_files_recursively(images_dir):

#       if filename.endswith(".jpg"):  # Проверяем, что файл является изображением в формате JPG

#           # Загружаем изображение в BGR
#           image = cv2.imread(filename, cv2.IMREAD_COLOR)

#           if (image is not None) and (image.shape[:2] > desired_size):  # Проверяем, что изображение успешно загружено
#               # Преобразуем BGR в RGB
#               image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#               image_resized = cv2.resize(image_rgb, desired_size)
#               images_list.append(image_resized)
#       if len(images_list) > max_samples: break

#   images_array = np.array(images_list, dtype=float)

#   # Проверяем размерности и тип массива
#   print("Shape of images_array:", images_array.shape)  # (количество изображений, высота, ширина, количество каналов)
#   print("Data type of images_array:", images_array.dtype)  # dtype будет uint8


#   return images_array

