from six.moves import urllib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random
import pandas as pd
import zipfile
import cv2
import os
import csv


# Download the class for extracting data
class Process_data:
    # Define the path
    def __init__(self, download_path=None, save_path=None, full_path=None):
        self.download_path = download_path if download_path else "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
        self.save_path = save_path if save_path else "image_datasets/image"
        self.full_path = full_path if full_path else "image_datasets/image/GTSRB/Final_Training/Images"

    # File download and decompression
    def download_and_unzipped_file(self, file_name):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        # Combine the download path and the archive save path
        image_url = os.path.join(self.download_path, file_name)
        zip_path = os.path.join(self.save_path, file_name)
        # If this folder is not detected, the folder will be downloaded and decompressed
        if not os.path.isdir(self.full_path):
            # Download the progress bar
            reporthook = self.creat_reporthook(file_name)
            urllib.request.urlretrieve(image_url, zip_path, reporthook=reporthook)
            print(" Download finished.")
            print("Start unzipping")
            # Unzip the zip file
            with zipfile.ZipFile(zip_path, "r") as image_zip:
                image_zip.extractall(path=self.save_path)
            print("Unzipping is complete")
            # Delete the zip file
            os.remove(zip_path)
        # If a folder exists, the download and decompression phase is skipped
        else:
            print("The dataset file is detected and the download is skipped")

    @staticmethod
    # Download progress
    def creat_reporthook(file_name):
        def reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\r{file_name} Downloading: {percent}% complete", end="")

        return reporthook

    @staticmethod
    # Read the CSV file
    def read_csvfile(path):
        annotations = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    with open(csv_path, 'r') as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=';')
                        for row in reader:
                            annotations.append(row)
        return annotations

    # Get a dataset with picture paths, points of interest, categories
    def get_processed_data(self, path):
        annotations = self.read_csvfile(path)
        image_path = []
        image_roi = []

        for csv_data in annotations:
            folder_path = f"{int(csv_data['ClassId']):05d}"
            roi = (int(csv_data['Roi.X1']), int(csv_data['Roi.Y1']),
                   int(csv_data['Roi.X2']), int(csv_data['Roi.Y2']))
            image_roi.append(roi)
            image_full_path = os.path.join(path, folder_path, csv_data['Filename'])
            image_path.append(image_full_path)

        df_images = pd.DataFrame({
            'Image_Path': image_path,
            'Image_roi': image_roi,
            'ClassId': [a['ClassId'] for a in annotations]
        })
        return df_images


# Classes for image processing
class Image_preprocessing:
    def __init__(self):
        # Initialize the training set image generator
        self.train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )
        # Initialize the image generator for development and test sets
        self.dev_test_datagen = ImageDataGenerator(rescale=1. / 255)
        self.train_data_generator = None
        self.test_data_generator = None

    def train_data_process(self, dataset):
        # Load the training image
        self.train_data_generator = self.train_datagen.flow_from_dataframe(
            dataframe=dataset,
            x_col='Image_Path',
            y_col='Class_Id',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=True
        )
        return self.train_data_generator

    def test_data_process(self, dataset):
        # Load tests as well as develop images
        self.test_data_generator = self.dev_test_datagen.flow_from_dataframe(
            dataframe=dataset,
            x_col='Image_Path',
            y_col='Class_Id',
            target_size=(224, 224),
            color_mode='rgb',
            class_mode='categorical',
            batch_size=32,
            shuffle=False
        )
        return self.test_data_generator

    def reset_train_generator(self):
        self.train_data_generator.reset()

    def reset_test_generator(self):
        self.test_data_generator.reset()

    # Point of interest clipping
    def crop_image(self, data_set):
        print("Points of interest are being clipped")
        crop_path = []
        crop_class = []
        for row in data_set.itertuples(index=False):
            image_path = row.Image_Path
            image_class = row.ClassId
            roi = row.Image_roi
            self.crop_and_save_image(image_path, roi, image_path)
            crop_path.append(image_path)
            crop_class.append(image_class)
        crop_df = pd.DataFrame({
            'Image_Path': crop_path,
            'Class_Id': crop_class
        })
        print("Points of interest are cropped")
        return crop_df

    # Trim the points of interest and save the file
    @staticmethod
    def crop_and_save_image(image_path, roi, save_path):
        image = Image.open(image_path)
        cropped_image = image.crop(roi)
        cropped_image.save(save_path)

    @staticmethod
    def show_image(image, class_id):
        plt.imshow(image)
        plt.title(f'Class ID: {class_id}')
        plt.axis('off')
        plt.show()


class Data_analysis:
    # Adjustable parameters
    def __init__(self):
        self.target_size = (30, 30)  # Size of the image
        self.blur_radius = 2  # Gaussian smoothing radius
        self.contrast_factor = 2  # Contrast intensity
        self.rotate = 15  # Random rotation angle
        self.translate_rate = 0.1  # Random translation rate

    @staticmethod
    def read_csvfile(path):
        annotations = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    with open(csv_path, 'r') as csvfile:
                        reader = csv.DictReader(csvfile, delimiter=';')
                        for row in reader:
                            annotations.append(row)
        return annotations

    def get_processed_data(self, path):
        annotations = self.read_csvfile(path)
        image_path = []
        image_roi = []
        image_height = []
        image_width = []
        for csv_data in annotations:
            folder_path = f"{int(csv_data['ClassId']):05d}"
            roi = (int(csv_data['Roi.X1']), int(csv_data['Roi.Y1']),
                   int(csv_data['Roi.X2']), int(csv_data['Roi.Y2']))
            height = csv_data['Height']
            width = csv_data['Width']
            image_roi.append(roi)
            image_full_path = os.path.join(path, folder_path, csv_data['Filename'])
            image_path.append(image_full_path)
            image_height.append(int(height))
            image_width.append(int(width))

        df_images = pd.DataFrame({
            'Image_Path': image_path,
            'Image_roi': image_roi,
            'ClassId': [a['ClassId'] for a in annotations],
            'Width': image_width,
            'Height': image_height
        })
        return df_images

    # Gaussian blur
    def gaussian_blur(self, image):
        return image.filter(ImageFilter.GaussianBlur(self.blur_radius))

        # Contrast enhancement

    def enhance_contrast(self, image):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(self.contrast_factor)

    # Resizing images
    def resize_image(self, image):
        return image.resize(self.target_size)

    # Data augmentation
    def augment_data(self, image):
        img_array = np.array(image)
        rows, cols = self.target_size
        # Setting random and numpy's random seeds
        random.seed(42)
        np.random.seed(42)

        # Random rotate (default from -15 to 15 degrees)
        angle = random.randint(-self.rotate, self.rotate)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        img_rotated = cv2.warpAffine(img_array, rotation_matrix, self.target_size)

        # Random transform (default translation within 10% of width and height)
        tx = np.random.randint(-int(cols * self.translate_rate), int(cols * self.translate_rate))
        ty = np.random.randint(-int(rows * self.translate_rate), int(rows * self.translate_rate))
        m_translate = np.float32([[1, 0, tx], [0, 1, ty]])
        img_translated = cv2.warpAffine(img_rotated, m_translate, self.target_size)

        # Normalisation
        img_normalised = img_translated / 255.0

        return img_normalised

    # Entry function
    def process_image(self, image_path, annotation_roi):
        with Image.open(image_path) as img:
            # Image Processing Pipeline
            # 1 Conversion to greyscale
            # img = self.greyscale(img)
            # 2 ROI cutting (Assuming annotation_roi is a dict with keys 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2')
            img_cropped = img.crop(annotation_roi)
            # 3 Apply Gaussian filter
            img_smoothed = self.gaussian_blur(img_cropped)
            # 4 Image enhancement
            img_enhanced = self.enhance_contrast(img_smoothed)
            # 5 Resizing the cropped image
            img_resized = self.resize_image(img_enhanced)
            # 6 Data augmentation
            img_augmented = self.augment_data(img_resized)

            return img_augmented

    @staticmethod
    def analyze_image_color(image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        channels = ['red', 'green', 'blue']
        stats = {}
        for i, color in enumerate(channels):
            channel = img_array[:, :, i]
            stats[color] = {
                'average_value': np.mean(channel),
                'standard_deviation': np.std(channel),
                'minimum': np.min(channel),
                'maximum': np.max(channel)
            }
        return stats
