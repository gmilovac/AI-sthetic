import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re

IMG_SIZE = 256
BATCH_SIZE = 32
TARGET_AESTHETIC_SCORE = 5.5 

AVA_METADATA_CSV = 'data/ground_truth_dataset.csv'
AVA_IMAGE_DIR_BASE = 'data/images/'

def load_and_parse_metadata(metadata_path):
    """
    Loads the AVA metadata CSV, prepares the image paths and labels, 
    and importantly, filters out image IDs that are not found on disk.
    
    Args:
        metadata_path (str): Path to the AVA metadata CSV file.
    
    Returns:
        tuple: (list of existing image paths, numpy array of corresponding aesthetic scores)
    """
    try:
        df = pd.read_csv(metadata_path)
    except FileNotFoundError:
        print(f"ERROR: Metadata file not found at {metadata_path}")
        return [], []
    
    if 'image_id' not in df.columns or 'mean_score' not in df.columns:
        print("CSV missing 'image_id' or 'mean_score'. Attempting to load 'AVA.txt' file format.")
        
        ava_txt_path = os.path.join(os.path.dirname(metadata_path), 'AVA_Files/AVA.txt')
        if os.path.exists(ava_txt_path):
            print(f"Found AVA.txt. Re-parsing metadata...")
            
            col_names = ['index', 'image_id'] + [f'score_{i}' for i in range(1, 11)] + \
                        ['mean_score', 'std_dev', 'tag_ids'] 
            
            df = pd.read_csv(ava_txt_path, sep='\s+', names=col_names, skiprows=1, index_col=False)
            df['image_id'] = df['image_id'].astype(int)
        else:
            print("ERROR: Metadata parsing failed. Check if 'ground_truth_dataset.csv' is correct.")
            return [], []


    df['image_id'] = df['image_id'].astype(int) 
    

    image_paths_full = [os.path.join(AVA_IMAGE_DIR_BASE, f"{img_id}.jpg") for img_id in df['image_id']]
    
    print("Checking file existence for all metadata entries...")
    existing_data = [(path, score) for path, score in zip(image_paths_full, df['mean_score'].values) if os.path.exists(path)]
    
    if not existing_data:
        print("CRITICAL ERROR: No image files were found in the specified directory.")
        return [], []

    image_paths = [item[0] for item in existing_data]
    aesthetic_scores = np.array([item[1] for item in existing_data])
    
    num_total = len(df)
    num_existing = len(image_paths)
    if num_existing < num_total:
        print(f"Warning: Filtered {num_total - num_existing} missing images.")
        print(f"Successfully loaded {num_existing} existing metadata entries.")
    else:
        print(f"Successfully loaded {num_existing} metadata entries.")
        
    return image_paths, aesthetic_scores

@tf.function
def preprocess_image_and_labels(image_path, aesthetic_score):
    """
    TensorFlow map function to load, resize, and normalize the image, 
    and convert the aesthetic score to binary labels.
    
    Note: Since we filtered the data in load_and_parse_metadata, 
    we no longer need the slower py_function fallback here, 
    but we will keep the original implementation for robustness.
    
    Args:
        image_path (tf.Tensor): Tensor representing the file path string.
        aesthetic_score (tf.Tensor): Tensor representing the continuous mean score.
        
    Returns:
        tuple: (image tensor, (real/fake label, aesthetic label))
    """
    img = tf.io.read_file(image_path)
    
    def py_decode_and_resize(img_path_tensor, img_str):
        img_path = img_path_tensor.numpy().decode('utf-8')
        try:
            img = tf.image.decode_jpeg(img_str, channels=3) 
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
            return img
        except tf.errors.InvalidArgumentError:
            tf.print(f"Warning: Failed to process image: {img_path}")
            return tf.constant(0.0, shape=[IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32)

    img = tf.py_function(
        func=py_decode_and_resize, 
        inp=[image_path, img], 
        Tout=tf.float32
    )
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])

    img = (img / 127.5) - 1.0

    # Aesthetic Label: Binary classification (Aesthetic >= 5.5 is True/1.0)
    
    aesthetic_score_float = tf.cast(aesthetic_score, tf.float32)
    
    threshold = tf.constant(TARGET_AESTHETIC_SCORE, dtype=tf.float32)
    
    aesthetic_label = tf.cast(aesthetic_score_float >= threshold, tf.float32)
    
    real_fake_label = tf.constant(1.0, dtype=tf.float32)

    return img, (real_fake_label, aesthetic_label)

def create_ava_dataset(metadata_path=AVA_METADATA_CSV, batch_size=BATCH_SIZE):
    """
    Builds the optimized tf.data.Dataset pipeline.
    """
    image_paths, aesthetic_scores = load_and_parse_metadata(metadata_path)
    
    if not image_paths or len(image_paths) == 0:
        print("Dataset creation failed: No valid image paths found.")
        return None

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, aesthetic_scores))

    dataset = dataset.shuffle(buffer_size=1024, reshuffle_each_iteration=True) 
    
    dataset = dataset.map(preprocess_image_and_labels, 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    try:
        print("Attempting to cache dataset. This may take a moment on first run.")
        dataset = dataset.cache(filename='./tf_cache')
    except:
        print("Warning: Failed to create file-based cache. Training may be slower.")
        pass
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    print(f"Starting Data Pipeline Creation...")
    print(f"Looking for metadata at: {AVA_METADATA_CSV}")
    print(f"Looking for images in base directory: {AVA_IMAGE_DIR_BASE}")

    train_ds = create_ava_dataset(AVA_METADATA_CSV, BATCH_SIZE)

    if train_ds:
        print("\n--- Pipeline Check ---")
        for images, labels in train_ds.take(1):
            real_fake_labels, aesthetic_labels = labels
            
            print(f"Batch Size: {images.shape[0]}")
            print(f"Image Batch Shape: {images.shape}")
            print(f"Real/Fake Labels Shape: {real_fake_labels.shape} (All 1.0)")
            print(f"Aesthetic Labels Shape: {aesthetic_labels.shape} (0.0 or 1.0)")
            
            min_val = tf.reduce_min(images).numpy()
            max_val = tf.reduce_max(images).numpy()
            print(f"Image Pixel Min/Max: {min_val:.2f} / {max_val:.2f}")

            mean_aesthetic = tf.reduce_mean(aesthetic_labels).numpy()
            print(f"Aesthetic Label Mean (Proportion of Aesthetic Images in Batch): {mean_aesthetic:.2f}")
            
            break