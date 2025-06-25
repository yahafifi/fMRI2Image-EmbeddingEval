# download_data.py
import os
import requests
import zipfile
import shutil
import config  # Import your config file

# --- Helper Functions ---
def create_directory(path):
    os.makedirs(path, exist_ok=True)

def download_file(url, save_path):
    print(f"Downloading {os.path.basename(save_path)} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded and saved at: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False
    except Exception as e:
        print(f"An error occurred while saving {save_path}: {e}")
        return False

def extract_zip_file(zip_path, extract_path, password=None):
    print(f"Extracting {zip_path} to {extract_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            pwd_bytes = password.encode() if password else None
            zip_ref.extractall(extract_path, pwd=pwd_bytes)
        print("Files extracted successfully.")
        return True
    except zipfile.BadZipFile:
        print(f"Error: {zip_path} is not a valid ZIP file.")
        return False
    except RuntimeError as e:
        if 'password required' in str(e).lower():
             print(f"Error: Password required or incorrect for {zip_path}.")
        else:
            print(f"Error extracting {zip_path}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during extraction: {e}")
        return False

def organize_god_imagenet_files(base_path):
    """ Specific organization for the GOD dataset's ImageNet stimuli zip """
    print(f"Organizing files in {base_path}...")
    images_subfolder = os.path.join(base_path, "images") # The zip might create this subdir

    # Check if the 'images' subfolder exists and contains the expected files/folders
    if not os.path.isdir(images_subfolder):
         print("Expected 'images' subfolder not found after extraction. Skipping organization.")
         # Attempt to find train/test directly in base_path if 'images' doesn't exist
         if not os.path.isdir(os.path.join(base_path, "training")):
              print(f"Critical: 'training' folder not found in {base_path} or {images_subfolder}.")
              return False
         images_subfolder = base_path # Treat base_path as the source if 'images' is missing

    # Move CSV files
    for file_name in ["image_test_id.csv", "image_training_id.csv"]:
        current_path = os.path.join(images_subfolder, file_name)
        target_path = os.path.join(base_path, file_name)
        if os.path.exists(current_path):
            try:
                shutil.move(current_path, target_path)
                print(f"Moved {file_name} to {base_path}")
            except Exception as e:
                print(f"Error moving {file_name}: {e}")
        # else:
            # print(f"Warning: {file_name} not found at {current_path}") # Less critical maybe

    # Move "training" and "test" folders
    for folder_name in ["training", "test"]:
        current_path = os.path.join(images_subfolder, folder_name)
        target_path = os.path.join(base_path, folder_name)
        if os.path.exists(current_path) and current_path != target_path: # Avoid moving if already correct
             try:
                shutil.move(current_path, target_path)
                print(f"Moved '{folder_name}' folder to {base_path}")
             except Exception as e:
                print(f"Error moving {folder_name}: {e}")
        elif not os.path.exists(target_path):
             print(f"Warning: '{folder_name}' folder not found at {current_path} or {target_path}")


    # Remove redundant "images" folder ONLY if it's now empty or just contains the moved items
    if images_subfolder != base_path and os.path.exists(images_subfolder):
        try:
            # Check if empty before removing, safer
            if not os.listdir(images_subfolder):
                 shutil.rmtree(images_subfolder)
                 print(f"Removed empty redundant folder: {images_subfolder}")
            else:
                 print(f"Folder {images_subfolder} not empty after moving files, keeping it.")
        except Exception as e:
            print(f"Error removing {images_subfolder}: {e}")


    print("File organization attempted.")
    # Add checks to ensure final structure is correct
    final_train_path = os.path.join(base_path, "training")
    final_test_path = os.path.join(base_path, "test")
    if os.path.isdir(final_train_path) and os.path.isdir(final_test_path):
        print("Verified: 'training' and 'test' folders exist in the target directory.")
        return True
    else:
        print("Error: Final structure verification failed. 'training' or 'test' folder missing.")
        return False

# --- Main Download Function ---
def download_all_data():
    print("--- Starting Data Download and Setup ---")
    # Create base directories
    create_directory(config.GOD_FMRI_PATH)
    create_directory(config.GOD_IMAGENET_PATH)
    create_directory(config.GOD_FEATURES_PATH) # Though maybe not used if we extract fresh
    create_directory(config.MODELS_BASE_PATH)

    # Define files to download {url: target_path}
    # Using URLs - verify these are still active and correct!
    files_to_download = {
        # GOD fMRI Data (Just one subject url and you can change for each subject)
        "": os.path.join(config.GOD_FMRI_PATH, f"Subject{config.SUBJECT_ID}.h5"),

        # GOD ImageNet Stimuli (Zip) 
        # Replace with a stable link if possible. 
        "": os.path.join(config.GOD_IMAGENET_PATH, "image_stimuli.zip"),

        # Precomputed ResNet50 Features (Optional) 
        "": os.path.join(config.GOD_FEATURES_PATH, "feature_imagenet_500_resnet50.pt"),
        "": os.path.join(config.GOD_FEATURES_PATH, "labels_imagenet_500_resnet50.pt"),

        # Precomputed Ridge Model (Optional)
        "": os.path.join(config.MODELS_BASE_PATH, f"ridge_sub{config.SUBJECT_ID}.sav"),

        # Metadata/Mappings (GitHub links seem more stable)
        "": config.CLASS_TO_WORDNET_JSON,

        # GOD Captions (Optional)
        "": os.path.join(config.KAGGLE_BASE_PATH, "GOD_train_caps_hm_and_wdnet.json"), 
    }

    # Download all files
    all_downloads_successful = True
    downloaded_files = {}
    for url, path in files_to_download.items():
         # Skip download if file already exists
        if os.path.exists(path):
            print(f"File already exists, skipping download: {path}")
            downloaded_files[path] = True # Mark as successful for later steps
            continue

        if download_file(url, path):
            downloaded_files[path] = True
        else:
            downloaded_files[path] = False
            all_downloads_successful = False # Mark failure if any download fails

    # --- Post-Download Processing ---

    # Extract GOD ImageNet Stimuli Zip
    zip_file_path = os.path.join(config.GOD_IMAGENET_PATH, "image_stimuli.zip")
    if downloaded_files.get(zip_file_path, False): # Check if download was successful
        # !! IMPORTANT: Provide the correct password if needed !!
        if extract_zip_file(zip_file_path, config.GOD_IMAGENET_PATH, password=""):
            # Organize the extracted files
           if not organize_god_imagenet_files(config.GOD_IMAGENET_PATH):
               print("Error during file organization. Check extraction contents.")
               all_downloads_successful = False
        else:
            print(f"Failed to extract {zip_file_path}. Cannot organize.")
            all_downloads_successful = False
    elif not os.path.isdir(os.path.join(config.GOD_IMAGENET_PATH, "training")):
         print(f"Warning: {zip_file_path} not downloaded and training/test folders not found. Stimuli missing.")


    if all_downloads_successful:
        print("--- Data Download and Setup Completed Successfully ---")
    else:
        print("--- Data Download and Setup Completed with Errors ---")

    return all_downloads_successful


if __name__ == "__main__":
    # This allows running the script directly to download data
    if not download_all_data():
         print("\nPlease check the download URLs and permissions, especially for Google Drive links.")
         print("Ensure the Kaggle environment has internet access enabled.")
