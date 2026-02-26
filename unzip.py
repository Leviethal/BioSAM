import os
import zipfile

def unzip_all_zips(root_folder, delete_zip=False):
    """
    Recursively finds and extracts all zip files inside root_folder.
    
    Args:
        root_folder (str): path to main folder
        delete_zip (bool): if True, deletes zip after extraction
    """

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                
                # Extract to folder with same name as zip (without .zip)
                extract_folder = os.path.join(root, file.replace(".zip", ""))

                print(f"Extracting: {zip_path}")
                print(f"To: {extract_folder}")

                os.makedirs(extract_folder, exist_ok=True)

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)

                if delete_zip:
                    os.remove(zip_path)
                    print(f"Deleted: {zip_path}")

    print("All zip files extracted successfully.")


if __name__ == "__main__":
    root_folder = r"FLARE22_LabeledCase50/images"
    unzip_all_zips(root_folder, delete_zip=False)