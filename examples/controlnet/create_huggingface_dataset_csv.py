import os
import json
import pandas as pd
from tqdm import tqdm
import pathlib
from pathlib import Path
import argparse

def bdd_to_hf_csv(image_folder: Path, condition_folder: Path, caption_folder: Path, output_folder: Path, output_csv_filename: str='hf_dataset_semseg.csv'):
    """
    Processes the dataset by matching images, condition files, and JSON captions that share the same base filename.
    Creates a CSV file that maps image paths, condition paths, and captions.
    Parameters:
    - image_folder: Path to the folder containing the image files.
    - caption_folder: Path to the folder containing the JSON files with captions.
    - condition_folder: Path to the folder containing the condition files (segmentation masks).
    - output_folder: Path to the folder where the output CSV file will be saved
    - output_csv_filename: Name of the output CSV file 
    """
    assert image_folder.exists()
    assert caption_folder.exists()
    assert condition_folder.exists()
    assert output_folder.exists()
    
    os.makedirs(output_folder, exist_ok=True)

    base_filenames = set(p.stem for p in image_folder.glob("*"))
    images_paths = []
    conditions_paths = []
    captions = []

    for file_stem in tqdm(base_filenames):
        image_path: Path = image_folder / (file_stem + ".jpg") 
        condition_path: Path = condition_folder / (file_stem + ".png") 
        caption_path: Path = caption_folder / (file_stem + ".txt") 

        if condition_path.exists() and caption_path.exists():
            images_paths.append(image_path)
            conditions_paths.append(condition_path)
            with open(caption_path, 'r') as jfile:
                json_data = json.load(jfile)
                captions.append(json_data.get('caption', ''))
        else:
            print(f"Missing files for {file_stem}, skipping.")

    df = pd.DataFrame({
        'image': images_paths,
        'condition': conditions_paths,
        'caption': captions
    })
    
    output_csv_path: Path = output_folder / output_csv_filename
    
    df.to_csv(output_csv_path, index=False)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-ip','--images_path',type=str, required=True, help="base path")
    parser.add_argument('-cap','--captions_path', type=str, required=True, help="base path")
    parser.add_argument('-cop','--conditions_path', type=str, required=True, help="base path")
    parser.add_argument('-op','--output_folder', type=str, required=True, help="base path")
    args = parser.parse_args
    
    bdd_to_hf_csv(image_folder=args["images_path"],condition_folder=args["condition_path"],caption_folder=args["caption_path"],output_folder=args["output_folder"])