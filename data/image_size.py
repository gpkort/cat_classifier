import os
from PIL import Image
import tensorflow as tf

def get_png_file_dimensions(directory):
    png_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg')]
    file_dimensions = {}
    for f in png_files:
        with Image.open(os.path.join(directory, f)) as img:
            file_dimensions[f] = img.size  # (width, height)
    return file_dimensions

def main() -> tuple[list[str], int]:
    size_list: list[str] = []
    file_count:int = 0

    for dir, subdirs, files in os.walk("image_data"):
        file_count = len(files)
        for i, file in enumerate(files):
            if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                with Image.open(os.path.join(dir, file)) as img:
                    sz: str = f"{img.size[0]}x{img.size[1]}"
                    if sz not in size_list:
                        size_list.append(sz)
                        
    return size_list, file_count
                
                
    # directory = "data/dataset-part1"
    # # directory = input("Enter the directory path: ")
    # # if not os.path.isdir(directory):
    # #     print("Invalid directory path")
    # #     return

    # file_dimensions = get_png_file_dimensions(directory)
    # if not file_dimensions:
    #     print("No .png files found in the directory")
    # else:
    #     for file, dimensions in file_dimensions.items():
    #         print(f"{file}: {dimensions[0]}x{dimensions[1]} (width x height)")

if __name__ == "__main__":
    print(main())
    