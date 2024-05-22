import pandas as pd
import os


def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def process_files(data_dir, max_files=100):
    # List all files in the directory
    files = os.listdir(data_dir)

    # Process only the first 100 files
    for i, file in enumerate(files):
        if i >= max_files:
            break  # Exit the loop after processing 100 files

        file_path = os.path.join(data_dir, file)

        # Check file format and read accordingly
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
            print(f"Contents of {file}:")
            print(df)
        elif file.endswith('.json'):
            df = pd.read_json(file_path)
            print(f"Contents of {file}:")
            print(df)
        else:
            print(f"Unsupported file format: {file}")


if __name__ == '__main__':
    print_hi('Kafo team')

    # Define the directory where the dataset is stored
    data_dir = r'C:\Users\HP\.ir_datasets\antique'

    # Process the files
    process_files(data_dir)
