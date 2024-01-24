import pandas as pd
import os

def save_column_to_individual_txt_files(csv_file_path, output_folder, column_name):
    """
    This function reads a CSV file and saves the content of a specified column from each row as an individual text file.

    :param csv_file_path: Path to the CSV file.
    :param output_folder: Path to the folder where individual text files will be saved.
    :param column_name: The name of the column to be saved as text files.
    """
    data_df = pd.read_csv(csv_file_path)

    # Ensure the column exists in the DataFrame
    if column_name not in data_df.columns:
        raise ValueError(f"Column '{column_name}' not found in the CSV file.")

    os.makedirs(output_folder, exist_ok=True)

    for index, row in data_df.iterrows():
        text_file_path = os.path.join(output_folder, f"row_{index}.txt")
        with open(text_file_path, 'w') as text_file:
            text_file.write(str(row[column_name]))

csv_file_path = '/Users/raosamvr/Downloads/pca/second_iter.csv' 
output_folder = '/Users/raosamvr/Downloads/pca/individual files/txt/'     
column_name = 'TEXT'             
save_column_to_individual_txt_files(csv_file_path, output_folder, column_name)

