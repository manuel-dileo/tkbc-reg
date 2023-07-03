import os
import csv
import sys
import random

# Function to read the CSV, TSV, or TXT file
def read_file(file_name):
    data = []
    delimiter = None
    if file_name.endswith('.csv'):
        delimiter = ','
    elif file_name.endswith('.tsv'):
        delimiter = '\t'
    elif file_name.endswith('.txt'):
        delimiter = ' '
    else:
        print("File format not supported. Please provide a CSV, TSV, or TXT file.")
        sys.exit(1)

    with open(file_name, 'r') as file:
        if delimiter is None:
            return data
        reader = csv.reader(file, delimiter=delimiter)
        for row in reader:
            data.append(row)
    return data

def drop_columns(data, file_name):
    if "bitcoin" in file_name:
        # Drop column relate to edge weight
        data = [row[:2] + row[3:] for row in data]
    elif "reddit" in file_name:
        # Drop the first row (header) + POST_ID + LINK_SENTIMENT and PROPERTIES columns
        data = [row[:2] + [row[3]] for row in data[1:]]

    return data

# Function to write the TSV file
def write_tsv_file(file_name, data):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        for row in data:
            writer.writerow(row)

# Function to add the "dummy" column between the first and second column
def add_dummy_column(data):
    for row in data:
        row.insert(1, "dummy")
    return data

# Function to split the data into train, valid, and test based on the values in the fourth column
def split_tsv_data(data, x, y, split='time'):
    # Calculate the number of rows for each split
    total_rows = len(data)
    train_rows = int(total_rows * (x / 100))
    valid_rows = int(total_rows * (y / 100))
    test_rows = total_rows - train_rows - valid_rows

    if split == 'time':
        # Sort the data based on the values in the fourth column
        data = sorted(data, key=lambda row: row[3])
    elif split == 'random':
        # Shuffle the data randomly
        random.shuffle(data)
    else:
        print('Invalid split method')
        return

    # Split the data based on the shuffled order
    train_data = data[:train_rows]
    valid_data = data[train_rows:train_rows + valid_rows]
    test_data = data[train_rows + valid_rows:]

    return train_data, valid_data, test_data

# Check if the input file and values of x and y are provided
if len(sys.argv) < 5:
    print("Usage: python split_roland_data.py <input_file> <x> <y> [--split time|random]")
    sys.exit(1)

# Get the input file name from the command line
file_path = sys.argv[1]
folder_path = os.path.dirname(file_path)
file_name = os.path.basename(file_path)

# Get the values of x and y from the command line
x = int(sys.argv[2])
y = int(sys.argv[3])

# Set the default split function to time
split_option = 'time'

# Check if the split parameter is provided
if len(sys.argv) >= 5 and sys.argv[4] == "--split":
    # Check if the split method is specified as random
    if len(sys.argv) >= 6 and sys.argv[5] == "random":
        split_option = 'random'

# Read the input file
data = read_file(file_path)
data = drop_columns(data, file_path)

# Add the "dummy" column
data_with_dummy = add_dummy_column(data)
write_tsv_file(f"{folder_path}/{os.path.splitext(file_name)[0]}.tsv", data_with_dummy)


# Split the data using the specified split function
train_data, valid_data, test_data = split_tsv_data(data_with_dummy, x, y, split_option)

# Write separate files for train, valid, and test (without extension)
write_tsv_file(f"{folder_path}/train", train_data)
write_tsv_file(f"{folder_path}/valid", valid_data)
write_tsv_file(f"{folder_path}/test", test_data)

print("Operations completed successfully.")