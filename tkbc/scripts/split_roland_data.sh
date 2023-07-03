cd ../

# Default split option
split_option="time"

# Check if the correct number of command-line arguments is provided
if [ "$#" -eq 2 ] && [ "$1" = "--split" ]; then
    split_option="$2"
elif [ "$#" -ne 0 ]; then
    echo "Usage: script.sh [--split time|random]"
    exit 1
fi

# Function to perform time-based split
perform_time_split() {
    # Time-based split logic here
    echo "Performing time-based split..."
    python split_roland_data.py src_data/bitcoinalpha/bitcoinalpha.csv 70 10 --split time
    python split_roland_data.py src_data/bitcoinotc/bitcoinotc.csv 70 10 --split time
    python split_roland_data.py src_data/collegemsg/CollegeMsg.txt 70 10 --split time
    python split_roland_data.py src_data/reddit-body/reddit-body.tsv 70 10 --split time
    python split_roland_data.py src_data/reddit-title/reddit-title.tsv 70 10 --split time
}

# Function to perform random split
perform_random_split() {
    # Random split logic here
    echo "Performing random split..."
    python split_roland_data.py src_data/bitcoinalpha/bitcoinalpha.csv 70 10 --split random
    python split_roland_data.py src_data/bitcoinotc/bitcoinotc.csv 70 10 --split random
    python split_roland_data.py src_data/collegemsg/CollegeMsg.txt 70 10 --split random
    python split_roland_data.py src_data/reddit-body/reddit-body.tsv 70 10 --split random
    python split_roland_data.py src_data/reddit-title/reddit-title.tsv 70 10 --split random
}


# Check the split option and perform the corresponding split
if [ "$split_option" = "time" ]; then
    perform_time_split
elif [ "$split_option" = "random" ]; then
    perform_random_split
else
    echo "Invalid split option. Please use either 'time' or 'random'."
fi