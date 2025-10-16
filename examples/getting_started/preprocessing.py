from amelio_cp import Process

"""
    This script goes and finds the raw data in differents formats (.mat, .csv, .xlsx),
    and processes them to create a single dataframe with all the features and label.

    For now, it looks for:
        - Kinematic data in .mat files (one per patient)
        - GPS data in a .csv file
            • This files is created by calculating the GPS values from the kinematic data.
        - Demographic data in an .xlsx file
"""
output_dir = "datasets/sample_1/processed_data"
data_dir = "datasets/sample_1/raw_data"
gps_path = "datasets/sample_1/GPS_25pp.csv"
demographic_path = "datasets/sample_1/6MWT_speed_25pp_caracteristics.xlsx"

all_data = Process().load_data(
    data_dir=data_dir, output_dir=output_dir, gps_path=gps_path, demographic_path=demographic_path, separate_legs=True
)

print(all_data)
