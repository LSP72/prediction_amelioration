from amelio_cp import Process


data_dir = '/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/dataset/sample_1/raw_data'
gps_path = '/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/dataset/sample_1/GPS_25pp.csv'
demographic_path = '/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/dataset/sample_1/6MWT_speed_25pp_caracteristics.xlsx'

all_data = Process().load_data(data_dir=data_dir,
                             gps_path=gps_path,
                             demographic_path=demographic_path,
                             separate_legs=True)

print(all_data)