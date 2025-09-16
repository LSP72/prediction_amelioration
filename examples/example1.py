from amelio_cp import Process
from amelio_cp.models.linear.rf_model import RFRModel


data_dir = '/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/dataset/sample_1/raw_data'
gps_path = '/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/dataset/sample_1/GPS_25pp.csv'
demographic_path = '/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/dataset/sample_1/6MWT_speed_25pp_caracteristics.xlsx'
output_dir = '/Users/mathildetardif/Documents/Python/Biomarkers/prediction_amelioration/dataset/sample_1/processed_data/'

all_data = Process().load_data(data_dir=data_dir,
                             output_dir=output_dir,
                             gps_path=gps_path,
                             demographic_path=demographic_path,
                             separate_legs=True)

print(all_data)


# Extracting features and labels
VIT_POST = all_data['VIT_POST']
END_POST = all_data['6MWT_POST']
data = all_data.drop(columns=['VIT_POST', '6MWT_POST'])

"""
SVR_VIT = SVRModel()
SVR_VIT.add_data(data, VIT_POST)
SVR_VIT.perf_estimate(n_iter=50)
"""

RF_END = RFRModel()
RF_END.add_data(data, END_POST)
RF_END.perf_estimate(n_iter=100)
