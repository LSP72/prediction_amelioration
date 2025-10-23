import pandas as pd
from pandas import DataFrame, Series
import amelio_cp.processing.old_featurExtractor as mat_to_df


class Process:
    def __init__(self):
        pass

    def load_csv(file_path: str) -> DataFrame:
        df = pd.read_csv(file_path, index_col=False)
        return df

    @staticmethod
    def calculate_ROM(df: DataFrame):

        print(df.columns)

        df["ROM_Hip_Sag"] = df["Max_Hip_flx/ext"] - df["Min_Hip_flx/ext"]
        df["ROM_Hip_Frontal"] = df["Max_Hip_abd/add"] - df["Min_Hip_abd/add"]
        df["ROM_Hip_Trns"] = df["Max_Hip_int/ext rot"] - df["Min_Hip_int/ext rot"]
        df["ROM_Knee_Sag"] = df["Max_Knee_flx/ext"] - df["Min_Knee_flx/ext"]
        df["ROM_Ankle_Sag"] = df["Max_Ankle_flx/ext"] - df["Min_Ankle_flx/ext"]

        # TODO: putting the following lines in the function that will do the
        # feature engineering
        df.drop(
            [
                "Min_Knee_abd/add",
                "Min_Knee_int/ext rot",
                "Min_Ankle_abd/add",
                "Min_Ankle_int/ext rot",
                "Max_Knee_abd/add",
                "Max_Knee_int/ext rot",
                "Max_Ankle_abd/add",
                "Max_Ankle_int/ext rot",
            ],
            axis=1,
            inplace=True,
        )

        return df

    @staticmethod
    def load_gps(file_path: str) -> DataFrame:
        df = pd.read_csv(file_path, index_col=False)
        df.drop(columns=["Unnamed: 0", "Post"], inplace=True)
        df.rename(columns={"Pre": "GPS"}, inplace=True)
        return df

    @staticmethod
    def load_demo(file_path: str):

        df = pd.read_excel(file_path)
        df["BMI"] = df["masse"] / ((df["taille"] / 100) ** 2)

        # The numbers are in the order so 1 is walker.
        df.replace(["walker", "cane", "none"], [1, 2, 3], inplace=True)

        # ----- Add Label -----
        df.drop(["delta6MWT", "deltaV"], axis=1, inplace=True)
        # This line excludes some of the features that are in the demographic excel file.
        df.drop(["Patient", "masse", "taille", "sex", "Diagnostique"], axis=1, inplace=True)

        return df

    def load_data(
        self, data_dir: str, output_dir: str, gps_path: str, demographic_path: str, separate_legs: bool = True
    ):
        """
        Load data from raw data

        Parameters
        ----------
        data_path : str
            DESCRIPTION.
        separate_legs : bool, optional
            DESCRIPTION. The default is True.
        gps_path : str
            gait analysis score.

        Returns
        -------
        all_data : TYPE
            DESCRIPTION.

        """
        # file info
        measurements = ["pctSimpleAppuie", "distFoulee", "vitCadencePasParMinute"]
        joint_names = ["Hip", "Knee", "Ankle"]
        side = ["Right", "Left"]

        # Extracting variables from .mat files
        kin_var = mat_to_df.MinMax_feature_extractor(
            directory=data_dir,
            output_dir=output_dir,
            measurements=measurements,
            separate_legs=separate_legs,
            joint_names=joint_names,
        )

        # Calculating ROM from the function above
        kin_var = self.calculate_ROM(kin_var)

        # ----- fixing the values of cadence -----
        kin_var["vitCadencePasParMinute"] *= 2

        # ----- Add GPS to the features -----
        #       Gait Profile Score
        gps = self.load_gps(gps_path)
        all_data = pd.concat((kin_var, gps), axis=1)

        # ----- Add participants demographic variables -----
        demo_var = self.load_demo(demographic_path)

        all_data = pd.concat((all_data, demo_var), axis=1)
        # TODO: concat in one funct

        return all_data

# TODO: enable the use of MCID calculation without all the data
    @staticmethod
    def calculate_MCID(all_data: DataFrame, variable: str) -> list:
        if variable == "VIT":
            delta_VIT = all_data[variable + "_POST"] - all_data[variable + "_PRE"]
            MCID_VIT = []
            for i in delta_VIT:
                if i >= 0.1:
                    MCID_VIT.append(1)
                else:
                    MCID_VIT.append(0)
            return pd.Series(MCID_VIT, index=all_data.index)

        elif variable == "6MWT":
            GMFCS_MCID = {1: range(4, 29), 2: range(4, 29), 3: range(9, 20), 4: range(10, 28)}
            delta_6MWT = all_data[variable + "_POST"] - all_data[variable + "_PRE"]
            MCID_6MWT = []
            for i in range(len(delta_6MWT)):
                if delta_6MWT.iloc[i] >= max(GMFCS_MCID[all_data["GMFCS"].iloc[i]]):
                    MCID_6MWT.append(1)
                else:
                    MCID_6MWT.append(0)
            return pd.Series(MCID_6MWT, index=all_data.index)
        else:
            raise ValueError("Variable not recognized. Use 'VIT', or '6MWT'.")
