import pandas as pd
import numpy as np
import scipy.stats

class DataPreprocessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, encoding='cp949', index_col=0)

    def encode_working(self):
        self.data['working_encoded'] = self.data['working'].map({'가동': 1, '정지': 0})

    def preprocess_datetime(self):
        self.data['datetime'] = pd.to_datetime(self.data['date'] + ' ' + self.data['time'])
        self.data['datetime_int'] = self.data['datetime'].astype(np.int64) // 10**9

    def remove_unnecessary_columns(self):
        columns_to_drop = ['count', 'EMS_operation_time', 'mold_code']
        self.data = self.data.drop(columns_to_drop, axis=1, errors='ignore')
        self.data = self.data.select_dtypes(include=[np.number])

    def remove_missing_values(self):
        if 'molten_volume' in self.data.columns:
            self.data.drop('molten_volume', axis=1, inplace=True)
        self.data.dropna(axis=0, inplace=True)
        self.data.reset_index(drop=True, inplace=True)

    def remove_outliers(self, percentile_range=(0.1, 99.9)):
        for col in self.data.select_dtypes(include=np.number).columns:
            if col != 'passorfail':
                lower = np.percentile(self.data[col], percentile_range[0])
                upper = np.percentile(self.data[col], percentile_range[1])
                self.data = self.data[(self.data[col] >= lower) & (self.data[col] <= upper)]
        self.data.reset_index(drop=True, inplace=True)

    def perform_feature_selection(self):
        t_test_results = []
        for col in self.data.columns:
            if col != 'passorfail':
                t_stat, p_value = scipy.stats.ttest_ind(
                    self.data[self.data['passorfail'] == 1][col],
                    self.data[self.data['passorfail'] == 0][col],
                    equal_var=False
                )
                if p_value < 0.05:
                    t_test_results.append(col)
        t_test_results.append('passorfail')
        self.data = self.data[t_test_results]

    def get_processed_data(self):
        return self.data

# Usage
# preprocessor = DataPreprocessor('./data/경진대회용 주조 공정최적화 데이터셋.csv')
# preprocessor.encode_working()
# preprocessor.preprocess_datetime()
# preprocessor.remove_unnecessary_columns()
# preprocessor.remove_missing_values()
# preprocessor.remove_outliers()
# preprocessor.perform_feature_selection()
# data = preprocessor.get_processed_data()
