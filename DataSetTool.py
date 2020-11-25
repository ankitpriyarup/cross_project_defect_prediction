import numpy as np
import os
import imblearn.over_sampling as over_sampling


class DataSetTool:
    # Metric compensation for version 08
    # Mij in Target = (Mij in Target * Mean(Mj in Source)) / Mean(Mj) in  Target
    @staticmethod
    def metric_compensation(source, target):
        # Iterate through each metric attribute
        for j in range(target.shape[1]):
            # Calculate the mean of each metric attribute
            metric_mean_source = np.mean(source[:, j])
            metric_mean_target = np.mean(target[:, j])
            # Iterate through each example
            for i in range(target.shape[0]):
                target[i, j] = (target[i, j] * metric_mean_source) / metric_mean_target
        return target

    # Metric compensation adjusted for version 17
    # Mij in Source = (Mij in Source * Mean(Mj in Target)) / Mean(Mj) in Source
    @staticmethod
    def metric_compensation_adopt(source, target):
        # Iterate through each metric attribute
        for j in range(source.shape[1]):
            # Calculate the mean of each metric attribute
            metric_mean_source = np.mean(source[:, j])
            metric_mean_target = np.mean(target[:, j])
            # Iterate through each example
            for i in range(source.shape[0]):
                source[i, j] = (source[i, j] * metric_mean_target) / metric_mean_source
        return source

    # Read all files in the folder and return the processed data set
    # metrics_num: Number of measures (the number of columns in the original data excluding the label column)
    # is_sample: Whether to resample
    # is_normalized: Whether the data is normalized
    @staticmethod
    def init_data(folder_path, metrics_num, is_sample=False, is_normalized=True):
        # Get all original files in the directory
        files = os.listdir(folder_path)
        data_list, label_list = [], []
        for file in files:
            # The real path of each subfile
            file_path = folder_path+file
            # Read file directly
            data_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=range(0, metrics_num+1))
            label_file = np.loadtxt(file_path, dtype=float, delimiter=',', usecols=metrics_num+1)
            if is_normalized:
                # Data normalization
                data_file -= data_file.min()
                data_file /= data_file.max()
                label_file -= label_file.min()
                label_file /= label_file.max()
            # Add to list
            data_list.append(data_file)
            label_list.append(label_file)
        # Resampling
        if is_sample:
            for index in range(len(data_list)):
                data_list[index], label_list[index] = over_sampling.SMOTE(kind='regular').fit_sample(data_list[index],
                                                                                                     label_list[index])
        return data_list, label_list
