# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 05:04:19 2022

@author: Jerry
"""

"""Code was inspired by tbmoon's code from his 'facenet' repository
    https://github.com/tbmoon/facenet/blob/master/datasets/write_csv_for_making_dataset.ipynb

    The code was modified to run much faster since 'dataframe.append()' creates a new dataframe per each iteration
    which significantly slows performance.
"""

import os
import glob
import pandas as pd
import time
import argparse
import random
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generating csv file for triplet loss!")
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the dataset folder to generate a csv file containing the paths of the training images for triplet loss training."
                    )
parser.add_argument('--csv_name', type=str,
                    help="Required name of the csv file to be generated. (default: 'glint360k.csv')"
                    )
args = parser.parse_args()
dataroot = args.dataroot
csv_name = args.csv_name


def generate_csv_file(dataroot, csv_name="glint360k.csv"):
    """Generates a csv file containing the image paths of the glint360k dataset for use in triplet selection in
    triplet loss training.

    Args:
        dataroot (str): absolute path to the training dataset.
        csv_name (str): name of the resulting csv file.
    """
    print("\nLoading image paths ...")
    files = glob.glob(dataroot + "/*/*") # dataroot = r'D:\NCKU\Course\Data_Science\part3\HW4\data\lfw'

    start_time = time.time()
    list_rows = []

    print("Number of files: {}".format(len(files)))
    print("\nGenerating csv file ...")

    progress_bar = enumerate(tqdm(files))

    for file_index, file in progress_bar:

        face_id = os.path.basename(file).split('.')[0]
        face_label = os.path.basename(os.path.dirname(file))

        # Better alternative than dataframe.append()
        row = {'id': face_id, 'name': face_label}
        list_rows.append(row)
    
    dataframe = pd.DataFrame(list_rows)
    dataframe = dataframe.sort_values(by=['name', 'id']).reset_index(drop=True)

    # Encode names as categorical classes
    dataframe['class'] = pd.factorize(dataframe['name'])[0]
    # dataframe.to_csv(path_or_buf=csv_name, index=False)

    # random.seed(4019)
    # test_number = int(len(dataframe)*0.2)
    # test_index = random.sample(list(dataframe.index), test_number)
    # test_dataframe = dataframe.copy().iloc[test_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    # save_name = 'test_' + csv_name
    # test_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    # train_index = list(set(list(dataframe.index)) ^ set(test_index))
    # train_100_dataframe = dataframe.copy().iloc[train_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    # save_name = 'train100_' + csv_name
    # train_100_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    # random.seed(4020)
    # sample_number = int(len(train_100_dataframe)*0.5)
    # sample_index = random.sample(list(train_100_dataframe.index), sample_number)
    # train_50_dataframe = train_100_dataframe.copy().iloc[sample_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    # save_name = 'train50_' + csv_name
    # train_50_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    # random.seed(4021)
    # sample_number = int(len(train_100_dataframe)*0.25)
    # sample_index = random.sample(list(train_100_dataframe.index), sample_number)
    # train_25_dataframe = train_100_dataframe.copy().iloc[sample_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    # save_name = 'train25_' + csv_name
    # train_25_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    # random.seed(4022)
    # sample_number = int(len(train_100_dataframe)*0.10)
    # sample_index = random.sample(list(train_100_dataframe.index), sample_number)
    # train_10_dataframe = train_100_dataframe.copy().iloc[sample_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    # save_name = 'train10_' + csv_name
    # train_10_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    train_index = list(dataframe.index)
    train_100_dataframe = dataframe.copy().iloc[train_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    save_name = 'train100_' + csv_name
    train_100_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    random.seed(4020)
    sample_number = int(len(train_100_dataframe)*0.5)
    sample_index = random.sample(list(train_100_dataframe.index), sample_number)
    train_50_dataframe = train_100_dataframe.copy().iloc[sample_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    save_name = 'train50_' + csv_name
    train_50_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    random.seed(4021)
    sample_number = int(len(train_100_dataframe)*0.25)
    sample_index = random.sample(list(train_100_dataframe.index), sample_number)
    train_25_dataframe = train_100_dataframe.copy().iloc[sample_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    save_name = 'train25_' + csv_name
    train_25_dataframe.to_csv(path_or_buf=save_name, index=False)
    
    random.seed(4022)
    sample_number = int(len(train_100_dataframe)*0.10)
    sample_index = random.sample(list(train_100_dataframe.index), sample_number)
    train_10_dataframe = train_100_dataframe.copy().iloc[sample_index, :].sort_values(by=['name', 'id']).reset_index(drop=True)
    save_name = 'train10_' + csv_name
    train_10_dataframe.to_csv(path_or_buf=save_name, index=False)
    

    elapsed_time = time.time()-start_time
    print("\nDone! Elapsed time: {:.2f} minutes.".format(elapsed_time/60))


if __name__ == '__main__':
    generate_csv_file(dataroot=dataroot, csv_name=csv_name)
