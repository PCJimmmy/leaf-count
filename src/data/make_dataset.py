# -*- coding: utf-8 -*-
import logging
import os
from collections import OrderedDict
from shutil import rmtree
import importlib


'''
DATA_CATEGORIES_TO_SCRIPTS_DICT:
Each key should map to one of the data categories, which are the subdirectories of the project 
directory under the opo-data-science s3 bucket. For scripts, do not include file extension.
Default args can be changed here as well.

e.g.
DATA_CATEGORIES_TO_SCRIPTS_DICT = OrderedDict([
    ('data_category_1', [(script_1, {default_arg_1 : 1, default_arg_2 : 2, ...]), (script_2, {...})]),
    ('data_category_2', [(script_3, {...}), (script_4, {...}), (script_5, {...})]),
    ('data_category_3', [(script_6, {...}), (script_7, {...})])
])
'''
DATA_CATEGORIES_TO_SCRIPTS_DICT = OrderedDict([
    ('images', [('extract_plant_images', {}),
                ('remove_selected_images', {}),
                ('resize_images', {'target_image_dimensions':[256, 256]}),
                ('augment_images', {})]),
    ('labels', [('augment_labels', {})])
])

DIR_NAME = os.path.dirname(__file__)
RAW_DIR = os.path.join(DIR_NAME, '../../data/raw')
INTERIM_DIR= os.path.join(DIR_NAME, '../../data/interim')
PROCESSED_DIR = os.path.join(DIR_NAME, '../../data/processed')
EXTERNAL_DIR = os.path.join(DIR_NAME, '../../data/external')


def main():
    """ 
    Runs data processing scripts to turn raw data from (../raw) into 
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data\n')

    for data_category in DATA_CATEGORIES_TO_SCRIPTS_DICT.keys():
        data_category_dir = INTERIM_DIR + '/' + data_category 
        try:
            os.mkdir(data_category_dir)
        except:
            rmtree(data_category_dir)
            os.mkdir(data_category_dir)

        # Data category is in either data/raw or data/external
        if data_category not in os.listdir(RAW_DIR):
            if data_category in os.listdir(EXTERNAL_DIR):
                script_input_path = EXTERNAL_DIR + '/' + data_category
            else: 
                print('Data category {} not found!'.format(data_category))
                break
        else: 
            script_input_path = RAW_DIR + '/' + data_category

        # Run scripts on specified data category
        for i, (script, args_dict) in enumerate(DATA_CATEGORIES_TO_SCRIPTS_DICT[data_category]):
            module = importlib.import_module('scripts.{}'.format(script))
            if i != len(DATA_CATEGORIES_TO_SCRIPTS_DICT[data_category]) - 1: 
                script_output_path = data_category_dir + '/' + script
            else: # last script requires special output filepath
                final_data_category_dir = PROCESSED_DIR + '/' + data_category
                script_output_path = final_data_category_dir
            try:
                os.mkdir(script_output_path)
            except:
                rmtree(script_output_path)
                os.mkdir(script_output_path)

            args_dict['input_path'] = script_input_path
            args_dict['output_path'] = script_output_path
            _ = getattr(module, 'main')(**args_dict)

            script_input_path = script_output_path  # recycle filepaths to make a pipeline


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()

