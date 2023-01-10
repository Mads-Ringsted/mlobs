# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import torch
import numpy as np

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # exchange with the corrupted mnist dataset
    xtrain = np.load(f'{input_filepath}/train_0.npz')['images']
    ytrain = np.load(f'{input_filepath}/train_0.npz')['labels']

    for i in range(1,5): 
        xtrain = np.concatenate((xtrain, np.load(f'{input_filepath}/train_{i}.npz')['images']), axis = 0)
        ytrain = np.concatenate((ytrain, np.load(f'{input_filepath}/train_{i}.npz')['labels']), axis = 0)
    xtest = np.load(f'{input_filepath}/test.npz')['images']
    ytest = np.load(f'{input_filepath}/test.npz')['labels']

    trainset = torch.utils.data.TensorDataset(torch.from_numpy(xtrain).float(), torch.from_numpy(ytrain).long())
    testset = torch.utils.data.TensorDataset(torch.from_numpy(xtest).float(), torch.from_numpy(ytest).long())

    torch.save(trainset, f'{output_filepath}/trainset.pt')
    torch.save(testset, f'{output_filepath}/testset.pt')





if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
