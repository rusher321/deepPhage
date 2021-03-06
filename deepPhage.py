#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
File: deepPhage.py
Created Date:  2021/3/6
Author: Huahui Ren <renhuahui@genomics.cn>
--------------
'''

# load package 

import os
import gzip
import math
import torch
import argparse


# Get the directory of the program
scirpt_path = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    args = argparse.ArgumentParser(
        description='phage sequence detector', formatter_class=RawTextHelpFormatter)
    args.add_argument('-c', '--config', default=None, type=str,
                      help='Path of config file')
    args.add_argument('-d', '--deviceid', default=None, type=str,
                      help='Indices of GPUs to enable. Quotated comma-separated device ID numbers. (default: all)')
    args.add_argument('-l', '--len', type=int, required=True,
                      help='Sequencing read length, should be not smaller than 50.')
    args.add_argument('-i', '--input', default=None, type=str, nargs='*', required=True,
                      help='Path of input sequence files (fasta and fastq), the second file will be considered as second end if two files given.')
    args.add_argument('-r', '--rrna', default=None, type=str, nargs='*',
                      help='Path of the output sequence file of detected phage (same number of files as input)')
    args.add_argument('-e', '--ensure', default="none", type=str, choices=['rrna', 'norrna', 'both', 'none'],
                      help='''Only output certain sequences with high confidence
                        nophage: output non-phage with high confidence, remove as many phage as possible;
                        phage: vice versa, output phage with high confidence;
                        both: both non-phage and phage prediction with high confidence;
                        none: give label based on the mean probability of read pair.
                        (Only applicable for paired end reads, discard the read pair when their predicitons are discordant)''')
    args.add_argument('-t', '--threads', default=10, type=int,
                      help='number of threads to use. (default: 10)')
    args.add_argument('-m', '--memory', default=32, type=int,
                      help='amount (GB) of GPU RAM. (default: 32)')
    args.add_argument('--chunk_size', default=None, type=int,
                      help='Use this parameter when having low memory. Parsing the file in chunks.\n{}.\n{}.'.format(
                          'Not needed when free RAM >=5 * your_file_size (uncompressed, sum of paired ends)',
                          'When chunk_size=256, memory=16 it will load 256 * 16 * 1024 reads each chunk (use ~20 GB for 100bp paired end)'
                      ))
    args.add_argument('-v', '--version', action='version',
                      version='%(prog)s {version}'.format(version=__version__))

    if not isinstance(args, tuple):
        args = args.parse_args()
    if args.config is None:
        config_file = os.path.join(scirpt_path, 'config.json')
    else:
        config_file = args.config
    config = ConfigParser.from_json(config_file)
    seq_pred = Predictor(config, args)
    seq_pred.load_model()
    seq_pred.detect()