#!/usr/bin/python


import sys
import getopt
import os
import pandas as pd
from collections import defaultdict
import argparse
import glob


def main():
    # Parsing the arguments
    parser = argparse.ArgumentParser(description='Merging Results Files')
    parser.add_argument('input_folder', help='input_folder')
    parser.add_argument('output_file', help='output_file')

    # These are params to be able to merge things appropriately
    parser.add_argument('--topk', default=None, help='When merging, we want to include the top k results for each unique key')
    parser.add_argument('--key_column', default=None, help='column of the key to group by')
    parser.add_argument('--sort_column', default=None, help='column of the to sort by')
    parser.add_argument('--per_library_topk', default=None, help='When set (>0), include the top k results for each unique key PER library (see --library_column) instead of overall. Takes precedence over --topk')
    parser.add_argument('--library_column', default=None, help='column identifying the source library, used with --per_library_topk')

    args = parser.parse_args()

    all_results_files = glob.glob(os.path.join(args.input_folder, "*.tsv"))

    all_results_list = []
    for i, results_file in enumerate(all_results_files):
        temp_df = pd.read_csv(results_file, sep="\t")

        if len(temp_df) > 0:
            all_results_list.append(temp_df)
    
    # merging results
    try:
        all_results_df = pd.concat(all_results_list, ignore_index=True)
    except:
        all_results_df = pd.DataFrame()
        
        # writing out
        all_results_df.to_csv(args.output_file, sep="\t", index=False)

        # exit
        exit(0)

    # Filtering when appropriate
    if args.per_library_topk is not None and int(args.per_library_topk) > 0:
        topk_filter = int(args.per_library_topk)

        all_results_df = all_results_df.sort_values(by=args.sort_column, ascending=False)
        group_cols = [args.key_column, args.library_column] if args.library_column else [args.key_column]
        all_results_df = all_results_df.groupby(group_cols).head(topk_filter)
    elif args.topk is not None:
        topk_filter = int(args.topk)

        all_results_df = all_results_df.sort_values(by=args.sort_column, ascending=False)
        all_results_df = all_results_df.groupby(args.key_column).head(int(args.topk))

    # writing results
    all_results_df.to_csv(args.output_file, sep="\t", index=False)

if __name__ == "__main__":
    main()
