import csv
from functools import partial
import os

from flatten_dict import flatten as flatten_dict
import pandas as pd

from style.utils.misc import assert_dir


def list2df(lst, flatten=False, recursive=(), columns=(), include_all_columns=False):
    if flatten:
        lst = [flatten_dict(d, reducer='path') for d in lst]
    df = pd.DataFrame.from_records(lst)
    for col in recursive:
        df[col] = df[col].map(partial(list2df, flatten=flatten))

    if columns:
        columns = list(columns)
        all_columns = list(df.columns)
        if include_all_columns:
            columns += [col for col in all_columns if col not in columns]
        df = df[columns]
    return df


def save_to_csv(path, data=(), fieldnames=None, when_exists='append', **row):
    fieldnames = fieldnames or list(row.keys())
    if when_exists == 'append':
        mode = 'at'
        writer_header = not os.path.isfile(path)
    elif when_exists == 'overwrite':
        mode = 'wt'
        writer_header = True
    else:
        raise Exception(f"Unknown option: {when_exists}")

    assert_dir(path)
    with open(path, mode, encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames)
        if writer_header:
            writer.writeheader()
        if row:
            writer.writerow(row)
        for d in data:
            writer.writerow(d)
