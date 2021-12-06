import os
import re
import glob
import numpy as np
import pandas as pd
import datetime
from . import utils
from .index import GenericDateIndex, GenericNameIndex, TopNStocksIndex


class Database:
    k_key_columns = (
        '证券代码',
        '证券名称',
        )

    def __init__(self, path):
        self.k_cached = os.path.join(os.getcwd(), 'cached')
        self.k_path   = path
        self.raw_data = {}
        os.makedirs(self.k_cached, exist_ok=True)

        self.index_data()

    def get_funds(self):
        return self.fund_stats['generic']

    def get_stocks(self):
        return self.stock_stats['generic']

    def get_fund_stats(self, name):
        return self.fund_stats[name]

    def get_stock_stats(self, name):
        return self.stock_stats[name]

    def get_fund_by_name(self, name):
        return self.fund_map_reverse[name]

    def get_fund_by_code(self, code):
        return self.fund_map[code]

    def get_stock_by_name(self, name):
        return self.stock_map_reverse[name]

    def get_stock_by_code(self, code):
        return self.stock_map[code]

    @staticmethod
    def compute_column_metadata(col):
        re_report_timestamp = re.compile('.报告期.(\d{4})年(.*)')
        re_trans_timestamp  = re.compile('.*日期..*(\d{4})-(\d{2})-(\d{2})')
        re_topn_name        = re.compile('.名次.*第(\d+)名')
        re_meta_name        = re.compile('\[([^]]+)\](\S+)')

        k_report_mapping = {
                '一季':         (3, 31),
                '二季/中报':     (6, 30),
                '三季':         (9, 30),
                '年报':         (12, 31),
                }

        tokens   = []
        metadata = {}
        for tok in col.split():
            m = re_report_timestamp.match(tok)
            if m:
                year, rpt = m.groups()
                month, day = k_report_mapping[rpt]
                assert 'date' not in metadata
                metadata['date'] = datetime.date(int(year), month, day)
                continue

            m = re_trans_timestamp.match(tok)
            if m:
                assert 'date' not in metadata
                year, month, day = m.groups()
                metadata['date'] = datetime.date(int(year), int(month), int(day))
                continue

            m = re_topn_name.match(tok)
            if m:
                topn = int(m.groups()[0])
                metadata['topN'] = topn
                continue

            m = re_meta_name.match(tok)
            if m:
                key, value = m.groups()
                metadata[key] = value
                continue

            tokens.append(tok)

        metadata['name'] = '_'.join(tokens)
        return metadata

    def load_files(self):
        if self.raw_data:
            return

        self.raw_data = {}
        for filename in glob.glob(os.path.join(self.k_path, '*.xls*'), recursive=True):
            print('loading {}'.format(filename))
            df = pd.read_excel(filename)
            df = df.replace('——', np.nan)
            df.columns = [' '.join(col.split()) for col in df.columns]
            self.raw_data[os.path.basename(filename)] = df

    def index_data(self):
        self.fund_stats = utils.pickle_cache(os.path.join(self.k_cached, 'indexed_fund_stats.pkl'),
                lambda : self.index_by_column_metadata('funds'))
        self.stock_stats = utils.pickle_cache(os.path.join(self.k_cached, 'indexed_stock_stats.pkl'),
                lambda : self.index_by_column_metadata('stocks'))

        self.index_basic_info()

        for df in self.fund_stats.values():
            df.rename(columns = {'证券代码': '基金代码', '证券名称': '基金名称'}, inplace=True)
        for df in self.stock_stats.values():
            df.rename(columns = {'证券代码': '股票代码', '证券名称': '股票名称'}, inplace=True)

    def index_by_column_metadata(self, prefix):
        self.load_files()
        dfs = [df for name, df in self.raw_data.items() if name.startswith(prefix)]

        indexers = [
            TopNStocksIndex(),
            GenericDateIndex(),
            GenericNameIndex(),
        ]

        for df in dfs:
            self.index_by_column_metadata_impl(indexers, df)

        result = {}
        for indexer in indexers:
            data = indexer.get_indexed_data()
            if data is None:
                continue

            cols =  list(self.k_key_columns)
            cols += list(sorted(set(data.columns.tolist()) - set(cols)))
            cols =  [col for col in cols if col in data.columns]

            result[indexer.name] = data[cols]

        return result

    def index_by_column_metadata_impl(self, indexers, df):
        columns_metadata = {}
        for col in df.columns:
            if col in self.k_key_columns:
                continue
            columns_metadata[col] = self.compute_column_metadata(col)

        for _, row in df.iterrows():
            if any(pd.isna(row[col]) for col in self.k_key_columns):
                continue

            for col, metadata in columns_metadata.items():
                for indexer in indexers:
                    if indexer.run(row, col, metadata):
                        break

    def index_basic_info(self):
        assert self.fund_stats and self.stock_stats

        # code -> name
        key_columns = list(self.k_key_columns)

        # for funds, there may be duplicated names mapping to different codes
        # so fund_map_reverse is shorter than fund_map
        fund_basics = self.get_funds()[key_columns]
        self.fund_map = dict(fund_basics.values.tolist())
        self.fund_map_reverse = {v: k for k, v in self.fund_map.items()}

        stock_basics = self.get_stocks()[key_columns]
        self.stock_map = dict(stock_basics.values.tolist())
        self.stock_map_reverse = {v: k for k, v in self.stock_map.items()}

