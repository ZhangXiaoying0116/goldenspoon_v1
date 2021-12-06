import pandas as pd
import numpy as np
from collections import defaultdict

class GenericIndexBase:
    def __init__(self):
        self.result = defaultdict(dict)

    def get_indexed_data(self):
        rows = []
        for k, v in self.result.items():
            if isinstance(v, dict):
                for key_name, key_value in zip(self.key_names, k):
                    v[key_name] = key_value
                rows.append(v)
            elif isinstance(v, (tuple, list)):
                for vv in v:
                    if not v: continue
                    for key_name, key_value in zip(self.key_names, k):
                        vv[key_name] = key_value
                    rows.append(vv)
            else:
                assert 0, 'unsupported type: {}'.format(type(v))

        if not rows:
            return None
        return pd.DataFrame(rows)

# generic indexer for columns without special metadata
class GenericNameIndex(GenericIndexBase):
    name = 'generic'
    key_names = ['证券代码', '证券名称']

    def run(self, row, col, metadata):
        if pd.isna(row[col]):
            return False

        indexed = self.result
        key = (row['证券代码'], row['证券名称'])

        if col in indexed[key] and not np.isclose(indexed[key][col], row[col]):
            print('WARN: column "{}" value already assigned: key: {}, value: {} vs {}'.format(
                col, key, indexed[key][col], row[col]))
        indexed[key][col] = row[col]
        return True

# generic indexer for columns with date
class GenericDateIndex(GenericIndexBase):
    name = 'date'
    key_names = ['证券代码', '日期']

    def run(self, row, col, metadata):
        if 'date' not in metadata:
            return False
        if pd.isna(row[col]):
            return False

        indexed = self.result
        key = (row['证券代码'], metadata['date'])

        name = ' '.join(
            [metadata['name']] +
            ['[%s]%s' % (k, v) for k, v in metadata.items() if k not in ('name', 'date')])

        if name in indexed[key] and not np.isclose(indexed[key][name], row[col]):
            print('WARN: column "{}" value already assigned: key: {}, value: {} vs {}'.format(
                name , key, indexed[key][name], row[col]))
        indexed[key][name] = row[col]
        return True

# indexer for funds topN stock holding stats
class TopNStocksIndex(GenericIndexBase):
    name = 'topn_stocks'
    key_names = ['基金代码', '日期']

    def run(self, row, col, metadata):
        k_columns = ('重仓股股票市值', '重仓股持仓占流通股比例', '前十大重仓股名称')
        if metadata['name'] not in k_columns:
            return False
        if pd.isna(row[col]):
            return False

        assert 'date' in metadata

        indexed = self.result
        key = (row['证券代码'], metadata['date'])
        if pd.isna(key[0]):
            return False

        if key not in indexed:
            indexed[key] = defaultdict(dict)

        if metadata['name'] == k_columns[-1]:
            stock_names = row[col].split(',')
            for i, name in enumerate(stock_names):
                indexed[key][i]['股票名称'] = name
        else:
            topN = metadata['topN']-1
            assert topN >= 0
            indexed[key][topN][metadata['name']] = row[col]
        return True

    def get_indexed_data(self):
        result = {}
        for k, v in self.result.items():
            indicators = []
            for data in v.values():
                assert '股票名称' in data, 'invalid data: {}'.format(v)
                for indicator, value in data.items():
                    if indicator == '股票名称':
                        continue
                    indicators.append({
                        '股票名称': data['股票名称'],
                        'indicator': indicator,
                        'value': value,
                        })
            result[k] = indicators
        self.result = result
        return super().get_indexed_data()

