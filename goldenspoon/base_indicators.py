import pandas as pd
import re
import datetime
import numpy as np
from dateutil import parser

class Indicator:
  def __init__(self, db, start_date=None, end_date=None):
    self.db = db
    self.start_date = start_date if start_date is None else parser.parse(start_date).date()
    self.end_date   = end_date   if end_date   is None else parser.parse(end_date).date()

  k_column_date = '日期'

  k_stock_column_name           = '股票名称'
  k_stock_column_code           = '股票代码'
  k_stock_column_industry       = '所属中证行业(2016) [行业类别]{level}级'
  k_stock_column_type           = '股票风格分类'
  k_stock_column_total_value    = '区间日均总市值 [起始交易日期]截止日3月前 [单位]亿元'
  k_stock_column_avg_price      = '当月成交均价 [复权方式]不复权'
  k_stock_column_close_price    = '月收盘价 [复权方式]不复权'
  k_stock_column_turnover_rate  = '月换手率 [单位]%'
  k_stock_column_amplitutde     = '月振幅 [单位]%'
  k_stock_column_margin_diff    = '融资融券差额 [单位]元'
  k_stock_column_share_ratio_of_funds   = '基金持股比例 [单位]% [比例类型]占流通股比例'
  k_stock_column_share_number_of_funds  = '基金持股数量 [单位]股'
  k_stock_column_num_of_funds           = '持股基金家数 [股本类型]流通股本'

  k_stock_exclude_timed_columns = [
    '市盈率(PE,TTM)',
    '所属概念板块',
    '流通股本 [单位]股',
    '股票规模指标',
    '股票风格分类',
    '自由流通市值 [单位]元',
  ]

  def get_stocks(self):
    '''
    return: pd.DataFrame with columns: [股票名称, 股票代码]
    '''
    return self.db.get_stocks()[[self.k_stock_column_name, self.k_stock_column_code]].dropna()

  def get_stocks_timed(self):
    result = self.db.get_stock_stats('date')

    columns = [c for c in result.columns if c not in self.k_stock_exclude_timed_columns]
    filtered = result[columns]

    if self.start_date:
      filtered = filtered[filtered[self.k_column_date] >= self.start_date]
    if self.end_date:
      filtered = filtered[filtered[self.k_column_date] <= self.end_date]

    return pd.concat([filtered,
                      result[[self.k_column_date, self.k_stock_column_code] + self.k_stock_exclude_timed_columns],
                    ], ignore_index=True, sort=False)

  def get_stocks_map(self):
    '''
    return: dict: 股票代码 -> 股票名称
    '''
    return self.db.stock_map

  def get_stocks_name_map(self):
    '''
    return: dict: 股票名称 -> 股票代码
    '''
    return self.db.stock_map_reverse

  def add_stock_name(self, df):
    if self.k_stock_column_code in df:
      df[self.k_stock_column_name] = df[self.k_stock_column_code].map(self.get_stocks_map())
    elif self.k_stock_column_name in df:
      df[self.k_stock_column_code] = df[self.k_stock_column_name].map(self.get_stocks_name_map())
    else:
      assert 0, 'invalid df columns: {}'.format(df.columns.tolist())

    columns =  [self.k_column_date, self.k_stock_column_code, self.k_stock_column_name]
    columns += [col for col in df.columns if col not in columns]
    return df[[col for col in columns if col in df.columns]]

  def get_stock_general(self):
    '''
    return: pd.DataFrame with columns: [股票名称, 股票代码, 所属行业(四级), 类型]
    '''
    untimed_columns =  [self.k_stock_column_name, self.k_stock_column_code]
    untimed_columns += [self.k_stock_column_industry.format(level=i) for i in [1,2,3,4]]
    timed_columns   =  [self.k_column_date,
                        self.k_stock_column_code,
                        self.k_stock_column_type]

    stocks_untimed = self.db.get_stocks()[untimed_columns]
    stocks_timed   = self.get_stocks_timed()[timed_columns].dropna() \
                      .sort_values(self.k_column_date, ascending=False) \
                      .groupby([self.k_column_date, self.k_stock_column_code]) \
                      .last()
    return pd.merge(stocks_untimed, stocks_timed, on=self.k_stock_column_code).dropna()

  def get_stock_performance(self):
    columns = [self.k_column_date,
               self.k_stock_column_code,
               self.k_stock_column_total_value,
               self.k_stock_column_close_price,
               self.k_stock_column_avg_price,
               self.k_stock_column_turnover_rate,
               self.k_stock_column_amplitutde,
               self.k_stock_column_margin_diff,
               ]
    stocks_timed = self.get_stocks_timed()[columns].dropna()
    return self.add_stock_name(stocks_timed).dropna()

  def get_stock_holding_funds_share(self):
    columns = [self.k_column_date,
               self.k_stock_column_code,
               self.k_stock_column_share_ratio_of_funds,
               self.k_stock_column_share_number_of_funds,
               self.k_stock_column_num_of_funds,
               ]
    return self.add_stock_name(self.get_stocks_timed()[columns].dropna()).dropna()

  def get_stock_holding_funds_number(self):
    columns = [self.k_column_date,
               self.k_stock_column_code,
               self.k_stock_column_num_of_funds,
               ]

    return self.add_stock_name(self.db.get_stock_stats('date')[columns].dropna()).dropna()

  def get_stock_topn_holding_funds(self):
    fund_topn_stocks = self.db.get_fund_stats('topn_stocks')
    fund_topn_stocks_share_ratio = fund_topn_stocks[fund_topn_stocks['indicator'] == '重仓股持仓占流通股比例'] \
                                    .groupby([self.k_column_date, self.k_stock_column_name])[self.k_fund_column_code] \
                                    .apply(set) \
                                    .reset_index()
    return self.add_stock_name(fund_topn_stocks_share_ratio.dropna()).dropna()


  k_fund_column_name  = '基金名称'
  k_fund_column_code  = '基金代码'
  k_fund_column_owner = '基金管理人简称'
  k_fund_column_type  = '投资类型(二级分类)'
  k_fund_column_total_value = '基金资产总值 [单位]元'
  k_fund_column_net_value   = '基金资产净值 [单位]元'
  k_fund_column_net_value_inc_rate  = '报告期净值增长率 [报告期净值数据项]过去3个月 [单位]%'
  #k_fund_column_income      = ''
  k_fund_column_manager             = '基金经理'
  k_fund_column_manager_longest     = '任职期限最长的现任基金经理 [名次]第1名'
  k_fund_column_manager_history     = '基金经理(历任)'

  k_fund_exclude_timed_columns = []

  def get_funds(self):
    return self.db.get_funds()[[self.k_fund_column_name, self.k_fund_column_code]].dropna()

  def get_funds_timed(self):
    result = self.db.get_fund_stats('date')

    columns = [c for c in result.columns if c not in self.k_fund_exclude_timed_columns]
    filtered = result[columns]

    if self.start_date:
      filtered = filtered[filtered[self.k_column_date] >= self.start_date]
    if self.end_date:
      filtered = filtered[filtered[self.k_column_date] <= self.end_date]

    return pd.concat([filtered,
                      result[[self.k_column_date, self.k_fund_column_code] + self.k_fund_exclude_timed_columns],
                    ], ignore_index=True, sort=False)

  def get_funds_map(self):
    return self.db.fund_map

  def get_funds_name_map(self):
    return self.db.fund_map_reverse

  def get_funds_general(self):
    untimed_columns = [self.k_fund_column_name,
                       self.k_fund_column_code,
                       self.k_fund_column_owner,
                       self.k_fund_column_type,
                       ]
    timed_columns   = [self.k_column_date,
                       self.k_fund_column_code,
                       self.k_fund_column_total_value,
                       ]

    funds_untimed = self.db.get_funds()[untimed_columns]
    funds_timed   = self.get_funds_timed()[timed_columns].dropna() \
                      .sort_values(self.k_column_date, ascending=False) \
                      .groupby([self.k_column_date, self.k_fund_column_code]) \
                      .last().reset_index()

    return pd.merge(funds_untimed, funds_timed, on=self.k_fund_column_code).dropna()

  def get_funds_manager(self):
    columns = [self.k_fund_column_code,
               self.k_fund_column_manager_history,
               ]
    re_manager = re.compile(r'(.+)\((\d.*)\)')
    date_fmt   = '%Y%m%d'

    result = []
    for fund_code, fund_managers in self.db.get_funds()[columns].values.tolist():
      for manager_str in fund_managers.split(','):
        m = re_manager.match(manager_str)
        assert m, 'invalid manager: {}'.format(manager_str)
        manager_name, manager_time = m.groups()

        if '-' in manager_time:
          start_time, end_time = manager_time.split('-')
          start_time = datetime.datetime.strptime(start_time, date_fmt)
          end_time   = datetime.datetime.strptime(end_time, date_fmt)
          current    = False
        elif '至今' in manager_time:
          start_time = datetime.datetime.strptime(manager_time[:8], date_fmt)
          end_time   = datetime.datetime.today()
          current    = True
        else:
          assert 0, 'invalid time {}'.format(manager_time)

        result.append({
            self.k_fund_column_code: fund_code,
            self.k_fund_column_manager: manager_name,
            'start': start_time,
            'end': end_time,
            'current': current,
        })

    return pd.DataFrame(result)

  def get_funds_manager_stats(self, only_current=True):
    df = self.get_funds_manager()

    if only_current:
      df = df[df['current'] == True]

    stats = df \
            .groupby(self.k_fund_column_manager)[self.k_fund_column_code] \
            .apply(set).reset_index()
    stats['基金名称'] = stats[self.k_fund_column_code].apply(lambda v: set([self.db.get_fund_by_code(vv) for vv in v]))
    stats['基金数量'] = stats[self.k_fund_column_code].apply(lambda v: len(v))

    return stats

  def get_funds_topn_stocks(self):
    fund_topn_stocks = self.db.get_fund_stats('topn_stocks')
    if self.start_date:
      fund_topn_stocks = fund_topn_stocks[fund_topn_stocks[self.k_column_date] >= self.start_date]
    if self.end_date:
      fund_topn_stocks = fund_topn_stocks[fund_topn_stocks[self.k_column_date] <= self.end_date]

    key_columns = [self.k_column_date,
                   self.k_fund_column_code,
                   self.k_stock_column_name,
                   'indicator']
    fund_topn_stocks = fund_topn_stocks.groupby(key_columns).sum().reset_index()
    return fund_topn_stocks.pivot(index=[self.k_column_date,
                                         self.k_fund_column_code,
                                         self.k_stock_column_name],
                                    columns=['indicator'],
                                    values=['value'])
