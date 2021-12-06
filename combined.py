#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pandas as pd
import numpy as np

sys.path.append('goldenspoon')

from goldenspoon import Database, Indicator

k_cyclical_industry_keys = ['原材料','汽车与汽车零部件','房地产', '交通运输','能源', '资本市场', '消费者服务', '银行','其他金融', '保险']

def compute_firstindustry_count(ind_stocks_general):
  firstindustry_count = ind_stocks_general['所属中证行业(2016) [行业类别]1级'].value_counts()
  # firstindustry_keys = firstindustry_count.keys() # ['工业','信息技术','可选消费','原材料','医药卫生','金融地产','主要消费','公用事业','电信业务','能源']
  return firstindustry_count

def compute_secondindustry_count(ind_stocks_general):
  secondindustry_count = ind_stocks_general['所属中证行业(2016) [行业类别]2级'].value_counts()
  # secondindustry_keys = secondindustry_count.keys()
  return secondindustry_count


class stocks_dynamic_indicators():
    def __init__(self, ind_stocks_general, ind_stocks_perf, ind_stocks_perf_with_fund, past_quater_number, minimal_fund_number):
        self.past_quater_number = past_quater_number
        self.minimal_fund_number = minimal_fund_number
        self.ind_stocks_general = ind_stocks_general
        self.ind_stocks_perf = ind_stocks_perf
        self.ind_stocks_perf_with_fund = ind_stocks_perf_with_fund

    def q_quater_mean_std(self, perf_metric,df_func):
        stock_id_list = []
        change_mean_list = []
        change_std_list = []

        firstindustry_count = compute_firstindustry_count(self.ind_stocks_general)

        for industry in firstindustry_count.keys():
            industry_stocks_general = self.ind_stocks_general.loc[self.ind_stocks_general['所属中证行业(2016) [行业类别]1级']==industry]
            industry_stock_id = industry_stocks_general['股票代码']

            change = []
            for stock_id in industry_stock_id:
                ind_stocks_perf_temp = (df_func.loc[df_func['股票代码']==stock_id])
                ind_stocks_quater_count = ind_stocks_perf_temp.shape[0]
                if ind_stocks_quater_count>0:
                    sample = ind_stocks_perf_temp[perf_metric]
                    if ind_stocks_quater_count>=self.past_quater_number:
                        baseline = ind_stocks_perf_temp.iloc[0:self.past_quater_number -1].mean()[perf_metric]
                        sample = sample [:self.past_quater_number]
                    else:
                        baseline = ind_stocks_perf_temp.iloc[0:ind_stocks_quater_count-1].mean()[perf_metric]
                        sample = sample [:ind_stocks_quater_count]
                    change = (sample - baseline)/baseline
                    change_mean = change.mean()
                    change_std = change.std()
                else:
                    change_mean = np.nan
                    change_std = np.nan
                stock_id_list.append(stock_id)
                change_mean_list.append(change_mean)
                change_std_list.append(change_std)
        return stock_id_list, change_mean_list, change_std_list

    #mean std function for stock fund affinity
    def q_quater_fund_owner_affinity_mean_std(self, perf_metric):
        stock_id_list = []
        change_mean_list = []
        change_std_list = []

        firstindustry_count = compute_firstindustry_count(self.ind_stocks_general)

        for industry in firstindustry_count.keys():
            industry_stocks_general = self.ind_stocks_general.loc[self.ind_stocks_general['所属中证行业(2016) [行业类别]1级']==industry]
            industry_stock_id = industry_stocks_general['股票代码']

            for stock_id in industry_stock_id:
                ind_stocks_perf_with_fund_temp = (self.ind_stocks_perf_with_fund.loc[self.ind_stocks_perf_with_fund['股票代码']==stock_id])
                ind_stocks_perf_temp = (self.ind_stocks_perf.loc[self.ind_stocks_perf['股票代码']==stock_id]).sort_values(by=['日期'])
                ind_stocks_quater_count = ind_stocks_perf_temp.shape[0]
                date = ind_stocks_perf_temp['日期']
                stock_quater_fund_owner_affinity = []
                if ind_stocks_quater_count>0:
                    if ind_stocks_quater_count>=self.past_quater_number:
                        date = date[-self.past_quater_number:]
                for quater in date:
                    quater_sample = ind_stocks_perf_with_fund_temp.loc[ind_stocks_perf_with_fund_temp['日期']==quater]
                    fund_owner_sample = quater_sample[perf_metric]
                    if fund_owner_sample.shape[0] >=self.minimal_fund_number:
                        fund_owner_sample_unique= fund_owner_sample.drop_duplicates()
                        quater_fund_owner_affinity = 1-fund_owner_sample_unique.shape[0]/fund_owner_sample.shape[0]
                        stock_quater_fund_owner_affinity.append(quater_fund_owner_affinity)
                stock_quater_fund_owner_affinity= np.asarray(stock_quater_fund_owner_affinity)
                change_mean = stock_quater_fund_owner_affinity.mean()
                change_std = stock_quater_fund_owner_affinity.std()

                stock_id_list.append(stock_id)
                change_mean_list.append(change_mean)
                change_std_list.append(change_std)
        return stock_id_list, change_mean_list, change_std_list

    #mean function for stock fund revisit
    def q_quater_fund_revisit_mean(self, perf_metric):
        stock_id_list = []
        change_mean_list = []

        firstindustry_count = compute_firstindustry_count(self.ind_stocks_general)

        for industry in firstindustry_count.keys():
            industry_stocks_general = self.ind_stocks_general.loc[self.ind_stocks_general['所属中证行业(2016) [行业类别]1级']==industry]
            industry_stock_id = industry_stocks_general['股票代码']
            revisit_ratio = []
            for stock_id in industry_stock_id:
                ind_stocks_perf_with_fund_temp = (self.ind_stocks_perf_with_fund.loc[self.ind_stocks_perf_with_fund['股票代码']==stock_id])
                ind_stocks_perf_temp = (self.ind_stocks_perf.loc[self.ind_stocks_perf['股票代码']==stock_id]).sort_values(by=['日期'])
                ind_stocks_quater_count = ind_stocks_perf_temp.shape[0]
                date = ind_stocks_perf_temp['日期']
                stock_quater_fund_revist = []
                stock_quater_fund_revist_unique = []
                if ind_stocks_quater_count>0:
                    if ind_stocks_quater_count>=self.past_quater_number :
                        date = date[-self.past_quater_number :]

                for quater in date:
                    quater_sample = ind_stocks_perf_with_fund_temp.loc[ind_stocks_perf_with_fund_temp['日期']==quater]
                    fund_revisit_sample = quater_sample[perf_metric]
                    if fund_revisit_sample.shape[0] >=self.minimal_fund_number:
                        stock_quater_fund_revist.append(fund_revisit_sample)
                        fund_revisit_sample_unique= fund_revisit_sample.drop_duplicates()
                        stock_quater_fund_revist_unique.append(fund_revisit_sample_unique)
                if len(stock_quater_fund_revist)>0:
                    revisit_ratio = (1-len(stock_quater_fund_revist_unique)/len(stock_quater_fund_revist))
                stock_id_list.append(stock_id)
                change_mean_list.append(revisit_ratio)
        return stock_id_list, change_mean_list

    #function for distinguish cyclical industry
    def stock_cyclical_industry(self, cyclical_industry_keys):
        stock_id_list = []
        cyclical_industry_list = []

        secondindustry_count = compute_secondindustry_count(self.ind_stocks_general)

        for industry in secondindustry_count.keys():
            industry_stocks_general = self.ind_stocks_general.loc[self.ind_stocks_general['所属中证行业(2016) [行业类别]2级']==industry]
            industry_stock_id = industry_stocks_general['股票代码']
            cyclical_insdustry = []
            for stock_id in industry_stock_id:
                if industry in cyclical_industry_keys:
                  cyclical_insdustry = 1
                else:
                  cyclical_insdustry = -1
                stock_id_list.append(stock_id)
                cyclical_industry_list.append(cyclical_insdustry)

        return stock_id_list, cyclical_industry_list

## get func --> core func
## --------------------------------------------------------------------------------------------------------------------------------------------------------------------
class stocks_industry_static_indicators():
    def __init__(self, ind_stocks_general_new, industry_level='所属中证行业(2016) [行业类别]1级'):
        self.industry_level = industry_level
        self.ind_stocks_general_new = ind_stocks_general_new

    def stocks_industry_static_indicators(self, industry_metric='股票风格分类'):
        indicator_types = self.ind_stocks_general_new[industry_metric].drop_duplicates().values.tolist()
        dict_industry_indicator = {}

        firstindustry_count = compute_firstindustry_count(self.ind_stocks_general_new)

        for firstindustry_key in firstindustry_count.keys():
            dict_indicator_percent ={}
            for type in indicator_types:
                firstindustry_indicator_count = self.ind_stocks_general_new[(self.ind_stocks_general_new[self.industry_level] == firstindustry_key) & (self.ind_stocks_general_new[industry_metric] == type)]
                firstindustry_indicator_percent = firstindustry_indicator_count['股票代码'].count() / firstindustry_count[firstindustry_key]
                dict_indicator_percent[type] = float(firstindustry_indicator_percent)
            dict_industry_indicator[firstindustry_key] = dict_indicator_percent

        dict_stock_indicator = {}
        def function(stock_id, stock_industry):
            dict_stock_indicator[stock_id] = dict_industry_indicator[stock_industry]
            return
        self.ind_stocks_general_new.apply(lambda x: function(x['股票代码'], x[self.industry_level]), axis = 1) ## ind_stocks_general_new存在
        return dict_stock_indicator

class stocks_industry_dynamic_indicators():
    def __init__(self, ind_stocks_general, past_quater_number):
        self.past_quater_number = past_quater_number
        self.ind_stocks_general = ind_stocks_general

    def q_quater_mean_std(self, perf_metric, df_func):
        stock_id_list = []
        industry_change_mean_list = []
        industry_change_std_list = []

        firstindustry_count = compute_firstindustry_count(self.ind_stocks_general)

        for industry in firstindustry_count.keys():
            industry_stocks_general = self.ind_stocks_general.loc[self.ind_stocks_general['所属中证行业(2016) [行业类别]1级']==industry]
            industry_stock_id = industry_stocks_general['股票代码']

            industry_stock_id_list = []
            change_mean_list = []
            change_std_list = []
            for stock_id in industry_stock_id:
                ind_stocks_perf_temp = (df_func.loc[df_func['股票代码']==stock_id])
                ind_stocks_quater_count = ind_stocks_perf_temp.shape[0]


                if ind_stocks_quater_count>0:
                    sample = ind_stocks_perf_temp[perf_metric]

                    if ind_stocks_quater_count>=self.past_quater_number:
                        baseline = ind_stocks_perf_temp.iloc[self.past_quater_number-1][perf_metric]
                        sample = sample [:self.past_quater_number]
                    else:
                        baseline = ind_stocks_perf_temp.iloc[ind_stocks_quater_count-1][perf_metric]
                        sample = sample [:ind_stocks_quater_count]

                    change = (sample - baseline)/baseline
                    change_mean = change.mean()
                    change_std = change.std()
                    change_mean_list.append(change_mean)
                    change_std_list.append(change_std)
                industry_stock_id_list.append(stock_id)
            change_mean = np.array(change_mean_list).mean()
            change_std = np.array(change_std_list).mean()
            for i in range(len(industry_stock_id_list)):
                industry_change_mean_list.append(change_mean)
                industry_change_std_list.append(change_std)
            stock_id_list.extend(industry_stock_id_list)

        return stock_id_list, industry_change_mean_list, industry_change_std_list

## main func --> get func
## --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def get_stocks_industry_static_indicators(ind_stocks_general_new):
    print('get_stocks_industry_static_indicators')
    industry_level='所属中证行业(2016) [行业类别]1级'
    stocks_industry_static = stocks_industry_static_indicators(ind_stocks_general_new, industry_level)

    indicator_stock_hushen300 = stocks_industry_static.stocks_industry_static_indicators(industry_metric='是否属于重要指数成份 [所属指数]沪深300指数')
    indicator_stock_styles = stocks_industry_static.stocks_industry_static_indicators(industry_metric='股票风格分类')
    indicator_stock_scales = stocks_industry_static.stocks_industry_static_indicators(industry_metric='股票规模指标')

    df_stock_hushen300 = dict_to_df(indicator_stock_hushen300)
    df_stock_styles = dict_to_df(indicator_stock_styles)
    df_stock_scales = dict_to_df(indicator_stock_scales)

    dfs = [df_stock_hushen300,df_stock_styles,df_stock_scales]
    df_final = merge_dfs(dfs)
    return df_final

def get_stocks_industry_dynamic_indicators(ind_stocks_general, ind_stocks_perf, ind_stocks_perf_holding_funds_share):
    past_quater_number = 4
    stocks_industry_dynamic = stocks_industry_dynamic_indicators(ind_stocks_general, past_quater_number)
    # a. Average rate of change and standard deviation of Stock Price over the past N quarters
    avg_price_stock_id_list, avg_price_mean, avg_price_std = stocks_industry_dynamic.q_quater_mean_std('当月成交均价 [复权方式]不复权',ind_stocks_perf)
    # b. Average rate of change and standard deviation of Total Fund Shareholding over the past N quarters
    fund_shareholding_id_list, fund_shareholding_mean, fund_shareholding_std = stocks_industry_dynamic.q_quater_mean_std('基金持股总值',ind_stocks_perf_holding_funds_share)
    # c. Average rate of change and standard deviation of Number of Fund Companies over the past N quarters
    fund_number_id_list, fund_number_mean, fund_number_std = stocks_industry_dynamic.q_quater_mean_std('持股基金家数 [股本类型]流通股本',ind_stocks_perf_holding_funds_share)
    assert len(avg_price_stock_id_list) == len(fund_shareholding_id_list) == len(fund_number_id_list)

    df_avg_price = meanstd_list_to_df(avg_price_stock_id_list, avg_price_mean, avg_price_std, flag='avg_price')
    df_fund_shareholding = meanstd_list_to_df(fund_shareholding_id_list, fund_shareholding_mean, fund_shareholding_std, flag='fund_shareholding')
    df_fund_number = meanstd_list_to_df(fund_number_id_list, fund_number_mean, fund_number_std, flag='fund_number')

    dfs = [df_avg_price,df_fund_shareholding,df_fund_number]
    df_final = merge_dfs(dfs)
    return df_final

def get_stocks_dynamic_indicators(ind_stocks_general, ind_stocks_perf, ind_stocks_holding_funds_share, ind_stocks_perf_with_fund):
    past_quater_number = 4
    minimal_fund_number = 4
    stocks_dynamic = stocks_dynamic_indicators(ind_stocks_general, ind_stocks_perf, ind_stocks_perf_with_fund, past_quater_number,minimal_fund_number)

    close_price_stock_id_list, close_price_mean, close_price_std = stocks_dynamic.q_quater_mean_std('月收盘价 [复权方式]不复权',ind_stocks_perf)## data base:ind_stocks_perf
    avg_price_stock_id_list, avg_price_mean, avg_price_std = stocks_dynamic.q_quater_mean_std('当月成交均价 [复权方式]不复权',ind_stocks_perf)## data base:ind_stocks_perf
    turnover_rate_stock_id_list, turnover_rate_mean, turnover_rate_std = stocks_dynamic.q_quater_mean_std('月换手率 [单位]%',ind_stocks_perf)## data base:ind_stocks_perf
    amplitutde_stock_id_list, amplitutde_mean, amplitutde_std = stocks_dynamic.q_quater_mean_std('月振幅 [单位]%',ind_stocks_perf)## data base:ind_stocks_perf
    margin_diff_stock_id_list, margin_diff_mean, margin_diff_std = stocks_dynamic.q_quater_mean_std('融资融券差额 [单位]元',ind_stocks_perf)## data base:ind_stocks_perf
    share_ratio_of_funds_stock_id_list, share_ratio_of_funds_mean, share_ratio_of_funds_std = stocks_dynamic.q_quater_mean_std('基金持股比例 [单位]% [比例类型]占流通股比例',ind_stocks_holding_funds_share) ## data base: ind_stocks_holding_funds_share
    num_of_funds_stock_id_list, num_of_funds_mean, num_of_funds_std = stocks_dynamic.q_quater_mean_std('持股基金家数 [股本类型]流通股本',ind_stocks_holding_funds_share) ## data base:ind_stocks_holding_funds_number
    fund_owner_affinity_stock_id_list, fund_owner_affinity_mean, fund_owner_affinity_std = stocks_dynamic.q_quater_fund_owner_affinity_mean_std('基金管理人简称')
    fund_revisit_stock_id_list, fund_revisit_mean = stocks_dynamic.q_quater_fund_revisit_mean('基金代码')
    cyclical_industry_stock_id_list, cyclical_industry = stocks_dynamic.stock_cyclical_industry(k_cyclical_industry_keys)
    assert len(close_price_stock_id_list) == len(avg_price_stock_id_list) == len(turnover_rate_stock_id_list) == len(amplitutde_stock_id_list) == len(margin_diff_stock_id_list) == \
           len(share_ratio_of_funds_stock_id_list) == len(num_of_funds_stock_id_list)== len(fund_owner_affinity_stock_id_list) == len(fund_revisit_stock_id_list) == len(cyclical_industry_stock_id_list)

    df_close_price = meanstd_list_to_df(close_price_stock_id_list, close_price_mean, close_price_std, flag='close_price')
    df_avg_price = meanstd_list_to_df(avg_price_stock_id_list, avg_price_mean, avg_price_std, flag='avg_price')
    df_turnover_rate = meanstd_list_to_df(turnover_rate_stock_id_list, turnover_rate_mean, turnover_rate_std, flag='turnover_rate')
    df_amplitutde = meanstd_list_to_df(amplitutde_stock_id_list, amplitutde_mean, amplitutde_std, flag='amplitutde')
    df_margin_diff = meanstd_list_to_df(margin_diff_stock_id_list, margin_diff_mean, margin_diff_std, flag='margin_diff')
    df_share_ratio_of_funds = meanstd_list_to_df(share_ratio_of_funds_stock_id_list, share_ratio_of_funds_mean, share_ratio_of_funds_std, flag='share_ratio_of_funds')
    df_num_of_funds= meanstd_list_to_df(num_of_funds_stock_id_list, num_of_funds_mean, num_of_funds_std, flag='num_of_funds')
    df_fund_owner_affinity= meanstd_list_to_df(fund_owner_affinity_stock_id_list, fund_owner_affinity_mean, fund_owner_affinity_std, flag='fund_owner_affinity')
    df_fund_revisit = cumstomlist_to_df(fund_revisit_stock_id_list, fund_revisit_mean,flags=['fund_revisit'])
    df_cyclical_industry = cumstomlist_to_df( cyclical_industry_stock_id_list, cyclical_industry, flags=['cyclical_industry'])
    dfs = [df_close_price,df_avg_price,df_turnover_rate,df_amplitutde,df_margin_diff,df_share_ratio_of_funds,df_num_of_funds,df_fund_owner_affinity,df_fund_revisit,df_cyclical_industry]

    df_final = merge_dfs(dfs)
    return df_final

def compute_indicators(db):
    # # raw data
    print("---------------------------get funds info-------------------------------------------")
    fund_generic = db.get_funds().info()
    print("fund_generic:\n",fund_generic)
    print("---------------------------get funds stats * date-------------------------------------------")
    fund_stats_date = db.get_fund_stats('date')
    print("fund_stats_date:\n",fund_stats_date)
    print("---------------------------get funds stats * topn_stocks-------------------------------------------")
    fund_stats_topn_stocks = db.get_fund_stats('topn_stocks')
    print("fund_stats_topn_stocks:\n",fund_stats_topn_stocks)
    print("---------------------------get stocks info-------------------------------------------")
    stock_generic = db.get_stocks().info()
    print("stock_generic:\n",stock_generic)
    print("---------------------------get stocks stats * date-------------------------------------------")
    stock_stats_date = db.get_stock_stats('date')
    print("stock_stats_date:\n",stock_stats_date)
    print("---------------------------finish get funds stock info-------------------------------------------")

    ind = Indicator(db)

    # ## Stock Indicators
    print('stock indicators')
    ind_stocks_general = ind.get_stock_general()
    ind_stocks_perf    = ind.get_stock_performance()
    ind_stocks_holding_funds_share = ind.get_stock_holding_funds_share()
    ind_stocks_holding_funds_number = ind.get_stock_holding_funds_number()
    ind_stocks_holding_topn_funds = ind.get_stock_topn_holding_funds()

    # ## Fund Indicators
    print('fund indicators')
    ind_funds_general = ind.get_funds_general()

    ## parameter
    ## --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## common part
    print('common part')

    # for class stocks_industry_static_indicators xy
    stocks_scale = db.get_stock_stats('date')[['股票代码', '股票规模指标']].dropna().groupby(['股票代码']).last().reset_index()
    stocks_scale.columns = ['股票代码', '股票规模指标']
    stocks_hushen300 = db.get_stocks()[['股票代码','是否属于重要指数成份 [所属指数]沪深300指数']]
    ind_stocks_general_new = pd.merge(ind_stocks_general, stocks_scale, how='inner', on=['股票代码'])
    ind_stocks_general_new = pd.merge(ind_stocks_general_new, stocks_hushen300, how='inner', on=['股票代码'])
    ind_stocks_general_new['是否属于重要指数成份 [所属指数]沪深300指数'].fillna('否', inplace=True)

    # for class stocks_industry_dynamic_indicators xy
    ind_stocks_perf_holding_funds_share = pd.merge(ind_stocks_perf, ind_stocks_holding_funds_share, how='outer', on=['日期', '股票代码'])
    ind_stocks_perf_holding_funds_share['基金持股总值']=ind_stocks_perf_holding_funds_share['基金持股数量 [单位]股']*ind_stocks_perf_holding_funds_share['月收盘价 [复权方式]不复权']

    # for class stocks_dynamic_indicators jy

    # for fund info mapping to stock jy
    ind = Indicator(db)
    funds_topn_stocks = ind.get_funds_topn_stocks().reset_index()
    print('funds_topn_stocks')
    funds_topn_stocks.columns = ['日期','基金代码','股票名称','仓股持仓占流通股比例','重仓股股票市值']
    new_df = pd.merge(ind_stocks_perf,funds_topn_stocks,how='outer',on=['日期','股票名称'])
    ind_stocks_perf_with_fund = pd.merge(new_df,ind_funds_general,how='outer',on=['日期','基金代码'])

    df_stocks_industry_static_indicators = get_stocks_industry_static_indicators(ind_stocks_general_new)
    print("df_stocks_industry_static_indicators:",df_stocks_industry_static_indicators)
    df_stocks_industry_dynamic_indicators = get_stocks_industry_dynamic_indicators(ind_stocks_general, ind_stocks_perf, ind_stocks_perf_holding_funds_share)
    print("df_stocks_industry_dynamic_indicators:",df_stocks_industry_dynamic_indicators)
    df_stocks_dynamic_indicators = get_stocks_dynamic_indicators(ind_stocks_general, ind_stocks_perf, ind_stocks_holding_funds_share, ind_stocks_perf_with_fund)
    print("df_stocks_dynamic_indicators:",df_stocks_dynamic_indicators)

    dfs = [
            df_stocks_industry_static_indicators,
            df_stocks_industry_dynamic_indicators,
            df_stocks_dynamic_indicators,
            ]

    return merge_dfs(dfs)

## tool part
## --------------------------------------------------------------------------------------------------------------------------------------------------------------------
def dict_to_df(my_dict):
    df = pd.DataFrame.from_dict(my_dict, orient='index').reset_index()
    df.rename(columns={'index':'id'},inplace=True)
    return df

def meanstd_list_to_df(id,mean,std,flag):
    dict = {"id" : id,
        flag + "_mean": mean,
        flag +  "_std" : std}
    df=pd.DataFrame(dict)
    return df

def cumstomlist_to_df(list1,list2,flags):
    dict = {
        "id": list1,
        flags[0]: list2}
    df=pd.DataFrame(dict)
    return df

def merge_dfs(dfs):
    df_final = pd.DataFrame(columns=['id'])
    for df in dfs:
        df_final = df_final.merge(df, on=['id'], how='outer')
    return df_final

## main func
## --------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
      db = Database('data')
      database = compute_indicators(db)
      print("final database:", database)
    except Exception as e:
        raise e

