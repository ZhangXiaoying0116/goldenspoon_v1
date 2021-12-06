import glob
import pandas as pd
import goldenspoon
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score
import os

def print_to_log(filename, _prefix, log_content, flag=None):
    if flag == 'dict':
        with open(filename,'w') as f:
            f.write(_prefix + '\n')
            for key in log_content.keys():
                f.write(str(key))
                f.write(" : ")
                if  str(type(log_content[key]))== "<class 'list'>":
                    for i in log_content[key]:
                        f.write(str(i)+'\n')
                else:
                    f.write(str(log_content[key]))
                f.write('\n' )
        f.close()
    else:
        with open(filename,'w') as f:
            f.write(_prefix + '\n' + \
                    str(log_content) + '\n' )
        f.close()

class Regress():
    def __init__(self,model, pickle_path, save_path, date_list, predict_n_month, predict_month_id, predict_label_type, all_indicator_use=True, indicator_use_list=[]):
        self.model = model
        self.pickle_path = pickle_path
        self.save_path = save_path
        self.date_list = date_list
        self.predict_n_month = predict_n_month
        self.predict_label_type = '_' + predict_label_type # 'predict_pricechange' or 'predict_price'
        self.predict_month_id = str(predict_month_id) + self.predict_label_type
        self.all_indicator_use = all_indicator_use
        self.indicator_use_list = indicator_use_list

    ## 1. Data generation
    def merge_df_indicator_label(self, df_indicator, df_label_list):
        df_indicator_label = df_indicator
        for m in range(len(df_label_list)):
            df_label = df_label_list[m]
            # data = df_label['日期'][0]
            df_label = df_label.reset_index()
            if m==0:
                df_label.columns = ['id', 'date', '0_label']
                df_label = df_label.drop(columns=['date'], axis=1)
            else:
                if self.predict_label_type == '_predict_pricechange':
                    df_label.columns = ['id', 'date', "predict_price", str(m) + self.predict_label_type]
                    df_label = df_label.drop(columns=['date'], axis=1)
                    df_label = df_label.drop(columns=['predict_price'], axis=1)
                elif self.predict_label_type == '_predict_price':
                    df_label.columns = ['id', 'date', str(m) + self.predict_label_type, "predict_pricechange"]
                    df_label = df_label.drop(columns=['date'], axis=1)
                    df_label = df_label.drop(columns=['predict_pricechange'], axis=1)
            df_indicator_label = df_indicator_label.merge(df_label, on=['id'], how='inner')
        return df_indicator_label

    def merge_indicator_label(self, df_indicator, label_pickle_list):
        df_label_list= []
        for filename in label_pickle_list:
            df_label = pd.read_pickle(filename)
            df_label_list.append(df_label)
            # print("TODO-----filename:{}, df_label:{}".format(filename, df_label))

        df_indicator_label = self.merge_df_indicator_label(df_indicator, df_label_list)
        return df_indicator_label

    def get_train_indicator_label(self, df_indicator_label):
        if self.all_indicator_use:
            X_df = df_indicator_label[self.all_indicator_list] ## list(all_indicator_list)[1:] not include first column ['id']
        else:
            if self.indicator_use_list==[]:
                assert("must provide indicators you want for data load")
            X_df = df_indicator_label[self.indicator_use_list]
        self.end_date_stock_price = df_indicator_label.iloc[:,df_indicator_label.columns.str.endswith('0_label')]
        Y_df = df_indicator_label.iloc[:,df_indicator_label.columns.str.endswith(self.predict_month_id)]
        Y_df = pd.concat([self.end_date_stock_price, Y_df], axis=1)
        return X_df, Y_df

    def data_preprocess(self):
        totaldate_X_df = pd.DataFrame()
        totaldate_Y_df = pd.DataFrame()
        for date in self.date_list:
            indicator_pickle = self.pickle_path + 'indicators.' + date + '.pickle'
            df_indicator = pd.read_pickle(indicator_pickle)
            self.all_indicator_list = list(df_indicator.columns.values)[1:]
            if not df_indicator.empty:
                label_pickle_list = []
                for m in range(self.predict_n_month+1): ## include 0_month
                    label_pickle = self.pickle_path + 'labels.' + str(m) + '_month.'+ date +'.pickle'
                    label_pickle_list.append(label_pickle)
                df_indicator_label = self.merge_indicator_label(df_indicator, label_pickle_list)
                X_df, Y_df = self.get_train_indicator_label(df_indicator_label)
                totaldate_X_df = pd.concat([totaldate_X_df, X_df], axis=0)
                totaldate_Y_df = pd.concat([totaldate_Y_df, Y_df], axis=0)
            else:
                print("{} is a empty dataframe!".format(indicator_pickle))
        x_train, x_test, y_train, y_test = train_test_split(totaldate_X_df, totaldate_Y_df, random_state=1)
        return x_train, x_test, y_train, y_test


    ## 2. result visualization
    def draw(self, y_pred, y_test, cur_stock_price_test, R2, show_interval=None):
        if show_interval == None:
            show_interval = len(y_pred)
        start_ = 0
        end_ = show_interval
        while start_ < len(y_pred):
            if end_ > len(y_pred):
                end_ = len(y_pred)
            plt.figure()
            show_len = end_ - start_
            plt.plot(np.arange(show_len), y_test[start_:end_], 'bo-', label='true value')
            plt.plot(np.arange(show_len), y_pred[start_:end_], 'ro-', label='predict value')
            if self.predict_label_type == '_predict_price':
                plt.plot(np.arange(show_len), cur_stock_price_test[start_:end_], 'g^-', label='current stock price')
            elif self.predict_label_type == '_predict_pricechange':
                plt.axhline(y=0, color='g', linestyle='-')
            plt.title('R2: %f'%R2)
            plt.legend()
            flag = 'demo_index_'+str(start_)+'-'+str(end_)+'.png'
            plt.savefig(self.save_path + flag)
            start_ += show_interval
            end_ += show_interval
        
        ## draw the y_pred and y_test scatter diagram
        plt.figure()
        x = np.linspace(0,1.0, len(y_pred))
        plt.scatter(x, y_pred, c='r', marker='*')
        plt.scatter(x, y_test, c='', marker='o',edgecolors='g')
        plt.savefig(self.save_path + 'y_scatter.png')


    ## 3. confusion_matrix
    def perf_measure(self, y_pred, y_true, cur_stock_price_test):
        y_pred = np.array(y_pred).flatten()
        y_true = np.array(y_true).flatten()
        cur_stock_price_test = np.array(cur_stock_price_test).flatten()

        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(len(y_true)):
            if self.predict_label_type == '_predict_price':
                y_true_case = y_true[i]-cur_stock_price_test[i]
                y_pred_case = y_pred[i]-cur_stock_price_test[i]
            elif self.predict_label_type == '_predict_pricechange':
                y_true_case = y_true[i]
                y_pred_case = y_pred[i]
            ## share prices are rising in y_true, also y_pred
            if (y_true_case) >= 0 and (y_pred_case) >= 0:
                TP += 1
            ## share prices are rising in y_pred, but falling in y_true
            if (y_true_case) < 0 and (y_pred_case) >= 0:
                FP += 1
            ## share prices are falling in y_true, also y_pred
            if (y_true_case) < 0 and (y_pred_case) < 0:
                TN += 1
            ## share prices are rising in y_true, but falling in y_pred
            if (y_true_case) >= 0 and (y_pred_case) < 0:
                FN += 1

        #[[TN|FP]
        # [FN|TP]]
        confusion_flag = np.array([['TN', 'FP'], ['FN', 'TP']])
        confusion_matrix = np.array([[TN, FP], [FN, TP]])
        plt.matshow(confusion_matrix, cmap=plt.cm.Greens)
        plt.colorbar()   
        for i in range(2):
            for j in range(2):     
                plt.annotate(confusion_flag[i,j] + ' : ' + str(confusion_matrix[i,j]), xy=(i, j), horizontalalignment='center', verticalalignment='center')
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
        flag = 'demo_confusion_matrix.png'
        plt.savefig(self.save_path + flag)

        return TP, FP, TN, FN

    ## 4. Linear regression
    def run(self):
        ## get train/test data
        x_train, x_test, y_train, y_test = self.data_preprocess()
        print("x_train.shape:{},\n x_test.shape:{},\n y_train.shape:{},\n y_test.shape:{},\n".format(x_train.shape, x_test.shape, y_train.shape, y_test.shape))

        cur_stock_price_train = y_train['0_label']
        cur_stock_price_test = y_test['0_label']
        y_train = y_train[self.predict_month_id]
        y_test = y_test[self.predict_month_id]

        ## model training
        self.model.fit(x_train, y_train)
        y_pred = self.model.predict(x_test)
        ## TODO p_value t_value
        R2_y_pred_y_test = r2_score(y_test, y_pred)

        if self.all_indicator_use:
            indicator_weight = list(zip(self.all_indicator_list, list(self.model.coef_)))
        else:
            indicator_weight = list(zip(self.indicator_use_list, list(self.model.coef_)))

        ## print result
        with open(self.save_path + 'indicator_weight.log','w') as f:
            for i_weight in indicator_weight:
                f.write(str(i_weight))
                f.write('\n' )
            f.close()

        TP, FP, TN, FN = self.perf_measure(y_pred, y_test, cur_stock_price_test)
        print("TP:{}, FP:{}, TN:{}, FN:{}".format(TP, FP, TN, FN))
        self.draw(y_pred, y_test, cur_stock_price_test, R2_y_pred_y_test, show_interval=50)
        return

if __name__ == '__main__':
    ## base parameter
    for predict_month_id in [1,2,3]:
        dataflag='regress_data'
        saveflag='regress_result'
        predict_n_month=3
        all_indicator_use=True
        predict_label_type='predict_pricechange' # 'predict_pricechange' or 'predict_price'
        indicator_use_list=[] ## full indicator list: ['是', '成长型', '混合型', '价值型', '小盘股', '中盘股', '大盘股', 'avg_price_mean_x', 'avg_price_std_x', 'fund_shareholding_mean', 'fund_shareholding_std', 'fund_number_mean', 'fund_number_std', 'close_price_mean', 'close_price_std', 'avg_price_mean_y', 'avg_price_std_y', 'turnover_rate_mean', 'turnover_rate_std', 'amplitutde_mean', 'amplitutde_std', 'margin_diff_mean', 'margin_diff_std', 'share_ratio_of_funds_mean', 'share_ratio_of_funds_std', 'num_of_funds_mean', 'num_of_funds_std', 'fund_owner_affinity_mean', 'fund_owner_affinity_std', 'cyclical_industry']
        pickle_path='./'+dataflag+'/'
        save_path=saveflag+'/' + str(predict_month_id) +'month/'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        date_list = ['2020-09-30','2020-12-31','2021-03-31','2021-06-30']
        r = Regress(linear_model.LinearRegression(),pickle_path,save_path,date_list,predict_n_month,predict_month_id,predict_label_type,all_indicator_use,indicator_use_list)
        r.run()
