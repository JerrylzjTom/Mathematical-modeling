# **一次参赛，终生受益！**

## MathorCup大数据挑战赛（2023.10.27-2023.11.03）

### 总体感受

参加的第一个数模比赛，感触很多，也学得了很多。由于之前没有什么经验，我就随意拉了两个人临时组了个队伍，队里三个小白，为了能平摊200元的报名费，我还拉了个学法学的同学。这次比赛，我是建模+编程+论文一把抓，因为队里其他人都对模型不了解，编程能力也不行，但幸运的是他们都比较负责，我安排的一些边缘任务他们还是能很高效的完成，给我省了很多时间。虽说有些吃力但最后还是完成了。还好这次比赛比较简单，我一个人可以应付三人工作，但还是希望能找两个能力稍微强一些的同志，合理分工。我还发现，数学建模过程和科研过程非常像，都是建模-实验-论文，参加数学建模比赛可以很大程度提升科研能力。我还发现，在比赛期间可以短时间速成很多东西，可能平时需要花几个星期学的事情，比赛期间迫于时间，你可能一天就能学会一项技能。我在知乎上看到一篇文章，有人提问数学建模比赛有什么意义，一个人回答说：数学建模比赛，是属于高付出但不知道有没有回报的过程。因为比赛没有标准答案！学生思维就是希望事事都有标准答案，因为有了标准答案我们就有了方向，有了可以奋斗的目标，这个目标是别人给你的。但数模比赛不是的，我们可以建很多模型，我们可以从不同角度解决问题，我们没有一个确定标准答案，这就导致我们完全不知道我们的方向是不是正确的，我们花几天时间建的模型会不会拟合效果很差……。我们进入社会后是没有标准答案的，我们做的每个决策我们都不知道会使事情变好还会使事情变差，我们每天都在摸石头过河，时时刻刻经历着高投入低回报的事情，这都很正常，不要幻想人生有标准答案。我们讨厌被标准，但也喜欢标准化的东西！考研、保研、绩点、竞赛、科研、工作，我们要想好自己要做什么，自己适合做什么，不要被周围人影响，因为他们的路不一定适合你，但无论选择什么，都坚信自己选择的是正确的，不要害怕！永远不要怀着努力就一定有回报的傲慢！

### 比赛过程

赛题在27号晚6点发布，A题我看是计算机视觉的题目，因为之前没有接触过这方面知识，我就直接决定选择B题。B题是大数据处理的题目，原本以为可以使用经典的机器学习模型解决，但仔细阅读题目后，发现要用时间序列模型解决，没有接触过时间序列的相关模型，建模初期还是比较迷茫的。我看网上的思路使用LSTM和ARIMA模型的居多，所以我就决定主要使用这两个模型。LSTM之前听说过，但还没有具体研究过原理和代码，ARIMA模型是第一次听说过。我在第一天晚上就把两个模型的代码写完了，我想着明天队友把处理后的数据给我，我直接把数据导入模型，但现实是残酷的。第二天，我把数据输入到LSTM模型中，发现因为数据维度的问题，代码一直跑不起来，因为之前没有用Pytorch搭项目的经验，所以debug的根本无从下手。然后我就转战ARIMA模型，代码输进去虽说能跑起来，但不知道在跑什么，也不知道结果代表着什么。所以28-29号两天都是迷茫的，根本不知道如何处理数据和模型。30号我在知乎上看到一个思路，是将30万个数据按类别分组。真是豁然开朗。原来是没有对数据进行分析，而且也不知道数据的含义是什么，导致模型跑的数据有很大问题。我把数据分组一共有1996组数据，意味着我要用模型跑1996次。由于已经第四天了，时间紧迫，我就只在ARIMA模型跑数据，LSTM模型随便编了几个数据。数据量比较大，第一问跑一次需要两个小时，第二问比较快，第三问能跑一天！

### 收获

1. 学会了Latex的基本命令，对latex语法基本熟悉，了解了数模比赛的论文排版
2. 学习了LSTM和ARIMA两个模型
3. 了解了数模比赛的基本流程
4. 意识到团队分工合作的重要性

### 不足

1. LSTM和ARIMA模型了解不够深入，完全套用代码，一知半解
2. 数据拟合很差，本质原因还是在模型理解不够深刻，以及数据分析和处理不够深入


## 数维杯
### 总体感受
这是第一次全英文数学建模，想着为参加美赛打打基础，熟悉一下英文写作。这次比赛只有我和lkh两个人，没有再找其他队友。这次选择了一道之前没有练习过的题目，是一道关于化学反应机理的题目，做题目前花了很长时间查文献，尽量搞清楚题目的所有概念。这次的数据很少，所以传统的机器学习和深度学习模型都用不了，唉，超大数据的题目和超小数据的题目遇到我都束手无策啊，还是要多打打kaggle，多练习数据分析的能力。这次比赛总体上感觉一般，模型没有创新，第四问我觉得用微分方程做最好，但之前没有系统学习过，所以我还是采用了传统的回归拟合来做。

### 收货
1. 使用latex更加熟练
2. 学习了多项式回归模型
3. 学习了灰色预测的方法
4. 学习了方差分析和非参数估计
### 不足
1. 对小样本数据处理经验不足
2. 模型创新性不够
3. 没有用微分方程解决问题

### 代码
#### 方差分析
```python
# 方差齐性检验
from scipy.stats import levene

def Test_equal_variance(data):
    statistic, p_value = levene(data[group],center='mean')

    # 输出检验结果
    print("Levene统计量:", statistic)
    print("p值:", p_value)

    # 判断是否拒绝原假设（p值小于显著性水平，通常设为0.05）
    alpha = 0.05
    if p_value < alpha:
        print("拒绝原假设，说明四组的方差不齐")
    else:
        print("接受原假设，说明四组的方差齐性没有显著差异")

# 不服从正态分布或方差不齐使用Kruskal test
from scipy.stats import kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def kruskal_test(data1, data2):

    all_data = np.concatenate([data1, data2])

    # Create labels for the groups
    labels = ['data1'] * len(data1) + ['data2'] * len(data2)
    # 判断是否存在显著差异
    kruskal_result = kruskal(data1, data2)

    # Check if there is a significant difference
    if kruskal_result.pvalue < 0.05:
        print("Kruskal-Wallis test: Reject the null hypothesis, significant difference exists")

        # Perform post hoc Tukey's HSD test
        tukey_results = pairwise_tukeyhsd(list(all_data), labels, alpha=0.05)

        # Display post hoc results
        print("Tukey's HSD post hoc test results:")
        print(tukey_results)

    else:
        print("Kruskal-Wallis test: Fail to reject the null hypothesis, no significant difference")
# 服从正态分布且方差齐性没有显著差异使用单因素方差分析

from scipy.stats import f_oneway
def ANOVA(data1, data2):
    anova_result = f_oneway(data1, data2)
    # 打印方差分析结果
    print("One-Way ANOVA p-value:", anova_result.pvalue)

    # 判断是否存在显著差异
    alpha = 0.05
    if anova_result.pvalue < alpha:
        print("One-Way ANOVA: Reject the null hypothesis, significant difference exists")
        # 将数据整合成一列
        all_data = np.concatenate([data1, data2])

        # Create labels for the groups
        labels = ['data1'] * len(data1) + [data2] * len(data2)

        # 执行 Tukey's HSD 进行事后多重比较
        tukey_results = pairwise_tukeyhsd(list(all_data), labels, alpha=alpha)

        # 显示事后多重比较结果
        print("Tukey's HSD post hoc test results:")
        print(tukey_results)

    else:
        print("One-Way ANOVA: Cannot reject the null hypothesis, no significant difference")
```
#### 堆叠图
```python
import matplotlib.pyplot as plt
import numpy as np

# data from https://allisonhorst.github.io/palmerpenguins/
import matplotlib.pyplot as plt

def stacked_bar(data):
    plt.figure(figsize=(15, 8))
    categories = data.columns.values
    # 创建堆叠图
    fig, ax = plt.subplots()

    # 绘制第一个变量的条形图，并设置label参数
    bar1 = ax.bar(categories, data[category1], label='Tar',color="#AB9A6F")

    # 绘制第二个变量的条形图，并设置bottom参数
    bar2 = ax.bar(categories, data[category2], bottom=Tar, label='Water',color="#D6E3B7")

    # 绘制第三个变量的条形图，并设置bottom参数
    bar3 = ax.bar(categories, data[category3], bottom=[i+j for i, j in zip(Tar,Water)], label='Char', color='#95A96A')

    bar4 = ax.bar(categories, data[categroy4], bottom=[i+j+k for i,j,k in zip(Tar,Water, Char)], label='Syngas', color="#45602D")
    # 添加图例
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # 设置标题和轴标签
    ax.set_xlabel('data')
    ax.set_ylabel('Yields(wt.%, daf)')
    # 调整图表布局，防止注释遮挡图表内容
    plt.tight_layout()
```
#### 柱状图
```python
def bar(data1, data2, data3):
    x1 = data1.index.values
    x = np.arange(5)
    width = 0.3

    f, axs = plt.subplots(2, 3,figsize=(20, 10))
    bars1 = axs[0,0].bar(x - width, data1, width, label='data1')
    bars2 = axs[0,0].bar(x, data2, width, label='data2')
    bars3 = axs[0,0].bar(x + width, data3, width, label='data3')

    def add_labels(bars):
        for bar in bars:
            yval = bar.get_height()
            axs[0,0].text(bar.get_x() + bar.get_width()/3, yval, round(yval, 2), ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    axs[0,0].set_xticks(x, x1)
    axs[0,0].set_xlabel("")
    axs[0,0].set_ylabel("")
```

#### 灰色预测
```python
class GrayForecast():
    def __init__(self, data, datacolumn=None):
    
        if isinstance(data, pd.core.frame.DataFrame):
            self.data=data
            try:
                self.data.columns = ['数据']
            except:
                if not datacolumn:
                    raise Exception('您传入的dataframe不止一列')
                else:
                    self.data = pd.DataFrame(data[datacolumn])
                    self.data.columns=['数据']
        elif isinstance(data, pd.core.series.Series):
            self.data = pd.DataFrame(data, columns=['数据'])
        else:
            self.data = pd.DataFrame(data, columns=['数据'])
    
        self.forecast_list = self.data.copy()
    
        if datacolumn:
            self.datacolumn = datacolumn
        else:
            self.datacolumn = None
        #save arg:
        #        data                DataFrame    数据
        #        forecast_list       DataFrame    预测序列
        #        datacolumn          string       数据的含义
    def level_check(self):
        # 数据级比校验
        n = len(self.data)
        lambda_k = np.zeros(n-1)
        for i in range(n-1):
            lambda_k[i] = self.data.ix[i]["数据"]/self.data.ix[i+1]["数据"]
            if lambda_k[i] < np.exp(-2/(n+1)) or lambda_k[i] > np.exp(2/(n+2)):
                flag = False
        else:
            flag = True
    
        self.lambda_k = lambda_k
    
        if not flag:
            print("级比校验失败，请对X(0)做平移变换")
            return False
        else:
            print("级比校验成功，请继续")
            return True
    
    #save arg:
    #        lambda_k            1-d list
    
    
    def GM_11_build_model(self, forecast=5):
        if forecast > len(self.data):
            raise Exception('您的数据行不够')
        X_0 = np.array(self.forecast_list['数据'].tail(forecast))
    #       1-AGO
        X_1 = np.zeros(X_0.shape)
        for i in range(X_0.shape[0]):
            X_1[i] = np.sum(X_0[0:i+1])
    #       紧邻均值生成序列
        Z_1 = np.zeros(X_1.shape[0]-1)
        for i in range(1, X_1.shape[0]):
            Z_1[i-1] = -0.5*(X_1[i]+X_1[i-1])
    
        B = np.append(np.array(np.mat(Z_1).T), np.ones(Z_1.shape).reshape((Z_1.shape[0], 1)), axis=1)
        Yn = X_0[1:].reshape((X_0[1:].shape[0], 1))
    
        B = np.mat(B)
        Yn = np.mat(Yn)
        a_ = (B.T*B)**-1 * B.T * Yn
    
        a, b = np.array(a_.T)[0]
    
        X_ = np.zeros(X_0.shape[0])
        def f(k):
            return (X_0[0]-b/a)*(1-np.exp(a))*np.exp(-a*(k))
    
        self.forecast_list.loc[len(self.forecast_list)] = f(X_.shape[0])
        
    def forecast(self, time=5, forecast_data_len=5):
        for i in range(time):
            self.GM_11_build_model(forecast=forecast_data_len)
            
    def log(self):
        res = self.forecast_list.copy()
        if self.datacolumn:
            res.columns = [self.datacolumn]
        return res
    def reset(self):
        self.forecast_list = self.data.copy()
        
        
    def plot(self, data):
        plt.scatter(data, label='Original Data', color='red')
        plt.plot(range(0,len(self.forecast_list)),self.forecast_list, label='Forecast Data')
        if self.datacolumn:
            plt.ylabel(self.datacolumn)
            plt.legend()

gf = GrayForecast(data, column)
gf.forecast(num) # 预测
gf.log() # 结果
```

## 亚太杯(2023.11.23-2023.11.27)


### 代码
#### 基本包
```python 
import numpy as np
import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设 置 字 体
plt.rcParams["axes.unicode_minus"]=False #该 语 句 解 决 图 像中 的 “ -” 负 号 的 乱 码 问 题
%matplotlib inline
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn
import scipy.stats as stats
from scipy.stats import shapiro, skew
import math
import plotly.express as px
import plotly.graph_objs as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from IPython.display import display
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
seed = 42
plotly_template = 'simple_white'
```

#### Describe
```python
def describe(df):
    '''
    This function plots a table containing Descriptive Statistics of the Dataframe
    '''
    mean_features = df.mean().round(2).apply(lambda x: "{:,.2f}".format(x)) 
    std_features = df.std().round(2).apply(lambda x: "{:,.2f}".format(x)) 
    q1 = df.quantile(0.25).round(2).apply(lambda x: "{:,.2f}".format(x))
    median = df.quantile(0.5).round(2).apply(lambda x: "{:,.2f}".format(x))
    q3 = df.quantile(0.75).round(2).apply(lambda x: "{:,.2f}".format(x))


    # Generating new Dataframe
    describe_df = pd.DataFrame({'Feature Name': mean_features.index,
                                'Mean': mean_features.values,
                                'Standard Deviation': std_features.values,
                                '25%': q1.values,
                                'Median': median.values,
                                '75%': q3.values})

    # Generating a Table w/ Pyplot
    fig = go.Figure(data = [go.Table(header=dict(values=list(describe_df.columns),
                                                 align = 'center',
                                                 fill_color = 'midnightblue',
                                               font=dict(color = 'white', size = 18)),
                                     cells=dict(values=[describe_df['Feature Name'],
                                                        describe_df['Mean'],
                                                        describe_df['Standard Deviation'],
                                                       describe_df['25%'],
                                                       describe_df['Median'],
                                                       describe_df['75%']],
                                                fill_color = 'gainsboro',
                                                align = 'center'))
                           ])

    fig.update_layout(title = {'text': f'<b>Descriptive Statistics of the Dataframe<br><sup> (Mean, Standard Deviation, 25%, Median, and 75%)</sup></b>'},
                      template = plotly_template,
                      height = 700, width = 950,
                      margin = dict(t = 100))

    fig.show()
```

#### Box
```python
def plot_boxplot_matrix(df):
    
    '''
    This function identifies all continuous features within the dataset and plots
    a matrix of boxplots for each attribute
    '''
    
    continuous_features = []
    for feat in df.columns:
        if df[feat].nunique() > 2:
            continuous_features.append(feat)
    
    num_cols = 2
    num_rows = (len(continuous_features) + 1) // num_cols


    fig = make_subplots(rows=num_rows, cols=num_cols)


    for i, feature in enumerate(continuous_features):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            go.Box(
                x=df[feature],
                name = ' '
            ),
            row=row,
            col=col
        )

        fig.update_yaxes(title_text = ' ', row=row, col=col)
        fig.update_xaxes(title_text= feature, row=row, col=col)
        fig.update_layout(
            title=f'<b>Boxplot Matrix<br> <sup> Continuous Features</sup></b>',
            showlegend=False,
            yaxis=dict(
            tickangle=-90  
        )
        )

    fig.update_layout(
        height=350 * num_rows,
        width=1000,
        margin=dict(t=100, l=80),
        template= plotly_template
    )


    fig.show()
plot_boxplot_matrix(df)
```

#### histogram
```python
def plot_histogram_matrix(df):
    
    '''
    This function identifies all continuous features within the dataset and plots
    a matrix of histograms for each attribute
    '''
    
    continuous_features = []
    for feat in df.columns:
        if df[feat].nunique() > 2:
            continuous_features.append(feat)
    num_cols = 2
    num_rows = (len(continuous_features) + 1) // num_cols

    fig = make_subplots(rows=num_rows, cols=num_cols)

    for i, feature in enumerate(continuous_features):
        row = i // num_cols + 1
        col = i % num_cols + 1

        fig.add_trace(
            go.Histogram(
                x=df[feature],
                name=feature
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=feature, row=row, col=col)
        fig.update_yaxes(title_text='Frequency', row=row, col=col)
        fig.update_layout(
            title=f'<b>Histogram Matrix<br> <sup> Continuous Features</sup></b>',
            showlegend=False
        )

    fig.update_layout(
        height=350 * num_rows,
        width=1000,
        margin=dict(t=100, l=80),
        template= plotly_template
    )

    fig.show()
plot_histogram_matrix(df)
```

#### Shapiro_wilk_test(检验是否服从正态分布)
```python
def shapiro_wilk_test(df):
    '''
    This function performs a Shapiro-Wilk test to check if the data is normally distributed or not, as well as skewness
    '''
    print(f'\033[1mShapiro-Wilk Test & Skewness:\033[0m')
    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  \n')

    numeric_columns = df.select_dtypes(include=['float', 'int']).columns

    for feature in numeric_columns:
        stats, p_value = shapiro(df[feature])

        if p_value < 0.05:
            text = f'{feature} Does Not Seem to be Normally Distributed'
        else:
            text = f'{feature} Seems to be Normally Distributed'

        print(f'{feature}')
        print(f'\n  Shapiro-Wilk Statistic: {stats:.2f}')
        print(f'\n  Shapiro-Wilk P-value: {p_value}')
        print(f'\n  Skewness: {np.round(skew(df[feature]), 2)}')
        print(f'\n  Conclusion: {text}')
        print('\n===============================================================================================')

    print('\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  \n')
    print(f'\033[1mEnd of Shapiro-Wilk Test\033[0m')
shapiro_wilk_test(df)
```

#### Correlation
```python
def plot_correlation(df):
    '''
    This function is resposible to plot a correlation map among features in the dataset
    '''
    corr = np.round(df.corr(), 2)
    mask = np.triu(np.ones_like(corr, dtype = bool))
    c_mask = np.where(~mask, corr, 100)

    c = []
    for i in c_mask.tolist()[1:]:
        c.append([x for x in i if x != 100])
    
    fig = ff.create_annotated_heatmap(z=c[::-1],
                                      x=corr.index.tolist()[:-1],
                                      y=corr.columns.tolist()[1:][::-1],
                                      colorscale = 'bluyl')

    fig.update_layout(title = {'text': '<b>Feature Correlation <br> <sup>Heatmap</sup></b>'},
                      height = 1050, width = 1050,
                      margin = dict(t=210, l = 80),
                      template = 'simple_white',
                      yaxis = dict(autorange = 'reversed'))

    fig.add_trace(go.Heatmap(z = c[::-1],
                             colorscale = 'bluyl',
                             showscale = True,
                             visible = False))
    
    
    
    fig.data[1].visible = True

    fig.show()
plot_correlation(df)
```

#### LSTM
```python
df = pd.read_excel("../Data/新能源汽车产销量.xlsx")

#Generate Time Seriese
df_date = df[['date', 'Production of new energy vehicles']]
df_date.set_index(df.date, inplace=True)
df_date.drop(columns='date', inplace=True)

# split the trainset and the testset
test_percent = 0.1
test_point = np.round(len(df_date)*test_percent)
test_index = int(len(df_date) - test_point)
train = df_date.iloc[:test_index]
test = df_date.iloc[test_index:]

# MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

# 导包
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss',patience=2)
length = 6
generator = TimeseriesGenerator(scaled_train,scaled_train,
                               length=length,batch_size=1)


validation_generator = TimeseriesGenerator(scaled_test,scaled_test,
                                          length=length,batch_size=1)
n_features = 1
model = Sequential()
model.add(LSTM(12, input_shape = (length, n_features)))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mse')
model.fit_generator(generator, epochs = 10, 
                    validation_data = validation_generator, 
                    callbacks = [early_stop])
model.save('model_LSTM.h5')
loss = pd.DataFrame(model.history.history)
loss.plot()

prediction = []
evaluation_batch = scaled_train[-length:]
current_batch = evaluation_batch.reshape(1, length, n_features)
for i in range (len(test)):
    current_prediction = model.predict(current_batch)[0]
    prediction.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis = 1)
prediction = scaler.inverse_transform(prediction)
test['LSTM Prediction'] = prediction
test.plot(figsize = (10, 4))
full_scaler = MinMaxScaler()
full_data_scale = scaler.transform(df_date)
length = 50 
generator = TimeseriesGenerator(full_data_scale, full_data_scale, length=length, batch_size=1)
model = Sequential()
model.add(LSTM(50, input_shape=(length, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
model.fit_generator(generator,epochs=6)

# full data to predict
prediction = []
evaluation_batch = full_data_scale[-length:]
current_batch = evaluation_batch.reshape(1, length, n_features)
for i in range (120):
    current_prediction = model.predict(current_batch)[0]
    prediction.append(current_prediction)
    current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis = 1)
prediction = scaler.inverse_transform(prediction)

from datetime import timedelta
from datetime import datetime
date_string = "2023-09"
date_format = "%Y-%m"
datetime_object = datetime.strptime(date_string, date_format)
future_months = [datetime_object + timedelta(days=30 * i) for i in range(1, 121)]
future_months
prediction_index = future_months
plt.plot(df_date.index,df_date['Production of new energy vehicles'])
plt.plot(prediction_index,prediction)
prelist = []
for i in prediction:
    prelist.append(i[0])
start_date = datetime(2023, 10, 1)

# 生成未来十年的日期列表（每个月）
date_list = [start_date + timedelta(days=30 * i) for i in range(10 * 12)]
date_list
forecast = pd.DataFrame(data={'forecast':prelist, 'Date': date_list})
```

#### ARIMA
##### 基本包
```python
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA as ARIMA
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
```
##### Stationarity test
```python
def test_stationarity(timeseries):
    #Determing rolling statistics
    MA = timeseries.rolling(window=12).mean()
    MSTD = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,5))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(MA, color='red', label='Rolling Mean')
    std = plt.plot(MSTD, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
```

##### Plot Time Series ACF and PACF
```python
def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()
```

##### Seasonal decompose
```python
def sesonal_decompose(data)
dec = sm.tsa.seasonal_decompose(data,period = 12,model = 'additive').plot()
plt.show()
```
##### Stationarity
```python
def stationarity(timeseries): #平稳性处理（timeseries 时间序列）
    ## 差分法,保存成新的列
    diff1 = timeseries.diff(1).dropna()  # 1阶差分 dropna() 删除缺失值
    diff2 = diff1.diff(1).dropna() #在一阶差分基础上再做一次一阶差分，即二阶查分
    ## 画图
    diff1.plot(color = 'red',title='diff 1',figsize=(10,4))
    diff2.plot(color = 'black',title='diff 2',figsize=(10,4))

    
    ## 平滑法
    rollmean = timeseries.rolling(window=4,center = False).mean() ## 滚动平均
    rollstd = timeseries.rolling(window=4,center = False).std() ## 滚动标准差
    ## 画图 
    rollmean.plot(color = 'yellow',title='Rolling Mean',figsize=(10,4))
    rollstd.plot(color = 'blue',title='Rolling Std',figsize=(10,4))
    
    return diff1,diff2,rollmean,rollstd
diff1,diff2,rollmean,rollstd = stationarity(df_date)
```
##### Test Whitenoise
```python
def testwhitenoise(data):
    m = 10# 检验10个自相关系数
    acf,q,p = sm.tsa.acf(data,nlags=m,qstat=True)
    out = np.c_[range(1,m+1),acf[1:],q,p]
    output = pd.DataFrame(out,columns=['lag','自相关系数','统计量Q值','p_values'])
    output = output.set_index('lag')# 设置第一列索引名称,可省略重复索引列1
    print(output)
```

##### Test Stationarity
```python
def teststeady(data,count=0):
    res_ADF = ADF(data)
    print('ADF检验结果为：', res_ADF)
    Pv = res_ADF[1]
    if Pv > 0.05:
        print('\033[1;31mP值：%s，原始序列不平稳，要进行差分！\033[0m' % round(Pv,5))
        count = count + 1
        print('\033[1;32m进行了%s阶差分后的结果如下\033[0m' % count)
        data = data.diff(1).dropna()
        teststeady(data,count)
    else:
        print('\033[1;34mP值：%s，原始序列平稳，继续建模\033[0m'% round(Pv,5))
    return data
```
##### Confirm p and q
```python
def confirm_p_q(data):
    fig = plt.figure(figsize=(8,6))
    testwhitenoise(data)
    train = teststeady(data)
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_pacf(train, lags=10, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_acf(train, lags=10, ax=ax2)
    plt.show()  ###可视化定阶

    pmax = int(len(data) / 10)
    qmax = int(len(data) / 10)
    AIC = sm.tsa.arma_order_select_ic(train,max_ar=pmax,max_ma=qmax,ic='aic')['aic_min_order']
    BIC = sm.tsa.arma_order_select_ic(train,max_ar=pmax,max_ma=qmax,ic='bic')['bic_min_order']
    HQIC = sm.tsa.arma_order_select_ic(train,max_ar=pmax,max_ma=qmax,ic='hqic')['hqic_min_order']
    print('AIC：',AIC)
    print('BIC：',BIC)
    print('HQIC：',HQIC)
    return AIC
```
##### Predict
```python
def prediction(data):
    ##tempmodel = ARMA(teststeady(data),pq).fit(disp=-1)
    tempmodel = sm.tsa.arima.ARIMA(data, order=(2,2,7)).fit()
    print(tempmodel.summary())
    #num = 10
    #predictoutside1 = tempmodel.forecast(num)[0]#预测样本外的
    predictoutside2 = tempmodel.predict(len(tempmodel.predict()),len(tempmodel.predict()) + 9,dynamic=True)##也是样本外预测，预测结果一致
    predictinside = tempmodel.predict()##样本内预测
    init_value = diff2.values[0]

    fig = plt.figure(figsize=(10, 6))
    predictinside = predictinside.cumsum()##差分还原
    pretrueinside = init_value + predictinside
    startprevalue = list(pretrueinside)[-1]
    predictoutside2 = predictoutside2.cumsum()##差分还原
    pretrueoutside = startprevalue + predictoutside2
    
    ##作图
    plt.plot(diff2.values,label='Original')
    plt.plot([init_value] + list(pretrueinside),label='Prediction')
    X = [i for i in range(len(diff2)-11,len(diff2))]
    #plt.plot(X,[startprevalue] + list(pretrueoutside), label='样本外预测值')
    allpredata = [init_value] + list(pretrueinside) + list(pretrueoutside)
    plt.legend()
    plt.show()
    return tempmodel,allpredata
``` 

