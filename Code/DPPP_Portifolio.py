import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import warnings
import random
from scipy import stats
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

file_path = r"上证50月度因子+收益+权重数据2005-2024.csv"
df = pd.read_csv(file_path,parse_dates=['date'])
df['weight'] = df['weight']/100
column = df.columns.tolist()[7:]

target_rows = 50
def softmax(x):
    exp_x = np.exp(x)  # 减去最大值防止溢出
    return exp_x / exp_x.sum()

def top_5_weights(weights,num=int(target_rows * 0.1)):
    top_5_indices = weights.nlargest(num).index
    new_weights = pd.Series(0, index=weights.index)
    new_weights.loc[top_5_indices] = 1 / num
    return new_weights

# 计算年化收益率
def annualized_return(returns):
    total_return = np.prod(1 + returns) - 1
    annualized_return = (1 + total_return) ** (12 / len(returns)) - 1
    return annualized_return

# 计算年化波动率
def annualized_volatility(returns):
    return np.std(returns) * np.sqrt(12)

# 计算夏普比率
def sharpe_ratio(returns, risk_free_rate=0.02):
    return (annualized_return(returns) - risk_free_rate) / annualized_volatility(returns)

# 计算Calmar比率
def calmar_ratio(returns):

    # 年化收益率
    annual_return = (np.prod(1 + returns) ** (12 / len(returns))) - 1  # 假设是月度收益率
    
    # 计算最大回撤
    cumulative = np.cumprod(1 + returns)  # 累积收益率
    peak = np.maximum.accumulate(cumulative)  # 累积峰值
    drawdown = (peak - cumulative) / peak  # 回撤
    max_drawdown = np.max(drawdown)  # 最大回撤

    # 计算 Calmar 比率
    return annual_return / max_drawdown if max_drawdown > 0 else np.nan

initial_train_end = '2020-12-31'  # 初始训练集结束时间
net_value_dfs = []

# gamma_value = 50
learning_rate = 1e-4
epochs = 50
batch_size = 5  
df_performance = pd.DataFrame()
for gamma_value in [10,20,30,40,50]:
    # 初始化模型
    input_size = df.shape[1] - 7  # 假设有5个非特征列
    def overall_utility(model, X, y, gamma=gamma_value):
        all_portfolio_returns = []  # 存储所有截面加权收益率
        for idx in range(len(X)):
            X_batch, y_batch = X[idx], y[idx]
            raw_weights = model(X_batch).squeeze()
            weights = torch.softmax(raw_weights, dim=0)  # 权重归一化
            
            # weights = raw_weights/raw_weights.sum()
            # # 论文原文方法
            # weights = (raw_weights-raw_weights.mean())/raw_weights.std()/50

            portfolio_return = torch.sum(weights * y_batch)
            utility_section = ((1+portfolio_return)**(1-gamma))/(1-gamma)
            lamda = 0.01
            squared_sum = lamda*torch.sum(weights**2)
            utility_section = -utility_section+squared_sum
            all_portfolio_returns.append(utility_section)
            
        utility = sum(all_portfolio_returns)
        return utility  # 负效用用于优化

    # 初始训练集
    current_train = df[df['date'] <= initial_train_end].copy()

    # 滚动窗口训练
    df_result = pd.DataFrame()

    def pad_to_rows(array, target_rows):
        current_rows, cols = array.shape
        if current_rows < target_rows:
            pad_width = [(0, target_rows - current_rows), (0, 0)]
            padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
        else:
            padded_array = array
        return padded_array
    def pad_to_length(array, target_rows):
        current_length = array.shape[0]
        if current_length < target_rows:
            pad_width = (0, target_rows - current_length)
            padded_array = np.pad(array, pad_width, mode='constant', constant_values=0)
        else:
            padded_array = array
        return padded_array

    class MLP(nn.Module):
        def __init__(self, input_size):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out1 = self.relu(self.fc1(x))
            out2 = self.relu(self.fc2(out1))
            out3 = self.fc3(out2)
            out3 = self.sigmoid(out3)
            return out3

    model = MLP(input_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # 获取测试集
    test_set = df[(df['date'] > initial_train_end) & (df['date'] <= '2025')].copy()
    # 将测试集按时间截面分组
    test_groups = list(test_set.groupby('date'))
    # 将训练集按时间截面分组
    train_groups = list(current_train.groupby('date'))
    train_losses = []
    test_losses = []
    # 训练循环
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_test_loss = 0
        
        # 随机打乱训练集的时间截面
        random.shuffle(train_groups)
        
        # 按批次训练
        for batch_idx in range(0, len(train_groups)-batch_size, batch_size):
            # 训练集上第N个batch截取（6个截面）
            batch_groups = train_groups[batch_idx:batch_idx + batch_size]
            
            # 合并批次数据
            batch_features = []
            batch_labels = []
            for _, group_df in batch_groups:  #对batch上的每一个截面循环处理
                features = group_df[column].values
                features = pad_to_rows(features, target_rows)
                labels = group_df['ret_o2c_next_month'].values
                labels = pad_to_length(labels, target_rows)
                batch_features.append(features)
                batch_labels.append(labels)
            # print(len(batch_features),len(batch_labels))
            # 转换为Tensor
            features_tensor = torch.tensor(batch_features, dtype=torch.float32)
            labels_tensor = torch.tensor(batch_labels, dtype=torch.float32)
            # print(features_tensor.shape,labels_tensor.shape)
            # 训练步骤
            optimizer.zero_grad()
            loss = overall_utility(model, features_tensor, labels_tensor, gamma=gamma_value)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # 计算测试集损失
        with torch.no_grad():
            for _, group_df in test_groups:
                test_features = torch.tensor(
                    group_df[column].values,dtype=torch.float32).unsqueeze(0)
                test_labels = torch.tensor(group_df['ret_o2c_next_month'].values, dtype=torch.float32).unsqueeze(0)
                test_loss = overall_utility(model, test_features, test_labels, gamma=gamma_value)
                epoch_test_loss += test_loss.item()
        
        # 记录损失
        train_losses.append(epoch_train_loss / len(train_groups ))
        test_losses.append(epoch_test_loss / len(test_groups))
        print(f'Gamma={gamma_value}:Epoch {epoch + 1}, Train Loss: {train_losses[-1]:.8f}, Test Loss: {test_losses[-1]:.8f}')
    print("="*50)
    #============================Training and Prediction Losses============================
    # 在主坐标轴上绘制 loss_train_list 的数据
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Times')
    ax1.set_ylabel('Training Loss', color=color)
    line1 = ax1.plot(train_losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 在副坐标轴上绘制 loss_pred_list 的数据
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Prediction Loss', color=color)
    line2 = ax2.plot(test_losses, color=color, label='Prediction Loss')
    ax2.tick_params(axis='y', labelcolor=color)
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    plt.title(f'Gamma={gamma_value}-Training and Prediction Losses')
    if not os.path.exists('DPPP_daily_result'):
        os.makedirs('DPPP_daily_result')
    save_path2 = os.path.join('DPPP_daily_result', f'Gamma={gamma_value}-Training and Prediction Losses.png')
    plt.savefig(save_path2)
    plt.close()
    # plt.show()
    #===============================================================================

    X_pred = test_set[column].values
    X_pred = torch.tensor(X_pred, dtype=torch.float32)
    with torch.no_grad():  
        raw_weights_pred = model(X_pred)  # 预测原始权重

    weights_pred = raw_weights_pred
    weights_pred_numpy = weights_pred.numpy().flatten()
    df_pred =test_set.copy()
    df_pred.loc[:, 'predicted_weights'] = weights_pred_numpy.astype(np.float32)
    df_pred = df_pred[['code', 'date', 'name', 'weight', 
                        'ret_o2c_next_month', 'predicted_weights']]

    df_pred['predicted_weights'] = df_pred.groupby(
        'date')['predicted_weights'].transform(top_5_weights)
    df_result = pd.concat([df_result,df_pred],axis=0)

    # group_num = 10
    # df_pred['group'] = pd.qcut(
    #     df_pred['predicted_weights'],
    #     q=group_num,  # 修改为10分组
    #     labels=range(1,group_num+1),
    #     duplicates='drop'
    # )

    # # 计算分组统计量（修正agg参数错误）
    # group_stats = df_pred.groupby('group').agg({
    #     'predicted_weights': 'mean',          # 预测因子均值（用于分组排序）
    #     'ret_o2c_next_month': ['mean', 'std', 'count']  # 实际收益率统计量
    # }).reset_index()

    # # 重命名列
    # group_stats.columns = ['group', 'predicted_weights_mean', 'ret_mean', 'ret_std', 'n_samples']

    # # ==================== 可视化 ====================
    # fig, ax1 = plt.subplots(figsize=(12, 6))

    # # 绘制左轴柱状图 (ret_mean)
    # color_bar = "#1f77b4"  # 柱状图颜色（蓝色系）
    # ax1.bar(
    #     group_stats["group"], 
    #     group_stats["ret_mean"], 
    #     color=color_bar, 
    #     alpha=0.7, 
    #     width=0.6, 
    #     label="收益率均值"
    # )
    # ax1.set_xlabel(f'分组 (1=最低预测, {group_num}=最高预测)', fontsize=12)
    # ax1.set_ylabel("收益率均值 (ret_mean)", fontsize=12, color=color_bar)
    # ax1.tick_params(axis="y", labelcolor=color_bar)
    # ax1.grid(axis="y", linestyle="--", alpha=0.5)  # 左轴横向网格线

    # # 创建右轴（共享同一横轴）
    # ax2 = ax1.twinx()

    # # 绘制右轴折线图 (ret_std)
    # color_line = "#d62728"  # 折线图颜色（红色系）
    # ax2.plot(
    #     group_stats["group"], 
    #     group_stats["ret_std"], 
    #     color=color_line, 
    #     marker="o", 
    #     linestyle="--", 
    #     linewidth=2, 
    #     markersize=8, 
    #     label="波动率 (ret_std)"
    # )
    # ax2.set_ylabel("波动率 (标准差)", fontsize=12, color=color_line)
    # ax2.tick_params(axis="y", labelcolor=color_line)
    # ax2.grid(axis="y", linestyle=":", alpha=0.5)  # 右轴横向网格线

    # # 合并图例并调整位置
    # lines1, labels1 = ax1.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    # ax1.legend(
    #     lines1 + lines2, 
    #     labels1 + labels2, 
    #     loc="upper left", 
    #     fontsize=10, 
    #     framealpha=0.9
    # )
    # plt.title("分层检验：收益率均值与波动率", fontsize=14, pad=20)
    # plt.xticks(group_stats["group"])  # 确保分组标签完整显示
    # fig.tight_layout()
    # save_path = os.path.join('DPPP_daily_result', f'Gamma={gamma_value}-分层检验.png')
    # plt.savefig(save_path)
    # plt.close()

    df_result['weight_ret'] = df_result['ret_o2c_next_month']*df_result['predicted_weights']
    df_result['baseline_ret'] = df_result['ret_o2c_next_month']*df_result['weight']
    df_net_value = df_result.groupby('date')[['weight_ret','baseline_ret']].sum()
    df_net_value = df_net_value.reset_index(drop=False)
    df_net_value['premium'] = df_net_value['weight_ret']-df_net_value['baseline_ret']

    # 手续费
    df_net_value['weight_ret'] = df_net_value['weight_ret']-0.00
    df_net_value['net_value'] = (1+df_net_value['weight_ret']).cumprod()
    df_net_value['baseline_net_value'] = (1+df_net_value['baseline_ret']).cumprod()
    df_net_value['premium'] = (1+df_net_value['premium']).cumprod()
    df_net_value['net_value'] = df_net_value['net_value']/df_net_value['net_value'].iloc[0]
    df_net_value['baseline_net_value'] = df_net_value['baseline_net_value']/df_net_value['baseline_net_value'].iloc[0]
    df_net_value['premium'] = df_net_value['premium']/df_net_value['premium'].iloc[0]

    plt.figure(figsize=(10, 6))
    plt.plot(df_net_value['date'], df_net_value['net_value'], label='DPPP', color='red')
    plt.plot(df_net_value['date'], df_net_value['baseline_net_value'], label='Baseline', color='blue')
    plt.plot(df_net_value['date'], df_net_value['premium'], label='Premium', color='orange')
    plt.title(f'Gamma_value:{gamma_value}-Cumulative Net Value Curve')
    plt.xlabel('Date')
    plt.ylabel('Net Value')
    # plt.grid(True)
    plt.legend()
    save_path1 = os.path.join('DPPP_daily_result', f"Gamma={gamma_value}-Net_Value.png")
    plt.savefig(save_path1)
    plt.close()
    # plt.show()

    # 计算投资组合和基准的年化收益、波动率、夏普比率、Calmar比率
    df_net_value['weight_ret'] = df_net_value['weight_ret'].fillna(0)  # 确保没有缺失值
    df_net_value['baseline_ret'] = df_net_value['baseline_ret'].fillna(0)  # 确保没有缺失值

    # 计算投资组合
    df_net_value['weight_ret_utility'] = ((1 + df_net_value['weight_ret'])**(1 - gamma_value)) / (1 - gamma_value)
    weight_ret_average_utility = df_net_value['weight_ret_utility'].mean()
    weight_ret_annualized_return = annualized_return(df_net_value['weight_ret'].values)
    weight_ret_annualized_volatility = annualized_volatility(df_net_value['weight_ret'].values)
    weight_ret_sharpe = sharpe_ratio(df_net_value['weight_ret'].values)
    df_net_value['weight_ret_cumulative_max'] = df_net_value['net_value'].cummax()
    df_net_value['weight_ret_drawdown'] = (df_net_value['weight_ret_cumulative_max'] - df_net_value['net_value'])/df_net_value['weight_ret_cumulative_max']
    weight_ret_max_drawdown = df_net_value['weight_ret_drawdown'].max()
    weight_ret_calmar = calmar_ratio(df_net_value['weight_ret'].values)
    weight_min_weight = df_result['predicted_weights'].min()
    weight_max_weight = df_result['predicted_weights'].max()
    win_ratio = (df_net_value['weight_ret']>df_net_value['baseline_ret']).mean()

    # 计算基准
    df_net_value['baseline_ret_utility'] = ((1 + df_net_value['baseline_ret'])**(1 - gamma_value)) / (1 - gamma_value)
    baseline_ret_average_utility = df_net_value['baseline_ret_utility'].mean()
    baseline_ret_annualized_return = annualized_return(df_net_value['baseline_ret'].values)
    baseline_ret_annualized_volatility = annualized_volatility(df_net_value['baseline_ret'].values)
    baseline_ret_sharpe = sharpe_ratio(df_net_value['baseline_ret'].values)
    df_net_value['baseline_ret_cumulative_max'] = df_net_value['baseline_net_value'].cummax()
    df_net_value['baseline_ret_drawdown'] = (df_net_value['baseline_ret_cumulative_max'] - df_net_value['baseline_net_value'])/df_net_value['baseline_ret_cumulative_max']
    baseline_ret_max_drawdown = df_net_value['baseline_ret_drawdown'].max()
    baseline_ret_calmar = calmar_ratio(df_net_value['baseline_ret'].values)
    baseline_min_weight = df_result['weight'].min()
    baseline_max_weight = df_result['weight'].max()

    #SR-p_value
    t_stat_utility, p_value_utility = stats.ttest_ind(df_net_value['weight_ret_utility'], df_net_value['baseline_ret_utility'])

    #utility-p_value
    t_stat, p_value = stats.ttest_ind(df_net_value['weight_ret'], df_net_value['baseline_ret'])


    performance_name = ['CRRA效用','p_utility[DPPP-VW]',
                        '年化收益','年化波动',
                        '最大回撤','Calmar比率',
                        '最小权重','最大权重','胜率',
                        '夏普比率',
                        'p_SR[DPPP-VW]',]

    strategy_data = [weight_ret_average_utility,p_value_utility,
                        weight_ret_annualized_return,
                        weight_ret_annualized_volatility,
                        weight_ret_max_drawdown,weight_ret_calmar,
                        weight_min_weight,weight_max_weight,win_ratio,
                        weight_ret_sharpe,
                        p_value]

    baseline_data = [baseline_ret_average_utility,np.nan,
                        baseline_ret_annualized_return,
                        baseline_ret_annualized_volatility,
                        baseline_ret_max_drawdown,baseline_ret_calmar,
                        baseline_min_weight,baseline_max_weight,np.nan,
                        baseline_ret_sharpe,
                        np.nan]

    performance_results = {
            f'业绩指标':performance_name,
            f'VW(γ={gamma_value})':baseline_data,
            f'DPPP(γ={gamma_value})':strategy_data
    }

    df_results = pd.DataFrame(performance_results)
    df_performance = pd.concat([df_performance,df_results],axis=1)

# 打印表格
print(df_performance)
print('-'*30)
save_path_performance = os.path.join('DPPP_daily_result', f'performance.xlsx')
df_performance.round(4).to_excel(save_path_performance,index=False)
# df_result.to_excel(fr"C:\Users\陈天洋\Desktop\KPPP\test\Gamma-{gamma_value}-成交明细.xlsx",index=False)

# # 修改 'net_value' 列的列名，添加 γ=gamma
# df_net_value = df_net_value.rename(columns={'net_value': f'γ={gamma_value}'})
# # 将修改后的 DataFrame 添加到列表
# net_value_dfs.append(df_net_value[['date', f'γ={gamma_value}']])
# merged_net_value_df = net_value_dfs[0][['date']]  # 初始化合并的 DataFrame，只保留 'date' 列
# for net_value_df in net_value_dfs:
#     merged_net_value_df = pd.merge(merged_net_value_df, net_value_df, on='date', how='left')

# # 添加 'baseline_net_value' 列，只需从任意 df_net_value 提取一次即可
# merged_net_value_df['baseline_net_value'] = df_net_value['baseline_net_value']

# # 调整列顺序，使 'date' 和 'baseline_net_value' 在前
# merged_net_value_df = merged_net_value_df[['date', 'baseline_net_value'] + [col for col in merged_net_value_df.columns if col not in ['date', 'baseline_net_value']]]

# # 输出结果
# print(merged_net_value_df)
    



