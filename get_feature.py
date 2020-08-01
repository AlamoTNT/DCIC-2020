import numpy as np
import pandas as pd
from gensim.models import Word2Vec

import utils
from utils import get_data, xy2lonlat, geohash_encode, tf_idf

TRAIN_PATH = './data/hy_round1_train_20200102'
TEST_PATH = './data/hy_round1_testA_20200102'

# 调用函数, 读取并整合数据
train = get_data(TRAIN_PATH, 'train')
test = get_data(TEST_PATH, 'test')
train = train.append(test)

train = xy2lonlat(train)
df = train.copy()

# 根据ID和time进行排序
df.sort_values(['ID', 'time'], inplace=True)
# 将时间处理成 2019-11-10 11:38:19的形式，并转成pandas时间格式
# df['time'] = df['time'].apply(lambda x: '2019-'+ x.split(' ')[0][:2]+ '-'+x.split(' ')[0][2:]+' '+x.split(' ')[1])
df['time'] = df['time'].apply(lambda x: '2019-' + x.split(' ')[0][:2] + '-' + x.split(' ')[0][2:] + ' ' + x.split(' ')[1])
df['time'] = pd.to_datetime(df['time'])

# 求每艘船的经度/维度/速度/时间的差分
df['lat_diff'] = df.groupby('ID')['lat'].diff(1)
df['lon_diff'] = df.groupby('ID')['lon'].diff(1)
df['speed_diff'] = df.groupby('ID')['speed'].diff(1)
df['diff_minutes'] = df.groupby('ID')['time'].diff(1).dt.seconds // 60

# 每艘船是否满足: 在10s时间内，经纬度变化量<0.01, 速度<0.1, 如果满足则该船是在锚点处，即停止。
# 使用apply实现, axis=1表示将函数应用与每一行
df['anchor'] = df.apply(lambda x: 1 if x['lat_diff'] < 0.01 and x['lon_diff'] < 0.01 and x['speed'] < 0.1 and x['diff_minutes'] <= 10 else 0, axis=1)

# 处理标签 将标签匹配成数字类别如围网对应0，使用map函数实现
df['type'] = df['type'].map({'围网': 0, '刺网': 1, '拖网': 2, 'unknown': -1})

# 获取锚点位置信息, 先对ID进行分组, 每个ID的anchor求和, 求出每艘船的锚点位置个数
# 使用groupby + agg组合, 分组聚合。返回dataframe
anchor_df = df.groupby('ID')['anchor'].agg('sum').reset_index()
anchor_df.columns = ['ID', 'anchor_cnt']

# 生成group_df, 每个ID对应的label, 以及每个ID出现的总次数
# mean的结果代价label, count等价于每个ID的数据的记录数。
group_df = df.groupby('ID')['type'].agg({'label': 'mean', 'cnt': 'count'}).reset_index()

# 将锚点位置信息合并group_df, 求出每艘船的停止率。停止次数/总次数。
group_df = group_df.merge(anchor_df, on='ID', how='left')
group_df['anchor_ratio'] = group_df['anchor_cnt'] / group_df['cnt']

stat_functions = ['min', 'max', 'mean', 'median', 'nunique', utils.q10, utils.q20, utils.q30, utils.q40, utils.q50, utils.q60, utils.q70, utils.q80, utils.q90]
stat_ways = ['min', 'max', 'mean', 'median', 'nunique', 'q_10', 'q_20', 'q_30', 'q_40', 'q_50', 'q_60', 'q_70', 'q_80', 'q_90']
stat_cols = ['lat', 'lon', 'speed', 'direction']

# 所有数据'lat', 'lon', 'speed', 'direction'统计特征
group_tmp = df.groupby('ID')[stat_cols].agg(stat_functions).reset_index()
group_tmp.columns = ['ID'] + ['{}_{}'.format(i,j) for i in stat_cols for j in stat_ways]

lat_lon_neq_zero = df[(df['lat_diff'] != 0) & (df['lon_diff'] != 0)]
speed_neq_zero = df[df['speed_diff'] != 0]

# 经纬度变化量不等于0的数据的'lat', 'lon', 'speed', 'direction'的统计特征
lat_lon_neq_group = lat_lon_neq_zero.groupby('ID')[stat_cols].agg(stat_functions).reset_index()
lat_lon_neq_group.columns = ['ID'] + ['pos_neq_zero_{}_{}'.format(i, j) for i in stat_cols for j in stat_ways]

# 速度不为0的数据'lat', 'lon', 'speed', 'direction'的统计特征
speed_neq_zero_group = speed_neq_zero.groupby('ID')[stat_cols].agg(stat_functions).reset_index()
speed_neq_zero_group.columns = ['ID'] + ['speed_neq_zero_group_{}_{}'.format(i,j) for i in stat_cols for j in stat_ways]

# 将这些统计特征合并到group_df中
group_df = group_df.merge(group_tmp, on='ID', how='left')
group_df = group_df.merge(lat_lon_neq_group, on='ID', how='left')
group_df = group_df.merge(speed_neq_zero_group, on='ID', how='left')

# 计算位置出现的次数
mode_df = df.groupby(['ID', 'lat', 'lon'])['time'].agg({'mode_cut': 'count'}).reset_index()
# 对ID分组，根据mode_cut进行排序
mode_df['rank'] = mode_df.groupby('ID')['mode_cut'].rank(method='first', ascending=False)
# 循环取出出现次数排名前三个的位置,并将其合并到group_df中
for i in range(1, 4):
    tmp_df = mode_df[mode_df['rank'] == i]
    del tmp_df['rank']
    tmp_df.columns = ['ID', 'rank_{}_mode_lat'.format(i), 'rank_{}_mode_lon'.format(i), 'rank{}_mode_cnt'.format(i)]
    group_df = group_df.merge(tmp_df, on='ID', how='left')

df['lat_lon'] = df.apply(lambda x: geohash_encode(x['lat'], x['lon'], 7), axis=1)

# 将每一个ID对应的所有的lat_lon 存到list中, ID [xvkwu60, xvkwu60, xvkwu60, xvkwu60, vkwu60, xvkwu6]
tmp = df.groupby('ID')['lat_lon'].agg(list).reset_index()  # agg聚合的结果是一个值(一个分组一个值)。
# list 转字符串 并以空格分隔, ID xvkwu60 xvkwu60 xvkwu60 xvkwu60 xvkwu60 xvkwu6
tmp['lat_lon'] = tmp['lat_lon'].apply(lambda x: ' '.join(x))
print('经纬度文本化的结果: ', tmp)

# agg聚合的结果是一个值(一个分组一个值), 所以下面代码执行返回一个ID, 以及ID对应的lat的梯度list
# 0	 [0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, ..
grad_df = df.groupby('ID')['lat'].apply(lambda x: np.gradient(x)).reset_index()
grad_df['lon'] = df.groupby('ID')['lon'].apply(lambda x: np.gradient(x)).reset_index()['lon']
grad_df['lat'] = grad_df['lat'].apply(lambda x: np.round(x, 4))
grad_df['lon'] = grad_df['lon'].apply(lambda x: np.round(x, 4))

# 注意下面这行format zip用法, 每一时刻的经纬度的梯度会一一匹配。
# 最后结果 -0.0058_0.0034 -0.0058_0.0034 -0.0047_0.0044 -... 经度梯度_维度梯度
grad_df['grad'] = grad_df.apply( lambda x: ' '.join(['{}_{}'.format(z[0], z[1]) for z in zip(x['lat'], x['lon'])]), axis=1)
print(grad_df['grad'])

tfidf_tmp = tf_idf(grad_df['grad'], 30, 'grad')
grad_tfidf = pd.concat([grad_df[['ID']], tfidf_tmp], axis=1)
group_df = group_df.merge(grad_tfidf, on='ID', how='left')
print('gradient tfidf finished.')

tmp = df.groupby('ID')['lat_lon'].apply(lambda x: x.sample(frac=0.1, random_state=1)).reset_index()
del tmp['level_1']
tmp.columns = ['ID', 'sample']
tmp = tmp.groupby('ID')['sample'].agg(list).reset_index()
tmp['sample'] = tmp['sample'].apply(lambda x: ' '.join(x))

tfidf_tmp = tf_idf(tmp['sample'], 30, 'sample')
sample_tfidf= pd.concat([tmp[['ID']], tfidf_tmp], axis=1)
group_df = group_df.merge(sample_tfidf, on='ID', how='left')
print('sample tfidf finished.')

# 训练 word2vec模型
data_frame = df.groupby('ID')['lat_lon'].agg(list).reset_index()
model = Word2Vec(data_frame['lat_lon'].values, size=30, window=5, min_count=1, sg=1, hs=1, workers=1, iter=10, seed=1,
                 hashfxn=utils.hashfxn)

# data_frame['lat_lon']中的每一个值，都是一个(n,30)的dataframe, n是指渔船的所有轨迹的个数。
# 30 是指每个轨迹被嵌入到30维。

data_frame['lat_lon'] = data_frame['lat_lon'].apply(lambda x: pd.DataFrame([model[c] for c in x]))
# 对每个ID，计算lat_lon的dataframe的30个特征的每一个特征均值，作为最终结果。
for m in range(30):
    data_frame['w2v_{}_mean'.format(m)] = data_frame['lat_lon'].apply(lambda x: x[m].mean())

del data_frame['lat_lon']
w2v_df = data_frame

group_df = group_df.merge(w2v_df, on='ID', how='left')
print('word2vec finished.')

group_df.to_csv('feature.csv')