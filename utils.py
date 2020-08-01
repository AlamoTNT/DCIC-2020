import os

import pandas as pd
from tqdm import tqdm
from pyproj import Proj
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


def get_data(file_path, model):
    file_path = file_path
    paths = os.listdir(file_path)  # 得到file_path目录下的文件名，存入数组paths中
    tmp = []
    for t in tqdm(range(len(paths))):
        p = paths[t]
        with open('{}/{}'.format(file_path, p), encoding='utf-8') as f:
            next(f)
            for line in f.readlines():
                # strip()去除字符串的头和尾的空格,并以,切割字符串，每一行数据形成一个字符数组。将字符数组添加到tmp数组中。
                tmp.append(line.strip().split(','))
    tmp_df = pd.DataFrame(tmp)
    print(tmp_df.head())
    if model == 'train':
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    else:
        tmp_df['type'] = 'unknown'
        tmp_df.columns = ['ID', 'lat', 'lon', 'speed', 'direction', 'time', 'type']
    tmp_df['lat'] = tmp_df['lat'].astype(float)
    tmp_df['lon'] = tmp_df['lon'].astype(float)
    tmp_df['speed'] = tmp_df['speed'].astype(float)
    tmp_df['direction'] = tmp_df['direction'].astype(int)
    return tmp_df


# 将原始数据得x,y 转为经纬度。
def xy2lonlat(df):
    x = df['lat'].values
    y = df['lon'].values
    p = Proj(
        "+proj=tmerc +lat_0=0 +lon_0=120 +k=1 +x_0=500000 +y_0=0 +a=6378140 +b=6356755.288157528 +units=m +no_defs")
    df['lon'], df['lat'] = p(x, y, inverse=True)
    return df


def geohash_encode(latitude, longitude, precision=12):
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)


def tf_idf(input_values, output_num, output_prefix, seed=1024):
    # 生成文本Tfidf矩阵
    tfidf_enc = TfidfVectorizer()
    tfidf_vec = tfidf_enc.fit_transform(input_values)
    # print('所有文本的关键字', tfidf_enc.get_feature_names())

    # print('词频矩阵的结果', tfidf_vec.toarray())

    # truncatedSVD作用于sklearn.feature_extraction.text向量化后返回的 term count/tf-idf矩阵。在这种情况下又被成为LSA（ latent semantic analysis）
    # LSA潜在语义分析，生成主题数为30维，
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(tfidf_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_tfidf_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


def count2vec(input_values, output_num, output_prefix, seed=1024):
    # CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵
    # 生成文本词频矩阵
    count_enc = CountVectorizer()
    count_vec = count_enc.fit_transform(input_values)
    # truncatedSVD作用于sklearn.feature_extraction.text向量化后返回的 term count/tf-idf矩阵。在这种情况下又被成为LSA（ latent semantic analysis）
    # LSA潜在语义分析，生成主题数为30维，
    svd_tmp = TruncatedSVD(n_components=output_num, n_iter=20, random_state=seed)
    svd_tmp = svd_tmp.fit_transform(count_vec)
    svd_tmp = pd.DataFrame(svd_tmp)
    svd_tmp.columns = ['{}_countvec_{}'.format(output_prefix, i) for i in range(output_num)]
    return svd_tmp


# 定义一些函数，返回数据的分位点数据
def q10(x):
    return x.quantile(0.1)


def q20(x):
    return x.quantile(0.2)


def q30(x):
    return x.quantile(0.3)


def q40(x):
    return x.quantile(0.4)


def q50(x):
    return x.quantile(0.5)


def q60(x):
    return x.quantile(0.6)


def q70(x):
    return x.quantile(0.7)


def q80(x):
    return x.quantile(0.8)


def q90(x):
    return x.quantile(0.9)


def hashfxn(astring):
    return ord(astring[0])


if __name__ == '__main__':
    pass
