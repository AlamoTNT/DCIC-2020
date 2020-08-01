import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
warnings.filterwarnings('ignore')


def f1_score_eval(preds, valid_df):
    labels = valid_df.get_label()
    preds = np.argmax(preds.reshape(3, -1), axis=0)
    scores = f1_score(y_true=labels, y_pred=preds, average='macro')
    return 'macro_f1_score', scores, True


def build_model(train_, test_, pred, label, cate_cols, split, seed=1024, is_shuffle=True, use_cart=False):
    n_class = 3
    train_pred = np.zeros((train_.shape[0], n_class))
    test_pred = np.zeros((test_.shape[0], n_class))
    n_splits = 5

    assert split in ['kf', 'skf'], '{} Not Support this type of split way'.format(split)

    if split == 'kf':
        folds = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
        kf_way = folds.split(train_[pred])
    else:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
        kf_way = folds.split(train_[pred], train_[label])

    print('Use {} features ...'.format(len(pred)))

    params = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'None',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'num_class': n_class,
        'nthread': 8,
        'verbose': -1,
    }
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        print('the {} training start ...'.format(n_fold))
        train_x, train_y = train_[pred].iloc[train_idx], train_[label].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_[label].iloc[valid_idx]

        if use_cart:
            dtrain = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols)
            dvalid = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols)
        else:
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            verbose_eval=100,
            feval=f1_score_eval
        )
        train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        test_pred += clf.predict(test_[pred], num_iteration=clf.best_iteration)/folds.n_splits
    print(classification_report(train_[label], np.argmax(train_pred, axis=1), digits=4))

    test_['label'] = np.argmax(test_pred, axis=1)
    return test_[['ID', 'label']]


if __name__ == '__main__':
    whole_df = pd.read_csv('feature.csv', index_col=[0])  # 不添加index_col, 特征集中包含'Unnamed: 0', 训练时会报错
    train = whole_df[whole_df['label'] != -1]
    test = whole_df[whole_df['label'] == -1]
    use_feats = [c for c in train.columns if c not in ['ID', 'label']]
    sub = build_model(train, test, use_feats, 'label', [], 'kf', is_shuffle=True, use_cart=False)
    sub['label'] = sub['label'].map({0: '围网', 1: '刺网', 2: '拖网'})
    sub.to_csv('submission.csv', encoding='utf-8', header=None, index=False)