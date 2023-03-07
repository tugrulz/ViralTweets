import pandas as pd
from glob import glob
from sklearn import metrics
from statistics import harmonic_mean



files = glob('output_original/*.csv')
theoretical = 1357228

dfs = []

for file in files:
    filename = file.split('/')[-1]
    df = pd.read_csv(file)
    df.columns = ['tpr', 'new_tweets', 'threshold']
    df['fpr'] = df['new_tweets'] / df['new_tweets'].max()
    df['fpr2'] = df['new_tweets'] / theoretical
    df = df.sort_values(by = ['tpr', 'new_tweets'])
    df = df.drop_duplicates(subset = ['tpr'], keep = 'first')
    df.to_csv('output_standardized/%s' % filename, index = False)
    df['metric'] = filename.split('.csv')[0]
    roc1 = metrics.auc(df['fpr'], df['tpr'])
    roc2 = metrics.auc(df['fpr2'], df['tpr'])
    df['roc1'] = roc1
    df['roc2'] = roc2

    #roc3
    df95 = df.copy()
    df95 = df95[df95.fpr2 <= 0.016]
    df95['fpr2'] = df95['fpr2']*(1/0.016)
    tprmax = df95.tpr.max()
    if(tprmax < 1):
        fpr2_max = df95.fpr2.max()
        multipli = 1/fpr2_max
        tpr_interpolated = tprmax*multipli

    tpr = df95['tpr']
    fpr = df95['fpr2']
    tpr.loc[-1] = tpr_interpolated
    fpr.loc[-1] = 1

    roc95 = metrics.auc(fpr, tpr)

    df['roc95'] = roc95
    df['fpr3'] = df.fpr2*(1/0.016)
    df['harmonic'] = harmonic_mean([roc95,roc1])
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv('merged_outputs.csv', index = False)


