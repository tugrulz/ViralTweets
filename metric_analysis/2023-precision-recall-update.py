import pandas as pd
from scipy.stats import hmean

df = pd.read_csv('merged_outputs.csv')
df['recall'] = df['tpr']
df['tp'] = df['tpr']*1008
df['fp'] = df['new_tweets'] - df['tp']
df['precision'] = df['tp'] / (df['tp'] + df['fp'])

df['f1'] = hmean(df[['precision', 'recall']], axis=1)
df['f2'] = 5 * (df['precision'] * df['recall']) / ((4 * df['precision'] ) +  df['recall'])
df['f3'] = 10 * (df['precision'] * df['recall']) / ((9 * df['precision'] ) +  df['recall'])
df['f4'] = 17 * (df['precision'] * df['recall']) / ((16 * df['precision'] ) +  df['recall'])
df['f5'] = 26 * (df['precision'] * df['recall']) / ((25 * df['precision'] ) +  df['recall'])

# df['f1'] = harmonic_mean([df['precision'], df['recall']])
metric_names = {
    'hard_threshold_viral_covered_vs_new_tweets_labeled' : 'RT > T',
    'virality_avg_retweets_viral_covered_vs_new_tweets_labeled' : 'RT > Avg. RT',
    'log_retweets_over_log_followers_viral_covered_vs_new_tweets_labeled' : 'log(RT / Followers)',
    'virality_median_retweets_viral_covered_vs_new_tweets_labeled 2': 'RT > Med. RT',
    'retweets_over_log_followers_viral_covered_vs_new_tweets_labeled': 'RT / log(Followers)',
    'roberta_paper_metric_viral_covered_vs_new_tweets_labeled': 'Influence Score',
    'virality_followers_viral_covered_vs_new_tweets_labeled': 'RT / Followers',
    'log_retweets_over_followers_viral_covered_vs_new_tweets_labeled': 'log(RT) / Followers',
    'virality_median_retweets_viral_covered_vs_new_tweets_labeled': 'unused',
    'virality_retweet_percentile_per_user_viral_covered_vs_new_tweets_labeled': 'RT Percentile'
}

df['metric_name'] = '?'
for key, name in metric_names.items():
    df.loc[df.metric == key, 'metric_name'] = name

df.to_csv('all_metric_stats.csv', index = False)
print()