import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


data = pd.read_csv('results.csv', sep='\t', index_col=0)
rank = data.copy().drop('alpha', 1).drop('epsilon', 1).drop('gamma', 1)#.drop('scores', 1).drop('scores_last_10', 1)

rank_desc = rank.copy()[['successes', 'successes_last_10']].rank(ascending=False)
print rank_desc
rank_asc = rank.copy().drop('successes',1).drop('successes_last_10', 1).rank(ascending=True)
print rank_asc
rank['rank'] = rank_asc.join(rank_desc).mean(axis=1)
print rank.sort_values('rank', ascending=True).head(10)


df = None


for col in list(data.columns.values):
    if col != 'alpha' and col != 'epsilon' and col != 'gamma' and col != 'eps_iters':

        list_ascending = ['perc_to_deadline', 'perc_to_deadline_last_10', 'penalties', 'penalties_last_10','steps', 'steps_last_10']
        ascending = True if col in list_ascending else False

        top5 = data.sort_values(col, ascending=ascending).head(10)

        if df is None:
            df = top5
        else:
            df= pd.concat([df,top5])


df_tuners = df.copy().drop_duplicates().sort_index()

df_tuners = df_tuners[['alpha', 'epsilon', 'gamma']]
sns.heatmap(df_tuners, fmt="f", linewidths=.5)
plt.yticks(rotation=0)
plt.show()

# Find duplicates
df_dup = df.copy()
df_dup['dup'] = df_dup.groupby(df.index.values,as_index=False).size()
sns_plot = sns.barplot(df_dup.index, y='dup', data=df_dup, palette="Blues_d", saturation=.5)
plt.show()

for col in list(data.columns.values):
    if col != 'alpha' and col != 'epsilon' and col != 'gamma' and col != 'eps_iters':
        print col
        sns_plot = sns.barplot(df.index, y=col, data=df, palette="Blues_d", saturation=.5)
        plt.show()
