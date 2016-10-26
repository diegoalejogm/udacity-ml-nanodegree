import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


img_path = './results/img/'

data = pd.read_csv('results2.csv', sep='\t', index_col=0)
df = None

for col in list(data.columns.values):
    if col != 'alpha' and col != 'epsilon' and col != 'gamma' and col != 'eps_iters':

        list_ascending = ['avg_perc_to_deadline', 'avg_perc_to_deadline_last_10']
        ascending = True if col in list_ascending else False

        top5 = data.sort_values(col, ascending=ascending).head(7)

        if df is None:
            df = top5
        else:
            df= pd.concat([df,top5])


df_tuners = df.copy().drop_duplicates().sort_index()

df_tuners = df_tuners[['alpha', 'epsilon', 'gamma']]
sns.heatmap(df_tuners, annot=True, fmt="f", linewidths=.5)
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
