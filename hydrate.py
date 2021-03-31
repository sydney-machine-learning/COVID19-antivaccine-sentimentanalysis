import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#keep only the tweet IDs before hydrating the tweets

rawli = ['corona_tweets_156.csv']
#'corona_tweets_107.csv', 'corona_tweets_114.csv', 'corona_tweets_121.csv', 'corona_tweets_128.csv', 'corona_tweets_135.csv', 'corona_tweets_142.csv', 'corona_tweets_149.csv', 'corona_tweets_156.csv', 'corona_tweets_163.csv', 'corona_tweets_170.csv', 'corona_tweets_177.csv', 'corona_tweets_184.csv', 'corona_tweets_191.csv', 'corona_tweets_198.csv', 'corona_tweets_205.csv', 'corona_tweets_212.csv', 'corona_tweets_219.csv', 'corona_tweets_226.csv', 'corona_tweets_233.csv', 'corona_tweets_240.csv', 'corona_tweets_247.csv', 'corona_tweets_254.csv', 'corona_tweets_261.csv', 'corona_tweets_268.csv', 'corona_tweets_275.csv', 'corona_tweets_282.csv', 'corona_tweets_289.csv', 'corona_tweets_296.csv', 'corona_tweets_303.csv', 'corona_tweets_310.csv', 'corona_tweets_317.csv'] #list of csv file titles
df_list = [] #list of dfs for raw csvs from the IEEE source
df_master = pd.DataFrame() #empty master dataframe
for raw in rawli:
    df_raw = pd.read_csv(r'/Users/cathy/Desktop/no geo tag raw data from IEEE/'+raw, names=['id','sentiment_score'])
    df_master = pd.concat([df_master,df_raw], axis=0)

df_ID = df_master['id']
df_ID.to_csv(r'/Users/cathy/Desktop/ID only 156.csv', index=False, header=None)
#df_master.to_csv(r'/Users/cathy/Desktop/master 30-100.csv', index=False, header=None)
print(df_ID.shape)


# #concatenate the hydrated files into a master_hydrated file
# hydrateli = ['hydrate March.csv', 'hydrate April.csv','hydrate May.csv','hydrate June.csv','hydrate July.csv','hydrate August.csv','hydrate September.csv','hydrate October.csv','hydrate November.csv','hydrate December.csv','hydrate January.csv']
# df_master_hydrated = pd.DataFrame()
# for hydrated in hydrateli:
#     df_hydrate = pd.read_csv(r'/Users/cathy/Desktop/'+hydrated, low_memory=False, header = 0)
#     df_master_hydrated = pd.concat([df_master_hydrated, df_hydrate], axis=0)
# #df_master_hydrated.to_csv(r'/Users/cathy/Desktop/master hydrated Mar to Jan.csv', index=False, header=None)
#
#
# df_merge = df_master_hydrated.merge(df_master, on='id', how='left')
# print (df_merge.shape)
#
# count = df_merge['text'].str.contains("vaccine").sum()
# print(count)
#
# vaccine_df = df_merge[df_merge['text'].str.contains("vaccine")]
# vaccine_df[['Longitude','Latitude']] = vaccine_df['coordinates'].str.split(',',expand=True)
# vaccine_df.to_csv(r'/Users/cathy/Desktop/vaccine dataset.csv', index=False, header = 0)
# #print(list(vaccine_df['text']))
