import pickle
import pandas as pd
import functools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

with open('/home/PJLAB/chenyun/ServiceComposition/simulator/dataset/qws2/qws2.txt', 'r') as f:
    data = f.readlines()
    
service_num = len(data)
columns=['Response Time', 'Availability', 'Throughput', 'Successability', 'Reliability',\
    'Compliance', 'Best Practices', 'Latency', 'Documentation', 'Service Name', 'WSDL Address']
df = pd.DataFrame(columns=columns)


for i in range(service_num):
    row = data[i].split(',')
    name = row[9]
    wsdl = ''.join(row[10:])
    row = [float(k) for k in row[:9]]
    row[1] /= 100
    row[3] /= 100
    row[4] /= 100
    row.append(name)
    row.append(wsdl)
    df.loc[len(df.index)] = row

with open('/home/PJLAB/chenyun/ServiceComposition/simulator/dataset/qws2/qws_df.pkl', 'wb') as f:
    pickle.dump(df, f)

for column in columns[:9]:
    print(df[column].describe())
    plt.hist(df[column], bins=100)
    plt.savefig(f'/home/PJLAB/chenyun/ServiceComposition/simulator/dataset/qws2/statistic_fig/{column}_hist.png')
    plt.close()
    
    sns.kdeplot(df[column], shade=True)
    plt.savefig(f'/home/PJLAB/chenyun/ServiceComposition/simulator/dataset/qws2/statistic_fig/{column}_kde.png')
    plt.close()
    
