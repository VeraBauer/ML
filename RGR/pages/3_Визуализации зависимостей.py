import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(layout="wide")

st.write("# Визуализации зависимостей")

st.sidebar.header("Визуализации")

data = pd.read_csv('..\data\smoke_detector.csv')
data.drop(data.columns[[0]], axis=1, inplace=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.set_facecolor('gainsboro')

data['Fire Alarm'].value_counts().plot(kind ='pie',  ylabel='Fire Alarm', ax=axes[0][0], autopct = '%.2f', cmap='GnBu') 

sns.histplot(ax=axes[0][1], data=data['Temperature[C]'])

corr = data.corr()
sns.heatmap(corr, ax=axes[1][0], xticklabels=corr.columns, yticklabels=corr.columns, annot_kws={"size":10}, cmap='GnBu')

sns.kdeplot( data=data, x='Temperature[C]', hue='Fire Alarm', ax=axes[1][1], palette='GnBu')
plt.legend(loc='upper left', labels=['no fire', 'yes fire'])

#axes[1][1].boxplot(x=data['Raw Ethanol'])

axes[0][0].set_title('Круговая диаграмма распределения случаев пожара')
axes[0][1].set_title('Гистограмма распределения температуры')
axes[1][0].set_title('Тепловая карта')
axes[1][1].set_title('Temperature vs fire density')
st.pyplot(fig)
