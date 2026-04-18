import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# グラフのスタイル設定
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# 1990年から現在までのS&P500とVIXの日次データを取得
print("データを取得中...")
vix = yf.download('^VIX', start='1990-01-01', end='2024-01-01')['Close']
sp500 = yf.download('^GSPC', start='1990-01-01', end='2024-01-01')['Close']

# データフレームの結合
df = pd.concat([vix, sp500], axis=1)
df.columns = ['VIX', 'SP500']
print("データ取得完了。データサイズ:", df.shape)