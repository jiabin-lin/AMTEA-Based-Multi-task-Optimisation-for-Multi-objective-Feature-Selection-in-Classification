#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write the statistical results on the training datasets into a table.
"""

from train_hv_statistic import train_hv_statistic
import pandas as pd

df_wine = train_hv_statistic('wine', 'Wine')

df_monks = train_hv_statistic('monks', 'Monks')

df_mushroom = train_hv_statistic('mushroom', 'Mushroom')

df_magic = train_hv_statistic('magic', 'Magic')

df_letter = train_hv_statistic('letter', 'Letter')

df_dermatology = train_hv_statistic('dermatology', 'Dermatology')

df_waveform = train_hv_statistic('waveform', 'Waveform')

# df_news = train_hv_statistic('news', 'News')

df = [df_wine, df_monks, df_mushroom, df_magic, df_letter, df_dermatology, df_waveform]

res = pd.concat(df)


res_latex = res.to_latex(index=False)
print(res_latex)

s = res.style.highlight_max(axis=1,
    props='textbf:--rwrap;'
)