#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Write the statistical results on the test datasets into a table.
"""


from test_hv_statistic import test_hv_statistic
import pandas as pd

df_wine = test_hv_statistic('wine', 'Wine')

df_monks = test_hv_statistic('monks', 'Monks')

df_mushroom = test_hv_statistic('mushroom', 'Mushroom')

df_magic = test_hv_statistic('magic', 'Magic')

df_letter = test_hv_statistic('letter', 'Letter')

df_dermatology = test_hv_statistic('dermatology', 'Dermatology')

df_waveform = test_hv_statistic('waveform', 'Waveform')

# df_news = train_hv_statistic('news', 'News')

df = [df_wine, df_monks, df_mushroom, df_magic, df_letter, df_dermatology, df_waveform]

res = pd.concat(df)


res_latex = res.to_latex(index=False)
print(res_latex)

s = res.style.highlight_max(axis=1,
    props='textbf:--rwrap;'
)