import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ._get_data import _load_all_offline_data
from ._split_and_normalize_data import split_and_normalize_data

def leaks_distribution_hist(fig_num, **params):

	print ("Trying to leaks_distribution_hist")

	fig_height = params.get("fig_height")
	fig_width = params.get("fig_width")
	font_size = params.get("font_size")
	ticks_font_size = params.get("ticks_font_size")
	num_hist_bins = params.get("num_hist_bins")
	color = params.get("color")
	dpi = params.get("dpi")
	leak_pred = params.get("leak_pred")



	_, Y, _ = _load_all_offline_data(**params)


	plt.rcParams["font.family"] = "Times New Roman"	
	plt.clf()
	plt.figure(figsize= (fig_width, fig_height))
	plt.xticks(fontsize = ticks_font_size)
	plt.yticks(np.arange(0,2001,500), fontsize = ticks_font_size)
	plt.xlabel('Leaks in Section 2', fontsize = font_size)
	plt.ylabel('Frequency', fontsize = font_size)
	plt.ylim([0,2000])
	
	dict_map = {1:'With Leak' , 0:'Without Leak'}
	Y['LL1'] = Y['LL1'].replace(dict_map)

	ax = sns.countplot(x=Y['LL1'], color='green')
	ax.set(xlabel='Leaks in Section 2', ylabel='Frequency')
	ax.bar_label(ax.containers[0])
	plt.savefig(f"Fig{fig_num}-R1C1.tif", dpi=dpi, bbox_inches="tight")