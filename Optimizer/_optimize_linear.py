from ._GA_tools import solve_sample_with_GA
from ._convert_df_to_signals import _convert_df_to_signals

def _optimize_linear(data, params, q_out = None):

	signals, leak_in_sections, leak_info = data

	signals = _convert_df_to_signals(signals, **params)

	holder = []
	for i, idx in enumerate(leak_info.index):
		sample_signal = signals[i]
		sample_leak_info = leak_info.loc[idx, :].values
		sample_leak_in_sections = leak_in_sections.loc[idx, :].values

		solution = solve_sample_with_GA(sample_leak_info,
										sample_signal,
										**params)

		holder.append([sample_leak_info, sample_leak_in_sections, solution])

	if q_out == None:
		return holder
	else:
		q_out.put(holder)