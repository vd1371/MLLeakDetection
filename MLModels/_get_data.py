from ._get_data_offline import get_data_offline
from ._get_data_online import get_data_online

def get_data(**kwargs):
	method = kwargs.get("method")

	if method == 'offline':
		return get_data_offline(**kwargs)

	elif method == "online":
		return get_data_online(**kwargs)