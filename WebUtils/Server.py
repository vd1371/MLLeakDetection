import socket
import tqdm
import os
import time

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from urllib.parse import parse_qs

class DataSender(BaseHTTPRequestHandler):

	def _set_response(self):
		self.send_response(200)
		self.send_header('Content-type', 'text/html')
		self.end_headers()

	def do_GET(self):
		self._set_response()

		try:
			query_components = parse_qs(urlparse(self.path).query)
			batch_number = int(query_components['batch_number'][0])

			with open(f"Data/LeakLocs-{batch_number}.csv", "rb") as f:
				self.wfile.write(f.read())

		except:
			self.wfile.write(b"NotFound")

def run_server(server_class=HTTPServer, handler_class=DataSender, addr="127.0.0.1", port=8000):

	server_address = (addr, port)
	httpd = server_class(server_address, handler_class)

	print(f"Starting httpd server on {addr}:{port}")
	httpd.serve_forever()


if __name__ == "__main__":

	run()