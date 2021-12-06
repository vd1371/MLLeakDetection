#Loading dependencies
import logging
import sys


class Logger(object):

	def __init__(self, logger_name = 'Logger', address = '',
				 level = logging.DEBUG, console_level = logging.ERROR,
				 file_level = logging.DEBUG, mode = 'w'):
		super(Logger, self).__init__()

		logging.basicConfig()
		
		self.instance = logging.getLogger(logger_name)
		self.instance.setLevel(level)
		self.instance.propagate = False

		console_handler = logging.StreamHandler(sys.stdout)
		console_handler.setLevel(console_level)
		self.instance.addHandler(console_handler)
		
		file_handler = logging.FileHandler(address, mode = mode)
		file_handler.setLevel(file_level)
		formatter = logging.Formatter('%(asctime)s-%(levelname)s- %(message)s')
		file_handler.setFormatter(formatter)
		self.instance.addHandler(file_handler)
	
	def _correct_message(self, message):
		output = "\n---------------------------------------------------------\n"
		output += message
		output += "\n---------------------------------------------------------\n"
		return output
		
	def debug(self, message):
		self.instance.debug(self._correct_message(message))

	def info(self, message):
		self.instance.info(self._correct_message(message))

	def warning(self, message):
		self.instance.warning(self._correct_message(message))

	def error(self, message):
		self.instance.error(self._correct_message(message))

	def critical(self, message):
		self.instance.critical(self._correct_message(message))

	def exception(self, message):
		self.instance.exception(self._correct_message(message))