from ._encode_solution import _encode_solution
from ._decode_solution import _decode_solution

class Solution:

	def __init__(self, value = 0, solution = None, flag = None, **params):
		self.value = value
		self.solution = solution
		self.flag = flag

	def encode(self, original_form, **params):
		return _encode_solution(original_form, **params)

	def decode(self, **params):
		return _decode_solution(self.solution, **params)

	def set_value(self, val):
		self.value = val

	def __str__(self):
		return f"{self.flag} - Value: {self.value}"
