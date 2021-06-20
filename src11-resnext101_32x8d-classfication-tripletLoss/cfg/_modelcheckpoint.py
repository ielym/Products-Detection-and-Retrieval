import torch
from ._history import ParaMeter

class SingleModelCheckPoint:

	def __init__(self, filepath, monitor, verbose, save_best_only, mode, save_weights_only):
		self.filepath = filepath
		self.monitor = monitor
		self.verbose = verbose
		self.save_best_only = save_best_only
		self.mode = mode
		self.save_weights_only = save_weights_only

		self.monitor_val = float('-inf') if mode == 'max' else float('+inf')

	def _savemodel(self, model, file_path):
		if self.save_weights_only == True:
			torch.save(model.state_dict(), file_path)
		elif self.save_weights_only == False:
			torch.save(model,file_path)

	def savemodel(self, model, **kwargs):
		for k, v in kwargs.items():
			if isinstance(kwargs[k], ParaMeter):
				kwargs[k] = v.avg
		if self.save_best_only == False:
			if kwargs['epoch'] % self.verbose == 0 or kwargs['epoch'] == 1:
				self._savemodel(model, self.filepath.format(**kwargs))
				print('Save model to {}'.format(self.filepath.format(**kwargs)))
		else:
			val = kwargs[self.monitor]
			if self.mode == 'max':
				if val >= self.monitor_val:
					self._savemodel(model, self.filepath.format(**kwargs))
					print('{} improve from {} to {}\nSave model to {}'.format(self.monitor, self.monitor_val, val,self.filepath.format(**kwargs)))
					self.monitor_val = val
				else:
					print('{} did not improve'.format(self.monitor))
			else:
				if val <= self.monitor_val:
					self._savemodel(model, self.filepath.format(**kwargs))
					print('{} improve from {} to {}\nSave model to {}'.format(self.monitor, self.monitor_val, val,self.filepath.format(**kwargs)))
					self.monitor_val = val
				else:
					print('{} did not improve'.format(self.monitor))


