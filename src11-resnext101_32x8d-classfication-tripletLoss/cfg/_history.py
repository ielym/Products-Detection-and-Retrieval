
class ParaMeter(object):
	def __init__(self, name, format=':f'):
		self.name = name
		self.format = format
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		formatstr = '{name} {val' + self.format + '} ({avg' + self.format + '})'
		return formatstr.format(**self.__dict__)

class History(object):
	def __init__(self):
		self.history_dict = {
			'BatchTime':ParaMeter('BatchTime', ':6.4f'),
			'DataTime':ParaMeter('DataTime', ':6.4f'),
		}

	def register(self, name, format):
		self.history_dict[name] = ParaMeter(name, format)

	def update(self, n, name, val):
		self.history_dict[name].update(val=val, n=n)

	def reset(self):
		for v in self.history_dict.values():
			v.reset()

	def display(self,epoch, batch=0, totalbatch=0, mode='train'):
		if mode == 'train':
			entries = ["[Epoch:{}] {}/{}\t".format(epoch, batch, totalbatch)]
			for k, v in self.history_dict.items():
				if not k.startswith('val_'):
					formatstr = '{} {' + v.format + '}'
					entries.append(formatstr.format(v.name, v.val))
			print('    '.join(entries))
		elif mode == 'validate':
			entries = ["[Epoch:{}] {:6.2f}".format(epoch, self.history_dict['BatchTime'].sum)]
			for k, v in self.history_dict.items():
				if k=='BatchTime' or k=='DataTime' :
					continue
				formatstr = '{} {' + v.format + '}'
				entries.append(formatstr.format(v.name, v.avg))
			print('    '.join(entries))