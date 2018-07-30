import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import collections
import time
import pickle

_since_beginning = collections.defaultdict(lambda: {})
_since_last_flush = collections.defaultdict(lambda: {})

_iter = [0]
def tick():
	_iter[0] += 1

def plot(name, value):
	_since_last_flush[name][_iter[0]] = value

def flush():
	prints = []
	# print(_since_last_flush)
	# print('>>>>>>>>>>>>>>>')
	for name, vals in _since_last_flush.items():
		#print('>>>>>>>>>>>>>>>')
		#print(vals.values())
		#print(list(vals.values()))
		#print(vals.values()[0][0])
		if 'time' not in name and 'dev disc cost' not in name and 'inception score' not in name:
			loss = [x[0] for x in list(vals.values())]
		else:
			loss = list(vals.values())
		#print(loss)
		prints.append(name + str(np.mean(np.array(loss))))
		_since_beginning[name].update(vals)
        
		#print('--------------')
		#print(_since_beginning[name])
		#print(list(_since_beginning[name].keys()))

		x_vals = np.sort(list(_since_beginning[name].keys()))
		y_vals = [_since_beginning[name][x] for x in x_vals]

		plt.clf()
		plt.plot(x_vals, y_vals)
		plt.xlabel('iteration')
		plt.ylabel(name)
		plt.savefig(name.replace(' ', '_')+'.jpg')

	print("iter " + str(_iter[0]) + '\t' +  "\t".join(prints))
	_since_last_flush.clear()

	with open('log.pkl', 'wb') as f:
		pickle.dump(dict(_since_beginning), f, pickle.HIGHEST_PROTOCOL)