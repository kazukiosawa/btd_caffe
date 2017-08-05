#!/usr/bin/python

from sktensor import dtensor, tucker
import numpy as np


def pci(T, R, rank, max_iter=1000, min_decrease=1e-5):
	shape = np.array(T).shape
	dim = range(len(rank))
	tensors = [dtensor(np.zeros(shape)) for r in range(R)]
	last = 1
	for i in range(max_iter):
		btd = []
		print "iter {0}".format(i+1)
		for r in range(R):
			Tres = T - (sum(tensors) - tensors[r])
			print "\t HOOI {0}".format(r+1)
			Td = tucker.hooi(Tres, rank, init='nvecs')
			btd.append(Td)
			coret = Td[0]
			factm = Td[1]
			Tapprox = coret.ttm(factm, dim)
			print "\t\t norm {0}".format(Tapprox.norm())
			tensors[r] = Tapprox
		Tres = T - sum(tensors)
		error = Tres.norm()/T.norm()
		decrease = last - error
		print "\t --------------------"
		print "\t Error {0}".format(error)
		print "\t Decrease {0}".format(decrease)
		if decrease <= min_decrease:
			break
		last = error
	return btd, tensors


def main():
	h = 3
	w = 3
	c = 512
	n = 512
	cr = 153
	nr = 153
	T = dtensor(np.random.rand(h*w, c, n))
	R = 4 
	rank = [9, int(cr/R), int(cr/R)]
	pci(T, R, rank)


if __name__ == '__main__':
	main()
