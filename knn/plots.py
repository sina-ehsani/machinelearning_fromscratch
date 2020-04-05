import matplotlib.pyplot as plt
import math
import numpy as np


def euclidist(array_a,array_b):
	distance = 0.0
	for i, j in zip(array_a,array_b):
	    distance += (i-j)**2
	return math.sqrt(distance)

if __name__ == '__main__':

	mu, sigma = 0, 0.5 # mean and standard deviation
	x = np.random.normal(mu, sigma, 500)
	y = np.random.normal(mu, sigma, 500)
	plt.plot(x, y, 'x')
	plt.axis('equal')
	plt.show()

	plt.hist(x+y)
	plt.show()

	dist=[]
	for i in range(len(x)):
	    for j in range(len(x)):
	        dist.append(euclidist([x[i],y[i]],[x[j],y[j]]))

	plt.hist(dist)
	# plt.axis('equal')
	plt.show()

	inner=[]
	for i in range(len(x)):
	    for j in range(len(x)):
	        inner.append(np.inner([x[i],y[i]],[x[j],y[j]]))

	plt.hist(inner)
	plt.show()
	counts, bins = np.histogram(inner)
	plt.hist(bins[:-1], bins, weights=counts)


	mu, sigma = 0, 0.01 # mean and standard deviation
	x=dict()
	for i in range(500):
	    x[i] = np.random.normal(mu, sigma, 100)

	dist2=[]
	for i in range(len(x)):
	    for j in range(len(x)):
	        dist2.append(euclidist(x[i],x[j]))

	plt.hist(dist2)
	plt.show()
	counts, bins = np.histogram(dist2)
	plt.hist(bins[:-1], bins, weights=counts)

	inner2=[]
	for i in range(len(x)):
	    for j in range(len(x)):
	        inner2.append(np.inner(x[i],x[j]))

	plt.hist(inner2)
	plt.show()
	counts, bins = np.histogram(inner2)
	plt.hist(bins[:-1], bins, weights=counts)


	y=list()
	for i in x.values():
	    y.extend(i)

	plt.hist(y)
	plt.show()