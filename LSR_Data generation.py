import numpy as np
import argparse
import math

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-N", "--number", type=int, default=5000,
	help="# of data points")
ap.add_argument("-p", "--dimension", type=int, default=10,
	help="dimension of data")
args = vars(ap.parse_args())

N=args["number"]
p=args["dimension"]

#data generation
W_nat=np.zeros(p)
for i in range(p):
	W_nat[i]=10*np.exp(-0.75*i)
X=np.random.multivariate_normal(np.zeros(p),np.identity(p),N)
SNR=2.0
cov=(1/SNR)
y=np.random.multivariate_normal(X.dot(W_nat),cov*np.identity(N),1)
y=y[0]
np.save('train_x.npy',X)
np.save('train_y.npy',y)
print("Samples have been generated\n")
