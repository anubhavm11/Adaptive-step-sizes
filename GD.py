import matplotlib.pyplot as plt
import numpy as np
import argparse
import math

N=500
p=10


def next_batch(X, y, batchSize):
	rng_state = np.random.get_state()
	np.random.shuffle(X)
	np.random.set_state(rng_state)
	np.random.shuffle(y)
	# loop over our dataset `X` in mini-batches of size `batchSize`
	for i in np.arange(0, X.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (X[int(i):int(i) + int(batchSize)], y[int(i):int(i) + int(batchSize)])

def error(A,b,w):
	preds=A.dot(w)
	errors=preds-b
	return errors

def loss(A,b,w):
	errors=error(A,b,w)
	return (0.5*np.sum(errors**2))

#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=5,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.1,
	help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=N/10,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())
t=100
epochs=args["epochs"]
alpha=args["alpha"]
batch_size=args["batch_size"]

#data generation
W_nat=np.zeros(p)
for i in range(p):
	W_nat[i]=10*np.exp(-0.75*i)
X=np.random.multivariate_normal(np.zeros(p),np.identity(p),N)
SNR=2.0
cov=(1/SNR)
y=np.random.multivariate_normal(X.dot(W_nat),cov*np.identity(N),1)
y=y[0]
print("Samples have been generated\n")

#computing best w for the data and getting its loss
w_MLE=(np.linalg.inv(np.matmul(X.T,X)).dot(X.T.dot(y)))
loss_MLE=loss(X,y,w_MLE)
print(w_MLE)
print(W_nat)
print(loss_MLE/N)



# #gradient descent
# lossHistoryGD=[]
# W_GD = np.random.uniform(size=(X.shape[1],))
# for epoch in np.arange(0, epochs):
# 	preds = X.dot(W_GD)
# 	error = preds - y
# 	loss = np.sum(error ** 2)
# 	lossHistoryGD.append(loss)
# 	print("[INFO] GD epoch #{}, loss={:.7f} alpha={}".format(epoch + 1, loss,alpha))
# 	gradient = X.T.dot(error) / X.shape[0]
# 	# if epoch%1000==0 :
# 	# 	alpha=alpha/2;
# 	W_GD += -alpha * gradient

#SGD final iterate
lossHistorySGD=[]
W_SGD = np.random.uniform(size=(X.shape[1],))
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	# loop over our data in batches
	for (batchX, batchY) in next_batch(X, y, batch_size):
		preds = batchX.dot(W_SGD)
		error = preds - batchY
		loss = 0.5*np.sum(error ** 2)
		
		gradient = batchX.T.dot(error) / batchX.shape[0]
		W_SGD += -alpha * gradient
		preds = X.dot(W_SGD)
		error = preds - y
		loss = 0.5*np.sum(error ** 2)
		epochLoss.append(loss)
		lossHistorySGD.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD epoch #{}, average loss={} alpha={}".format(epoch + 1, np.average(epochLoss)/N,alpha))

#SGD average iterate
lossHistorySGD2=[]
W_SGD2 = []
W_SGD2.append(np.random.uniform(size=(X.shape[1],)))
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	# loop over our data in batches
	for (batchX, batchY) in next_batch(X, y, batch_size):
		W2=np.average(W_SGD2,axis=0)
		preds = batchX.dot(W2)
		error = preds - batchY
		loss = 0.5*np.sum(error ** 2)
		
		gradient = batchX.T.dot(error) / batchX.shape[0]
		W_SGD2.append(W2-(alpha * gradient))
		if (len(W_SGD2)==t):
			W_SGD2.pop(0)
		preds = X.dot(np.average(W_SGD2,axis=0))
		error = preds - y
		loss = 0.5*np.sum(error ** 2)
		epochLoss.append(loss)
		lossHistorySGD2.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD average epoch #{}, average loss={} alpha={}".format(epoch + 1, np.average(epochLoss)/N,alpha))

#SGD 1/sqrt(t) step size
lossHistorySGD3=[]
n=0
W_SGD3 = np.random.uniform(size=(X.shape[1],))
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	# loop over our data in batches
	for (batchX, batchY) in next_batch(X, y, batch_size):
		n+=1
		preds = batchX.dot(W_SGD3)
		error = preds - batchY
		loss = 0.5*np.sum(error ** 2)
		
		gradient = batchX.T.dot(error) / batchX.shape[0]
		gamma=alpha/(np.sqrt(n))
		W_SGD3 += -gamma * gradient
		preds = X.dot(W_SGD3)
		error = preds - y
		loss = 0.5*np.sum(error ** 2)
		epochLoss.append(loss)
		lossHistorySGD3.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD 1/sqrt(t) step epoch #{}, average loss={} gamma={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

# # Implicit Stochastic Gradient Descent
# lossHistoryISGD=[]
# W_ISGD = np.random.uniform(size=(X.shape[1],))
# W_ISGD2 = np.zeros(p)
# for epoch in np.arange(0, epochs):
# 	# initialize the total loss for the epoch
# 	epochLoss = [] 
# 	# loop over our data in batches
# 	for i in range(X.shape[0]):
# 		preds = X[i].dot(W_ISGD)
# 		error = preds - y[i]
# 		loss = error ** 2
# 		epochLoss.append(loss)
# 		Z=np.sum(X[i]**2)
# 		np.copyto(W_ISGD2,W_SGD)
# 		W_ISGD=(W_ISGD2+(alpha*y[i]*X[i]))/(1+(alpha*Z))
# 	lossHistoryISGD.append(np.average(epochLoss))
# 	print("[INFO] ISGD epoch #{}, loss={:.7f} alpha={}".format(epoch + 1, np.average(epochLoss) ,alpha))

#SGD 1/2
lossHistorySGDH=[]
W_SGDH = np.random.uniform(size=(X.shape[1],))
W_SGDH2=np.zeros(p)
W_SGDH3=np.zeros(p)
gradient=np.zeros(p)
gradient2=np.zeros(p)
s=0
ni=0
n=0
burnin=epochs/20
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	# loop over our data in batches
	for (batchX, batchY) in next_batch(X, y, batch_size):
		
		preds = batchX.dot(W_SGDH)
		error = preds - batchY
		loss = 0.5*np.sum(error ** 2)
		np.copyto(gradient2,gradient)
		gradient = batchX.T.dot(error) / batchX.shape[0]
		np.copyto(W_SGDH3,W_SGDH2)
		np.copyto(W_SGDH2,W_SGDH)
		W_SGDH += -alpha * gradient
		if (epoch>2):
			s+=np.dot(W_SGDH-W_SGDH2,W_SGDH2-W_SGDH3)/(alpha**2)
			# s+=np.dot(gradient2,gradient)
		# print(s)
		if (n>ni+burnin) and (s<0) :
			ni=n
			s=0
			alpha/=2
		preds = X.dot(W_SGDH)
		error = preds - y
		loss = 0.5*np.sum(error ** 2)
		epochLoss.append(loss)
		lossHistorySGDH.append(loss)
		
	# lossHistorySGDH.append(np.average(epochLoss)/batch_size)
	n+=1
	print("[INFO] SGD 1/2 epoch #{}, average loss={:.7f} alpha={}".format(epoch + 1, np.average(epochLoss)/N,alpha))
print(W_nat)
print(w_MLE)
print(W_SGD)
print(W_SGDH)
fig = plt.figure()
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGD),endpoint=False) , np.array(lossHistorySGD)-loss_MLE,'b',np.linspace(0,epochs,len(lossHistorySGD3),endpoint=False) , np.array(lossHistorySGD3)-loss_MLE,'y',np.linspace(0,epochs,len(lossHistorySGD2),endpoint=False) , np.array(lossHistorySGD2)-loss_MLE,'r',np.linspace(0,epochs,len(lossHistorySGDH),endpoint=False), np.array(lossHistorySGDH)-loss_MLE,'g')
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()