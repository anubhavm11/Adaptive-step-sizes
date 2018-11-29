import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import seaborn as sns

N=5000
p=10

def next_batch(X, y, batchSize):
	new_order = np.random.permutation(N)
	X=X[new_order]
	y=y[new_order]
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
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=1,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

epochs=args["epochs"]
alpha=args["alpha"]

batch_size=args["batch_size"]

X=np.load('train_x.npy')
y=np.load('train_y.npy')
print("Samples have been loaded\n")
alpha/=batch_size

#computing best w for the data and getting its loss
w_MLE=(np.linalg.inv(np.matmul(X.T,X)).dot(X.T.dot(y)))
loss_MLE=loss(X,y,w_MLE)


#SGD final iterate
sns.set
fig = plt.figure()

fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")

step=[100,200,500,1000,2000,5000,10000,20000,30000,40000]
cmap = ['black','red','sienna','gold','palegreen','darkgreen','deepskyblue','navy','plum','palevioletred','magenta','slategray']
stepOpt=0
lossOpt=1000000
W_SGDInit=np.random.uniform(size=(X.shape[1],))
for burnin in step:
	s=0
	lossHistorySGDH=[]
	n=0
	ni=0
	gradient=np.zeros(p)
	gradient2=np.zeros(p)
	W_SGDH = W_SGDInit
	gamma=0.1
	for epoch in np.arange(0, epochs):
		# initialize the total loss for the epoch
		epochLoss = [] 
		new_order = np.random.permutation(N)
		X=X[new_order]
		y=y[new_order]
		# loop over our data in batches
		for i in range(N):
			n+=1
			preds = X[i].dot(W_SGDH)
			error = preds - y[i]
			loss = 0.5*error*error
			
			gradient = X[i]*(error)
			if (n>1000):
				s+=np.dot(gradient2,gradient)
			# print(s)
			gradient2=gradient
			# ars.append(s2);
			if (n>ni+burnin) and (s<0) :
				ni=n
				s=0
				# s2=0
				gamma/=2


			W_SGDH += -gamma * gradient	
			if n%100==0 :
				preds = X.dot(W_SGDH)
				error = preds - y
				loss = 0.5*np.sum(error ** 2)
				epochLoss.append(loss)
				lossHistorySGDH.append(loss)
				if loss<lossOpt :
					stepOpt=burnin
					lossOpt=loss
		# lossHistorySGD.append(np.average(epochLoss)/batch_size)
		print("[INFO] SGD epoch #{}, average loss={} burnin={} step-size={}".format(epoch + 1, np.average(epochLoss)/N,burnin,gamma))

	plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDH),endpoint=False) , (np.array(lossHistorySGDH)-loss_MLE)/N,label="Burnin ={}".format(burnin),color=cmap[step.index(burnin)])
plt.legend(loc='upper left')
plt.show()
plt.clf()
	
print("Optimal Burnin={}".format(burnin))