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
ap.add_argument("-e", "--epochs", type=float, default=50,
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

step=[0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1]
cmap = ['black','red','sienna','gold','palegreen','darkgreen','deepskyblue','navy','plum','palevioletred','magenta','slategray']
stepOpt=0
lossOpt=1000000
W_SGDAvgInit=np.random.uniform(size=(X.shape[1],))
for gamma in step:
	lossHistorySGDAvg=[]
	n=0
	W_SGD = W_SGDAvgInit
	W_SGDAvg = W_SGDAvgInit
	for epoch in np.arange(0, epochs):
		# initialize the total loss for the epoch
		epochLoss = [] 
		
		new_order = np.random.permutation(N)
		X=X[new_order]
		y=y[new_order]
		# loop over our data in batches
		for i in range(N):
			n+=1
			preds = X[i].dot(W_SGD)
			error = preds - y[i]
			loss = 0.5*error*error
			
			gradient = X[i]*(error)
			W_SGD=W_SGD-(gamma*gradient)
			W_SGDAvg=(W_SGD+(W_SGDAvg*n))/(n+1)
			if n%100==0 :
				preds = X.dot(W_SGDAvg)
				error = preds - y
				loss = 0.5*np.sum(error ** 2)
				
				epochLoss.append(loss)
				lossHistorySGDAvg.append(loss)
				if loss<lossOpt :
					stepOpt=gamma
					lossOpt=loss
		# lossHistorySGD.append(np.average(epochLoss)/batch_size)
		print("[INFO] SGD average epoch #{}, average loss={} alpha={}".format(epoch + 1, np.average(epochLoss)/N,gamma))
	plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDAvg),endpoint=False) , (np.array(lossHistorySGDAvg)-loss_MLE)/N,'b',label="Step size ={}".format(gamma),color=cmap[step.index(gamma)])
plt.legend(loc='upper left')
plt.show()
plt.clf()
	
print("Optimal step size={}".format(stepOpt))


step=[0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
cmap = ['black','red','sienna','gold','palegreen','darkgreen','deepskyblue','navy','plum','palevioletred','magenta','slategray']
stepOpt=0
lossOpt=1000000
W_SGDDecayAvgInit=np.random.uniform(size=(X.shape[1],))
for gamma in step:
	lossHistorySGDDecayAvg=[]
	n=0
	W_SGD = W_SGDDecayAvgInit
	W_SGDDecayAvg = W_SGDDecayAvgInit
	for epoch in np.arange(0, epochs):
		# initialize the total loss for the epoch
		epochLoss = [] 
		
		new_order = np.random.permutation(N)
		X=X[new_order]
		y=y[new_order]
		# loop over our data in batches
		for i in range(N):
			n+=1
			preds = X[i].dot(W_SGD)
			error = preds - y[i]
			loss = 0.5*error*error
			
			gradient = X[i]*(error)
			W_SGD=W_SGD-((gamma/np.sqrt(n))*gradient)
			W_SGDDecayAvg=(W_SGD+(W_SGDDecayAvg*n))/(n+1)
			if n%100==0 :
				preds = X.dot(W_SGDDecayAvg)
				error = preds - y
				loss = 0.5*np.sum(error ** 2)
				
				epochLoss.append(loss)
				lossHistorySGDDecayAvg.append(loss)
				if loss<lossOpt :
					stepOpt=gamma
					lossOpt=loss
		# lossHistorySGD.append(np.average(epochLoss)/batch_size)
		print("[INFO] SGD Decay average epoch #{}, average loss={} alpha={}".format(epoch + 1, np.average(epochLoss)/N,gamma/np.sqrt(n)))
	plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDDecayAvg),endpoint=False) , (np.array(lossHistorySGDDecayAvg)-loss_MLE)/N,'b',label="Step size ={}".format(gamma),color=cmap[step.index(gamma)])
plt.legend(loc='upper left')
plt.savefig('SGDDecayAvg.png')
plt.show()
plt.clf()

