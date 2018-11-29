import matplotlib.pyplot as plt
import numpy as np
import argparse
import math
import seaborn as sns

N=5000
p=10

def sigmoid_activation(x):
	# compute and return the sigmoid activation value for a
	# given input value
	return 1.0 / (1 + np.exp(-x))


#argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=50,
	help="# of epochs")
ap.add_argument("-a", "--alpha", type=float, default=0.1,
	help="learning rate")
ap.add_argument("-b", "--batch_size", type=int, default=1,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

epochs=args["epochs"]
alpha=args["alpha"]

batch_size=args["batch_size"]

#data generation
X=np.load('train_x.npy')
y=np.load('train_y.npy')
y=np.sign(y)
print("Samples have been loaded\n")
alpha/=batch_size


W_SGDInit = np.load('W_init.npy')


#SGD final iterate
lossHistorySGD=[]
alpha=0.05 #best value set
W_SGD = W_SGDInit
n=0
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	new_order = np.random.permutation(N)
	X=X[new_order]
	y=y[new_order]
	# loop over our data in batches
	for i in range(N):
		n+=1
		a = X[i].dot(W_SGD)
		gradient = (((sigmoid_activation(y[i]*a)-1))*y[i])*X[i]
		W_SGD += -alpha * gradient
		if(n%100==0):
			preds = X.dot(W_SGD)
			prod=y*preds
			loss = np.sum(np.log(1+np.exp(-x)) for x in prod)
			epochLoss.append(loss)
			lossHistorySGD.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD epoch #{}, average loss={} alpha={}".format(epoch + 1, np.average(epochLoss)/N,alpha))

lossHistorySGDAvg=[]
W_SGDAvg = W_SGDInit
W_SGD = W_SGDInit
n=0
gamma=0.5
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	new_order = np.random.permutation(N)
	X=X[new_order]
	y=y[new_order]
	# loop over our data in batches
	for i in range(N):
		n+=1
		a = X[i].dot(W_SGD)
		gradient = (((sigmoid_activation(y[i]*a)-1))*y[i])*X[i]
		W_SGD=W_SGD-gamma * gradient
		W_SGDAvg = ((W_SGDAvg*n)+W_SGD)/(n+1)
		if(n%100==0):
			preds = X.dot(W_SGDAvg)
			prod=y*preds
			loss = np.sum(np.log(1+np.exp(-x)) for x in prod)
			epochLoss.append(loss)
			lossHistorySGDAvg.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD Average epoch #{}, average loss={} step size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

lossHistorySGDDecay=[]
W_SGDDecay = W_SGDInit
n=0
gamma=20
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	new_order = np.random.permutation(N)
	X=X[new_order]
	y=y[new_order]
	# loop over our data in batches
	for i in range(N):
		n+=1
		a = X[i].dot(W_SGDDecay)
		gradient = (((sigmoid_activation(y[i]*a)-1))*y[i])*X[i]
		W_SGDDecay += -(gamma/np.sqrt(n)) * gradient
		if(i%100==0):
			preds = X.dot(W_SGDDecay)
			prod=y*preds
			loss = np.sum(np.log(1+np.exp(-x)) for x in prod)
			epochLoss.append(loss)
			lossHistorySGDDecay.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD Decay epoch #{}, average loss={} step size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))


#Decaying step-size with averaging
lossHistorySGDDecayAvg=[]
W_SGDDecayAvg = W_SGDInit
W_SGD = W_SGDInit
n=0
gamma=10
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	new_order = np.random.permutation(N)
	X=X[new_order]
	y=y[new_order]
	# loop over our data in batches
	for i in range(N):
		n+=1
		a = X[i].dot(W_SGD)
		gradient = (((sigmoid_activation(y[i]*a)-1))*y[i])*X[i]
		W_SGD=W_SGD-((gamma/np.sqrt(n)) * gradient)
		W_SGDDecayAvg = ((W_SGDDecayAvg*n)+W_SGD)/(n+1)
		if(n%100==0):
			preds = X.dot(W_SGDDecayAvg)
			prod=y*preds
			loss = np.sum(np.log(1+np.exp(-x)) for x in prod)
			epochLoss.append(loss)
			lossHistorySGDDecayAvg.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD Decay Average epoch #{}, average loss={} step size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

s=0
lossHistorySGDH=[]
n=0
ni=0
ars=[]
gradient=np.zeros(p)
gradient2=np.zeros(p)
W_SGDH = W_SGDInit
gamma=1
burnin=20
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	new_order = np.random.permutation(N)
	X=X[new_order]
	y=y[new_order]
	# loop over our data in batches
	for i in range(N):
		n+=1
		a = X[i].dot(W_SGDH)
		gradient = (((sigmoid_activation(y[i]*a)-1))*y[i])*X[i]
		# if (n>1000):
		s+=np.dot(gradient2,gradient)
		# print(s)
		gradient2=gradient
		ars.append(gamma);
		if (n>ni+burnin) and (s<0) :
			ni=n
			s=0
			# s2=0
			gamma/=2


		W_SGDH += -gamma * gradient	
		if n%100==0 :
			preds = X.dot(W_SGDH)
			prod=y*preds
			loss = np.sum(np.log(1+np.exp(-x)) for x in prod)
			epochLoss.append(loss)
			lossHistorySGDH.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD epoch #{}, average loss={} step-size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

cmap = ['black','red','sienna','gold','palegreen','darkgreen','deepskyblue','navy','plum','palevioletred','magenta','slategray']
sns.set
fig = plt.figure()
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGD),endpoint=False) , (np.array(lossHistorySGD))/N,label='Vanilla SGD',color=cmap[0])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDAvg),endpoint=False) , (np.array(lossHistorySGDAvg))/N,label='Average iterate SGD',color=cmap[1])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDDecay),endpoint=False) , (np.array(lossHistorySGDDecay))/N,label='SGD with step-size 1/sqrt(t)',color=cmap[2])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDDecayAvg),endpoint=False), (np.array(lossHistorySGDDecayAvg))/N,label='SGD with step-size 1/sqrt(t) (Averaged)',color=cmap[3])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDH),endpoint=False), (np.array(lossHistorySGDH))/N,label='SGD 1/2',color=cmap[4])
plt.legend(loc='upper left')
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Excess loss")
plt.show()
plt.clf()
plt.loglog(np.linspace(0,epochs,len(lossHistorySGD),endpoint=False) , (np.array(lossHistorySGD))/N,label='Vanilla SGD',color=cmap[0])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDAvg),endpoint=False) , (np.array(lossHistorySGDAvg))/N,label='Average iterate SGD',color=cmap[1])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDDecay),endpoint=False) , (np.array(lossHistorySGDDecay))/N,label='SGD with step-size 1/sqrt(t)',color=cmap[2])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDDecayAvg),endpoint=False), (np.array(lossHistorySGDDecayAvg))/N,label='SGD with step-size 1/sqrt(t) (Averaged)',color=cmap[3])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDH),endpoint=False), (np.array(lossHistorySGDH))/N,label='SGD 1/2',color=cmap[4])
plt.legend(loc='upper left')
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Excess loss")
plt.show()
plt.clf()
plt.semilogy(np.linspace(0,epochs,len(ars),endpoint=False) , np.array(ars),label='step-size of SGD1/2',color=cmap[0])
plt.legend(loc='upper left')
fig.suptitle("Variation in step-size of SGD1/2")
plt.xlabel("Epoch #")
plt.ylabel("step-size")
plt.show()