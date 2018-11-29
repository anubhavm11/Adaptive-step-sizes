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
# print(w_MLE)
# print(W_nat)
# print(loss_MLE/N)



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


W_SGDInit=np.load('W_init.npy')

#SGD final iterate
lossHistorySGD=[]
n=0
gamma=0.0001
W_SGD = W_SGDInit
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
		W_SGD += -gamma * gradient	
		if n%100==0 :
			preds = X.dot(W_SGD)
			error = preds - y
			loss = 0.5*np.sum(error ** 2)
			epochLoss.append(loss)
			lossHistorySGD.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD epoch #{}, average loss={} step-size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

#SGD average iterate
lossHistorySGDAvg=[]
n=0
gamma=0.01
W_SGD = W_SGDInit
W_SGDAvg = W_SGDInit
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
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD average epoch #{}, average loss={} step-size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

#SGD 1/sqrt(t) step size
lossHistorySGDDecay=[]
n=0
W_SGDDecay = W_SGDInit
gamma=0.01
for epoch in np.arange(0, epochs):
	# initialize the total loss for the epoch
	epochLoss = [] 
	
	new_order = np.random.permutation(N)
	X=X[new_order]
	y=y[new_order]
	# loop over our data in batches
	for i in range(N):
		n+=1
		preds = X[i].dot(W_SGDDecay)
		error = preds - y[i]
		loss = 0.5*error*error
		
		gradient = X[i]*(error)
		W_SGDDecay += -(gamma/np.sqrt(n)) * gradient
		if n%100==0 :
			preds = X.dot(W_SGDDecay)
			error = preds - y
			loss = 0.5*np.sum(error ** 2)
			
			epochLoss.append(loss)
			lossHistorySGDDecay.append(loss)
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD decay epoch #{}, average loss={} alpha={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

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

#Decaying step size with averaging
lossHistorySGDDecayAvg=[]
n=0
gamma=0.05
W_SGD = W_SGDInit
W_SGDDecayAvg = W_SGDInit
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
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD Decay average epoch #{}, average loss={} alpha={}".format(epoch + 1, np.average(epochLoss)/N,gamma/np.sqrt(n)))


#SGD 1/2
s=0
lossHistorySGDH=[]
n=0
ni=0
gradient=np.zeros(p)
gradient2=np.zeros(p)
W_SGDH = W_SGDInit
gamma=0.1
burnin=10000
ars=[]
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
		ars.append(gamma);
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
	# lossHistorySGD.append(np.average(epochLoss)/batch_size)
	print("[INFO] SGD epoch #{}, average loss={} step-size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

cmap = ['black','red','sienna','gold','palegreen','darkgreen','deepskyblue','navy','plum','palevioletred','magenta','slategray']
sns.set
fig = plt.figure()
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGD),endpoint=False) , (np.array(lossHistorySGD)-loss_MLE)/N,label='Vanilla SGD',color=cmap[0])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDAvg),endpoint=False) , (np.array(lossHistorySGDAvg)-loss_MLE)/N,label='Average iterate SGD',color=cmap[1])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDDecay),endpoint=False) , (np.array(lossHistorySGDDecay)-loss_MLE)/N,label='SGD with step-size 1/sqrt(t)',color=cmap[2])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDDecayAvg),endpoint=False), (np.array(lossHistorySGDDecayAvg)-loss_MLE)/N,label='SGD with step-size 1/sqrt(t) (Averaged)',color=cmap[3])
plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDH),endpoint=False), (np.array(lossHistorySGDH)-loss_MLE)/N,label='SGD 1/2',color=cmap[4])
plt.legend(loc='upper left')
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Excess loss")
plt.show()
plt.clf()
plt.loglog(np.linspace(0,epochs,len(lossHistorySGD),endpoint=False) , (np.array(lossHistorySGD)-loss_MLE)/N,label='Vanilla SGD',color=cmap[0])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDAvg),endpoint=False) , (np.array(lossHistorySGDAvg)-loss_MLE)/N,label='Average iterate SGD',color=cmap[1])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDDecay),endpoint=False) , (np.array(lossHistorySGDDecay)-loss_MLE)/N,label='SGD with step-size 1/sqrt(t)',color=cmap[2])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDDecayAvg),endpoint=False), (np.array(lossHistorySGDDecayAvg)-loss_MLE)/N,label='SGD 1/2',color=cmap[3])
plt.loglog(np.linspace(0,epochs,len(lossHistorySGDH),endpoint=False), (np.array(lossHistorySGDH)-loss_MLE)/N,label='SGD 1/2',color=cmap[4])
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