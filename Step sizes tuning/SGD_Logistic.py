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

X=np.load('train_x.npy')
y=np.load('train_y.npy')
y=np.sign(y)
print("Samples have been loaded\n")
alpha/=batch_size

#SGD final iterate
sns.set
fig = plt.figure()


fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")

step=[0.0001,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.5,1]
cmap = ['black','red','sienna','gold','palegreen','darkgreen','deepskyblue','navy','plum','palevioletred','magenta','slategray']
stepOpt=0
lossOpt=1000000
n=0
W_SGDInit=np.random.uniform(size=(X.shape[1],))
for gamma in step:
	lossHistorySGD=[]
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
			a = X[i].dot(W_SGD)
			gradient = (((sigmoid_activation(y[i]*a)-1))*y[i])*X[i]
			W_SGD += -gamma * gradient
			if(n%100==0):
				preds = X.dot(W_SGD)
				prod=y*preds
				loss = np.sum(np.log(1+np.exp(-x)) for x in prod)
				epochLoss.append(loss)
				lossHistorySGD.append(loss)
				if loss<lossOpt :
					stepOpt=gamma
					lossOpt=loss
		# lossHistorySGD.append(np.average(epochLoss)/batch_size)
		print("[INFO] SGD epoch #{}, average loss={} step size={}".format(epoch + 1, np.average(epochLoss)/N,gamma))

	plt.semilogy(np.linspace(0,epochs,len(lossHistorySGD),endpoint=False) , (np.array(lossHistorySGD))/N,label="Step size ={}".format(gamma),color=cmap[step.index(gamma)])
plt.legend(loc='upper left')
plt.show()
plt.clf()
	
print("Optimal step size={}".format(stepOpt))