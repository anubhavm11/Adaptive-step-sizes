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

#SGD final iterate
sns.set
fig = plt.figure()

fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")

step=[5,10,20,50,100,200,500,1000,2000]
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
	gamma=5
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
			# ars.append(s2);
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
				if loss<lossOpt :
					stepOpt=burnin
					lossOpt=loss
		# lossHistorySGD.append(np.average(epochLoss)/batch_size)
		print("[INFO] SGD epoch #{}, average loss={} step-size={} s={}".format(epoch + 1, np.average(epochLoss)/N,gamma,s))

	plt.semilogy(np.linspace(0,epochs,len(lossHistorySGDH),endpoint=False) , (np.array(lossHistorySGDH))/N,label="Burnin ={}".format(burnin),color=cmap[step.index(burnin)])
plt.legend(loc='upper left')
plt.show()
plt.clf()
	
print("Optimal Burnin={}".format(burnin))