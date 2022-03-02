import numpy as np
import scipy.io
import math
import matplotlib.pyplot as plt
import PIL

mat = scipy.io.loadmat('mnist_small.mat')
tick=1
X=mat['X']
Y=mat['Y']
N,D=X.shape
KList=[20]
# KList=[5,10,15,20]
# print(N)
numIter=200
batches=100
# print(X.shape)
# print(Y.shape)
Classes=[]
for i in range(0,10):
    # print(Y==i)
    Z=Y.reshape(Y.shape[0],)
    Z=list(np.where(Z==i))
    # print(np.max(Z[0]))
    # print(np.min(Z[0]))
    # print(len(Z))
    temp1=X[Z]
    # print(Z)
    Z[0]=Z[0].reshape(Z[0].shape[0],1)
    # print(Z.shape)
    # print(X[Z].shape)
    # print(temp1.shape)
    temp=np.concatenate((temp1,Z[0]),axis=1)
    # print(temp.shape)
    Classes.append(temp)
# exit(0)


def get_batch(batch_size):
    per_class_sample=int(batch_size/10)
    batch=np.zeros(per_class_sample)
    flag=0
    # print(len(Classes))
    for C in Classes:
        # print(C.shape[0])
        idx=np.random.randint(C.shape[0],size=per_class_sample)
        # print(idx)
        if(flag==0):
            batch=C[idx,:]
            flag=1
        else:
            batch=np.concatenate((batch,C[idx,:]),axis=0)
    return batch

def pdf_calc(pik,x,mu,sigma2):
    return((np.log(pik) - np.linalg.norm(mu - x)**2) / (2*sigma2))


def stepwise_EM(K):
    pi=np.ones((K,1))/K
    # print(pi)
    # exit(0)
    mu=np.random.randn(K,D)
    z=np.zeros((N,K))
    sigma2=1
    # ILL=[]
    for iter in range(1,2000):
        lr=math.pow((1+tick),-0.55)
        batch=get_batch(batches)
        idx=[]
        X_batch=[]
        for sample in batch:
            n=sample[-1]
            idx.append(n)
            for k in range(0,K):
                # print(pi[k])
                if(pi[k]==0):
                    print("breaking")
                    exit(0)
                z[n,k]=(pdf_calc(pi[k],sample[:-1],mu[k],sigma2))
            maxZ=np.max(z[n])
            # print("-------maxZ-----")
            # print(maxZ)
            # if(maxZ==0):
            #     print(z[n])
            #     print("Catch Here")
            #     exit(0)
            z[n]=np.exp(z[n]-maxZ-math.log(np.sum(np.exp(z[n]-maxZ))))
            X_batch.append(sample[:-1])
            # print(z[n,])
            # exit(0)
        # print("Iter completed")
        # print(z[idx])
        # Y=np.sum(z[idx],axis=0)
        # for i in Y:
        #     print(i)
        #     if(i==0):
        #         print(z[idx])
        #         print(idx)
        #         print("Breaking")
        #         exit(0)

        X_batch=np.asarray(X_batch)
        pi_hat=np.sum(z[idx],axis=0)/batches
        pi_hat=np.reshape(pi_hat.shape[0],1)
        # print(pi_hat.shape)

        pi=(1-lr)*pi+lr*pi_hat

        mu_hat=np.dot(z[idx].T,X_batch)/(np.sum(z[idx],axis=0).reshape(-1,1))
        # print("hey")
        mu=(1-lr)*mu+lr*mu_hat
        sigma2_hat=np.linalg.norm(X_batch-np.dot(z[idx],mu))**2/(batches*D)
        sigma2=(1-lr)*sigma2+lr*sigma2_hat
    # print(ILL)
    return mu


for K in KList:
    print("Started Online EM for K= "+str(K))
    mu=stepwise_EM(K)
    ctr=0
    for m in mu:
        ctr=ctr+1
        img_j = m.reshape((-1, 28))
        a2 = PIL.Image.fromarray(img_j.astype(np.uint8))
        a2.save('online_K_'+str(K)+"mu_"+str(ctr)+'.png')
