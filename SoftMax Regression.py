
import numpy as np
# import warnings
  
# # suppress overflow warnings
# warnings.filterwarnings('ignore')

def SGD(X, Y, w, b, epoch, alpha, eps, mb, N):
    # Stochastic Gradient Descent
    for i in range(epoch):
        for j in range(int(N/mb)):
            X_mb=X[:,j*mb:(j+1)*mb]
            Y_mb=Y[j*mb:(j+1)*mb,:]
        
            # Calculating Predicted labels
            z=np.dot(X_mb.T,w)+b
            Y_mb_hat=np.exp(z)/np.sum(np.exp(z),axis=1)[:,None]
            
            # Calculating the Gradients
            dw= (np.dot(X_mb,(Y_mb_hat-Y_mb))+alpha*w)/mb
            db= np.sum(Y_mb_hat-Y_mb)/mb

            # Updating the Weights and Bias
            w-=eps*dw
            b-=eps*db
    return w,b


# No of Classes
c=10

# Loading Training Dataset
X_data = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28))
Y_data = np.load("fashion_mnist_train_labels.npy")
Y_data_onehot=np.eye(c)[Y_data]

# Splitting into Training and Validation Dataset
X_tr, X_va= np.split(X_data,[int(.8 * len(X_data))])
X_tr= X_tr.T
X_va= X_va.T
Y_tr, Y_va= np.split(Y_data_onehot, [int(.8 * len(X_data))])

# Loading Test Dataset
X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28))
X_te= X_te.T
Y_te = np.load("fashion_mnist_test_labels.npy")
Y_te_onehot=np.eye(c)[Y_te]

# Number of Training Examples
N=np.shape(X_tr)[1]

# Hyper Parameter Values
#Uncomment below 4 lines to run it for one Hyperparameter Set
# Mb=[32]
# Epoch=[35]
# Alpha=[1]
# Eps=[0.0000005]
Mb=[2, 4, 8, 10, 16, 20, 30, 32] # Mini Batch Size
Epoch= [5, 10, 25, 35, 50, 75, 100, 250] # Number of Epochs
Alpha= [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001] # Regularization Strength
Eps= [0.00001, 0.000005, 0.000001, 0.0000005, 0.0000001, 0.00000005, 0.00000001, 0.000000005] # Learning Rate

mincost=np.Inf
H_star=np.array(4) #Initializing an array to store Best Hyperparameters

print("----------------------------------------------------------------------")
print("Performing Grid Search for 4096 combinations of Hyperparameters:")
print("----------------------------------------------------------------------")

W= np.zeros((np.shape(X_tr)[0],c))  #Randomly intializing the Weights
B= np.zeros(10) #Initializing bias = 0

for epoch in Epoch:
    for alpha in Alpha:
        for eps in Eps:
            for mb in Mb:
                # Stochastic Gradient Descent
                w,b=SGD(X_tr, Y_tr, W, B, epoch, alpha, eps, mb, N)

                # Calculating Predicted labels
                z_va=np.dot(X_va.T,w)+b 
                Y_va_hat= np.exp(z_va)/np.sum(np.exp(z_va),axis=1)[:,None]
                
                Y_va_hat_log = np.log(Y_va_hat, out=np.zeros_like(Y_va_hat), where=(Y_va_hat!=0))
                
                # Calculating MSE Loss on Validation Setimport warnings
                cost= -(np.trace(np.dot(Y_va,Y_va_hat_log.T)))/(np.shape(Y_va)[0]) + alpha*np.trace(np.dot(w.T,w))/(2*np.shape(Y_va)[0])

                # Updating Cost and Hyperparameters
                if(cost<mincost):
                    mincost=cost
                    H_star= [epoch, alpha, eps, mb]
                                     
print(" Grid Search Completed")
print(" Results after performing Grid Search:")
print(" Best Hyperparameters:") 
print("   Epochs= ",H_star[0])
print("   Alpha= ",H_star[1])
print("   Learning Rate= ",H_star[2])
print("   Mini Batch Size= ", H_star[3])
print(" Cost on Validation Set with Best Hyperparameters= ", mincost)
print("\n----------------------------------------------------------------------")
print("Training on Training + Validation Dataset:")
print("----------------------------------------------------------------------")

X_data= X_data.T

# Number of Training Examples
N_full=np.shape(X_data)[1]

# Stochastic Gradient Descent
w,b=SGD(X_data, Y_data_onehot, W, B, H_star[0], H_star[1], H_star[2], H_star[3], N_full)

print(" Training Completed")
print("\n----------------------------------------------------------------------")
print("Performance Evaluation")
print("----------------------------------------------------------------------")

# Calculating Predicted labels
z_te=np.dot(X_te.T,w)+b  
Y_te_hat= np.exp(z_te)/np.sum(np.exp(z_te),axis=1)[:,None]
Y_te_hat_log = np.log(Y_te_hat, out=np.zeros_like(Y_te_hat), where=(Y_te_hat!=0))
Y_pred = np.argmax(Y_te_hat,axis=1)

# Calculating MSE Loss on Test Set  
cost= -np.trace(np.dot(Y_te_onehot,Y_te_hat_log.T))/(np.shape(Y_te)[0])
accuracy = np.sum(Y_pred==Y_te)/np.shape(Y_te)[0]
print(" Cost on Test Dataset= ", cost)
print(" Accuracy on Test Dataset= %.2f" % (accuracy*100), "%")
print("\n")      
         