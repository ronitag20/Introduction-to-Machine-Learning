import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.linear_model import LogisticRegression

data_trn = np.loadtxt( "train.dat" )
data_tst = np.loadtxt( "test.dat" )

l=data_trn.shape[0]
m=data_tst.shape[0]

c=0
n=0
R=64
S=4
n_models = pow(2,S-1)*(pow(2,S)-1)
subsets = [None] * n_models
check=[]
models=[None]*n_models

# You are allowed to import any submodules of sklearn as well e.g. sklearn.svm etc
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES OTHER THAN SKLEARN WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_predict etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit( Z_train ):
################################
#  Non Editable Region Ending  #
################################
	for i in range(0,l):
		decimala =Z_train[i,-6]+2*Z_train[i,-7]+4*Z_train[i,-8]+8*Z_train[i,-9]
		decimalb =Z_train[i,-2]+2*Z_train[i,-3]+4*Z_train[i,-4]+8*Z_train[i,-5]
		if decimala<decimalb:
			c = sum(i + 2 for i in range(int(decimala)))
			n=int(16*decimala+decimalb-c-1)
			check=Z_train[i].copy()
		elif decimalb<decimala :
			c = sum(i + 2 for i in range(int(decimalb)))
			n=int(16*decimalb+decimala-c-1)
			check=Z_train[i].copy()
			check[-1]=1-check[-1]
		if decimala!=decimalb:
			if subsets[n] == None:
				subsets[n] = []
			subsets[n].append(check)
		check=[]

	x=[]
	for subset in (subsets):
		x.append(np.array(subset))
	model = [LogisticRegression(C=8.9,tol=0.058) for i in range(n_models)]
	for i in range(0,n_models):
		subset=x[i]
		model[i].fit(subset[:,:64],subset[:,-1])

	# Use this method to train your model using training CRPs
	# The first 64 columns contain the config bits
	# The next 4 columns contain the select bits for the first mux
	# The next 4 columns contain the select bits for the second mux
	# The first 64 + 4 + 4 = 72 columns constitute the challenge
	# The last column contains the response
	
	return model					# Return the trained model

c=0
n=0
subs = [None] * n_models
check=[]


################################
# Non Editable Region Starting #
################################
def my_predict( X_tst, model ):
################################
#  Non Editable Region Ending  #
################################
	pred=np.empty(m)
	for i in range(0,m):
		decimala =X_tst[i,-5]+2*X_tst[i,-6]+4*X_tst[i,-7]+8*X_tst[i,-8]
		decimalb =X_tst[i,-1]+2*X_tst[i,-2]+4*X_tst[i,-3]+8*X_tst[i,-4]
		if decimala<decimalb:
			c = sum(i + 2 for i in range(int(decimala)))
			n=int(16*decimala+decimalb-c-1)
			x = X_tst[i, :64].reshape((1, -1))
			pred[i]=model[n].predict(x)
		elif decimalb<decimala :
			c = sum(i + 2 for i in range(int(decimalb)))
			n=int(16*decimalb+decimala-c-1)
			x = X_tst[i, :64].reshape((1, -1))
			pred[i]=1-model[n].predict(x)

	# Use this method to make predictions on test challenges
	
	return pred
models=my_fit(data_trn)
pred=my_predict(data_tst[:,:-1],models)
accuracy=np.average(data_tst[ :, -1 ] == (pred))
print(accuracy)
