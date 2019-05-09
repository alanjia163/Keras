import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

np.random.seed(1)
# data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)
Y =0.5*X +2 +np.random.normal(0,0.05,(200,))


#plot
plt.scatter(X,Y)
plt.show()
#train data
X_train,Y_train = X[:160],Y[:160]
#test data
X_test,Y_test =X[160:],Y[160:]

# 按顺序执行
model =Sequential()
#add_layers
model.add(Dense(units=1,input_dim=1))
#loss,optimizer
model.compile(loss='mse',optimizer='sgd')

#tarin_step
for step in range(1000):
    #cost
    cost = model.train_on_batch(X_train,Y_train)
    if step% 100 == 0:
        print('train:',cost)

#test
print('Testing')
cost =model.evaluate(X_test,Y_test,batch_size=40)
print('test cost:',cost)
W,b = model.layers[0].get_weights()

print('Weights=',W,'bias',b)

#plot
Y_pred = model.predict(X_test)

plt.scatter(X_test, Y_test)
plt.plot(X_test,Y_pred)
plt.show()





