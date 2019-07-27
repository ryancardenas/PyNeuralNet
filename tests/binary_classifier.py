X, Y = pn.loadCSVData('./tests/clover.csv')
X, Y = X.T, Y.T
print(X.shape)
print(Y.shape)

np.random.seed(2)

num_iterations = 1000
learning_rate = 1
layout = ((1, 'tanh'),
          (1, 'sigmoid'),
         )

network = pn.buildNetwork(layout, X.shape[0])
newtork, costs, accs = pn.gradientDescent(X, Y, network, num_iterations, learning_rate,
                        costfunction='logistic', showprogress=True,
                        recording=True, debugmsg='')


print('Iteration:', num_iterations)
pn.plot2DBoundary(X, Y, network, lines=True, fill=False, marksize=20)
H = pn.predict(X, network)
acc, prec, rec = pn.evaluateModel(H, Y)
print('Accuracy:', acc[0,0]*100, '%')
print('Precision:', prec[0,0]*100, '%')
print('Recall:', rec[0,0]*100, '%')

plt.subplot(211)
plt.plot(costs, label='cost')
plt.title('cost')

plt.subplot(212)
plt.plot(accs, label='accuracy')
plt.title('accuracy')

plt.tight_layout()
plt.show()
