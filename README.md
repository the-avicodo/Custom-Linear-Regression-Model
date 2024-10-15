# Custom Linear Regression Model w/ NumPy
A thorough, yet simple model that shows the math behind linear regression
![image](https://github.com/user-attachments/assets/37105d32-34df-4e38-a633-0b8cbf9135e7)

### Project Overview
This project is a simple 2D linear regression model trained on generated sample data. This is to understand how linear regression works and how it is used in data analysis.

### Data Collection/Generation
For sample data, I generated a sample dataset using Scikit-Learn's sample generator model. I first generated sample data (with one feature, one target) and split it into training and testing data. Originally, I generated 250 points, however you can change the amount of generated points with the "num_entries" variable.

### Creating the model with NumPy
```py
def descent(w_new, lr): #lr = learning rate
    j=0 #number of iterations

    while True:
        #update weight values
        w_prev = w_new
        w0 = w_prev[0] - lr*grad(w_prev, X_train, y_train)[0]
        w1 = w_prev[1] - lr*grad(w_prev, X_train, y_train)[1]
        w_new = [w0,w1]
        print("Weights: w1 = {}, w0 = {}".format(w1, w0),
               "| Cost Function Value: {}".format(cost(w_new, X_train, y_train)))

        #stopping conditions
        if (w_new[0]-w_prev[0])**2 + (w_new[1]-w_prev[1])**2 <= pow(10, -6): #error threshold = 1e-6
            return w_new
        if j>500: #max iterations
            return w_new
        j+=1
```
To create the model, I used the cost(loss) function, gradient functions, and a descent for the iteration. The "cost" function returns the mean squared error of the iterated linear equation. The "grad" function returns an array of the partial derivatives of the cost function in respect to each weight(gradients). These are used to adjust the value of the weights during each iteration. The "descent" function performs the iterations. After the weights are initialized, each iteration updates the weight values by subtracting the product of the corresponding gradient and the learning rate("lr"), which is a small number that controls how much each gradient calculation influences the weights. I chose a learning of .1.

### Training the Model
After completing the difficult part, I initialized the weights:
```py
w = [0, -1]
```
Then I ran the gradient descent function:
```py
w = descent(w, .1)
```

### Evaluating the Model
I evaluated the model with the test data by comparing the Root Mean Squared Error(RMSE) with the standard deviation(SD). If the RMSE was less than half of the SD, then I considered an effective model.

![image](https://github.com/user-attachments/assets/778e2239-46e0-404c-af18-21353de71dea)

Thanks for reading!

