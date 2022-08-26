# Linear_regression
Linear Regression is the first machine learning project in the School 21 curriculum.

The goal of this project is to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm. A linear regression algorithm is implemented on one element — the car mileage.

## Estimator
The first program is used to predict the price of a car based on its mileage. When you launch the program, it asks you for mileage and should give you an approximate price of the car. The program uses the following hypothesis to predict the price:
### $estimatePrice(mileage) = θ_{0} + (θ_{1} ∗ mileage)$
```
$> python estimator.py 
Please enter mileage: 70000
Estimated price: 6948.720246903524
```
## Trainer
The second program is used to train your model. It will read the dataset and make a linear regression on this data.
Once the linear regression has completed, it saves the variables **theta0** and **theta1** for use in the first program.
I have been using the following formulas:
### $tmpθ_{0} = learningRate ∗ \displaystyle \frac{1}{m} \sum_{i=0}^{m - 1} (estimatePrice(mileage[i]) − price[i])$
### $tmpθ_{1} = learningRate ∗ \displaystyle \frac{1}{m} \sum_{i=0}^{m - 1} (estimatePrice(mileage[i]) − price[i]) ∗ milleage[i]$

```bash
$> python train.py -f data.csv
Model trained in 290 iterations
th0 = 8351.664125972411
th1 = -0.020042055415269825
```
You can also use "trainer" program with **-v** flag to visualize the learning process
<img src="https://github.com/etuffleb/linear_regression/blob/main/Figure_1.png" width="500"/>
<img src="https://github.com/etuffleb/linear_regression/blob/main/Figure_2.png" width="500"/>
