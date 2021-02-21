import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# generate random data-set
x = np.array([2.5,5.1,3.2,8.5,3.5,1.5,9.2,5.5,8.3,2.7,7.7,5.9,4.5,3.3,1.1,8.9,2.5,1.9,6.1,7.4,2.7]).reshape(-1,1)
y=np.array([21,47,27,75,30,20,88,60,81,25,85,62,41,42,17,95,30,24,67,69,30])


# sckit-learn implementation

# Model initialization
regression_model = LinearRegression()
# Fit the data(train the model)
regression_model.fit(x, y)
# Predict
y_predicted = regression_model.predict(x)

# model evaluation
rmse = mean_squared_error(y, y_predicted)
r2 = r2_score(y, y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)
print("predicted:",y_predicted)
# plotting values

# data points
plt.scatter(x, y, s=10)
plt.xlabel('x')
plt.ylabel('y')

# predicted values
plt.plot(x, y_predicted, color='r')
plt.show()

from tkinter import *
  
root = Tk() 
root.geometry("400x400") 
root.title(" Q&A ") 
  
def Take_input(): 
    INPUT = inputtxt.get("1.0", "end-1c") 
    print(INPUT) 
    INPUT=float(INPUT)
    lo=np.array([INPUT]).reshape(-1,1)
    correct=regression_model.predict(lo)
    if(INPUT!=None): 
        Output.insert(END, correct) 
    else: 
        Output.insert(END, "please give input") 
      
l = Label(text = "insert hours") 
inputtxt = Text(root, height = 10, 
                width = 25, 
                bg = "light yellow") 
  
Output = Text(root, height = 5,  
              width = 25,  
              bg = "light cyan") 
  
Display = Button(root, height = 2, 
                 width = 20,  
                 text ="Show", 
                 command = lambda:Take_input()) 
  
l.pack() 
inputtxt.pack() 
Display.pack() 
Output.pack() 
  
mainloop()