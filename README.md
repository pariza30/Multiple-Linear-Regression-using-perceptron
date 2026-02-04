# Multiple Linear Regression Using Perceptron Learning Rule

## Overview
This project implements **Multiple Linear Regression** using the **Perceptron-style weight update rule**.  
Instead of solving regression directly with the normal equation, the model learns weights iteratively using **Stochastic Gradient Descent (SGD)**, similar to how a perceptron learns.

The implementation is inspired by the perceptron learning approach used for logic gates, extended here for continuous regression output.

---

## Objective
- Predict a continuous target value (**income**) using multiple input features (**age**, **experience**)
- Train the regression model using iterative perceptron weight updates
- Demonstrate how linear regression can be learned through gradient-based optimization

---

## Dataset
The dataset contains:

- **age** → Input feature  
- **experience** → Input feature  
- **income** → Target output  

---

## Model Equation
The learned regression model follows:

\[
y = w_1x_1 + w_2x_2 + b
\]

Where:  
- \(x_1\) = age  
- \(x_2\) = experience  
- \(b\) = bias term  
- \(w_1, w_2\) = learned weights  

---

## Learning Method

### Prediction
\[
y_{pred} = w^Tx
\]

### Error
\[
error = y - y_{pred}
\]

### Weight Update Rule (Perceptron Style)
\[
w = w + \eta \cdot error \cdot x
\]

This is performed using **Stochastic Gradient Descent**, meaning weights are updated after every training sample.

---

## Implementation Steps
1. Load dataset  
2. Normalize input features for stable learning  
3. Add bias input  
4. Initialize weights randomly  
5. Train using perceptron learning rule  
6. Print mean squared error during training  
7. Compare predicted vs actual income values  

---

## Sample Output
After training, predictions closely match actual values:

Actual Income: 30450 | Predicted Income: 30001  
Actual Income: 35670 | Predicted Income: 33839  
Actual Income: 47830 | Predicted Income: 47242  

---

## Technologies Used
- Python  
- NumPy  
- Pandas  

---

## How to Run

```bash
pip install numpy pandas
python regression_perceptron.py
