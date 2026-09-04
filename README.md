H3>ENTER YOUR NAME V.Shreya</H3>
<H3>ENTER YOUR REGISTER NO.212224230266</H3>
<H3>EX. NO.5</H3>
<H3>DATE:4.9.26</H3>
<H1 ALIGN =CENTER>Implementation of XOR  using RBF</H1>
<H3>Aim:</H3>
To implement a XOR gate classification using Radial Basis Function  Neural Network.

<H3>Theory:</H3>
<P>Exclusive or is a logical operation that outputs true when the inputs differ.For the XOR gate, the TRUTH table will be as follows XOR truth table </P>

<P>XOR is a classification problem, as it renders binary distinct outputs. If we plot the INPUTS vs OUTPUTS for the XOR gate, as shown in figure below </P>




<P>The graph plots the two inputs corresponding to their output. Visualizing this plot, we can see that it is impossible to separate the different outputs (1 and 0) using a linear equation.
A Radial Basis Function Network (RBFN) is a particular type of neural network. The RBFN approach is more intuitive than MLP. An RBFN performs classification by measuring the input’s similarity to examples from the training set. Each RBFN neuron stores a “prototype”, which is just one of the examples from the training set. When we want to classify a new input, each neuron computes the Euclidean distance between the input and its prototype. Thus, if the input more closely resembles the class A prototypes than the class B prototypes, it is classified as class A ,else class B.
A Neural network with input layer, one hidden layer with Radial Basis function and a single node output layer (as shown in figure below) will be able to classify the binary data according to XOR output.
</P>





<H3>ALGORITHM:</H3>
Step 1: Initialize the input  vector for you bit binary data<Br>
Step 2: Initialize the centers for two hidden neurons in hidden layer<Br>
Step 3: Define the non- linear function for the hidden neurons using Gaussian RBF<br>
Step 4: Initialize the weights for the hidden neuron <br>
Step 5 : Determine the output  function as 
                 Y=W1*φ1 +W1 *φ2 <br>
Step 6: Test the network for accuracy<br>
Step 7: Plot the Input space and Hidden space of RBF NN for XOR classification.

<H3>PROGRAM:</H3>

```py
import numpy as np
import matplotlib.pyplot as plt

# XOR inputs and outputs
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

# RBF centers
centers = np.array([
    [0, 0],
    [1, 1]
])

sigma = 0.5

# Gaussian RBF
def rbf(x, c):
    return np.exp(-np.sum((x - c) ** 2) / (2 * sigma ** 2))

# Transform inputs
phi = np.array([
    [rbf(x, centers[0]), rbf(x, centers[1])]
    for x in X
])

print("Transformed inputs:")
print(phi)

# Plot transformed inputs
plt.figure(figsize=(7, 5))

for i in range(len(X)):
    if y[i] == 0:
        plt.scatter(phi[i, 0], phi[i, 1], label="Class_0" if i == 0 else "")
    else:
        plt.scatter(phi[i, 0], phi[i, 1], label="Class_1" if i == 1 else "")

# Separating line
x_line = np.linspace(0, 1.1, 100)
y_line = 1.17 - x_line

plt.plot(x_line, y_line, "--", label="Separating hyperplane")

plt.xlabel("φ1")
plt.ylabel("φ2")
plt.title("Transformed Inputs: Linearly Separable")
plt.legend()
plt.show()
```

<H3>OUTPUT:</H3>

<img width="1013" height="898" alt="image" src="https://github.com/user-attachments/assets/472b4ee1-1a3f-41f8-95d2-668cdd000e81" />


<H3>Result:</H3>
Thus , a Radial Basis Function Neural Network is implemented to classify XOR data.








