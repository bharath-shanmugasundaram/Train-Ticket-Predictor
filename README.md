# 🚆 Train Ticket Booking Prediction — Deep Neural Network (NumPy From Scratch)

This project builds a **5-layer deep neural network** completely from scratch using **NumPy**, designed to predict whether a **train ticket booking** will be successful or not. The dataset involves several categorical and numerical fields, all of which are transformed manually before training the neural network.

The project demonstrates how deep learning works internally:
✔ Manual feature preprocessing
✔ Multi-layer forward propagation
✔ Multi-layer backpropagation
✔ Gradient descent optimization
✔ Training loss monitoring
✔ Manual binary classification inference

Everything is implemented **without TensorFlow, PyTorch, or Scikit-learn**.

---

# 📊 Dataset Overview

Dataset used:
**\`train_ticket_booking_dataset_50000.csv\`**

It contains **50,000+ rows** with the following features:

| Feature | Description |
|--------|-------------|
| Day_of_Week | Day of booking |
| Time_of_Booking | Morning / Afternoon / Evening / Night |
| Train_Popularity | Low / Medium / High |
| Season | Normal / Holiday / Festival |
| Travel_Class | Sleeper / 3AC / 2AC / 1AC |
| Booking_Type | Tatkal / Normal |
| Booking_Status | Final output label (0 = Fail, 1 = Success) |

---

# 🔄 Data Preprocessing (Manual Encoding)

All categorical values are mapped manually to integers:

### **Day\_of\_Week**
Mon → 2
Tue → 3
Wed → 4
Thu → 5
Fri → 6
Sat → 7
Sun → 1

### **Time\_of\_Booking**
Morning → 1
Afternoon → 2
Evening → 3
Night → 4

### **Train\_Popularity**
Low → 1
Medium → 2
High → 3

### **Season**
Normal → 1
Holiday → 2
Festival → 3

### **Travel\_Class**
Sleeper → 1
3AC → 2
2AC → 3
1AC → 4

### **Booking\_Type**
Tatkal → 1
Normal → 2

### **Splitting the Dataset**
- First **50,000** rows → **Training set**
- Remaining rows → **Test set**, saved as \`Test\_dataset.csv\`

---

# 🧠 Neural Network Architecture

The network contains **5 hidden layers**, implemented with plain NumPy:

| Layer | Neurons | Activation |
|-------|------------|-------------|
| Input Layer | 13 features | — |
| Hidden Layer 1 | 13 neurons | Sigmoid |
| Hidden Layer 2 | 10 neurons | Sigmoid |
| Hidden Layer 3 | 7 neurons | Sigmoid |
| Hidden Layer 4 | 3 neurons | Sigmoid |
| Output Layer | 1 neuron | Sigmoid |

The final output is binary:
- \`0\` → Booking unsuccessful
- \`1\` → Booking successful

---

# 🧮 Activation Functions

## Sigmoid
\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

## Sigmoid Derivative
\[
\sigma'(a) = a(1 - a)
\]

---

# 🔢 Forward Propagation

Forward pass executes in this order:
Input X
↓
Z1 = W1·X + b1
A1 = sigmoid(Z1)
↓
Z2 = W2·A1 + b2
A2 = sigmoid(Z2)
↓
Z3 = W3·A2 + b3
A3 = sigmoid(Z3)
↓
Z4 = W4·A3 + b4
A4 = sigmoid(Z4)
↓
Z5 = W5·A4 + b5
A5 = sigmoid(Z5) ← Final output

---

# 🔄 Backpropagation

Error is propagated back through all 5 layers using:

- \`dZ5\`, \`dW5\`, \`db5\`
- \`dZ4\`, \`dW4\`, \`db4\`
- \`dZ3\`, \`dW3\`, \`db3\`
- \`dZ2\`, \`dW2\`, \`db2\`
- \`dZ1\`, \`dW1\`, \`db1\`

### Gradient Descent Update:
\[
W := W - \alpha \cdot dW
\]
\[
b := b - \alpha \cdot db
\]

Where **α = learning rate**.

---

# ⚙️ Training Configuration

| Hyperparameter | Value |
|----------------|--------|
| Epochs | 10,000 |
| Learning Rate | 0.1 |
| Batch Size | 64 |
| Loss function | Binary Cross-Entropy |
| Activation | Sigmoid (all layers) |

Loss is recorded at each epoch and stored in \`cost\`.

---

# 📉 Loss Visualization

Loss curve is plotted using:

\`\`\`python
sns.lineplot(cost)
\`\`\`
This shows whether the model is converging properly.

---

# 🧪 Prediction Pipeline

Once trained, each test sample is passed through all 5 layers.
Binary classification rule:

If A5 >= 0.5 → Predict 1
Else → Predict 0

This is done for all rows in the test dataset to compute accuracy.

---

# 🎯 Final Accuracy

Accuracy is calculated using:

$$
\text{Accuracy} = \frac{\text{correct\_predictions}}{\text{total\_test\_samples}} \times 100
$$

You will see output like:

\`\`\`
Accuracy : 87.24%
\`\`\`
(Your accuracy may vary depending on initialization and randomness.)

---

# 📦 Requirements

Install required libraries:

\`\`\`
pip install numpy pandas seaborn
\`\`\`

---

# ▶️ How to Run

To execute the complete training pipeline:

**Option 1 — Run as Python script**
\`\`\`
python train_model.py
\`\`\`

**Option 2 — Run Jupyter Notebook / VSCode**
Execute all cells in order.

---

# 📘 What You Will Learn

By studying this project, you will understand:

* How to **manually encode categorical ML dataset features**
* How to build a **deep feedforward neural network from scratch**
* How **forward propagation** works mathematically
* How **backpropagation** works layer-by-layer
* How to apply **gradient descent** without ML libraries
* How to measure accuracy of a classifier
* How to visualize loss curves
* How real-world categorical data is transformed for machine learning

This project is ideal for students learning **deep learning fundamentals**.

---

# 👨‍💻 Author

**Bharath**
Machine Learning Developer
Focused on understanding and building ML systems from scratch to gain deep foundational knowledge.
