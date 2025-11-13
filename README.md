```markdown
# 🚆 Train Ticket Booking Prediction — Deep Neural Network (NumPy From Scratch)

This project implements a **5-layer deep neural network** (built completely from scratch using NumPy) to predict whether a **train ticket booking** will be **successful or not**, based on various features such as day of week, booking time, train popularity, season, travel class, and booking type.

It demonstrates **manual deep learning implementation**, including feature preprocessing, multi-layer forward propagation, backpropagation, vectorized gradient descent, batching (partially implemented), and model evaluation — all **without using TensorFlow, Keras, or PyTorch**.

---

## 📌 Project Highlights

- 🔄 Converts all categorical features into numerical form using custom mapping  
- 🤖 Implements a **5-layer feedforward neural network**  
- 🧮 Uses **sigmoid activation** for all layers  
- 🧠 Fully manual **forward + backward propagation**  
- 🎯 Binary prediction output (0 or 1)  
- 📉 Cost tracking over epochs  
- 📊 Loss curve visualization using Seaborn  
- 🧪 Final model accuracy computed on a separate test dataset  

---

# 🧩 Dataset: `train_ticket_booking_dataset_50000.csv`

The dataset contains 50,000+ booking records. Some of the key features:

| Feature | Description |
|--------|-------------|
| `Day_of_Week` | Day the booking was made |
| `Time_of_Booking` | Morning / Afternoon / Evening / Night |
| `Train_Popularity` | Low / Medium / High |
| `Season` | Normal / Holiday / Festival |
| `Travel_Class` | Sleeper / 3AC / 2AC / 1AC |
| `Booking_Type` | Tatkal / Normal |
| `Booking_Status` | Target (0 = Fail, 1 = Success) |

---

## 🧮 Data Preprocessing

Categorical values are mapped numerically:

### **Day of Week**
```

Mon → 2
Tue → 3
Wed → 4
Thu → 5
Fri → 6
Sat → 7
Sun → 1

```

### **Time of Booking**
```

Morning → 1
Afternoon → 2
Evening → 3
Night → 4

```

### **Train Popularity**
```

Low → 1
Medium → 2
High → 3

```

### **Season**
```

Normal → 1
Holiday → 2
Festival → 3

```

### **Travel Class**
```

Sleeper → 1
3AC → 2
2AC → 3
1AC → 4

```

### **Booking Type**
```

Tatkal → 1
Normal → 2

```

The dataset is then split into:

- **Training set:** first 50,000 rows  
- **Test set:** exported as `Test_dataset.csv`

---

# 🧠 Neural Network Architecture

This model uses **5 fully-connected (dense) layers**:

| Layer | Size | Activation |
|-------|-------|------------|
| Input Layer | 13 features | — |
| Hidden Layer 1 | 13 neurons | Sigmoid |
| Hidden Layer 2 | 10 neurons | Sigmoid |
| Hidden Layer 3 | 7 neurons | Sigmoid |
| Hidden Layer 4 | 3 neurons | Sigmoid |
| Output Layer | 1 neuron | Sigmoid |

---

# 🔢 Forward Propagation Flow

```

X → W1 → A1 → W2 → A2 → W3 → A3 → W4 → A4 → W5 → A5 → Prediction

```

Where:

- `Z = W·X + b`
- `A = sigmoid(Z)`

---

# 🔄 Backpropagation

Gradients for all layers are manually computed:

- `dW5, db5`
- `dW4, db4`
- `dW3, db3`
- `dW2, db2`
- `dW1, db1`

Updating rule:

```

W -= learning_rate * dW
b -= learning_rate * db

````

---

# ⚙️ Training Setup

| Parameter | Value |
|-----------|-------|
| Epochs | 10,000 |
| Batch Size | 64 (partial use) |
| Learning Rate | 0.1 |
| Loss Function | Binary Cross Entropy |
| Activation | Sigmoid (all layers) |

Loss is printed every epoch and stored for graphing.

---

## 📉 Loss Curve

Plotted using:

```python
sns.lineplot(cost)
````

This visualizes model convergence.

---

# 🧪 Prediction Function

The prediction pipeline:

```python
def predict(X):
    Run forward propagation through all 5 layers
    If output >= 0.5 → Predict 1
    Else → Predict 0
```

---

# 🎯 Model Accuracy

The final accuracy is computed over **all rows in the test set**:

```python
Accuracy : XX.XX%
```

(Your output will vary depending on initialization and data.)

---

# 📦 Requirements

Install dependencies:

```bash
pip install numpy pandas seaborn
```

---

# ▶️ Running the Model

Simply execute:

```bash
python train_model.py
```

Or run all cells in your Jupyter notebook / VSCode environment.

---

# 🧠 What You Learn From This Project

* How to preprocess categorical data manually
* How to build a deep neural network *from scratch*
* How forward/backpropagation works internally
* How to implement multi-layer gradient descent in NumPy
* How to evaluate and visualize model performance

---

# 👨‍💻 Author

**Bharath**
Machine Learning Engineer
Exploring how deep learning works from first principles.

---

If you want, I can also generate:

✅ Architecture diagram
✅ Project folder structure
✅ Model explanation in mathematical format
✅ Code optimization + refactoring

Just tell me!
