import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def g(x):
    return 4 * sigmoid(x) - 2

def h(x):
    return 2 * sigmoid(x) - 1

def sigmoid_prime(y):
    return y * (1 - y)

def g_prime(net):
    return 1 - (g(net)**2) / 4

def h_prime(s):
    return (1 - h(s)**2) / 2

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        self.W_in_x = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.W_in_h = np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        self.b_in = np.zeros((hidden_size, 1))
        
        self.W_out_x = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.W_out_h = np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        self.b_out = np.zeros((hidden_size, 1))
        
        self.W_cell_x = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        self.W_cell_h = np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        self.b_cell = np.zeros((hidden_size, 1))
        
        self.W_final = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
        self.b_final = np.zeros((output_size, 1))
    
    def forward(self, X):
        T = X.shape[0]
        self.y_in = np.zeros((T, self.hidden_size, 1))
        self.y_out = np.zeros((T, self.hidden_size, 1))
        self.state = np.zeros((T + 1, self.hidden_size, 1))
        self.y_cell = np.zeros((T, self.hidden_size, 1))
        self.net_in = np.zeros((T, self.hidden_size, 1))
        self.net_out = np.zeros((T, self.hidden_size, 1))
        self.net_cell = np.zeros((T, self.hidden_size, 1))
        y_prev = np.zeros((self.hidden_size, 1))
        
        for t in range(T):
            x_t = X[t].reshape(self.input_size, 1)
            
            self.net_in[t] = self.W_in_x @ x_t + self.W_in_h @ y_prev + self.b_in
            self.y_in[t] = sigmoid(self.net_in[t])
            
            self.net_out[t] = self.W_out_x @ x_t + self.W_out_h @ y_prev + self.b_out
            self.y_out[t] = sigmoid(self.net_out[t])
            
            self.net_cell[t] = self.W_cell_x @ x_t + self.W_cell_h @ y_prev + self.b_cell
            self.state[t + 1] = self.state[t] + self.y_in[t] * g(self.net_cell[t])
            
            self.y_cell[t] = self.y_out[t] * h(self.state[t + 1])
            y_prev = self.y_cell[t]
        
        net_final = self.W_final @ self.y_cell[-1] + self.b_final
        y_final = net_final  # Linear output
        return y_final
    
    def backward(self, X, y_true, y_pred):
        T = X.shape[0]
        delta_final = y_pred - y_true  # Gradient for MSE
        
        dW_final = delta_final @ self.y_cell[-1].T
        db_final = delta_final
        
        delta_y_cell = np.zeros((T, self.hidden_size, 1))
        delta_state = np.zeros((T + 1, self.hidden_size, 1))
        
        delta_y_cell[T - 1] = self.W_final.T @ delta_final
        
        dW_in_x = np.zeros_like(self.W_in_x)
        dW_in_h = np.zeros_like(self.W_in_h)
        db_in = np.zeros_like(self.b_in)
        
        dW_out_x = np.zeros_like(self.W_out_x)
        dW_out_h = np.zeros_like(self.W_out_h)
        db_out = np.zeros_like(self.b_out)
        
        dW_cell_x = np.zeros_like(self.W_cell_x)
        dW_cell_h = np.zeros_like(self.W_cell_h)
        db_cell = np.zeros_like(self.b_cell)
        
        for t in range(T - 1, -1, -1):
            delta_out_t = sigmoid_prime(self.y_out[t]) * h(self.state[t + 1]) * delta_y_cell[t]
            delta_state[t + 1] = self.y_out[t] * h_prime(self.state[t + 1]) * delta_y_cell[t]
            
            delta_in_t = sigmoid_prime(self.y_in[t]) * g(self.net_cell[t]) * delta_state[t + 1]
            delta_cell_t = g_prime(self.net_cell[t]) * self.y_in[t] * delta_state[t + 1]
            
            x_t = X[t].reshape(self.input_size, 1)
            y_prev = self.y_cell[t - 1] if t > 0 else np.zeros((self.hidden_size, 1))
            
            dW_in_x += delta_in_t @ x_t.T
            dW_in_h += delta_in_t @ y_prev.T
            db_in += delta_in_t
            
            dW_out_x += delta_out_t @ x_t.T
            dW_out_h += delta_out_t @ y_prev.T
            db_out += delta_out_t
            
            dW_cell_x += delta_cell_t @ x_t.T
            dW_cell_h += delta_cell_t @ y_prev.T
            db_cell += delta_cell_t
            
            delta_y_prev = self.W_in_h.T @ delta_in_t + self.W_out_h.T @ delta_out_t + self.W_cell_h.T @ delta_cell_t
            
            if t > 0:
                delta_y_cell[t - 1] += delta_y_prev
                delta_state[t] = delta_state[t + 1]  # Constant error flow
        
        self.W_in_x -= self.lr * dW_in_x
        self.W_in_h -= self.lr * dW_in_h
        self.b_in -= self.lr * db_in
        
        self.W_out_x -= self.lr * dW_out_x
        self.W_out_h -= self.lr * dW_out_h
        self.b_out -= self.lr * db_out
        
        self.W_cell_x -= self.lr * dW_cell_x
        self.W_cell_h -= self.lr * dW_cell_h
        self.b_cell -= self.lr * db_cell
        
        self.W_final -= self.lr * dW_final
        self.b_final -= self.lr * db_final

def generate_adding_dataset(num_samples, seq_length=100, seed=42):
    np.random.seed(seed)
    X = np.zeros((num_samples, seq_length, 2))
    y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        X[i, :, 0] = np.random.uniform(0, 1, seq_length)
        markers = np.zeros(seq_length)
        pos1 = np.random.randint(0, seq_length // 2)
        pos2 = np.random.randint(seq_length // 2, seq_length)
        markers[pos1] = 1
        markers[pos2] = 1
        X[i, :, 1] = markers
        y[i] = X[i, pos1, 0] + X[i, pos2, 0]
    return X, y

train_X, train_y = generate_adding_dataset(1000)
test_X, test_y = generate_adding_dataset(200)

input_size = 2
hidden_size = 4  # Small for demo
output_size = 1
lstm = LSTM(input_size, hidden_size, output_size, learning_rate=0.1)

epochs = 10
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(train_X)):
        X_seq = train_X[i]
        y_true = train_y[i].reshape(1, 1)
        y_pred = lstm.forward(X_seq)
        loss = np.mean((y_pred - y_true)**2)
        total_loss += loss
        lstm.backward(X_seq, y_true, y_pred)
    print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(train_X)}")

test_loss = 0
for i in range(len(test_X)):
    X_seq = test_X[i]
    y_true = test_y[i].reshape(1, 1)
    y_pred = lstm.forward(X_seq)
    loss = np.mean((y_pred - y_true)**2)
    test_loss += loss
print(f"Test Loss: {test_loss / len(test_X)}")