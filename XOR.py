import numpy as np
import tensorflow as tf

class LogicGate(tf.Module): 
    def __init__(self): 
        super().__init__() 
        self.built = False  

    def __call__(self, x, train=True): 
        if not self.built: 
            input_dim = x.shape[-1]   
            self.w = tf.Variable(tf.random.normal([input_dim, 2]), name="weights")  
            self.b = tf.Variable(tf.zeros([2]), name="bias")  
            self.w_out = tf.Variable(tf.random.normal([2, 1]), name="weights_out")  
            self.b_out = tf.Variable(tf.zeros([1]), name="bias_out") 
            self.built = True 

        z_hidden = tf.add(tf.matmul(x, self.w), self.b)  
        hidden_output = tf.tanh(z_hidden)  
        
        z_out = tf.add(tf.matmul(hidden_output, self.w_out), self.b_out)
        return tf.sigmoid(z_out)  

def compute_loss(y_pred, y_true): 
    return tf.reduce_mean(tf.square(y_pred - y_true)) 

def xor_gate(a, b):
    return a ^ b  

def test_xor_gate():
    assert xor_gate(0, 0) == 0  
    assert xor_gate(0, 1) == 1 
    assert xor_gate(1, 0) == 1 
    assert xor_gate(1, 1) == 0  
    print("All tests passed! ")


def train_model(model, x_train, y_train, learning_rate=0.01, epochs=20000): 
    for epoch in range(epochs): 
        with tf.GradientTape() as tape: 
            y_pred = model(x_train)  
            loss = compute_loss(y_pred, y_train)  

        grads = tape.gradient(loss, model.variables) 
        for g, v in zip(grads, model.variables): 
            v.assign_sub(learning_rate * g) 

        if epoch % 1000 == 0: 
            acc = compute_accuracy(model, x_train, y_train) 
            print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}, Accuracy: {acc:.4f}") 

def compute_accuracy(model, x, y_true): 
    y_pred = model(x, train=False) 
    y_pred_rounded = tf.round(y_pred) 
    correct = tf.equal(y_pred_rounded, y_true) 
    return tf.reduce_mean(tf.cast(correct, tf.float32)).numpy() 

xor_table = np.array([[0, 0, 0], 
                      [1, 0, 1], 
                      [0, 1, 1], 
                      [1, 1, 0]], dtype=np.float32) 

x_train = xor_table[:, :2]   
y_train = xor_table[:, 2:]   

model = LogicGate() 
train_model(model, x_train, y_train) 

w1 = model.w.numpy()
w_out = model.w_out.numpy()
b = model.b.numpy()
b_out = model.b_out.numpy()
print(f"\nLearned weight for w1: {w1}") 
print(f"Learned weight for w_out: {w_out}") 
print(f"Learned bias for b: {b}") 
print(f"Learned bias for b_out: {b_out}\n")


y_pred = model(x_train, train=False).numpy().round().astype(np.uint8) 
print("Predicted Truth Table:") 
print(np.column_stack((xor_table[:, :2], y_pred))) 
