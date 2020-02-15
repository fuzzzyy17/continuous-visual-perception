x_data = [1, 2, 3]
y_data = [2, 4, 6]

w = 1

def forward(x):    # forward pass in net
    return x*w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2

def gradient(x, y): # d_loss/d_w
    return 2 * x * (x * w - y)


print ("pre-training prediction: 4 hrs work = ", forward(4), "marks")

for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - 0.01 * grad   # 0.01 = alpha
        print ("\tgrad: ", x_val, y_val, grad)
        l = loss(x_val, y_val)
    
    print("progress: ", epoch, "w = ", w, "loss = ", l)

print ("\npost-training prediction: 4 hrs work = ", forward(4), "marks")