import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
from convolution.conventionalConvolutionOOP import ConventionalConv
from fullyconnected.createweight import createWeights
from fullyconnected.fullyconnectedOOP import fullyConnected
import datetime

cancel_training = False

def update(loss, epoch, acc):
    loss_var.set(f"Current Loss :  {loss}")
    epoch_var.set(f"Epoch {epoch}, Acc: {acc}%")

def img_num(a):
    l = len(a)
    s = ''
    for i in range(0, 4 - l):
        s += '0'
    s += a
    return s

def open_file_manager():
    global selected_folder_path
    folder_path = filedialog.askdirectory(title="Select Folder")
    if folder_path:
        file_manager_label.config(text=f'Folder: {folder_path}')
        selected_folder_path = folder_path

def cancel_training_callback():
    global cancel_training
    cancel_training = True

def train_model():
    print('start')
    now = datetime.datetime.now()
    print(now)
    update(0,0,0)
    global cancel_training
    epochs = int(epochs_entry.get())
    hidden_size = int(hidden_layer_entry.get())
    learning_rate = float(learning_rate_entry.get())
    stride = int(stride_entry.get())
    trainSize = int(train_entry.get())
    img_arr = []
    ans_arr = []
    typeSize = 3
    total_size = typeSize*trainSize
    leftRange, rightRange = -0.1, 0.1

    for case in range(trainSize):
        for num in range(0, typeSize):
            path = selected_folder_path + '/{}'.format(num) + '/{}.jpg'.format(img_num(str(case)))
            convly = ConventionalConv(path, False, 2, 8, stride)
            img_arr.append(convly.finalOutput)
            ans = convly.makeAnswer(num, typeSize)
            ans_arr.append(ans)

    input_size = len(convly.finalOutput)
    output_size = typeSize
    CW = createWeights(leftRange, rightRange, input_size, hidden_size, output_size, total_size)

    for epoch in range(epochs + 1):
        if cancel_training:
            update(0, 0, 0)  # Update UI to show that training is canceled
            break

        fc = fullyConnected(img_arr, ans_arr, CW.weights_input_hidden, CW.bias_input_hidden,
                             CW.weights_hidden_output, CW.bias_hidden_output, "sigmoid", "MSE")
        fc.updateWeightSigmoid(fc.error, fc.hidden_layer_output, img_arr, CW.weights_hidden_output,
                               fc.input_layer_hidden, CW.bias_hidden_output, CW.weights_input_hidden,
                               CW.bias_input_hidden, learning_rate)
        trainAccuracy = CW.maxWeight(CW.weights_input_hidden, CW.weights_hidden_output, CW.bias_input_hidden,
                                     CW.bias_hidden_output, epoch, fc.hidden_layer_output, ans_arr)
        if epoch % 100 == 0:
            update(fc.loss, epoch, trainAccuracy)

        if fc.loss <= 0.005:
            update(fc.loss, epoch, trainAccuracy)
            break

    img_arr2 = []
    ans_arr2 = []
    for case in range(200, 200 + trainSize):
        for num in range(0, typeSize):
            path = selected_folder_path + '/{}'.format(num) + '/{}.jpg'.format(img_num(str(case)))
            convly2 = ConventionalConv(path, False, 2, 8, stride)
            img_arr2.append(convly2.finalOutput)
            ans2 = convly2.makeAnswer(num, typeSize)
            ans_arr2.append(ans2)
    fc2 = fullyConnected(img_arr2, ans_arr2, CW.max_weights_input_hidden, CW.max_bias_input_hidden,
                         CW.max_weights_hidden_output, CW.max_bias_hidden_output, "sigmoid", "MSE")
    final_ans.set("最終預測結果 : " + str(CW.acc(fc2.hidden_layer_output, ans_arr2)) + '%')
    end_time = datetime.datetime.now()
    spent = end_time - now
    spend_time.set("花費時間 : " + str(spent))
def start_training_thread():
    global cancel_training
    # Reset cancel flag before starting a new training thread
    cancel_training = False
    # Start the training thread
    training_thread = threading.Thread(target=train_model)
    training_thread.start()

# ----------------------------------------------------------
root = tk.Tk()
root.title("NKNU CNN底層實作")
selected_folder_path = ""

# 輸入 stride
ttk.Label(root, text="Stride:").grid(column=0, row=0, padx=10, pady=10)
stride_entry = ttk.Entry(root)
stride_entry.grid(column=1, row=0, padx=10, pady=10)

# 輸入 epoch
ttk.Label(root, text="Epochs:").grid(column=0, row=1, padx=10, pady=10)
epochs_entry = ttk.Entry(root)
epochs_entry.grid(column=1, row=1, padx=10, pady=10)

# Hidden layer number
ttk.Label(root, text="Hidden Layer Number:").grid(column=0, row=2, padx=10, pady=10)
hidden_layer_entry = ttk.Entry(root)
hidden_layer_entry.grid(column=1, row=2, padx=10, pady=10)

# 輸入學習率
ttk.Label(root, text="Learning Rate:").grid(column=0, row=3, padx=10, pady=10)
learning_rate_entry = ttk.Entry(root)
learning_rate_entry.grid(column=1, row=3, padx=10, pady=10)

ttk.Label(root, text="Train size:").grid(column=0, row=4, padx=10, pady=10)
train_entry = ttk.Entry(root)
train_entry.grid(column=1, row=4, padx=10, pady=10)


# 選擇激活函數的下拉選單
ttk.Label(root, text="                          Activation Function:").grid(column=2, row=3, padx=10, pady=10)
activation_functions = ["Sigmoid", "ReLU"]
activation_var = tk.StringVar()
activation_dropdown = ttk.Combobox(root, textvariable=activation_var, values=activation_functions)
activation_dropdown.grid(column=2, row=4, padx=10, pady=10)
activation_dropdown.set(activation_functions[0])  # Set default activation function

# 選擇檔案的按鈕
file_manager_button = ttk.Button(root, text="選擇檔案路徑", command=open_file_manager)
file_manager_button.grid(column=2, row=2, columnspan=2, pady=10)

# 顯示選擇的檔案路徑
file_manager_label = ttk.Label(root, text="Selected File: ")
file_manager_label.grid(column=2, row=1, columnspan=2, pady=10)

# 顯示 loss 的地方
loss_var = tk.StringVar()
loss_var.set("Current Loss : ")
loss_label = ttk.Label(root, textvariable=loss_var)
loss_label.grid(column = 0, row = 5, columnspan = 1, pady = 10)

epoch_var = tk.StringVar()
epoch_var.set("Epoch : ")
epoch_label = ttk.Label(root, textvariable=epoch_var)
epoch_label.grid(column = 0, row = 6, columnspan = 1, pady = 20)

# 最終結果
final_ans = tk.StringVar()
ans_label = ttk.Label(root, textvariable=final_ans)
ans_label.grid(column = 0, row = 7, columnspan = 1, pady = 20)

spend_time = tk.StringVar()
time_label = ttk.Label(root, textvariable = spend_time)
time_label.grid(column = 0, row = 8, columnspan = 1, pady = 20)

train_button = ttk.Button(root, text="開始訓練", command=start_training_thread)
train_button.grid(column = 2, row = 6, columnspan = 2, pady = 20)

# Add Cancel button
cancel_button = ttk.Button(root, text="取消訓練", command=cancel_training_callback)
cancel_button.grid(column = 2, row = 7 , pady=20)

root.mainloop()
