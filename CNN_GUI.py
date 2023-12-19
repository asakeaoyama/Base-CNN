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
    loss_var.set(f"Current Loss(訓練損失值) :  {loss}")
    epoch_var.set(f"Epoch(訓練期) {epoch}, Acc(訓練正確率): {acc}%")

def img_num(a):
    l = len(a)
    s = ''
    for i in range(0, 4 - l):
        s += '0'
    s += a
    return s

def open_file_manager(type):
    global selected_train_folder_path
    global selected_test_folder_path
    if (type == "train"):
        folder_path = filedialog.askdirectory(title="Select Train Folder")
        train_file_manager_label.config(text=f'Folder: {folder_path}')
        selected_train_folder_path = folder_path
    elif (type == "test"):
        folder_path = filedialog.askdirectory(title="Select Test Folder")
        test_file_manager_label.config(text=f'Folder: {folder_path}')
        selected_test_folder_path = folder_path

def cancel_training_callback():
    global cancel_training
    cancel_training = True

def train_model():
    print('start')
    now = datetime.datetime.now()
    print(now)
    final_ans.set("最終預測結果 : ")
    spend_time.set("花費時間 : ")
    update(0,0,0)
    global cancel_training
    epochs = int(epochs_entry.get())
    hidden_size = int(hidden_layer_entry.get())
    learning_rate = float(learning_rate_entry.get())
    stride = int(stride_entry.get())
    trainSize = int(train_entry.get())
    typeSize = int(type_entry.get())
    activation = activation_dropdown.get()
    filter = kernel_dropdown.get()
    img_arr = []
    ans_arr = []
    total_size = typeSize*trainSize
    leftRange, rightRange = -0.1, 0.1

    for case in range(trainSize):
        for num in range(typeSize):
            if(selected_train_folder_path[len(selected_train_folder_path)-5:len(selected_train_folder_path)] == "mbers"):
                path = selected_train_folder_path + '/{}'.format(num) + '/{}.png'.format(case)
                convly = ConventionalConv(path, True, 2, 8, stride, filter)
                img_arr.append(convly.finalOutput)
                ans = convly.makeAnswer(num, typeSize)
                ans_arr.append(ans)
            elif(selected_train_folder_path[len(selected_train_folder_path)-5:len(selected_train_folder_path)] == "train"):
                path = selected_train_folder_path + '/{}'.format(num) + '/{}.jpg'.format(img_num(str(case)))
                convly = ConventionalConv(path, False, 2, 8, stride, filter)
                img_arr.append(convly.finalOutput)
                ans2 = convly.makeAnswer(num, typeSize)
                ans_arr.append(ans2)

    input_size = len(convly.finalOutput)
    output_size = typeSize
    CW = createWeights(leftRange, rightRange, input_size, hidden_size, output_size, total_size)

    for epoch in range(epochs + 1):
        if cancel_training:
            update(0, 0, 0)  # Update UI to show that training is canceled
            break

        fc = fullyConnected(img_arr, ans_arr, CW.weights_input_hidden, CW.bias_input_hidden,
                             CW.weights_hidden_output, CW.bias_hidden_output, activation, "MSE")
        if (activation == "Sigmoid"):
            fc.updateWeightSigmoid(fc.error, fc.hidden_layer_output, img_arr, CW.weights_hidden_output,
                               fc.input_layer_hidden, CW.bias_hidden_output, CW.weights_input_hidden,
                               CW.bias_input_hidden, learning_rate)
        elif (activation == "ReLU"):
            fc.updateWeightRelu(fc.error, fc.hidden_layer_output, img_arr, CW.weights_hidden_output,
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
    for case in range(trainSize):
        for num in range(typeSize):
            if (selected_test_folder_path[len(selected_test_folder_path)-5:len(selected_test_folder_path)] == "mbers"):
                path = selected_test_folder_path + '/{}'.format(num) + '/{}.png'.format(case)
                convly2 = ConventionalConv(path, True, 2, 8, stride, filter)
                img_arr2.append(convly2.finalOutput)
                ans2 = convly2.makeAnswer(num, typeSize)
                ans_arr2.append(ans2)
            elif (selected_test_folder_path[len(selected_test_folder_path)-5:len(selected_test_folder_path)] == "valid"):
                path = selected_test_folder_path + '/{}'.format(num) + '/{}.jpg'.format(img_num(str(case)))
                convly2 = ConventionalConv(path, False, 2, 8, stride, filter)
                img_arr2.append(convly2.finalOutput)
                ans2 = convly2.makeAnswer(num, typeSize)
                ans_arr2.append(ans2)

    fc2 = fullyConnected(img_arr2, ans_arr2, CW.max_weights_input_hidden, CW.max_bias_input_hidden,
                         CW.max_weights_hidden_output, CW.max_bias_hidden_output, activation, "MSE")
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
selected_train_folder_path = ""
selected_test_folder_path = ""

# 輸入 stride
ttk.Label(root, text="Stride(步長):").grid(column=0, row=0, padx=10, pady=10)
stride_entry = ttk.Entry(root)
stride_entry.grid(column=1, row=0, padx=10, pady=10)

# 輸入 epoch
ttk.Label(root, text="Epochs(期):").grid(column=0, row=1, padx=10, pady=10)
epochs_entry = ttk.Entry(root)
epochs_entry.grid(column=1, row=1, padx=10, pady=10)

# Hidden layer number
ttk.Label(root, text="Hidden Layer Neuron(隱藏層神經元個數):").grid(column=0, row=2, padx=10, pady=10)
hidden_layer_entry = ttk.Entry(root)
hidden_layer_entry.grid(column=1, row=2, padx=10, pady=10)

# 輸入學習率
ttk.Label(root, text="Learning Rate(學習率):").grid(column=0, row=3, padx=10, pady=10)
learning_rate_entry = ttk.Entry(root)
learning_rate_entry.grid(column=1, row=3, padx=10, pady=10)

# 輸入每個種類的個數
ttk.Label(root, text="Train size(個數/種):").grid(column=0, row=4, padx=10, pady=10)
train_entry = ttk.Entry(root)
train_entry.grid(column=1, row=4, padx=10, pady=10)

# 輸入種類的個數
ttk.Label(root, text="Type size(種類個數):").grid(column=0, row=5, padx=10, pady=10)
type_entry = ttk.Entry(root)
type_entry.grid(column=1, row=5, padx=10, pady=10)

# 顯示選擇訓練的檔案路徑
train_file_manager_label = ttk.Label(root, text="Selected Train File: ")
train_file_manager_label.grid(column=2, row=1, columnspan=2, pady=10)

# 選擇訓練檔案的按鈕
train_file_manager_button = ttk.Button(root, text="選擇訓練檔案路徑", command=lambda: open_file_manager("train"))
train_file_manager_button.grid(column=2, row=2, columnspan=2, pady=10)

# 顯示測試訓練的檔案路徑
test_file_manager_label = ttk.Label(root, text="Selected Test File: ")
test_file_manager_label.grid(column=2, row=3, columnspan=2, pady=10)

# 選擇測試檔案的按鈕
test_file_manager_button = ttk.Button(root, text="選擇測試檔案路徑", command=lambda: open_file_manager("test"))
test_file_manager_button.grid(column=2, row=4, columnspan=2, pady=10)

# 選擇激活函數的下拉選單
ttk.Label(root, text="Activation Function(激活函數):").grid(column=2, row=5, padx=10, pady=10)
activation_functions = ["Sigmoid", "ReLU"]
activation_var = tk.StringVar()
activation_dropdown = ttk.Combobox(root, textvariable=activation_var, values=activation_functions)
activation_dropdown.grid(column=2, row=6, padx=10, pady=10)
activation_dropdown.set(activation_functions[0])  # Set default activation function

# 選擇濾波器的下拉選單
ttk.Label(root, text="Kernel/Filter(卷積核):").grid(column=2, row=7, padx=10, pady=10)
kernel_types = ["Normal Filter(01)", "Sobel Filter(-11)"]
kernel_var = tk.StringVar()
kernel_dropdown = ttk.Combobox(root, textvariable=kernel_var, values=kernel_types)
kernel_dropdown.grid(column=2, row=8, padx=10, pady=10)
kernel_dropdown.set(kernel_types[0])  # Set default activation function

# 顯示 loss 的地方
loss_var = tk.StringVar()
loss_var.set("Current Loss(訓練損失值) : ")
loss_label = ttk.Label(root, textvariable=loss_var)
loss_label.grid(column=0, row=6, columnspan=1, pady=10)

# 顯示 epoch 的地方
epoch_var = tk.StringVar()
epoch_var.set("Epoch(訓練期) : ")
epoch_label = ttk.Label(root, textvariable=epoch_var)
epoch_label.grid(column=0, row=7, columnspan=1, pady=20)

# 最終結果
final_ans = tk.StringVar()
final_ans.set("Final Accuracy(最終正確率) : ")
ans_label = ttk.Label(root, textvariable=final_ans)
ans_label.grid(column=0, row=8, columnspan=1, pady=20)

# 總時長
spend_time = tk.StringVar()
spend_time.set("Time(總時長) : ")
time_label = ttk.Label(root, textvariable=spend_time)
time_label.grid(column=0, row=9, columnspan=1, pady=20)

# start train button
train_button = ttk.Button(root, text="開始訓練", command=start_training_thread)
train_button.grid(column=2, row=9, columnspan=2, pady=20)

# cancel train button
cancel_button = ttk.Button(root, text="取消訓練", command=cancel_training_callback)
cancel_button.grid(column=2, row=10, columnspan=2, pady=20)

root.mainloop()
