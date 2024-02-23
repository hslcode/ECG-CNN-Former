# 这是一个MIT-BIH数据集的测试1 脚本。
import data
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
import random
from STFT import STFT
import matplotlib.pyplot as plt
from transformer import My_Transformer
from transformer import Net
from result_display import ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter

log_dir = 'runs/logs/'
save_dir = 'runs/'
# save_dir = 'runs/layernorm'
writer = SummaryWriter(log_dir)   #data_number,tf,Batch_size,epochs,head
Batch_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:',device)
epochs = 100

def main(re_loadData,re_stft,Is_Train):
    fs = 360
    n = 100
    X_train, Y_train, X_test, Y_test, X_val, Y_val = data.loadData(re_loadData)
    if re_stft:
        X_train_stft = data.make_data_stft(X_train,fs,n)
        np.save("X_train_stft.npy", X_train_stft)
        X_test_stft = data.make_data_stft(X_test,fs,n)
        np.save("X_test_stft.npy", X_test_stft)
        X_val_stft = data.make_data_stft(X_val,fs,n)
        np.save("X_val_stft.npy", X_val_stft)
    else:
        X_train_stft = np.load("X_train_stft.npy")
        X_test_stft = np.load("X_test_stft.npy")
        X_val_stft = np.load("X_val_stft.npy")
    Tran_loader = Data.DataLoader(data.MyDataSet(X_train_stft, X_train,Y_train), Batch_size, True)
    Test_loader = Data.DataLoader(data.MyDataSet(X_test_stft,X_test,Y_test), Batch_size, True)
    Val_loader = Data.DataLoader(data.MyDataSet(X_val_stft, X_val, Y_val), Batch_size, True)

    model = Net(device).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, betas=(0.7, 0.999), eps=1e-07, weight_decay=0, amsgrad=False)
    best_acc = 0
    if Is_Train:
        for epoch in range(epochs):
            model.train()
            loss_sum = 0
            acc_sum = 0
            num = 0
            for signal_TF, signal_org,Label in Tran_loader:
                signal_TF, signal_org,Label = signal_TF.to(device), signal_org.transpose(1,2).to(device),Label.to(device)
                outputs = model(signal_TF,signal_org.float())
                loss = criterion(outputs, Label.to(torch.int64) .view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                acc = float((torch.argmax(outputs, dim=1) == Label).cpu().numpy().astype(int).sum()) / float(
                    Label.size(0))
                acc_sum += acc
                loss_sum += loss.item()
                num += 1
            acc_avg = acc_sum / num
            loss_avg = loss_sum / num
            print('Epoch:', '%06d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(loss_avg))
            print('Epoch:', '%06d' % (epoch + 1), 'train acc_avg =', '{:.6f}'.format(acc_avg))
            writer.add_scalar('train loss', loss_avg, epoch + 1)
            writer.add_scalar('train acc', acc_avg, epoch + 1)

            model.eval()
            loss_sum = 0
            acc_sum = 0
            num = 0
            with torch.no_grad():
                for signal_TF, signal_org, Label in Val_loader:
                    signal_TF, signal_org, Label = signal_TF.to(device), signal_org.transpose(1, 2).to(device), Label.to(device)
                    outputs = model(signal_TF, signal_org.float())
                    loss = criterion(outputs, Label.to(torch.int64).view(-1))
                    acc = float((torch.argmax(outputs, dim=1) == Label).cpu().numpy().astype(int).sum()) / float(
                        Label.size(0))
                    acc_sum += acc
                    loss_sum += loss.item()
                    num += 1
                acc_avg = acc_sum/num
                loss_avg = loss_sum / num
                print('Epoch:', '%06d' % (epoch + 1), 'Val_loss =', '{:.6f}'.format(loss_avg))
                print('Epoch:', '%06d' % (epoch + 1), 'val acc_avg =', '{:.6f}'.format(acc_avg))
                writer.add_scalar('Val loss', loss_avg, epoch + 1)
                writer.add_scalar('val acc', acc_avg, epoch + 1)
            if acc_avg >= best_acc:
                best_acc = acc_avg
                torch.save(model, save_dir + 'best.pkl')
            if (epoch + 1) % 10 == 0:
                torch.save(model, save_dir + str(epoch + 1) + '.pkl')
        torch.save(model, save_dir + 'last.pkl')

    else:
        #model = torch.load(save_dir + 'last.pkl')
        model = torch.load('runs/last.pkl')
        labels = [label for label in ['N', 'A', 'V', 'L', 'R']]
        confusion = ConfusionMatrix(num_classes=5, labels=labels)
        with torch.no_grad():
            for signal_TF, signal_org,Label in Test_loader:
                predict = model(signal_TF.to(device),signal_org.transpose(1,2).float().to(device).to(device))
                predict = torch.softmax(predict, dim=1)
                predict = torch.argmax(predict, dim=1)
                Label = Label.cpu().detach().numpy()
                predict = predict.cpu().detach().numpy()
                confusion.update(predict, Label)
        confusion.summary()
        confusion.plot()


if __name__ == '__main__':
    re_loadData = 0
    re_stft = 0
    Is_Train =0
    #设置随机数，使实验具有可重复性
    seed_n = np.random.randint(5)
    random.seed(seed_n)
    np.random.seed(seed_n)
    torch.manual_seed(seed_n)
    torch.cuda.manual_seed(seed_n)
    torch.cuda.manual_seed_all(seed_n)
    main(re_loadData,re_stft,Is_Train)
