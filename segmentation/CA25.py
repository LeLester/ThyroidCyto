from CA25net import *
import scipy.io
import numpy as np
import time


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

epoch, batch = 100, 1

training_data = XuDataset(os.path.join('dataset', 'train'))
train_loader = DataLoader(training_data, batch_size=1, shuffle=True)
test_data = XuDataset(os.path.join('dataset', 'test'))
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


model = Cia(in_channels=3).to(device)
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

print('Begin training.')
start = time.time()

loss_f, macc_f, bacc_f, cacc_f = [], [], [], []

for ep in range(epoch):
    w = 0.5
    for bt, data in enumerate(train_loader):
        model.train()
        img, label, bound, _ = data
        img = img.cuda()
        label = label.cuda()
        bound = bound.cuda()
        mout, bout = model(img)
        loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0.3)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if ep % 2 == 0:
        lr = lr * 0.95
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        acc_all, bacc_all, cacc_all, loss_all = [], [], [], []
        with torch.no_grad():
            for verify in test_loader:
                img, label, bound, _ = verify
                img = img.cuda()
                label = label.cuda()
                bound = bound.cuda()
                model.eval()
                mout, bout = model(img)
                loss = w*my_loss(mout, label) + (1-w)*cia_loss(bout, bound, 0.5)
                loss_all.append(loss.cpu().numpy())
                acc = dice_acc(mout[0][0], label)
                acc_all.append(acc)
                acc = dice_acc(bout[0][0], bound>0)
                bacc_all.append(acc)
                acc = dice_acc(bout[0][1], bound>1)
                cacc_all.append(acc)

            acc_all = np.array(acc_all)
            loss_all = np.array(loss_all)
            bacc_all = np.array(bacc_all)
            cacc_all = np.array(cacc_all)
            print('epoch num : {} -- Loss: {} -- Mask acc : {} --  Boundary acc : {} -- Clustered edge acc : {}'.format(ep + 1, loss_all.mean(), acc_all.mean(), bacc_all.mean(), cacc_all.mean()))
            
            loss_f.append(loss_all.mean())
            macc_f.append(acc_all.mean())
            bacc_f.append(bacc_all.mean())
            cacc_f.append(cacc_all.mean())
            
            if ep > 9 and ep % 10 == 0 :
                torch.save({'model_state_dict': model.state_dict()}, 'weights_3c_2/ep{}_loss{}.ptf'.format(ep+1,bacc_all.mean()))

macc_f = np.array(macc_f)
loss_f = np.array(loss_f)
bacc_f = np.array(bacc_f)
cacc_f = np.array(cacc_f)
mdic = {"macc":macc_f, "loss":loss_f,"bacc":bacc_f, "cacc":cacc_f}

torch.save({'model_state_dict': model.state_dict()}, 'weights_3c_2/CA25_n.ptf')
end = time.time()
print('Total training time is {}h'.format((end-start)/3600))
print('Finished Training')
