import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backend.dataloader import load_data, load_labels

from backend.network import Network

LEARNING_RATE = 1e-3
EPOCHS = 100

USE_CUDA = False

def main():
    train_data, temp_mean, temp_var, time_mean, time_var = load_data('./data/train.csv')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

    print('Temperature mean: %f' % temp_mean)
    print('Tempature var: %f' % temp_var)
    print('Time mean: %f' % time_mean)
    print('Time var: %f' % time_var)

    with open('weights/normalize.csv', 'w') as f:
        f.write('%f\n%f\n%f\n%f' % (temp_mean, temp_var, time_mean, time_var))

    net = Network()

    opt = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    if USE_CUDA:
        net = net.cuda()

    for epoch in range(1, EPOCHS+1):
        running_loss = 0
        correct, total = 0, 0

        for X, Y in train_loader:
            if USE_CUDA:
                X, Y = X.cuda(), Y.cuda()

            opt.zero_grad()

            preds = net.forward(X)
            loss = F.nll_loss(preds, Y)
            loss.backward()

            correct += (preds.argmax(dim=1) == Y).sum().item()
            total += len(preds)
            running_loss += loss.item()

            opt.step()

        print('[Epoch %d] loss: %.4f acc: %.2f%%' % (epoch, running_loss / len(train_loader), 100 * correct / total))

        torch.save(net.state_dict(), 'weights/checkpoint-%04d.bin' % (epoch))

if __name__ == '__main__':
    main()
