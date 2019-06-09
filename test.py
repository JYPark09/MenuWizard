import torch

from backend.network import Network
from backend.dataloader import load_labels, normalize

def main():
    net = Network()
    net.load_state_dict(torch.load('weights/checkpoint-0500.bin'))

    with open('weights/normalize.csv', 'r') as f:
        temp_mean = float(f.readline())
        temp_var = float(f.readline())
        time_mean = float(f.readline())
        time_var = float(f.readline())

    inputTensor = torch.tensor([0,1,29.0,1,4,0.6,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]).view(1, -1)
    normalize(inputTensor[0], temp_mean,temp_var, time_mean, time_var)

    pred = torch.exp(net(inputTensor))
    labels = load_labels()

    for i in range(len(labels)):
        print('%s : %.2f%%' % (labels[i], pred[0][i] * 100))

    max_idx = pred[0].argmax()
    print('\nAI의 추천 메뉴: %s (%.2f%%)' % (labels[max_idx], 100 * pred[0][max_idx]))

if __name__ == '__main__':
    main()
