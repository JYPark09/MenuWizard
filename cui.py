import torch
import numpy as np

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

    print('점심 or 저녁: ', end='')
    dinner = int(input())

    print('평일 or 주말: ', end='')
    weekend = int(input())

    print('기온: ', end='')
    temperature = float(input())

    print('소수 or 다수: ', end='')
    many = int(input())

    print('수업한 시간: ', end='')
    time = int(input())

    print('어제 먹은 메뉴: ', end='')
    menu = int(input())

    print('귀찮은 정도: ', end='')
    lazy = float(input())

    vec = [dinner, weekend, temperature, many, time, lazy]
    
    menu_vec = [0]*32
    menu_vec[menu] = 1
    vec.extend(menu_vec)

    normalize(vec, temp_mean, temp_var, time_mean, time_var)
    vec = torch.tensor(vec).view(1, -1)

    pred = torch.exp(net(vec)).view(-1).detach().numpy()
    labels = load_labels()

    s = np.random.dirichlet([0.03] * 32)

    for i in range(32):
        pred[i] = 0.25 * s[i] + 0.75 * pred[i]

    total = sum(pred)
    for i in range(32):
        pred[i] /= total

    print('\n------------------------------------------------')

    for i in range(len(labels)):
        print('%s : %.2f%%' % (labels[i], pred[i] * 100))

    max_idx = pred.argmax()
    print('\nAI의 추천 메뉴: %s (%.2f%%)' % (labels[max_idx], 100 * pred[max_idx]))

if __name__ == '__main__':
    main()
