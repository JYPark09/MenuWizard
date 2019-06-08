f = open('data.csv', 'a')

try:
    while True:
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

        print('먹을 메뉴: ', end='')
        to_eat = int(input())

        vec = [to_eat, dinner, weekend, temperature, many, time, lazy]
        menu_vec = [0] * 32
        menu_vec[menu] = 1

        vec.extend(menu_vec)

        f.write(','.join(map(str, vec)) + '\n')
        print()

except KeyboardInterrupt:
    f.close()