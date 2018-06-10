if __name__ == '__main__':
    reward = 0
    for i in range(3):
        if i == 1:
            reward += 1.2
        else:
            reward += 0
    print(reward)