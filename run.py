from Env_modified import Env
from RL_brain_modified import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(300):
        print('episode = .................................', episode)
        with open('C:/E/dataSet/2018-06-20/result.txt', 'a') as res:
            res.write('episode = .................................' + str(episode) + '\n')
        total_reward = 0
        # initial observation
        init = True
        env = Env()
        episode_step = 0
        while True:

            if init is True:
                observation, reward, done, phase = env.reset(RL)
                total_reward += reward
                init = False

            # RL choose action based on observation
            action = RL.choose_action(observation, env.get_evn_time())
            # RL take action and get next observation and reward
            observation_, reward, done, phase = env.step(action)
            with open('C:/E/dataSet/2018-06-20/result.txt', 'a') as res:
                res.write(str(episode_step) + '    ' + '选择的hotspot    ' + str(action[0]) + '    等待时间    '
                          + str(action[1]) + '    获得的奖励    ' + str(reward) + '\n')
            print(str(episode_step) + '    ' + '选择的hotspot    ' + str(action[0]) + '    等待时间    '
                  + str(action[1]) + '    获得的奖励    ' + str(reward))
            total_reward += reward
            RL.store_transition(observation, action, reward, done, phase, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_
            episode_step += 1
            # break while loop when end of this episode
            if done:
                print('total reward         ', total_reward)
                with open('C:/E/dataSet/2018-06-20/result.txt', 'a') as res:
                    res.write('总奖励:    ' + str(total_reward) + '\n')
                break
            step += 1

    # end of game
    print('game over')


if __name__ == "__main__":
    # game
    RL = DeepQNetwork(learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_maze()
    RL.plot_cost()