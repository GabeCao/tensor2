from Evn_modified import Evn
from RL_brain_modified import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(3000):
        print('episode = ', episode)
        # initial observation
        init = True

        while True:

            if init is True:
                observation, _, _ = env.reset(RL)
                init = False

            # RL choose action based on observation
            action = RL.choose_action(observation, env.get_evn_time())

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            print(done)
            print(reward)
            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn(env.get_evn_time())

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')


if __name__ == "__main__":
    # game
    env = Evn()
    RL = DeepQNetwork(learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    run_maze()
    RL.plot_cost()