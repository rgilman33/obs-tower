from obstacle_tower_env import ObstacleTowerEnv
import sys
import argparse
import time          

# Custom
def run_episode(env):
    with torch.no_grad():
        done = False
        episode_reward = 0.0
        action = [0,0,0,0]
        (frame,key,time), reward, done, info = env.step(action); 
        hidden = make_hidden(1)

        while not done:

            frame = cv2.resize(frame, (128, 128), cv2.INTER_AREA)
            frame = np.transpose(frame, (2,0,1))
            frame = torch.FloatTensor(frame).to(device).unsqueeze(0); 

            state, hidden = get_state(frame, hidden)

            probs = controller(state);
            action = action_probs_to_action(probs); 
            action_dummy = action_to_dummies(action);

            action[0] = 1 # Force forward
            action[3] = 0 # No side to side
            
            (frame,key,time), reward, done, info = env.step(action)

            episode_reward += reward
            
        return episode_reward

def get_state(images, hidden):
    "Takes in images and lstm hidden, runs through vae and lstm, cats z and lstm output together into state"
    z, _, _ = encoder(images)
    z, _, _ = normalize_sequential(z, calc=False, mean=z_mean, absmax=z_max);
    z = z.unsqueeze(1); #print("z shape", z.shape)

    _, hidden, raw_output = lstm(z, hidden); #print("raw output shape", raw_output.shape)

    state = torch.cat([raw_output,z], dim=2).detach().squeeze(1); #print("state shape", state.shape)
    return state, hidden
"""
def run_episode(env):
    done = False
    episode_reward = 0.0
    
    while not done:
        action = env.action_space.sample(); print('action', action)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        
    return episode_reward"""

def run_evaluation(env):
    while not env.done_grading():
        run_episode(env)
        env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('environment_filename', default='./ObstacleTower/obstacletower', nargs='?')
    parser.add_argument('--docker_training', action='store_true')
    parser.set_defaults(docker_training=False)
    args = parser.parse_args()

    # changed retro to false, just like in training
    env = ObstacleTowerEnv(args.environment_filename, docker_training=args.docker_training, retro=False, timeout_wait=600) 
    time.sleep(30)
    
    if env.is_grading():

        from models import *

        encoder = Encoder(SZ, 3, 64).to(device)
        lstm = AWD_LSTM(emb_sz=nz, n_hid=N_HIDDEN, n_layers=N_LAYERS, hidden_p=DROP, 
                        input_p=DROP, weight_p=DROP).to(device); 

        encoder.load_state_dict(torch.load(ENCODER_MODEL_PATH, map_location=device))
        lstm.load_state_dict(torch.load(LSTM_MODEL_PATH, map_location=device))
        encoder.eval(); lstm.eval()
        set_trainable(encoder, False); set_trainable(lstm, False)

        z_mean = torch.load(Z_MEAN_PATH).to(device); z_max = torch.load(Z_MAX_PATH).to(device); 

        controller = Controller().to(device)
        controller.load_state_dict(torch.load("bestC.torch", map_location='cpu'))

        episode_reward = run_evaluation(env)
    else:
        while True:
            episode_reward = run_episode(env)
            print("Episode reward: " + str(episode_reward))
            env.reset()

    env.close()

