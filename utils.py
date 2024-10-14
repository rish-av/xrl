def get_representation(model, seq):
    rep, _ = model.encoder(seq)
    return rep


def sample_trajectories(states, actions, rewards, next_states, dones):
    trajectories = []
    current_trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}

    for i in range(len(states)):
        current_trajectory['states'].append(states[i])
        current_trajectory['actions'].append(actions[i])
        current_trajectory['rewards'].append(rewards[i])
        current_trajectory['next_states'].append(next_states[i])
        if dones[i]:
            trajectories.append(current_trajectory)
            if(len(trajectories) == 20):
                break
            current_trajectory = {'states': [], 'actions': [], 'rewards': [], 'next_states': []}
        
    if len(current_trajectory['states']) > 0:
        trajectories.append(current_trajectory)

    return trajectories