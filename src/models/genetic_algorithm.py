from src.models.train_animation_predictor import *


def init_weights(agent):
    for layer in agent.children():
        torch.nn.init.xavier_uniform_(layer.weight)


def create_random_agents(num_agents):
    agents = []

    for _ in range(num_agents):

        agent = AnimationPredictor(config.a_input_size, config.a_hidden_sizes, config.a_out_sizes)

        for param in agent.parameters():
            param.requires_grad = False

        init_weights(agent)
        agents.append(agent)

    return agents


def create_animation_vector(animation_prediction, value=-1):
    # Todo: Maybe put this function to another directory?
    if animation_prediction[0] == 1:
        for i in [8, 9, 10, 11]:
            animation_prediction[i] = value

    if animation_prediction[1] == 1:
        for i in [6, 7, 9, 10, 11]:
            animation_prediction[i] = value

    if animation_prediction[2] == 1:
        for i in [6, 7, 8, 10, 11]:
            animation_prediction[i] = value

    if animation_prediction[3] == 1:
        for i in [6, 7, 8, 9]:
            animation_prediction[i] = value

    if animation_prediction[4] == 1:
        for i in [6, 7, 8, 9, 10, 11]:
            animation_prediction[i] = value

    if animation_prediction[5] == 1:
        for i in [6, 7, 8, 9, 10, 11]:
            animation_prediction[i] = value
    return animation_prediction


def prepare_sm_input(path_vectors, animation_predictions, convert=True):
    if convert:
        return torch.tensor([list(torch.cat((create_animation_vector(animation_predictions[i]),
                                             path_vectors[i]), 0).detach().numpy())
                             for i in range(path_vectors.shape[0])])
    return torch.tensor([list(torch.cat((animation_predictions[i],
                                         path_vectors[i]), 0).detach().numpy())
                         for i in range(path_vectors.shape[0])])


def return_average_reward(path_vectors, animation_predictions):
    rewards = predict(prepare_sm_input(path_vectors, animation_predictions))
    # info(f'Reward distribution: {sorted(Counter(rewards.flatten().tolist()).items())}')
    return np.mean(rewards)


def compute_agent_rewards(agents, path_vectors, verbose=5):
    steps = len(agents) // verbose
    agent_rewards, agent_predictions = list(), list()
    num_agents = len(agents)
    for i, agent in enumerate(agents):
        animation_predictions = agent(path_vectors)
        avg_rewards = return_average_reward(path_vectors, animation_predictions)
        if i % steps == 0:
            info(f'Computed avg rewards for agent {i + 1}/{num_agents}: {avg_rewards}')
        agent_rewards.append(avg_rewards)
        agent_predictions.append(animation_predictions)
    return agent_rewards, agent_predictions


def crossover(agents, num_agents):
    children = list()

    for _ in range((num_agents - len(agents)) // 2):
        parent1 = np.random.choice(agents)
        parent2 = np.random.choice(agents)
        child1 = AnimationPredictor(config.a_input_size, config.a_hidden_sizes, config.a_out_sizes)
        child2 = AnimationPredictor(config.a_input_size, config.a_hidden_sizes, config.a_out_sizes)

        shapes = [param.shape for param in parent1.parameters()]

        genes1_flat = np.concatenate([param.flatten() for param in parent1.parameters()])
        genes2_flat = np.concatenate([param.flatten() for param in parent2.parameters()])

        genes_child1_flat = np.asarray(
            [random.choice([genes1_flat[i], genes2_flat[i]]) for i in range(genes1_flat.shape[0])])
        genes_child2_flat = np.asarray(
            [random.choice([genes1_flat[i], genes2_flat[i]]) for i in range(genes1_flat.shape[0])])

        genes1_unflat = []
        index = 0
        for shape in shapes:
            shape_flat = np.product(shape)
            genes1_unflat.append(genes_child1_flat[index: (index + shape_flat)].reshape(shape))
            index += shape_flat

        genes2_unflat = []
        index = 0
        for shape in shapes:
            shape_flat = np.product(shape)
            genes2_unflat.append(genes_child2_flat[index: (index + shape_flat)].reshape(shape))
            index += shape_flat

        c1_state_dict = child1.state_dict()
        for i, (name, param) in enumerate(c1_state_dict.items()):
            transformed_param = torch.tensor(genes1_unflat[i])
            c1_state_dict[name].copy_(transformed_param)

        c2_state_dict = child2.state_dict()
        for i, (name, param) in enumerate(c2_state_dict.items()):
            transformed_param = torch.tensor(genes2_unflat[i])
            c2_state_dict[name].copy_(transformed_param)

        children.append(child1)
        children.append(child2)
    agents.extend(children)
    return agents


def mutate(agent, mutation_power=0.02):
    child_agent = copy.deepcopy(agent)

    state_dict = child_agent.state_dict()

    for i, (name, param) in enumerate(state_dict.items()):
        # Transform the parameter as required.
        if len(param.shape) == 2:
            transformed_param = param + torch.tensor(mutation_power * np.random.randn(param.shape[0], param.shape[1]))

        if len(param.shape) == 1:
            transformed_param = param + torch.tensor(mutation_power * np.random.randn(param.shape[0]))

        # Update the parameter.
        state_dict[name].copy_(transformed_param)

    return child_agent
