import os
import sys
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from pathlib import Path
import torch
import pickle
import pandas as pd
import numpy as np
from src.models import config
from src.models.genetic_algorithm import *
from src.features.get_svg_size_pos import get_relative_pos_to_bounding_box_of_animated_paths
from src.models.logger import info


def retrieve_m1_predictions(input_data):
    m1_path_vectors = torch.tensor(input_data[config.m1_features].to_numpy(), dtype=torch.float)
    m1 = pickle.load(open(config.m1_path, 'rb'))
    path_predictions = m1.predict(m1_path_vectors)
    input_data['animated'] = path_predictions

    info(f'Number of animated paths: {sum(path_predictions)}/{len(path_predictions)}')
    return input_data


def retrieve_animation_midpoints(input_data, data_dir='data/raw/logos_dataset', drop=True):
    # Integrate midpoint of animation as feature
    animated_input_data = input_data[input_data['animated'] == 1]
    gb = animated_input_data.groupby('filename')['animation_id'].apply(list)

    input_data = pd.merge(left=input_data, right=gb, on='filename')
    input_data.rename(columns={'animation_id_x': 'animation_id', 'animation_id_y': 'animated_animation_ids'},
                      inplace=True)

    if drop:
        info("Non-animated paths won't be used for training")
        input_data = input_data[input_data['animated'] == 1]
    else:
        info("Non-animated paths will be used for training")
    input_data.reset_index(drop=True, inplace=True)
    info('Start extraction midpoint of animated paths as feature')
    input_data["rel_position_to_animations"] = input_data.apply(
        lambda row: get_relative_pos_to_bounding_box_of_animated_paths(f"{data_dir}/{row['filename']}.svg",
                                                                       int(row["animation_id"]),
                                                                       row["animated_animation_ids"]), axis=1)
    input_data["rel_x_position_to_animations"] = input_data["rel_position_to_animations"].apply(lambda row: row[0])
    input_data["rel_y_position_to_animations"] = input_data["rel_position_to_animations"].apply(lambda row: row[1])
    return input_data


def save_predictions(df, agents, test_paths, rewards, predictions, sorted_indices, generation):
    for i, agent in enumerate(sorted_indices):
        test_predictions = agents[agent](test_paths)
        test_reward = return_average_reward(test_paths, test_predictions)

        train_types = list()
        for prediction in predictions[agent]:
            train_types.append(np.argmax(prediction.detach().numpy()))
        train_type_counts = Counter(train_types)

        test_types = list()
        for prediction in test_predictions:
            test_types.append(np.argmax(prediction.detach().numpy()))
        test_type_counts = Counter(test_types)

        df = df.append(
            {'generation': generation, 'agent': agent, 'agent_rank': i,
             'train_mean_reward': rewards[agent], 'test_mean_reward': test_reward,
             'train_translate': train_type_counts[0], 'train_scale': train_type_counts[1],
             'train_rotate': train_type_counts[2], 'train_skew': train_type_counts[3],
             'train_fill': train_type_counts[4], 'train_opacity': train_type_counts[5],
             'test_translate': test_type_counts[0], 'test_scale': test_type_counts[1],
             'test_rotate': test_type_counts[2], 'test_skew': test_type_counts[3],
             'test_fill': test_type_counts[4], 'test_opacity': test_type_counts[5]},
            ignore_index=True)
    return df


def get_n_types(predictions):
    n_types = list()
    for agent_prediction in predictions:
        types = list()
        for prediction in agent_prediction:
            types.append(np.argmax(prediction.detach().numpy()))
        counts = Counter(types)
        n_types.append(len(counts))
    return n_types


def train_animation_predictor(train_paths, test_paths, hidden_sizes=config.a_hidden_sizes, out_size=config.a_out_sizes,
                              num_agents=100, top_parent_limit=10, generations=10, timestamp='', min_n_types=0):
    # disable gradients as we will not use them
    torch.set_grad_enabled(False)

    # initialize number of agents
    overall_start = datetime.now()
    agents = create_random_agents(num_agents=num_agents)

    info('AP model summary')
    print('=' * 100)
    print(f'Number of training instances: {len(train_paths)}')
    print(f'Number of agents: {num_agents}')
    print(f'Top parent limit: {top_parent_limit}')
    print(f'Number of generations: {generations}')
    print(f'Hidden sizes: {hidden_sizes}')
    print(f'Output size: {out_size}')
    print('=' * 100)

    training_process = pd.DataFrame(
        {'generation': [], 'agent': [], 'agent_rank': [],
         'train_mean_reward': [], 'test_mean_reward': [],
         'train_translate': [], 'train_scale': [],
         'train_rotate': [], 'train_skew': [],
         'train_fill': [], 'train_opacity': [],
         'test_translate': [], 'test_scale': [],
         'test_rotate': [], 'test_skew': [],
         'test_fill': [], 'test_opacity': []})

    for generation in range(generations):
        start = datetime.now()
        info(f'Generation {generation + 1}/{generations}')
        rewards, predictions = compute_agent_rewards(agents=agents, path_vectors=train_paths)
        n_types = get_n_types(predictions)
        reward_sorted_parent_indexes = np.argsort(rewards)[::-1].astype(int)
        sorted_parent_indexes = np.array(
            [index for index in reward_sorted_parent_indexes if n_types[index] >= min_n_types]
            + [index for index in
               reward_sorted_parent_indexes if
               n_types[index] < min_n_types])
        training_process = save_predictions(df=training_process, agents=agents, test_paths=test_paths,
                                            rewards=rewards, predictions=predictions,
                                            sorted_indices=sorted_parent_indexes, generation=generation)
        sorted_parent_indexes = sorted_parent_indexes[:top_parent_limit]
        top_agents = [agents[best_parent] for best_parent in sorted_parent_indexes]
        top_rewards = [rewards[best_parent] for best_parent in sorted_parent_indexes]

        children_agents = crossover(agents=top_agents, num_agents=num_agents)
        children_agents = [mutate(agent) for agent in children_agents]

        agents = children_agents

        stop = datetime.now()
        info(f'Operation time: {stop - start}')

        print('-' * 100)

        print(f'Mean rewards: {np.mean(rewards)} | Mean of top 10: {np.mean(top_rewards[:10])}')
        print(f'Top {top_parent_limit} agents: {sorted_parent_indexes}')
        print(f'Rewards for top {top_parent_limit} agents: {top_rewards[:top_parent_limit]}')

        print('=' * 100)

    overall_stop = datetime.now()
    info(f'Overall operation time: {overall_stop - overall_start}')

    training_process.to_csv(f'logs/{timestamp}_animation_predictions.csv', index=False)

    return top_agents[0]


def main(train_path='data/path_animation_decision/model_1_train.csv', test_path='data/path_animation_decision/model_1_test.csv', drop=True,
         num_agents=100, top_parent_limit=20, generations=50, timestamp='', model1=True):
    info(f'Train data source: {train_path}')
    info(f'Test data source: {test_path}')
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    if model1:
        # Apply model one to get predictions whether to animate paths or not
        train_data = retrieve_m1_predictions(train_data)
        test_data = retrieve_m1_predictions(test_data)

        # Retrieve features describing the midpoint of animated paths
        train_data = retrieve_animation_midpoints(train_data, drop=drop)
        test_data = retrieve_animation_midpoints(test_data, drop=drop)

        # Scale input data for surrogate model
        scaler = pickle.load(open(config.scaler_path, 'rb'))
        train_data[config.sm_features] = scaler.transform(train_data[config.sm_features])
        test_data[config.sm_features] = scaler.transform(test_data[config.sm_features])
        info('Scaled input data for surrogate model')
    else:
        info("Model 1 won't be applied to input data")

    # Prepare path vectors for animation prediction
    train_paths = torch.tensor(train_data[config.sm_features].to_numpy(), dtype=torch.float)
    test_paths = torch.tensor(test_data[config.sm_features].to_numpy(), dtype=torch.float)

    # Perform genetic algorithm for animation prediction
    top_model = train_animation_predictor(train_paths=train_paths, test_paths=test_paths, num_agents=num_agents,
                                          top_parent_limit=top_parent_limit, generations=generations,
                                          timestamp=timestamp)

    # Save best model for animation prediction
    torch.save(top_model, f'models/{timestamp}ap_best_model.pkl')
    torch.save(top_model.state_dict(), f'models/{timestamp}ap_best_model_state_dict.pth')

    return top_model


if __name__ == '__main__':
    os.chdir('../..')

    ap = ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-s", "--save", required=False, action='store_true', help="if set, output will be saved to file")
    ap.add_argument("-d", "--drop", required=False, action='store_false', help="if set, non-animated paths "
                                                                               "will be kept for training")
    ap.add_argument("-m1", "--model1", required=False, action='store_false', help="if set, model one won't be applied "
                                                                                  "to input data. Note: Also expects "
                                                                                  "input data to be scaled already")
    ap.add_argument("-train", "--train", required=False, default='data/animation_predictor'
                                                                 '/ap_train_data_scaled.csv',
                    help="path to training data")
    ap.add_argument("-test", "--test", required=False, default='data/animation_predictor/ap_test_data_scaled.csv',
                    help="path to test data")

    ap.add_argument("-a", "--n_agents", required=False, default=100, help="number of agents")
    ap.add_argument("-t", "--top_parents", required=False, default=20, help="number of top agents to be considered, "
                                                                            "should be even number")
    ap.add_argument("-g", "--generations", required=False, default=50, help="number of generations")
    args = vars(ap.parse_args())

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if args['save']:
        Path('logs/').mkdir(parents=True, exist_ok=True)
        log_file = f"logs/{timestamp}_ap_training.txt"
        info(f'Saving enabled, log file: {log_file}')
        sys.stdout = open(log_file, 'w')
        main(train_path=args['train'], test_path=args['test'],
             num_agents=int(args['n_agents']), top_parent_limit=int(args['top_parents']),
             generations=int(args['generations']), drop=args['drop'], timestamp=timestamp, model1=args['model1'])
        sys.stdout.close()
    else:
        main(train_path=args['train'], test_path=args['test'],
             num_agents=int(args['n_agents']), top_parent_limit=int(args['top_parents']),
             generations=int(args['generations']), drop=args['drop'], timestamp=timestamp, model1=args['model1'])
