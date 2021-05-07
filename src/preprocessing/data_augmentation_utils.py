import random
import pandas as pd


def balance_data_by_label_animation_type(df_original, df_fake, datapoints_per_group=100):
    df = pd.concat([df_original, df_fake], ignore_index=True)
    df['temp'] = df.apply(lambda row: row['rating_0']+row['rating_1']+row['rating_2']+row['rating_3'], axis=1)

    for animation_type in range(6):
        for label in range(5):
            df_temp = df[(df['temp'] == label) & (df[f'an_vec_{animation_type}'] == 1)].iloc[:datapoints_per_group]
            if label == 0 and animation_type == 0:
                df_new = df_temp
            else:
                df_new = pd.concat([df_new, df_temp], ignore_index=True)

    df_new.drop(columns=['temp'], axis=1, inplace=True)

    return df_new


def balance_data_by_animation_type(df_original, df_fake, datapoints_per_animation_type=1000):
    df = pd.concat([df_original, df_fake], ignore_index=True)
    df['temp'] = df.apply(lambda row: row['rating_0']+row['rating_1']+row['rating_2']+row['rating_3'], axis=1)

    for i, animation_type in enumerate(['translate', 'scale', 'rotate', 'skew', 'fill', 'opacity']):
        label_distribution = get_label_distribution_of_animation_type(df, animation_type)
        for label in range(5):
            nr_datapoints = int(round(label_distribution[label] * datapoints_per_animation_type))
            df_temp = df[(df['temp'] == label) & (df[f'an_vec_{i}'] == 1)].iloc[:nr_datapoints]
            if i == 0 and label == 0:
                df_new = df_temp
            else:
                df_new = pd.concat([df_new, df_temp], ignore_index=True)

    df_new.drop(columns=['temp'], axis=1, inplace=True)

    return df_new


def get_label_distribution_of_animation_type(df, animation_type):
    if animation_type == 'translate':
        i = 0
    elif animation_type == 'scale':
        i = 1
    elif animation_type == 'rotate':
        i = 2
    elif animation_type == 'skew':
        i = 3
    elif animation_type == 'fill':
        i = 4
    elif animation_type == 'opacity':
        i = 5
    else:
        raise NotImplementedError(f'Animation type {animation_type} is not implemented')
    df = df[df[f'an_vec_{i}'] == 1]
    animation_type_count = df.shape[0]
    return [label_count / animation_type_count for label_count in get_label_count(df)]


def get_animation_type_count(df):
    translate = df.groupby('an_vec_0').size().iloc[1]
    scale = df.groupby('an_vec_1').size().iloc[1]
    rotate = df.groupby('an_vec_2').size().iloc[1]
    skew = df.groupby('an_vec_3').size().iloc[1]
    fill = df.groupby('an_vec_4').size().iloc[1]
    opacity = df.groupby('an_vec_5').size().iloc[1]
    return [translate, scale, rotate, skew, fill, opacity]


def get_label_count(df):
    df['temp'] = df.apply(lambda row: row['rating_0'] + row['rating_1'] + row['rating_2'] + row['rating_3'], axis=1)
    vb = df.groupby('temp').size().iloc[0]
    b = df.groupby('temp').size().iloc[1]
    o = df.groupby('temp').size().iloc[2]
    g = df.groupby('temp').size().iloc[3]
    vg = df.groupby('temp').size().iloc[4]
    return [vb, b, o, g, vg]


def print_animation_type_count(df):
    print(f"translate: {df.groupby('an_vec_0').size().iloc[1]}")
    print(f"scale: {df.groupby('an_vec_1').size().iloc[1]}")
    print(f"rotate: {df.groupby('an_vec_2').size().iloc[1]}")
    print(f"skew: {df.groupby('an_vec_3').size().iloc[1]}")
    print(f"fill: {df.groupby('an_vec_4').size().iloc[1]}")
    print(f"opacity: {df.groupby('an_vec_5').size().iloc[1]}")


def print_label_count(df):
    df['temp'] = df.apply(lambda row: row['rating_0'] + row['rating_1'] + row['rating_2'] + row['rating_3'], axis=1)
    print(f"Label 0 (very bad): {df.groupby('temp').size().iloc[0]}")
    print(f"label 1 (bad): {df.groupby('temp').size().iloc[1]}")
    print(f"label 2 (okay): {df.groupby('temp').size().iloc[2]}")
    print(f"label 3 (good): {df.groupby('temp').size().iloc[3]}")
    print(f"label 4 (very good): {df.groupby('temp').size().iloc[4]}\n")


def print_label_count_of_all_animation_types(df):
    for animation_type in ['translate', 'scale', 'rotate', 'skew', 'fill', 'opacity']:
        print_label_count_by_animation_type(df, animation_type)


def print_label_count_by_animation_type(df, animation_type):
    if animation_type == 'translate':
        i = 0
    elif animation_type == 'scale':
        i = 1
    elif animation_type == 'rotate':
        i = 2
    elif animation_type == 'skew':
        i = 3
    elif animation_type == 'fill':
        i = 4
    elif animation_type == 'opacity':
        i = 5
    else:
        raise NotImplementedError(f'Animation type {animation_type} is not implemented')
    df = df[df[f'an_vec_{i}'] == 1]
    print(f"Animation type: {animation_type}")
    print_label_count(df)


def add_variance_in_animation_vectors(df):
    for i in range(6, 12):
        df[f'an_vec_{i}'] = df[f'an_vec_{i}'].apply(
            lambda row: row + (0.2 * random.random() - 0.1) if 0.1 < row < 0.9 else row)
        df[f'an_vec_{i}'] = df[f'an_vec_{i}'].apply(
            lambda row: row + (0.1 * random.random() - 0.05) if 0.9 < row < 0.95 else row)
        df[f'an_vec_{i}'] = df[f'an_vec_{i}'].apply(
            lambda row: row + (0.1 * random.random() - 0.05) if 0.05 < row < 0.1 else row)
        df[f'an_vec_{i}'] = df[f'an_vec_{i}'].apply(
            lambda row: row + (0.02 * random.random() - 0.01) if 0.95 < row < 0.99 else row)
        df[f'an_vec_{i}'] = df[f'an_vec_{i}'].apply(
            lambda row: row + (0.02 * random.random() - 0.01) if 0.01 < row < 0.05 else row)
        df[f'an_vec_{i}'] = df[f'an_vec_{i}'].apply(
            lambda row: row + (0.0002 * random.random() - 0.0001) if 0.95 < row < 0.9999 else row)
        df[f'an_vec_{i}'] = df[f'an_vec_{i}'].apply(
            lambda row: row + (0.0002 * random.random() - 0.0001) if 0.0001 < row < 0.05 else row)

    return df
