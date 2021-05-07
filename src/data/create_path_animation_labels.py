import pandas as pd
import pickle


def create_path_animation_labels(save_label_path, animation_paths_ratings, animated_logos_vectors):
    animation_paths_ratings_melt = pd.melt(animation_paths_ratings,
                                           id_vars=["alias", "animation_file", "animation_id", "logo_id", "time"],
                                           value_vars=["path_0_rating", "path_1_rating", "path_2_rating",
                                                       "path_3_rating",
                                                       "path_4_rating", "path_5_rating", "path_6_rating",
                                                       "path_7_rating"],
                                           var_name="path", value_name="label")

    animation_paths_ratings_id = animation_paths_ratings_melt.assign(
        order_id=animation_paths_ratings_melt.path.str.extract("(\d+)")).astype({"order_id": 'int'}).rename(
        columns={"animation_id": "animation_number"}).drop("path", 1)

    animated_logos_vectors_id = animated_logos_vectors.assign(
        logo_id=animated_logos_vectors.file.str.extract("logo_(\d+)"),
        animation_number=animated_logos_vectors.file.str.extract("animation_(\d+)")).astype({"logo_id": 'int',
                                                                                             "animation_number": 'int'})

    label_path_merged = pd.merge(animated_logos_vectors_id,
                                 animation_paths_ratings_id, how='left', on=["logo_id", "animation_number", "order_id"])

    output = open(save_label_path + "/path_animation_label.pkl", 'wb')
    pickle.dump(label_path_merged, output)
    output.close()
