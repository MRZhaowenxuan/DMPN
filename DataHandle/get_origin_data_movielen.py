from DataHandle.get_originx_data_base import Get_origin_data_base
import pandas as pd
import numpy as np
import os
from config.model_parameter import model_parameter

np.random.seed(1234)

class Get_movie_data(Get_origin_data_base):

    def __init__(self, FLAGS):
        super(Get_movie_data, self).__init__(FLAGS=FLAGS)
        self.data_path = "data/orgin_data/movie.csv"

        if FLAGS.init_origin_data == True:
            if os.path.exists(self.data_path):
                self.origin_data = pd.read_csv(self.data_path)
            else:
                self.movie_data = pd.read_csv("data/raw_data/movies.csv")
                self.ratings_data = pd.read_csv("data/raw_data/ratings.csv")
                self.get_movies_data()
        else:
            self.origin_data = pd.read_csv(self.data_path)

    def get_movies_data(self):
        self.origin_data = pd.merge(self.ratings_data, self.movie_data, on="movieId")
        self.origin_data = self.origin_data[["userId", "movieId", "timestamp", "genres"]]
        self.origin_data = self.origin_data.rename(columns={"userId": "user_id",
                                                            "movieId": "item_id",
                                                            "timestamp": "time_stamp",
                                                            "genres": "cat_id",
                                                            })

        self.filtered_data = self.filter(self.origin_data)
        self.filtered_data.to_csv(self.data_path, encoding="UTF8", index=False)
        self.origin_data = self.filtered_data


if __name__ == '__main__':
    model_parameter_ins = model_parameter()
    experiment = model_parameter_ins.flags.FLAGS.experiment_name
    FLAGS = model_parameter_ins.get_parameter(experiment).FLAGS
    ins = Get_movie_data(FLAGS=FLAGS)