import pandas as pd
import numpy as np
import os
import pickle
import re
import torch

import sys
sys.path.append("../../")
from scripts.utils import root_dir


meta_data_dir = os.path.join(root_dir, "data/ml-32m")
movies_filepath = os.path.join(meta_data_dir, "movies.csv")
tags_filepath = os.path.join(meta_data_dir, "tags.csv")

nmf_model_files_dir = os.path.join(root_dir, "models/nmf_model_files")
nmf_train_items_path = os.path.join(nmf_model_files_dir, "train_items.pkl")
nmf_users_path = os.path.join(nmf_model_files_dir, "uid2idx.pkl")
nmf_items_path = os.path.join(nmf_model_files_dir, "iid2idx.pkl")

movies_df = pd.read_csv(movies_filepath)
with open(nmf_train_items_path, 'rb') as f:
    train_items = pickle.load(f)
movies_df = movies_df[movies_df['movieId'].isin(train_items)]

assert movies_df.isna().sum().sum() == 0

# genres
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|') if isinstance(x, str) else [])
all_genres = set()
for genres in movies_df['genres']:
    all_genres.update(genres)

genre_dict = dict() 
genre_dict['(no genres listed)'] = 0

count = 1
for genre in all_genres:
    if genre not in genre_dict:
        genre_dict[genre] = count
        count += 1


# year
def get_year_from_title(title):
    if isinstance(title, str):
        # strip leading/trailing whitespace 
        title = title.strip()

        # regex match for str end with "(XXXX)"
        match = re.search(r'\(\d{4}\)$', title)
        
        if match:
            year = match.group(0)[1:-1]
            return int(year)
    return -1

movies_df['year'] = movies_df['title'].apply(get_year_from_title)

year_dict = dict()
year_dict[-1] = 0  # for movies without a year

count = 1
for year in movies_df['year'].unique():
    if year not in year_dict:
        year_dict[year] = count
        count += 1
        
# tags
with open(nmf_users_path, 'rb') as f:
    uid2idx = pickle.load(f)

tags_df = pd.read_csv(tags_filepath)
tags_df = tags_df[tags_df['movieId'].isin(train_items)]
tags_df = tags_df[tags_df['userId'].isin(uid2idx.keys())]
tags_grouped = tags_df.groupby('movieId')['tag'].apply(list).reset_index()
# get the top 3 frequent tags for each movie
def get_top_3_tags(tags_list):
    if not tags_list:
        return ['(no tags)', '(no tags)', '(no tags)']
    
    tag_counts = pd.Series(tags_list).value_counts()
    top_tags = tag_counts.nlargest(3).index.tolist()
    
    # If there are less than 3 tags, fill the rest with '(no tags)'
    while len(top_tags) < 3:
        top_tags.append('(no tags)')
    return top_tags
tags_grouped['tags'] = tags_grouped['tag'].apply(get_top_3_tags)
tags_grouped.drop(columns=['tag'], inplace=True)

all_tags = set()
for tags in tags_grouped['tags']:
    all_tags.update(set(tags))
    
tag_dict = dict()
tag_dict['(no tags)'] = 0
count = 1
for tag in all_tags:
    if tag not in tag_dict:
        tag_dict[tag] = count
        count += 1

movies_df = movies_df.merge(tags_grouped, on='movieId', how='left')
movies_df['tags'] = movies_df['tags'].apply(lambda x: x if isinstance(x, list) else ['(no tags)']*3)

movies_df['genres'] = movies_df['genres'].apply(lambda x: [genre_dict[genre] for genre in x])
movies_df['year'] = movies_df['year'].apply(lambda x: year_dict[x])
movies_df['tags'] = movies_df['tags'].apply(lambda x: [tag_dict[tag] for tag in x])

# pad genre based on the max length of genres in movies_df
max_genre_length = max(movies_df['genres'].apply(len))

def pad_genres(genres, max_length):
    if len(genres) < max_length:
        return genres + [0] * (max_length - len(genres))  # pad with 0s
    return genres[:max_length]  # truncate if longer than max_length
movies_df['genres'] = movies_df['genres'].apply(lambda x: pad_genres(x, max_genre_length))


# load items
with open(nmf_items_path, 'rb') as f:
    iid2idx = pickle.load(f)

movies_df['item_idx'] = movies_df['movieId'].map(iid2idx)
movies_df.drop(columns=['movieId'], inplace=True)
movies_df.drop(columns=['title'], inplace=True)
movies_df.rename(columns={'genres': 'genres_idx',
                           'year': 'year_idx',
                           'tags': 'tags_idx'}, inplace=True)


# save processed data
movies_df_path = os.path.join(nmf_model_files_dir, 'movies_df.pkl')
tags_dict_path = os.path.join(nmf_model_files_dir, 'tag_dict.pkl')
genre_dict_path = os.path.join(nmf_model_files_dir, 'genre_dict.pkl')
year_dict_path = os.path.join(nmf_model_files_dir, 'year_dict.pkl')

with open(movies_df_path, 'wb') as f:
    pickle.dump(movies_df, f)
with open(tags_dict_path, 'wb') as f:
    pickle.dump(tag_dict, f)
with open(genre_dict_path, 'wb') as f:
    pickle.dump(genre_dict, f)
with open(year_dict_path, 'wb') as f:
    pickle.dump(year_dict, f)