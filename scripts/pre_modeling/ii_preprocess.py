import pandas as pd
import numpy as np
from scripts.utils import root_dir
import os
from sklearn.model_selection import train_test_split

data_fetch_path = os.path.join(root_dir, 'data', 'ml-32m')
file_name = 'ratings.csv'
data_push_path = os.path.join(root_dir, 'data', 'ml-32m', 'processed')

def time_based_clip(df, min_year_month='2019-01'):
    """
    Clip the DataFrame to only include ratings from a specified minimum year and month.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['year_month'] = df['timestamp'].dt.to_period('M')
    df = df[df['year_month'] >= min_year_month].reset_index(drop=True)
    df.drop(columns=['year_month'], inplace=True)
    return df

def n_core_processing_both(df, max_iter=20, min_ratings=10):
    """
    Iteratively process the df s.t. every user ends up with at least 10 ratings and every item ends up with at least 10 ratings.
    """
    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration {iteration}")
        
        # Count ratings per user and item
        user_counts = df['userId'].value_counts()
        item_counts = df['movieId'].value_counts()
        
        # Filter users and items with at least 10 ratings
        users_to_keep = user_counts[user_counts >= min_ratings].index
        items_to_keep = item_counts[item_counts >= min_ratings].index
        
        # Filter the DataFrame
        df_filtered = df[(df['userId'].isin(users_to_keep)) & (df['movieId'].isin(items_to_keep))]
        
        # Check if any changes were made
        if len(df_filtered) == len(df):
            print("No more changes, stopping iteration.")
            break
        
        df = df_filtered
        
        if iteration >= max_iter:
            print("Reached maximum iterations, stopping.")
            break 

    print(f"Final number of ratings: {len(df)}")
    print(f"Number of users: {df['userId'].nunique()}")
    print(f"Number of items: {df['movieId'].nunique()}")
    print("min user ratings:", df['userId'].value_counts().min())
    print("min item ratings:", df['movieId'].value_counts().min())
    print("density:", len(df) / (df['userId'].nunique() * df['movieId'].nunique()))
    
    return df

def n_core_processing_users(df, min_ratings=10):
    """
    For validation and test sets, filter out users with less than min_ratings ratings.
    """
    user_counts = df['userId'].value_counts()
    users_to_keep = user_counts[user_counts >= min_ratings].index
    df_filtered = df[df['userId'].isin(users_to_keep)]
    
    print(f"Filtered DataFrame: {len(df_filtered)} ratings, {df_filtered['userId'].nunique()} users, {df_filtered['movieId'].nunique()} items")
    return df_filtered


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv(os.path.join(data_fetch_path, file_name))
    
    # Process the data
    df = time_based_clip(df, min_year_month='2018-01')

    # train-test-validation split based on timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
    test_df, val_df = train_test_split(test_df, test_size=0.5, shuffle=False)


    # n core processing for training set
    min_train_ratings = 20
    train_df = n_core_processing_both(train_df, min_ratings=min_train_ratings)

    train_users = set(list(train_df['userId'].unique()))
    train_items = set(list(train_df['movieId'].unique()))

    # keep only users and items in test and validation sets that are present in the training set
    test_df = test_df.loc[test_df['userId'].isin(train_users) & test_df['movieId'].isin(train_items)].reset_index(drop=True)
    val_df = val_df.loc[val_df['userId'].isin(train_users) & val_df['movieId'].isin(train_items)].reset_index(drop=True)

    # n core processing for test and validation sets
    min_val_ratings = 10
    test_df = n_core_processing_users(test_df, min_ratings=min_val_ratings)
    val_df = n_core_processing_users(val_df, min_ratings=min_val_ratings)

    # shuffle the dataframes
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the processed data
    if not os.path.exists(data_push_path):
        os.makedirs(data_push_path)
    
    train_df.to_csv(os.path.join(data_push_path, f'train_{min_train_ratings}_core.csv'), index=False)
    test_df.to_csv(os.path.join(data_push_path, f'test_{min_val_ratings}_core.csv'), index=False)
    val_df.to_csv(os.path.join(data_push_path, f'val_{min_val_ratings}_core.csv'), index=False)



