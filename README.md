# Two-Stage Deep Learning Recommender System for MovieLens

This repository contains the complete pipeline for building, training, and evaluating a modern two-stage deep learning recommender system using PyTorch. The models are trained on a dense subset of the MovieLens-32M dataset to predict user movie preferences.

The project follows a rigorous, bottom-up approach, starting with simple baselines to validate the data and evaluation framework before implementing a sophisticated two-tower retrieval and re-ranking architecture.

## Final Model Performance

| Model | nDCG@10 |
| :--- | :--- |
| **Two-Stage Retriever + Re-ranker** | **0.5466** |

---


## Project Structure

The repository is organized to separate data, models, and scripts, ensuring a clean and maintainable workflow.

```plaintext
movielens32m/
├── data/
│   └── ml-32m/
│       ├── raw/
│       └── processed/
├── models/
│   ├── mf_model_files/
│   ├── nmf_model_files/
│   ├── two_tower_retriever_files/
│   └── reranker_files/
├── scripts/
│   ├── pre_modeling/
│   │   ├── i_download_raw_data.py
│   │   └── ii_preprocess_ratings_data.py
│   ├── modeling/
│   │   ├── mf_model.py
│   │   ├── mf_training.py
│   │   ├── nmf_model.py
│   │   ├── nmf_training.py
│   │   ├── popularity_recommender.py
│   │   ├── two_tower_retriever_model.py
│   │   ├── two_tower_retriever_training.py
│   │   ├── reranker_model.py
│   │   ├── generate_reranker_training_data.py
│   │   └── reranker_training_and_evaluation.py
│   └── utils.py
└── README.md
```

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd movielens32m
    ```

2.  **Create `utils.py`:**
    Inside the `scripts/` directory, create a file named `utils.py` and add the following line to define the absolute path to your project root. This is crucial for all scripts to locate files correctly.
    ```python
    # scripts/utils.py
    import os
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ```

3.  **Install Dependencies:**
    It is recommended to use a virtual environment. Install the required packages from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run: A Step-by-Step Walkthrough

The project is designed to be run sequentially, building from data preprocessing to the final evaluation.

### Step 1: Data Downloading & Preprocessing

First, we download the raw data and then apply a rigorous preprocessing and filtering pipeline to create a high-quality, dense dataset suitable for training deep learning models.

1.  **Download Raw Data (`i_download_raw_data.py`):**
    This script downloads the MovieLens-32M zip file and extracts it into the `data/ml-32m/` directory.

2.  **Preprocess Ratings (`ii_preprocess_ratings_data.py`):**
    This is a critical step that performs two main functions:
    * **Temporal Split**: The data is split into training and validation sets based on timestamps to simulate a real-world prediction task.
    * **N-Core Filtering**: An iterative process is applied to ensure the data is dense.
        * **Training Set**: Every user and movie must have at least **20** interactions.
        * **Validation Set**: Every user must have at least **10** interactions.
    This creates the `train_20_core.csv` and `val_10_core.csv` files in the `data/ml-32m/processed/` directory.

### Step 2: Baseline Models (Validation & Benchmarking)

To ensure the data and evaluation logic were sound, several baseline models were tested. This is a crucial debugging step before moving to more complex architectures.

#### a. Popularity Recommender
A simple, non-personalized model that recommends the same list of globally popular movies to every user. This provides the absolute minimum performance to beat.
* **Script**: `popularity_recommender.py`
* **Result**: `nDCG@10: 0.0512`

#### b. Matrix Factorization (MF)
A classic collaborative filtering model trained using an implicit feedback approach (BCE Loss with negative sampling).
* **Scripts**: `mf_model.py`, `mf_training.py`
* **Result**: `nDCG@10: 0.0722`

#### c. Neural Matrix Factorization (NMF)
A more powerful model combining Matrix Factorization with an MLP to capture more complex user-item interactions. It was trained with a pairwise loss (BPR Loss).
* **Scripts**: `nmf_model.py`, `nmf_training.py`
* **Result**: `nDCG@10: 0.0923`

### Step 3: Two-Stage Architecture

With the baselines validating the pipeline, the final, more sophisticated architecture was implemented.

#### a. Stage 1: Two-Tower Retriever
This model learns separate embeddings for users and movies, allowing for efficient retrieval of a candidate set from the entire movie corpus.
* **Scripts**: `two_tower_retriever_model.py`, `two_tower_retriever_training.py`
* **Function**: This script preprocesses all metadata (genres, years, tags), engineers user history features, trains the two-tower model, and evaluates its recall.
* **Retriever Performance**:
    | Metric | Value |
    | :--- | :--- |
    | Recall@50 | 0.0479 |
    | Recall@100 | 0.0873 |
    | Recall@200 | 0.1534 |
    | **Recall@500** | **0.2970** |
    | Recall@1000 | 0.4487 |

A candidate pool size of **K=500** was chosen for the re-ranking stage, as it provided a good balance of recall and computational feasibility.

#### b. Stage 2: Re-ranker
This model takes the 500 candidates from the retriever and scores them for a final, precise ranking.
1.  **Generate Re-ranker Training Data**:
    * **Script**: `generate_reranker_training_data.py`
    * **Function**: Uses the trained retriever to generate a candidate set for each user and creates training triplets (user, positive_item, negative_item) *exclusively from within that set*. This is crucial to avoid training/serving skew.
2.  **Train Re-ranker and Final Evaluation**:
    * **Scripts**: `reranker_model.py`, `reranker_training_and_evaluation.py`
    * **Function**: Trains the re-ranker model, which uses frozen embeddings from the retriever as a key feature. After training, it performs a full two-stage evaluation to get the final system performance.

## Final Result & Conclusion

The end-to-end system was evaluated by first retrieving 500 candidates for each validation user and then using the trained re-ranker to produce the final top-10 list. The final score demonstrates the effectiveness of the architecture on the high-quality, dense dataset.

-   **Final nDCG@10**: **0.5466**

This project successfully demonstrates the power of a modern two-stage recommender system. By decoupling the recommendation process into a high-recall retriever and a high-precision re-ranker, the system achieves a state-of-the-art result. The iterative debugging process, starting with simple, validated baselines, was critical to the final model's success.
