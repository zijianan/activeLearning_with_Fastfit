import pandas as pd
import pigeonXT as pixt
from fastfit import FastFitTrainer, FastFit
from datasets import Dataset
import mysql.connector
from transformers import AutoTokenizer, pipeline, AutoModel
from scipy.spatial.distance import cosine
from tqdm import tqdm
import torch
from datasets import DatasetDict
import pigeonXT as pixt
from typing import List, Optional, Any, Tuple, Dict
from jupyter_ui_poll import ui_events
import time
from mysql.connector.connection import MySQLConnection
from mysql.connector.cursor import MySQLCursor
import warnings
import faiss
import numpy as np
# Database handler for MySQL operations
class MySQLDataHandler:
    def __init__(self, host: str, user: str, password: str, database: str):
        self.host = host
        self.user = user
        self.password = password
        self.database = database

    def connect(self) -> Tuple[MySQLCursor, MySQLConnection]:
        self.conn = mysql.connector.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            database=self.database
        )
        return self.conn.cursor(), self.conn

    def fetch_data(self, query: str) -> pd.DataFrame:
        cursor, conn = self.connect()
        cursor.execute(query)
        data = cursor.fetchall()
        df = pd.DataFrame(data, columns=[i[0] for i in cursor.description])
        cursor.close()
        conn.close()
        return df
# Annotation class for running in-notebook annotations using pigeonXT
class Annotation:
    @staticmethod
    def run_annotation(df: pd.DataFrame, labels: List[str], column_name: str)->pixt.annotate:
        # This will only setup the annotation, and needs to be confirmed via UI interaction
        return pixt.annotate(
            examples=df[[column_name]].rename(columns={column_name: 'example'}),
            options=labels,
            task_type='classification',
            buttons_in_a_row=3,
            reset_buttons_after_click=True,
            include_next=True
        )

class ModelTraining:
    def __init__(self, dataset: Dataset, model_name_or_path: str, label_column_name: str, text_column_name: str, save_path: str, num_train_epochs: int, load_from_FastFit: bool = False, tokenizer_name: Optional[str] = None):
        self.dataset = dataset
        self.model_name_or_path = model_name_or_path
        self.label_column_name = label_column_name
        self.text_column_name = text_column_name
        self.num_train_epochs = num_train_epochs
        self.save_path = save_path
        self.load_from_FastFit = load_from_FastFit
        self.tokenizer_name = tokenizer_name

    def train_model(self) -> FastFitTrainer:
        trainer = FastFitTrainer(
            model_name_or_path=self.model_name_or_path,
            label_column_name=self.label_column_name,
            text_column_name=self.text_column_name,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            max_text_length=128,
            dataloader_drop_last=False,
            num_repeats=2,
            optim="adafactor",
            clf_loss_factor=0.1,
            fp16=True,
            dataset=self.dataset,
            load_from_FastFit=self.load_from_FastFit,
            tokenizer_name=self.tokenizer_name
        )
        model = trainer.train()
        model.save_pretrained(self.save_path)
        results = trainer.evaluate()
        print(results)
        print("Accuracy: {:.1f}%".format(results["eval_accuracy"] * 100))
        return model
class ActiveLearning:
    """
    Initialize the Active Learning class with all required parameters for setting up and performing the learning process.

    Parameters
    ----------
    i : int
        Number of active learning epochs.
    theta : int
        Number of data points to annotate per epoch.
    model_name_or_path : str
        Base path or name of the model to use.
    label_column_name : str
        Column name in the dataset that contains labels.
    text_column_name : str
        Column name in the dataset that contains text entries.
    vali_data : Dataset
        Previously annotated data for model performance assessment.
    inference_datasets : Dataset
        Dataset for inference using a previously trained base model.
    uname : str
        User name for tracking annotations.
    tokenizer_name : str
        Name of the tokenizer to be used with the model.
    labels : List[str]
        List of possible labels for annotation.
    """
    def __init__(self, i: int, theta: int, model_name_or_path: str, label_column_name: str, text_column_name: str, vali_data: Dataset, inference_datasets: Dataset, uname: str, tokenizer_name: str, labels: List[str]):
        self.i = i
        self.theta = theta
        self.model_name_or_path = model_name_or_path
        self.label_column_name = label_column_name
        self.text_column_name = text_column_name
        self.vali_data = vali_data
        self.inference_data = inference_datasets
        self.uname = uname
        self.labels = labels
        self.tokenizer_name = tokenizer_name

        self.model = FastFit.from_pretrained(f"{self.model_name_or_path}0")
        self.model.to('cpu')
        torch.cuda.empty_cache()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device="cuda:0")

    def inference_model(self, current_model_path: str) -> None:
        """
        Load a model from the specified path and perform inference on the inference dataset.
        Parameters:
            current_model_path (str): Path to the current model for the active learning epoch.
        """
        self.model = FastFit.from_pretrained(current_model_path)
        self.classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device="cuda:0")

        def batch_predict(data: pd.DataFrame, batch_size: int = 32) -> Tuple[List[str], List[float]]:
            """
            Predict categories and scores in batches from the given data.
            Parameters:
                data (pd.DataFrame): The data to predict on.
                batch_size (int): Size of batches to use for predictions.
            Returns:
                Tuple[List[str], List[float]]: Predicted categories and their scores.
            """
            categories = []
            scores = []
            for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
                end = start + batch_size
                batch = data.iloc[start:end]
                predictions = self.classifier(batch[self.text_column_name].tolist())
                for prediction in predictions:
                    categories.append(prediction['label'])
                    scores.append(prediction['score'])
            return categories, scores

        categories, scores = batch_predict(self.inference_data)
        self.inference_data['tweet_category'] = categories
        self.inference_data['score'] = scores
        self.model.to('cpu')
        torch.cuda.empty_cache()

    def perform_epochs(self) -> None:
        """
        Perform multiple epochs of active learning, including inference, annotation, and training.
        """
        annotation_label = self.labels
        annotation_label.append('unknown')
        for epoch in range(self.i):
            current_model_path = f"{self.model_name_or_path}{epoch}"
            print(f"Epoch {epoch + 1}/{self.i}")
            self.inference_model(current_model_path)

            subset_for_annotation = self.inference_data.sort_values(by='score').head(self.theta)

            annotations = Annotation.run_annotation(subset_for_annotation, self.labels, self.text_column_name)
            
            with ui_events() as poll:
                while (annotations['label'] != '').sum() != len(subset_for_annotation):
                    poll(10)  # React to UI events (up to 10 at a time)
                    time.sleep(0.1)

            subset_for_annotation[self.label_column_name] = annotations['label']
            subset_for_annotation['uname'] = self.uname
            subset_for_annotation = subset_for_annotation[subset_for_annotation[self.label_column_name].isin(self.labels)]
            subset_for_annotation_dataset = DatasetDict({
                'train': Dataset.from_pandas(subset_for_annotation[[self.label_column_name, self.text_column_name]]),
                'test': self.vali_data,
                'validation': self.vali_data
            })

            training_instance = ModelTraining(
                dataset=subset_for_annotation_dataset,
                model_name_or_path=current_model_path,
                label_column_name=self.label_column_name,
                text_column_name=self.text_column_name,
                save_path=f"{self.model_name_or_path}{epoch+1}",
                load_from_FastFit=True,
                tokenizer_name=self.tokenizer_name,
                num_train_epochs=10
            )
            self.model = training_instance.train_model()
            self.model.to('cpu')
            torch.cuda.empty_cache()

class SimilaritySearchQuery:
    def __init__(self, fastfit: bool, model: str, simi_query: List[str], texts:List[str],model_path: str = None, tokenizer_name: str = None):
        self.fastfit = fastfit
        self.model = model
        self.simi_query = simi_query
        self.texts = texts
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
    def get_features(self,texts, model_name='princeton-nlp/sup-simcse-bert-base-uncased', batch_size=32):
        if self.fastfit:
            model = FastFit.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
        # Load pre-trained model tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Prepare to collect batches of embeddings
        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size)):
            # Process each batch
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        # Concatenate all batch embeddings
        all_embeddings = np.vstack(all_embeddings)
        return all_embeddings
    
    def calculate_dist(self,k=100):
        embeddings = self.get_features(self.texts, batch_size=32)
        embeddings = np.array(embeddings).squeeze()
        dimension = embeddings.shape[1]  # Dimension of the vectors
        index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        index.add(np.array(embeddings).astype('float32')) 
        # query_vec = self.get_features(self.simi_query)  
        query_vecs = [self.get_features(self.simi_query).squeeze().astype('float32') for query in self.simi_query]# Convert query to vector

        query_vecs = np.array(query_vecs).reshape(len(queries), -1)
        # Reshape query_vec to be two-dimensional
        all_results = []
        for query, query_vec in zip(self.simi_query, query_vecs):
            D, I = index.search(query_vec.reshape(1, -1), k)
            results = [(query, self.texts[idx], distance) for distance, idx in zip(D[0], I[0])]
            all_results.extend(results)

        # Create DataFrame
        df = pd.DataFrame(all_results, columns=["Query", "Text", "Distance"]).sort_values(by="Distance")
        return df


# Main execution block
if __name__ == "__main__":
    db_handler = MySQLDataHandler('localhost', 'user', 'password', 'database_name')
    df_trump = db_handler.fetch_data("SELECT * FROM tweets WHERE tweet_text LIKE '%trump%'")
    annotations = Annotation.run_annotation(df_trump, ['positive', 'negative', 'not relevant'])

    # Assuming `dataset` is already prepared and split
    training = ModelTraining(dataset, "bert-base-uncased", "tweet_category", "tweet_text", 10)
    results = training.train_model()
    print("Accuracy: {:.1f}%".format(results["eval_accuracy"] * 100))
    active_learning_instance = ActiveLearning(model_name_or_path=save_path, \
                                  label_column_name='label', \
                                  text_column_name='modeling_text',\
                                  theta=10,
                                  i=3,
                                  vali_data=dataset["validation"],
                                  inference_datasets=inference_data,
                                  tokenizer_name='roberta-large',
                                  uname=uname,
                                  labels=['neg','pos'])
    active_learning_instance.perform_epochs()