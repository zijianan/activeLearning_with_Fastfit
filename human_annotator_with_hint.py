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
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
import getpass
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from langchain_core.documents import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import (
  AutoTokenizer, 
  AutoModelForCausalLM, 
  BitsAndBytesConfig,
  pipeline
)
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

import warnings



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
    def __init__(self, fastfit: bool, simi_query: List[str], texts: List[str], model: str = 'princeton-nlp/sup-simcse-bert-base-uncased', model_path: str = None, tokenizer_name: str = None, batch_size: int = 32):
        """
        Initializes the TextSimilarityModel class with all required parameters for setting up the model.
        
        Parameters:
        - fastfit: bool, indicates whether to use FastFit model loading strategy.
        - simi_query: List[str], list of strings containing the query texts.
        - texts: List[str], list of strings containing the texts to compare against the queries.
        - model: str, model identifier for Hugging Face Transformers.
        - model_path: str, optional, path to a pretrained model for FastFit.
        - tokenizer_name: str, optional, tokenizer identifier for Hugging Face Transformers.
        - batch_size: int, number of texts to process in each batch during vectorization.
        """
        self.fastfit = fastfit
        self.model = model
        self.simi_query = [simi_query] if isinstance(simi_query, str) else simi_query
        self.texts = texts
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name if tokenizer_name else model
        self.batch_size = batch_size

    def get_features(self, texts: List[str]) -> np.ndarray:
        """
        Extracts embeddings for the given list of texts using a pretrained model specified during initialization.
        
        Parameters:
        - texts: List[str], a list of text strings from which to extract embeddings.
        
        Returns:
        - A numpy array containing embeddings for each text.
        """
        if self.fastfit:
            model = FastFit.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model)
            model = AutoModel.from_pretrained(self.model)

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        all_embeddings = []

        for i in tqdm(range(0, len(texts), self.batch_size)):
            batch_texts = texts[i:i+self.batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=256)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def calculate_dist(self, k=5, return_all=False) -> pd.DataFrame():
        """
        Calculates the distance between query embeddings and text embeddings using FAISS.
        Optionally returns only the top 'k' nearest results for each query or all results.
        
        Parameters:
        - k: int, number of nearest texts to return for each query (default is 5).
        - return_all: bool, if True, returns all results; otherwise, returns top 'k' results per query.
        
        Returns:
        - A pandas DataFrame containing the query, closest texts, and their corresponding distances.
        """
        embeddings = self.get_features(self.texts)
        query_vecs = self.get_features(self.simi_query)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))

        D, I = index.search(query_vecs.astype('float32'), max(k, len(self.texts)))  # Search for the maximum of 'k' or total texts

        all_results = []
        for idx, (distances, indices) in enumerate(zip(D, I)):
            results = [(self.simi_query[idx], self.texts[i], d) for d, i in zip(distances, indices)]
            all_results.extend(results)

        df = pd.DataFrame(all_results, columns=["Query", "Text", "Distance"])

        if not return_all:
            return df.groupby("Query").apply(lambda x: x.nsmallest(k, 'Distance')).reset_index(drop=True)
        return df
class RAGLLMQuery:
    """ A class for initializing and querying language models, handling both OpenAI and HuggingFace backends,
    with optional vector document retrieval using FAISS. """

    def __init__(self, openai: bool, model_name: str, timeout: int = 20, temperature: float = 0.2, max_tokens: int = 512) -> None:
        """
        Initializes the RAGLLMQuery object with configuration for the language model and device setup.
        """
        self.openai: bool = openai
        self.model_name: str = model_name
        self.temperature: float = temperature
        self.max_tokens: int = max_tokens
        self.timeout: int = timeout
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm: Optional[Any] = None  # Could be ChatOpenAI or HuggingFacePipeline, depending on use.
        self.retriever: Optional[Any] = None  # Specific type depends on the implementation of FAISS.
        self.prompt: Optional[PromptTemplate] = None

    def model_initial(self) -> None:
        """ Initializes the language model based on the openai flag. """
        if self.openai:
            os.environ["OPENAI_API_KEY"] = getpass.getpass(prompt='Please enter your OPENAI_API_KEY:')
            self.llm = ChatOpenAI(
                                model=self.model_name,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                timeout=self.timeout,
                                max_retries=2
                            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
            text_generation_pipeline = pipeline(
                model=self.model_name,
                tokenizer=tokenizer,
                task="text-generation",
                temperature=self.temperature,
                repetition_penalty=1.1,
                return_full_text=True,
                max_new_tokens=self.max_tokens,
                device=self.device,
            )
            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    def vector_doc(self, list_of_docs: List[str], def_search_kwargs: Dict[str, int] = {'k': 4}, model_name: str = 'sentence-transformers/all-mpnet-base-v2') -> None:
        """ Initializes document vector retrieval system using FAISS. """
        documents__ = [Document(page_content=doc) for doc in list_of_docs]
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device}
        )
        db = FAISS.from_documents(documents__, embeddings)
        self.retriever = db.as_retriever(search_type="similarity", search_kwargs=def_search_kwargs)

    def prompt_(self, system_prompt: str = "Answer the question based on your understanding on social media platform twitter. Here is context to help:") -> None:
        """ Configures the prompt template for language model queries. """
        prompt_template = f"""
        ### [INST]
        Instruction: {system_prompt}
        """ + """
        {context}

        ### QUESTION:
        {question}

        [/INST]
        """
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

    def rag_chain(self, query: str) -> Any:
        """ Executes a retrieval-augmented generation (RAG) query using the initialized language model and document retriever. """
        if self.llm is None:
            self.model_initial()
        if self.retriever is None:
            raise Exception("Use RAGLLMQuery.vector_doc(list_of_docs) to have a vector for the retrieval documents")
        warnings.warn("Currently you are using the default prompt:'Answer the question based on your understanding on social media platform twitter. Here is context to help:' you can customize your prompt by RAGLLMQuery.prompt_(system_prompt='')")
        self.prompt_()
        llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)
        rag_chain = ({"context": self.retriever, "question": RunnablePassthrough()} | llm_chain)

        result = rag_chain.invoke(query)
        return result
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