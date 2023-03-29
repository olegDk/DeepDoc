import os
import pandas as pd
from transformers import GPT2TokenizerFast
import openai
import pickle
import random
import time
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPEN_AI_API_KEYІ")

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"


# define a retry decorator
def retry_with_exponential_backoff(
    func,
    # initial_delay: float = 1,
    # exponential_base: float = 2,
    # jitter: bool = True,
    delay = 60,
    max_retries: int = 10,
    errors: tuple = (openai.error.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        # delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                print(f"Retry error... Sleeping for {delay} seconds")
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # # Increment the delay
                # delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


def merge_data():
    # Set the directory path where the data files are stored
    directory_path = f"./data/text/articles"

    # Define a list to store the file names
    file_names = []

    # Loop through the directory and append the file names to the list
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_names.append(filename)

    # Define the path to the output file
    output_file_path = os.path.join(directory_path, "merge.txt")

    # Open the output file in write mode
    with open(output_file_path, "w") as output_file:
        # Loop through the file names and open each file in read mode
        for filename in file_names:
            file_path = os.path.join(directory_path, filename)
            with open(file_path, "r") as input_file:
                # Read the contents of the input file and write it to the output file
                contents = input_file.read()
                output_file.write(contents)


def get_paragraphs(filter_lower: int = 32, filter_upper: int = 1024) -> list:

    # Set the directory path where the data files are stored
    directory_path = f"./data/text/articles/merge.txt"

    # Create an instance of the GPT2TokenizerFast classimple telegram bot python examples
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # Open the file "merge.txt" in read mode
    with open(directory_path, "r") as file:
        # Read the contents of the file
        content = file.read()

    # Split the text into paragraphs by newline separator
    paragraphs = content.split("\n")

    # Filter paragraphs by length
    filtered_paragraphs = []
    for paragraph in paragraphs:
        # Encode the paragraph using the tokenizer and count the number of tokens
        token_count = len(tokenizer.encode(paragraph))

        # Check if the number of tokens is within the desired range
        if token_count >= filter_lower and token_count <= filter_upper:
            filtered_paragraphs.append({"paragraph": paragraph, "tokens": token_count})

    return filtered_paragraphs


@retry_with_exponential_backoff
def get_embedding(text: str, model: str=EMBEDDING_MODEL) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]


def compute_doc_embeddings(df: pd.DataFrame) -> dict:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    """
    embeddings_dict = {}
    for idx, r in tqdm(df.iterrows()):
        try:
            embeddings_dict[idx] = get_embedding(r.paragraph)
        except Exception:
            continue
    
    return embeddings_dict


def prepare_embeddings() -> dict:
    merge_data()
    paragraphs = get_paragraphs()
    paragraphs_df = pd.DataFrame(paragraphs)
    #TODO remove duplicates!!!
    paragraphs_df.to_csv(f"./data/text/articles/paragraphs.csv")
    paragraphs_df = pd.read_csv(f"./data/text/articles/paragraphs.csv").drop(columns=["Unnamed: 0"])
    embeddings_dict = compute_doc_embeddings(paragraphs_df)
    with open(f'./data/text/articles/embeddings_dict.pkl', 'wb') as f:
        pickle.dump(embeddings_dict, f)

    return embeddings_dict


def load_embeddings() -> dict:
    with open(f"./data/text/articles/embeddings_dict.pkl", "rb") as i:
        embeddings_dict = pickle.load(i)
    
    return embeddings_dict


def load_contexts() -> pd.DataFrame:
    df = pd.read_csv(f"./data/text/articles/paragraphs.csv").drop(columns=["Unnamed: 0"])
    return df


def update_embeddings(texts: list):
    embeddings = load_embeddings()
    contexts = load_contexts()
    contexts_dict = contexts.to_dict()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    for text in texts:
        cur_idx = len(embeddings)
        embeddings[cur_idx] = get_embedding(text=text)
        contexts_dict['paragraph'][cur_idx] = text
        contexts_dict['tokens'][cur_idx] = len(tokenizer.encode(text))
    
    contexts = pd.DataFrame(contexts_dict)

    with open(f"./data/text/articles/embeddings_dict_updated.pkl", "wb") as o:
        pickle.dump(embeddings, o) 
    
    contexts.to_csv(f"./data/text/articles/paragraphs_updated.csv")


# embeddings_dict = prepare_embeddings()
# update_embeddings([f"I am co-founder and co-chief investment officer of Bridgewater Assosiates.", 
#                    f"I have 4 children. Their names are: Devon, Paul, Mark, Matt.",
#                    f"My wife's name is Barbara."])

# print()
