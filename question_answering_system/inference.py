import os
import numpy as np
import pandas as pd
from prepare_data import get_embedding
import tiktoken
import pickle
import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPEN_AI_API_KEY")

COMPLETIONS_MODEL = "text-davinci-003"
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_SECTION_LEN = 1024
SEPARATOR = "\n* "
ENCODING = "gpt2"  # encoding for text-davinci-003
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 300,
    "model": COMPLETIONS_MODEL,
}

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))


def load_embeddings() -> dict:
    with open(f"question_answering_system/data/text/articles/embeddings_dict_updated.pkl", "rb") as i:
        embeddings_dict = pickle.load(i)
    
    return embeddings_dict


def load_contexts() -> pd.DataFrame:
    df = pd.read_csv(f"question_answering_system/data/text/articles/paragraphs_updated.csv").drop(columns=["Unnamed: 0"])
    df.drop_duplicates(inplace=True)
    return df


embeddings = load_embeddings()
contexts = load_contexts()


def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


def order_document_sections_by_query_similarity(query: str, contexts: dict) -> list:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    
    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)
    
    return document_similarities


def construct_prompt(question: str, context_embeddings: dict, df: pd.DataFrame) -> str:
    """
    Fetch relevant 
    """
    most_relevant_document_sections = order_document_sections_by_query_similarity(question, context_embeddings)
    
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
     
    for _, section_index in most_relevant_document_sections:
        # Add contexts until we run out of space.
        # TODO deal with duplicates in contexts and embeddings_dict as well
        try:
            document_section = df.loc[section_index]
            chosen_sections_len += document_section.tokens + separator_len            
            chosen_sections.append(SEPARATOR + document_section.paragraph.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))

            if chosen_sections_len > MAX_SECTION_LEN:
                break
        except KeyError:
            continue
    
            
    # Useful diagnostic information
    print(f"Selected {len(chosen_sections)} document sections:")
    print("\n".join(chosen_sections_indexes))
    
    header = """Answer the question as truthfully as possible using the provided context 
        and pretending that you are the author of this text, 
        and if the answer is not contained within the text below, say "I don't know."\n\nContext:\n"""
    
    return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"


def answer_query_with_context(
    query: str,
    show_prompt: bool = False
) -> str:
    prompt = construct_prompt(
        query,
        embeddings,
        contexts
    )
    
    if show_prompt:
        print(prompt)

    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [{"role": "user", "content": prompt}]
    )
    
    final_response = completion.choices[0].message.content

    return final_response


ans = answer_query_with_context(query="How do you describe US-China relationships?")
print()
