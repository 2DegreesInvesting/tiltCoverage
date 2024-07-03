import json
import os
import uuid
import chromadb

import pandas as pd

from dotenv import load_dotenv

import sys
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.document_stores.chroma.utils import get_embedding_function


class MyChromaDocumentStore(ChromaDocumentStore):
    def __init__(
        self,
        collection_name="documents",
        embedding_function: str = "default",
        persist_path=None,
        **embedding_function_params,
    ):
        """
        Initializes the store. The __init__ constructor is not part of the Store Protocol
        and the signature can be customized to your needs. For example, parameters needed
        to set up a database client would be passed to this method.

        Note: for the component to be part of a serializable pipeline, the __init__
        parameters must be serializable, reason why we use a registry to configure the
        embedding function passing a string.

        :param collection_name: the name of the collection to use in the database.
        :param embedding_function: the name of the embedding function to use to embed the query
        :param persist_path: where to store the database. If None, the database will be `in-memory`.
        :param embedding_function_params: additional parameters to pass to the embedding function.
        """
        # Store the params for marshalling
        self._collection_name = collection_name
        self._embedding_function = embedding_function
        self._embedding_function_params = embedding_function_params
        self._persist_path = persist_path
        # Create the client instance
        if persist_path is None:
            self._chroma_client = chromadb.Client()
        else:
            self._chroma_client = chromadb.PersistentClient(path=persist_path)

        self._collection = self._chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=get_embedding_function(
                embedding_function, **embedding_function_params
            ),
            metadata={"hnsw:space": "cosine"},
        )


def check_file_extension(filename: str, extension: str):
    """Check if filename as the expected extension, otherwise raise RuntimeError"""

    ext = os.path.splitext(filename)[-1]
    if ext != extension:
        raise RuntimeError(f"Expected a {extension} file, got {ext}.")


def read_json(filename: str):
    """Opens and loads JSON files"""

    check_file_extension(filename=filename, extension=".json")

    with open(filename) as f:
        data = json.load(f)

    return data


def write_json(filename: str, data) -> None:
    """Creates a JSON file"""

    check_file_extension(filename=filename, extension=".json")

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


def exclude_col(df: pd.DataFrame, exclude: list[str]) -> pd.DataFrame:
    """Return the DataFrame having excluded specified columns

    Args:
        df (pd.DataFrame): DataFrame from which to exclude the specified columns
        exclude (list[str]): Names of columns to exclude from the DataFrame

    Returns:
        pd.DataFrame: Input DataFrame with the columns excluded
    """

    assert isinstance(exclude, list), "Incorrect list!"

    original = df.columns
    return df[[col for col in original if col not in exclude]]


def keep_col(df: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """Specify which columns of the DataFrame to keep. Used instead of
    exclude_col if it's easier to specify what to keep than to exclude.

    Args:
        df (pd.DataFrame): DataFrame in which to keep the specified columns
        keep (list[str]): Names of columns to keep in the DataFrame

    Returns:
        pd.DataFrame: Input DataFrame with only the columns specified
    """
    assert isinstance(keep, list), "Incorrect list!"

    original = df.columns
    return df[[col for col in original if col in keep]]


def directory_exists(directory: str) -> bool:
    """Checks whether the given directory exists"""
    return os.path.exists(directory) and os.path.isdir(directory)


def files_exist_in_dir(directory: str, list_filenames: list[str]) -> bool:
    """Checks whether the list of files exist in the directory

    Args:
        directory (str): Directory to search
        list_filenames (list[str]): List of file names to look up

    Returns:
        bool: Whether the files exist in the directory
    """
    # get list of files in the directory
    list_dir = os.listdir(path=directory)

    # check that all the files in list_filenames are in the directory
    all_in_dir = all([True for file in list_filenames if file in list_dir])

    return all_in_dir


def make_md5_uuid(name: str) -> str:
    """Make a UUID using a SHA-1 hash of a namespace UUID and a name"""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))


def load_env_file():
    """Load .env file. If not loaded, exit programme."""
    env_loaded = load_dotenv()

    if not env_loaded:
        print(
            "Your environment variables could not be loaded. Check that you have a .env file."
        )
        sys.exit(0)
