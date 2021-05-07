""" These functions connect with the firestore database to write and retrieve documents.
"""

import pandas as pd
import firebase_admin
from firebase_admin import firestore


def connect_to_firestore(credentials="../../secrets/animate_logos_admin_key.json",
                         database_url="https://animate-logos.firebaseio.com/"):
    """Connects to specified firestore instance with the admin credentials.

    Args:
        database_url: url of the database as specified with https://{project_id}.firebaseio.com/
        credentials: path to admin credential file

    Returns:
        A connection to the specified firestore database.
    """

    cred = firebase_admin.credentials.Certificate(credentials)
    firebase_admin.initialize_app(cred, {
        "databaseURL": database_url
    })

    return firestore.client()


def retrieve_documents_from_collection(firestore_client, collection="test_collection"):
    """Retrieves all documents of a specified firestore collection.

    Args:
        collection (string): firestore collection of the documents that should be retrieved
        firestore_client (string): specified firestore connection

    Returns:
        A list of dictionaries containing the content of the documents of the specified collection.
    """

    docs_ref = firestore_client.collection(collection)
    docs = docs_ref.get()
    results = []
    for doc in docs:
        results.append(doc.to_dict())

    results_dataframe = pd.DataFrame(results)
    return results_dataframe


def write_documents_to_collection(firestore_client, data, collection="test_write_collection"):
    """Writes documents from a dataframe to a specified collection.

    Args:
        collection (string): firestore collection where to the documents should be inserted
        data (object): dataframe
        firestore_client (object): specified firestore connection
    """

    documents = data.to_dict(orient="records")
    collection_ref = firestore_client.collection(collection)
    list(map(lambda x: collection_ref.add(x), documents))
