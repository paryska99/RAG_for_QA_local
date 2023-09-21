# import the necessary modules
import chromadb
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings


# Initialize the SentenceTransformer model
model = SentenceTransformer("C:\\Projekt\\AIAssistant\\gte-base")

# Initialize chromadb
client = chromadb.Client(settings=Settings(allow_reset=True))
collection_name = "AIAssistant"
collection = client.create_collection(name=collection_name)

def split_document_with_overlap(file_path, chunk_size, overlap_size, delimiters):
    chunks = []
    chunk_ids = []
    buffer = ""
    chunk_id = 1

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            buffer += line
            if len(buffer) >= chunk_size:
                end_index = chunk_size

                for delimiter in delimiters:
                    pos = buffer.rfind(delimiter, 0, end_index)
                    if pos != -1:
                        end_index = pos + len(delimiter)
                        break

                chunk = buffer[:end_index]
                chunks.append(chunk)
                chunk_ids.append(f"id{chunk_id}")

                buffer = buffer[end_index - overlap_size:]
                chunk_id += 1

    # Handle the last remaining part if there's any
    if buffer:
        chunks.append(buffer)
        chunk_ids.append(f"id{chunk_id}")

    return chunk_ids, chunks  # Corrected the return statement here

def get_relevant_documents(query, n_results=1):
    # Load and split the document
    example_docs = "C:\\Projekt\\AIAssistant\\example docs.txt"
    chunk_ids, split_data = split_document_with_overlap(example_docs, chunk_size=500, overlap_size=200, delimiters=["\n\n", "\n", ""])

    # Generate sentence embeddings
    sentence_embeddings = model.encode(split_data)
    sentence_embeddings_list = [embedding.tolist() for embedding in sentence_embeddings]

    # Add documents and embeddings to ChromaDB
    collection.add(embeddings=sentence_embeddings_list,
                   documents=split_data,
                   ids=chunk_ids)

    # Generate query embedding
    query_embedding = model.encode([query])  # Note: query needs to be a list
    query_embeddings_list = [query_emb.tolist() for query_emb in query_embedding]

    # Perform the query
    query_result = collection.query(query_embeddings=query_embeddings_list, n_results=n_results)

    print("Query result:", query_result)

    return query_result['documents']  # Assuming 'documents' is the key containing the results

# Testing the function
if __name__ == '__main__':
    query = "What is CryptoLib?"
    results = get_relevant_documents(query)
    # Assuming 'results' is a list of lists, so taking the first sublist
    results = results[0]
    for idx, result in enumerate(results, 1):  # start enumeration from 1 for better readability
        print(f"Document {idx}:")
        print(result)
        print("---")

