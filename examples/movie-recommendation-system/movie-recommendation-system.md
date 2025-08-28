# Movie recommendation system with Milvus vector store

This example demonstrates the use of the Ballerina Milvus vector store module for building a movie recommendation system. The system stores movie embeddings and queries them to find similar movies based on vector similarity and metadata filtering.

## Step 1: Import the modules

Import the required modules for AI operations, I/O operations, UUID generation, and Milvus vector store.

```ballerina
import ballerina/ai;
import ballerina/io;
import ballerina/uuid;
import ballerinax/ai.milvus;
```

## Step 2: Configure the application

Set up configurable variables for Milvus connection parameters.

```ballerina
configurable string serviceUrl = ?;
configurable string collectionName = ?;
configurable string username = ?;
configurable string password = ?;
```

## Step 3: Create a vector store instance

Initialize the Milvus vector store with your service URL, collection name, and authentication credentials.

```ballerina
milvus:VectorStore vectorStore = check new (serviceUrl, {
    collectionName
}, {
    auth: {
        username,
        password
    }
});
```

Now, the `milvus:VectorStore` instance can be used for storing and querying movie embeddings.

## Step 4: Add movie embeddings to the vector store

Store movie information with their vector embeddings and metadata in the Milvus vector store.

```ballerina
ai:Vector movieEmbedding = [0.1, 0.2, 0.3];

ai:Error? addResult = vectorStore.add([
    {
        id: uuid:createRandomUuid(),
        embedding: movieEmbedding,
        chunk: {
            type: "text",
            content: "The Shawshank Redemption",
            metadata: {
                "genre": "Drama"
            }
        }
    }
]);

if addResult is ai:Error {
    io:println("Error occurred while adding an entry to the vector store", addResult);
    return;
}
```


## Step 5: Query for movie recommendations

Search for similar movies using vector similarity and apply metadata filters to refine the results.

```ballerina
// This is the embedding of the search query. It should use the same model as the embedding of the movie entries.
ai:Vector searchEmbedding = [0.05, 0.1, 0.15];

ai:VectorMatch[] query = check vectorStore.query({
    embedding: searchEmbedding,
    filters: {
        filters: [
            {
                key: "genre",
                operator: ai:EQUAL,
                value: "Drama"
            }
        ]
    }
});
io:println("Query Results: ", query);
```


## Step 6: Understanding the results

The query results contain movie recommendations with similarity scores and metadata. Each result includes:

- **id**: Unique identifier for the movie entry
- **embedding**: The vector representation of the movie
- **chunk**: Contains the movie information including type, content, and metadata
- **similarityScore**: How similar the movie is to your search query (higher scores indicate better matches)

The system can be extended to:

- Add more movies with richer metadata (director, release year, rating, etc.)
- Use real embeddings from language models instead of sample vectors
- Implement more complex filtering logic
- Build a REST API around the recommendation system
