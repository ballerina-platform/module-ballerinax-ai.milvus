## Overview

The Ballerina Milvus vector store module provides a comprehensive API for integrating with Milvus vector database, enabling efficient storage, retrieval, and management of high-dimensional vectors. This module implements the Ballerina AI `VectorStore` interface and supports multiple vector search algorithms including dense, sparse, and hybrid vector search modes.

## Setup guide

To utilize the Milvus connector, you must have access to a running Milvus instance. You can use one of the following methods for that.

### Option 1: Using Docker

1. Make sure Docker is installed on your system.

2. Use the following command to start a Milvus standalone instance in docker

```bash
# Download the installation script
$ curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

#Start the Docker container
$ bash standalone_embed.sh start
```

For detailed installation instructions, refer to the official Milvus documentation.

- **Linux/macOS**: [Run Milvus in Docker](https://milvus.io/docs/install_standalone-docker.md)
- **Windows**: [Run Milvus in Docker on Windows](https://milvus.io/docs/install_standalone-windows.md)

### Option 2: Using Milvus Cloud by Zilliz

[Zilliz Cloud](https://cloud.zilliz.com/) provides a fully managed Milvus service. Follow these steps to set up your cloud instance:

1. **Sign up to Zilliz Cloud**: Visit [Zilliz Cloud](https://cloud.zilliz.com/) and create an account.

   <img src="https://raw.githubusercontent.com/ballerina-platform/module-ballerinax-milvus/main/ballerina/resources/sign_up.png" alt="Zilliz Cloud Sign Up" width="60%">

2. **Set up your account**: Complete the account setup process with your details.

   <img src="https://raw.githubusercontent.com/ballerina-platform/module-ballerinax-milvus/main/ballerina/resources/setup_account.png" alt="Account Setup" width="60%">

3. **Create a new cluster**: From the welcome page, select "Create Cluster" to start setting up your Milvus instance.

   <img src="https://raw.githubusercontent.com/ballerina-platform/module-ballerinax-milvus/main/ballerina/resources/welcome_page.png" alt="Welcome Page" width="60%">

4. **Configure cluster details**: Provide the necessary configuration details for your cluster, including cluster name, cloud provider, and region.

   <img src="https://raw.githubusercontent.com/ballerina-platform/module-ballerinax-milvus/main/ballerina/resources/create_cluster.png" alt="Create Cluster" width="60%">

5. **Download credentials**: Once your cluster is created, download the authentication credentials and connection details.

   <img src="https://raw.githubusercontent.com/ballerina-platform/module-ballerinax-milvus/main/ballerina/resources/cluster_creation.png" alt="Cluster Creation Complete" width="60%">

6. **Generate API Key**: Navigate to the API Keys section in your cluster dashboard and generate an API key for authentication.

## Quick Start

### Step 1: Import the module

```ballerina
import ballerina/ai;
import ballerinax/ai.milvus;
```

### Step 2: Initialize the Milvus vector store

```ballerina
ai:VectorStore vectorStore = check new milvus:VectorStore(
    serviceUrl = "add-milvus-service-url",
    apiKey = "add-api-key",
    config = {
        collectionName: "add-collection-name"
    }
);
```

### Step 3: Invoke the operations

```ballerina
ai:Error? result = vectorStore.add(
    [
        {
            id: "1",
            embedding: [1.0, 2.0, 3.0],
            chunk: {
                'type: "text", 
                content: "This is a chunk"
            }
        }
    ]
);

ai:VectorMatch[]|ai:Error matches = vectorStore.query({
    embedding: [1.1, 2.1, 3.1],
    filters: {
        // optional metadata filters
    }
});
```

## Examples

The Ballerina Milvus vector store module provides practical examples illustrating usage in various scenarios. Explore these [examples](https://github.com/ballerina-platform/module-ballerinax-ai.milvus/tree/main/examples).

1. [Movie recommendation system](https://github.com/ballerina-platform/module-ballerinax-ai.milvus/tree/main/examples/movie-recommendation-system)
   This example shows how to use Milvus vector store APIs to implement a movie recommendation system that stores movie embeddings and queries them to find similar movies based on vector similarity and metadata filtering.
