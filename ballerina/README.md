# Ballerina Milvus Vector Store Library

[![Build](https://github.com/ballerina-platform/module-ballerinax-ai.milvus/workflows/CI/badge.svg)](https://github.com/ballerina-platform/module-ballerinax-ai.milvus/actions?query=workflow%3ACI)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/ballerina-platform/module-ballerinax-ai.milvus.svg)](https://github.com/ballerina-platform/module-ballerinax-ai.milvus/commits/master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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

1. Create a Zilliz account: Visit [Zilliz Cloud](https://cloud.zilliz.com/) and create an account.

2. Create a new cluster.

3. Navigate to the API Keys section and generate an API key.

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
    config = {
        collectionName: "add-collection-name"
    }, 
    httpConfig = {
        auth: {
            token: "add-access-token" // required for Milvus Cloud
        }
    }
);
```

### Step 3: Invoke the operations

```ballerina
ai:Error? result = vectorStore.add(
    [
        {
            id: uuid:createRandomUuid(),
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
