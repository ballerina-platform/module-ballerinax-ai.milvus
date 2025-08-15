// Copyright (c) 2025 WSO2 LLC (http://www.wso2.com).
//
// WSO2 LLC. licenses this file to you under the Apache License,
// Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

import ballerina/ai;
import ballerinax/milvus;

# Milvus Vector Store implementation with support for Dense, Sparse, and Hybrid vector search modes.
#
# This class implements the ai:VectorStore interface and integrates with the Milvus vector database
# to provide functionality for vector upsert, query, and deletion.
#
public isolated class VectorStore {
    *ai:VectorStore;

    private final milvus:Client milvusClient;
    private final Configuration config;
    private final string chunkFieldName;
    private final int topK;

    # Initializes the Milvus vector store with the given configuration.
    #
    # + serviceUrl - The URL of the Milvus service.
    # + config - The configuration for the Milvus vector store.
    # + httpConfig - The HTTP configuration for the Milvus service.
    # 
    # + return - An error if the Milvus client initialization fails.
    public isolated function init(
            @display {label: "Service URL"} string serviceUrl,
            @display {label: "Milvus Configuration"} Configuration config,
            @display {label: "HTTP Configuration"} milvus:ConnectionConfig httpConfig = {}) returns ai:Error? {
        milvus:Client|error milvusClient = new (serviceUrl, httpConfig);
        if milvusClient is error {
            return error("Failed to initialize milvus vector store", milvusClient);
        }
        self.milvusClient = milvusClient;
        self.config = config.cloneReadOnly();
        self.topK = config.topK;
        lock {
            string? chunkFieldName = self.config.cloneReadOnly().chunkFieldName;
            self.chunkFieldName = chunkFieldName is () ? "content" : chunkFieldName;
        }
    }

    # Adds the given vector entries to the Milvus vector store.
    #
    # + entries - An array of ai:VectorEntry values to be added
    # + return - An ai:Error if vector addition fails, else ()
    public isolated function add(ai:VectorEntry[] entries) returns ai:Error? {
        if entries.length() == 0 {
            return;
        }
        lock {
            foreach ai:VectorEntry entry in entries.cloneReadOnly() {
                check self.milvusClient->upsert({
                    collectionName: self.config.collectionName,
                    data: {
                        id: check int:fromString(check entry.id.cloneWithType()),
                        vectors: check entry.embedding.cloneWithType(),
                        "properties": {
                            "type": entry.chunk.'type,
                            [self.chunkFieldName]: entry.chunk.content
                        }
                    }
                });
            }
        } on fail error e {
            return error("Failed to add vector entries", e);
        }
    }

    # Deletes vector entries from the store by their reference document ID.
    #
    # + id - The ID of the vector entry to delete.
    # + return - An `ai:Error` if the deletion fails; otherwise, `()` is returned indicating success.
    public isolated function delete(string id) returns ai:Error? {
        lock {
            int|error index = int:fromString(id);
            if index is error {
                return error ai:Error("Failed to convert id to int", index);
            }
            int|error deleteResult = self.milvusClient->delete({
                collectionName: self.config.collectionName,
                ids: [index]
            });
            if deleteResult is error {
                return error("Failed to delete vector entry", deleteResult);
            }
        }
    }

    # Queries Milvus using the provided embedding vector and returns the top matches.
    #
    # + query - The query to search for. Should match the configured query mode
    # 
    # + return - A list of matching ai:VectorMatch values, or an ai:Error on failure
    public isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error {
        ai:VectorMatch[] finalMatches = [];
        lock {
            ai:MetadataFilters? filters = query.cloneReadOnly().filters;
            string filterValue = filters is ai:MetadataFilters ? generateFilter(filters) : "";
            check self.milvusClient->loadCollection(self.config.collectionName);
            milvus:SearchResult[][] queryResult = check self.milvusClient->search({
                collectionName: self.config.collectionName,
                topK: self.topK,
                filter: filterValue,
                vectors: check query.cloneReadOnly().embedding.cloneWithType()
            });
            ai:VectorMatch[] matches = [];
            foreach milvus:SearchResult[] result in queryResult {
                foreach milvus:SearchResult item in result {
                    record{}? output = item.outputFields;
                    matches.push({
                        id: item.id.toString(),
                        embedding: output !is () 
                            ? output.hasKey("embedding") ? check output["embedding"].cloneWithType() : [] : [],
                        chunk: {
                            'type: output !is () 
                                ? output.hasKey("type") ? check output["type"].cloneWithType() : "" : "",
                            content: output !is () 
                                ? output.hasKey("content") ? check output["content"].cloneWithType() : "" : ""
                        },
                        similarityScore: item.similarityScore
                    });
                }
            }
            finalMatches = matches.cloneReadOnly();
        } on fail error e {
            return error("Failed to query vector store", e);
        }
        return finalMatches;
    }
}
