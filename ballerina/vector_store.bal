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

    # Initializes the Weaviate vector store with the given configuration.
    #
    public isolated function init(
            @display {label: "Service URL"} string serviceUrl,
            @display {label: "Weaviate Configuration"} Configuration config,
            @display {label: "HTTP Configuration"} milvus:ConnectionConfig httpConfig = {}) returns ai:Error? {
        milvus:Client|error milvusClient = new (serviceUrl, httpConfig);
        if milvusClient is error {
            return error("Failed to initialize weaviate vector store", milvusClient);
        }
        self.milvusClient = milvusClient;
        self.config = config.cloneReadOnly();
        self.topK = config.topK;
        lock {
            string? chunkFieldName = self.config.cloneReadOnly().chunkFieldName;
            self.chunkFieldName = chunkFieldName is () ? "content" : chunkFieldName;
        }
    }

    public isolated function add(ai:VectorEntry[] entries) returns ai:Error? {
        if entries.length() == 0 {
            return;
        }
        lock {
            foreach ai:VectorEntry entry in entries.cloneReadOnly() {
                check self.milvusClient->upsert({
                    collectionName: self.config.collectionName,
                    data: {
                        id: check entry.id.cloneWithType(),
                        vectors: check entry.embedding.cloneWithType(),
                        "properties": {
                            "type": entry.chunk.'type,
                            [self.chunkFieldName]: entry.chunk.content
                        }
                    }
                });
            }
        } on fail var e {
            return error ai:Error("Failed to add vector entries", e);
        }
    }

    public isolated function delete(string id) returns ai:Error? {
        lock {
            int|error index = int:fromString(id);
            if index is error {
                return error ai:Error("Failed to convert id to int", index);
            }
            int|error deleteResult = self.milvusClient->delete({
                collectionName: self.config.collectionName,
                id: [index]
            });
            if deleteResult is error {
                return error ai:Error("Failed to delete vector entry", deleteResult);
            }
        }
    }

    public isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error {
        ai:VectorMatch[] finalMatches = [];
        lock {
            ai:MetadataFilters? filters = query.cloneReadOnly().filters;
            string filterValue = filters is ai:MetadataFilters ? generateFilter(filters) : "";
            check self.milvusClient->loadCollection(self.config.collectionName);
            milvus:SearchResult[][] queryResult = self.milvusClient->search({
                collectionName: self.config.collectionName,
                topK: self.topK,
                filter: filterValue,
                vectors: check query.cloneReadOnly().embedding.cloneWithType()
            });
            ai:VectorMatch[] matches = [];
            foreach milvus:SearchResult[] result in queryResult {
                foreach milvus:SearchResult item in result {
                    matches.push({
                        id: item.primaryKey.toString(),
                        embedding: [],
                        chunk: {
                            'type: "",
                            content: ""
                        },
                        similarityScore: item.similarityScore
                    });
                }
            }
            finalMatches = matches.cloneReadOnly();
        } on fail var e {
            return error ai:Error("Failed to query vector store", e);
        }
        return finalMatches;
    }
}
