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
import ballerina/time;
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
    private final string primaryKeyField;
    private string[] outputFields = ["content", "type", "vector"];

    # Initializes the Milvus vector store with the given configuration.
    #
    # + serviceUrl - The URL of the Milvus service
    # + apiKey - The API key for the Milvus service
    # + config - The configuration for the Milvus vector store
    # + httpConfig - The HTTP configuration for the Milvus service
    # + return - An error if the Milvus client initialization fails.
    public isolated function init(
            @display {label: "Service URL"} string serviceUrl,
            @display {label: "API Key"} string apiKey,
            @display {label: "Milvus Configuration"} Configuration config,
            @display {label: "HTTP Configuration"} milvus:ConnectionConfig httpConfig = {}) returns ai:Error? {
        httpConfig.authConfig = {
            token: apiKey
        };
        milvus:Client|error milvusClient = new (serviceUrl, httpConfig);
        if milvusClient is error {
            return error("failed to initialize milvus vector store", milvusClient);
        }
        self.milvusClient = milvusClient;
        self.config = config.cloneReadOnly();
        self.primaryKeyField = config.primaryKeyField;
        lock {
            foreach string item in config.additionalFields.cloneReadOnly() {
                self.outputFields.push(item);
            }
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
                record {} properties = {};
                properties["type"] = entry.chunk.'type;
                properties[self.chunkFieldName] = entry.chunk.content;
                ai:Metadata? metadata = entry.chunk.metadata;
                if metadata !is () {
                    foreach string item in metadata.keys() {
                        anydata metadataValue = metadata.get(item);
                        if metadataValue is time:Utc {
                            string utcToString = time:utcToString(metadataValue);
                            properties[item] = utcToString;
                        } else {
                            properties[item] = metadataValue;
                        }
                    }
                }
                check self.milvusClient->upsert({
                    collectionName: self.config.collectionName,
                    data: {
                        primaryKey: {
                            fieldName: self.primaryKeyField,
                            value: check int:fromString(check entry.id.cloneWithType())
                        },
                        vectors: check entry.embedding.cloneWithType(),
                        properties
                    }
                });
            }
        } on fail error e {
            return error("failed to add vector entries", e);
        }
    }

    # Deletes vector entries from the store by their reference document ID.
    #
    # + ids - One or more identifiers of the vector entries to delete
    # + return - An `ai:Error` if the deletion fails, otherwise `()` indicating success
    public isolated function delete(string|string[] ids) returns ai:Error? {
        lock {
            string[] indexArray = ids is string[] ? ids.cloneReadOnly() : [ids.cloneReadOnly()];
            int[] indexes = indexArray.'map(id => check int:fromString(id));
            int _ = check self.milvusClient->delete({
                collectionName: self.config.collectionName,
                ids: indexes
            });
        } on fail error err {
            return error("failed to delete vector entries", err);
        }
    }

    # Queries Milvus using the provided embedding vector and returns the top matches.
    #
    # + query - The query to search for. Should match the configured query mode
    # + return - A list of matching ai:VectorMatch values, or an ai:Error on failure
    public isolated function query(ai:VectorStoreQuery query) returns ai:VectorMatch[]|ai:Error {
        lock {
            ai:MetadataFilters? filters = query.cloneReadOnly().filters;
            string filterValue = filters is ai:MetadataFilters ? generateFilter(filters) : "";
            check self.milvusClient->loadCollection(self.config.collectionName);
            if query.topK == 0 {
                return error("Invalid value for topK. The value cannot be 0.");
            }
            if query.embedding is () && filters is () {
                return error("Milvus does not allow empty embedding or filters at the same time.");
            }
            if query.embedding is () {
                milvus:QueryResult[][] queryResult = check self.milvusClient->query({
                    collectionName: self.config.collectionName,
                    filter: filterValue,
                    outputFields: self.outputFields
                });
                ai:VectorMatch[] matches = [];
                foreach milvus:QueryResult[] result in queryResult {
                    foreach milvus:QueryResult item in result {
                        string id = item.hasKey("id") ? check item["id"].cloneWithType() : "";
                        float similarityScore = item.hasKey("similarityScore") ?
                            check item["similarityScore"].cloneWithType() : 0.0;
                        ai:VectorMatch vectorMatch = 
                            check buildVectorMatch(id, item, similarityScore, self.outputFields.cloneReadOnly());
                        matches.push(vectorMatch);
                    }
                }
                return matches.cloneReadOnly();
            }
            milvus:SearchResult[][] queryResult = check self.milvusClient->search({
                collectionName: self.config.collectionName,
                topK: query.topK,
                filter: filterValue,
                vectors: check query.cloneReadOnly().embedding.cloneWithType(),
                outputFields: self.outputFields
            });
            ai:VectorMatch[] matches = [];
            foreach milvus:SearchResult[] result in queryResult {
                foreach milvus:SearchResult item in result {
                    ai:VectorMatch vectorMatch = check buildVectorMatch(item.id.toString(), item.outputFields, 
                        item.similarityScore, self.outputFields.cloneReadOnly());
                    matches.push(vectorMatch);
                }
            }
            return matches.cloneReadOnly();
        } on fail error e {
            return error("failed to query vector store", e);
        }
    }
}
