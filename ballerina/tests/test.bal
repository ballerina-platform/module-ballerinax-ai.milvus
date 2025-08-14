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

import ballerina/test;
import ballerina/ai;
import ballerinax/milvus;

string collectionName = "test_collection";
string id  = "10001";

VectorStore vectorStore = check new(
    serviceUrl = "http://localhost:19530", 
    config = {
        collectionName,
        chunkFieldName: "content",
        topK: 10
    }
);
milvus:Client milvusClient = check new(serviceUrl = "http://localhost:19530");

@test:BeforeSuite
function beforeSuite() returns error? {
    check milvusClient->createCollection({
        collectionName,
        dimension: 2
    });
}

@test:Config {
    groups: ["insert"]
}
function testInsert() returns error? {
    check vectorStore.add(entries = [
        {
            id,
            embedding: [0.1, 0.2],
            chunk: {
                'type: "text", 
                content: "test"
            }
        }
    ]);
    check milvusClient->createIndex({
        collectionName,
        primaryKey: "id",
        fieldNames: ["vector"]
    });
}

@test:Config {
    groups: ["delete"],
    dependsOn: [testQuery]
}
function testDelete() returns error? {
    check vectorStore.delete(id);
}

@test:Config {
    groups: ["query"],
    dependsOn: [testInsert]
}
function testQuery() returns error? {
    ai:VectorMatch[] matches = check vectorStore.query({
        embedding: [0.1, 0.2],
        filters: {
            filters: [
                {
                    key: "id",
                    value: id,
                    operator: ai:GREATER_THAN_OR_EQUAL
                },
                {
                    key: "id",
                    value: id,
                    operator: ai:LESS_THAN_OR_EQUAL
                }
            ],
            condition: ai:AND
        }
    });
    if matches.length() > 0 {
        test:assertEquals(matches[0].similarityScore, 1.0);
    }
}
