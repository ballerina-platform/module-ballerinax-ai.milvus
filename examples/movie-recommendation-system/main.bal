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
import ballerina/io;
import ballerinax/ai.milvus;

configurable string serviceUrl = ?;
configurable string collectionName = ?;
configurable string token = ?;

type MovieEntry record {
    string id;
    string title;
    float[] embedding;
    string genre;
    string director;
    string year;
};

public function main() returns error? {
    milvus:VectorStore vectorStore = check new (
        serviceUrl,
        authConfig = {
            token
        },
        config = {
            collectionName,
            primaryKeyField: "primary_key"
        }
    );

    MovieEntry[] movies = [
        {
            id: "1",
            title: "The Godfather",
            embedding: [0.12875, 0.289453, 0.34358],
            genre: "Crime",
            director: "Francis Ford Coppola",
            year: "1972"
        },
        {
            id: "2",
            title: "Hateful Eight",
            embedding: [0.4365, 0.535, 0.6845],
            genre: "Suspense",
            director: "Quentin Tarantino",
            year: "2015"
        },
        {
            id: "3",
            title: "Taxi Driver",
            embedding: [0.784573, 0.8453, 0.95434],
            genre: "Drama",
            director: "Martin Scorsese",
            year: "1976"
        }
    ];

    ai:Error? addResult = vectorStore.add(from MovieEntry movie in movies
        select {
            id: movie.id,
            embedding: movie.embedding,
            chunk: {
                'type: "movie",
                content: movie.title,
                metadata: {
                    "genre": movie.genre,
                    "director": movie.director,
                    "year": movie.year
                }
            }
        }
    );

    if addResult is ai:Error {
        io:println("Error occurred while adding data to vector store. ", addResult);
        return;
    }

    ai:VectorMatch[] query = check vectorStore.query({
        embedding: [0.1, 0.2, 0.3],
        filters: {
            filters: [
                {
                    'key: "genre",
                    operator: ai:EQUAL,
                    value: "Crime"
                }
            ]
        }
    });
    io:println("Query Results: ", query);
}
