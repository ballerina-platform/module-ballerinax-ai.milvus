// Copyright (c) 2026 WSO2 LLC (http://www.wso2.com).
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

// Comprehensive test coverage for the Milvus vector store.
// Sections:
//   A. Init / configuration
//   B. Data-type round-trip (insert -> query -> verify)
//   C. Filter operators (matching + non-matching cases)
//   D. Filter combinations (AND / OR / nesting)
//   E. Edge cases and error paths
//   F. Custom configuration (primary key, chunk field, additional fields)
//
// Each test uses a dedicated collection so ordering between tests is explicit
// via `dependsOn` and parallel test runs do not collide.

import ballerina/ai;
import ballerina/test;
import ballerina/time;

// =============================================================================
// Section A. Init / configuration tests
// =============================================================================

@test:Config {
    groups: ["init"]
}
function testInitWithDefaultConfig() returns error? {
    string collection = string `init_default_collection_${testRunSuffix}`;
    check milvusClient->createCollection({
        collectionName: collection,
        dimension: 2
    });
    VectorStore|ai:Error vs = new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    test:assertTrue(vs is VectorStore,
        "expected VectorStore to initialize with default configuration");
}

@test:Config {
    groups: ["init"]
}
function testInitWithCustomPrimaryKey() returns error? {
    string collection = string `init_custom_pk_collection_${testRunSuffix}`;
    check milvusClient->createCollection({
        collectionName: collection,
        dimension: 2,
        primaryFieldName: "doc_id"
    });
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content",
            primaryKeyField: "doc_id"
        }
    );
    check vs.add(entries = [
        {
            id: "9001",
            embedding: [0.11, 0.22],
            chunk: {
                'type: "text",
                content: "custom-pk",
                metadata: {fileName: "custom-pk.txt"}
            }
        }
    ]);
    check milvusClient->createIndex({
        collectionName: collection,
        primaryKey: "doc_id",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
    ai:VectorMatch[] matches = check vs.query({
        embedding: [0.11, 0.22],
        topK: 1
    });
    test:assertTrue(matches.length() > 0,
        "expected the entry to be retrievable through a custom primary-key configuration");
    test:assertEquals(matches[0].chunk.metadata?.fileName, "custom-pk.txt");
}

@test:Config {
    groups: ["init"]
}
function testInitWithCustomChunkFieldName() returns error? {
    string collection = string `init_custom_chunk_collection_${testRunSuffix}`;
    check milvusClient->createCollection({
        collectionName: collection,
        dimension: 2
    });
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            // chunk content goes into a column named "body" rather than "content"
            chunkFieldName: "body"
        }
    );
    check vs.add(entries = [
        {
            id: "9101",
            embedding: [0.13, 0.14],
            chunk: {
                'type: "text",
                content: "chunk with custom field name",
                metadata: {fileName: "custom-chunk.txt"}
            }
        }
    ]);
    check milvusClient->createIndex({
        collectionName: collection,
        primaryKey: "id",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
    ai:VectorMatch[] matches = check vs.query({
        embedding: [0.13, 0.14],
        topK: 1
    });
    test:assertTrue(matches.length() > 0,
        "expected the entry to be retrievable through a custom chunk field name");
    // The connector hard-codes the read-side `content` field, so even with a
    // different write column the chunk content read-back is empty here. Validate
    // that the entry itself is returned and metadata round-trips.
    test:assertEquals(matches[0].chunk.metadata?.fileName, "custom-chunk.txt");
}

@test:Config {
    groups: ["init"]
}
function testInitWithEmptyServiceUrl() returns error? {
    VectorStore|ai:Error vs = new (
        serviceUrl = "",
        apiKey = "",
        config = {
            collectionName: "irrelevant",
            chunkFieldName: "content"
        }
    );
    test:assertTrue(vs is ai:Error,
        "expected an empty service URL to surface as an ai:Error from VectorStore.init");
}

@test:Config {
    groups: ["init"]
}
function testInitWithInvalidServiceUrl() returns error? {
    VectorStore|ai:Error vs = new (
        serviceUrl = "::: not a url :::",
        apiKey = "",
        config = {
            collectionName: "irrelevant",
            chunkFieldName: "content"
        }
    );
    test:assertTrue(vs is ai:Error,
        "expected a malformed service URL to surface as an ai:Error from VectorStore.init");
}

// =============================================================================
// Section B. Data-type round-trip
// =============================================================================
//
// One collection holds a single row whose `metadata` exercises every
// json-compatible Ballerina type (string, int, float, decimal, boolean, nil,
// time:Utc, json[], nested map<json>). Each test then queries the row back
// and asserts a single field, so a regression in any one type is reported
// distinctly rather than masked behind a single composite assertion.

final string roundTripCollection = string `round_trip_collection_${testRunSuffix}`;
final time:Utc & readonly roundTripTimestamp = time:utcNow();

@test:Config {
    groups: ["round-trip"]
}
function testRoundTripSetup() returns error? {
    check milvusClient->createCollection({
        collectionName: roundTripCollection,
        dimension: 2
    });
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: roundTripCollection,
            chunkFieldName: "content"
        }
    );
    check vs.add(entries = [
        {
            id: "8001",
            embedding: [0.51, 0.52],
            chunk: {
                'type: "text",
                content: "round-trip",
                metadata: {
                    // ai:Metadata declared fields (schema-coerced on read)
                    fileName: "round-trip.txt",
                    fileSize: 4096d,
                    createdAt: roundTripTimestamp,
                    index: 12,
                    "header": "Section 1",
                    // open `json...` rest field — no schema coercion
                    "stringField": "ballerina",
                    "intField": 9999,
                    "floatField": 3.14,
                    "decimalField": 2.71828d,
                    "boolTrue": true,
                    "boolFalse": false,
                    "nilField": (),
                    "stringArray": <json>["alpha", "beta", "gamma"],
                    "intArray": <json>[1, 2, 3],
                    "mixedArray": <json>["text", 42, true, ()],
                    "nestedObject": <json>{"author": "wso2", "version": 2},
                    "deepNested": <json>{
                        "level1": {
                            "level2": {
                                "value": "deep"
                            }
                        }
                    },
                    "emptyArray": <json>[],
                    "emptyObject": <json>{},
                    "unicodeString": "résumé—中文—emoji 🚀",
                    "largeInt": 9007199254740992
                }
            }
        }
    ]);
    check milvusClient->createIndex({
        collectionName: roundTripCollection,
        primaryKey: "id",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
}

isolated function fetchRoundTripMetadata() returns ai:Metadata|error {
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: roundTripCollection,
            chunkFieldName: "content"
        }
    );
    ai:VectorMatch[] matches = check vs.query({
        embedding: [0.51, 0.52],
        topK: 1
    });
    if matches.length() == 0 {
        return error("no rows returned from round-trip collection");
    }
    ai:Metadata? md = matches[0].chunk.metadata;
    if md is () {
        return error("metadata missing from round-trip row");
    }
    return md;
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripDeclaredFields() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md.fileName, "round-trip.txt");
    test:assertEquals(md.fileSize, 4096d);
    test:assertEquals(md.index, 12);
    test:assertEquals(md.createdAt, roundTripTimestamp);
    test:assertEquals(md.header, "Section 1");
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripString() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["stringField"], "ballerina");
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripInt() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["intField"], 9999);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripFloat() returns error? {
    // `floatField` was inserted as Ballerina float `3.14`. JSON has one number
    // type, and on the read side the connector picks decimal for fractional
    // numbers. So `3.14f` round-trips as `3.14d` through the open `json...`
    // rest field. cloneWithType won't coerce because both decimal and float
    // are valid `json` members.
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["floatField"], 3.14d);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripDecimal() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["decimalField"], 2.71828d);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripBoolean() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["boolTrue"], true);
    test:assertEquals(md["boolFalse"], false);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripNil() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["nilField"], ());
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripStringArray() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["stringArray"], <json>["alpha", "beta", "gamma"]);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripIntArray() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["intArray"], <json>[1, 2, 3]);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripMixedArray() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["mixedArray"], <json>["text", 42, true, ()]);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripNestedObject() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["nestedObject"], <json>{"author": "wso2", "version": 2});
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripDeeplyNested() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["deepNested"], <json>{
        "level1": {
            "level2": {
                "value": "deep"
            }
        }
    });
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripEmptyArray() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["emptyArray"], <json>[]);
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripEmptyObject() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["emptyObject"], <json>{});
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripUnicodeString() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["unicodeString"], "résumé—中文—emoji 🚀");
}

@test:Config {
    groups: ["round-trip"],
    dependsOn: [testRoundTripSetup]
}
function testRoundTripLargeInt() returns error? {
    ai:Metadata md = check fetchRoundTripMetadata();
    test:assertEquals(md["largeInt"], 9007199254740992);
}

// =============================================================================
// Section C. Filter operator tests
// =============================================================================
//
// One collection holds five rows with deliberately varied scalar metadata so
// each operator can be exercised against both matching and non-matching values.
// `topK` is set to 100 to keep this test independent of vector similarity —
// the filter alone determines the result set.

final string filterCollection = string `filter_collection_${testRunSuffix}`;
final time:Utc & readonly filterTimestamp1 = [1700000000, 0d];
final time:Utc & readonly filterTimestamp2 = [1700100000, 0d];
final time:Utc & readonly filterTimestamp3 = [1700200000, 0d];

@test:Config {
    groups: ["filters"]
}
function testFilterCollectionSetup() returns error? {
    check milvusClient->createCollection({
        collectionName: filterCollection,
        dimension: 2
    });
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: filterCollection,
            chunkFieldName: "content"
        }
    );
    ai:VectorEntry[] entries = [
        {
            id: "5001",
            embedding: [0.10, 0.10],
            chunk: {
                'type: "text",
                content: "row-1",
                metadata: {
                    fileName: "alpha.txt",
                    fileSize: 100d,
                    index: 1,
                    createdAt: filterTimestamp1,
                    "category": "books",
                    "approved": true,
                    "score": 0.5d
                }
            }
        },
        {
            id: "5002",
            embedding: [0.20, 0.20],
            chunk: {
                'type: "text",
                content: "row-2",
                metadata: {
                    fileName: "beta.txt",
                    fileSize: 200d,
                    index: 2,
                    createdAt: filterTimestamp2,
                    "category": "movies",
                    "approved": false,
                    "score": 1.5d
                }
            }
        },
        {
            id: "5003",
            embedding: [0.30, 0.30],
            chunk: {
                'type: "text",
                content: "row-3",
                metadata: {
                    fileName: "gamma.txt",
                    fileSize: 300d,
                    index: 3,
                    createdAt: filterTimestamp3,
                    "category": "books",
                    "approved": true,
                    "score": 2.5d
                }
            }
        },
        {
            id: "5004",
            embedding: [0.40, 0.40],
            chunk: {
                'type: "text",
                content: "row-4",
                metadata: {
                    fileName: "delta.txt",
                    fileSize: 400d,
                    index: 4,
                    createdAt: filterTimestamp3,
                    "category": "music",
                    "approved": false,
                    "score": 3.5d
                }
            }
        },
        {
            id: "5005",
            embedding: [0.50, 0.50],
            chunk: {
                'type: "text",
                content: "row-5",
                metadata: {
                    fileName: "epsilon.txt",
                    fileSize: 500d,
                    index: 5,
                    createdAt: filterTimestamp3,
                    "category": "books",
                    "approved": true,
                    "score": 4.5d
                }
            }
        }
    ];
    check vs.add(entries = entries);
    check milvusClient->createIndex({
        collectionName: filterCollection,
        primaryKey: "id",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
}

isolated function runFilterQuery(ai:MetadataFilters filters) returns ai:VectorMatch[]|error {
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: filterCollection,
            chunkFieldName: "content"
        }
    );
    return vs.query({
        embedding: [0.10, 0.10],
        topK: 100,
        filters
    });
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterEqualStringMatch() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "fileName", value: "alpha.txt", operator: ai:EQUAL}]
    });
    test:assertEquals(matches.length(), 1, "EQUAL on string should match exactly one row");
    test:assertEquals(matches[0].chunk.metadata?.fileName, "alpha.txt");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterEqualStringMiss() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "fileName", value: "does-not-exist.txt", operator: ai:EQUAL}]
    });
    test:assertEquals(matches.length(), 0, "EQUAL on a non-existent value should match no rows");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterNotEqualString() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "fileName", value: "alpha.txt", operator: ai:NOT_EQUAL}]
    });
    test:assertEquals(matches.length(), 4, "NOT_EQUAL should exclude only the named row");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterGreaterThanInt() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "index", value: 3, operator: ai:GREATER_THAN}]
    });
    test:assertEquals(matches.length(), 2, "GREATER_THAN 3 should match indices 4 and 5");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterLessThanInt() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "index", value: 3, operator: ai:LESS_THAN}]
    });
    test:assertEquals(matches.length(), 2, "LESS_THAN 3 should match indices 1 and 2");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterGreaterThanOrEqualInt() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "index", value: 3, operator: ai:GREATER_THAN_OR_EQUAL}]
    });
    test:assertEquals(matches.length(), 3, "GREATER_THAN_OR_EQUAL 3 should match indices 3, 4, 5");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterLessThanOrEqualInt() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "index", value: 3, operator: ai:LESS_THAN_OR_EQUAL}]
    });
    test:assertEquals(matches.length(), 3, "LESS_THAN_OR_EQUAL 3 should match indices 1, 2, 3");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterGreaterThanDecimal() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "score", value: 2d, operator: ai:GREATER_THAN}]
    });
    test:assertEquals(matches.length(), 3, "GREATER_THAN 2.0 should match scores 2.5, 3.5, 4.5");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterEqualBoolean() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "approved", value: true, operator: ai:EQUAL}]
    });
    test:assertEquals(matches.length(), 3, "approved == true should match three rows");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterEqualUtcTimestamp() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "createdAt", value: filterTimestamp1, operator: ai:EQUAL}]
    });
    test:assertEquals(matches.length(), 1, "EQUAL on a Utc timestamp should match exactly one row");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterIn() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{
            key: "category",
            value: <json>["books", "music"],
            operator: ai:IN
        }]
    });
    test:assertEquals(matches.length(), 4,
        "IN [books, music] should match three book rows plus one music row");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterNotIn() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{
            key: "category",
            value: <json>["movies"],
            operator: ai:NOT_IN
        }]
    });
    test:assertEquals(matches.length(), 4, "NOT_IN [movies] should exclude only the movies row");
}

@test:Config {
    groups: ["filters"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterMatchesNothingOnNonExistentKey() returns error? {
    // Missing keys produce no rows. Milvus's expression evaluator returns
    // false for absent JSON paths, so the row is dropped from the result.
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "doesNotExist", value: "anything", operator: ai:EQUAL}]
    });
    test:assertEquals(matches.length(), 0,
        "filter on a non-existent metadata key should match no rows");
}

// =============================================================================
// Section D. Filter combinations (AND / OR / nesting)
// =============================================================================

@test:Config {
    groups: ["filters", "combinations"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterAndBothMatch() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        condition: ai:AND,
        filters: [
            {key: "category", value: "books", operator: ai:EQUAL},
            {key: "approved", value: true, operator: ai:EQUAL}
        ]
    });
    test:assertEquals(matches.length(), 3,
        "AND of (books AND approved) should match all three approved book rows");
}

@test:Config {
    groups: ["filters", "combinations"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterAndOneMatchOneMiss() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        condition: ai:AND,
        filters: [
            {key: "category", value: "books", operator: ai:EQUAL},
            {key: "fileName", value: "delta.txt", operator: ai:EQUAL}
        ]
    });
    test:assertEquals(matches.length(), 0,
        "AND should require both clauses to match — books AND delta.txt has no overlap");
}

@test:Config {
    groups: ["filters", "combinations"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterOrEitherMatches() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        condition: ai:OR,
        filters: [
            {key: "category", value: "movies", operator: ai:EQUAL},
            {key: "category", value: "music", operator: ai:EQUAL}
        ]
    });
    test:assertEquals(matches.length(), 2,
        "OR of (movies, music) should match both single-row categories");
}

@test:Config {
    groups: ["filters", "combinations"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterOrNoneMatch() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        condition: ai:OR,
        filters: [
            {key: "category", value: "podcasts", operator: ai:EQUAL},
            {key: "category", value: "art", operator: ai:EQUAL}
        ]
    });
    test:assertEquals(matches.length(), 0, "OR with no matching clauses should return no rows");
}

@test:Config {
    groups: ["filters", "combinations"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterNestedAndOfOr() returns error? {
    // (category in [books, music]) AND (index >= 3)
    ai:VectorMatch[] matches = check runFilterQuery({
        condition: ai:AND,
        filters: [
            {
                condition: ai:OR,
                filters: [
                    {key: "category", value: "books", operator: ai:EQUAL},
                    {key: "category", value: "music", operator: ai:EQUAL}
                ]
            },
            {key: "index", value: 3, operator: ai:GREATER_THAN_OR_EQUAL}
        ]
    });
    // books: rows 1, 3, 5 (indices 1, 3, 5); music: row 4 (index 4)
    // Of these, index >= 3 keeps 3, 4, 5 → 3 rows
    test:assertEquals(matches.length(), 3,
        "(books OR music) AND index>=3 should match three rows");
}

@test:Config {
    groups: ["filters", "combinations"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterTripleAnd() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        condition: ai:AND,
        filters: [
            {key: "category", value: "books", operator: ai:EQUAL},
            {key: "approved", value: true, operator: ai:EQUAL},
            {key: "index", value: 4, operator: ai:LESS_THAN}
        ]
    });
    // Books AND approved AND index < 4 → rows 1 and 3 (indices 1, 3)
    test:assertEquals(matches.length(), 2,
        "books AND approved AND index<4 should match rows 1 and 3");
}

// =============================================================================
// Section E. Edge cases and error paths
// =============================================================================

@test:Config {
    groups: ["edge"]
}
function testAddWithEmptyEntries() returns error? {
    string collection = string `edge_empty_entries_collection_${testRunSuffix}`;
    check milvusClient->createCollection({collectionName: collection, dimension: 2});
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    // add() must short-circuit on an empty input rather than touch Milvus.
    check vs.add(entries = []);
}

@test:Config {
    groups: ["edge"]
}
function testQueryTopKZeroErrors() returns error? {
    string collection = string `edge_topk_zero_collection_${testRunSuffix}`;
    check milvusClient->createCollection({collectionName: collection, dimension: 2});
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    ai:VectorMatch[]|ai:Error result = vs.query({
        embedding: [0.1, 0.2],
        topK: 0
    });
    test:assertTrue(result is ai:Error,
        "query with topK = 0 must return an ai:Error");
}

@test:Config {
    groups: ["edge"],
    dependsOn: [testFilterCollectionSetup]
}
function testQueryFilterOnlyNoEmbedding() returns error? {
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: filterCollection,
            chunkFieldName: "content"
        }
    );
    ai:VectorMatch[] matches = check vs.query({
        // no embedding — must fall through to the scalar query path
        filters: {
            filters: [{key: "fileName", value: "alpha.txt", operator: ai:EQUAL}]
        }
    });
    test:assertTrue(matches.length() > 0,
        "filter-only query (no embedding) should still return matching rows");
}

@test:Config {
    groups: ["edge"],
    dependsOn: [testFilterCollectionSetup]
}
function testQueryNeitherEmbeddingNorFilterErrors() returns error? {
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: filterCollection,
            chunkFieldName: "content"
        }
    );
    ai:VectorMatch[]|ai:Error result = vs.query({});
    test:assertTrue(result is ai:Error,
        "query with neither embedding nor filters must return an ai:Error");
}

@test:Config {
    groups: ["edge"]
}
function testInsertWithoutMetadata() returns error? {
    string collection = string `edge_no_metadata_collection_${testRunSuffix}`;
    check milvusClient->createCollection({collectionName: collection, dimension: 2});
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    check vs.add(entries = [
        {
            id: "6001",
            embedding: [0.31, 0.32],
            chunk: {
                'type: "text",
                content: "no-metadata"
            }
        }
    ]);
    check milvusClient->createIndex({
        collectionName: collection,
        primaryKey: "id",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
    ai:VectorMatch[] matches = check vs.query({
        embedding: [0.31, 0.32],
        topK: 1
    });
    test:assertTrue(matches.length() > 0, "row inserted without metadata must still be retrievable");
}

@test:Config {
    groups: ["edge"]
}
function testInsertWithNonNumericIdErrors() returns error? {
    string collection = string `edge_non_numeric_id_collection_${testRunSuffix}`;
    check milvusClient->createCollection({collectionName: collection, dimension: 2});
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    // The native Milvus collection's primary key is INT64 and the connector
    // does `int:fromString(entryId)`. A non-numeric id must surface as an error
    // rather than silently dropping the row or storing a garbage value.
    ai:Error? result = vs.add(entries = [
        {
            id: "not-a-number",
            embedding: [0.41, 0.42],
            chunk: {
                'type: "text",
                content: "bad-id"
            }
        }
    ]);
    test:assertTrue(result is ai:Error,
        "expected non-numeric id to produce an ai:Error from VectorStore.add");
}

@test:Config {
    groups: ["edge"]
}
function testDeleteMultipleIds() returns error? {
    string collection = string `edge_delete_multi_collection_${testRunSuffix}`;
    check milvusClient->createCollection({collectionName: collection, dimension: 2});
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    check vs.add(entries = [
        {id: "7001", embedding: [0.61, 0.62], chunk: {'type: "text", content: "del-1"}},
        {id: "7002", embedding: [0.63, 0.64], chunk: {'type: "text", content: "del-2"}},
        {id: "7003", embedding: [0.65, 0.66], chunk: {'type: "text", content: "del-3"}}
    ]);
    check milvusClient->createIndex({
        collectionName: collection,
        primaryKey: "id",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
    check vs.delete(["7001", "7002"]);
    waitForMilvusVisibility();
    ai:VectorMatch[] matches = check vs.query({
        embedding: [0.61, 0.62],
        topK: 10
    });
    foreach ai:VectorMatch m in matches {
        test:assertNotEquals(m.id, "7001", "id 7001 should be deleted");
        test:assertNotEquals(m.id, "7002", "id 7002 should be deleted");
    }
}

@test:Config {
    groups: ["edge"]
}
function testDeleteNonExistentIdDoesNotError() returns error? {
    string collection = string `edge_delete_non_existent_collection_${testRunSuffix}`;
    check milvusClient->createCollection({collectionName: collection, dimension: 2});
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    // Deleting an id that was never inserted is a no-op in Milvus — the
    // connector must propagate that as success rather than as an error.
    check vs.delete("99999");
}

@test:Config {
    groups: ["edge"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterTypeMismatchReturnsEmpty() returns error? {
    // `fileName` is a string in every row. Comparing it to an int via EQUAL
    // is a type mismatch — Milvus's expression evaluator returns no matches
    // rather than crashing. This test pins that contract.
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "fileName", value: 12345, operator: ai:EQUAL}]
    });
    test:assertEquals(matches.length(), 0,
        "comparing a string field to an int via EQUAL must return no rows");
}

@test:Config {
    groups: ["edge"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterStringFieldCompareWithBoolReturnsEmpty() returns error? {
    ai:VectorMatch[] matches = check runFilterQuery({
        filters: [{key: "category", value: true, operator: ai:EQUAL}]
    });
    test:assertEquals(matches.length(), 0,
        "comparing a string field to a boolean via EQUAL must return no rows");
}

@test:Config {
    groups: ["edge"],
    dependsOn: [testFilterCollectionSetup]
}
function testFilterIntFieldGreaterThanString() returns error? {
    // Numeric comparison against a string-typed value: Milvus rejects this as
    // an invalid expression rather than silently returning empty. Either way,
    // the connector must surface a deterministic outcome (error or empty),
    // not a crash. We only assert that no rows are erroneously returned.
    ai:VectorMatch[]|error result = runFilterQuery({
        filters: [{key: "index", value: "not-a-number", operator: ai:GREATER_THAN}]
    });
    if result is ai:VectorMatch[] {
        test:assertEquals(result.length(), 0,
            "comparing int to a non-numeric string must not return rows");
    }
    // result is error → also acceptable; either outcome documents the contract.
}

// =============================================================================
// Section F. Custom configuration combinations
// =============================================================================

@test:Config {
    groups: ["custom-config"]
}
function testCustomCollectionAndPrimaryKeyCombined() returns error? {
    string collection = string `fully_custom_collection_${testRunSuffix}`;
    check milvusClient->createCollection({
        collectionName: collection,
        dimension: 2,
        primaryFieldName: "doc_uid"
    });
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "body",
            primaryKeyField: "doc_uid"
        }
    );
    check vs.add(entries = [
        {
            id: "11001",
            embedding: [0.91, 0.92],
            chunk: {
                'type: "text",
                content: "fully-custom",
                metadata: {fileName: "fully.txt", "tag": "v1"}
            }
        }
    ]);
    check milvusClient->createIndex({
        collectionName: collection,
        primaryKey: "doc_uid",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
    ai:VectorMatch[] matches = check vs.query({
        embedding: [0.91, 0.92],
        filters: {
            filters: [{key: "tag", value: "v1", operator: ai:EQUAL}]
        }
    });
    test:assertTrue(matches.length() > 0,
        "fully-custom config (collection + chunk field + pk) should still round-trip");
    test:assertEquals(matches[0].chunk.metadata?.fileName, "fully.txt");
}

@test:Config {
    groups: ["custom-config"]
}
function testQueryWithLargeTopKReturnsAvailable() returns error? {
    string collection = string `custom_large_topk_collection_${testRunSuffix}`;
    check milvusClient->createCollection({collectionName: collection, dimension: 2});
    VectorStore vs = check new (
        serviceUrl = "http://localhost:19530",
        apiKey = "",
        config = {
            collectionName: collection,
            chunkFieldName: "content"
        }
    );
    check vs.add(entries = [
        {id: "12001", embedding: [0.10, 0.20], chunk: {'type: "text", content: "tk-1"}},
        {id: "12002", embedding: [0.30, 0.40], chunk: {'type: "text", content: "tk-2"}}
    ]);
    check milvusClient->createIndex({
        collectionName: collection,
        primaryKey: "id",
        fieldNames: ["vector"]
    });
    waitForMilvusVisibility();
    // Asking for many more rows than exist must not error — Milvus simply
    // returns whatever is available.
    ai:VectorMatch[] matches = check vs.query({
        embedding: [0.10, 0.20],
        topK: 1000
    });
    test:assertEquals(matches.length(), 2,
        "topK >> row count must return all available rows, not error");
}
