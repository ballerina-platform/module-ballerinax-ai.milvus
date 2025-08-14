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

isolated function generateFilter(ai:MetadataFilters|ai:MetadataFilter node) returns string {
    if node is ai:MetadataFilter {
        return string ` ${node.key} ${node.operator} ${generateValueField(node.value)} `;
    }
    string condition = string ` ${node.condition.toString().toUpperAscii()} `;
    string[] filters = [];
    (ai:MetadataFilters|ai:MetadataFilter)[] children = node.filters;
    foreach (ai:MetadataFilters|ai:MetadataFilter) child in children {
        string expression = generateFilter(child);
        if expression.length() > 0 {
            filters.push(expression);
        }
    }
    if filters.length() == 0 {
        return "";
    }
    if filters.length() == 1 {
        return filters[0];
    }
    return "(" + combineElements(filters, condition) + ")";
}

isolated function generateValueField(json value) returns string {
    if value is string {
        return "\"" + value + "\"";
    } else if value is json[] {
        string[] items = [];
        foreach json item in value {
            items.push(generateValueField(item));
        }
        return "[" + combineElements(items, ", ") + "]";
    }
    return value.toString();
}

isolated function combineElements(string[] parts, string separator) returns string {
    if parts.length() == 0 {
        return "";
    }
    string result = "";
    boolean first = true;
    foreach string part in parts {
        if !first {
            result += separator;
        }
        result += part;
        first = false;
    }
    return result;
}
