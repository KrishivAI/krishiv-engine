// SPDX-License-Identifier: Apache-2.0
/** The Flight / Flight SQL subset this client speaks, as protobuf text (Apache-2.0, apache/arrow format/). */
export const FLIGHT_PROTO = `// SPDX-License-Identifier: Apache-2.0
// The subset of Apache Arrow's Flight.proto and FlightSql.proto this client speaks (Apache-2.0, apache/arrow
// format/). Field numbers are Arrow's; do not renumber.
syntax = "proto3";
package arrow.flight.protocol;

message Any { string type_url = 1; bytes value = 2; }
message FlightDescriptor {
  enum DescriptorType { UNKNOWN = 0; PATH = 1; CMD = 2; }
  DescriptorType type = 1;
  bytes cmd = 2;
  repeated string path = 3;
}
message Ticket { bytes ticket = 1; }
message Location { string uri = 1; }
message FlightEndpoint { Ticket ticket = 1; repeated Location location = 2; }
message FlightInfo {
  bytes schema = 1;
  FlightDescriptor flight_descriptor = 2;
  repeated FlightEndpoint endpoint = 3;
  int64 total_records = 4;
  int64 total_bytes = 5;
}
message FlightData {
  FlightDescriptor flight_descriptor = 1;
  bytes data_header = 2;
  bytes app_metadata = 3;
  bytes data_body = 1000;
}
message PutResult { bytes app_metadata = 1; }

// FlightSql.proto
message CommandStatementQuery { string query = 1; }
message CommandStatementUpdate { string query = 1; }
message DoPutUpdateResult { int64 record_count = 1; }
`;
