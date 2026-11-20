// SPDX-License-Identifier: Apache-2.0
//
// The Arrow Flight SQL JDBC driver against `krishiv server` (design P3-T101 §0, §4.4 m11, §11).
// Run by `tests/scripts/flight_sql_jdbc.sh`, which `crates/krishiv-cli/tests/flight_sql.rs`
// (`jdbc_flight_sql_tpch_q1`) calls; that test compares what this prints with the Rust Flight
// SQL client's answer.
//
//   java -cp flight-sql-jdbc-driver.jar FlightSqlJdbc.java <url> <q1> <q1 with ?> <date for ?>
//     [<url with user= and password=> <url with another user=>]
//
// Prints, one per line:
//   QUERY <name>, then COL <label>|<java.sql.Types code>|<precision>|<scale> per column and
//     ROW <value>\t<value>... per row (decimals as plain strings, NULL as \N), for q1 run as a
//     Statement ("statement") and as a PreparedStatement with the date bound ("prepared", m11:
//     executeQuery on a parameterized prepared query must take the query path);
//   UPDATE <count after a CREATE TABLE, an INSERT of two rows by Statement.executeUpdate and an
//     INSERT of one by PreparedStatement.executeUpdate> (3 when each runs once);
//   TABLES <rows of getTables for the table>;
//   TYPES <the type names of getTypeInfo, comma-separated>;
//   CATALOGS, SCHEMAS <TABLE_CAT of getCatalogs, TABLE_SCHEM of getSchemas, comma-separated>;
//   COLUMNS <COLUMN_NAME:TYPE_NAME of getColumns for the table, comma-separated> (GetTables with
//     include_schema underneath);
//   with the two Basic URLs (design §3.1): BASIC <rows of q1 as a Statement over the first>,
//     BASIC_WRONG refused (the second, another principal's name with the token, cannot connect).
// Exits 1 with a message on the first failure.

import java.math.BigDecimal;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.List;

public final class FlightSqlJdbc {
    private FlightSqlJdbc() {}

    /** The values of one column of a metadata result, which is closed. */
    private static List<String> strings(ResultSet rs, String column) throws SQLException {
        List<String> out = new ArrayList<>();
        try (rs) {
            while (rs.next()) {
                out.add(rs.getString(column));
            }
        }
        return out;
    }

    private static void print(String name, ResultSet rs) throws SQLException {
        System.out.println("QUERY " + name);
        ResultSetMetaData md = rs.getMetaData();
        int n = md.getColumnCount();
        for (int i = 1; i <= n; i++) {
            System.out.println("COL " + md.getColumnLabel(i) + "|" + md.getColumnType(i) + "|"
                + md.getPrecision(i) + "|" + md.getScale(i));
        }
        while (rs.next()) {
            StringBuilder row = new StringBuilder("ROW ");
            for (int i = 1; i <= n; i++) {
                if (i > 1) {
                    row.append('\t');
                }
                Object v = rs.getObject(i);
                if (v == null) {
                    row.append("\\N");
                } else if (v instanceof BigDecimal d) {
                    row.append(d.toPlainString());
                } else {
                    row.append(v);
                }
            }
            System.out.println(row);
        }
    }

    private static long count(Statement st, String table) throws SQLException {
        try (ResultSet rs = st.executeQuery("SELECT count(*) FROM " + table)) {
            if (!rs.next()) {
                throw new SQLException("count(*) returned no row");
            }
            return rs.getLong(1);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4 && args.length != 6) {
            System.err.println(
                "usage: FlightSqlJdbc <url> <q1> <q1 with ?> <date for ?> [<basic url> <wrong basic url>]");
            System.exit(2);
        }
        String url = args[0];
        try (Connection conn = DriverManager.getConnection(url)) {
            try (Statement st = conn.createStatement(); ResultSet rs = st.executeQuery(args[1])) {
                print("statement", rs);
            }
            try (PreparedStatement ps = conn.prepareStatement(args[2])) {
                ps.setDate(1, java.sql.Date.valueOf(args[3]));
                try (ResultSet rs = ps.executeQuery()) {
                    print("prepared", rs);
                }
            }
            try (Statement st = conn.createStatement()) {
                st.executeUpdate("CREATE TABLE jdbc_t (n INT) USING parquet");
                st.executeUpdate("INSERT INTO jdbc_t VALUES (1), (2)");
                try (PreparedStatement ps = conn.prepareStatement("INSERT INTO jdbc_t VALUES (?)")) {
                    ps.setInt(1, 3);
                    ps.executeUpdate();
                }
                System.out.println("UPDATE " + count(st, "jdbc_t"));
            }
            DatabaseMetaData meta = conn.getMetaData();
            int tables = 0;
            try (ResultSet rs = meta.getTables(null, null, "jdbc_t", null)) {
                while (rs.next()) {
                    tables++;
                }
            }
            System.out.println("TABLES " + tables);
            List<String> types = new ArrayList<>();
            try (ResultSet rs = meta.getTypeInfo()) {
                while (rs.next()) {
                    types.add(rs.getString("TYPE_NAME"));
                }
            }
            System.out.println("TYPES " + String.join(",", types));
            System.out.println("CATALOGS " + String.join(",", strings(meta.getCatalogs(), "TABLE_CAT")));
            System.out.println("SCHEMAS " + String.join(",", strings(meta.getSchemas(), "TABLE_SCHEM")));
            List<String> columns = new ArrayList<>();
            try (ResultSet rs = meta.getColumns(null, null, "jdbc_t", null)) {
                while (rs.next()) {
                    columns.add(rs.getString("COLUMN_NAME") + ":" + rs.getString("TYPE_NAME"));
                }
            }
            System.out.println("COLUMNS " + String.join(",", columns));
        } catch (SQLException e) {
            System.err.println("JDBC failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }
        if (args.length == 6) {
            basic(args[4], args[5], args[1]);
        }
    }

    /** Basic: `user=` and `password=` in the URL (the driver's handshake), then the wrong name. */
    private static void basic(String url, String wrong, String q1) {
        try (Connection conn = DriverManager.getConnection(url);
                Statement st = conn.createStatement();
                ResultSet rs = st.executeQuery(q1)) {
            int rows = 0;
            while (rs.next()) {
                rows++;
            }
            System.out.println("BASIC " + rows);
        } catch (SQLException e) {
            System.err.println("JDBC failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }
        try (Connection conn = DriverManager.getConnection(wrong);
                Statement st = conn.createStatement();
                ResultSet rs = st.executeQuery("SELECT 1")) {
            System.err.println("JDBC failed: another principal's name with the token connected");
            System.exit(1);
        } catch (SQLException e) {
            System.out.println("BASIC_WRONG refused");
        }
    }
}
