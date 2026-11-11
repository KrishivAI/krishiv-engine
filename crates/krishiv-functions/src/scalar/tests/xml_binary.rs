// SPDX-License-Identifier: Apache-2.0
//! `from_xml` reading a `BINARY` field: Spark 4.0.x has no BINARY case in
//! `StaxXmlParser.castTo`, so every non-empty text is a miss there; 4.1+ decodes it
//! by the output style (source-derived, v4.0.1 / v4.1.1).

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef};
use krishiv_kernels::{KernelCtx, SparkVersion};

use super::{BindArg, BindOptions, bind_scalar, col, lit, strs};

fn read_b(version: SparkVersion) -> (bool, Option<Vec<u8>>) {
    let xml = strs(&[Some("<ROW><b>[01 FF]</b><n>7</n></ROW>")]);
    let schema: ArrayRef = strs(&[Some("b binary, n int")]);
    let opts = BindOptions::default().with_kernel(KernelCtx::new().with_spark_version(version));
    let f = bind_scalar(
        "from_xml",
        &[
            xml.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&schema)),
        ],
        &opts,
    )
    .unwrap();
    let out = f.invoke(&[col(&xml), lit(&schema)], 1).unwrap();
    let st = out.as_struct();
    let n = st.column(1).as_primitive::<arrow_array::types::Int32Type>();
    let b = st.column(0);
    let bytes = (!b.is_null(0)).then(|| match b.data_type() {
        arrow_schema::DataType::BinaryView => b.as_binary_view().value(0).to_vec(),
        arrow_schema::DataType::LargeBinary => b.as_binary::<i64>().value(0).to_vec(),
        _ => b.as_binary::<i32>().value(0).to_vec(),
    });
    (n.value(0) == 7, bytes)
}

#[test]
fn binary_reads_back_from_4_1_and_always_misses_on_4_0() {
    assert_eq!(read_b(SparkVersion::V4_0), (true, None));
    assert_eq!(read_b(SparkVersion::V4_1), (true, Some(vec![0x01, 0xFF])));
}

/// Spark 4.0 has no BINARY case: every style, in both modes, is a miss (NULL) or the parse-mode error.
#[test]
fn v4_0_misses_in_every_style_and_mode() {
    use arrow_array::builder::{MapBuilder, StringBuilder};
    use krishiv_kernels::binary_style::BinaryStyle;

    let mut mode = MapBuilder::new(None, StringBuilder::new(), StringBuilder::new());
    mode.keys().append_value("mode");
    mode.values().append_value("FAILFAST");
    mode.append(true).unwrap();
    let mode: ArrayRef = Arc::new(mode.finish());
    let xml = strs(&[Some("<ROW><b>abc</b></ROW>")]);
    let schema: ArrayRef = strs(&[Some("b binary")]);
    for style in [BinaryStyle::Utf8, BinaryStyle::Basic, BinaryStyle::Hex] {
        let opts = BindOptions::default()
            .with_binary_style(style)
            .with_kernel(KernelCtx::new().with_spark_version(SparkVersion::V4_0));
        let args = [
            xml.data_type().clone().into(),
            BindArg::Literal(Arc::clone(&schema)),
            BindArg::Literal(Arc::clone(&mode)),
        ];
        let f = bind_scalar("from_xml", &args, &opts).unwrap();
        let err = f
            .invoke(&[col(&xml), lit(&schema), lit(&mode)], 1)
            .unwrap_err();
        assert!(
            err.to_string().contains("MALFORMED_RECORD_IN_PARSING"),
            "{style:?}: {err}"
        );
    }
}
