// SPDX-License-Identifier: Apache-2.0
//! The gRPC codec of the Connect service: requests decode only through
//! `krishiv_connect::wire` (bounded size, count, depth and fields), so no
//! untrusted byte reaches prost's unbounded decoder; responses encode with
//! prost.

use std::marker::PhantomData;

use bytes::Buf;
use krishiv_connect::wire::{self, Decoded, Request};
use tonic::Status;
use tonic::codec::{Codec, DecodeBuf, Decoder, EncodeBuf, Encoder};

use crate::status::status_of;

/// Encodes responses `E` and decodes requests `D` through
/// [`wire::decode_request`].
#[derive(Debug)]
pub struct WireCodec<E, D>(PhantomData<fn() -> (E, D)>);

impl<E, D> Default for WireCodec<E, D> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<E, D> Codec for WireCodec<E, D>
where
    E: prost::Message + Send + 'static,
    D: Request + 'static,
{
    type Encode = E;
    type Decode = Decoded<D>;
    type Encoder = WireEncoder<E>;
    type Decoder = WireDecoder<D>;

    fn encoder(&mut self) -> Self::Encoder {
        WireEncoder(PhantomData)
    }

    fn decoder(&mut self) -> Self::Decoder {
        WireDecoder(PhantomData)
    }
}

/// Encodes a response message with prost.
#[derive(Debug)]
pub struct WireEncoder<E>(PhantomData<fn(E)>);

impl<E: prost::Message> Encoder for WireEncoder<E> {
    type Item = E;
    type Error = Status;

    fn encode(&mut self, item: E, dst: &mut EncodeBuf<'_>) -> Result<(), Status> {
        item.encode(dst)
            .map_err(|e| Status::internal(format!("cannot encode a response: {e}")))
    }
}

/// Decodes a request message through the bounded decoder.
#[derive(Debug)]
pub struct WireDecoder<D>(PhantomData<fn() -> D>);

impl<D: Request> Decoder for WireDecoder<D> {
    type Item = Decoded<D>;
    type Error = Status;

    fn decode(&mut self, src: &mut DecodeBuf<'_>) -> Result<Option<Decoded<D>>, Status> {
        // tonic hands over exactly one message; its bytes are usually one
        // chunk, which `copy_to_bytes` takes without copying.
        let bytes = src.copy_to_bytes(src.remaining());
        wire::decode_request::<D>(&bytes)
            .map(Some)
            .map_err(|e| status_of(&e))
    }
}
