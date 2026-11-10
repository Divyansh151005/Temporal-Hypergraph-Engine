# hypergraph-server: the Temporal Hypergraph Engine over gRPC.
#
#   docker build -t temporal-hypergraph-server .
#   docker run -d -p 50051:50051 -v thg-data:/data temporal-hypergraph-server
#
# Set HYPERGRAPH_API_KEY to require an `x-api-key` on every call. Any
# hypergraph-server flag (for example `--sync buffered`) can be appended to
# `docker run`.

FROM rust:1-bookworm AS builder
WORKDIR /src
# Every workspace member's manifest must be present for cargo to resolve the
# workspace, even though only grpc-server is built.
COPY Cargo.toml Cargo.lock rust-toolchain.toml README.md ./
COPY crates ./crates
COPY benchmarks ./benchmarks
COPY proto ./proto
RUN cargo build --release --locked -p grpc-server --bin hypergraph-server \
    && strip target/release/hypergraph-server

FROM debian:bookworm-slim
RUN useradd --system --uid 10001 --home-dir /data hypergraph \
    && mkdir -p /data \
    && chown hypergraph /data
COPY --from=builder /src/target/release/hypergraph-server /usr/local/bin/hypergraph-server
USER hypergraph
ENV PORT=50051
EXPOSE 50051
VOLUME /data
ENTRYPOINT ["hypergraph-server", "--data-dir", "/data"]
