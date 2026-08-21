#!/usr/bin/env bash
# Regenerates the committed client stubs from proto/hypergraph_service.proto.
#
#   scripts/gen_protos.sh            # Python + TypeScript
#   scripts/gen_protos.sh python     # Python only
#   scripts/gen_protos.sh ts         # TypeScript only
#
# Python needs `grpcio-tools`; TypeScript needs `npm ci` in
# integrations/vercel-ai-sdk (it provides ts-proto).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTO_DIR="$ROOT/proto"
PROTO="$PROTO_DIR/hypergraph_service.proto"
PYTHON="${PYTHON:-python3}"
TARGET="${1:-all}"

gen_python() {
  local out="$ROOT/integrations/python-client/src/hypergraph_client/_proto"
  mkdir -p "$out"
  rm -f "$out"/hypergraph_service_pb2*.py "$out"/hypergraph_service_pb2*.pyi
  "$PYTHON" -m grpc_tools.protoc -I"$PROTO_DIR" \
    --python_out="$out" --pyi_out="$out" --grpc_python_out="$out" "$PROTO"
  # grpc_tools emits a top-level import; make it package-relative.
  "$PYTHON" - "$out/hypergraph_service_pb2_grpc.py" <<'EOF'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
text = text.replace(
    "import hypergraph_service_pb2 as hypergraph__service__pb2",
    "from . import hypergraph_service_pb2 as hypergraph__service__pb2",
)
path.write_text(text)
EOF
  echo "python stubs -> $out"
}

gen_ts() {
  local pkg="$ROOT/integrations/vercel-ai-sdk"
  local out="$pkg/src/generated"
  local plugin="$pkg/node_modules/.bin/protoc-gen-ts_proto"
  if [[ ! -x "$plugin" ]]; then
    echo "missing $plugin; run 'npm ci' in $pkg first" >&2
    exit 1
  fi
  mkdir -p "$out"
  rm -f "$out"/*.ts
  "$PYTHON" -m grpc_tools.protoc -I"$PROTO_DIR" \
    --plugin=protoc-gen-ts_proto="$plugin" \
    --ts_proto_out="$out" \
    --ts_proto_opt=outputServices=grpc-js,esModuleInterop=true,env=node,forceLong=number,useOptionals=messages,importSuffix=.js \
    "$PROTO"
  echo "typescript stubs -> $out"
}

case "$TARGET" in
  python) gen_python ;;
  ts) gen_ts ;;
  all) gen_python; gen_ts ;;
  *) echo "usage: $0 [python|ts|all]" >&2; exit 2 ;;
esac
