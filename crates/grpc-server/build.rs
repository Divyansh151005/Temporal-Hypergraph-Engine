use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("PROTOC").is_none() {
        std::env::set_var("PROTOC", protoc_bin_vendored::protoc_bin_path()?);
    }
    let proto_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../proto");
    let proto = proto_dir.join("hypergraph_service.proto");
    println!("cargo:rerun-if-changed={}", proto.display());
    tonic_build::configure()
        .bytes(["."])
        .compile_protos(&[proto], &[proto_dir])?;
    Ok(())
}
