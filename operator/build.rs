use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=src");

    let blueprint_metadata = serde_json::json!({
        "name": "embedding-inference",
        "description": "Text embedding operator (dense vector generation) via TEI on Tangle",
        "version": env!("CARGO_PKG_VERSION"),
        "manager": {
            "Evm": "EmbeddingBSM"
        },
        "master_revision": "Latest",
        "jobs": [
            {
                "name": "embed",
                "job_index": 0,
                "description": "Generate dense vector embeddings from text input",
                "inputs": ["(string,string)"],
                "outputs": ["(bytes,uint32)"],
                "required_results": 1,
                "execution": "local"
            }
        ]
    });

    let json = serde_json::to_string_pretty(&blueprint_metadata).unwrap();
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace_root = manifest_dir.parent().expect("workspace root");
    std::fs::write(workspace_root.join("blueprint.json"), json.as_bytes()).unwrap();
}
