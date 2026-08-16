use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::{NamedTempFile, tempdir};

use crate::{ForestError, SavedModel};

fn escaped_path(path: &Path) -> String {
    format!("{:?}", path.to_string_lossy())
}

pub fn compile_model(model: &SavedModel, output: impl AsRef<Path>) -> Result<(), ForestError> {
    model.validate()?;
    let directory = tempdir().map_err(|error| ForestError::new(format!("could not create build directory: {error}")))?;
    let model_path = directory.path().join("model.ffm");
    model.save(&model_path)?;
    let source_dir = directory.path().join("src");
    fs::create_dir(&source_dir).map_err(|error| ForestError::new(format!("could not create build source: {error}")))?;
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dependency = if manifest_dir.join("Cargo.toml").is_file() {
        format!("fastforest = {{ path = {} }}", escaped_path(&manifest_dir))
    } else {
        format!("fastforest = \"={}\"", env!("CARGO_PKG_VERSION"))
    };
    let manifest = format!(
        "[package]\nname = \"fastforest-embedded\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[dependencies]\n{dependency}\n"
    );
    fs::write(directory.path().join("Cargo.toml"), manifest)
        .map_err(|error| ForestError::new(format!("could not write build manifest: {error}")))?;
    let source = format!(
        "const MODEL: &[u8] = include_bytes!({});\nfn main() {{\n    if let Err(error) = fastforest::cli::run_embedded_predict(MODEL, std::env::args_os()) {{\n        eprintln!(\"{{error}}\");\n        std::process::exit(1);\n    }}\n}}\n",
        escaped_path(&model_path)
    );
    fs::write(source_dir.join("main.rs"), source)
        .map_err(|error| ForestError::new(format!("could not write build source: {error}")))?;
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let status = Command::new(cargo)
        .args(["build", "--release", "--quiet"])
        .current_dir(directory.path())
        .status()
        .map_err(|error| ForestError::new(format!("could not run Cargo: {error}")))?;
    if !status.success() {
        return Err(ForestError::new("Cargo could not build the standalone predictor"));
    }
    let executable = directory
        .path()
        .join("target/release")
        .join(if cfg!(windows) { "fastforest-embedded.exe" } else { "fastforest-embedded" });
    let output = output.as_ref();
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let temporary = NamedTempFile::new_in(parent)
        .map_err(|error| ForestError::new(format!("could not create executable output: {error}")))?;
    fs::copy(&executable, temporary.path())
        .map_err(|error| ForestError::new(format!("could not copy standalone predictor: {error}")))?;
    temporary
        .persist(output)
        .map_err(|error| ForestError::new(format!("could not save {:?}: {}", output, error.error)))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FileFitOptions, fit_csv};

    #[test]
    #[ignore = "builds a complete standalone release executable"]
    fn standalone_executable_story() {
        let directory = tempdir().unwrap();
        let data = directory.path().join("data.csv");
        fs::write(&data, "x,y\n0,0\n1,2\n2,4\n3,6\n4,8\n5,10\n").unwrap();
        let model = fit_csv(
            &data,
            &FileFitOptions {
                target: "y".to_owned(),
                n_trees: Some(4),
                seed: Some(42),
                ..FileFitOptions::default()
            },
        )
        .unwrap();
        let executable = directory.path().join(if cfg!(windows) { "predict.exe" } else { "predict" });
        compile_model(&model, &executable).unwrap();
        let predictions = directory.path().join("predictions.csv");
        let status = Command::new(&executable)
            .args([data.as_os_str(), "--output".as_ref(), predictions.as_os_str()])
            .status()
            .unwrap();
        assert!(status.success());
        assert_eq!(fs::read_to_string(predictions).unwrap().lines().count(), 7);
    }
}
