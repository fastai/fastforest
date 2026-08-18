use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;

use crate::{ClassifierForest, Encoder, Forest, ForestError};

const MAGIC: &[u8; 8] = b"FFM\0\x02\0\0\0";

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SavedValue {
    pub kind: u8,
    pub value: String,
}

impl SavedValue {
    pub fn validate(&self) -> Result<(), ForestError> {
        if self.kind > 5 {
            return Err(ForestError::new("saved model contains an unknown scalar type"));
        }
        match self.kind {
            2 if self.value != "0" && self.value != "1" => Err(ForestError::new("saved model contains an invalid boolean")),
            3 => self.value.parse::<i64>().map(|_| ()).map_err(|_| ForestError::new("saved model contains an invalid integer")),
            4 => self.value.parse::<f64>().map(|_| ()).map_err(|_| ForestError::new("saved model contains an invalid float")),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub markers: Vec<SavedValue>,
    pub date_columns: Vec<(usize, String)>,
    pub parameters: Vec<(String, SavedValue)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SavedModel {
    Regression { encoder: Encoder, forest: Forest, metadata: ModelMetadata },
    Classification { encoder: Encoder, forest: ClassifierForest, metadata: ModelMetadata, classes: Vec<SavedValue> },
}

#[derive(Serialize, Deserialize)]
struct Envelope {
    writer_version: String,
    model: SavedModel,
}

impl SavedModel {
    pub fn regression(encoder: Encoder, forest: Forest, metadata: ModelMetadata) -> Self {
        Self::Regression { encoder, forest, metadata }
    }

    pub fn classification(encoder: Encoder, forest: ClassifierForest, metadata: ModelMetadata, classes: Vec<SavedValue>) -> Self {
        Self::Classification { encoder, forest, metadata, classes }
    }

    pub fn validate(&self) -> Result<(), ForestError> {
        let (encoder, metadata) = match self {
            Self::Regression { encoder, forest, metadata } => {
                encoder.validate_loaded()?;
                forest.validate_loaded(encoder.encoded_to_raw().len())?;
                (encoder, metadata)
            }
            Self::Classification { encoder, forest, metadata, classes } => {
                encoder.validate_loaded()?;
                forest.validate_loaded(encoder.encoded_to_raw().len())?;
                if classes.len() != forest.n_classes() {
                    return Err(ForestError::new("saved classifier class dimensions are inconsistent"));
                }
                classes.iter().try_for_each(SavedValue::validate)?;
                (encoder, metadata)
            }
        };
        if metadata.markers.len() != encoder.input_names().len() {
            return Err(ForestError::new("saved missing-value metadata has the wrong length"));
        }
        metadata.markers.iter().try_for_each(SavedValue::validate)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, ForestError> {
        self.validate()?;
        let payload = bincode::serde::encode_to_vec(
            Envelope { writer_version: env!("CARGO_PKG_VERSION").to_owned(), model: self.clone() },
            bincode::config::standard(),
        )
        .map_err(|error| ForestError::new(format!("could not encode model: {error}")))?;
        let mut bytes = Vec::with_capacity(MAGIC.len() + payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ForestError> {
        if !bytes.starts_with(MAGIC) {
            return Err(ForestError::new("unsupported or malformed FastForest model"));
        }
        let (envelope, used): (Envelope, usize) = bincode::serde::decode_from_slice(&bytes[MAGIC.len()..], bincode::config::standard())
            .map_err(|error| ForestError::new(format!("could not decode model: {error}")))?;
        if used != bytes.len() - MAGIC.len() {
            return Err(ForestError::new("saved model contains trailing data"));
        }
        envelope.model.validate()?;
        Ok(envelope.model)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, ForestError> {
        let path = path.as_ref();
        let mut file = File::open(path).map_err(|error| ForestError::new(format!("could not open {:?}: {error}", path)))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|error| ForestError::new(format!("could not read {:?}: {error}", path)))?;
        Self::from_bytes(&bytes)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), ForestError> {
        let path = path.as_ref();
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        let bytes = self.to_bytes()?;
        let mut temporary =
            NamedTempFile::new_in(parent).map_err(|error| ForestError::new(format!("could not create model file: {error}")))?;
        temporary
            .write_all(&bytes)
            .and_then(|_| temporary.as_file_mut().sync_all())
            .map_err(|error| ForestError::new(format!("could not write model: {error}")))?;
        temporary.persist(path).map_err(|error| ForestError::new(format!("could not save {:?}: {}", path, error.error)))?;
        Ok(())
    }

    pub fn encoder(&self) -> &Encoder {
        match self {
            Self::Regression { encoder, .. } | Self::Classification { encoder, .. } => encoder,
        }
    }

    pub fn metadata(&self) -> &ModelMetadata {
        match self {
            Self::Regression { metadata, .. } | Self::Classification { metadata, .. } => metadata,
        }
    }
}
