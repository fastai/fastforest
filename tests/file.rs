use std::fs;
use std::fs::File;
use std::process::Command;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray};
use arrow_ipc::reader::FileReader;
use arrow_ipc::writer::FileWriter;
use arrow_schema::{DataType, Field, Schema};
use fastforest::{
    CsvSample, CsvViewOptions, FileFitOptions, SavedModel, Task, compile_model, convert_csv_to_arrow, fit_arrow, fit_csv, fit_file,
    predict_arrow, predict_csv, predict_file, view_csv,
};

#[test]
fn mixed_file_and_model_story() {
    fastforest::cli::run_fit(["fastforest-fit", "--help"]).unwrap();
    fastforest::cli::run_predict(["fastforest-predict", "--help"]).unwrap();
    fastforest::cli::run_convert(["fastforest-convert", "--help"]).unwrap();
    fastforest::cli::run_view(["viewcsv", "--help"]).unwrap();
    fastforest::cli::run_compile(["fastforest-compile", "--help"]).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("mixed.csv");
    let mut csv = String::from("number,color,target,label\n");
    for row in 0..240 {
        let color = ["red", "green", "blue"][row % 3];
        let target = row as f32 * 0.5 + (row % 3) as f32 * 4.;
        let label = ["low", "middle", "high"][row % 3];
        csv.push_str(&format!("{row},{color},{target},{label}\n"));
    }
    fs::write(&data, csv).unwrap();

    let summary = directory.path().join("summary.csv");
    fs::write(&summary, "dataset,loss,fit_seconds,task\na,0.123456,1.23456,classification\nb,0.00123456,2.34567,classification\n").unwrap();
    assert_eq!(
        view_csv(&summary, &CsvViewOptions::default()).unwrap(),
        "dataset,loss,fit_seconds\na,0.1235,1.23\nb,0.001235,2.35\nConstants:\ntask=classification\n"
    );
    let sampled = view_csv(&data, &CsvViewOptions { sample: Some(CsvSample::Rows(20)), seed: 42, ..CsvViewOptions::default() }).unwrap();
    assert!(sampled.starts_with("20 randomly sampled rows from 240\n"));

    let regression =
        fit_csv(&data, &FileFitOptions { target: "target".into(), n_trees: Some(12), seed: Some(42), ..FileFitOptions::default() })
            .unwrap();
    let model_path = directory.path().join("regression.ffm");
    regression.save(&model_path).unwrap();
    let loaded = SavedModel::load(&model_path).unwrap();
    let predictions = directory.path().join("regression-predictions.csv");
    predict_csv(&loaded, &data, &predictions, 17, false).unwrap();
    let output = fs::read_to_string(predictions).unwrap();
    assert_eq!(output.lines().count(), 241);
    assert_eq!(output.lines().next(), Some("prediction"));

    let classifier = fit_csv(
        &data,
        &FileFitOptions {
            task: Task::Classification,
            target: "label".into(),
            n_trees: Some(12),
            seed: Some(42),
            ..FileFitOptions::default()
        },
    )
    .unwrap();
    let probabilities = directory.path().join("probabilities.csv");
    predict_csv(&classifier, &data, &probabilities, 19, true).unwrap();
    let output = fs::read_to_string(probabilities).unwrap();
    assert_eq!(output.lines().count(), 241);
    assert_eq!(output.lines().next(), Some("proba_high,proba_low,proba_middle"));

    let arrow = directory.path().join("mixed.arrow");
    let schema = Arc::new(Schema::new(vec![
        Field::new("number", DataType::Int64, false),
        Field::new("color", DataType::Utf8, false),
        Field::new("target", DataType::Float32, false),
        Field::new("label", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..240)) as ArrayRef,
            Arc::new(StringArray::from((0..240).map(|row| ["red", "green", "blue"][row % 3]).collect::<Vec<_>>())) as ArrayRef,
            Arc::new(Float32Array::from_iter_values((0..240).map(|row| row as f32 * 0.5 + (row % 3) as f32 * 4.))) as ArrayRef,
            Arc::new(StringArray::from((0..240).map(|row| ["low", "middle", "high"][row % 3]).collect::<Vec<_>>())) as ArrayRef,
        ],
    )
    .unwrap();
    let mut writer = FileWriter::try_new(File::create(&arrow).unwrap(), &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    let model = fit_arrow(
        &arrow,
        &FileFitOptions {
            task: Task::Classification,
            target: "label".into(),
            n_trees: Some(12),
            seed: Some(42),
            ..FileFitOptions::default()
        },
    )
    .unwrap();
    let arrow_predictions = directory.path().join("arrow-predictions.arrow");
    predict_arrow(&model, &arrow, &arrow_predictions, 19, true).unwrap();
    let reader = FileReader::try_new(File::open(&arrow_predictions).unwrap(), None).unwrap();
    assert_eq!(reader.map(|batch| batch.unwrap().num_rows()).sum::<usize>(), 240);

    let numeric_csv = directory.path().join("numeric.csv");
    let mut numeric = String::from("x0,x1,target\n");
    for row in 0..211 {
        numeric.push_str(&format!("{row},{},{:.3}\n", row % 7, row as f32 * 0.25));
    }
    fs::write(&numeric_csv, numeric).unwrap();
    let numeric_arrow = directory.path().join("numeric.arrow");
    convert_csv_to_arrow(&numeric_csv, &numeric_arrow, 23).unwrap();
    let model = fit_file(
        &numeric_arrow,
        &FileFitOptions { target: "target".into(), n_trees: Some(8), seed: Some(42), ..FileFitOptions::default() },
    )
    .unwrap();
    let output = directory.path().join("predictions.arrow");
    predict_file(&model, &numeric_arrow, &output, 17, false).unwrap();
    let reader = FileReader::try_new(File::open(output).unwrap(), None).unwrap();
    assert_eq!(reader.map(|batch| batch.unwrap().num_rows()).sum::<usize>(), 211);
}

#[test]
#[ignore = "builds a complete standalone release executable"]
fn standalone_executable_story() {
    let directory = tempfile::tempdir().unwrap();
    let data = directory.path().join("data.csv");
    fs::write(&data, "x,y\n0,0\n1,2\n2,4\n3,6\n4,8\n5,10\n").unwrap();
    let model =
        fit_csv(&data, &FileFitOptions { target: "y".to_owned(), n_trees: Some(4), seed: Some(42), ..FileFitOptions::default() }).unwrap();
    let executable = directory.path().join(if cfg!(windows) { "predict.exe" } else { "predict" });
    compile_model(&model, &executable).unwrap();
    let predictions = directory.path().join("predictions.csv");
    let status = Command::new(&executable).args([data.as_os_str(), "--output".as_ref(), predictions.as_os_str()]).status().unwrap();
    assert!(status.success());
    assert_eq!(fs::read_to_string(predictions).unwrap().lines().count(), 7);
}
