fn main() {
    if let Err(error) = fastforest::cli::run_predict(std::env::args_os()) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
