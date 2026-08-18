fn main() {
    if let Err(error) = fastforest::cli::run_view(std::env::args_os()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
