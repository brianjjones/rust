use cfg_if::cfg_if;
use std::env;

/// Find the path to an Tensorflow library. This will try:
/// - the `TENSORFLOW_DIR` environment variable with several subdirectories appended
/// - the environment's library path (e.g. `LD_LIBRARY_PATH` in Linux)
/// - Tensorflow's default installation paths for the OS
pub fn find(library_name: &str) -> Option<PathBuf> {
    println!("BJONES inside find..");
    let file = format!(
        "{}{}{}",
        env::consts::DLL_PREFIX,
        library_name,
        env::consts::DLL_SUFFIX
    );
    log::info!("Attempting to find library: {}", file);

    // We search for the library in various different places and early-return if we find it.
    macro_rules! check_and_return {
        ($path: expr) => {

            let path = $path;
            println!("BJONES looking in {:?}", path);
            log::debug!("Searching in: {}", path.display());
            if path.is_file() {
                println!("BJONES IT WAS FOUND {:?}", path);
                log::info!("Found library at path: {}", path.display());
                return Some(path);
            }
            println!("BJONES FAILED --> {:?}", path);
        };
    }

    // Search using the `TENSORFLOW_DIR` environment variable, this may be set by users.
    if let Some(install_dir) = env::var_os(ENV_TENSORFLOW_DIR) {
        check_and_return!(PathBuf::from(install_dir));
    }

    // Search in the OS library path (i.e. `LD_LIBRARY_PATH` on Linux, `PATH` on Windows, and
    // `DYLD_LIBRARY_PATH` on MacOS).
    // https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html
    if let Some(path) = env::var_os(ENV_LIBRARY_PATH) {
        for lib_dir in env::split_paths(&path) {
            check_and_return!(lib_dir.join(&file));
        }
    }

    // Search in Tensorflow's default installation directories (if they exist).
    for default_dir in DEFAULT_INSTALLATION_DIRECTORIES
        .iter()
        .map(PathBuf::from)
        .filter(|d| d.is_dir())
    {
        check_and_return!(default_dir);
    }

    None
}

const ENV_TENSORFLOW_DIR: &'static str = "TENSORFLOW_DIR";

cfg_if! {
    if #[cfg(any(target_os = "linux"))] {
        const ENV_LIBRARY_PATH: &'static str = "LD_LIBRARY_PATH";
    } else if #[cfg(target_os = "macos")] {
        const ENV_LIBRARY_PATH: &'static str = "DYLD_LIBRARY_PATH";
    } else if #[cfg(target_os = "windows")] {
        const ENV_LIBRARY_PATH: &'static str = "PATH";
    } else {
        // This may not work but seems like a sane default for target OS' not listed above.
        const ENV_LIBRARY_PATH: &'static str = "LD_LIBRARY_PATH";
    }
}

cfg_if! {
    if #[cfg(any(target_os = "linux", target_os = "macos"))] {
        const DEFAULT_INSTALLATION_DIRECTORIES: &'static [&'static str] =
            &["/usr/local/lib", "/usr/local/lib/libtensorflow"];
    } else if #[cfg(target_os = "windows")] {
        const DEFAULT_INSTALLATION_DIRECTORIES: &'static [&'static str] = &[
            "C:\\Program Files (x86)\\Tensorflow",
            "C:\\Program Files (x86)\\tensorflow",
        ];
    } else {
        const DEFAULT_INSTALLATION_DIRECTORIES: &'static [&'static str] = &[];
    }
}

