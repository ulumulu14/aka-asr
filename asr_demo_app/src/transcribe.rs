//! Most Rust ↔ Python.
//!
//! Przed kompilacją transcriber.py wstrzykuje do sys.path:
//! - site-packages venv (embedded interpreter nie aktywuje venv sam z siebie)
//! - katalog główny repo (importy model / preprocessing / train w transcriber.py)
//!
//! Potem eksponuje bufor audio jako numpy.ndarray shape `[frames, channels]`
//! i woła `transcriber.transcribe(samples, sample_rate, checkpoint_path) -> str`.
//!
//! Wymaga włączonej feature `auto-initialize` w pyo3 — interpreter Python
//! jest inicjalizowany automatycznie przy pierwszym `Python::with_gil`.

use anyhow::{anyhow, Context, Result};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::ffi::CString;
use std::path::{Path, PathBuf};

/// site-packages venv: $VIRTUAL_ENV, a gdy nieustawione — <repo>/.venv
fn venv_site_packages(repo_root: Option<&Path>) -> Option<PathBuf> {
    let venv = std::env::var_os("VIRTUAL_ENV")
        .map(PathBuf::from)
        .or_else(|| {
            repo_root
                .map(|r| r.join(".venv"))
                .filter(|p| p.exists())
        })?;

    // Windows: Lib/site-packages
    let win = venv.join("Lib").join("site-packages");
    if win.exists() {
        return Some(win);
    }
    // Linux/macOS: lib/pythonX.Y/site-packages
    for entry in std::fs::read_dir(venv.join("lib")).ok()?.flatten() {
        let sp = entry.path().join("site-packages");
        if sp.exists() {
            return Some(sp);
        }
    }
    None
}

pub fn run(
    samples: Vec<f32>,
    channels: u16,
    sample_rate: u32,
    checkpoint: &Path,
    script_path: &Path,
) -> Result<String> {
    let script_abs = script_path
        .canonicalize()
        .with_context(|| format!("nie udało się znaleźć {}", script_path.display()))?;
    let script_source = std::fs::read_to_string(&script_abs)
        .with_context(|| format!("nie udało się odczytać {}", script_abs.display()))?;

    // transcriber.py leży w asr_demo_app/ — repo z modelem to katalog wyżej
    let repo_root = script_abs
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf);
    let site_packages = venv_site_packages(repo_root.as_deref());

    let ckpt_str = checkpoint.to_string_lossy().into_owned();
    let code = CString::new(script_source).context("transcriber.py zawiera bajt NUL")?;
    // absolutna ścieżka jako __file__ — transcriber.py wyprowadza z niej repo root
    let filename = CString::new(script_abs.to_string_lossy().as_bytes())?;
    let module_name = CString::new("transcriber")?;

    let chans = channels.max(1) as usize;
    let frames = samples.len() / chans;
    let mut samples = samples;
    samples.truncate(frames * chans); // strumień mógł uciąć ostatnią ramkę w pół

    Python::attach(|py| -> PyResult<String> {
        let sys_path = py.import("sys")?.getattr("path")?;
        for p in [site_packages.as_ref(), repo_root.as_ref()]
            .into_iter()
            .flatten()
        {
            sys_path.call_method1("insert", (0, p.to_string_lossy().into_owned()))?;
        }

        // [frames, channels] z interleaved Vec<f32> (cpal layout: L,R,L,R,...)
        // from_shape_vec wypełnia row-major, więc wiersz = jedna ramka czasowa
        let arr = ndarray::Array2::from_shape_vec((frames, chans), samples)
            .map_err(|e| PyValueError::new_err(format!("nieprawidłowy kształt audio: {e}")))?;
        let py_array = arr.into_pyarray(py);

        // kompilacja modułu z pliku — pozwala edytować transcriber.py bez rekompilacji binarki
        let module = PyModule::from_code(py, &code, &filename, &module_name)?;

        module
            .getattr("transcribe")?
            .call1((py_array, sample_rate, ckpt_str))?
            .extract::<String>()
    })
    .map_err(|e| anyhow!("inferencja Pythona nie powiodła się:\n{e}"))
}
