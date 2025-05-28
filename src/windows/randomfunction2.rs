use rfd::FileDialog;
use std::fs::{self, File};
use std::io::BufReader;
use csv::Reader;
use std::io;
use std::path::{Path, PathBuf};
use dioxus::prelude::*;
use std::env;

pub fn pick_csv_and_get_headers_and_clone(
    destination_folder: &GlobalSignal<Option<PathBuf>>,
) -> Result<(PathBuf, Vec<String>), Box<dyn std::error::Error>> {
    // Make sure destination folder is set
    let binding = destination_folder.read();
    let Some(dest_folder) = binding.as_ref() else {
        return Err("Destination folder is not set".into());
    };

    // Open file picker
    let file_path = FileDialog::new()
        .add_filter("CSV files", &["csv"])
        .pick_file()
        .ok_or("No file selected")?;

    // Get the file name and clone path
    let file_name = file_path
        .file_name()
        .ok_or("Invalid file path")?
        .to_string_lossy()
        .to_string();
    let dest_path = dest_folder.join(&file_name);

    // Copy the file
    fs::copy(&file_path, &dest_path)?;

    // Open and parse headers from the original file
    let file = File::open(&file_path)?;
    let mut reader = Reader::from_reader(BufReader::new(file));
    let headers = reader
        .headers()?
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();

    Ok((dest_path, headers))
}


pub fn ensure_project_dir(project_signal: &GlobalSignal<Option<PathBuf>>) -> std::io::Result<()> {
    // Read the path from the signal
    if let Some(project_path) = project_signal.read().as_ref() {
        // Ensure the directory exists
        fs::create_dir_all(project_path)?;
        println!("Directory ensured: {}", project_path.display());
    } else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Project path not set",
        ));
    }
    Ok(())
}
pub fn clone_selected_folder_from_signal(
    destination_signal: &GlobalSignal<Option<PathBuf>>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // Get the target destination path from the signal
    let destination_root = destination_signal.read().clone().ok_or("Destination path not set")?;

    // Open folder picker dialog
    let selected_folder = FileDialog::new()
        .set_directory(".")
        .pick_folder()
        .ok_or("No folder selected")?;

    // Get the folder's name
    let folder_name = selected_folder
        .file_name()
        .ok_or("Selected folder has no valid name")?;

    // Compose new destination path
    let new_path = destination_root.join(folder_name);

    // Recursively copy all contents
    copy_dir_all(&selected_folder, &new_path)?;

    Ok(new_path)
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;

        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

use crate::data::datas;
use crate::architecture::Layers;
use crate::settings::Settings;


pub fn save_data_to_file(data: datas,name:&str) {
    let json = serde_json::to_string(&data).unwrap();
    fs::write(name, json).unwrap();
}
pub fn save_data_to_file2(data: Vec<Layers>,name:&str) {
    let json = serde_json::to_string(&data).unwrap();
    fs::write(name, json).unwrap();
}
pub fn save_data_to_file3(data: Settings,name:&str) {
    let json = serde_json::to_string(&data).unwrap();
    fs::write(name, json).unwrap();
}

pub fn load_data_from_file() -> Option<datas> {
    if let Some(path) = &*PROJECT.read() {
        if let Err(e) = env::set_current_dir(path) {
            eprintln!("Failed to set working directory: {}", e);
        }
    }
    fs::read_to_string("data.json")
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
}
pub fn load_data_from_file2() -> Option<Vec<Layers>> {
    if let Some(path) = &*PROJECT.read() {
        if let Err(e) = env::set_current_dir(path) {
            eprintln!("Failed to set working directory: {}", e);
        }
    }
    println!("Current dir: {:?}", std::env::current_dir());
    fs::read_to_string("architecture.json")
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
}
pub fn load_data_from_file3() -> Option<Settings> {
    if let Some(path) = &*PROJECT.read() {
        if let Err(e) = env::set_current_dir(path) {
            eprintln!("Failed to set working directory: {}", e);
        }
    }
    fs::read_to_string("settings.json")
        .ok()
        .and_then(|json| serde_json::from_str(&json).ok())
}


use crate::PROJECT;

