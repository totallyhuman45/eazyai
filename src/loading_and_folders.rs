use rfd::FileDialog;
use std::fs;
use std::path::Path;
use std::path::PathBuf;





pub fn select_folder() -> Option<PathBuf> {
    let start_path = dirs::home_dir()
        .map(|home| home.join("Documents").join("YourAiProjects"))
        .unwrap_or_else(|| PathBuf::from("."));

    FileDialog::new()
        .set_directory(start_path)
        .pick_folder()
}


pub fn ensure_projects_folder_in_documents() -> PathBuf {
    let mut path = dirs::document_dir().expect("Couldn't find the Documents directory");
    path.push("YourAiProjects"); // You can change this folder name if you want

    if !path.exists() {
        fs::create_dir_all(&path).expect("Failed to create Projects folder in Documents");
    }

    path
}
pub fn create_folder_in_directory(directory: &str, folder_name: String) -> Option<PathBuf> {
    // Construct the full path by joining the directory path and the folder name
    let path = Path::new(directory).join(&folder_name);

    // Attempt to create the directory
    if fs::create_dir_all(&path).is_ok() {
        println!("Folder '{}' created successfully in '{}'", folder_name, directory);
        Some(path) // Return the path if the folder is created successfully
    } else {
        println!("Error creating folder: {}", folder_name);
        None // Return None if there was an error
    }
}