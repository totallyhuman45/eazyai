use crate::frame::DOCS;
use crate::{windows, PROJECT};
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::vec;
use windows::randomfunction2::*;


static FILEAMOUNT: GlobalSignal<String> = GlobalSignal::new(|| "choose file type".to_string());

static DATA: GlobalSignal<datas> = GlobalSignal::new(|| datas::defalt);

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum datas {
    Onecsv(One_Csv),
    Foldercomparison(Folder_Comparison),
    //Csvandfolder(Csv_and_folder),
    defalt,
}
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct One_Csv {
    pub csv_path: PathBuf,
    pub input_indices: Vec<i32>,
    pub label_index: i32,
    pub num_columns: i32,
    pub columns: Vec<String>,
}
#[derive(PartialEq, Debug, Clone, Serialize, Deserialize)]
pub struct Folder_Comparison {
    pub imgs_path1: PathBuf,
    pub imgs_path2: PathBuf,
}
// #[derive(PartialEq)]
// #[derive(Debug)]
// struct Csv_and_folder{
//     csv_path: PathBuf,
//     imgs_path: PathBuf,
// }

pub fn Data() -> Element {
    use_effect(move || {
        let x = load_data_from_file();
        match x {
            Some(t) => {
                *DATA.write() = t.clone();
                match t.clone() {
                    datas::Onecsv(_) => *FILEAMOUNT.write() = String::from("One Csv"),
                    datas::Foldercomparison(_) => {
                        *FILEAMOUNT.write() = String::from("Folder Comparison")
                    }
                    _ => {}
                }
                print!("loaded")
            }
            _ => print!("do nothing"),
        }
    });

    rsx! {
        if DOCS.read().clone(){
            div {
                style: "display: flex; font-family: sans-serif; gap: 20px; padding: 20px; box-sizing: border-box;",
                div {
                    style: "
                        flex: 1;
                        background-color: #d0d0d0;
                        padding: 20px;
                        text-align: left;
                        border-radius: 16px;
                        color: #000000;

                    ",
                    div {
                        class: "black-text",
                        style: { r#"
                                h1, h2, h3, h4, h5, h6 {
                                    color: black !important;
                                }
                            "# },
                        h2 { "Model Types and Setup" }

                        p { "As we talked about before, this course offers two types of AI models you can build:" }

                        h3 { "1. Data-Based Prediction Model (CSV)" }

                        p {
                            "You’ll work with a CSV (spreadsheet) file. It contains rows of information about people or events. Your AI will learn to guess if something will happen — a simple "
                            b { "true or false" }
                            " (1 or 0)."
                        }

                        b { "Example Project:" }
                        ul {
                            li { b { "Churn detection" } " — Your model will use customer data to predict whether someone will make a purchase or leave a service." }
                        }

                        h4 { "How to Start:" }
                        ul {
                            li { "Download a CSV like this one: " a { href: "https://www.kaggle.com/datasets/barun2104/telecom-churn", "📈 Telecom Churn Dataset by Barun Kumar" } }
                            li { "Make sure the CSV only contains " b { "numbers" } " (no words or text allowed)." }
                            li { "select that file here it may take a few seconds to load." }
                            li { "select the output of the model so in this example churn and select the input and in this case the information used to predict churn" }
                            li { "then save the data to a file" }

                        }

                        h4 { "now its time to learn how to load the data in python using pandas" }
                        h4 { "Code to Load a CSV:" }
                        pre {
                            code { r#"
                    these are our imports as we go you will need to import more packages,
                    for now pandas and sklearn
                    import pandas as pd
                    from sklearn.model_selection import train_test_split

                    simply load the csv from the file path this is a pandas function
                    # Load the CSV
                    df = pd.read_csv(\"your_file.csv\")  # Replace with your file path

                    this is simply a way to monitor your loading and if its working
                    # Show first few rows
                    print(df.head())

                    chose output column
                    # Pick the column you want to predict
                    target_column = \"Churn\"


                    this isnt needed for all data loading but for what where makeing this works
                    # Only use numeric columns
                    df = df.select_dtypes(include=[\"number\"])

                    split the data into a input and output variables
                    # Split into features (X) and target (y)
                    X = df.drop(columns=[target_column])
                    y = df[target_column]

                    split the data into training and test sets
                    # Split into training and test sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    "#
                            }
                        }

                        hr {}

                        h3 { "2. Image Classification Model (Folder Comparison)" }

                        p { "In this more advanced project, you’ll load images from " b { "two separate folders" } " and train your AI to tell the difference." }

                        b { "Example Project:" }
                        ul {
                            li { "One folder has pictures of " b { "cats" } ", the other has pictures of " b { "dogs" } ". The model learns to recognize them and labels new images as either a cat or dog." }
                        }

                        h4 { "How to Start:" }
                        ul {
                            li { "Download a dataset like this: " a { href: "https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset", "📁 Cats vs Dogs Dataset by Sachin" } }
                            li {
                                "Place images in " b { "two separate folders" } " like this:"
                                pre { code { "cat_folder/\n    cat1.jpg\n    cat2.jpg\n\ndog_folder/\n    dog1.jpg\n    dog2.jpg" } }
                            }
                            li {"then select your two folders this may load for a second"}
                            li{"then save the data to a file"}
                        }

                        h4 { "Code to Load Images from Two Folders:" }
                        pre {
                            code { r#"
                    these are our imports as we go you will need to import more packages,
                    for now os, numpy, tensorflow, keras, sklearn
                    import os
                    import numpy as np
                    import tensorflow as tf
                    from tensorflow.keras.utils import load_img, img_to_array
                    from sklearn.model_selection import train_test_split

                    set the image sizes we dont do this in the couse yet we will do this layer
                    # Set image size
                    image_size = (180, 180)

                    the folder paths we setup just like the ones we setup for model where makeing now
                    # Folder paths
                    cat_folder = \"path/to/cat_folder\"
                    dog_folder = \"path/to/dog_folder\"

                    a simple function to load images and labels
                    # Load images and labels
                    def load_images_from_folder(folder, label):
                        images = []
                        labels = []
                        for filename in os.listdir(folder):
                            if filename.lower().endswith((\".jpg\", \".png\", \".jpeg\")):
                                img_path = os.path.join(folder, filename)
                                img = load_img(img_path, target_size=image_size)
                                img_array = img_to_array(img) / 255.0  # Normalize
                                images.append(img_array)
                                labels.append(label)
                        return images, labels

                    use the function we made to load images and labels
                    # Load cat and dog images
                    cat_images, cat_labels = load_images_from_folder(cat_folder, 0)
                    dog_images, dog_labels = load_images_from_folder(dog_folder, 1)

                    combine the images and labels
                    # Combine them
                    X = np.array(cat_images + dog_images)
                    y = np.array(cat_labels + dog_labels)

                    split the data into training and testing sets
                    # Train/test split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

                    print out our data to see if it works
                    print(\"Images loaded:\", X.shape)
                    print(\"Labels:\", np.unique(y))
                    "#
                            }
                        }

                        hr {}

                    }

                }
            }
        }

        div {
            style: "display: flex; font-family: sans-serif; gap: 20px; padding: 20px; box-sizing: border-box;",
            div {
                style: "
                    min-height: 1000px;
                    flex: 1;
                    background-color: #d0d0d0;
                    padding: 20px;
                    text-align: left;
                    border-radius: 16px;
                    text
                ",
                Dropdown{}

                if FILEAMOUNT == String::from("One Csv"){
                    One_csv{}
                }
                if FILEAMOUNT == String::from("Folder Comparison"){
                    FolderComparison{}
                }
            }
        }
    }
}

#[component]
fn Dropdown() -> Element {
    let mut is_open = use_signal(|| false);
    let selected = use_signal(|| "Select an option".to_string());

    let options = vec!["Folder Comparison", "One Csv"];

    rsx! {
        div {
            style: "position: relative; width: 200px;",
            button {
                onclick: move |_| is_open.set(!is_open()),
                style: "width: 100%; padding: 8px; cursor: pointer;",
                "{FILEAMOUNT}"
            }
            if is_open() {
                div {
                    for x in options{
                        button{
                            onclick: move |_|
                            {
                                is_open.set(!is_open());
                                *FILEAMOUNT.write() = x.to_string();
                                if x == "One Csv"{
                                    *DATA.write() = datas::Onecsv(One_Csv::default());
                                }
                                if x == "Folder Comparison"{
                                    *DATA.write() = datas::Foldercomparison(Folder_Comparison::default());
                                }
                                println!("{:?}",FILEAMOUNT)
                            },
                            {x}
                        }
                    }
                }
            }
        }

    }
}

#[component]
fn One_csv() -> Element {
    println!("{:?}", *DATA.read());
    let data = DATA.read();
    let coulums = match &*data {
        datas::Onecsv(struc) => struc.columns.clone(),
        _ => panic!("Unsupported variant"),
    };
    let input_indices = match &*data {
        datas::Onecsv(struc) => struc.input_indices.clone(),
        _ => panic!("Unsupported variant"),
    };
    let label_index = match &*data {
        datas::Onecsv(struc) => struc.label_index,
        _ => panic!("Unsupported variant"),
    };
    let image_path = "icons8-folder-50.png"; // macOS/Linux
    rsx! {
        div{
            style:"color :#000000",
            "load csv"
        }
        button {
            onclick: move |_| {
                let mut data = DATA.write(); // Writing to the signal inside the event handler
                match &mut *data {
                    datas::Onecsv(struc) => struc.load(),
                    _ => panic!("unexpected variant"),
                }
            },
            img {
                src: "{image_path}",
                style: "width: 50px;"
            }
        }
        div {
            class: "button-bar",
            div{
                style: "color: #000000",
                 "input columns"
            }

            for (t,x) in coulums.iter().enumerate() {
                div {
                    style: "display: flex; flex-direction: column; align-items: center; flex: 1 1 30%; max-width: 300px;",

                    button{
                        // if input_indices.contains(&(t as i32)) {
                        //     style: "background-color: green;"
                        // }else{
                        //     style:"background-color:grey;"
                        // }

                        style: format_args!(
                            "background-color: {};",
                            if input_indices.contains(&(t as i32)) { "green" } else { "gray" }
                        ),
                        onclick: move |_| {
                            let mut data = DATA.write(); // Writing to the signal inside the event handler
                            match &mut *data {
                                datas::Onecsv(struc) =>
                                if struc.input_indices.contains(&(t as i32)){
                                    struc.input_indices.retain(|&x| x != (t as i32));
                                    println!("delete")
                                }else{
                                    struc.input_indices.push(t as i32)
                                },
                                _ => panic!("unexpected variant"),
                            }},
                        "{x}"
                    }
                    for _ in 0..10 {
                        span { "|" }
                    }
                }
            }
        }
         div {
            class: "button-bar",
            div{
                style: "color: #000000",
                 "output column"
            }

            for (t,x) in coulums.iter().enumerate() {
                div {
                    style: "display: flex; flex-direction: column; align-items: center; flex: 1 1 30%; max-width: 300px;",

                    button{
                        // if input_indices.contains(&(t as i32)) {
                        //     style: "background-color: green;"
                        // }else{
                        //     style:"background-color:grey;"
                        // }

                        style: format_args!(
                            "background-color: {};",
                            if label_index == t as i32 { "green" } else { "gray" }
                        ),
                        onclick: move |_| {
                            let mut data = DATA.write(); // Writing to the signal inside the event handler
                            match &mut *data {
                                datas::Onecsv(struc) =>
                                    struc.label_index = t as i32,
                                _ => panic!("unexpected variant"),
                            }},
                        "{x}"
                    }
                    for _ in 0..10 {
                        span { "|" }
                    }
                }
            }
         }

         div{
            button{

                onclick: move |_| {
                    let data = DATA.read().clone();
                    if let Some(base_path) = &*PROJECT.read() {
                        let mut full_path = base_path.clone();
                        full_path.push("data.json");

                        if let Some(path_str) = full_path.to_str() {
                            save_data_to_file(data,path_str);
                        } else {
                            println!("Failed to convert path to string");
                        }
                    } else {
                        println!("No base path set!");
                    }

                },
                "save to file"
            }
        }
    }
}

#[component]
fn FolderComparison() -> Element {
    let image_path = "icons8-folder-50.png"; // macOS/Linux
    rsx! {
        div{
            style:"color :#000000",
            "load folders"
        }
        div{
            style: "color: #000000",
            "import folder one"
            button {
                onclick: move |_| {
                    let mut data = DATA.write(); // Writing to the signal inside the event handler
                    match &mut *data {
                        datas::Foldercomparison(struc) => struc.load1(),
                        _ => panic!("unexpected variant"),
                    }
                },
                img {
                    src: "{image_path}",
                    style: "width: 50px;"
                }
            }
        }
        div{
            style: "color: #000000",
            "import folder two"
            button {
                onclick: move |_| {
                    let mut data = DATA.write(); // Writing to the signal inside the event handler
                    match &mut *data {
                        datas::Foldercomparison(struc) => struc.load2(),
                        _ => panic!("unexpected variant"),
                    }
                },
                img {
                    src: "{image_path}",
                    style: "width: 50px;"
                }
            }
        }
        div{
            style: "color: #000000",
            "this would be used for something like image separation for example one folder of dogs one folder of cats if the model is given a photo of a dog it will output 0 if it is given a model of a cat it will output 1."
        }
        div{
            button{

                onclick: move |_| {
                    let data = DATA.read().clone();
                    if let Some(base_path) = &*PROJECT.read() {
                        let mut full_path = base_path.clone();
                        full_path.push("data.json");

        // Convert to &str for the function
                        if let Some(path_str) = full_path.to_str() {
                            save_data_to_file(data,path_str);
                        } else {
                            println!("Failed to convert path to string");
                        }
                    } else {
                        println!("No base path set!");
                    }

                },
                "save to file"
            }
        }
    }
}

impl One_Csv {
    pub fn load(&mut self) {
        if let Err(e) = ensure_project_dir(&PROJECT) {
            panic!("Error: {}", e);
        }

        match pick_csv_and_get_headers_and_clone(&PROJECT) {
            Ok((filename, headers)) => {
                println!("File: {:?}", filename);
                self.csv_path = filename;
                println!("Columns: {:?}", headers);
                self.columns = headers.clone();
                self.num_columns = headers.len() as i32
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}
impl Default for One_Csv {
    fn default() -> Self {
        Self {
            csv_path: PathBuf::new(),
            input_indices: Vec::new(),
            label_index: 0,
            num_columns: 0,
            columns: Vec::new(),
        }
    }
}
impl Folder_Comparison {
    pub fn load1(&mut self) {
        match clone_selected_folder_from_signal(&PROJECT) {
            Ok(cloned_path) => {
                println!("Folder successfully cloned to: {:?}", cloned_path);
                self.imgs_path1 = cloned_path;
            }
            Err(e) => {
                eprintln!("Error cloning folder: {}", e);
            }
        }
    }
    pub fn load2(&mut self) {
        match clone_selected_folder_from_signal(&PROJECT) {
            Ok(cloned_path) => {
                println!("Folder successfully cloned to: {:?}", cloned_path);
                self.imgs_path2 = cloned_path;
            }
            Err(e) => {
                eprintln!("Error cloning folder: {}", e);
            }
        }
    }
}
impl Default for Folder_Comparison {
    fn default() -> Self {
        Self {
            imgs_path1: PathBuf::new(),
            imgs_path2: PathBuf::new(),
        }
    }
}
