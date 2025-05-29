use crate::windows::architecture::ActivationFunctions;
use crate::windows::architecture::Layers;
use crate::windows::data::datas;
use crate::windows::loading_and_folders2::*;
use crate::windows::settings::{optimizer, Settings};
use crate::PROJECT;
use dioxus::prelude::*;

pub fn Run() -> Element {
    let data: Memo<Option<datas>> = use_memo(load_data_from_file);
    let architecture = use_memo(load_data_from_file2);
    let settings = use_memo(load_data_from_file3);

    println!("{:?}", architecture);
    let mut script: Signal<PathBuf> = use_signal(PathBuf::new);
    rsx! {

        div{
           style: format_args!(
               "display: inline-block; background-color: {}; border-radius: 8px; padding: 4px 8px;",
                if data().is_some() { "green" } else { "gray" }
            ),
            "data"
        }
        div{
           style: format_args!(
               "display: inline-block; background-color: {}; border-radius: 8px; padding: 4px 8px;",
                if architecture().is_some() { "green" } else { "gray" }
            ),
            "architecture"
        }
        div{
           style: format_args!(
               "display: inline-block; background-color: {}; border-radius: 8px; padding: 4px 8px;",
                if settings().is_some() { "green" } else { "gray" }
            ),
            "settings"
        }
        div{
            button{
                onclick: move |_| {
                    if architecture().is_some() && data().is_some() && settings().is_some(){
                        script.set(write_py_code_to_file(&PROJECT.read().clone().unwrap(),&compile(data().unwrap(),architecture().unwrap(),settings().unwrap())));
                    }
                },

               "compile to python file"
            }
        }
        div{
            h1{"1.download python"}
            "for this you should get version 3.12.10 "
            Link{}
        }
        div{
            h1{"2.pip install imports"}
            div{
               "open terminal or  and paste in this"
            }
            div{
               "python3 -m pip install tensorflow pandas scikit-learn matplotlib"
            }
        }
        div{
            h1{"3.run code"}
            div{
               "put this in terminal and watch your code run"
            }
            div{
               "python3 {script().display()}"
            }
        }
    }
}

fn compile(data: datas, architecture: Vec<Layers>, settings: Settings) -> String {
    let datatype = match data.clone() {
        datas::Onecsv(x) => true,
        _ => false,
    };

    let data_code = match data {
        datas::Onecsv(x) => {
            let csv_path_str = x.csv_path.to_str().unwrap_or("Invalid path");
            let input_indices_str = x
                .input_indices
                .iter()
                .map(|s| s.to_string()) // Convert i32 to String
                .collect::<Vec<String>>()
                .join(", ");
            let label_index = x.label_index;
            let data_code = format!(
                r#"import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# === CONFIGURATION ===
file_path = "{}"  # <- Update this
input_indices = [{}]  # <- Update based on actual CSV columns
label_index = {}  # <- Index of the label column

# === LOAD CSV USING PANDAS ===
df = pd.read_csv(file_path)

# Convert to NumPy arrays
X = df.iloc[:, input_indices].values.astype("float32")
y = df.iloc[:, label_index].values.astype("float32")
input_dim = X.shape[1]  # <- Fix: define input_dim\n"#,
                csv_path_str, input_indices_str, label_index
            );
            data_code
        }
        datas::Foldercomparison(y) => {
            format!(
                r#"import os
import shutil
import tempfile
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# === CONFIG ===
image_size = ({}, {})
batch_size = {}
folder_a = {:?}
folder_b = {:?}

# === TEMP DIRECTORY ===
temp_dir = tempfile.mkdtemp()

# === IMAGE VALIDATION ===
def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
            img_format = img.format.lower()
            if img_format not in ["jpeg", "png", "bmp", "gif"]:
                return False
            if os.path.getsize(path) == 0:
                return False
            return img.mode in ('RGB', 'L')
    except:
        return False

# === COPY AND CONVERT CLEAN IMAGES TO RGB ===
class_dirs = [("class_0", folder_a), ("class_1", folder_b)]
for class_name, src_dir in class_dirs:
    dst_dir = os.path.join(temp_dir, class_name)
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            src_path = os.path.join(src_dir, fname)
            try:
                with Image.open(src_path) as img:
                    img = img.convert("RGB")  # Force 3-channel
                    if os.path.getsize(src_path) == 0:
                        continue
                    save_path = os.path.join(dst_dir, fname)
                    img.save(save_path)
            except Exception as e:
                print(f"Skipped invalid")

# === LOAD DATASET ===
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    temp_dir,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    temp_dir,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)

# === CACHE & PREFETCH ===
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
"#,
                match architecture.first().unwrap() {
                    Layers::Inputimg(x) => x.width,
                    _ => panic!(),
                },
                match architecture.first().unwrap() {
                    Layers::Inputimg(x) => x.height,
                    _ => panic!(),
                },
                settings.batch,
                y.imgs_path1,
                y.imgs_path2
            )
        }
        _ => "h".to_string(),
    };
    let mut layers_code: String = "h".to_string();
    let mut settings_code: String = "h".to_string();

    if datatype {
        layers_code = format!(
            "# === DEFINE MODEL ===\nmodel = tf.keras.Sequential([{}])\n",
            layers_to_code(architecture.clone())
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        settings_code = format!(
            r#"model.compile(optimizer='{}',
loss='binary_crossentropy', # Binary classification loss always binary_crossentropy
metrics=['accuracy']) # always accuracy

model.summary()

# === TRAIN MODEL ===

history = model.fit(X, y, epochs={}, batch_size={}, validation_split=0.2)

# === OPTIONAL: SAVE MODEL ===

model.save("model.keras")

# === PLOTS: Accuracy and Loss ===

plt.figure(figsize=(12, 5))

# Accuracy plot

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()

# Save plot as image

import os

# Get the directory where the CSV file is located
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, "training_metrics.png")

# Save the plot to that directory
plt.savefig(output_path)
plt.close()"#,
            opdimizer_to_string(settings.clone().optimizer),
            settings.clone().epochs,
            settings.batch
        );
    } else {
        layers_code = format!(
            "# === DEFINE MODEL ===\nmodel = tf.keras.Sequential([{}])\n",
            layers_to_code2(architecture.clone(), architecture)
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );

        settings_code = format!(
            r#"
# === COMPILE & TRAIN ===
model.compile(optimizer='{}',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs={}
)

# === SAVE MODEL ===
model.save("binary_image_classifier.h5")

# === PLOT STATS ===
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

import os

# Get the directory where the CSV file is located
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, "training_metrics.png")

# Save the plot to that directory
plt.savefig(output_path)

# === CLEANUP (optional) ===
# shutil.rmtree(temp_dir)
"#,
            opdimizer_to_string(settings.optimizer),
            settings.epochs.clone()
        )
    }

    format!("{}{}{}", data_code, layers_code, settings_code)
}
fn layers_to_code(vec: Vec<Layers>) -> Vec<String> {
    let mut x: Vec<String> = vec![];
    for t in vec {
        x.push(match t {
            Layers::Inputi32(y) => "tf.keras.layers.InputLayer(shape=(input_dim,))".to_string(), // needs input_dimensions to be a var called input dim
            Layers::Dense(y) => format!(
                "tf.keras.layers.Dense({}, activation='{}')",
                y.neurons,
                activation_to_string(y.activation)
            ),
            Layers::Dropout(y) => format!("tf.keras.layers.Dropout({})", y.rate),
            Layers::BatchNormalization(y) => "tf.keras.layers.BatchNormalization()".to_string(),
            Layers::Activation(y) => format!(
                "tf.keras.layers.Activation('{}')",
                activation_to_string(y.activation)
            ),
            Layers::Inputimg(y) => format!("tf.keras.Input(shape=(*image_size, {}))", y.channels),
            Layers::Conv2D(y) => format!(
                "tf.keras.layers.Conv2D(
                filters={},
                kernel_size=({}, {}),
                activation='{}',
                input_shape=({}, {}, {})  # Only needed on the first layer
            )
            ",
                y.filters,
                y.kernel_size.0,
                y.kernel_size.1,
                activation_to_string(y.activation),
                y.input_shape.0,
                y.input_shape.1,
                y.input_shape.2
            ),
            Layers::MaxPooling2D(y) => format!(
                "tf.keras.layers.MaxPooling2D(pool_size=({}, {}), strides=({}, {}))",
                y.pool_size.0, y.pool_size.1, y.strides.0, y.strides.1
            ),
            Layers::Flatten(y) => "tf.keras.layers.Flatten()".to_string(),
            Layers::Output(y) => format!(
                "tf.keras.layers.Dense(1, activation='{}')",
                activation_to_string(y.activation)
            ),
            _ => "nope".to_string(),
        })
    }
    x
}

fn layers_to_code2(vec: Vec<Layers>, architecture: Vec<Layers>) -> Vec<String> {
    let mut x: Vec<String> = vec![];
    for t in vec {
        x.push(match t {
            Layers::Inputi32(y) => "tf.keras.layers.InputLayer(shape=(input_dim,))".to_string(), // needs input_dimensions to be a var called input dim
            Layers::Dense(y) => format!(
                "tf.keras.layers.Dense({}, activation='{}')",
                y.neurons,
                activation_to_string(y.activation)
            ),
            Layers::Dropout(y) => format!("tf.keras.layers.Dropout({})", y.rate),
            Layers::BatchNormalization(y) => "tf.keras.layers.BatchNormalization()".to_string(),
            Layers::Activation(y) => format!(
                "tf.keras.layers.Activation('{}')",
                activation_to_string(y.activation)
            ),
            Layers::Inputimg(y) => format!(
                "tf.keras.Input(shape=(*image_size, {})),layers.Rescaling(1./255)",
                y.channels
            ),
            Layers::Conv2D(y) => format!(
                "tf.keras.layers.Conv2D(
                filters={},
                kernel_size=({}, {}),
                activation='{}',
                input_shape=({}, {}, {})  # Only needed on the first layer
            )
            ",
                y.filters,
                y.kernel_size.0,
                y.kernel_size.1,
                activation_to_string(y.activation),
                match architecture.first().unwrap() {
                    Layers::Inputimg(x) => x.width,
                    _ => panic!(),
                },
                match architecture.first().unwrap() {
                    Layers::Inputimg(x) => x.height,
                    _ => panic!(),
                },
                match architecture.first().unwrap() {
                    Layers::Inputimg(x) => x.channels,
                    _ => panic!(),
                }
            ),
            Layers::MaxPooling2D(y) => format!(
                "tf.keras.layers.MaxPooling2D(pool_size=({}, {}), strides=({}, {}))",
                y.pool_size.0, y.pool_size.1, y.strides.0, y.strides.1
            ),
            Layers::Flatten(y) => "tf.keras.layers.Flatten()".to_string(),
            Layers::Output(y) => format!(
                "tf.keras.layers.Dense(1, activation='{}')",
                activation_to_string(y.activation)
            ),
            _ => "nope".to_string(),
        })
    }
    x
}

fn activation_to_string(activation: ActivationFunctions) -> String {
    match activation {
        ActivationFunctions::Sigmoid => "sigmoid".to_string(),
        ActivationFunctions::ReLU => "relu".to_string(),
        ActivationFunctions::Softmax => "softmax".to_string(),
        ActivationFunctions::Tanh => "tanh".to_string(),
        ActivationFunctions::Linear => "linear".to_string(),
        ActivationFunctions::None => "linear".to_string(),
    }
}
fn opdimizer_to_string(optimizer: optimizer) -> String {
    match optimizer {
        optimizer::SGD => "SGD".to_string(),
        optimizer::RMSprop => "RMSprop".to_string(),
        optimizer::Adagrad => "Adagrad".to_string(),
        optimizer::Adadelta => "Adadelta".to_string(),
        optimizer::Adam => "Adam".to_string(),
        optimizer::Adamax => "Adamax".to_string(),
        optimizer::Nadam => "Nadam".to_string(),
    }
}

use std::fs;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;

fn write_py_code_to_file(dir: &Path, py_code: &str) -> PathBuf {
    // Ensure the directory exists
    if !dir.exists() {
        fs::create_dir_all(dir);
    }

    // Create a filename, for example "script.py"
    let file_path = dir.join("script.py");

    // Open the file for writing (this creates or truncates)
    let file = fs::File::create(&file_path);

    // Write the Python code string into the file
    let _ = file.expect("REASON").write_all(py_code.as_bytes());

    // Return the full path to the new file
    file_path
}
#[component]
fn Link() -> Element {
    rsx! {
        a {
            href: "https://www.python.org/downloads/release/python-31210/",
            "python download"
        }
    }
}
