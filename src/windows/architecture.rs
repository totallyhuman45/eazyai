use crate::frame::DOCS;
use crate::{windows, PROJECT};
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use std::vec;
use std::vec::Vec;
use windows::loading_and_folders2::*;

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone, Default)]
pub enum ActivationFunctions {
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
    Linear,
    #[default]
    None,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum Layers {
    Inputi32(Inputi32),
    Dense(Dense),
    Dropout(Dropout),
    BatchNormalization(BatchNormalization),
    Activation(Activation),
    Inputimg(Inputimg),
    Conv2D(Conv2D),
    MaxPooling2D(MaxPooling2D),
    Flatten(Flatten),
    Output(Output),
    None,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct Inputi32 {}
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct Dense {
    pub neurons: i32,
    pub activation: ActivationFunctions,
}
#[derive(Debug, Default, Serialize, Deserialize, Clone, PartialEq)]
pub struct Dropout {
    pub rate: f64,
}
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct BatchNormalization {}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct Activation {
    pub activation: ActivationFunctions,
}
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct Inputimg {
    pub height: i32,
    pub width: i32,
    pub channels: i32,
}
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct Conv2D {
    pub filters: i32,
    pub kernel_size: (i32, i32),
    pub activation: ActivationFunctions,
    pub input_shape: (i32, i32, i32),
}
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct MaxPooling2D {
    pub pool_size: (i32, i32),
    pub strides: (i32, i32),
}
#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct Flatten {}

#[derive(Debug, Serialize, Deserialize, Clone, Default, PartialEq)]
pub struct Output {
    //units is usaly 1 Number of output classes (for binary classification, it's 1).
    pub activation: ActivationFunctions,
}
static ARCHITECTURE: GlobalSignal<Vec<Layers>> = GlobalSignal::new(|| {
    vec![
        Layers::Inputi32(Inputi32 {}),
        Layers::Output(Output {
            activation: ActivationFunctions::None,
        }),
    ]
});

pub fn Architecture() -> Element {
    use_effect(move || {
        let x = load_data_from_file2();
        match x {
            Some(t) => {
                *ARCHITECTURE.write() = t;
                print!("loaded")
            }
            _ => print!("do nothing"),
        }
    });

    let amount_of_layers = use_signal(|| ARCHITECTURE.read().clone().len() as i32); // Default to 0
    let binding = ARCHITECTURE.read().clone();
    let iterate = binding.iter().enumerate().collect::<Vec<_>>();
    println!("{:?}", ARCHITECTURE.read().clone());
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
                                h1 { "Neural Network Layer Types and Activations Explained" }

                                h2 { "Layer Types" }
                                p { "Common layer types used in neural networks for image tasks:" }
                                ul {
                                    li { "Inputi32" }
                                    li { "Dense" }
                                    li { "Dropout" }
                                    li { "BatchNormalization" }
                                    li { "Activation" }
                                    li { "Inputimg" }
                                    li { "Conv2D" }
                                    li { "MaxPooling2D" }
                                    li { "Flatten" }
                                    li { "Output" }
                                }

                                h2 { "Activation Functions" }

                                div {
                                    h3 { "ReLU (Rectified Linear Unit)" }
                                    p { "What it does: Converts negative values to 0; keeps positives unchanged." }
                                    p { "Why useful: Fast, simple, adds non-linearity." }
                                    p { "When to use: Hidden layers in CNNs, Dense layers." }
                                }

                                div {
                                    h3 { "Sigmoid" }
                                    p { "What it does: Squashes values between 0 and 1 (like probabilities)." }
                                    p { "Why useful: Good for binary classification." }
                                    p { "When to use: Final layer in binary classification." }
                                }

                                div {
                                    h3 { "Softmax" }
                                    p { "What it does: Converts raw scores into probabilities that sum to 1." }
                                    p { "Why useful: Highlights the most likely class." }
                                    p { "When to use: Final layer in multiclass classification." }
                                }

                                div {
                                    h3 { "Tanh (Hyperbolic Tangent)" }
                                    p { "What it does: Squashes values between -1 and 1." }
                                    p { "Why useful: Useful for outputs that swing both positive and negative." }
                                    p { "When to use: Hidden layers when symmetric outputs are desired." }
                                }

                                div {
                                    h3 { "Linear" }
                                    p { "What it does: Passes input unchanged." }
                                    p { "Why useful: Used in regression tasks for raw numeric output." }
                                    p { "When to use: Output layer in regression problems." }
                                }

                                div {
                                    h3 { "None (No activation)" }
                                    p { "What it does: Same as linear, no activation applied." }
                                    p { "When to use: If applying activation later manually or for regression." }
                                }

                                h2 { "Quick Activation Guide" }
                                table {
                                    thead {
                                        tr {
                                            th { "Activation" }
                                            th { "Use Case" }
                                        }
                                    }
                                    tbody {
                                        tr { td { "ReLU" } td { "Hidden layers (fast & effective)" } }
                                        tr { td { "Sigmoid" } td { "Binary classification output" } }
                                        tr { td { "Softmax" } td { "Multiclass classification output" } }
                                        tr { td { "Tanh" } td { "Hidden layers (symmetric output)" } }
                                        tr { td { "Linear" } td { "Regression output" } }
                                        tr { td { "None" } td { "No activation (like linear)" } }
                                    }
                                }

                                h2 { "Layer Descriptions" }

                                div {
                                    h3 { "Input / Inputimg" }
                                    p { "Defines the input data shape." }
                                    pre {
                                        code {
                                            r#"
                    tf.keras.Input(shape=(180, 180, 3))
                    # Expects images 180×180 pixels with 3 RGB channels.
                                            "#
                                        }
                                    }
                                }

                                div {
                                    h3 { "Dense" }
                                    p { "Fully connected layer." }
                                    p { "Parameters:" }
                                    ul {
                                        li { "units: number of neurons, e.g., 128" }
                                        li { "activation: activation function, e.g., 'relu'" }
                                    }
                                    pre {
                                        code { r#"Dense(128, activation='relu')"# }
                                    }
                                }

                                div {
                                    h3 { "Dropout" }
                                    p { "Randomly disables a fraction of neurons during training to prevent overfitting." }
                                    p { "Parameter:" }
                                    ul {
                                        li { "rate: fraction of neurons to drop, e.g., 0.5" }
                                    }
                                    pre {
                                        code { r#"Dropout(0.5)"# }
                                    }
                                }

                                div {
                                    h3 { "BatchNormalization" }
                                    p { "Normalizes layer outputs to mean ~0 and variance ~1." }
                                    p { "Helps training stability and speed." }
                                    p { "Optional params: axis, momentum" }
                                    pre {
                                        code { r#"BatchNormalization()"# }
                                    }
                                }

                                div {
                                    h3 { "Activation" }
                                    p { "Applies an activation function as a separate layer." }
                                    p { "Parameter: Activation function name, e.g., 'relu'" }
                                    pre {
                                        code { r#"Activation('relu')"# }
                                    }
                                }

                                div {
                                    h3 { "Conv2D" }
                                    p { "Applies convolution filters to extract image features." }
                                    p { "Parameters:" }
                                    ul {
                                        li { "filters: number of filters, e.g., 32" }
                                        li { "kernel_size: filter size, e.g., (3, 3)" }
                                        li { "activation: activation function after convolution" }
                                    }
                                    pre {
                                        code { r#"Conv2D(filters=32, kernel_size=(3, 3), activation='relu')"# }
                                    }
                                }

                                div {
                                    h3 { "MaxPooling2D" }
                                    p { "Downsamples feature maps by taking max in each window." }
                                    p { "Parameters:" }
                                    ul {
                                        li { "pool_size: e.g., (2, 2)" }
                                        li { "strides: step size, defaults to pool_size" }
                                    }
                                    pre {
                                        code { r#"MaxPooling2D(pool_size=(2, 2), strides=(2, 2))"# }
                                    }
                                    p { "(Reduces width and height by half)" }
                                }

                                div {
                                    h3 { "Flatten" }
                                    p { "Converts multi-dimensional input into 1D vector for Dense layers." }
                                    p { "No parameters needed." }
                                    pre {
                                        code { r#"Flatten()"# }
                                    }
                                }

                                div {
                                    h3 { "Output" }
                                    p { "Typically a final Dense layer." }
                                    p { "Examples based on task:" }

                                    p { "Binary classification:" }
                                    pre {
                                        code { r#"Dense(1, activation='sigmoid')"# }
                                    }

                                    p { "Multiclass classification:" }
                                    pre {
                                        code { r#"Dense(num_classes, activation='softmax')"# }
                                    }

                                    p { "Regression:" }
                                    pre {
                                        code { r#"Dense(1, activation='linear')  # or activation=None"# }
                                    }
                                }
                                div {
                                    h2{"lets look at our example of our churn model"}
                                    ul {
                                        li { strong { "InputLayer (input_shape=(input_dim,))" }, " — Accepts input we would use inputi32 for this model." }
                                        li { strong { "Dense(32, activation='relu')" }, " — First hidden layer with 32 neurons to learn initial feature representations." }
                                        li { strong { "Dense(16, activation='relu')" }, " — Second hidden layer with 16 neurons to refine and compress features." }
                                        li { strong { "Dense(1, activation='sigmoid')" }, " — Output layer with 1 neuron producing a probability for binary classification." }
                                    }
                                }
                                div {
                                    h2{"lets look at our example of our image classification model"}
                                    ul {
                                            li { strong { "Input (shape=(*image_size, 3))" }, " — Accepts images with dimensions specified by image_size and 3 color channels (RGB). for this model wou would use the inputimg" }
                                            li { strong { "Rescaling (1./255)" }, " — Normalizes pixel values from 0-255 to 0-1 for better model training." }
                                            li { strong { "Conv2D (32, 3, activation='relu')" }, " — First convolutional layer with 32 filters to extract basic features like edges." }
                                            li { strong { "MaxPooling2D (pool_size=(2, 2), strides=(2, 2))" }, " — Reduces spatial dimensions to make feature maps smaller and computation efficient." }
                                            li { strong { "Conv2D (64, 3, activation='relu')" }, " — Second convolutional layer with 64 filters to capture more complex patterns." }
                                            li { strong { "MaxPooling2D(pool_size=(2, 2), strides=(2, 2))" }, " — Further spatial downsampling to reduce feature map size." }
                                            li { strong { "Conv2D (128, 3, activation='relu')" }, " — Third convolutional layer with 128 filters to detect high-level features." }
                                            li { strong { "MaxPooling2D (pool_size=(2, 2), strides=(2, 2))" }, " — Final pooling to reduce dimensionality before flattening." }
                                            li { strong { "Flatten ()" }, " — Converts 3D feature maps into 1D vector for Dense layers." }
                                            li { strong { "Dense(128, activation='relu')" }, " — Fully connected layer to learn complex feature combinations." }
                                            li { strong { "Dense(1, activation='sigmoid')" }, " — Output layer producing probability for binary classification." }
                                        }
                                }

                            }
                }
            }
        }
        div {
            style: "display: flex; font-family: sans-serif; gap: 20px; padding: 20px; box-sizing: border-box;",
            div {
                style: "
                    display: flex;
                    flex-direction: column;
                    flex: 1;
                    background-color: #d0d0d0;
                    padding: 20px;
                    text-align: left;
                    border-radius: 16px;
                ",
                Amount{amount_of_layers}
                div {
                    style: "display: flex; gap: 1rem; padding: 1rem; flex-wrap: wrap;",
                    for (t,x) in iterate {
                        // Add a key for each Layeredit to help Dioxus efficiently update the DOM
                        Layeredit {layer:x.clone() ,index: t as i32 }

                    }
                }
                div{
                    button{

                        onclick: move |_| {
                            let data = ARCHITECTURE.read().clone();
                            if let Some(base_path) = &*PROJECT.read() {
                                let mut full_path = base_path.clone();
                                full_path.push("architecture.json");

                                if let Some(path_str) = full_path.to_str() {
                                    save_data_to_file2(data,path_str);
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
    }
}
#[component]
fn Amount(mut amount_of_layers: Signal<i32>) -> Element {
    rsx! {
        div {
            style: "color: #000000;",
            label { "Select a number of layers: " }
            input {
                r#type: "number",
                value: "{amount_of_layers}",
                oninput: move |evt| {
                    if let Ok(num) = evt.parsed::<i32>() {
                        if num < amount_of_layers(){
                            let x = remove_to_end(ARCHITECTURE.read().clone());
                            *ARCHITECTURE.write() = x;                        }
                        if num > amount_of_layers(){
                            let x = add_to_end(ARCHITECTURE.read().clone());
                            *ARCHITECTURE.write() = x;
                        }
                        amount_of_layers.set(num);
                        println!("Selected layers: {:?}", *ARCHITECTURE.read());
                    }
                },
                min: "2",
                max: "15"
            }
            p { "Current value: {amount_of_layers}" }
        }
    }
}
fn add_to_end(vec: Vec<Layers>) -> Vec<Layers> {
    let mut vece = vec.clone();
    let x = vec.last().unwrap();
    vece.pop();
    vece.push(Layers::None);
    vece.push(x.clone());
    vece
}
fn remove_to_end(vec: Vec<Layers>) -> Vec<Layers> {
    let mut vece = vec.clone();
    let x = vec.last().unwrap();
    vece.pop();
    vece.pop();
    vece.push(x.clone());
    vece
}

#[component]
fn TypeComponent(ind: i32) -> Element {
    // UI logic here
    let arch = ARCHITECTURE.read();
    let curr = arch.get(ind as usize).unwrap().clone();
    let options = [
        "Inputi32",
        "Dense",
        "Dropout",
        "BatchNormalization",
        "Activation",
        "Inputimg",
        "Conv2D",
        "MaxPooling2D",
        "Flatten",
        "Output",
        "None",
    ];
    let options2 = vec![
        Layers::Inputi32(Inputi32::default()),
        Layers::Dense(Dense::default()),
        Layers::Dropout(Dropout::default()),
        Layers::BatchNormalization(BatchNormalization::default()),
        Layers::Activation(Activation::default()),
        Layers::Inputimg(Inputimg::default()),
        Layers::Conv2D(Conv2D::default()),
        Layers::MaxPooling2D(MaxPooling2D::default()),
        Layers::Flatten(Flatten::default()),
        Layers::Output(Output::default()),
        Layers::None,
    ];
    let curr_num = match curr.clone() {
        Layers::Inputi32(x) => 0,
        Layers::Dense(x) => 1,
        Layers::Dropout(x) => 2,
        Layers::BatchNormalization(x) => 3,
        Layers::Activation(x) => 4,
        Layers::Inputimg(x) => 5,
        Layers::Conv2D(x) => 6,
        Layers::MaxPooling2D(x) => 7,
        Layers::Flatten(x) => 8,
        Layers::Output(x) => 9,
        Layers::None => 10,
        _ => panic!(),
    };

    let curr1 = curr.clone();
    let curr2 = curr.clone();
    let options21 = options2.clone();
    let options22 = options2.clone();

    let curr_read = options.get(curr_num).unwrap();

    rsx! {

        "{curr_read}"
        div{
            button{
                onclick: move |_| {
                    let mut currr_num = match curr1.clone() {
                        Layers::Inputi32(x) => 0,
                        Layers::Dense(x)=> 1,
                        Layers::Dropout(x)=> 2,
                        Layers::BatchNormalization(x)=> 3,
                        Layers::Activation(x)=> 4,
                        Layers::Inputimg(x)=> 5,
                        Layers::Conv2D(x)=> 6,
                        Layers::MaxPooling2D(x)=> 7,
                        Layers::Flatten(x)=> 8,
                        Layers::Output(x)=> 9,
                        Layers::None=> 10,
                        _ => panic!()
                    };
                    if currr_num == 10{
                        currr_num = 0
                    }else{
                        currr_num += 1
                    }
                    let x = options21.get(currr_num).unwrap().clone(); // Ensure you have an owned value
                    println!("{:?}", currr_num.clone());
                    *ARCHITECTURE.write().get_mut(ind as usize).unwrap() = x;


                },
                "▲"
            }
            button{
                onclick: move |_| {

                    let mut currr_num = match curr2.clone() {
                        Layers::Inputi32(x) => 0,
                        Layers::Dense(x)=> 1,
                        Layers::Dropout(x)=> 2,
                        Layers::BatchNormalization(x)=> 3,
                        Layers::Activation(x)=> 4,
                        Layers::Inputimg(x)=> 5,
                        Layers::Conv2D(x)=> 6,
                        Layers::MaxPooling2D(x)=> 7,
                        Layers::Flatten(x)=> 8,
                        Layers::Output(x)=> 9,
                        Layers::None=> 10,
                        _ => panic!()
                    };
                    if currr_num == 0{
                        currr_num = 10
                    }else{
                        currr_num -= 1
                    }
                    let x = options22.get(currr_num).unwrap().clone(); // Ensure you have an owned value
                    println!("{:?}", currr_num.clone());
                    *ARCHITECTURE.write().get_mut(ind as usize).unwrap() = x;


                },
                "▼"
            }
        }
    }
}

#[component]
fn Layeredit(layer: Layers, index: i32) -> Element {
    println!("{:?}", ARCHITECTURE.write().get_mut(index as usize));
    rsx! {
        div{
            style: "border-radius: 12px; background-color: #f0f0f0; padding: 1rem; color: #000000; max-width: 200px; word-wrap: break-word; min-height: 20vh;",

            TypeComponent{ind: index}

            match layer {
                Layers::Output(x) => {
                    Output::acctivation_dropdown(index)
                },
                Layers::Flatten(x) => rsx! { "Flatten" },
                Layers::Inputi32(x) => rsx! { "Inputi32" },
                Layers::BatchNormalization(x) => rsx! { "BatchNormalization" },
                Layers::Dense(x) => rsx!{
                    {Dense::acctivation_dropdown(index)}
                    {Dense::i32picker(index)}
                },
                Layers::Inputimg(x) => Inputimg::i32picker(index),
                Layers::Dropout(x) => Dropout::i32picker(index),
                Layers::Activation(x) => Activation::acctivation_dropdown(index),
                Layers::Conv2D(x) => rsx!{{Conv2D::i32picker(index)}{Conv2D::acctivation_dropdown(index)}},
                Layers::MaxPooling2D(x) => MaxPooling2D::i32picker(index),
                Layers::None => rsx! { "empty" },
                _ => rsx! { "undefined" }
            }

        }
    }
}

fn Output_node(layer: Output, indexs: i32) -> Element {
    rsx! {"output"}
}

trait dropdown {
    fn acctivation_dropdown(index: i32) -> Element;
}

impl dropdown for Output {
    fn acctivation_dropdown(index: i32) -> Element {
        let options = ["ReLU", "Sigmoid", "Softmax", "Tanh", "Linear", "None"];
        let options2 = vec![
            ActivationFunctions::ReLU,
            ActivationFunctions::Sigmoid,
            ActivationFunctions::Softmax,
            ActivationFunctions::Tanh,
            ActivationFunctions::Linear,
            ActivationFunctions::None,
        ];
        let curr = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Output(x) => x.clone().activation,
            _ => panic!(),
        };
        rsx! {
            "activation function"

            for (r, t) in options.iter().enumerate() {
                {
                    let value = options2.clone();
                    rsx! {
                    button {
                        style: format_args!(
                            "background-color: {};",
                            if curr == *value.get(r).unwrap() { "green" } else { "gray" }
                        ),
                        onclick: move |_| {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Output(x) => x.activation = value[r].clone(),
                                _ => panic!()
                            };
                        },
                        "{t}"
                    }
                }
                }
            }
        }
    }
}

impl dropdown for Dense {
    fn acctivation_dropdown(index: i32) -> Element {
        let options = ["ReLU", "Sigmoid", "Softmax", "Tanh", "Linear", "None"];
        let options2 = vec![
            ActivationFunctions::ReLU,
            ActivationFunctions::Sigmoid,
            ActivationFunctions::Softmax,
            ActivationFunctions::Tanh,
            ActivationFunctions::Linear,
            ActivationFunctions::None,
        ];
        let curr = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Dense(x) => x.clone().activation,
            _ => panic!(),
        };
        rsx! {
            "activation function"
            for (r, t) in options.iter().enumerate() {
                {
                    let value = options2.clone();
                    rsx! {
                    button {
                        style: format_args!(
                            "background-color: {};",
                            if curr == *value.get(r).unwrap() { "green" } else { "gray" }
                        ),
                        onclick: move |_| {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Dense(x) => x.activation = value[r].clone(),
                                _ => panic!()
                            };
                        },
                        "{t}"
                    }
                }
                }
            }
        }
    }
}

impl dropdown for Activation {
    fn acctivation_dropdown(index: i32) -> Element {
        let options = ["ReLU", "Sigmoid", "Softmax", "Tanh", "Linear", "None"];
        let options2 = vec![
            ActivationFunctions::ReLU,
            ActivationFunctions::Sigmoid,
            ActivationFunctions::Softmax,
            ActivationFunctions::Tanh,
            ActivationFunctions::Linear,
            ActivationFunctions::None,
        ];
        let curr = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Activation(x) => x.clone().activation,
            _ => panic!(),
        };
        rsx! {
            "activation function"
            for (r, t) in options.iter().enumerate() {
                {
                    let value = options2.clone();
                    rsx! {
                    button {
                        style: format_args!(
                            "background-color: {};",
                            if curr == *value.get(r).unwrap() { "green" } else { "gray" }
                        ),
                        onclick: move |_| {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Activation(x) => x.activation = value[r].clone(),
                                _ => panic!()
                            };
                        },
                        "{t}"
                    }
                }
                }
            }
        }
    }
}

impl dropdown for Conv2D {
    fn acctivation_dropdown(index: i32) -> Element {
        let options = ["ReLU", "Sigmoid", "Softmax", "Tanh", "Linear", "None"];
        let options2 = vec![
            ActivationFunctions::ReLU,
            ActivationFunctions::Sigmoid,
            ActivationFunctions::Softmax,
            ActivationFunctions::Tanh,
            ActivationFunctions::Linear,
            ActivationFunctions::None,
        ];
        let curr = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Conv2D(x) => x.clone().activation,
            _ => panic!(),
        };
        rsx! {
            "activation function"
            for (r, t) in options.iter().enumerate() {
                {
                    let value = options2.clone();
                    rsx! {
                    button {
                        style: format_args!(
                            "background-color: {};",
                            if curr == *value.get(r).unwrap() { "green" } else { "gray" }
                        ),
                        onclick: move |_| {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Conv2D(x) => x.activation = value[r].clone(),
                                _ => panic!()
                            };
                        },
                        "{t}"
                    }
                }
                }
            }
        }
    }
}

trait i32s {
    fn i32picker(index: i32) -> Element;
}

impl i32s for Dense {
    fn i32picker(index: i32) -> Element {
        let curr = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Dense(x) => x.neurons,
            _ => panic!(),
        };
        rsx! {
            div {
                style: "color: #000000;",
                label { "Select a number: " }
                input {
                    r#type: "number",
                    value: "{curr}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Dense(x) => {x.neurons = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current neurons: {curr}" }
            }
        }
    }
}

impl i32s for Inputimg {
    fn i32picker(index: i32) -> Element {
        let curr1 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Inputimg(x) => x.height,
            _ => panic!(),
        };
        let curr2 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Inputimg(x) => x.width,
            _ => panic!(),
        };
        let curr3 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Inputimg(x) => x.channels,
            _ => panic!(),
        };
        rsx! {
            div {
                style: "color: #000000;",
                label { "Select a number: " }
                input {
                    r#type: "number",
                    value: "{curr1}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Inputimg(x) => {x.height = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current height: {curr1}" }
            }
            div {
                style: "color: #000000;",
                label { "Select a number: " }
                input {
                    r#type: "number",
                    value: "{curr2}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Inputimg(x) => {x.width = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current width: {curr2}" }
            }
            div {
                style: "color: #000000;",
                label { "Select a number: " }
                input {
                    r#type: "number",
                    value: "{curr3}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Inputimg(x) => {x.channels = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current channels: {curr3}" }
            }
        }
    }
}

impl i32s for Dropout {
    fn i32picker(index: i32) -> Element {
        let curr = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Dropout(x) => x.rate,
            _ => panic!(),
        };
        rsx! {
            div {
                style: "color: #000000;",
                label { "Select a number: " }
                input {
                    r#type: "number",
                    value: "{curr}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<f64>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Dropout(x) => {x.rate = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current rate: {curr}" }
            }
        }
    }
}

impl i32s for Conv2D {
    fn i32picker(index: i32) -> Element {
        let curr1 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Conv2D(x) => x.filters,
            _ => panic!(),
        };
        let curr2 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Conv2D(x) => x.kernel_size.0,
            _ => panic!(),
        };
        let curr3 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Conv2D(x) => x.kernel_size.1,
            _ => panic!(),
        };
        let curr4 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Conv2D(x) => x.input_shape.0,
            _ => panic!(),
        };
        let curr5 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Conv2D(x) => x.input_shape.1,
            _ => panic!(),
        };
        let curr6 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::Conv2D(x) => x.input_shape.2,
            _ => panic!(),
        };
        rsx! {
            div {
                style: "color: #000000;",
                label { "Select a number: " }
                input {
                    r#type: "number",
                    value: "{curr1}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Conv2D(x) => {x.filters = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current filters: {curr1}" }
            }
            div {
                style: "color: #000000;",
                label { "Select a numbers: " }
                input {
                    r#type: "number",
                    value: "{curr2}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Conv2D(x) => {x.kernel_size.0 = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                input {
                    r#type: "number",
                    value: "{curr3}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::Conv2D(x) => {x.kernel_size.1 = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current kernel size: ({curr2}, {curr3})" }
            }

        }
    }
}

impl i32s for MaxPooling2D {
    fn i32picker(index: i32) -> Element {
        let curr1 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::MaxPooling2D(x) => x.pool_size.0,
            _ => panic!(),
        };
        let curr2 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::MaxPooling2D(x) => x.pool_size.1,
            _ => panic!(),
        };
        let curr3 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::MaxPooling2D(x) => x.strides.0,
            _ => panic!(),
        };
        let curr4 = match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
            Layers::MaxPooling2D(x) => x.strides.1,
            _ => panic!(),
        };
        rsx! {
            div {
                style: "color: #000000;",
                label { "Select a numbers: " }
                input {
                    r#type: "number",
                    value: "{curr1}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::MaxPooling2D(x) => {x.pool_size.0 = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                input {
                    r#type: "number",
                    value: "{curr2}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::MaxPooling2D(x) => {x.pool_size.1 = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current kernel size: ({curr1}, {curr2})" }
            }
            div {
                style: "color: #000000;",
                label { "Select a numbers: " }
                input {
                    r#type: "number",
                    value: "{curr3}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::MaxPooling2D(x) => {x.strides.0 = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                input {
                    r#type: "number",
                    value: "{curr4}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            match ARCHITECTURE.write().get_mut(index as usize).unwrap() {
                                Layers::MaxPooling2D(x) => {x.strides.1 = num;},
                                _ => panic!()
                            }
                            println!("Selected number: {:?}", *ARCHITECTURE.read());
                        }
                    },
                }
                p { "Current strides: ({curr3}, {curr4})" }
            }
        }
    }
}
