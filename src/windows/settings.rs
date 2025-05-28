use crate::frame::DOCS;
use crate::{windows, PROJECT};
use dioxus::prelude::*;
use serde::{Deserialize, Serialize};
use windows::randomfunction2::*;

static SETTINGS: GlobalSignal<Settings> = GlobalSignal::new(|| Settings {
    optimizer: optimizer::Adam,
    epochs: 0,
    batch: 0,
});

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    pub optimizer: optimizer,
    pub epochs: i32,
    pub batch: i32,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum optimizer {
    SGD,
    RMSprop,
    Adagrad,
    Adadelta,
    Adam,
    Adamax,
    Nadam,
}

pub fn Settings() -> Element {
    use_effect(move || {
        let x = load_data_from_file3();
        match x {
            Some(t) => {
                *SETTINGS.write() = t;
                print!("loaded")
            }
            _ => print!("do nothing"),
        }
    });

    let optimizer_list = [
        "SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam",
    ];
    let optimizer_list2 = [
        optimizer::SGD,
        optimizer::RMSprop,
        optimizer::Adagrad,
        optimizer::Adadelta,
        optimizer::Adam,
        optimizer::Adamax,
        optimizer::Nadam,
    ];
    let curr = match SETTINGS.read().optimizer {
        optimizer::SGD => 0,
        optimizer::RMSprop => 1,
        optimizer::Adagrad => 2,
        optimizer::Adadelta => 3,
        optimizer::Adam => 4,
        optimizer::Adamax => 5,
        optimizer::Nadam => 6,
        _ => 0,
    };
    let curr2 = SETTINGS.read().epochs;
    let curr3 = SETTINGS.read().batch;

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
                        div {
                            h2 { "Optimizers" }
                            p { "Optimizers control how the model updates its weights to reduce error during training." }
                            p { "They’re a core part of the learning process." }
                            h3 { "Common optimizers:" }
                            ul {
                                li {
                                    strong { "SGD (Stochastic Gradient Descent): " }
                                    "Simple, reliable, but can be slow without tuning. Good for basic tasks."
                                }
                                li {
                                    strong { "RMSprop: " }
                                    "Adapts learning using a moving average of squared gradients. Great for RNNs or noisy data."
                                }
                                li {
                                    strong { "Adagrad: " }
                                    "Adjusts learning rates individually for each parameter. Great for sparse data (e.g., text), but learning slows over time."
                                }
                                li {
                                    strong { "Adadelta: " }
                                    "Builds on Adagrad to avoid the shrinking learning rate issue. Good for longer training runs."
                                }
                                li {
                                    strong { "Adam: " }
                                    "Combines momentum and adaptive learning rate. Fast, effective, and widely used for most problems."
                                }
                                li {
                                    strong { "Adamax: " }
                                    "A more stable variant of Adam using the infinity norm. Helpful with large or unbounded gradients."
                                }
                                li {
                                    strong { "Nadam: " }
                                    "Adam + Nesterov momentum = potentially faster convergence in some cases."
                                }
                            }
                            p { strong { "Examples: " } "For both our examples we use the Adam optimizer." }

                            h2 { "Epochs" }
                            p { "An epoch is one full pass through the training dataset." }
                            p { "Training for multiple epochs lets the model gradually improve its predictions." }
                            ul {
                                li { "Too few = underfitting." }
                                li { "Too many = overfitting." }
                            }
                            p { "Think of epochs as how many times your model practices." }
                            p { "If you're trying it for the first time, test with a lower epoch count before training the final version." }

                            h2 { "Batch Size" }
                            p { "Batch size is how many samples the model sees before it updates the weights." }
                            ul {
                                li { strong { "Smaller batch: " } "Slower training but more accurate updates." }
                                li { strong { "Larger batch: " } "Faster training, but may require more memory and might generalize less well." }
                            }
                            p { "Common values: 32, 64, 128" }
                            p { "Choose based on your system’s memory and model behavior. If you choose too high of a batch size, it might not work well on less powerful machines." }
                            h2{"how this translates to the code."}
                            div {
                                h3 { "Model Compilation" }
                                p { "Before training, the model must be compiled. This step sets up how the model will learn and be evaluated." }

                                p{"model.compile(optimizer='adam',
                                            loss='binary_crossentropy',
                                            metrics=['accuracy'])"}
                                ul {
                                    li {
                                        strong { "optimizer='adam': " }
                                        "Adam is a popular, adaptive optimizer that helps the model learn efficiently. It combines the benefits of momentum and adaptive learning rates."
                                    }
                                    li {
                                        strong { "loss='binary_crossentropy': " }
                                        "Since this is a binary classification problem (two possible outputs), we use 'binary_crossentropy' to measure how well the model’s predictions match the true labels."
                                    }
                                    li {
                                        strong { "metrics=['accuracy']: " }
                                        "We track accuracy to see how often the model's predictions are correct. It's the most common metric for classification tasks."
                                    }
                                }

                                p {
                                    "Together, these settings tell the model: “Use the Adam optimizer to update weights, learn using binary cross-entropy loss, and track accuracy while training.”"
                                }

                            }
                            div {
                                h3 { "Training the Model" }
                                p { "This line starts training the model with model.fit()" }
                                p{"model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)"}
                                ul {
                                    li {
                                        strong { "X: " }
                                        "The input data (features) that the model learns from."
                                    }
                                    li {
                                        strong { "y: " }
                                        "The target labels (what the model is trying to predict)."
                                    }
                                    li {
                                        strong { "epochs=10: " }
                                        "The model will go through the entire dataset 10 times to improve its accuracy each round."
                                    }
                                    li {
                                        strong { "batch_size=32: " }
                                        "The model updates its weights every 32 samples instead of after every single one. This speeds up training and smooths learning."
                                    }
                                    li {
                                        strong { "validation_split=0.2: " }
                                        "20% of the data will be held out from training and used to validate how well the model is performing after each epoch."
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        div{"chose optimizer"}
        div{
            for (x, opt) in optimizer_list2.iter().cloned().enumerate() {
                button{
                    style: format_args!(
                            "background-color: {};",
                            if curr == x { "green" } else { "gray" }
                    ),
                    onclick: move |_| {
                        SETTINGS.write().optimizer = opt.clone()
                    },
                    "{optimizer_list.get(x).unwrap().clone()}"
                }
            }
        }
        div {
                style: "color: #ffffff;",
                label { "how many epochs: " }
                input {
                    r#type: "number",
                    value: "{curr2}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            SETTINGS.write().epochs = num;
                            println!("Selected number: {:?}", *SETTINGS.read());
                        }
                    },
                }
                p { "Current epochs: {curr2}" }
            }
        div {
                style: "color: #ffffff;",
                label { "batch size: " }
                input {
                    r#type: "number",
                    value: "{curr3}",
                    oninput: move |evt| {
                        if let Ok(num) = evt.parsed::<i32>() {
                            SETTINGS.write().batch = num;
                            println!("Selected number: {:?}", *SETTINGS.read());
                        }
                    },
                }
                p { "Current batch size: {curr3}" }
            }
        div {
            button {
                onclick: move |_| {
                    let data = SETTINGS.read().clone();
                    if let Some(base_path) = &*PROJECT.read() {
                        let mut full_path = base_path.clone();
                        full_path.push("settings.json");
                        if let Some(path_str) = full_path.to_str() {
                            save_data_to_file3(data,path_str);
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
