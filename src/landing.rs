use crate::loading_and_folders::*;
use crate::{Route, PROJECT};
use dioxus::core_macro::rsx;
use dioxus::dioxus_core::Element;
use dioxus::hooks::use_signal;
use dioxus::prelude::*;
use dioxus::prelude::{asset, Asset};

#[component]
pub fn Landing() -> Element {
    static CSS: Asset = asset!("/assets/main.css");
    let mut show_input = use_signal(|| false); // Manage state to show input
    let mut inbox = use_signal(|| "".to_string());
    let nav = navigator(); // ← programmatic navigator

    if let Some(path) = (*PROJECT)() {
        nav.replace(Route::Data {});
    }
    rsx! {

            document::Stylesheet { href: CSS }
            div {
                h1 { "Welcome to the Course!" }
                p {
                    "In this course, you’ll learn how to build your own AI models while exploring how they work under the hood. We'll be using ",
                    strong { "Python" },
                    " and the powerful ",
                    strong { "TensorFlow + Keras" },
                    " library, which simplifies machine learning by letting you define, train, and evaluate your models with ease."
                }

                h2 { "How the Course Works" }
                p {
                    "At the top of your screen, you'll see an ",
                    strong { "Info" },
                    " button. Click it any time for step-by-step guidance and explanations of how each concept translates into real Python code. Each page also includes a ",
                    strong { "Save" },
                    " button—be sure to save your work before exiting the program so you don’t lose progress!"
                }
                p {
                    "There will also be the steps to take at the top of the page. You should do these steps in order from left to right."
                }

                h2 { "What You’ll Build" }
                ol {
                    li {
                        strong { "Data-Based Prediction Model" }
                        p {
                            "You’ll start by working with CSV (spreadsheet) files that contain rows of user behavior or event data. Your AI will learn to predict a simple true or false outcome (1 or 0)."
                        }
                        p {
                            em { "Project Example: " },
                            "Churn detection — based on customer activity, your model will predict whether a person is likely to make a purchase or leave the service."
                        }
                    }
                    li {
                        strong { "Image Classification Model" }
                        p {
                            "In this more advanced project, you'll train an AI to tell the difference between images in two folders. For example, one folder contains pictures of cats, the other dogs. The AI will learn to recognize the difference and label new images as either a cat or a dog."
                        }
                    }
                }
                p {
                    "we highly recommend starting with the Data-Based Prediction Model as it is the most straightforward and accessible project for beginners."
                }
                h2 { "What You Need to Know" }
                p {
                    "This course assumes you have a basic understanding of Python and some familiarity with AI or machine learning concepts."
                }

                p {
                    "If you're new to Python or just want a refresher, here are some free, high-quality resources:"
                }

                h3 { "Python Basics" }
                ul {
                    li {
                        a { href: "https://www.youtube.com/watch?v=kqtD5dpn9C8", target: "_blank", "Python Tutorial for Beginners – Programming with Mosh" }
                    }
                }

                h3 { "Machine Learning & Neural Networks" }
                ul {
                    li {
                        a { href: "https://www.youtube.com/watch?v=aircAruvnKk", target: "_blank", "3Blue1Brown’s Deep Learning Series (Highly Recommended!)" }
                        ul {
                            li { a { href: "https://www.youtube.com/watch?v=aircAruvnKk", target: "_blank", "What is a Neural Network?" } }
                            li { a { href: "https://www.youtube.com/watch?v=IHZwWFHWa-w", target: "_blank", "Gradient Descent" } }
                            li { a { href: "https://www.youtube.com/watch?v=Ilg3gGewQ5U", target: "_blank", "Backpropagation" } }
                            li { a { href: "https://www.youtube.com/watch?v=tIeHLnjs5U8", target: "_blank", "How to Train a Neural Network" } }
                        }
                    }
                    li {
                        a { href: "https://www.youtube.com/watch?v=SmZmBKc7Lrs", target: "_blank", "AI & Machine Learning Explained – Artem Kirsanov" }
                    }
                }

                p {
                    "These videos go deeper into the math behind AI, but ",
                    strong { "don’t worry" },
                    "—in this course, we’ll stay focused on high-level concepts and practical implementation using TensorFlow and Keras."
                }

                p {
                    "By the end of this course, you'll understand how AI models work and how to build your own—from data preprocessing to training and making predictions. Let’s get started!"
                }
                p {
                    "if you've started already load your save file at the name of the project you made in the youraiprojects folder made in documents."
                }
                p {
                    "if you havent started yet press new project and set the name of the project you want to create"
                }
            }

            h1{"eazy ai"}
            button {
                onclick: move |_| {
                    let selected = select_folder();
                    *PROJECT.write() = selected.clone();
                    println!("Project selected: {:?}", PROJECT.read());
                },
                "select project"
            }

            button {
                onclick: move |_| {
                    // Toggle the visibility of the new project input
                    show_input.set(true);  // Update the state using `set`
    ;
                },
                "new project"
            }

            // Show input field for new project when `show_input` is true
            if show_input() {
                div {
                    input {
                        placeholder: "Enter new project name",
                        oninput: move |e| {
                            // Handle input change here (update state)
                            // For example, store project name or take further action
                            inbox.set(e.value())
                        },
                    }
                    button {
                        onclick: move |_| {
                            let selected = create_folder_in_directory("/Users/keller/Documents/YourAiProjects", inbox());
                            println!("Your project selected: {:?}", selected);
                            *PROJECT.write() = selected.clone();
                        },
                        "enter"
                    },
                    button {
                        onclick: move |_| {
                            // Toggle the visibility of the new project input
                            show_input.set(false);  // Update the state using `set`

                        },
                        "❌"
                    }
                }

            Outlet::<Route> {} // <--- Render the Outlet component

            }
        }
}
