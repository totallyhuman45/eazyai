use dioxus::core_macro::rsx;
use dioxus::dioxus_core::Element;
use dioxus::hooks::use_signal;
use dioxus::prelude::{asset, Asset};
use dioxus::prelude::*;
use crate::random_functions::*;
use crate::{Route, PROJECT};

#[component]
pub fn Landing() -> Element {
    static CSS: Asset = asset!("/assets/main.css");
    let mut show_input = use_signal(|| false); // Manage state to show input
    let mut inbox = use_signal(|| "".to_string());
    let nav = navigator();              // ← programmatic navigator

    if let Some(path) = (*PROJECT)() {
        nav.replace(Route::Data {});
    }
    rsx! {

        document::Stylesheet { href: CSS }

        h1{"eazy ai"}
        button {
            onclick: move |_| {
                let selected = select_folder("/Users/keller/Documents/YourAppProjects");
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

