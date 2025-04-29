mod random_functions;
mod landing;
mod frame;

use frame::*;
use landing::*;
use random_functions::*;

use dioxus::prelude::*;
use dioxus_router::prelude::*;

use std::fs;
use std::path::PathBuf;

use rfd::FileDialog;
use dioxus_router::prelude::*;

static PROJECT: GlobalSignal<Option<PathBuf>> = GlobalSignal::new(|| None);

#[derive(Routable, PartialEq, Clone)]
enum Route {
    #[route("/")]
    Landing,

    // #[route("/Frame")]
    // Frame,
    #[layout(Frame)]
    #[route("/Frame/Data")]
    Data,
    #[route("/Frame/Architecture")]
    Architecture,
    #[route("/Frame/Settings")]
    Settings,
    #[route("/Frame/Run")]
    Run,
    #[route("/Frame/Results")]
    Results,
}

fn Data() -> Element {
    rsx!{"Data"}
}
fn Architecture() -> Element {
    rsx!{"Architecture"}
}
fn Settings() -> Element {
    rsx!{"Settings"}
}
fn Run() -> Element {
    rsx!{"Run"}
}
fn Results() -> Element {
    rsx!{"Results"}
}
fn main() {
    println!("{:?}", ensure_projects_folder_in_documents());
    dioxus::launch(App);
}


#[component]
fn App() -> Element {

    rsx! {
        Router::<Route> {}
    }
}

