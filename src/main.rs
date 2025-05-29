mod frame;
mod landing;
mod loading_and_folders;

mod windows;

use windows::*;

use architecture::*;
use data::*;
use results::*;
use run::*;
use settings::*;

use frame::*;
use landing::*;
use loading_and_folders::*;

use dioxus::prelude::*;

use std::path::PathBuf;

static PROJECT: GlobalSignal<Option<PathBuf>> = GlobalSignal::new(|| None);

#[derive(Routable, PartialEq, Clone, Debug)]
enum Route {
    #[route("/")]
    Landing,
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
