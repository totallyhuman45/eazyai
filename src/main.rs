mod random_functions;
mod landing;
mod frame;

mod windows;

use windows::*;

use architecture::*;
use data::*;
use settings::*;
use run::*;
use results::*;

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

