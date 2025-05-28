mod frame;
mod landing;
mod random_functions;

mod windows;

use windows::*;

use architecture::*;
use data::*;
use results::*;
use run::*;
use settings::*;

use frame::*;
use landing::*;
use random_functions::*;

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
