use dioxus::core_macro::rsx;
use dioxus::dioxus_core::Element;
use dioxus::hooks::use_signal;
use dioxus::prelude::{asset, Asset};
use std::path::PathBuf;
use dioxus::prelude::*;
use crate::{Route, PROJECT};
use crate::landing::*;
use dioxus::prelude::*;
use crate::Route::*;

static CSS: Asset = asset!("/assets/frame.css");



#[component]
pub fn Frame() -> Element {
    // cant use signals in this because of stupid signal rules
    let mut open = use_signal(|| 0);
    println!("{:?}",open);
    rsx! {

        document::Stylesheet { href: CSS }

        div {
            class: "button-bar",
            button { onclick: move |_| open.set(0), "Data" }
            button { onclick: move |_| open.set(1), "Architecture" }
            button { onclick: move |_| open.set(2), "Settings" }
            button { onclick: move |_| open.set(3), "Run" }
            button { onclick: move |_| open.set(4), "Results" }
        }
        div {
            class: "content",

        }
        Outlet::<Route> {} // <--- Render the Outlet component

    }
}