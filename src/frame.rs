use crate::Route;
use dioxus::core_macro::rsx;
use dioxus::dioxus_core::Element;
use dioxus::hooks::use_signal;
use dioxus::prelude::*;
use dioxus::prelude::{asset, Asset};

static CSS: Asset = asset!("/assets/frame.css");

pub static DOCS: GlobalSignal<bool> = Global::new(|| false);

#[component]
pub fn Frame() -> Element {
    // cant use signals in this because of stupid signal rules
    let mut open = use_signal(|| 0);
    let nav = navigator(); // ← programmatic navigator
    println!("{:?}", open);
    match open() {
        0 => nav.push(Route::Data {}),
        1 => nav.push(Route::Architecture {}),
        2 => nav.push(Route::Settings {}),
        3 => nav.push(Route::Run {}),
        4 => nav.push(Route::Results {}),
        _ => panic!("how"),
    };
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
        button {
            onclick: move |_| *DOCS.write() ^= true,
            style: "width: 100%; display: block;",
            "info"
        }
        Outlet::<Route> {} // <--- Render the Outlet component

    }
}
