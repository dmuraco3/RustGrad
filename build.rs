extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {


    println!("cargo:rustc-link-lib=framework=Accelerate"); // This is a very important line, 
    // Links the binary to system Accelerate framework

    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-isysroot/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
        .clang_arg("-Framework Accelerate")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

}