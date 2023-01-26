
use std::time::Instant;

use RustGrad::Tensor::{STensor1D, STensor2D, STensor3D};


fn main() {

    let x1 = STensor2D::<f32, 100, 100>::rand(); 
   
    let x2 = STensor2D::<f32, 100, 100>::rand(); 

    let start = Instant::now();
    let result = x1 * x2;   
    println!("{:?}", start.elapsed());
    

}

