use core::{panic, fmt};
use std::{ops::{Add, AddAssign, Div, SubAssign, Sub, MulAssign, DivAssign, Mul, Index, IndexMut}, iter::Sum, fmt::{Debug, Display, Formatter}, process::Output, marker::PhantomData};

use rand::{Rng, rngs::ThreadRng, distributions::{Standard, self}, prelude::Distribution};

pub mod macros;

pub mod gpu;

pub mod ops;

pub struct Const<const R: usize>;

pub trait TensorTrait<T>: Debug + Copy + Sum 
    + Add<Output = T>
    + Sub<Output = T>
    + Mul<Output = T>
    + Div 
    + AddAssign 
    + SubAssign 
    + MulAssign 
    + DivAssign 
    + 'static     
{
    fn zero() -> Self;
    fn one() -> Self;
}

impl TensorTrait<f32> for f32 {
    fn zero() -> f32 {
        0.0
    }
    fn one() -> f32 {
        1.0
    }
}

impl TensorTrait<f64> for f64 {
    fn zero() -> f64 {
        0.0
    }
    fn one() -> f64 {
        1.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct STensor0D<T: TensorTrait<T>> (
    T
);

impl <T: TensorTrait<T>> STensor0D<T> {

    pub fn new(data: T) -> Self {
        STensor0D (data)
    }

    pub fn rand() -> Self 
    where 
        Standard: Distribution<T>
    {
        let mut rng = rand::thread_rng();
        STensor0D (rng.gen::<T>())
    }
}

impl <T: TensorTrait<T>> Display for STensor0D<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }   
}


#[derive(Debug, Clone, Copy)]
pub struct STensor1D<T: TensorTrait<T>, const R: usize> {
    pub data: [STensor0D<T>;R],
    pub shape: [usize;1],
}
impl <T: TensorTrait<T>, const R: usize> STensor1D<T, R> {

    pub fn new(data: [T;R]) -> Self {
        let mut output = STensor1D {
            data: [STensor0D::new(T::zero());R],
            shape: [R]

        };

        data.iter().zip(output.data.iter_mut()).for_each(|(data, new)| {
            *new = STensor0D::new(*data)
        });

        return output
    }

    pub fn transpose(self) -> STensor2D<T, 1, R> {
        let mut output = STensor2D::new(
            [[T::zero()];R]
        );

        self.data.iter().zip(output.data.iter_mut()).for_each(|(data, new)| {
           *new = STensor1D::new([data.0])
        });

        return output
    }
    
}

impl <T: TensorTrait<T>, const R: usize> Display for STensor1D<T, R> {

    fn fmt(&self, f: &mut Formatter) -> fmt::Result { 
        write!(f,"[")?;
        for x in 0..R {
            if x != R-1 {
                write!(f, "{:?}, ", self.data[x].0)?
            } else {
                write!(f, "{:?}", self.data[x].0)?
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct STensor2D<T: TensorTrait<T>, const R: usize, const C: usize> {
    pub data: [STensor1D<T,R>;C],
    pub shape: [usize;2],
}


impl <T: TensorTrait<T>, const R: usize, const C: usize> STensor2D<T, R, C> {

    pub fn new(data: [[T;R];C]) -> Self {
        let mut new_tensor = STensor2D {
            data: [STensor1D::new([T::zero();R]);C],
            shape: [R,C]
        };
        data.iter().zip(new_tensor.data.iter_mut()).for_each(|(data, new)| {
            *new = STensor1D::new(*data)
        });
        return new_tensor
    }

    pub fn transpose(self) -> STensor2D<T, C, R> {
        let temp = self;
        let mut output = STensor2D{
            data: [STensor1D::new([T::zero();C]);R],
            shape: [C,R],
        };
        for x in 0..C {
            for y in 0..R {
                output.data[y].data[x] = temp.data[x].data[y]
            }
        }
        return output
    }

    pub fn zeros() -> STensor2D<T, R, C> {
        return STensor2D::new([[T::zero();R];C])
    }

    pub fn rand() -> STensor2D<T, R, C> 
    where  
        Standard: Distribution<T>
    {
        let mut rng  = rand::thread_rng();

        let mut new_tensor = STensor2D::<T, R, C>::zeros();


        for y in 0..C {
            for x in 0..R {
                new_tensor.data[y].data[x] = STensor0D::rand();
            }
        }

        return new_tensor
    }


}

impl <T: TensorTrait<T>, const R: usize, const C: usize> Display for STensor2D<T, R, C> {

    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "[")?;

        for x in 0..R {
            write!(f, "\t[")?;
            for y in 0..C {
                if y != C-1 {
                    write!(f, "{:?}, ", self.data[x].data[y].0)?
                } else {
                    write!(f, "{:?}", self.data[x].data[y].0)?
                }
            }
            if x != R-1 {
                write!(f, "],\n")?
            } else {
                write!(f, "]\n")?
            }
        }

        writeln!(f, "]")

    }
}


#[derive(Debug, Clone, Copy)]
pub struct STensor3D<T: TensorTrait<T>, const R: usize, const C: usize, const P: usize> {
    pub data: [STensor2D<T, R, C>; P],
    pub shape: [usize;3]
}

impl <T: TensorTrait<T>, const R: usize, const C: usize, const P: usize> STensor3D<T, R, C, P>{
    pub fn new(data: [[[T;R];C];P]) -> Self {
        let mut new_tensor = STensor3D {
            data: [STensor2D::new([[T::zero();R];C]);P],
            shape: [R,C,P]
        };

        data.iter().zip(new_tensor.data.iter_mut()).for_each(|(data,  new)| {
            *new  = STensor2D::new(*data)
        });

        return new_tensor
    }

    pub fn transpose(self) -> STensor3D<T, P, C, R> {

        let temp = self;

        let mut output = STensor3D{
            data: [STensor2D::new([[T::zero();P];C]);R],
            shape: [P,C,R],
        };

        for x in 0..P {
            for y in 0..C {
                for z in 0..R {
                    output.data[z].data[y].data[x] = temp.data[x].data[y].data[z]
                }
            }
        }

        return output
    }

}

impl <T: TensorTrait<T>, const R: usize, const C: usize, const P: usize> Display for STensor3D<T, R, C, P> {

    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "[")?;

        for x in 0..R {

            writeln!(f, "\t[")?;

            for y in 0..C {
                write!(f, "\t\t[")?;
                for z in 0..P {
                    if z != P-1 {
                        write!(f, "{:?}, ", self.data[x].data[y].data[z].0)?
                    } else {
                        write!(f, "{:?}", self.data[x].data[y].data[z].0)?
                        
                    }
                }
                if y != C-1 {
                    write!(f, "],\n")?
                } else {
                    write!(f, "]\n")?
                }
            }
            if x != R-1 {
                writeln!(f, "\t],")?
            } else {
                writeln!(f, "\t]")?
            }
        }
        writeln!(f, "]")

    }
}