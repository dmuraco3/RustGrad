use std::{ops::Mul, process::Termination, time::Instant};

use crate::Tensor::{TensorTrait, STensor0D, STensor1D, STensor2D, STensor3D};

impl <T: TensorTrait<T>> Mul for STensor0D<T> {
    type Output = STensor0D<T>;

    #[inline(never)]
    fn mul(self, rhs: Self) -> Self::Output {
        return STensor0D::new(self.0 * rhs.0)
    }
}

impl <T: TensorTrait<T>, const R: usize> Mul for STensor1D<T, R> {
    type Output = STensor0D<T>;
    /// ## Mul represents a dot product function on > rank 0 Tensors
    fn mul(self, rhs: Self) -> Self::Output {
        let mut total = STensor0D::new(T::zero());

        for x in 0..R {
            total = (self.data[x] * rhs.data[x]) + total;
        }

        return total
    }
} 

impl <T: TensorTrait<T>, const R: usize, const C: usize> Mul<STensor2D<T,C,R>> for STensor2D<T, R, C> 
{
    type Output = STensor2D<T, C, C>;

    #[inline(never)]
    fn mul(self, rhs: STensor2D<T, C, R>) -> Self::Output {

        let mut new_tensor = STensor2D::<T, C, C>::zeros();

        let rhs = rhs.transpose();

        // self.data.iter().enumerate().for_each(|(index_x, lhs)| {
        //     rhs.data.iter().enumerate().for_each(|(index_y, rhs)| {
        //         new_tensor.data[index_x].data[index_y] = lhs.data.iter().zip(rhs.data.iter()).fold(STensor0D::new(T::zero()), |acc, (lhs, rhs)| acc + *lhs * *rhs)
        //     });
        // });

        self.data.iter().zip(new_tensor.data.iter_mut()).enumerate().for_each(|(index_x, (lhs, new))| {
            rhs.data.iter().zip(new.data.iter_mut()).enumerate().for_each(|(index_y, (rhs, new))| {
                *new = lhs.data.iter().zip(rhs.data.iter()).fold(STensor0D::new(T::zero()), |acc, (lhs, rhs)| acc + *lhs * *rhs)
            });
        });

        return new_tensor
    }
}
