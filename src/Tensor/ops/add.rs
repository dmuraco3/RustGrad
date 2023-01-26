use std::ops::Add;

use crate::Tensor::{TensorTrait, STensor0D, STensor1D, STensor2D, STensor3D};

impl <T: TensorTrait<T>> Add for STensor0D<T> {
    type Output = STensor0D<T>;

    fn add(self, rhs: Self) -> Self::Output {
        STensor0D (self.0 + rhs.0)
    }

}

impl <T: TensorTrait<T>, const R: usize> Add for STensor1D<T, R> {
    type Output = STensor1D<T, R>;

    #[inline(never)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut new_tensor = STensor1D::new([T::zero();R]);

        for x in 0..R {
            new_tensor.data[x] = self.data[x] + rhs.data[x]
        }

        return new_tensor
    }
}

impl <T: TensorTrait<T>, const R: usize, const C: usize> Add for STensor2D<T, R, C> {
    type Output = STensor2D<T, R, C>;

    #[inline(never)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut new_tensor = STensor2D::new([[T::zero();R];C]);

        for y in 0..C {
            new_tensor.data[y] = self.data[y] + rhs.data[y];
        }

        return new_tensor
    }
}

impl <T: TensorTrait<T>, const R: usize, const C: usize, const P: usize> Add for STensor3D<T, R, C, P> {
    type Output = STensor3D<T, R, C, P>;

    #[inline(never)]
    fn add(self, rhs: Self) -> Self::Output {
        let mut new_tensor = STensor3D::new([[[T::zero();R];C];P]);

        for y in 0..P {
            new_tensor.data[y] = self.data[y] + rhs.data[y];
        }

        return new_tensor;
    }

}