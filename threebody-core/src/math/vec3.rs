use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::float::approx_eq;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn norm_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn norm(self) -> f64 {
        self.norm_sq().sqrt()
    }

    pub fn normalized(self) -> Option<Self> {
        let n = self.norm();
        if n == 0.0 {
            None
        } else {
            Some(self / n)
        }
    }

    pub fn approx_eq(self, other: Self, rel: f64, abs: f64) -> bool {
        approx_eq(self.x, other.x, rel, abs)
            && approx_eq(self.y, other.y, rel, abs)
            && approx_eq(self.z, other.z, rel, abs)
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl Div<f64> for Vec3 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::Vec3;

    #[test]
    fn add_sub_scalar() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b - a, Vec3::new(3.0, 3.0, 3.0));
        assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
        assert_eq!(b / 2.0, Vec3::new(2.0, 2.5, 3.0));
    }

    #[test]
    fn dot_cross_norm() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(a.dot(b), 0.0);
        assert_eq!(a.cross(b), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(a.norm(), 1.0);
    }

    #[test]
    fn normalize_and_zero() {
        let a = Vec3::new(3.0, 0.0, 0.0);
        let n = a.normalized().unwrap();
        assert!(n.approx_eq(Vec3::new(1.0, 0.0, 0.0), 1e-12, 1e-12));

        let z = Vec3::zero();
        assert!(z.normalized().is_none());
    }

    #[test]
    fn approx_eq_edges() {
        let a = Vec3::new(1.0, 1.0, 1.0);
        let b = Vec3::new(1.0 + 1e-9, 1.0 - 1e-9, 1.0);
        assert!(a.approx_eq(b, 1e-8, 1e-10));
    }
}
