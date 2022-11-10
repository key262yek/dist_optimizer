use crate::distribution::PDF;
use crate::error::Error;
use std::collections::HashMap;
/// Optimization method using grdient descent
use std::hash::Hash;

impl<T: Eq + Hash + Copy> PDF<T> {
    pub fn single_slope(&mut self, idx: T, h: f32, f: &dyn Fn(&Self) -> f32) -> Result<f32, Error> {
        let f0 = f(self);

        let v = self[idx];
        let v0 = self.constrained_value();

        if v0 > h {
            self.modify_pdf(idx, v + h);
            let f1 = f(self);
            self.modify_pdf(idx, v);

            return Ok((f1 - f0) / h);
        } else if v > h {
            self.modify_pdf(idx, v - h);
            let f1 = f(self);
            self.modify_pdf(idx, v);

            return Ok((f0 - f1) / h);
        } else {
            return Err(Error::make_error_msg(
                format!("Too big step size h = {}", h).to_string(),
            ));
        }
    }

    // pub fn gradient(&mut self, f: &dyn Fn(&Self) -> f32) -> Result<HashMap<T, f32>, Error> {
    //     const h: f32 = 1e-4;
    //     let mut grad: HashMap<T, f32> = HashMap::new();
    //     for &key in self.map.keys() {
    //             let slope = self.single_slope(key, h, f)?;
    //             grad.insert(key, slope);
    //     }    
    //     return Ok(grad);
    // }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::collections::HashMap;

    fn functional<T: Eq + Hash>(pdf: &PDF<T>) -> f32 {
        return pdf.prob.map(|s| s.powi(2)).sum();
    }

    #[test]
    fn test_single_slope() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let mut pdf = PDF::new(map, 1);
        assert_abs_diff_eq!(
            pdf.single_slope(2, 1e-4, &functional).unwrap(),
            0.2,
            epsilon = 1e-3
        );
        assert_abs_diff_eq!(
            pdf.single_slope(3, 1e-4, &functional).unwrap(),
            0.0,
            epsilon = 1e-3
        );
    }
}
