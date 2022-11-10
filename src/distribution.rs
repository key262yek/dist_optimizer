use approx::AbsDiffEq;
/// Define general CDF and conversion from iterator
use rand::{distributions::Distribution, Rng};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::From;
use std::hash::Hash;
use std::ops::{Bound, Index};
use ndarray::Array1;

#[derive(Debug, Clone)]
pub struct PDF<T: Eq + Hash> {
    pub(crate) map: HashMap<T, usize>,
    pub(crate) prob: Array1<f32>, 
    constrained_item: (T, usize),
}

impl<T: Eq + Hash + Copy> PDF<T> {
    pub fn new(map: HashMap<T, f32>, constraint: T) -> Self {
        let sum: f32 = map.values().sum();
        if sum.abs_diff_ne(&1.0, 1e-10) {
            panic!("PDF is not normalized. sum : {:?}", sum);
        }

        let mut new_map : HashMap<T, usize> = HashMap::new();
        let mut prob: Vec<f32> = Vec::new();

        let mut idx = 0usize;
        for (&key, &value) in map.iter(){
            new_map.insert(key, idx);
            prob.push(value);
            idx += 1;
        }

        let prob = Array1::from_iter(prob);
        let constrained_idx = new_map[&constraint];
        Self {
            map: new_map,
            prob,
            constrained_item: (constraint, constrained_idx),
        }
    }

    pub fn get_pdf(&self, k: T) -> Option<f32> {
        let idx = self.map.get(&k);
        idx.map(|&x| self.prob[x])
    }

    pub fn modify_pdf(&mut self, k: T, v: f32) {
        let idx = *self.map.get(&k).unwrap();
        let constrained_idx = self.constrained_item.1;
        let dv = v - self.prob[idx];

        if self.prob[constrained_idx] < dv {
            panic!("Increase of probability is too big to normalize");
        }

        self.prob[idx] += dv;
        self.prob[constrained_idx] -= dv;
    }

    pub fn constrained_value(&self) -> f32 {
        return self.prob[self.constrained_item.1];
    }
}

impl<T: Eq + Hash + Copy> Index<T> for PDF<T> {
    type Output = f32;

    fn index(&self, k: T) -> &f32 {
        &self.prob[self.map[&k]]
    }
}

impl<'a, T: Eq + Hash + Copy> Index<T> for &'a PDF<T> {
    type Output = f32;

    fn index(&self, k: T) -> &f32 {
        &self.prob[self.map[&k]]
    }
}

impl<'a, T: Eq + Hash + Copy> Index<T> for &'a mut PDF<T> {
    type Output = f32;

    fn index(&self, k: T) -> &f32 {
        &self.prob[self.map[&k]]
    }
}

pub struct PDFIter<T: Eq + Hash>{
    items: std::collections::hash_map::IntoIter<T, usize>,
    prob: Array1<f32>,
}

impl<T: Eq + Hash> std::iter::Iterator for PDFIter<T>{
    type Item = (T, f32);

    fn next(&mut self) -> Option<Self::Item>{
        self.items.next().map(|(x, idx)| (x, self.prob[idx]))
    }
}

impl<T: Eq + Hash> IntoIterator for PDF<T> {
    type Item = (T, f32);
    type IntoIter = PDFIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        PDFIter{
            items: self.map.into_iter(),
            prob: self.prob,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CDFItem<T: Copy> {
    start: Bound<f32>,
    end: Bound<f32>,
    value: T,
}

impl<T: Copy> CDFItem<T> {
    fn cmp(&self, p: f32) -> Ordering {
        // Return whether the range is greater than p, is less than p or contains p
        match self.start {
            Bound::Included(v) => {
                if p < v {
                    return Ordering::Greater;
                }
            }
            Bound::Excluded(v) => {
                if p <= v {
                    return Ordering::Greater;
                }
            }
            Bound::Unbounded => unreachable!(),
        }

        match self.end {
            Bound::Included(v) => {
                if p > v {
                    return Ordering::Less;
                }
            }
            Bound::Excluded(v) => {
                if p >= v {
                    return Ordering::Less;
                }
            }
            Bound::Unbounded => unreachable!(),
        }

        return Ordering::Equal;
    }

    fn len(&self) -> f32 {
        match (self.start, self.end) {
            (Bound::Included(s), Bound::Included(e))
            | (Bound::Included(s), Bound::Excluded(e))
            | (Bound::Excluded(s), Bound::Included(e))
            | (Bound::Excluded(s), Bound::Excluded(e)) => e - s,
            (Bound::Unbounded, Bound::Included(_e)) | (Bound::Unbounded, Bound::Excluded(_e)) => {
                -f32::INFINITY
            }
            (_, Bound::Unbounded) => f32::INFINITY,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CDF<T: Copy>(Vec<CDFItem<T>>);

impl<T: Copy> CDF<T> {
    pub fn get_value(&self, p: f32) -> Option<T> {
        if p < 0.0 || 1.0 < p {
            return None;
        }

        let x = self.0.binary_search_by(|v| v.cmp(p));
        if let Ok(idx) = x {
            Some(self.0[idx].value)
        } else {
            unreachable!();
        }
    }
}

impl<I, T> From<I> for CDF<T>
where
    I: IntoIterator<Item = (T, f32)> + Clone,
    T: Copy,
{
    fn from(s: I) -> Self {
        let mut items: Vec<CDFItem<T>> = vec![];
        let normalize_factor: f32 = s.clone().into_iter().map(|(_, p)| p).sum();
        let mut temp = 0f32;

        for (k, p) in s.into_iter() {
            items.push(CDFItem {
                start: Bound::Included(temp),
                end: Bound::Excluded(temp + p / normalize_factor),
                value: k,
            });
            temp = temp + p / normalize_factor;
        }

        let last_idx = items.len() - 1;
        if let CDFItem {
            end: Bound::Excluded(p),
            ..
        } = items[last_idx]
        {
            items[last_idx].end = Bound::Included(p);
        }

        Self(items)
    }
}

impl<T> Distribution<T> for CDF<T>
where
    T: Copy,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let p: f32 = rng.gen();
        return self.get_value(p).unwrap();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::collections::BTreeMap;

    #[test]
    fn test_get_pdf() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map, 1);
        assert_abs_diff_eq!(pdf.get_pdf(1).unwrap(), &0.3, epsilon = 1e-5);
        assert_abs_diff_eq!(pdf.get_pdf(2).unwrap(), &0.4, epsilon = 1e-5);
        assert_abs_diff_eq!(pdf.get_pdf(3).unwrap(), &0.3, epsilon = 1e-5);
        assert_eq!(pdf.get_pdf(4), None);
    }

    #[test]
    fn test_index() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map, 1);
        assert_abs_diff_eq!(pdf[1], 0.3, epsilon = 1e-5);
        assert_abs_diff_eq!(pdf[2], 0.4, epsilon = 1e-5);
        assert_abs_diff_eq!(pdf[3], 0.3, epsilon = 1e-5);
    }

    #[test]
    #[should_panic]
    fn test_index2() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map, 1);
        let _x = pdf[4];
    }

    #[test]
    fn test_modify_pdf() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let mut pdf = PDF::new(map, 1);
        pdf.modify_pdf(2, 0.5);
        assert_abs_diff_eq!(pdf[1], 0.2, epsilon = 1e-5);
        assert_abs_diff_eq!(pdf[2], 0.5, epsilon = 1e-5);

        pdf.modify_pdf(3, 0.2);
        assert_abs_diff_eq!(pdf[1], 0.3, epsilon = 1e-5);
        assert_abs_diff_eq!(pdf[3], 0.2, epsilon = 1e-5);
    }

    #[test]
    #[should_panic]
    fn test_modify_pdf2() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let mut pdf = PDF::new(map, 1);
        pdf.modify_pdf(2, 0.8);
    }

    #[test]
    fn test_intoiter() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let mut vec: Vec<(usize, f32)> = PDF::new(map, 1).into_iter().collect();
        vec.sort_by_key(|v| v.0);
        assert_eq!(vec, vec![(1, 0.3), (2, 0.4), (3, 0.3)]);
    }

    #[test]
    fn test_pdf_into_cdf() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map.clone(), 1);
        let cdf: CDF<usize> = pdf.into();
        for item in cdf.0.iter() {
            assert_abs_diff_eq!(item.len(), map[&item.value]);
        }
    }

    #[test]
    fn test_cdf_item_order() {
        let item = CDFItem::<f32> {
            start: Bound::Included(1.0),
            end: Bound::Excluded(2.0),
            value: 3.0,
        };

        assert_eq!(item.cmp(0.0), Ordering::Greater);
        assert_eq!(item.cmp(1.0), Ordering::Equal);
        assert_eq!(item.cmp(1.5), Ordering::Equal);
        assert_eq!(item.cmp(2.0), Ordering::Less);
        assert_eq!(item.cmp(3.0), Ordering::Less);
    }

    #[test]
    fn test_len_cdf_item() {
        let item = CDFItem::<f32> {
            start: Bound::Included(1.0),
            end: Bound::Excluded(2.0),
            value: 3.0,
        };
        assert_abs_diff_eq!(item.len(), 1.0);
    }

    #[test]
    fn test_get_value() {
        let cdf = CDF::<usize>(vec![
            CDFItem {
                start: Bound::Included(0.0),
                end: Bound::Excluded(0.5),
                value: 3,
            },
            CDFItem {
                start: Bound::Included(0.5),
                end: Bound::Excluded(0.7),
                value: 4,
            },
            CDFItem {
                start: Bound::Included(0.7),
                end: Bound::Included(1.0),
                value: 5,
            },
        ]);

        assert_eq!(cdf.get_value(0.0), Some(3));
        assert_eq!(cdf.get_value(0.1), Some(3));
        assert_eq!(cdf.get_value(0.2), Some(3));
        assert_eq!(cdf.get_value(0.3), Some(3));
        assert_eq!(cdf.get_value(0.4), Some(3));
        assert_eq!(cdf.get_value(0.5), Some(4));
        assert_eq!(cdf.get_value(0.6), Some(4));
        assert_eq!(cdf.get_value(0.7), Some(5));
        assert_eq!(cdf.get_value(0.8), Some(5));
        assert_eq!(cdf.get_value(0.9), Some(5));
        assert_eq!(cdf.get_value(1.0), Some(5));
    }

    #[test]
    fn test_from() {
        let items = [(2, 1.0), (3, 2.0), (4, 3.0), (5, 4.0)];
        let cdf: CDF<usize> = items.into();

        assert_eq!(cdf.get_value(0.0), Some(2));
        assert_eq!(cdf.get_value(0.1), Some(3));
        assert_eq!(cdf.get_value(0.2), Some(3));
        assert_eq!(cdf.get_value(0.3), Some(4));
        assert_eq!(cdf.get_value(0.5), Some(4));
        assert_eq!(cdf.get_value(0.6), Some(5));
        assert_eq!(cdf.get_value(0.7), Some(5));
        assert_eq!(cdf.get_value(1.0), Some(5));
    }

    #[test]
    fn test_sample() {
        const INC: u128 = 0xa02bdbf7bb3c0a7ac28fa16a64abf96;
        const SEED: u128 = 1234567890;
        let mut pcg = rand_pcg::Pcg64::new(SEED, INC);

        let items = [(2, 1.0), (3, 2.0), (4, 3.0), (5, 4.0)];
        let cdf: CDF<usize> = items.into();

        let mut count: BTreeMap<usize, usize> = BTreeMap::new();
        count.insert(2, 0);
        count.insert(3, 0);
        count.insert(4, 0);
        count.insert(5, 0);

        for _i in 0..10000 {
            let n = cdf.sample(&mut pcg);
            *count.get_mut(&n).unwrap() += 1;
        }

        assert_eq!(*count.get_mut(&2).unwrap(), 990);
        assert_eq!(*count.get_mut(&3).unwrap(), 2045);
        assert_eq!(*count.get_mut(&4).unwrap(), 3029);
        assert_eq!(*count.get_mut(&5).unwrap(), 3936);
    }
}
