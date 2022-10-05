use std::cmp::Ordering;
use std::collections::HashMap;
use std::convert::From;
/// Define general CDF and conversion from iterator
use std::hash::Hash;
use std::ops::{Bound, Index};

#[derive(Debug, Clone)]
pub struct PDF<T: Eq + Hash>(HashMap<T, f32>);

impl<T: Eq + Hash> PDF<T> {
    pub fn new(map: HashMap<T, f32>) -> Self {
        Self(map)
    }

    pub fn get_pdf(&self, k: &T) -> Option<&f32> {
        self.0.get(k)
    }
}

impl<T: Eq + Hash + Copy> Index<T> for PDF<T> {
    type Output = f32;

    fn index(&self, k: T) -> &f32 {
        &self.0[&k]
    }
}

impl<T: Eq + Hash> IntoIterator for PDF<T> {
    type Item = (T, f32);
    type IntoIter = std::collections::hash_map::IntoIter<T, f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
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
    pub fn get_cdf(&self, p: f32) -> Option<T> {
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

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_get_pdf() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map);
        assert_eq!(pdf.get_pdf(&1), Some(&0.3));
        assert_eq!(pdf.get_pdf(&2), Some(&0.4));
        assert_eq!(pdf.get_pdf(&3), Some(&0.3));
        assert_eq!(pdf.get_pdf(&4), None);
    }

    #[test]
    fn test_index() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map);
        assert_eq!(pdf[1], 0.3);
        assert_eq!(pdf[2], 0.4);
        assert_eq!(pdf[3], 0.3);
    }

    #[test]
    #[should_panic]
    fn test_index2() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map);
        let _x = pdf[4];
    }

    #[test]
    fn test_intoiter() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let mut vec: Vec<(usize, f32)> = PDF::new(map).into_iter().collect();
        vec.sort_by_key(|v| v.0);
        assert_eq!(vec, vec![(1, 0.3), (2, 0.4), (3, 0.3)]);
    }

    #[test]
    fn test_pdf_into_cdf() {
        let mut map: HashMap<usize, f32> = HashMap::new();
        map.insert(1, 0.3);
        map.insert(2, 0.4);
        map.insert(3, 0.3);

        let pdf = PDF::new(map.clone());
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
    fn test_get_cdf() {
        let cdf = CDF::<usize>(vec![
            CDFItem {
                start: Bound::Included(1.0),
                end: Bound::Excluded(2.0),
                value: 3,
            },
            CDFItem {
                start: Bound::Included(2.0),
                end: Bound::Excluded(3.0),
                value: 4,
            },
            CDFItem {
                start: Bound::Included(3.0),
                end: Bound::Included(5.0),
                value: 5,
            },
        ]);

        assert_eq!(cdf.get_cdf(1.0), Some(3));
        assert_eq!(cdf.get_cdf(1.5), Some(3));
        assert_eq!(cdf.get_cdf(2.0), Some(4));
        assert_eq!(cdf.get_cdf(4.0), Some(5));
        assert_eq!(cdf.get_cdf(5.0), Some(5));
    }

    #[test]
    fn test_from() {
        let items = [(2, 1.0), (3, 2.0), (4, 3.0), (5, 4.0)];
        let cdf: CDF<usize> = items.into();

        assert_eq!(cdf.get_cdf(0.0), Some(2));
        assert_eq!(cdf.get_cdf(0.1), Some(3));
        assert_eq!(cdf.get_cdf(0.2), Some(3));
        assert_eq!(cdf.get_cdf(0.3), Some(4));
        assert_eq!(cdf.get_cdf(0.5), Some(4));
        assert_eq!(cdf.get_cdf(0.6), Some(5));
        assert_eq!(cdf.get_cdf(0.7), Some(5));
        assert_eq!(cdf.get_cdf(1.0), Some(5));
    }
}
