/// Define general CDF and conversion from iterator
use crate::error::{Error, ErrorCode};
use std::cmp::Ordering;
use std::convert::From;
use std::ops::Bound;

#[derive(Debug, Clone, Copy)]
struct DistItem<T: Copy> {
    start: Bound<f32>,
    end: Bound<f32>,
    value: T,
}

impl<T: Copy> DistItem<T> {
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
}

#[derive(Clone, Debug)]
pub struct CDF<T: Copy>(Vec<DistItem<T>>);

impl<T: Copy> CDF<T> {
    pub fn get_value(&self, p: f32) -> Result<T, Error> {
        if p < 0.0 || 1.0 < p {
            return Err(Error::make_error_syntax(ErrorCode::InvalidArgumentInput));
        }

        let x = self.0.binary_search_by(|v| v.cmp(p));
        if let Ok(idx) = x {
            Ok(self.0[idx].value)
        } else {
            unreachable!();
        }
    }
}

impl<I, T> From<I> for CDF<T>
where
    I: IntoIterator<Item = (f32, T)> + Clone,
    T: Copy,
{
    fn from(s: I) -> Self {
        let mut items: Vec<DistItem<T>> = vec![];
        let normalize_factor: f32 = s.clone().into_iter().map(|(k, _)| k).sum();
        let mut temp = 0f32;

        for (k, v) in s.into_iter() {
            items.push(DistItem {
                start: Bound::Included(temp),
                end: Bound::Excluded(temp + k / normalize_factor),
                value: v,
            });
            temp = temp + k / normalize_factor;
        }

        let last_idx = items.len() - 1;
        if let DistItem {
            end: Bound::Excluded(v),
            ..
        } = items[last_idx]
        {
            items[last_idx].end = Bound::Included(v);
        }

        Self(items)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_dist_item_order() {
        let item = DistItem::<f32> {
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
    fn test_get_value() {
        let cdf = CDF::<usize>(vec![
            DistItem {
                start: Bound::Included(1.0),
                end: Bound::Excluded(2.0),
                value: 3,
            },
            DistItem {
                start: Bound::Included(2.0),
                end: Bound::Excluded(3.0),
                value: 4,
            },
            DistItem {
                start: Bound::Included(3.0),
                end: Bound::Included(5.0),
                value: 5,
            },
        ]);

        assert_eq!(cdf.get_value(1.0), Ok(3));
        assert_eq!(cdf.get_value(1.5), Ok(3));
        assert_eq!(cdf.get_value(2.0), Ok(4));
        assert_eq!(cdf.get_value(4.0), Ok(5));
        assert_eq!(cdf.get_value(5.0), Ok(5));
    }

    #[test]
    fn test_from() {
        let items = [(1.0, 2), (2.0, 3), (3.0, 4), (4.0, 5)];
        let cdf: CDF<usize> = items.into();

        assert_eq!(cdf.get_value(0.0), Ok(2));
        assert_eq!(cdf.get_value(0.1), Ok(3));
        assert_eq!(cdf.get_value(0.2), Ok(3));
        assert_eq!(cdf.get_value(0.3), Ok(4));
        assert_eq!(cdf.get_value(0.5), Ok(4));
        assert_eq!(cdf.get_value(0.6), Ok(5));
        assert_eq!(cdf.get_value(0.7), Ok(5));
        assert_eq!(cdf.get_value(1.0), Ok(5));
    }
}
