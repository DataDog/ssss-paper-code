#[derive(Clone, Debug)]
pub(crate) struct Distribution {
    cdf: Vec<f64>,
}

impl Distribution {
    /// The pmf needs not be normalized.
    pub(crate) fn new_from_pmf(pmf: Vec<usize>) -> Self {
        let mut cdf = vec![0.0; pmf.len()];
        if !pmf.is_empty() {
            cdf[0] = pmf[0] as f64;
            (1..pmf.len()).for_each(|i| cdf[i] = cdf[i - 1] + pmf[i] as f64);
        }

        // Normalize.
        let sum = *cdf.last().unwrap();
        cdf.iter_mut().for_each(|c| *c /= sum);

        Self { cdf }
    }

    pub(crate) fn new_from_cdf(cdf: Vec<f64>) -> Self {
        Self { cdf }
    }

    pub(crate) fn cdf(&self, i: isize) -> f64 {
        match i {
            i if i < 0 => 0.0,
            i if i >= self.cdf.len() as isize => *self.cdf.last().unwrap(),
            i => self.cdf[i as usize],
        }
    }

    pub(crate) fn pmf(&self, i: isize) -> f64 {
        self.cdf(i) - self.cdf(i - 1)
    }

    pub(crate) fn pmf_iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        (0..self.cdf.len())
            .map(|i| (i, self.pmf(i as isize)))
            .filter(|&(_, p)| p != 0.0)
    }
}

impl FromIterator<usize> for Distribution {
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let mut pmf = vec![];
        iter.into_iter().for_each(|z| {
            if z >= pmf.len() {
                pmf.resize(z + 1, 0);
            }
            pmf[z] += 1;
        });
        Self::new_from_pmf(pmf)
    }
}

pub(crate) fn geometric(n: usize, d: usize) -> Distribution {
    Distribution::new_from_cdf(
        (0..=n)
            .map(|x| 1.0 - 2.0_f64.powi(-(x as i32)) / d as f64)
            .collect(),
    )
}
