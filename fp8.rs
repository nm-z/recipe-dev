//! FP8 storage codecs shared by the runtime and template generator.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Encoding {
	E4M3,
	E5M2,
}

impl Encoding {
	pub(crate) fn pack(self, value: f64) -> u64 {
		let sign = (value.to_bits() >> 63) << 7;
		let (man, bias, largest, nan) = match self {
			Self::E4M3 => (3, 7, 126, 127),
			Self::E5M2 => (2, 15, 123, 126),
		};
		if value.is_nan() { return sign | nan; }
		if value.is_infinite() {
			return sign | if self == Self::E5M2 { 124 } else { largest };
		}
		let magnitude = value.abs();
		if magnitude == 0.0 { return sign; }
		if magnitude >= self.unpack(largest) { return sign | largest; }
		let exponent = ((magnitude.to_bits() >> 52) & 2047) as i32 - 1023;
		let target = exponent + bias;
		let bits = if target <= 0 {
			(magnitude * 2.0_f64.powi(bias + man - 1)).round_ties_even() as u64
		} else {
			let significand = (magnitude * 2.0_f64.powi(man - exponent)).round_ties_even() as u64;
			(((target - 1) as u64) << man) + significand
		};
		sign | bits.min(largest)
	}

	pub(crate) fn unpack(self, bits: u64) -> f64 {
		let (man, bias, mask) = match self { Self::E4M3 => (3, 7, 15), Self::E5M2 => (2, 15, 31) };
		let magnitude = bits & 127;
		if self == Self::E4M3 && magnitude == 127 || self == Self::E5M2 && magnitude > 124 { return f64::NAN; }
		let exponent = ((bits >> man) & mask) as i32;
		let fraction = bits & ((1 << man) - 1);
		let value = if self == Self::E5M2 && magnitude == 124 {
			f64::INFINITY
		} else if exponent == 0 {
			fraction as f64 * 2.0_f64.powi(1 - bias - man)
		} else {
			((1 << man) + fraction) as f64 * 2.0_f64.powi(exponent - bias - man)
		};
		if bits & 128 != 0 { -value } else { value }
	}
}

#[cfg(test)]
mod tests {
	use super::Encoding::*;
	#[test]
	fn all_finite_codes_round_trip() {
		for format in [E4M3, E5M2] {
			for bits in 0..256 {
				let value = format.unpack(bits);
				if !value.is_nan() { assert_eq!(format.pack(value), bits, "{format:?} {bits}"); }
			}
		}
	}
	#[test]
	fn range_subnormals_and_even_ties() {
		assert_eq!(E4M3.unpack(126), 448.0);
		assert_eq!(E4M3.pack(500.0), 126);
		assert_eq!(E4M3.unpack(1), 2.0_f64.powi(-9));
		assert_eq!(E4M3.pack(1.0625), 56);
		assert_eq!(E4M3.pack(1.1875), 58);
		assert_eq!(E5M2.unpack(123), 57344.0);
		assert_eq!(E5M2.unpack(1), 2.0_f64.powi(-16));
		assert_eq!(E5M2.pack(1.125), 60);
		assert_eq!(E5M2.pack(1.375), 62);
	}
}
