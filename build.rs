use std::{
	env,
	error::Error,
	fs, io,
	path::{Path, PathBuf},
};
#[derive(Clone, Copy, PartialEq, Eq)]
struct FloatLayout {
	sign: u8,
	exp: u8,
	man: u8,
}
impl FloatLayout {
	const fn new(sign: u8, exp: u8, man: u8) -> Self {
		Self { sign, exp, man }
	}
	const fn bits(self) -> u8 {
		self.sign + self.exp + self.man
	}
	fn unpack(self, bits: u64) -> f64 {
		let exponent_limit = (1u64 << self.exp) - 1;
		let mantissa_limit = 1u64 << self.man;
		let exponent = bits >> self.man & exponent_limit;
		let mantissa = bits & (mantissa_limit - 1);
		let magnitude = match (exponent, mantissa) {
			(value, 0) if value == exponent_limit => f64::INFINITY,
			(value, _) if value == exponent_limit => f64::NAN,
			(0, 0) => 0.0,
			(0, value) => 2.0f64.powi(1 - ((1i32 << (self.exp - 1)) - 1)) * value as f64 / mantissa_limit as f64,
			(value, man) => 2.0f64.powi(value as i32 - ((1i32 << (self.exp - 1)) - 1)) * (1.0 + man as f64 / mantissa_limit as f64),
		};
		if bits >> (self.exp + self.man) != 0 { -magnitude } else { magnitude }
	}
}
#[derive(Clone, Copy, PartialEq, Eq)]
struct FloatFormat {
	arithmetic: FloatLayout,
	storage: FloatLayout,
}
impl FloatFormat {
	const FP8: Self = Self::native(1, 5, 2);
	const FP16: Self = Self::native(1, 5, 10);
	const FP32: Self = Self::native(1, 8, 23);
	const FP64: Self = Self::native(1, 11, 52);
	const BF16: Self = Self::native(1, 8, 7);
	const TF32: Self = Self { arithmetic: FloatLayout::new(1, 8, 10), storage: FloatLayout::new(1, 8, 23) };
	const fn native(sign: u8, exp: u8, man: u8) -> Self {
		let layout = FloatLayout::new(sign, exp, man);
		Self { arithmetic: layout, storage: layout }
	}
	const fn bytes(self) -> usize {
		self.storage.bits().div_ceil(8) as usize
	}
	fn pack(self, value: f64) -> u64 {
		let rounded = self.arithmetic.unpack(self.arithmetic.pack_from(value));
		let bits = match self.storage.bits() {
			64 => rounded.to_bits(),
			32 => u64::from((rounded as f32).to_bits()),
			16 if self == Self::FP16 => self.storage.pack_from(rounded),
			16 => u64::from((rounded as f32).to_bits() >> 16),
			8 => self.storage.pack_from(rounded),
			_ => unreachable!(),
		};
		bits
	}
	fn unpack(self, bits: u64) -> f64 {
		match self.storage.bits() {
			64 => f64::from_bits(bits),
			32 => f64::from(f32::from_bits(bits as u32)),
			16 if self == Self::FP16 => self.arithmetic.unpack(bits),
			16 => f64::from(f32::from_bits((bits as u32) << 16)),
			8 => self.arithmetic.unpack(bits),
			_ => unreachable!(),
		}
	}
}
impl FloatLayout {
	fn pack_from(self, value: f64) -> u64 {
		let sign = value.to_bits() >> 63 << (self.exp + self.man);
		let exponent_limit = (1u64 << self.exp) - 1;
		let mantissa_limit = 1u64 << self.man;
		if value.is_nan() {
			return sign | exponent_limit << self.man | 1u64 << (self.man - 1);
		}
		if value.is_infinite() {
			return sign | exponent_limit << self.man;
		}
		if value == 0.0 {
			return sign;
		}
		let bias = (1i32 << (self.exp - 1)) - 1;
		let magnitude = value.abs();
		let mut exponent = magnitude.log2().floor() as i32;
		if exponent < 1 - bias {
			let mantissa = (magnitude / 2.0f64.powi(1 - bias) * mantissa_limit as f64).round_ties_even() as u64;
			return if mantissa == mantissa_limit { sign | 1u64 << self.man } else { sign | mantissa };
		}
		let mut mantissa = ((magnitude / 2.0f64.powi(exponent) - 1.0) * mantissa_limit as f64).round_ties_even() as u64;
		if mantissa == mantissa_limit {
			mantissa = 0;
			exponent += 1
		}
		let stored = exponent + bias;
		if stored >= exponent_limit as i32 { sign | exponent_limit << self.man } else { sign | (stored as u64) << self.man | mantissa }
	}
}
#[derive(Clone, Copy)]
struct IntFormat {
	bits: u8,
}
impl IntFormat {
	const INT1: Self = Self { bits: 1 };
	const INT4: Self = Self { bits: 4 };
	const INT8: Self = Self { bits: 8 };
	const fn bytes(self) -> usize {
		self.bits.div_ceil(8) as usize
	}
	fn pack(self, value: f64) -> u64 {
		(value.round_ties_even() as i64).clamp(-(1i64 << (self.bits - 1)), (1i64 << (self.bits - 1)) - 1) as u64 & ((1u64 << self.bits) - 1)
	}
}
type BuildResult<T> = Result<T, Box<dyn Error>>;
const PARALLEL: &str = r#"declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @recipe.workgroup.size.x()
define internal i32 @global_id() #1 { entry:
%lane = call i32 @llvm.amdgcn.workitem.id.x() %group = call i32 @llvm.amdgcn.workgroup.id.x()
%width = call i32 @recipe.workgroup.size.x() %base = mul i32 %group, %width %id = add i32 %base, %lane ret i32 %id }
@RECIPE_GRID_BARRIER@"#;
const AMD_GRID_BARRIER: &str = r#"declare void @__ockl_grid_sync()
define internal void @grid_barrier(i32 %threads) #1 { entry: call void @__ockl_grid_sync() ret void }"#;
// PTX only accepts ordered atomics on sm_70 and newer, so the counting barrier uses
// relaxed atomics with explicit fences. A release fence before each arrival publishes the
// block's writes; the last arriver acquires them, republishes with a release fence, and
// flips the phase; each waiter acquires after it observes the flip. The fences lower to
// membar, which every NVIDIA architecture supports, so one barrier serves them all.
const NVIDIA_GRID_BARRIER: &str = r#"@grid.count = internal addrspace(1) global i32 0, align 4
@grid.phase = internal addrspace(1) global i32 0, align 4
define internal void @grid_barrier(i32 %threads) #1 { entry:
call void @llvm.amdgcn.s.barrier() %lane = call i32 @llvm.amdgcn.workitem.id.x()
%leader = icmp eq i32 %lane, 0 br i1 %leader, label %arrive, label %joined arrive:
%width = call i32 @recipe.workgroup.size.x() %groups = udiv i32 %threads, %width
%phase = load atomic i32, ptr addrspace(1) @grid.phase monotonic, align 4
fence release
%prior = atomicrmw add ptr addrspace(1) @grid.count, i32 1 monotonic %limit = sub i32 %groups, 1
%last = icmp eq i32 %prior, %limit br i1 %last, label %release, label %wait release:
fence acquire
store atomic i32 0, ptr addrspace(1) @grid.count monotonic, align 4 %next = xor i32 %phase, 1
fence release
store atomic i32 %next, ptr addrspace(1) @grid.phase monotonic, align 4 br label %joined wait:
%seen = load atomic i32, ptr addrspace(1) @grid.phase monotonic, align 4 %ready = icmp ne i32 %seen, %phase
br i1 %ready, label %waited, label %wait waited:
fence acquire br label %joined joined: call void @llvm.amdgcn.s.barrier() ret void }"#;
const AMD_WIDTH: &str = r#"declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
define internal i32 @recipe.workgroup.size.x() #1 { entry: %args = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
%address = getelementptr i8, ptr addrspace(4) %args, i32 4 %value = load i16, ptr addrspace(4) %address, align 2
%width = zext i16 %value to i32 ret i32 %width }"#;
fn parallel_ir(ir: String, width: &str, grid_barrier: &str) -> String {
	let mut ir = ir.replace("call i32 @llvm.amdgcn.workitem.id.x()", "call i32 @global_id()").replace("call void @llvm.amdgcn.s.barrier()", "call void @grid_barrier(i32 %threads)");
	let target = ir.find("target triple").and_then(|start| ir[start..].find('\n').map(|end| start + end + 1)).expect("kernel target triple is absent");
	ir.insert_str(target, &format!("{}\n", PARALLEL.replace("declare i32 @recipe.workgroup.size.x()", width).replace("@RECIPE_GRID_BARRIER@", grid_barrier)));
	ir.replace("recipe.local.id.x", "llvm.amdgcn.workitem.id.x").replace("recipe.group.id.x", "llvm.amdgcn.workgroup.id.x").replace("recipe.local.barrier", "llvm.amdgcn.s.barrier")
}
fn word(text: String, from: &str, to: &str) -> String {
	let (mut output, mut rest) = (String::with_capacity(text.len()), text.as_str());
	while let Some(index) = rest.find(from) {
		let end = index + from.len();
		let identifier = |value: char| value.is_ascii_alphanumeric() || value == '_' || value == '.';
		let bounded = rest[..index].chars().next_back().is_none_or(|value| !identifier(value)) && rest[end..].chars().next().is_none_or(|value| !identifier(value));
		output.push_str(&rest[..index]);
		output.push_str(if bounded { to } else { from });
		rest = &rest[end..];
	}
	output.push_str(rest);
	output
}
fn numeric_region(ir: &str) -> BuildResult<(usize, usize)> {
	let start = ir.find("; NUMERIC BEGIN").ok_or_else(|| io::Error::other("numeric operation block is absent"))?;
	let end = ir[start..].find("; NUMERIC END").map(|offset| start + offset + "; NUMERIC END".len()).ok_or_else(|| io::Error::other("numeric operation block is incomplete"))?;
	Ok((start, end))
}
/// The five transcendental entry points. They are defined by `shared_math`
/// rather than resolved against OCML, CUDA libdevice, or the host library, so
/// every backend evaluates the same coefficients in the same order.
const MATH_NAMES: [&str; 5] = ["recipe.math.exp", "recipe.math.tanh", "recipe.math.cos", "recipe.math.sin", "recipe.math.log"];

fn constant(value: f64) -> String {
	format!("0x{:016X}", value.to_bits())
}
fn math_literal(arithmetic: &str, value: f64) -> String {
	if arithmetic == "double" { constant(value) } else { constant(f64::from(value as f32)) }
}

/// Horner evaluation of `sum(coefficients[i] * variable^i)` as a chain of one
/// multiply and one add per term, innermost coefficient first.
fn horner(prefix: &str, variable: &str, coefficients: &[f64]) -> (String, String) {
	horner_typed(prefix, variable, coefficients, "double")
}

fn horner_typed(prefix: &str, variable: &str, coefficients: &[f64], arithmetic: &str) -> (String, String) {
	let mut ir = String::new();
	let intrinsic = if arithmetic == "double" { "f64" } else { "f32" };
	let mut current = math_literal(arithmetic, coefficients[coefficients.len() - 1]);
	for (step, coefficient) in coefficients.iter().rev().skip(1).enumerate() {
		ir.push_str(&format!(
			"%{prefix}.{step} = call {arithmetic} @llvm.fma.{intrinsic}({arithmetic} {current}, {arithmetic} {variable}, {arithmetic} {})\n",
			math_literal(arithmetic, *coefficient)
		));
		current = format!("%{prefix}.{step}");
	}
	(ir, current)
}

fn exponential_math(arithmetic: &str, name: &str, bias: i32, maximum: i32, shift: u8, high: f64, low: f64, terms: u32) -> String {
	let intrinsic = if arithmetic == "double" { "f64" } else { "f32" };
	let integer = if arithmetic == "double" { "i64" } else { "i32" };
	let stored = if arithmetic == "double" { "%stored = zext i32 %clamped to i64\n" } else { "%stored = add i32 %clamped, 0\n" };
	let infinity = math_literal(arithmetic, f64::INFINITY);
	let (polynomial, value) = horner_typed("exp.poly", "%r", &reciprocal_factorials(1, 1, terms), arithmetic);
	format!(
		"define internal {arithmetic} @{name}.pow2(i32 %k) #1 {{\nentry:\n%biased = add i32 %k, {bias}\n%low = icmp slt i32 %biased, 0\n%floor = select i1 %low, i32 0, i32 %biased\n%high = icmp sgt i32 %floor, {maximum}\n%clamped = select i1 %high, i32 {maximum}, i32 %floor\n{stored}%bits = shl {integer} %stored, {shift}\n%result = bitcast {integer} %bits to {arithmetic}\nret {arithmetic} %result\n}}\ndefine internal {arithmetic} @{name}.scale({arithmetic} %value, i32 %k) #1 {{\nentry:\n%half = ashr i32 %k, 1\n%rest = sub i32 %k, %half\n%first = call {arithmetic} @{name}.pow2(i32 %half)\n%second = call {arithmetic} @{name}.pow2(i32 %rest)\n%scaled = fmul {arithmetic} %value, %first\n%result = fmul {arithmetic} %scaled, %second\nret {arithmetic} %result\n}}\ndefine internal {arithmetic} @{name}({arithmetic} %x) #1 {{\nentry:\n%unordered = fcmp uno {arithmetic} %x, %x\nbr i1 %unordered, label %quiet, label %high.test\nquiet:\nret {arithmetic} %x\nhigh.test:\n%high = fcmp ogt {arithmetic} %x, {}\nbr i1 %high, label %overflow, label %low.test\noverflow:\nret {arithmetic} {infinity}\nlow.test:\n%low = fcmp olt {arithmetic} %x, {}\nbr i1 %low, label %underflow, label %reduce\nunderflow:\nret {arithmetic} 0.0\nreduce:\n%scaled = fmul {arithmetic} %x, {}\n%shifted = fadd {arithmetic} %scaled, {}\n%kf = call {arithmetic} @llvm.floor.{intrinsic}({arithmetic} %shifted)\n%k = fptosi {arithmetic} %kf to i32\n%upper = fmul {arithmetic} %kf, {}\n%partial = fsub {arithmetic} %x, %upper\n%lower = fmul {arithmetic} %kf, {}\n%r = fsub {arithmetic} %partial, %lower\n{polynomial}%body = fmul {arithmetic} %r, {value}\n%expanded = fadd {arithmetic} {}, %body\n%result = call {arithmetic} @{name}.scale({arithmetic} %expanded, i32 %k)\nret {arithmetic} %result\n}}\n",
		math_literal(arithmetic, high),
		math_literal(arithmetic, low),
		math_literal(arithmetic, std::f64::consts::LOG2_E),
		math_literal(arithmetic, 0.5),
		math_literal(arithmetic, f64::from_bits(0x3FE62E42FEE00000)),
		math_literal(arithmetic, 1.90821492927058770002e-10),
		math_literal(arithmetic, 1.0)
	)
}

/// Reciprocal factorials `1/(first!)`, `1/((first + step)!)`, ... up to `last`,
/// each the correctly rounded double of an exact ratio because every factorial
/// through 22! is itself exact in double.
fn reciprocal_factorials(first: u32, step: u32, last: u32) -> Vec<f64> {
	let (mut factorial, mut values) = (1.0_f64, Vec::new());
	for term in 1..=last {
		factorial *= f64::from(term);
		if term >= first && (term - first) % step == 0 {
			values.push(1.0 / factorial);
		}
	}
	values
}

/// The same terms with the alternating signs of a sine or cosine series, which
/// starts negative at the leading term of the polynomial in `r * r`.
fn alternating_factorials(first: u32, last: u32) -> Vec<f64> {
	reciprocal_factorials(first, 2, last).into_iter().enumerate().map(|(term, value)| if term % 2 == 0 { -value } else { value }).collect()
}

/// Deterministic implementations of the five transcendentals, evaluated in
/// double regardless of the arithmetic type. A backend library would be free to
/// choose its own range reduction, coefficients, and operation order, and two
/// backends that make different choices cannot produce the same bytes. Every
/// step below is an IEEE-754 operation with a single correctly rounded result,
/// so the sequence is fixed by the program.
///
/// Trigonometric argument reduction is the classic three-term Cody-Waite split,
/// which loses accuracy once the argument passes roughly 2^20. It loses the same
/// accuracy everywhere, which is what the byte-identity contract asks for.
fn shared_math(arithmetic: &str) -> String {
	let mut block = String::new();
	if arithmetic != "double" {
		block.push_str("declare double @llvm.floor.f64(double) declare double @llvm.fabs.f64(double) declare double @llvm.fma.f64(double, double, double)\n");
	}
	let infinity = constant(f64::INFINITY);
	let negative_infinity = constant(f64::NEG_INFINITY);
	let quiet = constant(f64::NAN);
	// Two half-exponents keep every intermediate normal, so no scale step rounds
	// through the subnormal range on the way to a normal answer.
	block.push_str(&exponential_math("double", "recipe.math.exp.wide", 1023, 2046, 52, 709.782712893383973096, -745.133219101941108420, 13));
	if arithmetic != "double" {
		block.push_str(&exponential_math(arithmetic, "recipe.math.exp.narrow", 127, 254, 23, 88.72283905206835, -103.972084045410, 8));
	}
	let (expm1, expm1_value) = horner("expm1.poly", "%x", &reciprocal_factorials(1, 1, 16));
	block.push_str(&format!("define internal double @recipe.math.expm1(double %x) #1 {{\nentry:\n%absolute = call double @llvm.fabs.f64(double %x)\n%small = fcmp olt double %absolute, {}\nbr i1 %small, label %series, label %general\nseries:\n{expm1}%result = fmul double %x, {expm1_value}\nret double %result\ngeneral:\n%value = call double @recipe.math.exp.wide(double %x)\n%shifted = fsub double %value, 1.0\nret double %shifted\n}}\n", constant(0.35)));
	// tanh(x) = u / (u + 2) for u = e^(2x) - 1. Neither the numerator nor the
	// denominator cancels, so small arguments keep their significance without a
	// separate polynomial branch.
	block.push_str(&format!("define internal double @recipe.math.tanh.wide(double %x) #1 {{\nentry:\n%unordered = fcmp uno double %x, %x\nbr i1 %unordered, label %quiet, label %range\nquiet:\nret double %x\nrange:\n%absolute = call double @llvm.fabs.f64(double %x)\n%saturated = fcmp oge double %absolute, {}\nbr i1 %saturated, label %unit, label %compute\nunit:\n%negative = fcmp olt double %x, 0.0\n%signed = select i1 %negative, double -1.0, double 1.0\nret double %signed\ncompute:\n%doubled = fmul double %x, 2.0\n%u = call double @recipe.math.expm1(double %doubled)\n%denominator = fadd double %u, 2.0\n%result = fdiv double %u, %denominator\nret double %result\n}}\n", constant(20.0)));
	let logarithm_terms = (0..12).map(|term| 1.0 / f64::from(2 * term + 1)).collect::<Vec<_>>();
	let (logarithm, logarithm_value) = horner("log.poly", "%z", &logarithm_terms);
	block.push_str(&format!("define internal double @recipe.math.log.wide(double %x) #1 {{\nentry:\n%unordered = fcmp uno double %x, %x\nbr i1 %unordered, label %quiet, label %sign.test\nquiet:\nret double %x\nsign.test:\n%negative = fcmp olt double %x, 0.0\nbr i1 %negative, label %invalid, label %zero.test\ninvalid:\nret double {quiet}\nzero.test:\n%zero = fcmp oeq double %x, 0.0\nbr i1 %zero, label %pole, label %infinite.test\npole:\nret double {negative_infinity}\ninfinite.test:\n%infinite = fcmp oeq double %x, {infinity}\nbr i1 %infinite, label %unbounded, label %normalize\nunbounded:\nret double {infinity}\nnormalize:\n%raw = bitcast double %x to i64\n%subnormal = icmp ult i64 %raw, 4503599627370496\n%boosted = fmul double %x, {}\n%source = select i1 %subnormal, double %boosted, double %x\n%correction = select i1 %subnormal, i32 -54, i32 0\n%bits = bitcast double %source to i64\n%field = lshr i64 %bits, 52\n%narrow = trunc i64 %field to i32\n%unbiased = sub i32 %narrow, 1023\n%exponent = add i32 %unbiased, %correction\n%fraction = and i64 %bits, 4503599627370495\n%unit = or i64 %fraction, 4607182418800017408\n%m = bitcast i64 %unit to double\n%large = fcmp ogt double %m, {}\n%halved = fmul double %m, 0.5\n%mantissa = select i1 %large, double %halved, double %m\n%bump = select i1 %large, i32 1, i32 0\n%e = add i32 %exponent, %bump\n%ef = sitofp i32 %e to double\n%numerator = fsub double %mantissa, 1.0\n%denominator = fadd double %mantissa, 1.0\n%s = fdiv double %numerator, %denominator\n%z = fmul double %s, %s\n{logarithm}%series = fmul double %s, {logarithm_value}\n%twice = fmul double %series, 2.0\n%upper = fmul double %ef, {}\n%lower = fmul double %ef, {}\n%tail = fadd double %twice, %lower\n%result = fadd double %upper, %tail\nret double %result\n}}\n",
		constant(f64::from_bits(0x4350000000000000)),
		constant(std::f64::consts::SQRT_2),
		constant(f64::from_bits(0x3FE62E42FEE00000)),
		constant(1.90821492927058770002e-10)));
	let (sine, sine_value) = horner("sin.poly", "%z", &alternating_factorials(3, 19));
	let (cosine, cosine_value) = horner("cos.poly", "%z", &alternating_factorials(2, 18));
	block.push_str(&format!("define internal double @recipe.math.trig(double %x, i32 %offset) #1 {{\nentry:\n%unordered = fcmp uno double %x, %x\n%absolute = call double @llvm.fabs.f64(double %x)\n%infinite = fcmp oeq double %absolute, {infinity}\n%invalid = or i1 %unordered, %infinite\nbr i1 %invalid, label %undefined, label %reduce\nundefined:\nret double {quiet}\nreduce:\n%scaled = fmul double %x, {}\n%shifted = fadd double %scaled, 0.5\n%kf = call double @llvm.floor.f64(double %shifted)\n%first = fmul double %kf, {}\n%partial = fsub double %x, %first\n%second = fmul double %kf, {}\n%closer = fsub double %partial, %second\n%third = fmul double %kf, {}\n%r = fsub double %closer, %third\n%quarter = fmul double %kf, 0.25\n%whole = call double @llvm.floor.f64(double %quarter)\n%rounds = fmul double %whole, 4.0\n%residue = fsub double %kf, %rounds\n%k = fptosi double %residue to i32\n%shift = add i32 %k, %offset\n%quadrant = and i32 %shift, 3\n%z = fmul double %r, %r\n{sine}%sin.body = fmul double %z, {sine_value}\n%sin.scaled = fmul double %r, %sin.body\n%sin = fadd double %r, %sin.scaled\n{cosine}%cos.body = fmul double %z, {cosine_value}\n%cos = fadd double 1.0, %cos.body\n%parity = and i32 %quadrant, 1\n%cosine = icmp eq i32 %parity, 1\n%magnitude = select i1 %cosine, double %cos, double %sin\n%upper = icmp uge i32 %quadrant, 2\n%negated = fneg double %magnitude\n%result = select i1 %upper, double %negated, double %magnitude\nret double %result\n}}\n",
		constant(std::f64::consts::FRAC_2_PI),
		constant(f64::from_bits(0x3FF921FB54400000)),
		constant(f64::from_bits(0x3DD0B4611A600000)),
		constant(f64::from_bits(0x3BA3198A2E000000))));
	block.push_str("define internal double @recipe.math.sin.wide(double %x) #1 { entry: %result = call double @recipe.math.trig(double %x, i32 0) ret double %result }\ndefine internal double @recipe.math.cos.wide(double %x) #1 { entry: %result = call double @recipe.math.trig(double %x, i32 1) ret double %result }\n");
	for name in MATH_NAMES {
		let wide = format!("{name}.wide");
		if arithmetic == "double" {
			block.push_str(&format!("define internal double @{name}(double %value) #1 {{ entry: %result = call double @{wide}(double %value) ret double %result }}\n"));
		} else if name == "recipe.math.exp" {
			block.push_str(&format!(
				"define internal {arithmetic} @{name}({arithmetic} %value) #1 {{ entry: %result = call {arithmetic} @recipe.math.exp.narrow({arithmetic} %value) ret {arithmetic} %result }}\n"
			));
		} else {
			// Evaluating in double and rounding once is both deterministic and at
			// least as accurate as a native narrow evaluation would be.
			block.push_str(&format!("define internal {arithmetic} @{name}({arithmetic} %value) #1 {{ entry: %wide = fpext {arithmetic} %value to double %computed = call double @{wide}(double %wide) %result = fptrunc double %computed to {arithmetic} ret {arithmetic} %result }}\n"));
		}
	}
	block
}
fn native_codec(ty: &str, rounded: bool) -> String {
	let round = if rounded {
		"define internal float @recipe.round(float %value) #1 { entry: %bits = bitcast float %value to i32 %absolute = and i32 %bits, 2147483647 %special = icmp uge i32 %absolute, 2139095040 %shifted = lshr i32 %bits, 13 %least = and i32 %shifted, 1 %bias = add i32 4095, %least %biased = add i32 %bits, %bias %masked = and i32 %biased, -8192 %encoded = select i1 %special, i32 %bits, i32 %masked %result = bitcast i32 %encoded to float ret float %result }\n"
	} else {
		""
	};
	let conversion = if rounded { format!("%result = call {ty} @recipe.round({ty} %value)") } else { format!("ret {ty} %value") };
	let returned = if rounded { format!("ret {ty} %result") } else { String::new() };
	format!(
		"{round}define internal {ty} @recipe.decode({ty} %value) #1 {{ entry: {conversion} {returned} }}\ndefine internal {ty} @recipe.encode({ty} %value) #1 {{ entry: {conversion} {returned} }}\n"
	)
}

fn numeric_operations(prefix: &str, value: &str, arithmetic: &str, encoded: bool, vector: Option<bool>) -> String {
	let intrinsic = if arithmetic == "double" { "f64" } else { "f32" };
	let math = MATH_NAMES;
	let mut block = String::new();
	for (name, operation) in [("add", "fadd"), ("sub", "fsub"), ("mul", "fmul"), ("div", "fdiv")] {
		if encoded {
			block.push_str(&format!("define internal {value} @{prefix}.{name}({value} %left, {value} %right) #1 {{ entry: %left.wide = call {arithmetic} @recipe.decode({value} %left) %right.wide = call {arithmetic} @recipe.decode({value} %right) %wide = {operation} {arithmetic} %left.wide, %right.wide %result = call {value} @recipe.encode({arithmetic} %wide) ret {value} %result }}\n"))
		} else {
			block.push_str(&format!("define internal {value} @{prefix}.{name}({value} %left, {value} %right) #1 {{ entry: %result = {operation} {value} %left, %right ret {value} %result }}\n"))
		}
	}
	if encoded {
		block.push_str(&format!("define internal {value} @{prefix}.madd({value} %sum, {value} %left, {value} %right) #1 {{ entry: %sum.wide = call {arithmetic} @recipe.decode({value} %sum) %left.wide = call {arithmetic} @recipe.decode({value} %left) %right.wide = call {arithmetic} @recipe.decode({value} %right) %wide = call {arithmetic} @llvm.fma.{intrinsic}({arithmetic} %left.wide, {arithmetic} %right.wide, {arithmetic} %sum.wide) %result = call {value} @recipe.encode({arithmetic} %wide) ret {value} %result }}\n"));
	} else {
		block.push_str(&format!("define internal {value} @{prefix}.madd({value} %sum, {value} %left, {value} %right) #1 {{ entry: %result = call {value} @llvm.fma.{intrinsic}({value} %left, {value} %right, {value} %sum) ret {value} %result }}\n"));
	}
	if let Some(declare) = vector {
		if let Some(intrinsic) = match value {
			"half" => Some("f16"),
			"float" => Some("f32"),
			"double" => Some("f64"),
			_ => None,
		} {
			// The state family reuses the model family's intrinsic when both name the
			// same type, so the declaration is emitted once.
			if declare {
				block.push_str(&format!(
					"declare <RECIPE_REGISTER_M x {value}> @llvm.fma.vRECIPE_REGISTER_M{intrinsic}(<RECIPE_REGISTER_M x {value}>, <RECIPE_REGISTER_M x {value}>, <RECIPE_REGISTER_M x {value}>)\n"
				));
			}
			block.push_str(&format!("define internal <RECIPE_REGISTER_M x {value}> @{prefix}.madd.vector(<RECIPE_REGISTER_M x {value}> %sum, <RECIPE_REGISTER_M x {value}> %left, <RECIPE_REGISTER_M x {value}> %right) #1 {{\nentry:\n%result = call <RECIPE_REGISTER_M x {value}> @llvm.fma.vRECIPE_REGISTER_M{intrinsic}(<RECIPE_REGISTER_M x {value}> %left, <RECIPE_REGISTER_M x {value}> %right, <RECIPE_REGISTER_M x {value}> %sum)\nret <RECIPE_REGISTER_M x {value}> %result\n}}\n"));
		} else {
			block.push_str(&format!("define internal <RECIPE_REGISTER_M x {value}> @{prefix}.madd.vector(<RECIPE_REGISTER_M x {value}> %sum, <RECIPE_REGISTER_M x {value}> %left, <RECIPE_REGISTER_M x {value}> %right) #1 {{\nentry:\nbr label %loop\nloop:\n%p = phi i32 [ 0, %entry ], [ %p.next, %step ]\n%result = phi <RECIPE_REGISTER_M x {value}> [ poison, %entry ], [ %next, %step ]\n%more = icmp ult i32 %p, RECIPE_REGISTER_M\nbr i1 %more, label %step, label %done\nstep:\n%sum.value = extractelement <RECIPE_REGISTER_M x {value}> %sum, i32 %p\n%left.value = extractelement <RECIPE_REGISTER_M x {value}> %left, i32 %p\n%right.value = extractelement <RECIPE_REGISTER_M x {value}> %right, i32 %p\n%value = call {value} @{prefix}.madd({value} %sum.value, {value} %left.value, {value} %right.value)\n%next = insertelement <RECIPE_REGISTER_M x {value}> %result, {value} %value, i32 %p\n%p.next = add i32 %p, 1\nbr label %loop\ndone:\nret <RECIPE_REGISTER_M x {value}> %result\n}}\n"));
		}
	}
	if encoded {
		block.push_str(&format!("define internal {value} @{prefix}.neg({value} %value) #1 {{ entry: %wide = call {arithmetic} @recipe.decode({value} %value) %negative = fneg {arithmetic} %wide %result = call {value} @recipe.encode({arithmetic} %negative) ret {value} %result }}\n"));
	} else {
		block.push_str(&format!("define internal {value} @{prefix}.neg({value} %value) #1 {{ entry: %result = fneg {value} %value ret {value} %result }}\n"));
	}
	for predicate in ["oeq", "oge", "ogt", "ole", "olt", "one", "ord"] {
		if encoded {
			block.push_str(&format!("define internal i1 @{prefix}.{predicate}({value} %left, {value} %right) #1 {{ entry: %left.wide = call {arithmetic} @recipe.decode({value} %left) %right.wide = call {arithmetic} @recipe.decode({value} %right) %result = fcmp {predicate} {arithmetic} %left.wide, %right.wide ret i1 %result }}\n"))
		} else {
			block.push_str(&format!("define internal i1 @{prefix}.{predicate}({value} %left, {value} %right) #1 {{ entry: %result = fcmp {predicate} {value} %left, %right ret i1 %result }}\n"))
		}
	}
	for (name, operation) in [("from.u1", "uitofp i1"), ("from.u32", "uitofp i32"), ("from.s32", "sitofp i32")] {
		let source = if name == "from.u1" { "i1" } else { "i32" };
		if encoded {
			block.push_str(&format!("define internal {value} @{prefix}.{name}({source} %value) #1 {{ entry: %wide = {operation} %value to {arithmetic} %result = call {value} @recipe.encode({arithmetic} %wide) ret {value} %result }}\n"))
		} else {
			block.push_str(&format!("define internal {value} @{prefix}.{name}({source} %value) #1 {{ entry: %result = {operation} %value to {value} ret {value} %result }}\n"))
		}
	}
	for (name, operation) in [("to.u32", "fptoui"), ("to.s32", "fptosi")] {
		if encoded {
			block.push_str(&format!(
				"define internal i32 @{prefix}.{name}({value} %value) #1 {{ entry: %wide = call {arithmetic} @recipe.decode({value} %value) %result = {operation} {arithmetic} %wide to i32 ret i32 %result }}\n"
			))
		} else {
			block.push_str(&format!("define internal i32 @{prefix}.{name}({value} %value) #1 {{ entry: %result = {operation} {value} %value to i32 ret i32 %result }}\n"))
		}
	}
	let from_f32 = if arithmetic == "double" { format!("%wide = fpext float %value to double") } else { format!("%wide = fadd float %value, 0.0") };
	let from_f16 = format!("%wide = fpext half %value to {arithmetic}");
	let to_f16 = format!("%result = fptrunc {arithmetic} %wide to half");
	for (name, source, conversion) in [("from.f32", "float", from_f32), ("from.f16", "half", from_f16)] {
		if encoded {
			block.push_str(&format!(
				"define internal {value} @{prefix}.{name}({source} %value) #1 {{ entry: {conversion} %result = call {value} @recipe.encode({arithmetic} %wide) ret {value} %result }}\n"
			))
		} else {
			block.push_str(&format!("define internal {value} @{prefix}.{name}({source} %value) #1 {{ entry: {conversion} ret {value} %wide }}\n"))
		}
	}
	if encoded {
		block.push_str(&format!("define internal half @{prefix}.to.f16({value} %value) #1 {{ entry: %wide = call {arithmetic} @recipe.decode({value} %value) {to_f16} ret half %result }}\n"));
	} else {
		block.push_str(&format!("define internal half @{prefix}.to.f16({value} %value) #1 {{ entry: %wide = fadd {value} %value, 0.0 {to_f16} ret half %result }}\n"));
	}
	for (name, symbol) in [
		("abs", format!("llvm.fabs.{intrinsic}")),
		("floor", format!("llvm.floor.{intrinsic}")),
		("sqrt", format!("llvm.sqrt.{intrinsic}")),
		("exp", math[0].to_owned()),
		("tanh", math[1].to_owned()),
		("cos", math[2].to_owned()),
		("sin", math[3].to_owned()),
		("log", math[4].to_owned()),
	] {
		if encoded {
			block.push_str(&format!("define internal {value} @{prefix}.{name}({value} %value) #1 {{ entry: %wide = call {arithmetic} @recipe.decode({value} %value) %computed = call {arithmetic} @{symbol}({arithmetic} %wide) %result = call {value} @recipe.encode({arithmetic} %computed) ret {value} %result }}\n"))
		} else {
			block.push_str(&format!("define internal {value} @{prefix}.{name}({value} %value) #1 {{ entry: %result = call {value} @{symbol}({value} %value) ret {value} %result }}\n"))
		}
	}
	block.push_str(&format!("define internal {value} @{prefix}.sigmoid({value} %value) #1 {{ entry: %negative = call {value} @{prefix}.neg({value} %value) %exponential = call {value} @{prefix}.exp({value} %negative) %one = call {value} @{prefix}.from.u1(i1 true) %denominator = call {value} @{prefix}.add({value} %exponential, {value} %one) %result = call {value} @{prefix}.div({value} %one, {value} %denominator) ret {value} %result }}\n"));
	block
}

fn numeric_program(value: &str, arithmetic: &str, codec: &str) -> String {
	let intrinsic = if arithmetic == "double" { "f64" } else { "f32" };
	let mut block = if codec.starts_with("; NUMERIC BEGIN") {
		format!("{codec}\n")
	} else {
		format!(
			"; NUMERIC BEGIN\ndeclare {arithmetic} @llvm.sqrt.{intrinsic}({arithmetic}) declare {arithmetic} @llvm.fabs.{intrinsic}({arithmetic}) declare {arithmetic} @llvm.floor.{intrinsic}({arithmetic})\n{codec}\n"
		)
	};
	block.push_str(&format!("\ndeclare {arithmetic} @llvm.fma.{intrinsic}({arithmetic}, {arithmetic}, {arithmetic})\n"));
	block.push_str(&shared_math(arithmetic));
	block.push_str(&numeric_operations("recipe", value, arithmetic, true, Some(true)));
	// No floating-point atomic is emitted. Every reduction names its own owner and
	// a fixed order, so a read-modify-write race has nowhere left to happen.
	if !codec.contains("@recipe.set.format") {
		block.push_str("define internal void @recipe.set.format(i32 %exp, i32 %man) #1 { entry: ret void }\n")
	}
	block.push_str(&numeric_operations("recipe.state", arithmetic, arithmetic, false, Some(value != arithmetic)));
	block.push_str(&format!("define internal {arithmetic} @recipe.state.from.model({value} %value) #1 {{ entry: %result = call {arithmetic} @recipe.decode({value} %value) ret {arithmetic} %result }}\ndefine internal {value} @recipe.model.from.state({arithmetic} %value) #1 {{ entry: %result = call {value} @recipe.encode({arithmetic} %value) ret {value} %result }}\n; NUMERIC END"));
	block
}

fn native_ir(ir: String, suffix: &str, llvm: &str, format: FloatFormat) -> BuildResult<String> {
	let (start, end) = numeric_region(&ir)?;
	let bits = format.storage.bits();
	let numeric = numeric_program(llvm, llvm, &native_codec(llvm, format == FloatFormat::TF32));
	let mut kernel = format!("{}@RECIPE_NUMERIC@{}", &ir[..start], &ir[end..]);
	kernel = word(kernel, "double", llvm).replace("@contraction_tile", &format!("@contraction_tile{suffix}")).replace("align 8", &format!("align {}", bits / 8));
	// The arithmetic type is the model type here, so the widening accumulator folds away.
	kernel = kernel.replace("RECIPE_STATE_ALIGN", &(bits / 8).to_string()).replace("RECIPE_STATE", llvm);
	if bits < 64 {
		let literal = |value: f64| match llvm {
			"half" => format!("0xH{:04X}", format.pack(value)),
			"bfloat" => format!("0xR{:04X}", format.pack(value)),
			_ => format!("0x{:016X}", format.unpack(format.pack(value)).to_bits()),
		};
		kernel = kernel
			.replace(&format!("{llvm} 0.1"), &format!("{llvm} {}", literal(0.1)))
			.replace("0x3CB0000000000000", &literal(f64::from_bits(0x3CB0000000000000)))
			.replace("0x3FEFFFFFFFFFFFFE", &literal(f64::from_bits(0x3FEFFFFFFFFFFFFE)))
	}
	Ok(kernel.replace("@RECIPE_NUMERIC@", &numeric))
}
fn custom_numeric() -> String {
	let mut block = String::from(
		r#"; NUMERIC BEGIN
@recipe_f_exp = internal addrspace(3) global i32 undef, align 4
@recipe_f_man = internal addrspace(3) global i32 undef, align 4
declare double @llvm.sqrt.f64(double)
declare double @llvm.fabs.f64(double)
declare double @llvm.floor.f64(double)
declare i64 @llvm.ctlz.i64(i64, i1)
declare double @llvm.roundeven.f64(double)
define internal void @recipe.set.format(i32 %exp, i32 %man) #1 { entry: store atomic i32 %exp, ptr addrspace(3) @recipe_f_exp monotonic, align 4 store atomic i32 %man, ptr addrspace(3) @recipe_f_man monotonic, align 4 ret void }
define internal double @recipe.f.power(i64 %exponent) #3 { entry: %high = icmp sgt i64 %exponent, 1023 br i1 %high, label %infinity, label %low.test infinity: ret double 0x7FF0000000000000 low.test: %low = icmp slt i64 %exponent, -1074 br i1 %low, label %zero, label %finite zero: ret double 0.0 finite: %normal = icmp sge i64 %exponent, -1022 br i1 %normal, label %power.normal, label %power.subnormal power.normal: %biased = add i64 %exponent, 1023 %normal.bits = shl i64 %biased, 52 %normal.result = bitcast i64 %normal.bits to double ret double %normal.result power.subnormal: %shift = add i64 %exponent, 1074 %subnormal.bits = shl i64 1, %shift %subnormal.result = bitcast i64 %subnormal.bits to double ret double %subnormal.result }
define internal double @recipe.round(double %value) #3 { entry: %source = bitcast double %value to i64 %sign.source = lshr i64 %source, 63 %absolute.bits = and i64 %source, 9223372036854775807 %absolute = bitcast i64 %absolute.bits to double %exp.word = load atomic i32, ptr addrspace(3) @recipe_f_exp monotonic, align 4 %man.word = load atomic i32, ptr addrspace(3) @recipe_f_man monotonic, align 4 %exp = zext i32 %exp.word to i64 %man = zext i32 %man.word to i64 %total = add i64 %exp, %man %sign = shl i64 %sign.source, %total %exp.shift = sub i64 %exp, 1 %bias.one = shl i64 1, %exp.shift %bias = sub i64 %bias.one, 1 %exponent.one = shl i64 1, %exp %exponent.limit = sub i64 %exponent.one, 1 %mantissa.limit = shl i64 1, %man %nan = fcmp uno double %value, %value br i1 %nan, label %encode.nan, label %infinite.test encode.nan: %quiet.shift = sub i64 %man, 1 %quiet = shl i64 1, %quiet.shift %special.exponent = shl i64 %exponent.limit, %man %nan.base = or i64 %sign, %special.exponent %nan.bits = or i64 %nan.base, %quiet %nan.result = call double @recipe.f.decode(i64 %nan.bits) ret double %nan.result infinite.test: %infinite = fcmp oeq double %absolute, 0x7FF0000000000000 br i1 %infinite, label %encode.infinity, label %zero.test encode.infinity: %infinity.exponent = shl i64 %exponent.limit, %man %infinity.bits = or i64 %sign, %infinity.exponent %infinity.result = call double @recipe.f.decode(i64 %infinity.bits) ret double %infinity.result zero.test: %zero = fcmp oeq double %absolute, 0.0 br i1 %zero, label %encode.zero, label %finite encode.zero: %zero.bits = shl i64 %sign.source, 63 %zero.result = bitcast i64 %zero.bits to double ret double %zero.result finite: %minimum = sub i64 1, %bias %source.exponent.shifted = lshr i64 %absolute.bits, 52 %source.exponent = and i64 %source.exponent.shifted, 2047 %source.mantissa = and i64 %absolute.bits, 4503599627370495 %source.normal = icmp ne i64 %source.exponent, 0 br i1 %source.normal, label %source.normal.exponent, label %source.subnormal.exponent source.normal.exponent: %normal.unbiased = sub i64 %source.exponent, 1023 br label %source.exponent.ready source.subnormal.exponent: %leading.zeros = call i64 @llvm.ctlz.i64(i64 %source.mantissa, i1 false) %highest = sub i64 63, %leading.zeros %subnormal.unbiased = sub i64 %highest, 1074 br label %source.exponent.ready source.exponent.ready: %unbiased = phi i64 [ %normal.unbiased, %source.normal.exponent ], [ %subnormal.unbiased, %source.subnormal.exponent ] %subnormal = icmp slt i64 %unbiased, %minimum br i1 %subnormal, label %encode.subnormal, label %encode.normal encode.subnormal: %minimum.power = call double @recipe.f.power(i64 %minimum) %subnormal.ratio = fdiv double %absolute, %minimum.power %subnormal.scaled = uitofp i64 %mantissa.limit to double %subnormal.value = fmul double %subnormal.ratio, %subnormal.scaled %subnormal.rounded = call double @llvm.roundeven.f64(double %subnormal.value) %subnormal.mantissa = fptoui double %subnormal.rounded to i64 %subnormal.carry = icmp eq i64 %subnormal.mantissa, %mantissa.limit %subnormal.encoded = select i1 %subnormal.carry, i64 %mantissa.limit, i64 %subnormal.mantissa %subnormal.bits = or i64 %sign, %subnormal.encoded %subnormal.result = call double @recipe.f.decode(i64 %subnormal.bits) ret double %subnormal.result encode.normal: %power = call double @recipe.f.power(i64 %unbiased) %ratio = fdiv double %absolute, %power %fraction = fsub double %ratio, 1.0 %mantissa.scale = uitofp i64 %mantissa.limit to double %mantissa.value = fmul double %fraction, %mantissa.scale %mantissa.rounded = call double @llvm.roundeven.f64(double %mantissa.value) %mantissa.initial = fptoui double %mantissa.rounded to i64 %carry = icmp eq i64 %mantissa.initial, %mantissa.limit %carry.value = zext i1 %carry to i64 %final.unbiased = add i64 %unbiased, %carry.value %mantissa = select i1 %carry, i64 0, i64 %mantissa.initial %stored = add i64 %final.unbiased, %bias %overflow = icmp sge i64 %stored, %exponent.limit br i1 %overflow, label %encode.infinity, label %pack pack: %stored.bits = shl i64 %stored, %man %normal.base = or i64 %sign, %stored.bits %normal.bits = or i64 %normal.base, %mantissa %normal.result = call double @recipe.f.decode(i64 %normal.bits) ret double %normal.result }
define internal double @recipe.f.decode(i64 %bits) #3 { entry: %exp.word = load atomic i32, ptr addrspace(3) @recipe_f_exp monotonic, align 4 %man.word = load atomic i32, ptr addrspace(3) @recipe_f_man monotonic, align 4 %exp = zext i32 %exp.word to i64 %man = zext i32 %man.word to i64 %total = add i64 %exp, %man %negative.bit = lshr i64 %bits, %total %negative = icmp ne i64 %negative.bit, 0 %exp.one = shl i64 1, %exp %exp.limit = sub i64 %exp.one, 1 %man.limit = shl i64 1, %man %shifted = lshr i64 %bits, %man %exponent = and i64 %shifted, %exp.limit %man.mask = sub i64 %man.limit, 1 %mantissa = and i64 %bits, %man.mask %special = icmp eq i64 %exponent, %exp.limit br i1 %special, label %decode.special, label %finite decode.special: %infinity = icmp eq i64 %mantissa, 0 %special.value = select i1 %infinity, double 0x7FF0000000000000, double 0x7FF8000000000000 br label %signed finite: %zero.exp = icmp eq i64 %exponent, 0 br i1 %zero.exp, label %decode.subnormal, label %decode.normal decode.subnormal: %zero.man = icmp eq i64 %mantissa, 0 br i1 %zero.man, label %decode.zero, label %subnormal decode.zero: br label %signed subnormal: %bias.shift = sub i64 %exp, 1 %bias.one = shl i64 1, %bias.shift %bias = sub i64 %bias.one, 1 %subnormal.exponent = sub i64 1, %bias %subnormal.power = call double @recipe.f.power(i64 %subnormal.exponent) %subnormal.man = uitofp i64 %mantissa to double %subnormal.limit = uitofp i64 %man.limit to double %subnormal.fraction = fdiv double %subnormal.man, %subnormal.limit %subnormal.value = fmul double %subnormal.power, %subnormal.fraction br label %signed decode.normal: %normal.bias.shift = sub i64 %exp, 1 %normal.bias.one = shl i64 1, %normal.bias.shift %normal.bias = sub i64 %normal.bias.one, 1 %normal.exponent = sub i64 %exponent, %normal.bias %normal.power = call double @recipe.f.power(i64 %normal.exponent) %normal.man = uitofp i64 %mantissa to double %normal.limit = uitofp i64 %man.limit to double %normal.fraction = fdiv double %normal.man, %normal.limit %normal.significand = fadd double 1.0, %normal.fraction %normal.value = fmul double %normal.power, %normal.significand br label %signed signed: %magnitude = phi double [ %special.value, %decode.special ], [ 0.0, %decode.zero ], [ %subnormal.value, %subnormal ], [ %normal.value, %decode.normal ] %negated = fneg double %magnitude %result = select i1 %negative, double %negated, double %magnitude ret double %result }
"#,
	);
	block.push_str("define internal double @recipe.decode(double %value) #1 { entry: %result = call double @recipe.round(double %value) ret double %result }\ndefine internal double @recipe.encode(double %value) #1 { entry: %result = call double @recipe.round(double %value) ret double %result }\n");
	numeric_program("double", "double", &block)
}
fn custom_ir(ir: String, suffix: &str) -> BuildResult<String> {
	let (start, end) = numeric_region(&ir)?;
	// The custom float carries its values in double, so the arithmetic type is
	// the model type and the widening accumulator folds away.
	Ok(format!("{}@RECIPE_NUMERIC@{}", &ir[..start], &ir[end..])
		.replace("@contraction_tile", &format!("@contraction_tile{suffix}"))
		.replace("RECIPE_STATE_ALIGN", "8")
		.replace("RECIPE_STATE", "double")
		.replace("@RECIPE_NUMERIC@", &custom_numeric()))
}
fn fp8_codec() -> &'static str {
	// The codec stays out of line. It is the one narrow-float body that branches,
	// so inlining it at every arithmetic site multiplies the kernel.
	r#"define internal float @recipe.decode(i8 %value) #3 { entry: %wide = zext i8 %value to i32 %sign = and i32 %wide, 128 %exponent.shifted = lshr i32 %wide, 2 %exponent = and i32 %exponent.shifted, 31 %mantissa = and i32 %wide, 3 %zero.exponent = icmp eq i32 %exponent, 0 br i1 %zero.exponent, label %subnormal, label %nonzero subnormal: %mantissa.float = uitofp i32 %mantissa to float %negative = icmp ne i32 %sign, 0 %subnormal.value = select i1 %negative, float 0xBEF0000000000000, float 0x3EF0000000000000 %scaled = fmul float %mantissa.float, %subnormal.value ret float %scaled nonzero: %special = icmp eq i32 %exponent, 31 %biased.exponent = add i32 %exponent, 112 %float.exponent = select i1 %special, i32 255, i32 %biased.exponent %float.sign = shl i32 %sign, 24 %float.exponent.bits = shl i32 %float.exponent, 23 %float.mantissa = shl i32 %mantissa, 21 %signed = or i32 %float.sign, %float.exponent.bits %bits = or i32 %signed, %float.mantissa %result = bitcast i32 %bits to float ret float %result }
define internal i8 @recipe.encode(float %value) #3 { entry: %bits = bitcast float %value to i32 %sign.shifted = lshr i32 %bits, 24 %sign = and i32 %sign.shifted, 128 %absolute = and i32 %bits, 2147483647 %exponent.shifted = lshr i32 %absolute, 23 %exponent = and i32 %exponent.shifted, 255 %mantissa = and i32 %absolute, 8388607 %special = icmp eq i32 %exponent, 255 br i1 %special, label %encode.special, label %finite encode.special: %nan = icmp ne i32 %mantissa, 0 %special.mantissa = select i1 %nan, i32 2, i32 0 %special.base = or i32 %sign, 124 %special.bits = or i32 %special.base, %special.mantissa %special.result = trunc i32 %special.bits to i8 ret i8 %special.result finite: %zero = icmp eq i32 %exponent, 0 br i1 %zero, label %encode.zero, label %range encode.zero: %zero.result = trunc i32 %sign to i8 ret i8 %zero.result range: %unbiased = sub i32 %exponent, 127 %overflow = icmp sgt i32 %unbiased, 15 br i1 %overflow, label %encode.infinity, label %normal.test encode.infinity: %infinity.bits = or i32 %sign, 124 %infinity.result = trunc i32 %infinity.bits to i8 ret i8 %infinity.result normal.test: %normal = icmp sge i32 %unbiased, -14 br i1 %normal, label %encode.normal, label %subnormal.test encode.normal: %stored = add i32 %unbiased, 15 %top = lshr i32 %mantissa, 21 %remainder = and i32 %mantissa, 2097151 %above = icmp ugt i32 %remainder, 1048576 %tie = icmp eq i32 %remainder, 1048576 %odd.bit = and i32 %top, 1 %odd = icmp ne i32 %odd.bit, 0 %tie.odd = and i1 %tie, %odd %round = or i1 %above, %tie.odd %increment = zext i1 %round to i32 %rounded = add i32 %top, %increment %carry = lshr i32 %rounded, 2 %final.exponent = add i32 %stored, %carry %rounded.overflow = icmp uge i32 %final.exponent, 31 br i1 %rounded.overflow, label %encode.infinity, label %normal.pack normal.pack: %final.mantissa = and i32 %rounded, 3 %exponent.bits = shl i32 %final.exponent, 2 %normal.base = or i32 %sign, %exponent.bits %normal.bits = or i32 %normal.base, %final.mantissa %normal.result = trunc i32 %normal.bits to i8 ret i8 %normal.result subnormal.test: %too.small = icmp slt i32 %unbiased, -17 br i1 %too.small, label %encode.zero, label %encode.subnormal encode.subnormal: %significand = or i32 %mantissa, 8388608 %shift = sub i32 7, %unbiased %subnormal.top = lshr i32 %significand, %shift %one = shl i32 1, %shift %mask = sub i32 %one, 1 %subnormal.remainder = and i32 %significand, %mask %half.shift = sub i32 %shift, 1 %half = shl i32 1, %half.shift %subnormal.above = icmp ugt i32 %subnormal.remainder, %half %subnormal.tie = icmp eq i32 %subnormal.remainder, %half %subnormal.odd.bit = and i32 %subnormal.top, 1 %subnormal.odd = icmp ne i32 %subnormal.odd.bit, 0 %subnormal.tie.odd = and i1 %subnormal.tie, %subnormal.odd %subnormal.round = or i1 %subnormal.above, %subnormal.tie.odd %subnormal.increment = zext i1 %subnormal.round to i32 %subnormal.mantissa = add i32 %subnormal.top, %subnormal.increment %subnormal.bits = or i32 %sign, %subnormal.mantissa %subnormal.result = trunc i32 %subnormal.bits to i8 ret i8 %subnormal.result }"#
}
fn bf16_codec() -> &'static str {
	r#"define internal float @recipe.decode(i16 %value) #1 { entry: %wide = zext i16 %value to i32 %bits = shl i32 %wide, 16 %result = bitcast i32 %bits to float ret float %result }
define internal i16 @recipe.encode(float %value) #1 { entry: %bits = bitcast float %value to i32 %absolute = and i32 %bits, 2147483647 %special = icmp uge i32 %absolute, 2139095040 %upper = lshr i32 %bits, 16 %mantissa = and i32 %absolute, 8388607 %nan = icmp ne i32 %mantissa, 0 %quiet = or i32 %upper, 64 %special.bits = select i1 %nan, i32 %quiet, i32 %upper %lower = and i32 %bits, 65535 %above = icmp ugt i32 %lower, 32768 %tie = icmp eq i32 %lower, 32768 %odd.bit = and i32 %upper, 1 %odd = icmp ne i32 %odd.bit, 0 %tie.odd = and i1 %tie, %odd %round = or i1 %above, %tie.odd %increment = zext i1 %round to i32 %rounded = add i32 %upper, %increment %encoded = select i1 %special, i32 %special.bits, i32 %rounded %result = trunc i32 %encoded to i16 ret i16 %result }"#
}
fn encoded_ir(ir: String, suffix: &str, bytes: usize, codec: &str, pack: impl Fn(f64) -> u64) -> BuildResult<String> {
	let (start, end) = numeric_region(&ir)?;
	let llvm = match bytes {
		1 => "i8",
		2 => "i16",
		4 => "i32",
		_ => "i64",
	};
	let numeric = numeric_program(llvm, "float", codec);
	let mut kernel = word(format!("{}@RECIPE_NUMERIC@{}", &ir[..start], &ir[end..]), "double", llvm)
		.replace("@contraction_tile", &format!("@contraction_tile{suffix}"))
		.replace("align 8", &format!("align {bytes}"));
	kernel = kernel.replace("RECIPE_STATE_ALIGN", "4").replace("RECIPE_STATE", "float");
	for (source, value) in [("-2.0", -2.0), ("-1.0", -1.0), ("0.0", 0.0), ("0.1", 0.1), ("0.5", 0.5), ("1.0", 1.0), ("2.0", 2.0)] {
		kernel = word(kernel, source, &pack(value).to_string())
	}
	for bits in [0x3CB0000000000000, 0x3FEFFFFFFFFFFFFE, 0xFFF0000000000000, 0x7FF8000000000000] {
		kernel = kernel.replace(&format!("0x{bits:016X}"), &format!("{}", pack(f64::from_bits(bits))))
	}
	Ok(kernel.replace("@RECIPE_NUMERIC@", &numeric))
}
fn half_ir(ir: String) -> BuildResult<String> {
	let (start, end) = numeric_region(&ir)?;
	let codec = "define internal float @recipe.decode(half %value) #1 { entry: %result = fpext half %value to float ret float %result }\ndefine internal half @recipe.encode(float %value) #1 { entry: %result = fptrunc float %value to half ret half %result }";
	let numeric = numeric_program("half", "float", codec);
	let mut kernel = word(format!("{}@RECIPE_NUMERIC@{}", &ir[..start], &ir[end..]), "double", "half").replace("@contraction_tile", "@contraction_tile_f16").replace("align 8", "align 2");
	kernel = kernel.replace("RECIPE_STATE_ALIGN", "4").replace("RECIPE_STATE", "float");
	for (source, value) in [("-2.0", -2.0), ("-1.0", -1.0), ("0.0", 0.0), ("0.1", 0.1), ("0.5", 0.5), ("1.0", 1.0), ("2.0", 2.0)] {
		kernel = word(kernel, source, &format!("0xH{:04X}", FloatFormat::FP16.pack(value)))
	}
	for bits in [0x3CB0000000000000, 0x3FEFFFFFFFFFFFFE, 0xFFF0000000000000, 0x7FF8000000000000] {
		kernel = kernel.replace(&format!("0x{bits:016X}"), &format!("0xH{:04X}", FloatFormat::FP16.pack(f64::from_bits(bits))))
	}
	Ok(kernel.replace("@RECIPE_NUMERIC@", &numeric))
}
fn int_codec(format: IntFormat) -> String {
	let (bits, mask) = (format.bits, (1u16 << format.bits) - 1);
	let shift = 32 - bits;
	let minimum = -(1i16 << (bits - 1));
	let maximum = (1i16 << (bits - 1)) - 1;
	format!(
		"declare float @llvm.roundeven.f32(float)\ndefine internal float @recipe.decode(i8 %value) #1 {{ entry: %wide = zext i8 %value to i32 %masked = and i32 %wide, {mask} %shifted = shl i32 %masked, {shift} %signed = ashr i32 %shifted, {shift} %result = sitofp i32 %signed to float ret float %result }}\ndefine internal i8 @recipe.encode(float %value) #1 {{ entry: %nan = fcmp uno float %value, %value %rounded = call float @llvm.roundeven.f32(float %value) %below = fcmp olt float %rounded, {minimum}.0 %above = fcmp ogt float %rounded, {maximum}.0 %lowered = select i1 %below, float {minimum}.0, float %rounded %clamped = select i1 %above, float {maximum}.0, float %lowered %finite = select i1 %nan, float 0.0, float %clamped %integer = fptosi float %finite to i32 %masked = and i32 %integer, {mask} %result = trunc i32 %masked to i8 ret i8 %result }}"
	)
}
fn setting<'a>(manifest: &'a str, key: &str) -> BuildResult<&'a str> {
	let prefix = format!("{key} = ");
	manifest.lines().find_map(|line| line.trim().strip_prefix(&prefix)).ok_or_else(|| io::Error::other(format!("{key} must be configured")).into())
}
fn number<'a>(manifest: &'a str, key: &str) -> BuildResult<&'a str> {
	let value = setting(manifest, key)?;
	value.parse::<f64>().map_err(|error| io::Error::other(format!("{key} must be numeric: {error}")))?;
	Ok(value)
}
fn text<'a>(manifest: &'a str, key: &str) -> BuildResult<&'a str> {
	setting(manifest, key)?.strip_prefix('"').and_then(|value| value.strip_suffix('"')).ok_or_else(|| io::Error::other(format!("{key} must be quoted")).into())
}
const CPU_REPLACEMENTS: &[(&str, &str)] = &[
	(
		"@contraction_tile = external addrspace(3) global [0 x double], align 16",
		"@contraction_tile = internal thread_local global [RECIPE_CONTRACTION_CPU_SHARED_VALUES x double] zeroinitializer, align 16",
	),
	(" addrspace(3)", ""),
	("call i32 @llvm.amdgcn.workitem.id.x()", "call i32 @recipe.cpu.thread.id()"),
	("call i32 @recipe.local.id.x()", "add i32 0, 0"),
	("call i32 @recipe.group.id.x()", "call i32 @recipe.cpu.thread.id()"),
	("call i32 @recipe.workgroup.size.x()", "add i32 1, 0"),
	("call void @llvm.amdgcn.s.barrier()", ""),
	("call void @recipe.local.barrier()", ""),
	("call void @grid_barrier(i32 %threads)", "call void @recipe.cpu.barrier()"),
	("declare i32 @llvm.amdgcn.workitem.id.x()", ""),
	("declare void @llvm.amdgcn.s.barrier()", ""),
	("declare i64 @__ockl_steadyctr_u64()", ""),
	("attributes #0 = { nounwind \"amdgpu-flat-work-group-size\"=\"RECIPE_WORKGROUP_SIZE,RECIPE_WORKGROUP_SIZE\" }", "attributes #0 = { nounwind }"),
];
const CPU_PARALLEL: &str = r#"@recipe.cpu.thread = internal thread_local global i32 0, align 4
@recipe.cpu.barrier.context = internal thread_local global ptr null, align 8
@recipe.cpu.barrier.wait = internal thread_local global ptr null, align 8
define void @recipe_model_thread(i32 %thread, ptr %context, ptr %wait) #0 { entry: store i32 %thread, ptr @recipe.cpu.thread, align 4 store ptr %context, ptr @recipe.cpu.barrier.context, align 8 store ptr %wait, ptr @recipe.cpu.barrier.wait, align 8 ret void }
define internal i32 @recipe.cpu.thread.id() #1 { entry: %thread = load i32, ptr @recipe.cpu.thread, align 4 ret i32 %thread }
define internal void @recipe.cpu.barrier() #1 { entry: %context = load ptr, ptr @recipe.cpu.barrier.context, align 8 %wait = load ptr, ptr @recipe.cpu.barrier.wait, align 8 call void %wait(ptr %context) ret void }"#;
/// Compile-time contraction shape. A reverse K extent is cut into one contiguous
/// partition per `split_span` elements, capped at `partitions`, so the summation
/// order is a property of the program rather than of the device it runs on.
#[derive(Clone, Copy)]
struct Schedule {
	swizzle_m: u32,
	partitions: u32,
	split_span: u32,
	matrix_split_span: u32,
	local_chunks: u32,
}
fn precision_sources(ir: String, schedule: Schedule) -> BuildResult<[(&'static str, String); 10]> {
	let ir = ir
		.replace("RECIPE_CONTRACTION_SWIZZLE_M", &schedule.swizzle_m.to_string())
		.replace("RECIPE_CONTRACTION_K_PARTITIONS", &schedule.partitions.to_string())
		.replace("RECIPE_CONTRACTION_MATRIX_SPLIT_SPAN", &schedule.matrix_split_span.to_string())
		.replace("RECIPE_CONTRACTION_SPLIT_SPAN", &schedule.split_span.to_string())
		.replace("RECIPE_CONTRACTION_LOCAL_CHUNKS", &schedule.local_chunks.to_string());
	Ok([
		("", native_ir(ir.clone(), "", "double", FloatFormat::FP64)?),
		("-f32", native_ir(ir.clone(), "_f32", "float", FloatFormat::FP32)?),
		("-f16", half_ir(ir.clone())?),
		("-f8", encoded_ir(ir.clone(), "_f8", FloatFormat::FP8.bytes(), fp8_codec(), |value| FloatFormat::FP8.pack(value))?),
		("-bf16", encoded_ir(ir.clone(), "_bf16", FloatFormat::BF16.bytes(), bf16_codec(), |value| FloatFormat::BF16.pack(value))?),
		("-tf32", native_ir(ir.clone(), "_tf32", "float", FloatFormat::TF32)?),
		("-int8", encoded_ir(ir.clone(), "_int8", IntFormat::INT8.bytes(), &int_codec(IntFormat::INT8), |value| IntFormat::INT8.pack(value))?),
		("-int4", encoded_ir(ir.clone(), "_int4", IntFormat::INT4.bytes(), &int_codec(IntFormat::INT4), |value| IntFormat::INT4.pack(value))?),
		("-int1", encoded_ir(ir.clone(), "_int1", IntFormat::INT1.bytes(), &int_codec(IntFormat::INT1), |value| IntFormat::INT1.pack(value))?),
		("-f", custom_ir(ir, "_f")?),
	])
}
fn wmma_source(source: &str) -> String {
	source.lines().filter(|line| !line.starts_with("; RECIPE_WMMA ")).collect::<Vec<_>>().join("\n")
}
fn wmma_method(source: &str, key: &str) -> BuildResult<(String, String)> {
	let marker = format!("{key} ");
	let catalog = source.lines().find_map(|line| line.strip_prefix("; RECIPE_WMMA ")).ok_or_else(|| io::Error::other("WMMA methods are absent"))?;
	let method = catalog.split(" || ").find_map(|method| method.strip_prefix(&marker)).ok_or_else(|| io::Error::other(format!("WMMA method {key} is absent")))?;
	let (kind, body) = method.split_once(' ').ok_or_else(|| io::Error::other(format!("WMMA method {key} has no body")))?;
	Ok((kind.to_owned(), body.replace("\\n", "\n")))
}
fn compose_wmma(ir: String, source: &str, key: &str) -> BuildResult<String> {
	let (kind, body) = wmma_method(source, key)?;
	if kind == "call" {
		return Ok(ir.replace("@recipe.wmma(", &body));
	}
	if kind != "definition" {
		return Err(io::Error::other(format!("WMMA method {key} has an invalid kind")).into());
	}
	let declaration = ir.lines().find(|line| line.starts_with("declare ") && line.contains("@recipe.wmma(")).ok_or_else(|| io::Error::other("specialized WMMA declaration is absent"))?.to_owned();
	Ok(ir.replace(&declaration, &body))
}
fn compose_contraction(mut ir: String, matrix: bool) -> String {
	for (operation, vector, native) in [
		("@contraction_product_accumulate(", "@contraction_vector_accumulate(", "@contraction_matrix_accumulate("),
		("@contraction_a_index(", "@contraction_vector_a_index(", "@contraction_matrix_a_index("),
		("@contraction_b_index(", "@contraction_vector_b_index(", "@contraction_matrix_b_index("),
		("@contraction_output_m(", "@contraction_vector_output_m(", "@contraction_matrix_output_m("),
		("@contraction_output_n(", "@contraction_vector_output_n(", "@contraction_matrix_output_n("),
		("@contraction_store_lane(", "@contraction_vector_store_lane(", "@contraction_matrix_store_lane("),
	] {
		ir = ir.replace(operation, if matrix { native } else { vector })
	}
	ir
}
fn compile_amd(manifest: &str, out: &PathBuf, schedule: Schedule) -> BuildResult<()> {
	let source = fs::read_to_string("amd-nv-cpu.ll")?;
	let ir = parallel_ir(wmma_source(&source), AMD_WIDTH, AMD_GRID_BARRIER);
	let mut values = Vec::new();
	for (suffix, contents) in precision_sources(ir, schedule)? {
		let path = out.join(format!("recipe-amd{suffix}.ll"));
		fs::write(&path, compose_contraction(contents.clone(), false))?;
		values.push(format!("{}={}", if suffix.is_empty() { "default" } else { suffix }, path.display()));
		if ["-f16", "-bf16", "-int8", "-int4"].contains(&suffix) {
			for architecture in ["gfx11", "gfx12"] {
				let template = format!("{architecture}{suffix}");
				let method = if template == "gfx12-int4" { "gfx12-int8" } else { &template };
				let path = out.join(format!("recipe-amd-{template}.ll"));
				fs::write(&path, compose_wmma(compose_contraction(contents.clone(), true), &source, method)?)?;
				values.push(format!("{template}={}", path.display()));
			}
		}
	}
	println!("cargo:rustc-env=RECIPE_AMD_IR={}", values.join("\x3b"));
	println!("cargo:rustc-env=RECIPE_HSA_COMPILER={}", text(manifest, "hsa-compiler")?);
	for (key, environment) in [
		("hsa-device-library", "RECIPE_HSA_DEVICE_LIBRARY"),
		("hsa-clock-library", "RECIPE_HSA_CLOCK_LIBRARY"),
		("hsa-abi-library", "RECIPE_HSA_ABI_LIBRARY"),
		("hsa-finite-library", "RECIPE_HSA_FINITE_LIBRARY"),
		("hsa-math-library", "RECIPE_HSA_MATH_LIBRARY"),
		("hsa-device-library-directory", "RECIPE_HSA_DEVICE_LIBRARY_DIRECTORY"),
	] {
		println!("cargo:rustc-env={environment}={}", text(manifest, key)?);
	}
	Ok(())
}
fn compile_nvidia(manifest: &str, out: &PathBuf, schedule: Schedule) -> BuildResult<()> {
	let ir = wmma_source(&fs::read_to_string("amd-nv-cpu.ll")?);
	let ir = parallel_ir(ir, "declare i32 @recipe.workgroup.size.x()", NVIDIA_GRID_BARRIER)
		.replace("amdgcn-amd-amdhsa", "nvptx64-nvidia-cuda")
		.replace("llvm.amdgcn.workitem.id.x", "llvm.nvvm.read.ptx.sreg.tid.x")
		.replace("llvm.amdgcn.workgroup.id.x", "llvm.nvvm.read.ptx.sreg.ctaid.x")
		.replace("recipe.workgroup.size.x", "llvm.nvvm.read.ptx.sreg.ntid.x")
		.replace("llvm.amdgcn.s.barrier", "llvm.nvvm.barrier0")
		.replace("attributes #0 = { nounwind \"amdgpu-flat-work-group-size\"=\"RECIPE_WORKGROUP_SIZE,RECIPE_WORKGROUP_SIZE\" }", "attributes #0 = { nounwind }")
		.replace(", addrspace(5)", "")
		.replace(" addrspace(5)", "");
	let mut values = Vec::new();
	for (suffix, contents) in precision_sources(ir, schedule)? {
		let path = out.join(format!("recipe-nvidia{suffix}.ll"));
		fs::write(&path, compose_contraction(contents, false))?;
		values.push(format!("{}={}", if suffix.is_empty() { "default" } else { suffix }, path.display()));
	}
	println!("cargo:rustc-env=RECIPE_NV_IR={}", values.join("\x3b"));
	println!("cargo:rustc-env=RECIPE_NV_COMPILER={}", text(manifest, "nvidia-compiler")?);
	println!("cargo:rustc-env=RECIPE_NV_DEVICE_LIBRARY={}", text(manifest, "nvidia-device-library")?);
	println!("cargo:rustc-env=RECIPE_NV_PTX_VERSION=+{}", text(manifest, "nvidia-ptx")?);
	println!("cargo:rustc-env=RECIPE_NV_PTX_GENERATOR={}", text(manifest, "nvidia-ptx-generator")?);
	Ok(())
}
fn compile_cpu(manifest: &str, out: &PathBuf, schedule: Schedule) -> BuildResult<()> {
	let target = env::var("TARGET")?;
	let mut ir = wmma_source(&fs::read_to_string("amd-nv-cpu.ll")?).replace("amdgcn-amd-amdhsa", &target);
	for (pattern, replacement) in CPU_REPLACEMENTS {
		ir = ir.replace(pattern, replacement);
	}
	ir.push_str(CPU_PARALLEL);
	let clang = text(manifest, "cpu-compiler")?;
	if !Path::new(clang).exists() {
		return Err(io::Error::other(format!("cpu-compiler {clang:?} is absent")).into());
	}
	let mut values = Vec::new();
	for (suffix, contents) in precision_sources(ir, schedule)? {
		let contents = contents
			.replace(" addrspace(1)", "")
			.replace(" addrspace(3)", "")
			.replace(", addrspace(5)", "")
			.replace(" addrspace(5)", "")
			.replace("RECIPE_CONTRACTION_CPU_SHARED_VALUES", number(manifest, "contraction-cpu-shared-values")?);
		let path = out.join(format!("recipe-cpu{suffix}.ll"));
		fs::write(&path, compose_contraction(contents, false))?;
		values.push(format!("{}={}", if suffix.is_empty() { "default" } else { suffix }, path.display()));
	}
	println!("cargo:rustc-env=RECIPE_CPU_IR={}", values.join("\x3b"));
	println!("cargo:rustc-env=RECIPE_CPU_COMPILER={clang}");
	println!("cargo:rustc-env=RECIPE_CPU_TARGET={target}");
	Ok(())
}
fn main() -> BuildResult<()> {
	let manifest = fs::read_to_string("Cargo.toml")?;
	let positive = |key: &str| -> BuildResult<u32> {
		setting(&manifest, key)?.parse::<u32>().ok().filter(|value| *value != 0).ok_or_else(|| io::Error::other(format!("{key} must be a positive integer")).into())
	};
	let schedule = Schedule {
		swizzle_m: positive("contraction-swizzle-m-tiles")?,
		partitions: positive("contraction-k-partitions")?,
		split_span: positive("contraction-split-span")?,
		matrix_split_span: positive("contraction-matrix-split-span")?,
		local_chunks: positive("contraction-local-chunks")?,
	};
	for (key, environment) in [
		("epochs", "RECIPE_TRAIN_EPOCHS"),
		("learning-rate", "RECIPE_TRAIN_LEARNING_RATE"),
		("initial-weight", "RECIPE_TRAIN_INITIAL_WEIGHT"),
		("adamw-beta1", "RECIPE_ADAMW_BETA1"),
		("adamw-beta2", "RECIPE_ADAMW_BETA2"),
		("adamw-epsilon", "RECIPE_ADAMW_EPSILON"),
		("adamw-weight-decay", "RECIPE_ADAMW_WEIGHT_DECAY"),
		("kmeans-iterations", "RECIPE_KMEANS_ITERATIONS"),
		("svm-iterations", "RECIPE_SVM_ITERATIONS"),
		("svm-learning-rate", "RECIPE_SVM_LEARNING_RATE"),
		("svm-regularization", "RECIPE_SVM_REGULARIZATION"),
		("svm-epsilon", "RECIPE_SVM_EPSILON"),
		("tree-depth", "RECIPE_TREE_DEPTH"),
		("tree-min-rows", "RECIPE_TREE_MIN_ROWS"),
		("forest-feature-fraction", "RECIPE_FOREST_FEATURE_FRACTION"),
		("bayes-prior-precision", "RECIPE_BAYES_PRIOR_PRECISION"),
		("bayes-noise-variance", "RECIPE_BAYES_NOISE_VARIANCE"),
		("bayes-variance-epsilon", "RECIPE_BAYES_VARIANCE_EPSILON"),
		("boost-iterations", "RECIPE_BOOST_ITERATIONS"),
		("boost-learning-rate", "RECIPE_BOOST_LEARNING_RATE"),
		("catboost-ordered-prior", "RECIPE_CATBOOST_ORDERED_PRIOR"),
		("catboost-border-count", "RECIPE_CATBOOST_BORDER_COUNT"),
		("xgboost-l2-regularization", "RECIPE_XGBOOST_L2_REGULARIZATION"),
		("xgboost-minimum-gain", "RECIPE_XGBOOST_MINIMUM_GAIN"),
		("lightgbm-histogram-bins", "RECIPE_LIGHTGBM_HISTOGRAM_BINS"),
		("lightgbm-leaves", "RECIPE_LIGHTGBM_LEAVES"),
		("quantization-block-weights", "RECIPE_QUANTIZATION_BLOCK_WEIGHTS"),
		("surrogate-epochs", "RECIPE_SURROGATE_EPOCHS"),
		("surrogate-rate", "RECIPE_SURROGATE_RATE"),
		("surrogate-width", "RECIPE_SURROGATE_WIDTH"),
		("random-seed", "RECIPE_RANDOM_SEED"),
		("progress-refresh-hz", "RECIPE_PROGRESS_REFRESH_HZ"),
		("normalization-epsilon", "RECIPE_NORMALIZATION_EPSILON"),
		("categorical-ratio", "RECIPE_CATEGORICAL_RATIO"),
		("leak-slope", "RECIPE_LEAK_SLOPE"),
		("prelu-slope", "RECIPE_PRELU_SLOPE"),
		("elu-alpha", "RECIPE_ELU_ALPHA"),
		("selu-alpha", "RECIPE_SELU_ALPHA"),
		("selu-scale", "RECIPE_SELU_SCALE"),
		("gelu-scale", "RECIPE_GELU_SCALE"),
		("gelu-cubic", "RECIPE_GELU_CUBIC"),
		("huber-threshold", "RECIPE_HUBER_THRESHOLD"),
		("output-tolerance", "RECIPE_OUTPUT_TOLERANCE"),
		("gradient-tolerance", "RECIPE_GRADIENT_TOLERANCE"),
		("backend-tolerance", "RECIPE_BACKEND_TOLERANCE"),
		("contraction-cpu-shared-values", "RECIPE_CONTRACTION_CPU_SHARED_VALUES"),
		("contraction-register-m", "RECIPE_CONTRACTION_REGISTER_M"),
		("contraction-register-n", "RECIPE_CONTRACTION_REGISTER_N"),
		("contraction-fragment-k", "RECIPE_CONTRACTION_FRAGMENT_K"),
		("contraction-k-partitions", "RECIPE_CONTRACTION_K_PARTITIONS"),
		("contraction-split-span", "RECIPE_CONTRACTION_SPLIT_SPAN"),
		("contraction-matrix-split-span", "RECIPE_CONTRACTION_MATRIX_SPLIT_SPAN"),
		("contraction-chunk-k", "RECIPE_CONTRACTION_CHUNK_K"),
		("contraction-resident-waves-per-workgroup", "RECIPE_CONTRACTION_RESIDENT_WAVES_PER_WORKGROUP"),
		("contraction-matrix-max-waves-per-workgroup", "RECIPE_CONTRACTION_MATRIX_MAX_WAVES_PER_WORKGROUP"),
		("attention-query-tile", "RECIPE_ATTENTION_QUERY_TILE"),
		("topology-probe-bytes", "RECIPE_TOPOLOGY_PROBE_BYTES"),
		("cpu-worker-threads", "RECIPE_CPU_WORKER_THREADS"),
	] {
		println!("cargo:rustc-env={environment}={}", number(&manifest, key)?);
	}
	for (key, environment) in [("hsa-runtime", "RECIPE_HSA_RUNTIME"), ("nvidia-runtime", "RECIPE_NV_RUNTIME")] {
		println!("cargo:rustc-env={environment}={}", text(&manifest, key)?);
	}
	let placement = setting(&manifest, "multi-device")?;
	println!(
		"cargo:rustc-env=RECIPE_MULTI_DEVICE={}",
		match placement {
			"false" | "true" => placement,
			"\"auto\"" => "auto",
			value => return Err(io::Error::other(format!("multi-device must be false, true, or \"auto\", not {value}")).into()),
		}
	);
	let out = PathBuf::from(env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("OUT_DIR must be configured"))?);
	println!("cargo::rustc-check-cfg=cfg(amd)");
	println!("cargo::rustc-check-cfg=cfg(nvidia)");
	let toolchain = |compiler: &str, library: &str| -> BuildResult<bool> { Ok(Path::new(text(&manifest, compiler)?).exists() && Path::new(text(&manifest, library)?).exists()) };
	compile_cpu(&manifest, &out, schedule)?;
	// GPU driver stubs and library search paths are host-arch: cross-compiled builds are CPU-only.
	let native = env::var("TARGET")? == env::var("HOST")?;
	let amd = native && toolchain("hsa-compiler", "hsa-device-library")?;
	let nvidia = native && toolchain("nvidia-compiler", "nvidia-device-library")? && Path::new(text(&manifest, "nvidia-ptx-generator")?).exists();
	if amd {
		println!("cargo:rustc-cfg=amd");
		compile_amd(&manifest, &out, schedule)?;
	}
	if nvidia {
		println!("cargo:rustc-cfg=nvidia");
		compile_nvidia(&manifest, &out, schedule)?;
	}
	println!("cargo:rerun-if-changed=Cargo.toml");
	println!("cargo:rerun-if-changed=amd-nv-cpu.ll");
	Ok(())
}
