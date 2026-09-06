//! Recipe executes one model graph after automatically probing a compiled discrete GPU backend.
//! Attention uses learned Q/K/V and output projections.
#![allow(non_upper_case_globals)]
mod program_ir {
	//! Compile-time lowering for the scalar, predictor, route, and normalization
	//! pieces of a concrete model.
	//!
	//! The caller supplies the already selected value type and pointer spelling.
	//! These routines return LLVM text containing only fixed SSA and direct memory
	//! operations. They intentionally do not emit descriptors, instruction arrays,
	//! opcode switches, or runtime graph traversal.

	use std::collections::BTreeMap;
	use std::fmt::{self, Write as _};

	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	#[repr(i32)]
	pub enum ScalarOpcode {
		Add = 0,
		Constant = 1,
		Parameter = 2,
		Subtract = 3,
		Multiply = 4,
		Divide = 5,
		Absolute = 6,
		Exp = 7,
		Log = 8,
		Sin = 10,
		Cos = 11,
		Tanh = 12,
		Greater = 13,
		StraightThrough = 14,
		Select = 15,
	}

	impl ScalarOpcode {
		fn from_i32(value: i32) -> Result<Self, EmitError> {
			match value {
				0 => Ok(Self::Add),
				1 => Ok(Self::Constant),
				2 => Ok(Self::Parameter),
				3 => Ok(Self::Subtract),
				4 => Ok(Self::Multiply),
				5 => Ok(Self::Divide),
				6 => Ok(Self::Absolute),
				7 => Ok(Self::Exp),
				8 => Ok(Self::Log),
				10 => Ok(Self::Sin),
				11 => Ok(Self::Cos),
				12 => Ok(Self::Tanh),
				13 => Ok(Self::Greater),
				14 => Ok(Self::StraightThrough),
				15 => Ok(Self::Select),
				_ => Err(EmitError::InvalidOpcode { kind: "scalar", value }),
			}
		}
	}

	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	pub enum PredictorOpcode {
		Feature = 0,
		Row = 1,
		Constant = 2,
		Load = 3,
		Store = 4,
		Duplicate = 5,
		Add = 6,
		Subtract = 7,
		Multiply = 8,
		Divide = 9,
		Greater = 10,
		Choose = 11,
		Nearest = 12,
		Affine = 13,
		Gaussian = 14,
	}

	impl PredictorOpcode {
		fn from_i32(value: i32) -> Result<Self, EmitError> {
			match value {
				0 => Ok(Self::Feature),
				1 => Ok(Self::Row),
				2 => Ok(Self::Constant),
				3 => Ok(Self::Load),
				4 => Ok(Self::Store),
				5 => Ok(Self::Duplicate),
				6 => Ok(Self::Add),
				7 => Ok(Self::Subtract),
				8 => Ok(Self::Multiply),
				9 => Ok(Self::Divide),
				10 => Ok(Self::Greater),
				11 => Ok(Self::Choose),
				12 => Ok(Self::Nearest),
				13 => Ok(Self::Affine),
				14 => Ok(Self::Gaussian),
				_ => Err(EmitError::InvalidOpcode { kind: "predictor", value }),
			}
		}
	}

	pub enum EmitError {
		WrongWidth { kind: &'static str, width: usize },
		InvalidOpcode { kind: &'static str, value: i32 },
		InvalidOperand { kind: &'static str, value: f64 },
		InvalidReference { kind: &'static str, index: i32 },
		StackUnderflow { kind: &'static str },
		StackDepth { kind: &'static str, depth: usize },
		LocalIndex { index: usize, locals: usize },
	}

	impl fmt::Display for EmitError {
		fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
			match self {
				Self::WrongWidth { kind, width } => write!(f, "{kind} program has invalid width {width}"),
				Self::InvalidOpcode { kind, value } => write!(f, "{kind} program has invalid opcode {value}"),
				Self::InvalidOperand { kind, value } => write!(f, "{kind} program has invalid operand {value}"),
				Self::InvalidReference { kind, index } => write!(f, "{kind} program references unavailable value {index}"),
				Self::StackUnderflow { kind } => write!(f, "{kind} program stack underflows"),
				Self::StackDepth { kind, depth } => write!(f, "{kind} program ends at stack depth {depth}"),
				Self::LocalIndex { index, locals } => write!(f, "predictor local {index} is outside {locals} locals"),
			}
		}
	}

	pub type LiteralFn<'a> = dyn Fn(f64, &str) -> String + 'a;

	#[derive(Clone, Copy)]
	pub struct ScalarContext<'a> {
		pub value_type: &'a str,
		pub pointer_type: &'a str,
		pub alignment: usize,
		pub first: &'a str,
		pub second: &'a str,
		pub weights: &'a str,
		pub decode: usize,
		pub prefix: &'a str,
		pub literal: &'a LiteralFn<'a>,
	}

	pub struct ScalarForward {
		pub code: String,
		pub value: String,
	}

	pub struct ScalarReverse {
		pub code: String,
		pub first_adjoint: String,
		pub second_adjoint: String,
		pub parameter_adjoint: BTreeMap<usize, String>,
	}

	struct ScalarInstruction {
		opcode: ScalarOpcode,
		left: f64,
		right: f64,
	}

	fn integer(value: f64, kind: &'static str) -> Result<i32, EmitError> {
		if !value.is_finite() || value.fract() != 0.0 || value < i32::MIN as f64 || value > i32::MAX as f64 {
			return Err(EmitError::InvalidOperand { kind, value });
		}
		Ok(value as i32)
	}

	fn binary(code: &mut String, value_type: &str, name: &str, operation: &str, left: &str, right: &str) -> String {
		let _ = writeln!(code, "{name} = call {value_type} @recipe.{operation}({value_type} {left}, {value_type} {right})");
		name.to_owned()
	}

	fn predicate(code: &mut String, value_type: &str, name: &str, operation: &str, left: &str, right: &str) -> String {
		let _ = writeln!(code, "{name}.condition = call i1 @recipe.{operation}({value_type} {left}, {value_type} {right})");
		let _ = writeln!(code, "{name} = call {value_type} @recipe.from.u1(i1 {name}.condition)");
		name.to_owned()
	}

	fn parse_scalar(code: &[f64]) -> Result<Vec<ScalarInstruction>, EmitError> {
		if code.len() % 3 != 0 {
			return Err(EmitError::WrongWidth { kind: "scalar", width: code.len() });
		}
		code.chunks_exact(3)
			.map(|instruction| Ok(ScalarInstruction { opcode: ScalarOpcode::from_i32(integer(instruction[0], "scalar opcode")?)?, left: instruction[1], right: instruction[2] }))
			.collect()
	}

	fn scalar_operand(value: f64, values: &[String], first: &str, second: &str) -> Result<String, EmitError> {
		let index = integer(value, "scalar reference")?;
		match index {
			-2 => Ok(second.to_owned()),
			-1 => Ok(first.to_owned()),
			0.. => values.get(index as usize).cloned().ok_or(EmitError::InvalidReference { kind: "scalar", index }),
			_ => Err(EmitError::InvalidReference { kind: "scalar", index }),
		}
	}

	/// Emit a scalar program as straight-line SSA. The returned value is the
	/// program result. Scalar `StraightThrough` returns its left operand in the
	/// forward path, matching the real block's inference semantics.
	pub fn emit_scalar_forward(code: &[f64], context: ScalarContext<'_>) -> Result<ScalarForward, EmitError> {
		let instructions = parse_scalar(code)?;
		let mut output = String::new();
		let mut values = Vec::with_capacity(instructions.len());
		for (index, instruction) in instructions.iter().enumerate() {
			let name = format!("%{}.scalar.{index}", context.prefix);
			let value = match instruction.opcode {
				ScalarOpcode::Constant => (context.literal)(instruction.left, context.value_type),
				ScalarOpcode::Parameter => {
					let parameter = integer(instruction.left, "scalar parameter")?;
					if parameter < 0 {
						return Err(EmitError::InvalidOperand { kind: "scalar parameter", value: instruction.left });
					}
					if context.decode == 0 {
						let pointer = format!("{name}.ptr");
						let _ = writeln!(
							output,
							"{pointer} = getelementptr inbounds {ty}, {ptrty} {weights}, i32 {parameter}",
							ty = context.value_type,
							ptrty = context.pointer_type,
							weights = context.weights,
							parameter = parameter
						);
						let _ = writeln!(
							output,
							"{name} = load {ty}, {ptrty} {pointer}, align {align}",
							ty = context.value_type,
							ptrty = context.pointer_type,
							pointer = pointer,
							align = context.alignment
						);
					} else {
						let _ = writeln!(
							output,
							"{name} = call {ty} @recipe.model.decode({ptrty} {weights}, i32 {parameter}, i32 {decode})",
							ty = context.value_type,
							ptrty = context.pointer_type,
							weights = context.weights,
							parameter = parameter,
							decode = context.decode
						);
					}
					name
				}
				ScalarOpcode::StraightThrough => scalar_operand(instruction.left, &values, context.first, context.second)?,
				ScalarOpcode::Select => {
					let condition = scalar_operand(instruction.left, &values, context.first, context.second)?;
					let value = scalar_operand(instruction.right, &values, context.first, context.second)?;
					let zero = (context.literal)(0.0, context.value_type);
					let _ = writeln!(output, "{name}.condition = call i1 @recipe.ogt({ty} {condition}, {ty} {zero})", ty = context.value_type);
					let _ = writeln!(output, "{name} = select i1 {name}.condition, {ty} {value}, {ty} {zero}", ty = context.value_type);
					name
				}
				ScalarOpcode::Add | ScalarOpcode::Subtract | ScalarOpcode::Multiply | ScalarOpcode::Divide | ScalarOpcode::Greater => {
					let left = scalar_operand(instruction.left, &values, context.first, context.second)?;
					let right = scalar_operand(instruction.right, &values, context.first, context.second)?;
					match instruction.opcode {
						ScalarOpcode::Add => binary(&mut output, context.value_type, &name, "add", &left, &right),
						ScalarOpcode::Subtract => binary(&mut output, context.value_type, &name, "sub", &left, &right),
						ScalarOpcode::Multiply => binary(&mut output, context.value_type, &name, "mul", &left, &right),
						ScalarOpcode::Divide => binary(&mut output, context.value_type, &name, "div", &left, &right),
						ScalarOpcode::Greater => predicate(&mut output, context.value_type, &name, "ogt", &left, &right),
						_ => unreachable!(),
					}
				}
				ScalarOpcode::Absolute | ScalarOpcode::Exp | ScalarOpcode::Log | ScalarOpcode::Sin | ScalarOpcode::Cos | ScalarOpcode::Tanh => {
					let left = scalar_operand(instruction.left, &values, context.first, context.second)?;
					let operation = match instruction.opcode {
						ScalarOpcode::Absolute => "abs",
						ScalarOpcode::Exp => "exp",
						ScalarOpcode::Log => "log",
						ScalarOpcode::Sin => "sin",
						ScalarOpcode::Cos => "cos",
						ScalarOpcode::Tanh => "tanh",
						_ => unreachable!(),
					};
					let _ = writeln!(output, "{name} = call {ty} @recipe.{operation}({ty} {left})", ty = context.value_type);
					name
				}
			};
			values.push(value);
		}
		let value = values.last().cloned().ok_or(EmitError::WrongWidth { kind: "scalar", width: 0 })?;
		Ok(ScalarForward { code: output, value })
	}

	fn add_adjoint(code: &mut String, value_type: &str, prefix: &str, old: &mut String, contribution: &str, sequence: &mut usize) {
		let name = format!("%{prefix}.adjoint.{}", *sequence);
		*sequence += 1;
		*old = binary(code, value_type, &name, "add", old, contribution);
	}

	fn negate(code: &mut String, value_type: &str, prefix: &str, value: &str, sequence: &mut usize) -> String {
		let name = format!("%{prefix}.neg.{}", *sequence);
		*sequence += 1;
		let _ = writeln!(code, "{name} = call {value_type} @recipe.neg({value_type} {value})");
		name
	}

	/// Emit the reverse of a scalar program using the forward SSA values. The
	/// result contains expressions for the two input adjoints and parameter
	/// adjoints. The caller owns the flat adjoint and gradient arenas and stores
	/// these expressions at the node's fixed element/parameter offsets.
	pub fn emit_scalar_reverse(code: &[f64], context: ScalarContext<'_>, incoming: &str) -> Result<ScalarReverse, EmitError> {
		let instructions = parse_scalar(code)?;
		let mut output = String::new();
		let mut values = Vec::with_capacity(instructions.len());
		let mut parameter_for = vec![None; instructions.len()];
		for (index, instruction) in instructions.iter().enumerate() {
			let value = match instruction.opcode {
				ScalarOpcode::Constant => (context.literal)(instruction.left, context.value_type),
				ScalarOpcode::Parameter => {
					let parameter = integer(instruction.left, "scalar parameter")?;
					if parameter < 0 {
						return Err(EmitError::InvalidOperand { kind: "scalar parameter", value: instruction.left });
					}
					parameter_for[index] = Some(parameter as usize);
					format!("%{}.scalar.{index}", context.prefix)
				}
				ScalarOpcode::StraightThrough => scalar_operand(instruction.left, &values, context.first, context.second)?,
				ScalarOpcode::Add
				| ScalarOpcode::Subtract
				| ScalarOpcode::Multiply
				| ScalarOpcode::Divide
				| ScalarOpcode::Greater
				| ScalarOpcode::Select
				| ScalarOpcode::Absolute
				| ScalarOpcode::Exp
				| ScalarOpcode::Log
				| ScalarOpcode::Sin
				| ScalarOpcode::Cos
				| ScalarOpcode::Tanh => format!("%{}.scalar.{index}", context.prefix),
			};
			values.push(value);
		}
		let mut adjoints = vec![(context.literal)(0.0, context.value_type); instructions.len()];
		let mut first = (context.literal)(0.0, context.value_type);
		let mut second = (context.literal)(0.0, context.value_type);
		let mut parameters = BTreeMap::new();
		let mut sequence = 0;
		if let Some(last) = adjoints.last_mut() {
			*last = incoming.to_owned();
		}
		let operand = |value: f64, values: &[String]| scalar_operand(value, values, context.first, context.second);
		let add_operand = |code: &mut String, value: f64, contribution: &str, adjoints: &mut [String], first: &mut String, second: &mut String, sequence: &mut usize| -> Result<(), EmitError> {
			let index = integer(value, "scalar reference")?;
			match index {
				-2 => add_adjoint(code, context.value_type, context.prefix, second, contribution, sequence),
				-1 => add_adjoint(code, context.value_type, context.prefix, first, contribution, sequence),
				0.. => {
					let slot = usize::try_from(index).map_err(|_| EmitError::InvalidReference { kind: "scalar", index })?;
					let target = adjoints.get_mut(slot).ok_or(EmitError::InvalidReference { kind: "scalar", index })?;
					add_adjoint(code, context.value_type, context.prefix, target, contribution, sequence)
				}
				_ => return Err(EmitError::InvalidReference { kind: "scalar", index }),
			}
			Ok(())
		};
		for (index, instruction) in instructions.iter().enumerate().rev() {
			let adjoint = adjoints[index].clone();
			let left = if matches!(instruction.opcode, ScalarOpcode::Constant | ScalarOpcode::Parameter) { String::new() } else { operand(instruction.left, &values)? };
			let right = if matches!(
				instruction.opcode,
				ScalarOpcode::Add | ScalarOpcode::Subtract | ScalarOpcode::Multiply | ScalarOpcode::Divide | ScalarOpcode::Greater | ScalarOpcode::Select | ScalarOpcode::StraightThrough
			) {
				operand(instruction.right, &values)?
			} else {
				String::new()
			};
			match instruction.opcode {
				ScalarOpcode::Add => {
					add_operand(&mut output, instruction.left, &adjoint, &mut adjoints, &mut first, &mut second, &mut sequence)?;
					add_operand(&mut output, instruction.right, &adjoint, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::StraightThrough => {
					add_operand(&mut output, instruction.right, &adjoint, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Subtract => {
					add_operand(&mut output, instruction.left, &adjoint, &mut adjoints, &mut first, &mut second, &mut sequence)?;
					let negative = negate(&mut output, context.value_type, context.prefix, &adjoint, &mut sequence);
					add_operand(&mut output, instruction.right, &negative, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Multiply => {
					let left_contribution = binary(&mut output, context.value_type, &format!("%{}.mul.left.{sequence}", context.prefix), "mul", &adjoint, &right);
					sequence += 1;
					let right_contribution = binary(&mut output, context.value_type, &format!("%{}.mul.right.{sequence}", context.prefix), "mul", &adjoint, &left);
					sequence += 1;
					add_operand(&mut output, instruction.left, &left_contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
					add_operand(&mut output, instruction.right, &right_contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Divide => {
					let left_contribution = binary(&mut output, context.value_type, &format!("%{}.div.left.{sequence}", context.prefix), "div", &adjoint, &right);
					sequence += 1;
					let square = binary(&mut output, context.value_type, &format!("%{}.div.square.{sequence}", context.prefix), "mul", &right, &right);
					sequence += 1;
					let numerator = binary(&mut output, context.value_type, &format!("%{}.div.numerator.{sequence}", context.prefix), "mul", &adjoint, &left);
					sequence += 1;
					let raw = binary(&mut output, context.value_type, &format!("%{}.div.raw.{sequence}", context.prefix), "div", &numerator, &square);
					sequence += 1;
					let right_contribution = negate(&mut output, context.value_type, context.prefix, &raw, &mut sequence);
					add_operand(&mut output, instruction.left, &left_contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
					add_operand(&mut output, instruction.right, &right_contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Absolute => {
					let negative = format!("%{}.abs.negative.{sequence}", context.prefix);
					sequence += 1;
					let positive = format!("%{}.abs.positive.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(output, "{negative} = call i1 @recipe.olt({ty} {left}, {ty} {zero})", ty = context.value_type, zero = (context.literal)(0.0, context.value_type));
					let _ = writeln!(output, "{positive} = call i1 @recipe.ogt({ty} {left}, {ty} {zero})", ty = context.value_type, zero = (context.literal)(0.0, context.value_type));
					let negated = negate(&mut output, context.value_type, context.prefix, &adjoint, &mut sequence);
					let upper = format!("%{}.abs.upper.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(
						output,
						"{upper} = select i1 {positive}, {ty} {adjoint}, {ty} {zero}",
						ty = context.value_type,
						adjoint = adjoint,
						zero = (context.literal)(0.0, context.value_type)
					);
					let contribution = format!("%{}.abs.contribution.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(output, "{contribution} = select i1 {negative}, {ty} {negated}, {ty} {upper}", ty = context.value_type, negated = negated, upper = upper);
					add_operand(&mut output, instruction.left, &contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Exp => {
					let contribution = binary(&mut output, context.value_type, &format!("%{}.exp.{sequence}", context.prefix), "mul", &adjoint, &values[index]);
					sequence += 1;
					add_operand(&mut output, instruction.left, &contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Log => {
					let contribution = binary(&mut output, context.value_type, &format!("%{}.log.{sequence}", context.prefix), "div", &adjoint, &left);
					sequence += 1;
					add_operand(&mut output, instruction.left, &contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Sin => {
					let cosine = format!("%{}.sin.cosine.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(output, "{cosine} = call {ty} @recipe.cos({ty} {left})", ty = context.value_type);
					let contribution = binary(&mut output, context.value_type, &format!("%{}.sin.{sequence}", context.prefix), "mul", &adjoint, &cosine);
					sequence += 1;
					add_operand(&mut output, instruction.left, &contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Cos => {
					let sine = format!("%{}.cos.sine.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(output, "{sine} = call {ty} @recipe.sin({ty} {left})", ty = context.value_type);
					let raw = binary(&mut output, context.value_type, &format!("%{}.cos.raw.{sequence}", context.prefix), "mul", &adjoint, &sine);
					sequence += 1;
					let contribution = negate(&mut output, context.value_type, context.prefix, &raw, &mut sequence);
					add_operand(&mut output, instruction.left, &contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Tanh => {
					let square = binary(&mut output, context.value_type, &format!("%{}.tanh.square.{sequence}", context.prefix), "mul", &values[index], &values[index]);
					sequence += 1;
					let one = (context.literal)(1.0, context.value_type);
					let base = binary(&mut output, context.value_type, &format!("%{}.tanh.base.{sequence}", context.prefix), "sub", &one, &square);
					sequence += 1;
					let contribution = binary(&mut output, context.value_type, &format!("%{}.tanh.{sequence}", context.prefix), "mul", &adjoint, &base);
					sequence += 1;
					add_operand(&mut output, instruction.left, &contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Select => {
					let condition = format!("%{}.select.condition.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(output, "{condition} = call i1 @recipe.ogt({ty} {left}, {ty} {zero})", ty = context.value_type, zero = (context.literal)(0.0, context.value_type));
					let contribution = format!("%{}.select.contribution.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(
						output,
						"{contribution} = select i1 {condition}, {ty} {adjoint}, {ty} {zero}",
						ty = context.value_type,
						adjoint = adjoint,
						zero = (context.literal)(0.0, context.value_type)
					);
					add_operand(&mut output, instruction.right, &contribution, &mut adjoints, &mut first, &mut second, &mut sequence)?;
				}
				ScalarOpcode::Greater | ScalarOpcode::Constant | ScalarOpcode::Parameter => {}
			}
		}
		for (index, parameter) in parameter_for.into_iter().enumerate() {
			if let Some(parameter) = parameter {
				parameters
					.entry(parameter)
					.and_modify(|value: &mut String| add_adjoint(&mut output, context.value_type, context.prefix, value, &adjoints[index], &mut sequence))
					.or_insert_with(|| adjoints[index].clone());
			}
		}
		Ok(ScalarReverse { code: output, first_adjoint: first, second_adjoint: second, parameter_adjoint: parameters })
	}

	#[derive(Clone, Copy)]
	pub struct PredictorContext<'a> {
		pub value_type: &'a str,
		pub pointer_type: &'a str,
		pub alignment: usize,
		pub input: &'a str,
		pub row: &'a str,
		pub features: usize,
		pub weights: &'a str,
		pub parameters: usize,
		pub prefix: &'a str,
		pub literal: &'a LiteralFn<'a>,
	}

	pub struct PredictorForward {
		pub code: String,
		pub value: String,
	}

	fn parse_predictor(code: &[f64]) -> Result<Vec<(PredictorOpcode, f64)>, EmitError> {
		if code.len() % 2 != 0 {
			return Err(EmitError::WrongWidth { kind: "predictor", width: code.len() });
		}
		code.chunks_exact(2).map(|instruction| Ok((PredictorOpcode::from_i32(integer(instruction[0], "predictor opcode")?)?, instruction[1]))).collect()
	}

	/// Emit a predictor without a runtime stack, local array, or opcode switch.
	/// Stores and loads are resolved at compile time into SSA values.
	pub fn emit_predictor_forward(code: &[f64], locals: usize, context: PredictorContext<'_>) -> Result<PredictorForward, EmitError> {
		let instructions = parse_predictor(code)?;
		let mut output = String::new();
		let mut stack = Vec::new();
		let mut local_values = vec![(context.literal)(0.0, context.value_type); locals];
		let mut sequence = 0;
		let pop = |stack: &mut Vec<String>| stack.pop().ok_or(EmitError::StackUnderflow { kind: "predictor" });
		let push_binary = |output: &mut String, stack: &mut Vec<String>, operation: &str, value_type: &str, prefix: &str, sequence: &mut usize| -> Result<(), EmitError> {
			let right = pop(stack)?;
			let left = pop(stack)?;
			let name = format!("%{prefix}.predictor.{sequence}");
			*sequence += 1;
			stack.push(binary(output, value_type, &name, operation, &left, &right));
			Ok(())
		};
		for (opcode, argument) in instructions {
			match opcode {
				PredictorOpcode::Feature => {
					let feature = integer(argument, "predictor feature")?;
					if feature < 0 || feature as usize >= context.features {
						return Err(EmitError::InvalidOperand { kind: "predictor feature", value: argument });
					}
					let row_base = format!("%{}.predictor.row.base.{sequence}", context.prefix);
					sequence += 1;
					let index = format!("%{}.predictor.feature.index.{sequence}", context.prefix);
					sequence += 1;
					let pointer = format!("%{}.predictor.feature.ptr.{sequence}", context.prefix);
					sequence += 1;
					let value = format!("%{}.predictor.feature.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(output, "{row_base} = mul i32 {row}, {features}", row = context.row, features = context.features);
					let _ = writeln!(output, "{index} = add i32 {row_base}, {feature}");
					let _ = writeln!(
						output,
						"{pointer} = getelementptr inbounds {ty}, {ptrty} {input}, i32 {index}",
						ty = context.value_type,
						ptrty = context.pointer_type,
						input = context.input
					);
					let _ = writeln!(
						output,
						"{value} = load {ty}, {ptrty} {pointer}, align {align}",
						ty = context.value_type,
						ptrty = context.pointer_type,
						pointer = pointer,
						align = context.alignment
					);
					stack.push(value);
				}
				PredictorOpcode::Row => {
					let value = format!("%{}.predictor.row.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(output, "{value} = call {ty} @recipe.from.u32(i32 {row})", ty = context.value_type, row = context.row);
					stack.push(value);
				}
				PredictorOpcode::Constant => stack.push((context.literal)(argument, context.value_type)),
				PredictorOpcode::Load => {
					let local = integer(argument, "predictor local")?;
					if local < 0 || local as usize >= locals {
						return Err(EmitError::LocalIndex { index: local.max(0) as usize, locals });
					}
					stack.push(local_values[local as usize].clone());
				}
				PredictorOpcode::Store => {
					let local = integer(argument, "predictor local")?;
					if local < 0 || local as usize >= locals {
						return Err(EmitError::LocalIndex { index: local.max(0) as usize, locals });
					}
					local_values[local as usize] = pop(&mut stack)?;
				}
				PredictorOpcode::Duplicate => stack.push(stack.last().cloned().ok_or(EmitError::StackUnderflow { kind: "predictor" })?),
				PredictorOpcode::Add => push_binary(&mut output, &mut stack, "add", context.value_type, context.prefix, &mut sequence)?,
				PredictorOpcode::Subtract => push_binary(&mut output, &mut stack, "sub", context.value_type, context.prefix, &mut sequence)?,
				PredictorOpcode::Multiply => push_binary(&mut output, &mut stack, "mul", context.value_type, context.prefix, &mut sequence)?,
				PredictorOpcode::Divide => push_binary(&mut output, &mut stack, "div", context.value_type, context.prefix, &mut sequence)?,
				PredictorOpcode::Greater => {
					let right = pop(&mut stack)?;
					let left = pop(&mut stack)?;
					let name = format!("%{}.predictor.greater.{sequence}", context.prefix);
					sequence += 1;
					stack.push(predicate(&mut output, context.value_type, &name, "ogt", &left, &right));
				}
				PredictorOpcode::Choose => {
					let no = pop(&mut stack)?;
					let yes = pop(&mut stack)?;
					let condition = pop(&mut stack)?;
					let condition_true = format!("%{}.predictor.condition.{sequence}", context.prefix);
					sequence += 1;
					let value = format!("%{}.predictor.choose.{sequence}", context.prefix);
					sequence += 1;
					let _ = writeln!(
						output,
						"{condition_true} = call i1 @recipe.one({ty} {condition}, {ty} {zero})",
						ty = context.value_type,
						zero = (context.literal)(0.0, context.value_type)
					);
					let _ = writeln!(output, "{value} = select i1 {condition_true}, {ty} {yes}, {ty} {no}", ty = context.value_type);
					stack.push(value);
				}
				PredictorOpcode::Nearest => {
					let count = integer(argument.abs(), "nearest count")?;
					if count <= 0 {
						return Err(EmitError::InvalidOperand { kind: "nearest count", value: argument });
					}
					let (count, exclude) = (count as usize, argument < 0.0);
					let rows = context.parameters / (context.features + 1);
					if rows == 0 || rows * (context.features + 1) != context.parameters {
						return Err(EmitError::InvalidOperand { kind: "nearest table width", value: context.parameters as f64 });
					}
					let ty = context.value_type;
					let (ptr, align) = (context.pointer_type, context.alignment);
					let p = format!("{}.nearest.{sequence}", context.prefix);
					sequence += 1;
					let (zero, maximum) = ((context.literal)(0.0, ty), (context.literal)(f64::MAX, ty));
					// Row loop head: induction variable plus the k best (distance, target) pairs as phis.
					let _ = writeln!(output, "br label %{p}.entry\n{p}.entry:\nbr label %{p}.head\n{p}.head:");
					let _ = writeln!(output, "%{p}.i = phi i32 [ 0, %{p}.entry ], [ %{p}.i.next, %{p}.latch ]");
					for slot in 0..count {
						let _ = writeln!(output, "%{p}.d{slot} = phi {ty} [ {maximum}, %{p}.entry ], [ %{p}.d{slot}.new, %{p}.latch ]");
						let _ = writeln!(output, "%{p}.t{slot} = phi {ty} [ {zero}, %{p}.entry ], [ %{p}.t{slot}.new, %{p}.latch ]");
					}
					let _ = writeln!(output, "%{p}.more = icmp ult i32 %{p}.i, {rows}\nbr i1 %{p}.more, label %{p}.distance, label %{p}.done");
					// Squared distance between the query row and stored row i, accumulated per feature.
					let _ = writeln!(output, "{p}.distance:\nbr label %{p}.d.head\n{p}.d.head:");
					let _ = writeln!(output, "%{p}.j = phi i32 [ 0, %{p}.distance ], [ %{p}.j.next, %{p}.d.body ]");
					let _ = writeln!(output, "%{p}.acc = phi {ty} [ {zero}, %{p}.distance ], [ %{p}.acc.next, %{p}.d.body ]");
					let _ = writeln!(output, "%{p}.d.more = icmp ult i32 %{p}.j, {features}\nbr i1 %{p}.d.more, label %{p}.d.body, label %{p}.d.done", features = context.features);
					let _ = writeln!(output, "{p}.d.body:");
					let _ = writeln!(
						output,
						"%{p}.q.base = mul i32 {row}, {features}\n%{p}.q.index = add i32 %{p}.q.base, %{p}.j\n%{p}.q.ptr = getelementptr inbounds {ty}, {ptr} {input}, i32 %{p}.q.index\n%{p}.q = load {ty}, {ptr} %{p}.q.ptr, align {align}",
						row = context.row,
						features = context.features,
						input = context.input
					);
					let _ = writeln!(
						output,
						"%{p}.w.base = mul i32 %{p}.i, {features}\n%{p}.w.index = add i32 %{p}.w.base, %{p}.j\n%{p}.w.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.w.index\n%{p}.w = load {ty}, {ptr} %{p}.w.ptr, align {align}",
						features = context.features,
						weights = context.weights
					);
					let _ = writeln!(
						output,
						"%{p}.diff = call {ty} @recipe.sub({ty} %{p}.q, {ty} %{p}.w)\n%{p}.square = call {ty} @recipe.mul({ty} %{p}.diff, {ty} %{p}.diff)\n%{p}.acc.next = call {ty} @recipe.add({ty} %{p}.acc, {ty} %{p}.square)"
					);
					let _ = writeln!(output, "%{p}.j.next = add i32 %{p}.j, 1\nbr label %{p}.d.head\n{p}.d.done:");
					let candidate_distance = if exclude {
						let _ = writeln!(output, "%{p}.self = icmp eq i32 %{p}.i, {row}\n%{p}.candidate = select i1 %{p}.self, {ty} {maximum}, {ty} %{p}.acc", row = context.row);
						format!("%{p}.candidate")
					} else {
						format!("%{p}.acc")
					};
					let _ = writeln!(
						output,
						"%{p}.target.index = add i32 {base}, %{p}.i\n%{p}.target.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.target.index\n%{p}.target = load {ty}, {ptr} %{p}.target.ptr, align {align}",
						base = rows * context.features,
						weights = context.weights
					);
					// Bubble the candidate through the k slots. A displaced entry precedes every later
					// equal-distance slot because rows are visited in ascending index order.
					let (mut carry_distance, mut carry_target) = (candidate_distance, format!("%{p}.target"));
					let mut carry_precedes = "false".to_owned();
					for slot in 0..count {
						let _ = writeln!(output, "%{p}.nearer{slot} = call i1 @recipe.ogt({ty} %{p}.d{slot}, {ty} {carry_distance})");
						let _ = writeln!(output, "%{p}.equal{slot} = call i1 @recipe.oeq({ty} %{p}.d{slot}, {ty} {carry_distance})");
						let _ = writeln!(output, "%{p}.tie{slot} = and i1 %{p}.equal{slot}, {carry_precedes}");
						let _ = writeln!(output, "%{p}.swap{slot} = or i1 %{p}.nearer{slot}, %{p}.tie{slot}");
						let _ = writeln!(output, "%{p}.d{slot}.new = select i1 %{p}.swap{slot}, {ty} {carry_distance}, {ty} %{p}.d{slot}");
						let _ = writeln!(output, "%{p}.t{slot}.new = select i1 %{p}.swap{slot}, {ty} {carry_target}, {ty} %{p}.t{slot}");
						let _ = writeln!(output, "%{p}.carry.d{slot} = select i1 %{p}.swap{slot}, {ty} %{p}.d{slot}, {ty} {carry_distance}");
						let _ = writeln!(output, "%{p}.carry.t{slot} = select i1 %{p}.swap{slot}, {ty} %{p}.t{slot}, {ty} {carry_target}");
						let _ = writeln!(output, "%{p}.carry.precedes{slot} = or i1 {carry_precedes}, %{p}.swap{slot}");
						carry_distance = format!("%{p}.carry.d{slot}");
						carry_target = format!("%{p}.carry.t{slot}");
						carry_precedes = format!("%{p}.carry.precedes{slot}");
					}
					let _ = writeln!(output, "br label %{p}.latch\n{p}.latch:\n%{p}.i.next = add i32 %{p}.i, 1\nbr label %{p}.head\n{p}.done:");
					let mut sum = zero;
					for slot in 0..count {
						let name = format!("%{p}.sum{slot}");
						let _ = writeln!(output, "{name} = call {ty} @recipe.add({ty} {sum}, {ty} %{p}.t{slot})");
						sum = name;
					}
					let _ = writeln!(output, "%{p}.result = call {ty} @recipe.div({ty} {sum}, {ty} {count})", count = (context.literal)(count as f64, ty));
					stack.push(format!("%{p}.result"));
				}
				PredictorOpcode::Affine => {
					if context.features == 0 || context.parameters != 3 * context.features {
						return Err(EmitError::InvalidOperand { kind: "affine table width", value: context.parameters as f64 });
					}
					let ty = context.value_type;
					let (ptr, align) = (context.pointer_type, context.alignment);
					let p = format!("{}.affine.{sequence}", context.prefix);
					sequence += 1;
					let zero = (context.literal)(0.0, ty);
					// Feature loop head: induction variable plus the running sum as phis.
					let _ = writeln!(output, "br label %{p}.entry\n{p}.entry:\nbr label %{p}.head\n{p}.head:");
					let _ = writeln!(output, "%{p}.j = phi i32 [ 0, %{p}.entry ], [ %{p}.j.next, %{p}.body ]");
					let _ = writeln!(output, "%{p}.acc = phi {ty} [ {zero}, %{p}.entry ], [ %{p}.acc.next, %{p}.body ]");
					let _ = writeln!(output, "%{p}.more = icmp ult i32 %{p}.j, {features}\nbr i1 %{p}.more, label %{p}.body, label %{p}.done", features = context.features);
					// The table is three feature-length planes (means, scales, weights),
					// accumulated per feature as (x - mean) * scale * weight. The
					// weights pointer is already advanced to this node's parameter
					// span, so the plane indices are node-relative.
					let _ = writeln!(output, "{p}.body:");
					let _ = writeln!(
						output,
						"%{p}.q.base = mul i32 {row}, {features}\n%{p}.q.index = add i32 %{p}.q.base, %{p}.j\n%{p}.q.ptr = getelementptr inbounds {ty}, {ptr} {input}, i32 %{p}.q.index\n%{p}.q = load {ty}, {ptr} %{p}.q.ptr, align {align}",
						row = context.row,
						features = context.features,
						input = context.input
					);
					let _ = writeln!(
						output,
						"%{p}.mean.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.j\n%{p}.mean = load {ty}, {ptr} %{p}.mean.ptr, align {align}",
						weights = context.weights
					);
					let _ = writeln!(
						output,
						"%{p}.scale.index = add i32 %{p}.j, {features}\n%{p}.scale.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.scale.index\n%{p}.scale = load {ty}, {ptr} %{p}.scale.ptr, align {align}",
						features = context.features,
						weights = context.weights
					);
					let _ = writeln!(
						output,
						"%{p}.weight.index = add i32 %{p}.scale.index, {features}\n%{p}.weight.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.weight.index\n%{p}.weight = load {ty}, {ptr} %{p}.weight.ptr, align {align}",
						features = context.features,
						weights = context.weights
					);
					let _ = writeln!(
						output,
						"%{p}.centered = call {ty} @recipe.sub({ty} %{p}.q, {ty} %{p}.mean)\n%{p}.scaled = call {ty} @recipe.mul({ty} %{p}.centered, {ty} %{p}.scale)\n%{p}.term = call {ty} @recipe.mul({ty} %{p}.scaled, {ty} %{p}.weight)\n%{p}.acc.next = call {ty} @recipe.add({ty} %{p}.acc, {ty} %{p}.term)"
					);
					let _ = writeln!(output, "%{p}.j.next = add i32 %{p}.j, 1\nbr label %{p}.head\n{p}.done:");
					stack.push(format!("%{p}.acc"));
				}
				PredictorOpcode::Gaussian => {
					let width = 2 * context.features + 2;
					if context.features == 0 || context.parameters == 0 || context.parameters % width != 0 {
						return Err(EmitError::InvalidOperand { kind: "gaussian table width", value: context.parameters as f64 });
					}
					let classes = context.parameters / width;
					let ty = context.value_type;
					let (ptr, align) = (context.pointer_type, context.alignment);
					let p = format!("{}.gaussian.{sequence}", context.prefix);
					sequence += 1;
					let lowest = (context.literal)(f64::MIN, ty);
					// The table is four planes: per-class means, per-class scales, class
					// bases, and class labels. The score for a class starts at its base and
					// accumulates (x - mean)^2 * scale for each feature.
					let (scales, bases, labels) = (classes * context.features, 2 * classes * context.features, 2 * classes * context.features + classes);
					let _ = writeln!(output, "br label %{p}.entry\n{p}.entry:");
					let _ = writeln!(
						output,
						"%{p}.first.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 {labels}\n%{p}.first = load {ty}, {ptr} %{p}.first.ptr, align {align}\nbr label %{p}.head\n{p}.head:",
						weights = context.weights
					);
					let _ = writeln!(output, "%{p}.c = phi i32 [ 0, %{p}.entry ], [ %{p}.c.next, %{p}.latch ]");
					let _ = writeln!(output, "%{p}.best = phi {ty} [ {lowest}, %{p}.entry ], [ %{p}.best.new, %{p}.latch ]");
					let _ = writeln!(output, "%{p}.label = phi {ty} [ %{p}.first, %{p}.entry ], [ %{p}.label.new, %{p}.latch ]");
					let _ = writeln!(output, "%{p}.more = icmp ult i32 %{p}.c, {classes}\nbr i1 %{p}.more, label %{p}.score, label %{p}.done");
					let _ = writeln!(
						output,
						"{p}.score:\n%{p}.base.index = add i32 %{p}.c, {bases}\n%{p}.base.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.base.index\n%{p}.base = load {ty}, {ptr} %{p}.base.ptr, align {align}\nbr label %{p}.f.head\n{p}.f.head:",
						weights = context.weights
					);
					let _ = writeln!(output, "%{p}.j = phi i32 [ 0, %{p}.score ], [ %{p}.j.next, %{p}.f.body ]");
					let _ = writeln!(output, "%{p}.acc = phi {ty} [ %{p}.base, %{p}.score ], [ %{p}.acc.next, %{p}.f.body ]");
					let _ = writeln!(output, "%{p}.f.more = icmp ult i32 %{p}.j, {features}\nbr i1 %{p}.f.more, label %{p}.f.body, label %{p}.f.done", features = context.features);
					let _ = writeln!(output, "{p}.f.body:");
					let _ = writeln!(
						output,
						"%{p}.q.base = mul i32 {row}, {features}\n%{p}.q.index = add i32 %{p}.q.base, %{p}.j\n%{p}.q.ptr = getelementptr inbounds {ty}, {ptr} {input}, i32 %{p}.q.index\n%{p}.q = load {ty}, {ptr} %{p}.q.ptr, align {align}",
						row = context.row,
						features = context.features,
						input = context.input
					);
					let _ = writeln!(
						output,
						"%{p}.mean.base = mul i32 %{p}.c, {features}\n%{p}.mean.index = add i32 %{p}.mean.base, %{p}.j\n%{p}.mean.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.mean.index\n%{p}.mean = load {ty}, {ptr} %{p}.mean.ptr, align {align}",
						features = context.features,
						weights = context.weights
					);
					let _ = writeln!(
						output,
						"%{p}.scale.index = add i32 %{p}.mean.index, {scales}\n%{p}.scale.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.scale.index\n%{p}.scale = load {ty}, {ptr} %{p}.scale.ptr, align {align}",
						weights = context.weights
					);
					let _ = writeln!(
						output,
						"%{p}.centered = call {ty} @recipe.sub({ty} %{p}.q, {ty} %{p}.mean)\n%{p}.square = call {ty} @recipe.mul({ty} %{p}.centered, {ty} %{p}.centered)\n%{p}.term = call {ty} @recipe.mul({ty} %{p}.square, {ty} %{p}.scale)\n%{p}.acc.next = call {ty} @recipe.add({ty} %{p}.acc, {ty} %{p}.term)"
					);
					let _ = writeln!(output, "%{p}.j.next = add i32 %{p}.j, 1\nbr label %{p}.f.head\n{p}.f.done:");
					let _ = writeln!(
						output,
						"%{p}.target.index = add i32 %{p}.c, {labels}\n%{p}.target.ptr = getelementptr inbounds {ty}, {ptr} {weights}, i32 %{p}.target.index\n%{p}.target = load {ty}, {ptr} %{p}.target.ptr, align {align}",
						weights = context.weights
					);
					let _ = writeln!(output, "%{p}.swap = call i1 @recipe.ogt({ty} %{p}.acc, {ty} %{p}.best)");
					let _ = writeln!(output, "%{p}.best.new = select i1 %{p}.swap, {ty} %{p}.acc, {ty} %{p}.best");
					let _ = writeln!(output, "%{p}.label.new = select i1 %{p}.swap, {ty} %{p}.target, {ty} %{p}.label");
					let _ = writeln!(output, "br label %{p}.latch\n{p}.latch:\n%{p}.c.next = add i32 %{p}.c, 1\nbr label %{p}.head\n{p}.done:");
					stack.push(format!("%{p}.label"));
				}
			}
		}
		if stack.len() != 1 {
			return Err(EmitError::StackDepth { kind: "predictor", depth: stack.len() });
		}
		Ok(PredictorForward { code: output, value: stack.remove(0) })
	}

	#[derive(Clone, Copy, Debug, PartialEq, Eq)]
	pub enum NormalizeMode {
		Batch,
		Layer,
		/// Root-mean-square statistics use layer-shaped groups with a zero mean.
		Rms,
		/// Stored batch statistics used by evaluation and inference.
		Evaluation,
		/// Row L2 statistics use layer-shaped groups with a zero mean, no item
		/// average, and an epsilon floor on the norm.
		L2,
	}

	impl NormalizeMode {
		/// Layer-shaped modes group one row position; the group holds `width`
		/// channels, so a row of `channels` splits into `channels / width` heads.
		pub fn per_row(self) -> bool {
			matches!(self, Self::Layer | Self::Rms | Self::L2)
		}
	}

	#[derive(Clone, Copy)]
	pub struct NormalizeContext<'a> {
		pub value_type: &'a str,
		pub pointer_type: &'a str,
		pub alignment: usize,
		pub source_value: &'a str,
		pub context: &'a str,
		pub rows: &'a str,
		pub channels: usize,
		pub length: usize,
		/// Channels per group in the per-row modes.
		pub width: usize,
		/// Channels the per-row modes normalize; the rest pass through.
		pub span: usize,
		/// The per-channel scale applied after normalization, when the node carries one.
		pub weight: Option<&'a str>,
		pub mode: NormalizeMode,
		pub prefix: &'a str,
	}

	impl NormalizeContext<'_> {
		fn shape(&self) -> GroupShape<'_> {
			GroupShape { mode: self.mode, channels: self.channels, length: self.length, width: self.width, span: self.span, rows: self.rows }
		}
	}

	pub struct NormalizeFragment {
		pub code: String,
		pub value: String,
	}

	#[derive(Clone, Copy)]
	struct GroupShape<'a> {
		mode: NormalizeMode,
		channels: usize,
		length: usize,
		width: usize,
		span: usize,
		rows: &'a str,
	}

	struct GroupIndex {
		group: String,
		groups: String,
		channel: String,
		/// The predicate that holds where the element normalizes, when a channel of
		/// the row passes through instead.
		inside: Option<String>,
	}

	/// Names the statistics group of one element. A per-row mode splits the leading
	/// `span` channels into groups of `width`, so one attention head owns one group;
	/// a channel past the span has no group and clamps to the first one.
	fn emit_group_index(code: &mut String, prefix: &str, shape: GroupShape<'_>, element: &str) -> GroupIndex {
		let (channels, length, elements) = (shape.channels, shape.length, shape.channels * shape.length);
		let (row, local, position) = (format!("%{prefix}.row"), format!("%{prefix}.local"), format!("%{prefix}.position"));
		let (group, groups, channel) = (format!("%{prefix}.group"), format!("%{prefix}.groups"), format!("%{prefix}.channel"));
		let _ = writeln!(code, "{row} = udiv i32 {element}, {elements}");
		let _ = writeln!(code, "{local} = urem i32 {element}, {elements}");
		let _ = writeln!(code, "{position} = urem i32 {local}, {length}");
		let _ = writeln!(code, "{channel} = udiv i32 {local}, {length}");
		if !shape.mode.per_row() {
			let _ = writeln!(code, "{group} = add i32 {channel}, 0");
			let _ = writeln!(code, "{groups} = add i32 0, {channels}");
			return GroupIndex { group, groups, channel, inside: None };
		}
		let (span, width) = (shape.span, shape.width);
		let inside = (span < channels).then(|| format!("%{prefix}.inside"));
		let head = format!("%{prefix}.head");
		match &inside {
			Some(inside) => {
				let _ = writeln!(code, "{inside} = icmp ult i32 {channel}, {span}");
				let _ = writeln!(code, "%{prefix}.head.whole = udiv i32 {channel}, {width}");
				let _ = writeln!(code, "{head} = select i1 {inside}, i32 %{prefix}.head.whole, i32 0");
			}
			None => {
				let _ = writeln!(code, "{head} = udiv i32 {channel}, {width}");
			}
		}
		let plane = length * (span / width);
		let _ = writeln!(code, "%{prefix}.head.base = mul i32 {head}, {length}");
		let _ = writeln!(code, "%{prefix}.row.base = mul i32 {row}, {plane}");
		let _ = writeln!(code, "%{prefix}.row.group = add i32 %{prefix}.row.base, %{prefix}.head.base");
		let _ = writeln!(code, "{group} = add i32 %{prefix}.row.group, {position}");
		let _ = writeln!(code, "{groups} = mul i32 {rows}, {plane}", rows = shape.rows);
		GroupIndex { group, groups, channel, inside }
	}

	/// Emit one normalized element from the fixed statistics arena. The arena is
	/// laid out as `mean[group]` followed by `scale[group]`, with `groups` fixed by
	/// the normalization mode. A training caller must run a separate fixed stats
	/// pass before this fragment; evaluation and inference reuse the stored arena.
	pub fn emit_normalize(context: NormalizeContext<'_>, element: &str) -> NormalizeFragment {
		let mut output = String::new();
		let prefix = format!("{}.normalize", context.prefix);
		let index = emit_group_index(&mut output, &prefix, context.shape(), element);
		let (group, groups) = (&index.group, &index.groups);
		let scale_index = format!("%{prefix}.scale.index");
		let mean_pointer = format!("%{prefix}.mean.ptr");
		let scale_pointer = format!("%{prefix}.scale.ptr");
		let mean = format!("%{prefix}.mean");
		let scale = format!("%{prefix}.scale");
		let centered = format!("%{prefix}.centered");
		let value = format!("%{prefix}.value");
		let _ = writeln!(output, "{scale_index} = add i32 {groups}, {group}");
		let _ = writeln!(
			output,
			"{mean_pointer} = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 {group}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			context_ptr = context.context
		);
		let _ = writeln!(
			output,
			"{scale_pointer} = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 {scale_index}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			context_ptr = context.context
		);
		let _ = writeln!(
			output,
			"{mean} = load {ty}, {ptrty} {mean_pointer}, align {align}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			mean_pointer = mean_pointer,
			align = context.alignment
		);
		let _ = writeln!(
			output,
			"{scale} = load {ty}, {ptrty} {scale_pointer}, align {align}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			scale_pointer = scale_pointer,
			align = context.alignment
		);
		let _ = writeln!(output, "{centered} = call {ty} @recipe.sub({ty} {source}, {ty} {mean})", ty = context.value_type, source = context.source_value);
		let _ = writeln!(output, "{value} = call {ty} @recipe.mul({ty} {centered}, {ty} {scale})", ty = context.value_type);
		let value = match context.weight {
			None => value,
			Some(weight) => {
				let weight_pointer = format!("%{prefix}.weight.ptr");
				let weight_value = format!("%{prefix}.weight");
				let scaled = format!("%{prefix}.scaled");
				let column = weight_column(&mut output, &prefix, &index);
				let _ = writeln!(output, "{weight_pointer} = getelementptr inbounds {ty}, {ptrty} {weight}, i32 {column}", ty = context.value_type, ptrty = context.pointer_type);
				let _ =
					writeln!(output, "{weight_value} = load {ty}, {ptrty} {weight_pointer}, align {align}", ty = context.value_type, ptrty = context.pointer_type, align = context.alignment);
				let _ = writeln!(output, "{scaled} = call {ty} @recipe.mul({ty} {value}, {ty} {weight_value})", ty = context.value_type);
				scaled
			}
		};
		let Some(inside) = &index.inside else { return NormalizeFragment { code: output, value } };
		let passed = format!("%{prefix}.passed");
		let _ = writeln!(output, "{passed} = select i1 {inside}, {ty} {value}, {ty} {source}", ty = context.value_type, source = context.source_value);
		NormalizeFragment { code: output, value: passed }
	}

	/// The scale column of one element. Only the normalized span carries a scale, so
	/// a passing channel reads the first column and drops the product.
	fn weight_column(code: &mut String, prefix: &str, index: &GroupIndex) -> String {
		let Some(inside) = &index.inside else { return index.channel.clone() };
		let column = format!("%{prefix}.weight.channel");
		let _ = writeln!(code, "{column} = select i1 {inside}, i32 {channel}, i32 0", channel = index.channel);
		column
	}

	#[derive(Clone, Copy)]
	pub struct NormalizeReverseContext<'a> {
		pub value_type: &'a str,
		pub pointer_type: &'a str,
		pub alignment: usize,
		pub state_type: &'a str,
		pub state_zero: &'a str,
		pub context: &'a str,
		pub rows: &'a str,
		pub channels: usize,
		pub length: usize,
		/// Channels per group in the per-row modes.
		pub width: usize,
		/// Channels the per-row modes normalize; the rest pass through.
		pub span: usize,
		/// The per-channel scale the forward pass applied, when the node carries one.
		pub weight: Option<&'a str>,
		/// The node input, which a weighted node re-normalizes in reverse.
		pub source: &'a str,
		pub mode: NormalizeMode,
		pub prefix: &'a str,
	}

	impl NormalizeReverseContext<'_> {
		fn shape(&self) -> GroupShape<'_> {
			GroupShape { mode: self.mode, channels: self.channels, length: self.length, width: self.width, span: self.span, rows: self.rows }
		}
	}

	pub struct NormalizeReverseFragment {
		pub code: String,
		pub contribution: String,
	}

	/// Accumulate each group's delta sum and delta-output projection in the state
	/// format, like the loss reduction, and store them as per-item means in the
	/// model format. Batch groups span every row, so the raw counts and sums can
	/// exceed the finite range of narrow model formats; the means cannot.
	pub fn emit_normalize_reverse_stats(context: NormalizeReverseContext<'_>, delta: &str, output_value: &str) -> String {
		let mut code = String::new();
		let elements = context.channels * context.length;
		let prefix = context.prefix;
		let group = format!("%{prefix}.group");
		let groups = format!("%{prefix}.groups");
		let items = format!("%{prefix}.items");
		let heads = context.span / context.width;
		match context.mode {
			NormalizeMode::Batch => {
				let _ = writeln!(code, "{groups} = add i32 0, {}", context.channels);
				let _ = writeln!(code, "{items} = mul i32 {}, {}", context.rows, context.length);
			}
			NormalizeMode::Layer | NormalizeMode::Rms | NormalizeMode::L2 => {
				let _ = writeln!(code, "{groups} = mul i32 {}, {}", context.rows, context.length * heads);
				let _ = writeln!(code, "{items} = add i32 0, {}", context.width);
			}
			NormalizeMode::Evaluation => return code,
		}
		let _ = writeln!(code, "br label %{prefix}.entry");
		let _ = writeln!(code, "{prefix}.entry:");
		let _ = writeln!(code, "br label %{prefix}.group.loop");
		let _ = writeln!(code, "{prefix}.group.loop:");
		let _ = writeln!(code, "{group} = phi i32 [ %tid, %{prefix}.entry ], [ %{prefix}.group.next, %{prefix}.store ]");
		let _ = writeln!(code, "%{prefix}.group.more = icmp ult i32 {group}, {groups}");
		let _ = writeln!(code, "br i1 %{prefix}.group.more, label %{prefix}.item.loop, label %{prefix}.done");
		let _ = writeln!(code, "{prefix}.item.loop:");
		let _ = writeln!(code, "%{prefix}.p = phi i32 [ 0, %{prefix}.group.loop ], [ %{prefix}.p.next, %{prefix}.item.step ]");
		let _ = writeln!(code, "%{prefix}.sum = phi {ty} [ {zero}, %{prefix}.group.loop ], [ %{prefix}.sum.next, %{prefix}.item.step ]", ty = context.state_type, zero = context.state_zero);
		let _ = writeln!(
			code,
			"%{prefix}.projected = phi {ty} [ {zero}, %{prefix}.group.loop ], [ %{prefix}.projected.next, %{prefix}.item.step ]",
			ty = context.state_type,
			zero = context.state_zero
		);
		let _ = writeln!(code, "%{prefix}.item.more = icmp ult i32 %{prefix}.p, {items}");
		let _ = writeln!(code, "br i1 %{prefix}.item.more, label %{prefix}.item.step, label %{prefix}.store");
		let _ = writeln!(code, "{prefix}.item.step:");
		match context.mode {
			NormalizeMode::Batch => {
				let _ = writeln!(code, "%{prefix}.row = udiv i32 %{prefix}.p, {}", context.length);
				let _ = writeln!(code, "%{prefix}.position = urem i32 %{prefix}.p, {}", context.length);
				let _ = writeln!(code, "%{prefix}.row.base = mul i32 %{prefix}.row, {elements}");
				let _ = writeln!(code, "%{prefix}.channel.base = mul i32 {group}, {}", context.length);
			}
			NormalizeMode::Layer | NormalizeMode::Rms | NormalizeMode::L2 => {
				let _ = writeln!(code, "%{prefix}.row = udiv i32 {group}, {}", context.length * heads);
				let _ = writeln!(code, "%{prefix}.row.local = urem i32 {group}, {}", context.length * heads);
				let _ = writeln!(code, "%{prefix}.head = udiv i32 %{prefix}.row.local, {}", context.length);
				let _ = writeln!(code, "%{prefix}.position = urem i32 %{prefix}.row.local, {}", context.length);
				let _ = writeln!(code, "%{prefix}.head.base = mul i32 %{prefix}.head, {}", context.width);
				let _ = writeln!(code, "%{prefix}.channel = add i32 %{prefix}.head.base, %{prefix}.p");
				let _ = writeln!(code, "%{prefix}.row.base = mul i32 %{prefix}.row, {elements}");
				let _ = writeln!(code, "%{prefix}.channel.base = mul i32 %{prefix}.channel, {}", context.length);
			}
			NormalizeMode::Evaluation => unreachable!(),
		}
		let _ = writeln!(code, "%{prefix}.local = add i32 %{prefix}.channel.base, %{prefix}.position");
		let _ = writeln!(code, "%{prefix}.index = add i32 %{prefix}.row.base, %{prefix}.local");
		let _ = writeln!(code, "%{prefix}.delta.ptr = getelementptr inbounds {ty}, {ptrty} {delta}, i32 %{prefix}.index", ty = context.value_type, ptrty = context.pointer_type);
		let _ = writeln!(
			code,
			"%{prefix}.output.ptr = getelementptr inbounds {ty}, {ptrty} {output}, i32 %{prefix}.index",
			ty = context.value_type,
			ptrty = context.pointer_type,
			output = output_value
		);
		let _ = writeln!(code, "%{prefix}.delta.model = load {ty}, {ptrty} %{prefix}.delta.ptr, align {align}", ty = context.value_type, ptrty = context.pointer_type, align = context.alignment);
		let _ = writeln!(code, "%{prefix}.output.model = load {ty}, {ptrty} %{prefix}.output.ptr, align {align}", ty = context.value_type, ptrty = context.pointer_type, align = context.alignment);
		// A weighted node stored `weight * normalized`, so the reverse formula takes
		// the delta scaled by the weight and re-normalizes the input for the projection.
		let (delta_model, output_model) = match context.weight {
			Some(weight) => {
				let _ = writeln!(code, "%{prefix}.scale.index = add i32 {groups}, {group}");
				let _ = writeln!(
					code,
					"%{prefix}.scale.ptr = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 %{prefix}.scale.index",
					ty = context.value_type,
					ptrty = context.pointer_type,
					context_ptr = context.context
				);
				let _ = writeln!(
					code,
					"%{prefix}.scale = load {ty}, {ptrty} %{prefix}.scale.ptr, align {align}",
					ty = context.value_type,
					ptrty = context.pointer_type,
					align = context.alignment
				);
				let _ = writeln!(
					code,
					"%{prefix}.source.ptr = getelementptr inbounds {ty}, {ptrty} {source}, i32 %{prefix}.index",
					ty = context.value_type,
					ptrty = context.pointer_type,
					source = context.source
				);
				let _ = writeln!(
					code,
					"%{prefix}.source.model = load {ty}, {ptrty} %{prefix}.source.ptr, align {align}",
					ty = context.value_type,
					ptrty = context.pointer_type,
					align = context.alignment
				);
				let _ = writeln!(code, "%{prefix}.normalized = call {ty} @recipe.mul({ty} %{prefix}.source.model, {ty} %{prefix}.scale)", ty = context.value_type);
				let _ = writeln!(code, "%{prefix}.weight.ptr = getelementptr inbounds {ty}, {ptrty} {weight}, i32 %{prefix}.channel", ty = context.value_type, ptrty = context.pointer_type);
				let _ = writeln!(
					code,
					"%{prefix}.weight = load {ty}, {ptrty} %{prefix}.weight.ptr, align {align}",
					ty = context.value_type,
					ptrty = context.pointer_type,
					align = context.alignment
				);
				let _ = writeln!(code, "%{prefix}.delta.weighted = call {ty} @recipe.mul({ty} %{prefix}.delta.model, {ty} %{prefix}.weight)", ty = context.value_type);
				(format!("%{prefix}.delta.weighted"), format!("%{prefix}.normalized"))
			}
			None => (format!("%{prefix}.delta.model"), format!("%{prefix}.output.model")),
		};
		let _ = writeln!(code, "%{prefix}.delta = call {state} @recipe.state.from.model({ty} {delta_model})", state = context.state_type, ty = context.value_type);
		let _ = writeln!(code, "%{prefix}.output = call {state} @recipe.state.from.model({ty} {output_model})", state = context.state_type, ty = context.value_type);
		if matches!(context.mode, NormalizeMode::Rms | NormalizeMode::L2) {
			let _ = writeln!(code, "%{prefix}.sum.next = call {ty} @recipe.state.add({ty} %{prefix}.sum, {ty} {zero})", ty = context.state_type, zero = context.state_zero);
		} else {
			let _ = writeln!(code, "%{prefix}.sum.next = call {ty} @recipe.state.add({ty} %{prefix}.sum, {ty} %{prefix}.delta)", ty = context.state_type);
		}
		let _ = writeln!(code, "%{prefix}.projection = call {ty} @recipe.state.mul({ty} %{prefix}.delta, {ty} %{prefix}.output)", ty = context.state_type);
		let _ = writeln!(code, "%{prefix}.projected.next = call {ty} @recipe.state.add({ty} %{prefix}.projected, {ty} %{prefix}.projection)", ty = context.state_type);
		let _ = writeln!(code, "%{prefix}.p.next = add i32 %{prefix}.p, 1");
		let _ = writeln!(code, "br label %{prefix}.item.loop");
		let _ = writeln!(code, "{prefix}.store:");
		let _ = writeln!(code, "%{prefix}.items.value = call {ty} @recipe.state.from.u32(i32 {items})", ty = context.state_type);
		let _ = writeln!(code, "%{prefix}.sum.mean = call {ty} @recipe.state.div({ty} %{prefix}.sum, {ty} %{prefix}.items.value)", ty = context.state_type);
		// An L2 group scales by its norm rather than its root mean square, so its
		// reverse projection is the whole sum over the group.
		let projected_divisor = if context.mode == NormalizeMode::L2 { "one" } else { "items.value" };
		let _ = writeln!(code, "%{prefix}.one = call {ty} @recipe.state.from.u32(i32 1)", ty = context.state_type);
		let _ = writeln!(code, "%{prefix}.projected.mean = call {ty} @recipe.state.div({ty} %{prefix}.projected, {ty} %{prefix}.{projected_divisor})", ty = context.state_type);
		let _ = writeln!(code, "%{prefix}.sum.model = call {ty} @recipe.model.from.state({state} %{prefix}.sum.mean)", ty = context.value_type, state = context.state_type);
		let _ = writeln!(code, "%{prefix}.projected.model = call {ty} @recipe.model.from.state({state} %{prefix}.projected.mean)", ty = context.value_type, state = context.state_type);
		let _ = writeln!(code, "%{prefix}.sum.base = mul i32 {groups}, 2");
		let _ = writeln!(code, "%{prefix}.projected.base = mul i32 {groups}, 3");
		let _ = writeln!(code, "%{prefix}.sum.index = add i32 %{prefix}.sum.base, {group}");
		let _ = writeln!(code, "%{prefix}.projected.index = add i32 %{prefix}.projected.base, {group}");
		let _ = writeln!(
			code,
			"%{prefix}.sum.ptr = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 %{prefix}.sum.index",
			ty = context.value_type,
			ptrty = context.pointer_type,
			context_ptr = context.context
		);
		let _ = writeln!(
			code,
			"%{prefix}.projected.ptr = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 %{prefix}.projected.index",
			ty = context.value_type,
			ptrty = context.pointer_type,
			context_ptr = context.context
		);
		let _ = writeln!(code, "store {ty} %{prefix}.sum.model, {ptrty} %{prefix}.sum.ptr, align {align}", ty = context.value_type, ptrty = context.pointer_type, align = context.alignment);
		let _ = writeln!(
			code,
			"store {ty} %{prefix}.projected.model, {ptrty} %{prefix}.projected.ptr, align {align}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			align = context.alignment
		);
		let _ = writeln!(code, "%{prefix}.group.next = add i32 {group}, %threads");
		let _ = writeln!(code, "br label %{prefix}.group.loop");
		let _ = writeln!(code, "{prefix}.done:");
		code
	}

	/// Emit the fixed reverse formula for a normalized element. In training modes
	/// the stats pass must have populated the per-item delta mean and projection
	/// mean for each group in the context arena; keeping the group reductions as
	/// means bounds them by the item magnitudes, so no batch-sized count or sum
	/// ever has to be representable in the model arithmetic format. Evaluation
	/// uses stored scale directly because its stats are not differentiated.
	pub fn emit_normalize_reverse(context: NormalizeReverseContext<'_>, element: &str, delta: &str, output_value: &str) -> NormalizeReverseFragment {
		let mut code = String::new();
		let prefix = format!("{}.normalize.reverse", context.prefix);
		let index = emit_group_index(&mut code, &prefix, context.shape(), element);
		let (group, groups) = (&index.group, &index.groups);
		let passing = delta;
		let scale_index = format!("%{prefix}.scale.index");
		let sum_base = format!("%{prefix}.sum.base");
		let projected_base = format!("%{prefix}.projected.base");
		let sum_index = format!("%{prefix}.sum.index");
		let projected_index = format!("%{prefix}.projected.index");
		let scale_pointer = format!("%{prefix}.scale.ptr");
		let sum_pointer = format!("%{prefix}.sum.ptr");
		let projected_pointer = format!("%{prefix}.projected.ptr");
		let scale = format!("%{prefix}.scale");
		let sum = format!("%{prefix}.sum");
		let projected = format!("%{prefix}.projected");
		let _ = writeln!(code, "{scale_index} = add i32 {groups}, {group}");
		let _ = writeln!(code, "{sum_base} = mul i32 {groups}, 2");
		let _ = writeln!(code, "{projected_base} = mul i32 {groups}, 3");
		let _ = writeln!(code, "{sum_index} = add i32 {sum_base}, {group}");
		let _ = writeln!(code, "{projected_index} = add i32 {projected_base}, {group}");
		let _ = writeln!(
			code,
			"{scale_pointer} = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 {scale_index}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			context_ptr = context.context
		);
		let _ = writeln!(
			code,
			"{sum_pointer} = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 {sum_index}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			context_ptr = context.context
		);
		let _ = writeln!(
			code,
			"{projected_pointer} = getelementptr inbounds {ty}, {ptrty} {context_ptr}, i32 {projected_index}",
			ty = context.value_type,
			ptrty = context.pointer_type,
			context_ptr = context.context
		);
		let align = context.alignment;
		let _ = writeln!(code, "{scale} = load {ty}, {ptrty} {scale_pointer}, align {align}", ty = context.value_type, ptrty = context.pointer_type);
		if context.mode == NormalizeMode::Evaluation {
			let contribution = format!("%{}.normalize.reverse.fixed", context.prefix);
			let _ = writeln!(code, "{contribution} = call {ty} @recipe.mul({ty} {delta}, {ty} {scale})", ty = context.value_type);
			return NormalizeReverseFragment { code, contribution };
		}
		let _ = writeln!(code, "{sum} = load {ty}, {ptrty} {sum_pointer}, align {align}", ty = context.value_type, ptrty = context.pointer_type);
		let _ = writeln!(code, "{projected} = load {ty}, {ptrty} {projected_pointer}, align {align}", ty = context.value_type, ptrty = context.pointer_type);
		let output_projection = format!("%{prefix}.output.projection");
		let centered = format!("%{prefix}.centered");
		let numerator = format!("%{prefix}.numerator");
		let contribution = format!("%{prefix}.contribution");
		// A weighted node stored `weight * normalized`: the formula takes the delta
		// scaled by the weight and the re-normalized input.
		let (delta, output_value) = match context.weight {
			Some(weight) => {
				let source_pointer = format!("%{prefix}.source.ptr");
				let source_value = format!("%{prefix}.source");
				let normalized = format!("%{prefix}.normalized");
				let weight_pointer = format!("%{prefix}.weight.ptr");
				let weight_value = format!("%{prefix}.weight");
				let weighted = format!("%{prefix}.delta.weighted");
				let _ = writeln!(
					code,
					"{source_pointer} = getelementptr inbounds {ty}, {ptrty} {source}, i32 {element}",
					ty = context.value_type,
					ptrty = context.pointer_type,
					source = context.source
				);
				let _ = writeln!(code, "{source_value} = load {ty}, {ptrty} {source_pointer}, align {align}", ty = context.value_type, ptrty = context.pointer_type);
				let _ = writeln!(code, "{normalized} = call {ty} @recipe.mul({ty} {source_value}, {ty} {scale})", ty = context.value_type);
				let column = weight_column(&mut code, &prefix, &index);
				let _ = writeln!(code, "{weight_pointer} = getelementptr inbounds {ty}, {ptrty} {weight}, i32 {column}", ty = context.value_type, ptrty = context.pointer_type);
				let _ = writeln!(code, "{weight_value} = load {ty}, {ptrty} {weight_pointer}, align {align}", ty = context.value_type, ptrty = context.pointer_type);
				let _ = writeln!(code, "{weighted} = call {ty} @recipe.mul({ty} {delta}, {ty} {weight_value})", ty = context.value_type);
				(weighted, normalized)
			}
			None => (delta.to_owned(), output_value.to_owned()),
		};
		let _ = writeln!(code, "{output_projection} = call {ty} @recipe.mul({ty} {output_value}, {ty} {projected})", ty = context.value_type);
		let _ = writeln!(code, "{centered} = call {ty} @recipe.sub({ty} {delta}, {ty} {sum})", ty = context.value_type);
		let _ = writeln!(code, "{numerator} = call {ty} @recipe.sub({ty} {centered}, {ty} {output_projection})", ty = context.value_type);
		let _ = writeln!(code, "{contribution} = call {ty} @recipe.mul({ty} {scale}, {ty} {numerator})", ty = context.value_type);
		// A channel outside the span keeps its own adjoint, so the delta passes through.
		let Some(inside) = &index.inside else { return NormalizeReverseFragment { code, contribution } };
		let passed = format!("%{prefix}.passed");
		let _ = writeln!(code, "{passed} = select i1 {inside}, {ty} {contribution}, {ty} {passing}", ty = context.value_type);
		NormalizeReverseFragment { code, contribution: passed }
	}
}

use program_ir::{PredictorOpcode, ScalarOpcode};
use std::sync::atomic::AtomicUsize;

#[derive(Clone)]
pub(crate) struct NativeLayout {
	pub values: Vec<usize>,
	pub contexts: Vec<usize>,
	pub adjoints: Vec<usize>,
	pub values_bytes: usize,
	pub contexts_bytes: usize,
	pub adjoints_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum BackendTarget {
	Cpu { target: String },
	Amd { architecture: String },
	Nvidia { architecture: String },
}

impl BackendTarget {
	fn backend(&self) -> Backend {
		match self {
			Self::Cpu { .. } => Backend::Cpu,
			Self::Amd { .. } => Backend::Amd,
			Self::Nvidia { .. } => Backend::Nvidia,
		}
	}

	fn artifact_extension(&self) -> &'static str {
		match self {
			Self::Cpu { .. } => "so",
			Self::Amd { .. } => "hsaco",
			Self::Nvidia { .. } => "ptx",
		}
	}

	fn validate(&self) -> Result<()> {
		match self {
			Self::Cpu { target } => {
				let (target, compiler, cpu, features) = cpu_identity(target)?;
				let configured = option_env!("RECIPE_CPU_TARGET").ok_or_else(|| RecipeError::new("CPU native target is unavailable"))?;
				require(target == configured, format!("CPU target {target:?} does not match configured target {configured:?}"))?;
				require(!compiler.is_empty() && !cpu.is_empty() && !features.is_empty(), "CPU native target identity is incomplete")?;
			}
			Self::Amd { architecture } => {
				let suffix = architecture.strip_prefix("gfx").unwrap_or("");
				require(!suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit() || byte.is_ascii_lowercase()), "AMD architecture must be an exact gfx target")?;
			}
			Self::Nvidia { architecture } => {
				let suffix = architecture.strip_prefix("sm_").unwrap_or("");
				require(!suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit() || byte.is_ascii_lowercase()), "NVIDIA architecture must be an exact sm target")?;
			}
		}
		Ok(())
	}
}

fn cpu_identity_field<'a>(field: &'a str, name: &str) -> Result<&'a str> {
	let prefix = format!("{name}=");
	field.strip_prefix(&prefix).filter(|value| !value.is_empty()).ok_or_else(|| RecipeError::new(format!("CPU native target field {name:?} is absent")))
}

const LLVM_OPAQUE_POINTER_DEFAULT_MAJOR: u32 = 15;
const APPLE_CLANG_BROKEN_LICM_PROMOTION_PREFIX: &str = "Apple clang version 14.";
fn cpu_llvm_major(compiler: &str) -> Result<u32> {
	compiler
		.split_once("clang version ")
		.and_then(|(_, version)| version.split('.').next())
		.and_then(|major| major.parse().ok())
		.filter(|major| *major != 0)
		.ok_or_else(|| RecipeError::new("CPU compiler LLVM major version is absent"))
}

fn cpu_identity(target: &str) -> Result<(&str, &str, &str, &str)> {
	let mut fields = target.split(';');
	let target = cpu_identity_field(fields.next().unwrap_or_default(), "target")?;
	let compiler = cpu_identity_field(fields.next().unwrap_or_default(), "compiler")?;
	let cpu = cpu_identity_field(fields.next().unwrap_or_default(), "cpu")?;
	let features = cpu_identity_field(fields.next().unwrap_or_default(), "features")?;
	require(fields.next().is_none(), "CPU native target identity has extra fields")?;
	require(target.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte) || byte == b'.'), "CPU target is empty or malformed")?;
	require(compiler.bytes().all(|byte| !byte.is_ascii_control() && byte != b'|' && byte != b';'), "CPU compiler identity is malformed")?;
	require(cpu.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(&byte) || byte == b'.'), "CPU model identity is malformed")?;
	let mut previous = None;
	for feature in features.split(',') {
		let bytes = feature.as_bytes();
		require(
			bytes.len() > 1 && matches!(bytes[0], b'+' | b'-') && bytes[1..].iter().all(|byte| byte.is_ascii_alphanumeric() || b"_-".contains(byte) || *byte == b'.'),
			"CPU feature identity is malformed",
		)?;
		require(previous.is_none_or(|prior: &str| prior < feature), "CPU feature identity is not canonical")?;
		previous = Some(feature);
	}
	Ok((target, compiler, cpu, features))
}

fn native_cpu_target() -> Result<BackendTarget> {
	let compiler = native_cpu_compiler()?;
	let target = option_env!("RECIPE_CPU_TARGET").ok_or_else(|| RecipeError::new("CPU native target is unavailable"))?;
	let output = Command::new(compiler)
		.args(["-target", target, "-march=native", "-###", "-x", "ir", "-c", "/dev/null", "-o", "/dev/null"])
		.output()
		.map_err(|error| RecipeError::new(format!("cannot query CPU native target: {error}")))?;
	require(output.status.success(), format!("CPU native target query failed: {}", String::from_utf8_lossy(&output.stderr).lines().next().unwrap_or("no compiler diagnostic")))?;
	let mut text = String::from_utf8_lossy(&output.stderr).into_owned();
	text.push_str(&String::from_utf8_lossy(&output.stdout));
	let tokens = text.split_whitespace().map(|token| token.trim_matches('"')).collect::<Vec<_>>();
	let cpu = tokens.windows(2).find_map(|pair| (pair[0] == "-target-cpu").then_some(pair[1])).ok_or_else(|| RecipeError::new("CPU native target query omitted target CPU"))?;
	let mut features = tokens.windows(2).filter_map(|pair| (pair[0] == "-target-feature").then_some(pair[1])).collect::<Vec<_>>();
	features.sort_unstable();
	features.dedup();
	require(!features.is_empty(), "CPU native target query omitted target features")?;
	let version = text.lines().find(|line| line.contains("clang version")).map(str::trim).ok_or_else(|| RecipeError::new("CPU native target query omitted compiler identity"))?;
	let identity = format!("target={target};compiler={compiler}@{version};cpu={cpu};features={}", features.join(","));
	let target = BackendTarget::Cpu { target: identity };
	target.validate()?;
	Ok(target)
}

pub(crate) struct NativeArtifact {
	pub(crate) backend: BackendTarget,
	pub(crate) layout: NativeLayout,
	pub(crate) precision: NativePrecision,
	pub(crate) artifact: Vec<u8>,
	pub(crate) path: PathBuf,
	pub(crate) storage: Vec<u8>,
	pub(crate) training: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct NativePrecision {
	model: Compute,
	state: Compute,
	source: &'static str,
	model_type: &'static str,
	state_type: &'static str,
	epoch_layout: &'static [u8],
}

const NATIVE_FORWARD_SYMBOL: &str = "recipe_model_forward";
const NATIVE_EPOCH_SYMBOL: &str = "recipe_model_epoch";
const NATIVE_MODEL_LOAD_SYMBOL: &str = "recipe_model_load";
const NATIVE_CPU_THREAD_SYMBOL: &str = "recipe_model_thread";
const NATIVE_FORWARD_LAYOUT: &[u8] = b"88884444";
const NATIVE_EPOCH_LAYOUT_FP64: &[u8] = b"88888888888844888888844";
const NATIVE_EPOCH_LAYOUT_FP32: &[u8] = b"88888888888844444444444";
const NATIVE_MODEL_LOAD_LAYOUT: &[u8] = b"884";
macro_rules! native_precisions {
	($($pattern:pat $(if $guard:expr)? => ($source:literal, $model_type:literal, $state:expr, $state_type:literal, $layout:expr)),+ $(,)?) => {
		impl NativePrecision {
			fn new(model: Compute) -> Result<Self> {
				match model {
					$($pattern $(if $guard)? => Ok(Self { model, state: $state, source: $source, model_type: $model_type, state_type: $state_type, epoch_layout: $layout }),)+
					_ => Err(RecipeError::new(format!("{} has no native precision composition", model.label()))),
				}
			}
		}
	};
}
native_precisions! {
	Compute::F(_) => ("-f", "double", Compute::FP64, "double", NATIVE_EPOCH_LAYOUT_FP64),
	Compute::Fp(format) if format == FloatFormat::FP64 => ("default", "double", Compute::FP64, "double", NATIVE_EPOCH_LAYOUT_FP64),
	Compute::Fp(format) if format == FloatFormat::FP32 => ("-f32", "float", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
	Compute::Fp(format) if format == FloatFormat::FP16 => ("-f16", "half", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
	Compute::Fp(format) if format == FloatFormat::FP8 => ("-f8", "i8", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
	Compute::Bf(format) if format == FloatFormat::BF16 => ("-bf16", "i16", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
	Compute::Tf(format) if format == FloatFormat::TF32 => ("-tf32", "float", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
	Compute::Int(format) if format == IntFormat::INT8 => ("-int8", "i8", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
	Compute::Int(format) if format == IntFormat::INT4 => ("-int4", "i8", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
	Compute::Int(format) if format == IntFormat::INT1 => ("-int1", "i8", Compute::FP32, "float", NATIVE_EPOCH_LAYOUT_FP32),
}
fn align(value: usize, boundary: usize) -> Result<usize> {
	let boundary = boundary.max(1);
	let remainder = value % boundary;
	if remainder == 0 { Ok(value) } else { checked_add(value, boundary - remainder, "native arena alignment") }
}

fn encode_floats(values: &[f64], precision: Compute) -> Vec<u8> {
	let bytes = precision.bytes();
	values.iter().flat_map(|value| precision.pack(*value).to_le_bytes().into_iter().take(bytes)).collect()
}

/// The stored representation a packed node keeps, or `None` when the node decodes at
/// load. A gather's table is its context rather than a weight span, so it never packs.
fn packed_weight(graph: &Graph, index: usize, inference: bool) -> Option<&StoredWeight> {
	if !inference || !graph.nodes[index].packed {
		return None;
	}
	arena_weight(&graph.nodes[index], graph.stored.get(index)?)
}

/// Byte offset of every node's weights, and the arena size. A packed node keeps its stored bytes.
fn native_weight_arena(graph: &Graph, precision: Compute, inference: bool) -> Result<(Vec<usize>, usize)> {
	let mut offsets = Vec::with_capacity(graph.nodes.len());
	let mut bytes = 0;
	for index in 0..graph.nodes.len() {
		let offset = align(bytes, precision.bytes())?;
		let span = match packed_weight(graph, index, inference) {
			Some(weight) => weight.bytes.len(),
			None => checked_mul(graph.nodes[index].parameters, precision.bytes(), "native weight arena")?,
		};
		offsets.push(offset);
		bytes = checked_add(offset, span, "native weight arena")?;
	}
	Ok((offsets, bytes))
}

fn native_weight_bytes(graph: &Graph, precision: Compute, inference: bool) -> Result<Vec<u8>> {
	let (offsets, bytes) = native_weight_arena(graph, precision, inference)?;
	let mut arena = vec![0_u8; bytes.max(1)];
	for (index, node) in graph.nodes.iter().enumerate() {
		// A packed weight's runs land in the arena straight from where they are
		// mapped or held, so the copy into the arena is the only one made.
		match packed_weight(graph, index, inference) {
			Some(weight) => weight.bytes.copy_into(&mut arena[offsets[index]..offsets[index] + weight.bytes.len()]),
			None => {
				let encoded = encode_floats(&graph.parameters[node.offset..node.offset + node.parameters], precision);
				arena[offsets[index]..offsets[index] + encoded.len()].copy_from_slice(&encoded);
			}
		}
	}
	Ok(arena)
}

impl NativeLayout {
	pub(crate) fn for_graph(graph: &Graph, rows: usize, precision: Compute) -> Result<Self> {
		let element = precision.bytes();
		let mut values = Vec::with_capacity(graph.nodes.len());
		let mut contexts = Vec::with_capacity(graph.nodes.len());
		let mut adjoints = Vec::with_capacity(graph.nodes.len());
		let (mut value_offset, mut context_offset, mut adjoint_offset) = (0, 0, 0);
		for node in &graph.nodes {
			value_offset = align(value_offset, element.max(8))?;
			context_offset = align(context_offset, element.max(8))?;
			adjoint_offset = align(adjoint_offset, element.max(8))?;
			values.push(value_offset);
			contexts.push(context_offset);
			adjoints.push(adjoint_offset);
			value_offset = checked_add(value_offset, graph_rows_buffer(node.output, rows, element)?, "model value arena")?;
			context_offset = checked_add(context_offset, node_context(node, rows, element)?, "model context arena")?;
			adjoint_offset = checked_add(adjoint_offset, graph_rows_buffer(node.output, rows, element)?, "model adjoint arena")?;
		}
		Ok(Self { values, contexts, adjoints, values_bytes: value_offset.max(element), contexts_bytes: context_offset.max(element), adjoints_bytes: adjoint_offset.max(element) })
	}
}

struct NodePlan {
	node: Node,
	value: usize,
	context: usize,
	adjoint: usize,
	stored: Option<StoredWeight>,
	storage_offset: usize,
	weight_offset: usize,
	packed: bool,
}

impl NodePlan {
	/// The decoder selector a consuming kernel passes back, or zero for a dense node.
	fn decode(&self, index: usize) -> usize {
		if self.packed { index + 1 } else { 0 }
	}
}

#[derive(Clone, Copy)]
enum NativeMatrix {
	Gfx11,
	Gfx12,
}

impl NativeMatrix {
	fn key(self) -> &'static str {
		match self {
			Self::Gfx11 => "gfx11",
			Self::Gfx12 => "gfx12",
		}
	}
}

pub(crate) struct NativeModelIr {
	graph: Graph,
	layout: NativeLayout,
	precision: NativePrecision,
	rows: usize,
	schedule: NativeSchedule,
	plans: Vec<NodePlan>,
	storage_bytes: usize,
}

impl NativeModelIr {
	pub(crate) fn from_graph(graph: &Graph, rows: usize, precision: Compute, schedule: NativeSchedule, inference: bool) -> Result<Self> {
		require(rows != 0, "native model rows must be positive")?;
		let layout = NativeLayout::for_graph(graph, rows, precision)?;
		let (weight_offsets, _) = native_weight_arena(graph, precision, inference)?;
		let precision = NativePrecision::new(precision)?;
		let mut plans = Vec::with_capacity(graph.nodes.len());
		let mut storage_bytes = 0usize;
		for (index, node) in graph.nodes.iter().cloned().enumerate() {
			let id = || node.identity(index);
			// A lookup reads its ids on the host, so it names no device source.
			require((node.source >= -1 || (node.source == -2 && node.op == Primitive::Lookup)) && node.source < index as i32, format!("{} has invalid source node {}", id(), node.source))?;
			require(node.second >= -2 && node.second < index as i32, format!("{} has invalid second source node {}", id(), node.second))?;
			// A packed node reads its stored bytes, and one bound to a mapped
			// weight owns no arithmetic span at all, so only a dense node's
			// parameter range is checked against the graph's parameters.
			let packed = packed_weight(graph, index, inference).is_some();
			require(
				packed || node.offset.checked_add(node.parameters).is_some_and(|end| end <= graph.parameters.len()),
				format!("{} parameter range exceeds {} values", id(), graph.parameters.len()),
			)?;
			let width = if node.op == Primitive::Predictor { 2 } else { 3 };
			let program_width = node.program_count.checked_mul(width).ok_or_else(|| RecipeError::new(format!("{} program length overflows", id())))?;
			require(node.program_offset.checked_add(program_width).is_some_and(|end| end <= graph.programs.len()), format!("{} program range exceeds {} values", id(), graph.programs.len()))?;
			let stored = graph.stored.get(index).cloned().unwrap_or(None);
			if let Some(weight) = &stored {
				require(weight.count == node.weights(), format!("{} stored weight count {} does not match tensor count {}", id(), weight.count, node.weights()))?;
			}
			let storage_offset = align(storage_bytes, alignment("float"))?;
			if let Some(weight) = arena_weight(&node, &stored).filter(|_| !packed) {
				storage_bytes = checked_add(storage_offset, weight.bytes.len(), "native storage arena")?;
			}
			plans.push(NodePlan {
				node,
				value: layout.values[index],
				context: layout.contexts[index],
				adjoint: layout.adjoints[index],
				stored,
				storage_offset,
				weight_offset: weight_offsets[index],
				packed,
			});
		}
		Ok(Self { graph: graph.clone(), layout, precision, rows, schedule, plans, storage_bytes })
	}
	fn storage(&self) -> Vec<u8> {
		let mut storage = Vec::with_capacity(self.storage_bytes);
		for plan in &self.plans {
			if let Some(weight) = arena_weight(&plan.node, &plan.stored).filter(|_| !plan.packed) {
				storage.resize(plan.storage_offset, 0);
				weight.bytes.extend_into(&mut storage);
			}
		}
		storage
	}
}

fn template_path(mapping: &str, suffix: &str) -> Result<PathBuf> {
	let key = if suffix.is_empty() { "default" } else { suffix };
	let path = mapping
		.split(';')
		.find_map(|entry| entry.split_once('=').filter(|(name, _)| *name == key).map(|(_, path)| PathBuf::from(path)))
		.ok_or_else(|| RecipeError::new(format!("native LLVM template {key:?} is absent")))?;
	Ok(path)
}

fn backend_template(backend: Backend, precision: NativePrecision, matrix: Option<NativeMatrix>) -> Result<String> {
	let suffix = precision.source;
	let mapping = match backend {
		Backend::Cpu => option_env!("RECIPE_CPU_IR").ok_or_else(|| RecipeError::new("CPU native LLVM templates are unavailable"))?,
		Backend::Amd => option_env!("RECIPE_AMD_IR").ok_or_else(|| RecipeError::new("AMD native LLVM templates are unavailable"))?,
		Backend::Nvidia => option_env!("RECIPE_NV_IR").ok_or_else(|| RecipeError::new("NVIDIA native LLVM templates are unavailable"))?,
	};
	let key = matrix.map_or_else(|| suffix.to_owned(), |method| format!("{}{suffix}", method.key()));
	let mut ir = fs::read_to_string(template_path(mapping, &key)?).map_err(|error| RecipeError::new(format!("cannot read native LLVM template: {error}")))?;
	if let Compute::F(format) = precision.model {
		for address_space in [" addrspace(3)", ""] {
			ir = ir.replace(&format!("load atomic i32, ptr{address_space} @recipe_f_exp monotonic, align 4"), &format!("add i32 0, {}", format.arithmetic.exp));
			ir = ir.replace(&format!("load atomic i32, ptr{address_space} @recipe_f_man monotonic, align 4"), &format!("add i32 0, {}", format.arithmetic.man));
		}
		let narrow = if format.arithmetic == FloatFormat::FP16.arithmetic {
			Some("half")
		} else if format.arithmetic == FloatFormat::FP32.arithmetic {
			Some("float")
		} else {
			None
		};
		if let Some(narrow) = narrow {
			ir = strip_definition(ir, "recipe.round");
			ir.push_str(&format!(
				"define internal double @recipe.round(double %value) #1 {{ entry: %narrow = fptrunc double %value to {narrow} %result = fpext {narrow} %narrow to double ret double %result }}\n"
			));
		}
	}
	Ok(ir)
}

fn pointer_type(backend: Backend) -> &'static str {
	if backend == Backend::Cpu { "ptr" } else { "ptr addrspace(1)" }
}

fn definition_span(ir: &str, name: &str) -> Option<(usize, usize)> {
	let signature = format!("@{name}(");
	let (start, open) = ir.match_indices("define ").find_map(|(start, _)| {
		let open = start + ir[start..].find('{')?;
		ir[start..open].contains(&signature).then_some((start, open))
	})?;
	let mut depth = 0usize;
	for (index, byte) in ir[open..].bytes().enumerate() {
		match byte {
			b'{' => depth += 1,
			b'}' => {
				depth = depth.saturating_sub(1);
				if depth == 0 {
					return Some((start, open + index + 1));
				}
			}
			_ => {}
		}
	}
	None
}

fn strip_definition(mut ir: String, name: &str) -> String {
	if let Some((start, end)) = definition_span(&ir, name) {
		ir.replace_range(start..end, "")
	}
	ir
}

fn prune_internal_definitions(mut ir: String) -> String {
	loop {
		let names = ir
			.match_indices("define internal ")
			.filter_map(|(start, _)| {
				let signature = &ir[start..ir[start..].find('{').map(|offset| start + offset)?];
				Some(signature.rsplit_once('@')?.1.split_once('(')?.0.to_owned())
			})
			.collect::<Vec<_>>();
		// A reference reads "@name(", so one pass over the module's '@' positions counts every name at once instead of searching the whole module again once per name, and no name outruns the window so a later '(' names nothing. The definitions arrive in ascending order, so removing their spans in reverse leaves the earlier spans valid.
		let (bytes, window, mut counts) = (ir.as_bytes(), names.iter().map(|name| name.len() + 1).max().unwrap_or(0), HashMap::new());
		ir.match_indices('@')
			.filter_map(|(at, _)| bytes[at + 1..(at + 1 + window).min(bytes.len())].iter().position(|&byte| byte == b'(').map(|stop| &bytes[at + 1..at + 1 + stop]))
			.for_each(|name| *counts.entry(name).or_insert(0usize) += 1);
		let spans: Vec<_> = names.iter().filter(|name| counts[name.as_bytes()] == 1).filter_map(|name| definition_span(&ir, name)).collect();
		if spans.is_empty() {
			return ir;
		}
		for (start, end) in spans.into_iter().rev() {
			ir.replace_range(start..end, "")
		}
	}
}

fn barrier(backend: Backend) -> &'static str {
	match backend {
		Backend::Cpu => "call void @recipe.cpu.barrier()",
		Backend::Amd | Backend::Nvidia => "call void @grid_barrier(i32 %threads)",
	}
}

fn ptr_gep(backend: Backend, base: &str, offset: usize, name: &str) -> String {
	let pointer = pointer_type(backend);
	format!("%{name} = getelementptr i8, {pointer} %{base}, i64 {offset}\n")
}

mod quantized {
	use super::{Backend, NativePrecision, half, native_literal, pointer_type, unfp16};

	#[derive(Clone, Copy)]
	pub(super) enum QuantIntOp {
		Add,
		Subtract,
		Multiply,
		Divide,
		Remainder,
		ShiftLeft,
		ShiftRight,
		And,
		Or,
		Xor,
	}

	#[derive(Clone, Copy)]
	pub(super) enum QuantValueOp {
		Add,
		Subtract,
		Multiply,
	}

	pub(super) trait QuantOps {
		type Int: Clone;
		type Value: Clone;
		fn index(&self) -> Self::Int;
		fn integer(&self, value: u64) -> Self::Int;
		fn int(&mut self, operation: QuantIntOp, left: Self::Int, right: Self::Int) -> Self::Int;
		fn equal(&mut self, left: Self::Int, right: Self::Int) -> Self::Int;
		fn less(&mut self, left: Self::Int, right: Self::Int) -> Self::Int;
		fn select_int(&mut self, condition: Self::Int, yes: Self::Int, no: Self::Int) -> Self::Int;
		fn sign_extend(&mut self, value: Self::Int, bits: u8) -> Self::Int;
		fn load(&mut self, bits: u8, offset: Self::Int) -> Self::Int;
		fn half(&mut self, offset: Self::Int) -> Self::Value;
		fn float(&mut self, offset: Self::Int) -> Self::Value;
		fn half_bits(&mut self, bits: Self::Int) -> Self::Value;
		fn table(&mut self, name: &'static str, values: &'static [u16], index: Self::Int) -> Self::Int;
		fn signed_table(&mut self, name: &'static str, values: &'static [i8], index: Self::Int) -> Self::Int;
		fn value_table(&mut self, name: &str, values: &[f64], index: Self::Int) -> Self::Value;
		fn number(&mut self, value: Self::Int, signed: bool) -> Self::Value;
		fn literal(&self, value: f64) -> Self::Value;
		fn value(&mut self, operation: QuantValueOp, left: Self::Value, right: Self::Value) -> Self::Value;
		fn select_value(&mut self, condition: Self::Int, yes: Self::Value, no: Self::Value) -> Self::Value;
		fn signed(&mut self, magnitude: Self::Value, sign: Self::Int) -> Self::Value;
	}

	fn quant_int<Q: QuantOps>(quant: &mut Q, operation: QuantIntOp, left: Q::Int, right: u64) -> Q::Int {
		quant.int(operation, left, quant.integer(right))
	}
	fn quant_bits<Q: QuantOps>(quant: &mut Q, value: Q::Int, shift: Q::Int, width: u8) -> Q::Int {
		let shifted = quant.int(QuantIntOp::ShiftRight, value, shift);
		quant_int(quant, QuantIntOp::And, shifted, (1_u64 << width) - 1)
	}
	fn quant_parity_sign<Q: QuantOps>(quant: &mut Q, signs: Q::Int, lane: Q::Int) -> Q::Int {
		let shifted = quant.int(QuantIntOp::ShiftRight, signs.clone(), lane.clone());
		let direct = quant_int(quant, QuantIntOp::And, shifted, 1);
		let high = quant_int(quant, QuantIntOp::ShiftRight, signs.clone(), 4);
		let parity4 = quant.int(QuantIntOp::Xor, signs, high);
		let high = quant_int(quant, QuantIntOp::ShiftRight, parity4.clone(), 2);
		let parity2 = quant.int(QuantIntOp::Xor, parity4, high);
		let high = quant_int(quant, QuantIntOp::ShiftRight, parity2.clone(), 1);
		let parity1 = quant.int(QuantIntOp::Xor, parity2, high);
		let parity = quant_int(quant, QuantIntOp::And, parity1, 1);
		let last = quant.equal(lane, quant.integer(7));
		quant.select_int(last, parity, direct)
	}

	#[derive(Clone, Copy)]
	pub(super) enum IqPacking {
		S,
		Xs,
		Xxs,
	}

	#[derive(Clone, Copy)]
	pub(super) struct IqLayout {
		pub(super) man: u8,
		pub(super) exp: u8,
		pub(super) sign: u8,
		pub(super) packing: IqPacking,
		pub(super) table_name: &'static str,
		pub(super) table: &'static [u16],
	}

	#[derive(Clone, Copy)]
	pub(super) struct Iq1Layout {
		pub(super) man: u8,
		pub(super) exp: u8,
		pub(super) sign: u8,
		pub(super) medium: bool,
		pub(super) table_name: &'static str,
		pub(super) table: &'static [u16],
	}

	#[derive(Clone, Copy)]
	pub(super) struct ScalarLayout {
		pub(super) sign: u8,
		pub(super) exp: u8,
		pub(super) man: u8,
		pub(super) variant: u8,
	}

	#[derive(Clone, Copy)]
	pub(super) struct Iq4Layout {
		pub(super) sign: u8,
		pub(super) exp: u8,
		pub(super) man: u8,
		pub(super) xs: bool,
		pub(super) table_name: &'static str,
		pub(super) table: &'static [i8],
	}

	pub(super) fn dequant_iq<Q: QuantOps>(quant: &mut Q, layout: IqLayout) -> Q::Value {
		let local = quant.index();
		let lane = quant_int(quant, QuantIntOp::Remainder, local.clone(), 8);
		let scale = quant.half(quant.integer(0));
		let (grid, factor_code, sign, table_lane, multiplier, odd_factor) = match (layout.man, layout.packing) {
			(2, IqPacking::S) => {
				let value_block = quant_int(quant, QuantIntOp::Divide, local.clone(), 16);
				let slot = quant_int(quant, QuantIntOp::Divide, local.clone(), 8);
				let low_offset = quant_int(quant, QuantIntOp::Add, slot.clone(), 2);
				let low = quant.load(8, low_offset);
				let high_slot = quant_int(quant, QuantIntOp::Divide, slot.clone(), 4);
				let high_offset = quant_int(quant, QuantIntOp::Add, high_slot, 66);
				let high = quant.load(8, high_offset);
				let high_lane = quant_int(quant, QuantIntOp::Remainder, slot.clone(), 4);
				let high_shift = quant_int(quant, QuantIntOp::Multiply, high_lane, 2);
				let high_bits = quant_bits(quant, high, high_shift, 2);
				let high_bits = quant_int(quant, QuantIntOp::ShiftLeft, high_bits, 8);
				let grid = quant.int(QuantIntOp::Or, low, high_bits);
				let sign_offset = quant_int(quant, QuantIntOp::Add, slot, 34);
				let signs = quant.load(8, sign_offset);
				let sign = quant_bits(quant, signs, lane.clone(), layout.sign);
				let factor_block = quant_int(quant, QuantIntOp::Divide, value_block.clone(), 2);
				let factor_offset = quant_int(quant, QuantIntOp::Add, factor_block, 74);
				let factor = quant.load(8, factor_offset);
				let factor_lane = quant_int(quant, QuantIntOp::Remainder, value_block, 2);
				let factor_shift = quant_int(quant, QuantIntOp::Multiply, factor_lane, layout.exp as u64);
				let factor = quant_bits(quant, factor, factor_shift, layout.exp);
				(grid, factor, sign, lane, 0.25, false)
			}
			(2, IqPacking::Xs) => {
				let value_block = quant_int(quant, QuantIntOp::Divide, local.clone(), 16);
				let slot = quant_int(quant, QuantIntOp::Divide, local.clone(), 8);
				let word_offset = quant_int(quant, QuantIntOp::Multiply, slot, 2);
				let word_offset = quant_int(quant, QuantIntOp::Add, word_offset, 2);
				let word = quant.load(16, word_offset);
				let grid = quant_int(quant, QuantIntOp::And, word.clone(), 511);
				let signs = quant_int(quant, QuantIntOp::ShiftRight, word, 9);
				let sign = quant_parity_sign(quant, signs, lane.clone());
				let factor_block = quant_int(quant, QuantIntOp::Divide, value_block.clone(), 2);
				let factor_offset = quant_int(quant, QuantIntOp::Add, factor_block, 66);
				let factor = quant.load(8, factor_offset);
				let factor_lane = quant_int(quant, QuantIntOp::Remainder, value_block, 2);
				let factor_shift = quant_int(quant, QuantIntOp::Multiply, factor_lane, layout.exp as u64);
				let factor = quant_bits(quant, factor, factor_shift, layout.exp);
				(grid, factor, sign, lane, 0.25, false)
			}
			(2, IqPacking::Xxs) | (3, IqPacking::Xxs) => {
				let value_block = quant_int(quant, QuantIntOp::Divide, local.clone(), 32);
				let group_local = quant_int(quant, QuantIntOp::Remainder, local.clone(), 32);
				let group = quant_int(quant, QuantIntOp::Divide, group_local, 8);
				let grids_per_block = 8;
				let grid_block = quant_int(quant, QuantIntOp::Multiply, value_block.clone(), grids_per_block);
				let grid_group = if layout.man == 2 {
					group.clone()
				} else {
					let group = quant_int(quant, QuantIntOp::Multiply, group.clone(), 2);
					let half = quant_int(quant, QuantIntOp::Divide, lane.clone(), 4);
					quant.int(QuantIntOp::Add, group, half)
				};
				let grid_offset = quant.int(QuantIntOp::Add, grid_block, grid_group);
				let grid_offset = quant_int(quant, QuantIntOp::Add, grid_offset, 2);
				let grid = quant.load(8, grid_offset);
				let word_stride = if layout.man == 2 { 8 } else { 4 };
				let word_base = if layout.man == 2 { 6 } else { 66 };
				let word_offset = quant_int(quant, QuantIntOp::Multiply, value_block, word_stride);
				let word_offset = quant_int(quant, QuantIntOp::Add, word_offset, word_base);
				let word = quant.load(32, word_offset);
				let sign_shift = quant_int(quant, QuantIntOp::Multiply, group, 7);
				let signs = quant_bits(quant, word.clone(), sign_shift, 7);
				let sign = quant_parity_sign(quant, signs, lane.clone());
				let factor = quant_bits(quant, word, quant.integer(28), layout.exp);
				let table_lane = if layout.man == 2 { lane } else { quant_int(quant, QuantIntOp::Remainder, lane, 4) };
				(grid, factor, sign, table_lane, if layout.man == 2 { 0.25 } else { 0.5 }, false)
			}
			(3, IqPacking::S) => {
				let value_block = quant_int(quant, QuantIntOp::Divide, local.clone(), 32);
				let group = quant_int(quant, QuantIntOp::Divide, local.clone(), 4);
				let low_offset = quant_int(quant, QuantIntOp::Add, group.clone(), 2);
				let low = quant.load(8, low_offset);
				let high_group = quant_int(quant, QuantIntOp::Divide, group.clone(), 8);
				let high_offset = quant_int(quant, QuantIntOp::Add, high_group, 66);
				let high = quant.load(8, high_offset);
				let high_lane = quant_int(quant, QuantIntOp::Remainder, group, 8);
				let high = quant_bits(quant, high, high_lane, 1);
				let high = quant_int(quant, QuantIntOp::ShiftLeft, high, 8);
				let grid = quant.int(QuantIntOp::Or, low, high);
				let sign_group = quant_int(quant, QuantIntOp::Divide, local.clone(), 8);
				let sign_offset = quant_int(quant, QuantIntOp::Add, sign_group, 74);
				let signs = quant.load(8, sign_offset);
				let sign = quant_bits(quant, signs, lane.clone(), layout.sign);
				let factor_block = quant_int(quant, QuantIntOp::Divide, value_block.clone(), 2);
				let factor_offset = quant_int(quant, QuantIntOp::Add, factor_block, 106);
				let factor = quant.load(8, factor_offset);
				let factor_lane = quant_int(quant, QuantIntOp::Remainder, value_block, 2);
				let factor_shift = quant_int(quant, QuantIntOp::Multiply, factor_lane, layout.exp as u64);
				let factor = quant_bits(quant, factor, factor_shift, layout.exp);
				(grid, factor, sign, quant_int(quant, QuantIntOp::Remainder, lane, 4), 1.0, true)
			}
			_ => unreachable!(),
		};
		let table_word = quant.table(layout.table_name, layout.table, grid);
		let man_shift = quant_int(quant, QuantIntOp::Multiply, table_lane, layout.man as u64);
		let man_code = quant_bits(quant, table_word, man_shift, layout.man);
		let man_code = quant_int(quant, QuantIntOp::Multiply, man_code, 2);
		let man_code = quant_int(quant, QuantIntOp::Add, man_code, 1);
		let mantissa = quant.number(man_code, false);
		let exponent = if odd_factor {
			let factor_code = quant_int(quant, QuantIntOp::Multiply, factor_code, 2);
			let factor_code = quant_int(quant, QuantIntOp::Add, factor_code, 1);
			quant.number(factor_code, false)
		} else {
			let factor = quant.number(factor_code, false);
			quant.value(QuantValueOp::Add, factor, quant.literal(0.5))
		};
		let scaled = quant.value(QuantValueOp::Multiply, scale, exponent);
		let scaled = quant.value(QuantValueOp::Multiply, scaled, quant.literal(multiplier));
		let magnitude = quant.value(QuantValueOp::Multiply, scaled, mantissa);
		quant.signed(magnitude, sign)
	}

	pub(super) fn dequant_iq1<Q: QuantOps>(quant: &mut Q, layout: Iq1Layout) -> Q::Value {
		let local = quant.index();
		let lane = quant_int(quant, QuantIntOp::Remainder, local.clone(), 8);
		let (grid, scale, factor_code, delta_bit) = if layout.medium {
			let value_block = quant_int(quant, QuantIntOp::Divide, local.clone(), 16);
			let group_local = quant_int(quant, QuantIntOp::Remainder, local.clone(), 16);
			let group = quant_int(quant, QuantIntOp::Divide, group_local, 8);
			let high_offset = quant_int(quant, QuantIntOp::Add, value_block.clone(), 32);
			let high = quant.load(8, high_offset);
			let grid_block = quant_int(quant, QuantIntOp::Multiply, value_block.clone(), 2);
			let grid_offset = quant.int(QuantIntOp::Add, grid_block, group.clone());
			let grid_low = quant.load(8, grid_offset);
			let group_shift = quant_int(quant, QuantIntOp::Multiply, group.clone(), 4);
			let grid_high = quant_bits(quant, high.clone(), group_shift.clone(), 3);
			let grid_high = quant_int(quant, QuantIntOp::ShiftLeft, grid_high, 8);
			let grid = quant.int(QuantIntOp::Or, grid_low, grid_high);
			let delta_shift = quant_int(quant, QuantIntOp::Add, group_shift, 3);
			let delta = quant_bits(quant, high, delta_shift, layout.sign);
			let packed = quant.load(64, quant.integer(48));
			let s0 = quant_bits(quant, packed.clone(), quant.integer(12), 4);
			let s1 = quant_bits(quant, packed.clone(), quant.integer(24), 8);
			let s1 = quant_int(quant, QuantIntOp::And, s1, 240);
			let scale = quant.int(QuantIntOp::Or, s0, s1);
			let s2 = quant_bits(quant, packed.clone(), quant.integer(36), 12);
			let s2 = quant_int(quant, QuantIntOp::And, s2, 3840);
			let scale = quant.int(QuantIntOp::Or, scale, s2);
			let s3 = quant_bits(quant, packed.clone(), quant.integer(48), 16);
			let s3 = quant_int(quant, QuantIntOp::And, s3, 61440);
			let scale = quant.int(QuantIntOp::Or, scale, s3);
			let scale = quant.half_bits(scale);
			let scale_word = quant_int(quant, QuantIntOp::Divide, value_block.clone(), 4);
			let scale_word = quant_int(quant, QuantIntOp::Multiply, scale_word, 16);
			let scale_local = quant_int(quant, QuantIntOp::Remainder, value_block, 4);
			let scale_local = quant_int(quant, QuantIntOp::Multiply, scale_local, layout.exp as u64);
			let scale_shift = quant.int(QuantIntOp::Add, scale_word, scale_local);
			let factor = quant_bits(quant, packed, scale_shift, layout.exp);
			(grid, scale, factor, delta)
		} else {
			let value_block = quant_int(quant, QuantIntOp::Divide, local.clone(), 32);
			let group_local = quant_int(quant, QuantIntOp::Remainder, local.clone(), 32);
			let group = quant_int(quant, QuantIntOp::Divide, group_local, 8);
			let high_block = quant_int(quant, QuantIntOp::Multiply, value_block.clone(), 2);
			let high_offset = quant_int(quant, QuantIntOp::Add, high_block, 34);
			let high = quant.load(16, high_offset);
			let grid_block = quant_int(quant, QuantIntOp::Multiply, value_block, 4);
			let grid_base = quant_int(quant, QuantIntOp::Add, grid_block, 2);
			let grid_offset = quant.int(QuantIntOp::Add, grid_base, group.clone());
			let grid_low = quant.load(8, grid_offset);
			let group_shift = quant_int(quant, QuantIntOp::Multiply, group, layout.exp as u64);
			let grid_high = quant_bits(quant, high.clone(), group_shift, 3);
			let grid_high = quant_int(quant, QuantIntOp::ShiftLeft, grid_high, 8);
			let grid = quant.int(QuantIntOp::Or, grid_low, grid_high);
			let delta = quant_bits(quant, high.clone(), quant.integer(15), layout.sign);
			let factor = quant_bits(quant, high, quant.integer(12), layout.exp);
			(grid, quant.half(quant.integer(0)), factor, delta)
		};
		let table_word = quant.table(layout.table_name, layout.table, grid);
		let man_shift = quant_int(quant, QuantIntOp::Multiply, lane, layout.man as u64);
		let man_code = quant_bits(quant, table_word, man_shift, layout.man);
		let man_code = quant_int(quant, QuantIntOp::Subtract, man_code, 1);
		let mantissa = quant.number(man_code, true);
		let delta = quant.select_value(delta_bit, quant.literal(-0.125), quant.literal(0.125));
		let mantissa = quant.value(QuantValueOp::Add, mantissa, delta);
		let factor_code = quant_int(quant, QuantIntOp::Multiply, factor_code, 2);
		let factor_code = quant_int(quant, QuantIntOp::Add, factor_code, 1);
		let exponent = quant.number(factor_code, false);
		let scaled = quant.value(QuantValueOp::Multiply, scale, exponent);
		quant.value(QuantValueOp::Multiply, scaled, mantissa)
	}

	pub(super) fn dequant_q45k<Q: QuantOps>(quant: &mut Q, man: u8) -> Q::Value {
		let local = quant.index();
		let sub = quant_int(quant, QuantIntOp::Divide, local.clone(), 32);
		let within = quant_int(quant, QuantIntOp::Remainder, local, 32);
		let pair = quant_int(quant, QuantIntOp::Divide, sub.clone(), 2);
		let packed_offset = quant_int(quant, QuantIntOp::Multiply, pair, 32);
		let packed_offset = quant.int(QuantIntOp::Add, packed_offset, within);
		let packed_offset = quant_int(quant, QuantIntOp::Add, packed_offset, if man == 4 { 16 } else { 48 });
		let packed = quant.load(8, packed_offset);
		let half = quant_int(quant, QuantIntOp::Remainder, sub.clone(), 2);
		let shift = quant_int(quant, QuantIntOp::Multiply, half, 4);
		let low_code = quant_bits(quant, packed, shift, 4);
		let code = if man == 4 {
			low_code
		} else {
			let high_offset = quant_int(quant, QuantIntOp::Remainder, quant.index(), 32);
			let high_offset = quant_int(quant, QuantIntOp::Add, high_offset, 16);
			let high = quant.load(8, high_offset);
			let high_shift = quant_int(quant, QuantIntOp::Divide, sub.clone(), 1);
			let high = quant_bits(quant, high, high_shift, 1);
			let high = quant_int(quant, QuantIntOp::ShiftLeft, high, 4);
			quant.int(QuantIntOp::Or, low_code, high)
		};
		let low_scale_offset = quant_int(quant, QuantIntOp::Add, sub.clone(), 4);
		let low_scale = quant.load(8, low_scale_offset);
		let low_scale = quant_int(quant, QuantIntOp::And, low_scale, 63);
		let low_minimum_offset = quant_int(quant, QuantIntOp::Add, sub.clone(), 8);
		let low_minimum = quant.load(8, low_minimum_offset);
		let low_minimum = quant_int(quant, QuantIntOp::And, low_minimum, 63);
		let high_packed_offset = quant_int(quant, QuantIntOp::Add, sub.clone(), 8);
		let high_packed = quant.load(8, high_packed_offset);
		let high_scale_bits = quant.load(8, sub.clone());
		let high_scale_low = quant_int(quant, QuantIntOp::And, high_packed.clone(), 15);
		let high_scale_top = quant_int(quant, QuantIntOp::ShiftRight, high_scale_bits, 6);
		let high_scale_top = quant_int(quant, QuantIntOp::ShiftLeft, high_scale_top, 4);
		let high_scale = quant.int(QuantIntOp::Or, high_scale_low, high_scale_top);
		let high_minimum_offset = quant_int(quant, QuantIntOp::Add, sub.clone(), 4);
		let high_minimum_bits = quant.load(8, high_minimum_offset);
		let high_minimum_low = quant_int(quant, QuantIntOp::ShiftRight, high_packed, 4);
		let high_minimum_top = quant_int(quant, QuantIntOp::ShiftRight, high_minimum_bits, 6);
		let high_minimum_top = quant_int(quant, QuantIntOp::ShiftLeft, high_minimum_top, 4);
		let high_minimum = quant.int(QuantIntOp::Or, high_minimum_low, high_minimum_top);
		let low = quant.less(sub, quant.integer(4));
		let scale_code = quant.select_int(low.clone(), low_scale, high_scale);
		let minimum_code = quant.select_int(low, low_minimum, high_minimum);
		let scale = quant.half(quant.integer(0));
		let minimum = quant.half(quant.integer(2));
		let scale_code = quant.number(scale_code, false);
		let minimum_code = quant.number(minimum_code, false);
		let code = quant.number(code, false);
		let step = quant.value(QuantValueOp::Multiply, scale, scale_code);
		let base = quant.value(QuantValueOp::Multiply, minimum, minimum_code);
		let product = quant.value(QuantValueOp::Multiply, step, code);
		quant.value(QuantValueOp::Subtract, product, base)
	}

	pub(super) fn dequant_q6k<Q: QuantOps>(quant: &mut Q) -> Q::Value {
		let local = quant.index();
		let chunk = quant_int(quant, QuantIntOp::Divide, local.clone(), 128);
		let chunk_local = quant_int(quant, QuantIntOp::Remainder, local, 128);
		let group = quant_int(quant, QuantIntOp::Divide, chunk_local.clone(), 32);
		let within = quant_int(quant, QuantIntOp::Remainder, chunk_local, 32);
		let low_group = quant_int(quant, QuantIntOp::And, group.clone(), 1);
		let low_extra = quant_int(quant, QuantIntOp::Multiply, low_group, 32);
		let low_local = quant.int(QuantIntOp::Add, within.clone(), low_extra);
		let low_chunk = quant_int(quant, QuantIntOp::Multiply, chunk.clone(), 64);
		let low_offset = quant.int(QuantIntOp::Add, low_chunk, low_local);
		let low = quant.load(8, low_offset);
		let high_chunk = quant_int(quant, QuantIntOp::Multiply, chunk.clone(), 32);
		let high_offset = quant.int(QuantIntOp::Add, high_chunk, within.clone());
		let high_offset = quant_int(quant, QuantIntOp::Add, high_offset, 128);
		let high = quant.load(8, high_offset);
		let low_half = quant_int(quant, QuantIntOp::Divide, group.clone(), 2);
		let low_shift = quant_int(quant, QuantIntOp::Multiply, low_half, 4);
		let low_bits = quant_bits(quant, low, low_shift, 4);
		let high_shift = quant_int(quant, QuantIntOp::Multiply, group.clone(), 2);
		let high_bits = quant_bits(quant, high, high_shift, 2);
		let high_bits = quant_int(quant, QuantIntOp::ShiftLeft, high_bits, 4);
		let code = quant.int(QuantIntOp::Or, low_bits, high_bits);
		let code = quant_int(quant, QuantIntOp::Subtract, code, 32);
		let scale_half = quant_int(quant, QuantIntOp::Divide, within, 16);
		let scale_group = quant_int(quant, QuantIntOp::Multiply, group, 2);
		let scale_local = quant.int(QuantIntOp::Add, scale_group, scale_half);
		let scale_chunk = quant_int(quant, QuantIntOp::Multiply, chunk, 8);
		let scale_offset = quant.int(QuantIntOp::Add, scale_chunk, scale_local);
		let scale_offset = quant_int(quant, QuantIntOp::Add, scale_offset, 192);
		let factor = quant.load(8, scale_offset);
		let factor = quant.sign_extend(factor, 8);
		let scale = quant.half(quant.integer(208));
		let factor = quant.number(factor, true);
		let code = quant.number(code, true);
		let scaled = quant.value(QuantValueOp::Multiply, scale, factor);
		quant.value(QuantValueOp::Multiply, scaled, code)
	}

	pub(super) fn dequant_scalar<Q: QuantOps>(quant: &mut Q, layout: ScalarLayout) -> Q::Value {
		let local = quant.index();
		let header = if layout.variant == 1 { 4 } else { 2 };
		let code = match layout.man {
			4 => {
				let offset = quant_int(quant, QuantIntOp::Remainder, local.clone(), 16);
				let offset = quant_int(quant, QuantIntOp::Add, offset, header);
				let byte = quant.load(8, offset);
				let half = quant_int(quant, QuantIntOp::Divide, local, 16);
				let shift = quant_int(quant, QuantIntOp::Multiply, half, 4);
				quant_bits(quant, byte, shift, 4)
			}
			5 => {
				let low_offset = quant_int(quant, QuantIntOp::Remainder, local.clone(), 16);
				let low_offset = quant_int(quant, QuantIntOp::Add, low_offset, header + 4);
				let low = quant.load(8, low_offset);
				let half = quant_int(quant, QuantIntOp::Divide, local.clone(), 16);
				let shift = quant_int(quant, QuantIntOp::Multiply, half, 4);
				let low = quant_bits(quant, low, shift, 4);
				let high_offset = quant_int(quant, QuantIntOp::Divide, local.clone(), 8);
				let high_offset = quant_int(quant, QuantIntOp::Add, high_offset, header);
				let high = quant.load(8, high_offset);
				let lane = quant_int(quant, QuantIntOp::Remainder, local, 8);
				let high = quant_bits(quant, high, lane, 1);
				let high = quant_int(quant, QuantIntOp::ShiftLeft, high, 4);
				quant.int(QuantIntOp::Or, low, high)
			}
			8 => {
				let offset = quant_int(quant, QuantIntOp::Add, local, header);
				quant.load(8, offset)
			}
			_ => unreachable!(),
		};
		let scale = if layout.exp == 5 { quant.half(quant.integer(0)) } else { unreachable!() };
		let code = if layout.man == 8 {
			let code = quant.sign_extend(code, 8);
			quant.number(code, true)
		} else if layout.variant == 0 {
			let offset = quant_int(quant, QuantIntOp::ShiftLeft, quant.integer(1), u64::from(layout.man - layout.sign));
			let code = quant.int(QuantIntOp::Subtract, code, offset);
			quant.number(code, true)
		} else {
			quant.number(code, false)
		};
		let product = quant.value(QuantValueOp::Multiply, code, scale);
		if layout.variant == 1 && layout.man != 8 {
			let minimum = quant.half(quant.integer(2));
			quant.value(QuantValueOp::Add, product, minimum)
		} else {
			product
		}
	}

	pub(super) fn dequant_q2k<Q: QuantOps>(quant: &mut Q) -> Q::Value {
		let local = quant.index();
		let order = quant_int(quant, QuantIntOp::Divide, local.clone(), 16);
		let section = quant_int(quant, QuantIntOp::Divide, order.clone(), 8);
		let shift_group = quant_int(quant, QuantIntOp::Remainder, order.clone(), 8);
		let shift_group = quant_int(quant, QuantIntOp::Divide, shift_group, 2);
		let shift = quant_int(quant, QuantIntOp::Multiply, shift_group, 2);
		let half_index = quant_int(quant, QuantIntOp::Remainder, order, 2);
		let offset = quant_int(quant, QuantIntOp::Remainder, local, 16);
		let metadata_offset = quant_int(quant, QuantIntOp::Multiply, section.clone(), 8);
		let metadata_offset = quant.int(QuantIntOp::Add, metadata_offset, shift.clone());
		let metadata_offset = quant.int(QuantIntOp::Add, metadata_offset, half_index.clone());
		let metadata = quant.load(8, metadata_offset);
		let scale_code = quant_int(quant, QuantIntOp::And, metadata.clone(), 15);
		let minimum_code = quant_int(quant, QuantIntOp::ShiftRight, metadata, 4);
		let code_offset = quant_int(quant, QuantIntOp::Multiply, section, 32);
		let half_offset = quant_int(quant, QuantIntOp::Multiply, half_index, 16);
		let code_offset = quant.int(QuantIntOp::Add, code_offset, half_offset);
		let code_offset = quant.int(QuantIntOp::Add, code_offset, offset);
		let code_offset = quant_int(quant, QuantIntOp::Add, code_offset, 16);
		let code = quant.load(8, code_offset);
		let code = quant_bits(quant, code, shift, 2);
		let scale = quant.half(quant.integer(80));
		let minimum = quant.half(quant.integer(82));
		let scale_code = quant.number(scale_code, false);
		let minimum_code = quant.number(minimum_code, false);
		let code = quant.number(code, false);
		let scaled = quant.value(QuantValueOp::Multiply, scale, scale_code);
		let product = quant.value(QuantValueOp::Multiply, scaled, code);
		let minimum = quant.value(QuantValueOp::Multiply, minimum, minimum_code);
		quant.value(QuantValueOp::Subtract, product, minimum)
	}

	pub(super) fn dequant_q3k<Q: QuantOps>(quant: &mut Q) -> Q::Value {
		let local = quant.index();
		let block = quant_int(quant, QuantIntOp::Divide, local.clone(), 16);
		let low_block = quant_int(quant, QuantIntOp::Subtract, block.clone(), 8);
		let low = quant.less(block.clone(), quant.integer(8));
		let low_block = quant.select_int(low.clone(), block.clone(), low_block);
		let low_offset = quant_int(quant, QuantIntOp::Add, low_block, 96);
		let low_scale = quant.load(8, low_offset);
		let low_shift = quant.select_int(low, quant.integer(0), quant.integer(4));
		let low_scale = quant_bits(quant, low_scale, low_shift, 4);
		let high_block = quant_int(quant, QuantIntOp::Remainder, block.clone(), 4);
		let high_offset = quant_int(quant, QuantIntOp::Add, high_block, 104);
		let high_scale = quant.load(8, high_offset);
		let high_shift = quant_int(quant, QuantIntOp::Divide, block.clone(), 4);
		let high_shift = quant_int(quant, QuantIntOp::Multiply, high_shift, 2);
		let high_scale = quant_bits(quant, high_scale, high_shift, 2);
		let high_scale = quant_int(quant, QuantIntOp::ShiftLeft, high_scale, 4);
		let scale_code = quant.int(QuantIntOp::Or, low_scale, high_scale);
		let scale_code = quant_int(quant, QuantIntOp::Subtract, scale_code, 32);
		let section = quant_int(quant, QuantIntOp::Divide, local.clone(), 128);
		let code_offset = quant_int(quant, QuantIntOp::Multiply, section, 32);
		let within = quant_int(quant, QuantIntOp::Remainder, local.clone(), 32);
		let code_offset = quant.int(QuantIntOp::Add, code_offset, within.clone());
		let code_offset = quant_int(quant, QuantIntOp::Add, code_offset, 32);
		let code = quant.load(8, code_offset);
		let local128 = quant_int(quant, QuantIntOp::Remainder, local.clone(), 128);
		let code_shift = quant_int(quant, QuantIntOp::Divide, local128, 32);
		let code_shift = quant_int(quant, QuantIntOp::Multiply, code_shift, 2);
		let code = quant_bits(quant, code, code_shift, 2);
		let sign_byte = quant.load(8, within);
		let sign_shift = quant_int(quant, QuantIntOp::Divide, local, 32);
		let sign = quant_bits(quant, sign_byte, sign_shift, 1);
		let subtract = quant.select_int(sign, quant.integer(0), quant.integer(4));
		let code = quant.int(QuantIntOp::Subtract, code, subtract);
		let scale = quant.half(quant.integer(108));
		let scale_code = quant.number(scale_code, true);
		let code = quant.number(code, true);
		let scaled = quant.value(QuantValueOp::Multiply, scale, scale_code);
		quant.value(QuantValueOp::Multiply, scaled, code)
	}

	pub(super) fn dequant_q8k<Q: QuantOps>(quant: &mut Q) -> Q::Value {
		let local = quant.index();
		let offset = quant_int(quant, QuantIntOp::Add, local, 4);
		let code = quant.load(8, offset);
		let code = quant.sign_extend(code, 8);
		let code = quant.number(code, true);
		let scale = quant.float(quant.integer(0));
		quant.value(QuantValueOp::Multiply, scale, code)
	}

	pub(super) fn dequant_iq4<Q: QuantOps>(quant: &mut Q, layout: Iq4Layout) -> Q::Value {
		let local = quant.index();
		let scale = quant.half(quant.integer(0));
		let (code, exponent) = if layout.xs {
			let block = quant_int(quant, QuantIntOp::Divide, local.clone(), 32);
			let high = quant.load(16, quant.integer(2));
			let low_offset = quant_int(quant, QuantIntOp::Divide, block.clone(), 2);
			let low_offset = quant_int(quant, QuantIntOp::Add, low_offset, 4);
			let low = quant.load(8, low_offset);
			let low_shift = quant_int(quant, QuantIntOp::Remainder, block.clone(), 2);
			let low_shift = quant_int(quant, QuantIntOp::Multiply, low_shift, 4);
			let low = quant_bits(quant, low, low_shift, layout.man);
			let high_shift = quant_int(quant, QuantIntOp::Multiply, block.clone(), 2);
			let high = quant_bits(quant, high, high_shift, layout.sign + 1);
			let high = quant_int(quant, QuantIntOp::ShiftLeft, high, layout.man as u64);
			let exponent = quant.int(QuantIntOp::Or, low, high);
			let exponent = quant_int(quant, QuantIntOp::Subtract, exponent, 1_u64 << (layout.exp - 1));
			let packed_offset = quant_int(quant, QuantIntOp::Multiply, block, 16);
			let within = quant_int(quant, QuantIntOp::Remainder, local.clone(), 16);
			let packed_offset = quant.int(QuantIntOp::Add, packed_offset, within);
			let packed_offset = quant_int(quant, QuantIntOp::Add, packed_offset, 8);
			let packed = quant.load(8, packed_offset);
			let half = quant_int(quant, QuantIntOp::Remainder, local, 32);
			let high_half = quant.less(half, quant.integer(16));
			let shift = quant.select_int(high_half, quant.integer(0), quant.integer(4));
			(quant_bits(quant, packed, shift, layout.man), exponent)
		} else {
			let offset = quant_int(quant, QuantIntOp::Remainder, local.clone(), 16);
			let offset = quant_int(quant, QuantIntOp::Add, offset, 2);
			let packed = quant.load(8, offset);
			let low = quant.less(local, quant.integer(16));
			let shift = quant.select_int(low, quant.integer(0), quant.integer(4));
			(quant_bits(quant, packed, shift, layout.man), quant.integer(1))
		};
		let level = quant.signed_table(layout.table_name, layout.table, code);
		let level = quant.number(level, true);
		let exponent = quant.number(exponent, true);
		let scaled = quant.value(QuantValueOp::Multiply, scale, exponent);
		quant.value(QuantValueOp::Multiply, scaled, level)
	}

	pub(super) fn dequant_nf4<Q: QuantOps>(quant: &mut Q, block: usize, table_name: &str, table: &[f64], scales_name: &str, scales: &[f64]) -> Q::Value {
		let local = quant.index();
		let byte_offset = quant_int(quant, QuantIntOp::Divide, local.clone(), 2);
		let packed = quant.load(8, byte_offset);
		let half = quant_int(quant, QuantIntOp::Remainder, local.clone(), 2);
		let shift = quant_int(quant, QuantIntOp::Multiply, half, 4);
		let code = quant_bits(quant, packed, shift, 4);
		let level = quant.value_table(table_name, table, code);
		let scale_index = quant_int(quant, QuantIntOp::Divide, local, block as u64);
		let scale = quant.value_table(scales_name, scales, scale_index);
		quant.value(QuantValueOp::Multiply, level, scale)
	}

	pub(super) struct HostQuantOps<'a> {
		pub(super) bytes: &'a [u8],
		pub(super) index: usize,
	}

	impl QuantOps for HostQuantOps<'_> {
		type Int = u64;
		type Value = f64;
		fn index(&self) -> Self::Int {
			self.index as u64
		}
		fn integer(&self, value: u64) -> Self::Int {
			value
		}
		fn int(&mut self, operation: QuantIntOp, left: Self::Int, right: Self::Int) -> Self::Int {
			match operation {
				QuantIntOp::Add => left + right,
				QuantIntOp::Subtract => left.wrapping_sub(right),
				QuantIntOp::Multiply => left * right,
				QuantIntOp::Divide => left / right,
				QuantIntOp::Remainder => left % right,
				QuantIntOp::ShiftLeft => left << right,
				QuantIntOp::ShiftRight => left >> right,
				QuantIntOp::And => left & right,
				QuantIntOp::Or => left | right,
				QuantIntOp::Xor => left ^ right,
			}
		}
		fn equal(&mut self, left: Self::Int, right: Self::Int) -> Self::Int {
			u64::from(left == right)
		}
		fn less(&mut self, left: Self::Int, right: Self::Int) -> Self::Int {
			u64::from(left < right)
		}
		fn select_int(&mut self, condition: Self::Int, yes: Self::Int, no: Self::Int) -> Self::Int {
			if condition != 0 { yes } else { no }
		}
		fn sign_extend(&mut self, value: Self::Int, bits: u8) -> Self::Int {
			((value << (64 - bits)) as i64 >> (64 - bits)) as u64
		}
		fn load(&mut self, bits: u8, offset: Self::Int) -> Self::Int {
			let offset = offset as usize;
			(0..usize::from(bits / 8)).fold(0, |value, byte| value | u64::from(self.bytes[offset + byte]) << (8 * byte))
		}
		fn half(&mut self, offset: Self::Int) -> Self::Value {
			f64::from(half(&self.bytes[offset as usize..]))
		}
		fn float(&mut self, offset: Self::Int) -> Self::Value {
			f64::from(f32::from_le_bytes(self.bytes[offset as usize..offset as usize + 4].try_into().unwrap()))
		}
		fn half_bits(&mut self, bits: Self::Int) -> Self::Value {
			f64::from(unfp16(bits as u16))
		}
		fn table(&mut self, _name: &'static str, values: &'static [u16], index: Self::Int) -> Self::Int {
			u64::from(values[index as usize])
		}
		fn signed_table(&mut self, _name: &'static str, values: &'static [i8], index: Self::Int) -> Self::Int {
			values[index as usize] as i64 as u64
		}
		fn value_table(&mut self, _name: &str, values: &[f64], index: Self::Int) -> Self::Value {
			values[index as usize]
		}
		fn number(&mut self, value: Self::Int, signed: bool) -> Self::Value {
			if signed { value as i64 as f64 } else { value as f64 }
		}
		fn literal(&self, value: f64) -> Self::Value {
			value
		}
		fn value(&mut self, operation: QuantValueOp, left: Self::Value, right: Self::Value) -> Self::Value {
			match operation {
				QuantValueOp::Add => left + right,
				QuantValueOp::Subtract => left - right,
				QuantValueOp::Multiply => left * right,
			}
		}
		fn select_value(&mut self, condition: Self::Int, yes: Self::Value, no: Self::Value) -> Self::Value {
			if condition != 0 { yes } else { no }
		}
		fn signed(&mut self, magnitude: Self::Value, sign: Self::Int) -> Self::Value {
			if sign != 0 { -magnitude } else { magnitude }
		}
	}

	pub(super) struct NativeQuantOps {
		pub(super) globals: String,
		pub(super) ir: String,
		pub(super) backend: Backend,
		pub(super) precision: NativePrecision,
		pub(super) next: usize,
	}

	impl NativeQuantOps {
		fn name(&mut self) -> String {
			let name = format!("%quant.{}", self.next);
			self.next += 1;
			name
		}
		fn instruction(&mut self, instruction: String) -> String {
			let name = self.name();
			self.ir.push_str(&format!("{name} = {instruction}\n"));
			name
		}
	}

	impl QuantOps for NativeQuantOps {
		type Int = String;
		type Value = String;
		fn index(&self) -> Self::Int {
			"%local".to_owned()
		}
		fn integer(&self, value: u64) -> Self::Int {
			value.to_string()
		}
		fn int(&mut self, operation: QuantIntOp, left: Self::Int, right: Self::Int) -> Self::Int {
			let operation = match operation {
				QuantIntOp::Add => "add",
				QuantIntOp::Subtract => "sub",
				QuantIntOp::Multiply => "mul",
				QuantIntOp::Divide => "udiv",
				QuantIntOp::Remainder => "urem",
				QuantIntOp::ShiftLeft => "shl",
				QuantIntOp::ShiftRight => "lshr",
				QuantIntOp::And => "and",
				QuantIntOp::Or => "or",
				QuantIntOp::Xor => "xor",
			};
			self.instruction(format!("{operation} i64 {left}, {right}"))
		}
		fn equal(&mut self, left: Self::Int, right: Self::Int) -> Self::Int {
			let condition = self.instruction(format!("icmp eq i64 {left}, {right}"));
			self.instruction(format!("zext i1 {condition} to i64"))
		}
		fn less(&mut self, left: Self::Int, right: Self::Int) -> Self::Int {
			let condition = self.instruction(format!("icmp ult i64 {left}, {right}"));
			self.instruction(format!("zext i1 {condition} to i64"))
		}
		fn select_int(&mut self, condition: Self::Int, yes: Self::Int, no: Self::Int) -> Self::Int {
			let condition = self.instruction(format!("icmp ne i64 {condition}, 0"));
			self.instruction(format!("select i1 {condition}, i64 {yes}, i64 {no}"))
		}
		fn sign_extend(&mut self, value: Self::Int, bits: u8) -> Self::Int {
			let narrow = self.instruction(format!("trunc i64 {value} to i{bits}"));
			self.instruction(format!("sext i{bits} {narrow} to i64"))
		}
		fn load(&mut self, bits: u8, offset: Self::Int) -> Self::Int {
			let pointer = pointer_type(self.backend);
			let address = self.instruction(format!("getelementptr inbounds i8, {pointer} %block, i64 {offset}"));
			let loaded = self.instruction(format!("load i{bits}, {pointer} {address}, align {}", if bits == 8 { 1 } else { 2 }));
			if bits == 64 { loaded } else { self.instruction(format!("zext i{bits} {loaded} to i64")) }
		}
		fn half(&mut self, offset: Self::Int) -> Self::Value {
			let pointer = pointer_type(self.backend);
			let ty = self.precision.model_type;
			let address = self.instruction(format!("getelementptr inbounds i8, {pointer} %block, i64 {offset}"));
			let loaded = self.instruction(format!("load half, {pointer} {address}, align 2"));
			self.instruction(format!("call {ty} @recipe.from.f16(half {loaded})"))
		}
		fn float(&mut self, offset: Self::Int) -> Self::Value {
			let pointer = pointer_type(self.backend);
			let ty = self.precision.model_type;
			let address = self.instruction(format!("getelementptr inbounds i8, {pointer} %block, i64 {offset}"));
			let loaded = self.instruction(format!("load float, {pointer} {address}, align 4"));
			self.instruction(format!("call {ty} @recipe.from.f32(float {loaded})"))
		}
		fn half_bits(&mut self, bits: Self::Int) -> Self::Value {
			let ty = self.precision.model_type;
			let bits = self.instruction(format!("trunc i64 {bits} to i16"));
			let half = self.instruction(format!("bitcast i16 {bits} to half"));
			self.instruction(format!("call {ty} @recipe.from.f16(half {half})"))
		}
		fn table(&mut self, name: &'static str, values: &'static [u16], index: Self::Int) -> Self::Int {
			let address = self.instruction(format!("getelementptr inbounds [{} x i16], ptr @recipe_model_{name}, i32 0, i64 {index}", values.len()));
			let loaded = self.instruction(format!("load i16, ptr {address}, align 2"));
			self.instruction(format!("zext i16 {loaded} to i64"))
		}
		fn signed_table(&mut self, name: &'static str, values: &'static [i8], index: Self::Int) -> Self::Int {
			let address = self.instruction(format!("getelementptr inbounds [{} x i8], ptr @recipe_model_{name}, i32 0, i64 {index}", values.len()));
			let loaded = self.instruction(format!("load i8, ptr {address}, align 1"));
			self.instruction(format!("sext i8 {loaded} to i64"))
		}
		fn value_table(&mut self, name: &str, values: &[f64], index: Self::Int) -> Self::Value {
			let ty = self.precision.model_type;
			if !self.globals.contains(&format!("@recipe_model_{name} =")) {
				self.globals.push_str(&format!(
					"@recipe_model_{name} = private unnamed_addr constant [{} x {ty}] [{}]\n",
					values.len(),
					values.iter().map(|value| format!("{ty} {}", native_literal(self.precision.model, ty, *value))).collect::<Vec<_>>().join(", ")
				));
			}
			let address = self.instruction(format!("getelementptr inbounds [{} x {ty}], ptr @recipe_model_{name}, i32 0, i64 {index}", values.len()));
			self.instruction(format!("load {ty}, ptr {address}, align {}", super::alignment(ty)))
		}
		fn number(&mut self, value: Self::Int, signed: bool) -> Self::Value {
			let ty = self.precision.model_type;
			let value = self.instruction(format!("trunc i64 {value} to i32"));
			self.instruction(format!("call {ty} @recipe.from.{}32(i32 {value})", if signed { "s" } else { "u" }))
		}
		fn literal(&self, value: f64) -> Self::Value {
			native_literal(self.precision.model, self.precision.model_type, value)
		}
		fn value(&mut self, operation: QuantValueOp, left: Self::Value, right: Self::Value) -> Self::Value {
			let ty = self.precision.model_type;
			let operation = match operation {
				QuantValueOp::Add => "add",
				QuantValueOp::Subtract => "sub",
				QuantValueOp::Multiply => "mul",
			};
			self.instruction(format!("call {ty} @recipe.{operation}({ty} {left}, {ty} {right})"))
		}
		fn select_value(&mut self, condition: Self::Int, yes: Self::Value, no: Self::Value) -> Self::Value {
			let ty = self.precision.model_type;
			let condition = self.instruction(format!("icmp ne i64 {condition}, 0"));
			self.instruction(format!("select i1 {condition}, {ty} {yes}, {ty} {no}"))
		}
		fn signed(&mut self, magnitude: Self::Value, sign: Self::Int) -> Self::Value {
			let ty = self.precision.model_type;
			let negative = self.instruction(format!("call {ty} @recipe.neg({ty} {magnitude})"));
			let sign = self.instruction(format!("icmp ne i64 {sign}, 0"));
			self.instruction(format!("select i1 {sign}, {ty} {negative}, {ty} {magnitude}"))
		}
	}
}
use quantized::{HostQuantOps, Iq1Layout, Iq4Layout, IqLayout, IqPacking, NativeQuantOps, QuantOps, ScalarLayout, dequant_nf4};

impl NativeModelIr {
	pub(crate) fn emit_fixed_primitives(&self, backend: Backend, matrix: bool, reverse: bool, training: bool) -> Result<String> {
		let mut ir = String::new();
		let order = if reverse {
			self.plans.iter().rev().enumerate().map(|(position, plan)| (self.plans.len() - position - 1, plan)).collect::<Vec<_>>()
		} else {
			self.plans.iter().enumerate().collect::<Vec<_>>()
		};
		for (index, plan) in order {
			let pointers = self.emit_pointers(backend, index, plan, reverse, &mut ir)?;
			let node = &plan.node;
			// The reverse pass differentiates the whole sequence at once.
			let window = if reverse { NodeWindow { begin: "0".to_owned(), span: node.output.length.to_string() } } else { self.emit_node_window(index, node, &mut ir)? };
			let (begin, span) = (&window.begin, &window.span);
			match (reverse, node.op) {
				(false, Primitive::Contraction) => {
					let extent = self.schedule.contractions[index].ok_or_else(|| RecipeError::new("native contraction schedule is absent"))?.forward;
					require(node.argument[1] == 0.0 || node.argument[1] == 1.0, "contraction ReLU flag is invalid")?;
					let call = format!(
						"call void @contraction_forward_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {source}, i32 %rows, i32 {in_channels}, i32 {in_length}, i32 {out_channels}, i32 {out_length}, i32 {begin}, i32 {span}, i32 {kernel}, i1 {bias}, i1 {relu}, i1 false, i1 false, i1 false, i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads, i32 0, i32 {decode} )\n",
						pointer = pointer_type(backend),
						bias = node.argument[2] == 0.0,
						decode = plan.decode(index),
						source = pointers.source,
						weights = pointers.weights,
						value = pointers.value,
						in_channels = node.input.channels,
						in_length = node.input.length,
						out_channels = node.output.channels,
						out_length = node.output.length,
						kernel = integer_argument(node.argument[0], "contraction kernel")?,
						relu = node.argument[1] == 1.0,
						tile_m = extent.m,
						tile_n = extent.n,
						tile_k = extent.k
					);
					ir.push_str(&call);
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Gather) => {
					let (layout, _) = embedding_row(node)?;
					let per_row = checked_mul(node.output.channels, node.output.length, "gather row elements")?;
					let (pointer, ty) = (pointer_type(backend), self.precision.model_type);
					let prefix = format!("n{index}.gather");
					emit_fixed_loop(&mut ir, index, "gather", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"%{prefix}.row = udiv i32 {p}, {per_row}\n%{prefix}.within = urem i32 {p}, {per_row}\n%{prefix}.channel = udiv i32 %{prefix}.within, {length}\n%{prefix}.position = urem i32 %{prefix}.within, {length}\n%{prefix}.base = mul i32 %{prefix}.row, {length}\n%{prefix}.token = add i32 %{prefix}.base, %{prefix}.position\n%{prefix}.id.ptr = getelementptr inbounds i32, {pointer} {source}, i32 %{prefix}.token\n%{prefix}.id = load i32, {pointer} %{prefix}.id.ptr, align 4\n%{prefix}.value = call {ty} @recipe_model_quantized_{name}({pointer} {table}, i32 %{prefix}.id, i32 %{prefix}.channel, i32 {width})\n%{prefix}.out = getelementptr inbounds {ty}, {pointer} {value}, i32 {p}\nstore {ty} %{prefix}.value, {pointer} %{prefix}.out, align {align}\n",
							source = pointers.source,
							table = pointers.context,
							value = pointers.value,
							length = node.output.length,
							width = node.output.channels,
							name = layout.name,
							align = alignment(ty)
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::TopK) => {
					// One router decision per row and position: a `[1, length]` shape.
					let positions = Shape { channels: 1, length: node.output.length };
					emit_fixed_loop(&mut ir, index, "topk", self.rows, positions, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @topk_forward_body( {pointer} {source}, {pointer} {value}, i32 {p}, i32 {experts}, i32 {length}, i32 {top}, i32 {scoring}, i32 {renormalize} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							value = pointers.value,
							experts = node.output.channels,
							length = node.output.length,
							top = node.argument[0],
							scoring = node.argument[1],
							renormalize = node.argument[2]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Expand) => {
					emit_fixed_loop(&mut ir, index, "expand", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @expand_forward_body( {pointer} {source}, {pointer} {value}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							value = pointers.value,
							channels = node.input.channels,
							length = node.input.length,
							lanes = node.argument[0]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(reverse, Primitive::Rope) => {
					let (input, output) = if reverse { (&pointers.delta, &pointers.source_adjoint) } else { (&pointers.source, &pointers.value) };
					let ty = self.precision.model_type;
					let base = native_literal(self.precision.model, ty, node.argument[1]);
					emit_fixed_loop(&mut ir, index, if reverse { "rope.reverse" } else { "rope" }, self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @rope_body( {pointer} {input}, {pointer} {output}, i32 {p}, i32 {channels}, i32 {length}, i32 {head_width}, i32 {dims}, i32 {rotated}, {ty} {base}, i1 {reverse} )\n",
							pointer = pointer_type(backend),
							channels = node.output.channels,
							length = node.output.length,
							head_width = node.argument[2],
							dims = node.argument[0],
							rotated = node.argument[3]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::ExpertIn) => {
					emit_fixed_loop(&mut ir, index, "expert.in", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @expert_in_forward_body( {pointer} {source}, {pointer} {routing}, {pointer} {weights}, {pointer} {value}, i32 {p}, i32 {channels}, i32 {length}, i32 {hidden}, i32 {experts}, i32 {top}, i32 {decode} )\n",
							pointer = pointer_type(backend),
							decode = plan.decode(index),
							source = pointers.source,
							routing = pointers.second,
							weights = pointers.weights,
							value = pointers.value,
							channels = node.input.channels,
							length = node.output.length,
							hidden = node.argument[2],
							experts = node.argument[0],
							top = node.argument[1]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Read) => {
					emit_fixed_loop(&mut ir, index, "read", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @read_forward_body( {pointer} {source}, {pointer} {gate}, {pointer} {value}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes}, i1 {gated} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							gate = pointers.second,
							value = pointers.value,
							channels = node.output.channels,
							length = node.output.length,
							lanes = node.argument[0],
							gated = node.second >= 0
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Dconv) => {
					emit_fixed_loop(&mut ir, index, "dconv", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @dconv_forward_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, i32 {p}, i32 {channels}, i32 {length}, i32 {kernel}, i32 {dilation}, i32 {decode} )\n",
							pointer = pointer_type(backend),
							decode = plan.decode(index),
							source = pointers.source,
							weights = pointers.weights,
							value = pointers.value,
							channels = node.output.channels,
							length = node.output.length,
							kernel = node.argument[0],
							dilation = node.argument[1]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Lookup) => {
					// The host stages the gathered rows of the window in the context as
					// [row][position][channel]; the device lays them out as its value plane.
					let (pointer, ty) = (pointer_type(backend), self.precision.model_type);
					let prefix = format!("n{index}.lookup");
					let per_row = checked_mul(node.output.channels, node.output.length, "lookup row elements")?;
					emit_fixed_loop(&mut ir, index, "lookup", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"%{prefix}.row = udiv i32 {p}, {per_row}\n%{prefix}.within = urem i32 {p}, {per_row}\n%{prefix}.channel = udiv i32 %{prefix}.within, {length}\n%{prefix}.position = urem i32 %{prefix}.within, {length}\n%{prefix}.token = mul i32 %{prefix}.row, {length}\n%{prefix}.slot = add i32 %{prefix}.token, %{prefix}.position\n%{prefix}.base = mul i32 %{prefix}.slot, {channels}\n%{prefix}.index = add i32 %{prefix}.base, %{prefix}.channel\n%{prefix}.in = getelementptr inbounds {ty}, {pointer} {context}, i32 %{prefix}.index\n%{prefix}.value = load {ty}, {pointer} %{prefix}.in, align {align}\n%{prefix}.out = getelementptr inbounds {ty}, {pointer} {value}, i32 {p}\nstore {ty} %{prefix}.value, {pointer} %{prefix}.out, align {align}\n",
							context = pointers.context,
							value = pointers.value,
							length = node.output.length,
							channels = node.output.channels,
							align = alignment(ty)
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Fold) => {
					emit_fixed_loop(&mut ir, index, "fold", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @fold_forward_body( {pointer} {source}, {pointer} {value}, i32 {p}, i32 {groups}, i32 {width}, i32 {length} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							value = pointers.value,
							groups = node.output.channels,
							width = node.argument[0],
							length = node.output.length
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Outer) => {
					emit_fixed_loop(&mut ir, index, "outer", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @outer_forward_body( {pointer} {source}, {pointer} {gate}, {pointer} {value}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes}, i1 {gated} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							gate = pointers.second,
							value = pointers.value,
							channels = node.input.channels,
							length = node.input.length,
							lanes = node.argument[0],
							gated = node.second >= 0
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Delta) => {
					// One row and head per element: a `[heads, 1]` shape walked whole.
					let shape = delta_shape(node, self.rows)?;
					let pairs = Shape { channels: shape.heads as usize, length: 1 };
					let whole = NodeWindow { begin: "0".to_owned(), span: "1".to_owned() };
					emit_fixed_loop(&mut ir, index, "delta", self.rows, pairs, &whole, |ir, p| {
						ir.push_str(&format!(
							"call void @delta_forward_body( {pointer} {source}, {pointer} {second}, {pointer} {weights}, {pointer} {value}, {pointer} {context}, i32 {p}, {arguments}, i32 {decode} )\n",
							pointer = pointer_type(backend),
							decode = plan.decode(index),
							source = pointers.source,
							second = pointers.second,
							weights = pointers.weights,
							value = pointers.value,
							context = pointers.context,
							arguments = shape.arguments
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::ExpertOut) => {
					emit_fixed_loop(&mut ir, index, "expert.out", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @expert_out_forward_body( {pointer} {source}, {pointer} {routing}, {pointer} {weights}, {pointer} {value}, i32 {p}, i32 {channels}, i32 {length}, i32 {hidden}, i32 {experts}, i32 {top}, i32 {decode} )\n",
							pointer = pointer_type(backend),
							decode = plan.decode(index),
							source = pointers.source,
							routing = pointers.second,
							weights = pointers.weights,
							value = pointers.value,
							channels = node.output.channels,
							length = node.output.length,
							hidden = node.argument[2],
							experts = node.argument[0],
							top = node.argument[1]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Pool) => {
					let size = integer_argument(node.argument[0], "pool size")?;
					emit_fixed_loop(&mut ir, index, "pool", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @pool_forward_body( {pointer} {source}, {pointer} {value}, {pointer} {context}, i32 {p}, i32 {from}, i32 {to}, i32 {size}, i32 {channels} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							value = pointers.value,
							context = pointers.context,
							p = p,
							from = node.input.elements(),
							to = node.output.elements(),
							size = size,
							channels = node.input.channels
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Attention) => {
					let extent = self.schedule.attention[index].ok_or_else(|| RecipeError::new("native attention schedule is absent"))?;
					let attention = if matrix && extent.m as usize == node.output.length { "attention_forward_matrix_body" } else { "attention_forward_body" };
					let selectors = attention_selectors(node, &self.precision)?;
					let (heads, from, channels) = (integer_argument(node.argument[0], "attention heads")?, node.output.elements(), node.output.channels);
					let blocks = attention_blocks(node);
					if blocks != 0 {
						// The indexer reads its own projection, the node's second source, and
						// keeps its state in the context arena: one running sum of indexer
						// keys per block. The index loop visits only the blocks the window's
						// positions land in and extends their sums by those positions, and
						// the select loop scores only the window's queries, so a step costs
						// the blocks it touches and a whole forward costs the sequence once.
						let (pointer, source, context) = (pointer_type(backend), &pointers.second, &pointers.context);
						let shared = format!("i32 %rows, i32 {from}, i32 {heads}, i32 {channels}, {selectors}");
						let keep = integer_argument(node.argument[4], "indexer blocks kept")?;
						let block = integer_argument(node.argument[3], "indexer block")?;
						let (first, count) = (format!("%n{index}.index.first"), format!("%n{index}.index.count"));
						let end = format!("%n{index}.end");
						ir.push_str(&format!(
							"{first} = udiv i32 {begin}, {block}\n%n{index}.index.stop = add i32 {end}, {last}\n%n{index}.index.last = udiv i32 %n{index}.index.stop, {block}\n%n{index}.index.touched = sub i32 %n{index}.index.last, {first}\n%n{index}.index.empty = icmp eq i32 {span}, 0\n{count} = select i1 %n{index}.index.empty, i32 0, i32 %n{index}.index.touched\n",
							last = block - 1
						));
						let touched = NodeWindow { begin: first, span: count };
						emit_fixed_loop(&mut ir, index, "index", self.rows, Shape { channels: 1, length: blocks }, &touched, |ir, p| {
							ir.push_str(&format!("call void @attention_index_body( {pointer} {source}, {pointer} {context}, i32 {p}, i32 {begin}, i32 {end}, {shared} )\n"));
						})?;
						ir.push_str(barrier(backend));
						emit_fixed_loop(&mut ir, index, "select", self.rows, Shape { channels: 1, length: node.output.length }, &window, |ir, p| {
							ir.push_str(&format!("call void @attention_select_body( {pointer} {source}, {pointer} {context}, i32 {p}, i32 {keep}, {shared} )\n"));
						})?;
						ir.push_str(barrier(backend));
					}
					// The matrix body scores the whole sequence at once. Its keys past
					// the window are zero and the causal mask drops them, so it stays
					// correct on a step but reworks the positions the window skips.
					let extended = if attention == "attention_forward_body" { format!("i32 {begin}, i32 {span}, ") } else { String::new() };
					ir.push_str(&format!("call void @{attention}( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {context}, i32 %rows, i32 {from}, i32 {heads}, i32 {channels}, {extended}i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads, {selectors} )\n", pointer = pointer_type(backend), source = pointers.source, weights = pointers.weights, value = pointers.value, context = pointers.context, tile_m = extent.m, tile_n = extent.n, tile_k = extent.k));
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Scan) => {
					let extent = self.schedule.contractions[index].ok_or_else(|| RecipeError::new("native scan schedule is absent"))?.forward;
					ir.push_str(&format!("call void @scan_forward_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {context}, i32 %rows, i32 {in_channels}, i32 {in_length}, i32 {out_channels}, i32 {begin}, i32 {span}, i32 {gates}, i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads, i32 0, i32 {decode} )\n", decode = plan.decode(index), pointer = pointer_type(backend), source = pointers.source, weights = pointers.weights, value = pointers.value, context = pointers.context, in_channels = node.input.channels, in_length = node.input.length, out_channels = node.output.channels, gates = integer_argument(node.argument[0], "scan gates")?, tile_m = extent.m, tile_n = extent.n, tile_k = extent.k));
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Elementwise) => {
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let literal = |value: f64, ty: &str| native_literal(self.precision.model, ty, value);
					let prefix = format!("n{index}.scalar");
					let first = format!("%{prefix}.first");
					let second = format!("%{prefix}.second");
					let second_operand = if pointers.second == pointers.source { first.as_str() } else { second.as_str() };
					let code_end = node
						.program_offset
						.checked_add(node.program_count.checked_mul(3).ok_or_else(|| RecipeError::new("scalar program length overflows"))?)
						.ok_or_else(|| RecipeError::new("scalar program range overflows"))?;
					let code = self.graph.programs.get(node.program_offset..code_end).ok_or_else(|| RecipeError::new(format!("node {index} scalar program range is invalid")))?;
					let forward = program_ir::emit_scalar_forward(
						code,
						program_ir::ScalarContext {
							value_type: ty,
							pointer_type: pointer,
							alignment: alignment(ty),
							first: &first,
							second: second_operand,
							weights: &pointers.weights,
							decode: plan.decode(index),
							prefix: &prefix,
							literal: &literal,
						},
					)
					.map_err(|error| RecipeError::new(error.to_string()))?;
					emit_fixed_loop(&mut ir, index, "scalar", self.rows, node.output, &window, |ir, p| {
						let first_pointer = format!("%{prefix}.first.ptr");
						let output_pointer = format!("%{prefix}.output.ptr");
						ir.push_str(&format!(
							"{first_pointer} = getelementptr inbounds {ty}, {pointer} {source}, i32 {p}\n{first} = load {ty}, {pointer} {first_pointer}, align {align}\n",
							source = pointers.source,
							align = alignment(ty)
						));
						if pointers.second != pointers.source {
							let second_pointer = format!("%{prefix}.second.ptr");
							ir.push_str(&format!("{second_pointer} = getelementptr inbounds {ty}, {pointer} {second_source}, i32 {p}\n", second_source = pointers.second));
							ir.push_str(&format!("{second} = load {ty}, {pointer} {second_pointer}, align {align}\n", align = alignment(ty)));
						}
						ir.push_str(&forward.code);
						ir.push_str(&format!(
							"{output_pointer} = getelementptr inbounds {ty}, {pointer} {value}, i32 {p}\nstore {ty} {result}, {pointer} {output_pointer}, align {align}\n",
							value = pointers.value,
							result = forward.value,
							align = alignment(ty)
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Predictor) => {
					let locals = integer_argument(node.argument[0], "predictor locals")?;
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let literal = |value: f64, ty: &str| native_literal(self.precision.model, ty, value);
					let prefix = format!("n{index}.predictor");
					let code_end = node
						.program_offset
						.checked_add(node.program_count.checked_mul(2).ok_or_else(|| RecipeError::new("predictor program length overflows"))?)
						.ok_or_else(|| RecipeError::new("predictor program range overflows"))?;
					let code = self.graph.programs.get(node.program_offset..code_end).ok_or_else(|| RecipeError::new(format!("node {index} predictor program range is invalid")))?;
					let locals = usize::try_from(locals).map_err(|_| RecipeError::new("predictor locals exceed usize"))?;
					let row = format!("%{prefix}.row");
					let forward = program_ir::emit_predictor_forward(
						code,
						locals,
						program_ir::PredictorContext {
							value_type: ty,
							pointer_type: pointer,
							alignment: alignment(ty),
							input: &pointers.source,
							row: &row,
							features: node.input.elements(),
							weights: &pointers.weights,
							parameters: node.parameters,
							prefix: &prefix,
							literal: &literal,
						},
					)
					.map_err(|error| RecipeError::new(error.to_string()))?;
					emit_fixed_loop(&mut ir, index, "predictor", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!("{row} = udiv i32 {p}, {elements}\n", elements = node.output.elements()));
						ir.push_str(&forward.code);
						let output_pointer = format!("%{prefix}.output.ptr");
						ir.push_str(&format!(
							"{output_pointer} = getelementptr inbounds {ty}, {pointer} {value}, i32 {p}\nstore {ty} {result}, {pointer} {output_pointer}, align {align}\n",
							value = pointers.value,
							result = forward.value,
							align = alignment(ty)
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Contraction) => {
					let tiles = self.schedule.contractions[index].ok_or_else(|| RecipeError::new("native contraction schedule is absent"))?;
					require(node.argument[1] == 0.0 || node.argument[1] == 1.0, "contraction ReLU flag is invalid")?;
					let kernel = integer_argument(node.argument[0], "contraction kernel")?;
					let composed_previous = kernel <= 1;
					let matrix_gradient = matrix;
					let accumulate_previous = self.plans[index + 1..].iter().any(|candidate| candidate.node.source == node.source || candidate.node.second == node.source);
					ir.push_str(&format!("call void @contraction_reverse_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {delta}, {pointer} {source_adjoint}, {pointer} %gradient, i1 {write_input}, i1 {bias}, i1 {relu}, i1 {matrix_gradient}, i32 %rows, i32 {in_channels}, i32 {in_length}, i32 {out_channels}, i32 {out_length}, i32 {kernel}, i32 {offset}, i32 {gradient_m}, i32 {gradient_n}, i32 {gradient_k}, i32 {previous_m}, i32 {previous_n}, i32 {previous_k}, i32 %threads )\n", pointer = pointer_type(backend), source = pointers.source, weights = pointers.weights, value = pointers.value, delta = pointers.delta, source_adjoint = pointers.source_adjoint, write_input = !composed_previous, bias = node.argument[2] == 0.0, matrix_gradient = matrix_gradient, in_channels = node.input.channels, in_length = node.input.length, out_channels = node.output.channels, out_length = node.output.length, kernel = kernel, offset = plan.node.offset, relu = node.argument[1] == 1.0, gradient_m = tiles.gradient.m, gradient_n = tiles.gradient.n, gradient_k = tiles.gradient.k, previous_m = tiles.previous.m, previous_n = tiles.previous.n, previous_k = tiles.previous.k));
					if composed_previous {
						ir.push_str(&format!("call void @contraction_forward_body( {pointer} {delta}, {pointer} {weights}, {pointer} {source_adjoint}, {pointer} {value}, i32 %rows, i32 {out_channels}, i32 {out_length}, i32 {in_channels}, i32 {in_length}, i32 0, i32 {in_length}, i32 0, i1 false, i1 {relu}, i1 true, i1 true, i1 {accumulate}, i32 {previous_m}, i32 {previous_n}, i32 {previous_k}, i32 %threads, i32 0, i32 0 )\n", pointer = pointer_type(backend), delta = pointers.delta, weights = pointers.weights, source_adjoint = pointers.source_adjoint, value = pointers.value, out_channels = node.output.channels, out_length = node.output.length, in_channels = node.input.channels, in_length = node.input.length, relu = node.argument[1] == 1.0, accumulate = accumulate_previous, previous_m = tiles.previous.m, previous_n = tiles.previous.n, previous_k = tiles.previous.k));
					}
					ir.push_str(barrier(backend));
				}
				// The gather reads the packed table the run was given and the optimizer
				// leaves it frozen, so the embedding contributes no reverse pass.
				(true, Primitive::Gather) => {}
				(true, Primitive::Expand) => {
					emit_fixed_loop(&mut ir, index, "expand.reverse", self.rows, node.input, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @expand_reverse_body( {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes} )\n",
							pointer = pointer_type(backend),
							delta = pointers.delta,
							adjoint = pointers.source_adjoint,
							channels = node.input.channels,
							length = node.input.length,
							lanes = node.argument[0]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::TopK) => {
					let positions = Shape { channels: 1, length: node.output.length };
					emit_fixed_loop(&mut ir, index, "topk.reverse", self.rows, positions, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @topk_reverse_body( {pointer} {source}, {pointer} {value}, {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {experts}, i32 {length}, i32 {scoring}, i32 {renormalize} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							value = pointers.value,
							delta = pointers.delta,
							adjoint = pointers.source_adjoint,
							experts = node.output.channels,
							length = node.output.length,
							scoring = node.argument[1],
							renormalize = node.argument[2]
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Dconv) => {
					emit_fixed_loop(&mut ir, index, "dconv.reverse", self.rows, node.output, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @dconv_reverse_input_body( {pointer} {weights}, {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {kernel}, i32 {dilation} )\n",
							pointer = pointer_type(backend),
							weights = pointers.weights,
							delta = pointers.delta,
							adjoint = pointers.source_adjoint,
							channels = node.output.channels,
							length = node.output.length,
							kernel = node.argument[0],
							dilation = node.argument[1]
						));
					})?;
					ir.push_str(barrier(backend));
					// One tap per channel and kernel position: a `[channels, kernel]` shape walked whole.
					let kernel = integer_argument(node.argument[0], "depthwise kernel")? as usize;
					let taps = Shape { channels: node.output.channels, length: kernel };
					let whole = NodeWindow { begin: "0".to_owned(), span: kernel.to_string() };
					emit_fixed_loop(&mut ir, index, "dconv.weight.reverse", 1, taps, &whole, |ir, p| {
						ir.push_str(&format!(
							"call void @dconv_reverse_weight_body( {pointer} {source}, {pointer} {delta}, {pointer} %gradient, i32 {p}, i32 %rows, i32 {channels}, i32 {length}, i32 {kernel}, i32 {dilation}, i32 {offset} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							delta = pointers.delta,
							channels = node.output.channels,
							length = node.output.length,
							kernel = node.argument[0],
							dilation = node.argument[1],
							offset = node.offset
						));
					})?;
					ir.push_str(barrier(backend));
				}
				// The lookup's table stays on the host and is never trained, so the
				// rows it stages contribute no reverse pass.
				(true, Primitive::Lookup) => {}
				(true, Primitive::Fold) => {
					emit_fixed_loop(&mut ir, index, "fold.reverse", self.rows, node.input, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @fold_reverse_body( {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {groups}, i32 {width}, i32 {length} )\n",
							pointer = pointer_type(backend),
							delta = pointers.delta,
							adjoint = pointers.source_adjoint,
							groups = node.output.channels,
							width = node.argument[0],
							length = node.output.length
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::ExpertIn) => {
					let (channels, length) = (node.input.channels, node.output.length);
					let (hidden, experts, top) = (node.argument[2], node.argument[0], node.argument[1]);
					emit_fixed_loop(&mut ir, index, "expert.in.reverse", self.rows, node.input, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @expert_in_reverse_input_body( {pointer} {routing}, {pointer} {weights}, {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {hidden}, i32 {experts}, i32 {top} )\n",
							pointer = pointer_type(backend),
							routing = pointers.second,
							weights = pointers.weights,
							delta = pointers.delta,
							adjoint = pointers.source_adjoint
						));
					})?;
					ir.push_str(barrier(backend));
					emit_expert_buckets(&mut ir, backend, index, self.rows, node, &pointers)?;
					let offset = narrow(plan.node.offset, "expert gradient offset")?;
					// One table entry per element: a `[1, parameters]` shape walked whole.
					let table = Shape { channels: 1, length: node.parameters };
					let whole = NodeWindow { begin: "0".to_owned(), span: node.parameters.to_string() };
					emit_fixed_loop(&mut ir, index, "expert.in.gradient", 1, table, &whole, |ir, p| {
						ir.push_str(&format!(
							"call void @expert_in_reverse_weight_body( {pointer} {source}, {pointer} {delta}, {pointer} {context}, {pointer} %gradient, i32 {p}, i32 {channels}, i32 {length}, i32 {hidden}, i32 {experts}, i32 {top}, i32 {offset} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							delta = pointers.delta,
							context = pointers.context
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Read) => {
					emit_fixed_loop(&mut ir, index, "read.reverse", self.rows, node.input, &window, |ir, p| {
						ir.push_str(&format!("call void @read_reverse_body( {pointer} {source}, {pointer} {gate}, {pointer} {delta}, {pointer} {adjoint}, {pointer} {gate_adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes}, i1 {gated} )\n", pointer = pointer_type(backend), source = pointers.source, gate = pointers.second, delta = pointers.delta, adjoint = pointers.source_adjoint, gate_adjoint = pointers.second_adjoint, channels = node.output.channels, length = node.output.length, lanes = node.argument[0], gated = node.second >= 0));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Outer) => {
					emit_fixed_loop(&mut ir, index, "outer.reverse", self.rows, node.input, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @outer_reverse_branch_body( {pointer} {gate}, {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes}, i1 {gated} )\n",
							pointer = pointer_type(backend),
							gate = pointers.second,
							delta = pointers.delta,
							adjoint = pointers.source_adjoint,
							channels = node.input.channels,
							length = node.input.length,
							lanes = node.argument[0],
							gated = node.second >= 0
						));
					})?;
					ir.push_str(barrier(backend));
					if node.second >= 0 {
						// One gate per row, lane, and position: a `[lanes, length]` shape walked whole.
						let gates = Shape { channels: integer_argument(node.argument[0], "outer lanes")? as usize, length: node.input.length };
						emit_fixed_loop(&mut ir, index, "outer.gate.reverse", self.rows, gates, &window, |ir, p| {
							ir.push_str(&format!(
								"call void @outer_reverse_gate_body( {pointer} {source}, {pointer} {delta}, {pointer} {gate_adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes} )\n",
								pointer = pointer_type(backend),
								source = pointers.source,
								delta = pointers.delta,
								gate_adjoint = pointers.second_adjoint,
								channels = node.input.channels,
								length = node.input.length,
								lanes = node.argument[0]
							));
						})?;
						ir.push_str(barrier(backend));
					}
				}
				(true, Primitive::Delta) => {
					let shape = delta_shape(node, self.rows)?;
					let keys = Shape { channels: shape.key_heads as usize, length: 1 };
					let pairs = Shape { channels: shape.heads as usize, length: 1 };
					let whole = NodeWindow { begin: "0".to_owned(), span: "1".to_owned() };
					// One row and key head per element, so the value heads sharing a key head
					// walk in one thread and own the query and key adjoint elements they share.
					emit_fixed_loop(&mut ir, index, "delta.reverse", self.rows, keys, &whole, |ir, p| {
						ir.push_str(&format!(
							"call void @delta_reverse_body( {pointer} {source}, {pointer} {second}, {pointer} {weights}, {pointer} {context}, {pointer} {delta}, {pointer} {adjoint}, {pointer} {gate} , i32 {p}, {arguments} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							second = pointers.second,
							weights = pointers.weights,
							context = pointers.context,
							delta = pointers.delta,
							adjoint = pointers.source_adjoint,
							gate = pointers.second_adjoint,
							arguments = shape.arguments
						));
					})?;
					ir.push_str(barrier(backend));
					// Then one decay scale per value head, folding that head's row partials.
					emit_fixed_loop(&mut ir, index, "delta.decay.reverse", 1, pairs, &whole, |ir, p| {
						ir.push_str(&format!(
							"call void @delta_reverse_decay_body( {pointer} {context}, {pointer} %gradient, i32 {p}, i32 %rows, i32 {heads}, i32 {partials}, i32 {offset} )\n",
							pointer = pointer_type(backend),
							context = pointers.context,
							heads = shape.heads,
							partials = shape.partials,
							offset = node.offset
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::ExpertOut) => {
					let (channels, length) = (node.output.channels, node.output.length);
					let (hidden, experts, top) = (node.argument[2], node.argument[0], node.argument[1]);
					emit_fixed_loop(&mut ir, index, "expert.out.reverse", self.rows, node.input, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @expert_out_reverse_values_body( {pointer} {routing}, {pointer} {weights}, {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {hidden}, i32 {experts}, i32 {top} )\n",
							pointer = pointer_type(backend),
							routing = pointers.second,
							weights = pointers.weights,
							delta = pointers.delta,
							adjoint = pointers.source_adjoint
						));
					})?;
					ir.push_str(barrier(backend));
					let positions = Shape { channels: 1, length };
					emit_fixed_loop(&mut ir, index, "expert.out.routing", self.rows, positions, &window, |ir, p| {
						ir.push_str(&format!(
							"call void @expert_out_reverse_routing_body( {pointer} {source}, {pointer} {routing}, {pointer} {weights}, {pointer} {delta}, {pointer} {adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {hidden}, i32 {experts}, i32 {top} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							routing = pointers.second,
							weights = pointers.weights,
							delta = pointers.delta,
							adjoint = pointers.second_adjoint
						));
					})?;
					ir.push_str(barrier(backend));
					emit_expert_buckets(&mut ir, backend, index, self.rows, node, &pointers)?;
					let offset = narrow(plan.node.offset, "expert gradient offset")?;
					let table = Shape { channels: 1, length: node.parameters };
					let whole = NodeWindow { begin: "0".to_owned(), span: node.parameters.to_string() };
					emit_fixed_loop(&mut ir, index, "expert.out.gradient", 1, table, &whole, |ir, p| {
						ir.push_str(&format!(
							"call void @expert_out_reverse_weight_body( {pointer} {source}, {pointer} {routing}, {pointer} {delta}, {pointer} {context}, {pointer} %gradient, i32 {p}, i32 {channels}, i32 {length}, i32 {hidden}, i32 {experts}, i32 {top}, i32 {offset} )\n",
							pointer = pointer_type(backend),
							source = pointers.source,
							routing = pointers.second,
							delta = pointers.delta,
							context = pointers.context
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Pool) => {
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let prefix = format!("n{index}.pool.reverse");
					emit_fixed_loop(&mut ir, index, "pool.reverse", self.rows, node.output, &window, |ir, p| {
						let context_pointer = format!("%{prefix}.context.ptr");
						let context_wide = format!("%{prefix}.context.index.wide");
						let context_index = format!("%{prefix}.context.index");
						let delta_pointer = format!("%{prefix}.delta.ptr");
						let delta_value = format!("%{prefix}.delta.value");
						let source_pointer = format!("%{prefix}.source.adjoint.ptr");
						let source_value = format!("%{prefix}.source.adjoint.value");
						let source_sum = format!("%{prefix}.source.adjoint.sum");
						ir.push_str(&format!("{context_pointer} = getelementptr inbounds i64, {pointer} {context}, i32 {p}\n{context_wide} = load i64, {pointer} {context_pointer}, align 8\n{context_index} = trunc i64 {context_wide} to i32\n{delta_pointer} = getelementptr inbounds {ty}, {pointer} {delta}, i32 {p}\n{delta_value} = load {ty}, {pointer} {delta_pointer}, align {align}\n{source_pointer} = getelementptr inbounds {ty}, {pointer} {source_adjoint}, i32 {context_index}\n{source_value} = load {ty}, {pointer} {source_pointer}, align {align}\n{source_sum} = call {ty} @recipe.add({ty} {source_value}, {ty} {delta_value})\nstore {ty} {source_sum}, {pointer} {source_pointer}, align {align}\n", context = pointers.context, delta = pointers.delta, source_adjoint = pointers.source_adjoint, align = alignment(ty), pointer = pointer, ty = ty));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Attention) => {
					let extent = self.schedule.attention[index].ok_or_else(|| RecipeError::new("native attention schedule is absent"))?;
					let attention = if matrix && extent.m as usize == node.output.length { "attention_reverse_matrix_body" } else { "attention_reverse_body" };
					let selectors = attention_selectors(node, &self.precision)?;
					let (heads, from, channels) = (integer_argument(node.argument[0], "attention heads")?, node.output.elements(), node.output.channels);
					ir.push_str(&format!("call void @{attention}( {pointer} {source}, {pointer} {value}, {pointer} {context}, {pointer} {delta}, {pointer} {source_adjoint}, i32 %rows, i32 {from}, i32 {heads}, i32 {channels}, i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads, {selectors} )\n", pointer = pointer_type(backend), source = pointers.source, value = pointers.value, context = pointers.context, delta = pointers.delta, source_adjoint = pointers.source_adjoint, tile_m = extent.m, tile_n = extent.n, tile_k = extent.k));
					ir.push_str(barrier(backend));
					if attention_blocks(node) != 0 {
						// The indexer gradient lands in the side projection's adjoint, which
						// the attention node alone writes.
						let (pointer, source, context, source_adjoint) = (pointer_type(backend), &pointers.second, &pointers.context, &pointers.second_adjoint);
						emit_fixed_loop(&mut ir, index, "index.reverse", self.rows, Shape { channels: 1, length: node.output.length }, &window, |ir, p| {
							ir.push_str(&format!(
								"call void @attention_index_reverse_body( {pointer} {source}, {pointer} {context}, {pointer} {source_adjoint}, i32 {p}, i32 %rows, i32 {from}, i32 {heads}, i32 {channels}, {selectors} )\n"
							));
						})?;
						ir.push_str(barrier(backend));
					}
				}
				(true, Primitive::Scan) => {
					let tiles = self.schedule.contractions[index].ok_or_else(|| RecipeError::new("native scan schedule is absent"))?;
					ir.push_str(&format!("call void @scan_reverse_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {context}, {pointer} {delta}, {pointer} {source_adjoint}, {pointer} %gradient, i1 true, i32 %rows, i32 {in_channels}, i32 {in_length}, i32 {out_channels}, i32 {gates}, i32 {parameters}, i32 {offset}, i32 {gradient_m}, i32 {gradient_n}, i32 {gradient_k}, i32 {previous_m}, i32 {previous_n}, i32 {previous_k}, i32 %threads )\n", pointer = pointer_type(backend), source = pointers.source, weights = pointers.weights, value = pointers.value, context = pointers.context, delta = pointers.delta, source_adjoint = pointers.source_adjoint, in_channels = node.input.channels, in_length = node.input.length, out_channels = node.output.channels, gates = integer_argument(node.argument[0], "scan gates")?, parameters = node.parameters, offset = plan.node.offset, gradient_m = tiles.gradient.m, gradient_n = tiles.gradient.n, gradient_k = tiles.gradient.k, previous_m = tiles.previous.m, previous_n = tiles.previous.n, previous_k = tiles.previous.k));
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Predictor) => {
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Elementwise) => {
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let literal = |value: f64, ty: &str| native_literal(self.precision.model, ty, value);
					let prefix = format!("n{index}.scalar.reverse");
					let first = format!("%{prefix}.first");
					let second = format!("%{prefix}.second");
					let second_operand = if pointers.second == pointers.source { first.as_str() } else { second.as_str() };
					let code_end = node
						.program_offset
						.checked_add(node.program_count.checked_mul(3).ok_or_else(|| RecipeError::new("scalar reverse program length overflows"))?)
						.ok_or_else(|| RecipeError::new("scalar reverse program range overflows"))?;
					let code = self.graph.programs.get(node.program_offset..code_end).ok_or_else(|| RecipeError::new(format!("node {index} scalar reverse program range is invalid")))?;
					let forward = program_ir::emit_scalar_forward(
						code,
						program_ir::ScalarContext {
							value_type: ty,
							pointer_type: pointer,
							alignment: alignment(ty),
							first: &first,
							second: second_operand,
							weights: &pointers.weights,
							decode: plan.decode(index),
							prefix: &prefix,
							literal: &literal,
						},
					)
					.map_err(|error| RecipeError::new(error.to_string()))?;
					let incoming = format!("%{prefix}.incoming");
					let reverse = program_ir::emit_scalar_reverse(
						code,
						program_ir::ScalarContext {
							value_type: ty,
							pointer_type: pointer,
							alignment: alignment(ty),
							first: &first,
							second: second_operand,
							weights: &pointers.weights,
							decode: plan.decode(index),
							prefix: &prefix,
							literal: &literal,
						},
						&incoming,
					)
					.map_err(|error| RecipeError::new(error.to_string()))?;
					let gradients = reverse.parameter_adjoint.iter().map(|(&parameter, value)| Ok((parameter, value.clone()))).collect::<Result<Vec<_>>>()?;
					let scalar_body = |ir: &mut String, p: &str| {
						let first_pointer = format!("%{prefix}.first.ptr");
						let incoming_pointer = format!("%{prefix}.incoming.ptr");
						let first_adjoint_pointer = format!("%{prefix}.first.adjoint.ptr");
						ir.push_str(&format!(
							"{first_pointer} = getelementptr inbounds {ty}, {pointer} {source}, i32 {p}\n{first} = load {ty}, {pointer} {first_pointer}, align {align}\n",
							source = pointers.source,
							align = alignment(ty)
						));
						if pointers.second != pointers.source {
							let second_pointer = format!("%{prefix}.second.ptr");
							ir.push_str(&format!(
								"{second_pointer} = getelementptr inbounds {ty}, {pointer} {second_source}, i32 {p}\n{second} = load {ty}, {pointer} {second_pointer}, align {align}\n",
								second_source = pointers.second,
								align = alignment(ty)
							));
						}
						ir.push_str(&format!(
							"{incoming_pointer} = getelementptr inbounds {ty}, {pointer} {delta}, i32 {p}\n{incoming} = load {ty}, {pointer} {incoming_pointer}, align {align}\n",
							delta = pointers.delta,
							align = alignment(ty)
						));
						ir.push_str(&forward.code);
						ir.push_str(&reverse.code);
						ir.push_str(&format!("{first_adjoint_pointer} = getelementptr inbounds {ty}, {pointer} {source_adjoint}, i32 {p}\n", source_adjoint = pointers.source_adjoint));
						if node.second >= 0 {
							let second_adjoint_pointer = format!("%{prefix}.second.adjoint.ptr");
							ir.push_str(&accumulate_owned(&first_adjoint_pointer, &reverse.first_adjoint, ty, pointer, &format!("{prefix}.first.owned")));
							ir.push_str(&format!(
								"{second_adjoint_pointer} = getelementptr inbounds {ty}, {pointer} {second_adjoint}, i32 {p}\n",
								second_adjoint = pointers.second_adjoint
							));
							ir.push_str(&accumulate_owned(&second_adjoint_pointer, &reverse.second_adjoint, ty, pointer, &format!("{prefix}.second.owned")));
						} else {
							let combined = format!("%{prefix}.combined");
							ir.push_str(&format!(
								"{combined} = call {ty} @recipe.add({ty} {first_adjoint}, {ty} {second_adjoint})\n",
								first_adjoint = reverse.first_adjoint,
								second_adjoint = reverse.second_adjoint
							));
							ir.push_str(&accumulate_owned(&first_adjoint_pointer, &combined, ty, pointer, &format!("{prefix}.combined.owned")));
						}
					};
					if gradients.is_empty() {
						emit_fixed_loop(&mut ir, index, "scalar.reverse", self.rows, node.output, &window, scalar_body)?;
						ir.push_str(barrier(backend));
					} else {
						// A trainable scalar is one destination shared by every element, so
						// the summation order has to belong to the program rather than to
						// the schedule. Each partition sums its own contiguous run of
						// elements in ascending order into its own scratch row, and one
						// owner then folds the rows in ascending partition order.
						let count = checked_mul(self.rows, node.output.elements(), "scalar reverse count")?;
						let partitions = count.min(NATIVE_SCALAR_PARTITIONS).max(1);
						emit_partitioned_loop(
							&mut ir,
							index,
							"scalar.reverse",
							PartitionedLoop {
								count,
								partitions,
								columns: node.parameters,
								value_type: ty,
								pointer_type: pointer,
								scratch: &pointers.context,
								zero: &literal(0.0, ty),
								gradients: &gradients,
							},
							scalar_body,
						)?;
						ir.push_str(barrier(backend));
						let (columns, offset) = (narrow(node.parameters, "scalar gradient columns")?, narrow(plan.node.offset, "scalar gradient offset")?);
						ir.push_str(&format!(
							"call void @reduce_rows({pointer} {context}, {pointer} %gradient, i32 {partitions}, i32 {columns}, i32 {columns}, i32 0, i32 {offset}, i32 %threads)\n",
							context = pointers.context
						));
						ir.push_str(barrier(backend));
					}
				}
				(false, Primitive::Normalize) => {
					let mode = normalize_mode(node.argument[0])?;
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let prefix = format!("n{index}.normalize");
					let weight = (node.parameters != 0).then_some(pointers.weights.as_str());
					if mode != program_ir::NormalizeMode::Evaluation && (training || mode.per_row()) {
						ir.push_str(&self.emit_normalize_stats(backend, index, node, &pointers, mode)?);
						ir.push_str(barrier(backend));
					}
					emit_fixed_loop(&mut ir, index, "normalize", self.rows, node.output, &window, |ir, p| {
						let source_pointer = format!("%{prefix}.source.ptr");
						let source_value = format!("%{prefix}.source.value");
						ir.push_str(&format!(
							"{source_pointer} = getelementptr inbounds {ty}, {pointer} {source}, i32 {p}\n{source_value} = load {ty}, {pointer} {source_pointer}, align {align}\n",
							source = pointers.source,
							align = alignment(ty)
						));
						let fragment = program_ir::emit_normalize(
							program_ir::NormalizeContext {
								value_type: ty,
								pointer_type: pointer,
								alignment: alignment(ty),
								source_value: &source_value,
								context: &pointers.context,
								rows: "%rows",
								channels: node.output.channels,
								length: node.output.length,
								width: normalize_width(node),
								span: normalize_span(node),
								weight,
								mode,
								prefix: &prefix,
							},
							p,
						);
						ir.push_str(&fragment.code);
						let output_pointer = format!("%{prefix}.output.ptr");
						ir.push_str(&format!(
							"{output_pointer} = getelementptr inbounds {ty}, {pointer} {value}, i32 {p}\nstore {ty} {result}, {pointer} {output_pointer}, align {align}\n",
							value = pointers.value,
							result = fragment.value,
							align = alignment(ty)
						));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Normalize) => {
					let mode = normalize_mode(node.argument[0])?;
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let prefix = format!("n{index}.normalize.reverse");
					let weight = (node.parameters != 0).then_some(pointers.weights.as_str());
					if mode != program_ir::NormalizeMode::Evaluation {
						let stats_prefix = format!("{prefix}.stats");
						let state_zero = native_literal(self.precision.state, self.precision.state_type, 0.0);
						ir.push_str(&program_ir::emit_normalize_reverse_stats(
							program_ir::NormalizeReverseContext {
								value_type: ty,
								pointer_type: pointer,
								alignment: alignment(ty),
								state_type: self.precision.state_type,
								state_zero: &state_zero,
								context: &pointers.context,
								rows: "%rows",
								channels: node.output.channels,
								length: node.output.length,
								width: normalize_width(node),
								span: normalize_span(node),
								weight,
								source: &pointers.source,
								mode,
								prefix: &stats_prefix,
							},
							&pointers.delta,
							&pointers.value,
						));
						ir.push_str(barrier(backend));
					}
					emit_fixed_loop(&mut ir, index, "normalize.reverse", self.rows, node.output, &window, |ir, p| {
						let delta_pointer = format!("%{prefix}.delta.ptr");
						let delta_value = format!("%{prefix}.delta.value");
						let output_pointer = format!("%{prefix}.output.ptr");
						let output_value = format!("%{prefix}.output.value");
						ir.push_str(&format!("{delta_pointer} = getelementptr inbounds {ty}, {pointer} {delta}, i32 {p}\n{delta_value} = load {ty}, {pointer} {delta_pointer}, align {align}\n{output_pointer} = getelementptr inbounds {ty}, {pointer} {value}, i32 {p}\n{output_value} = load {ty}, {pointer} {output_pointer}, align {align}\n", delta = pointers.delta, value = pointers.value, align = alignment(ty)));
						let state_zero = native_literal(self.precision.state, self.precision.state_type, 0.0);
						let fragment = program_ir::emit_normalize_reverse(
							program_ir::NormalizeReverseContext {
								value_type: ty,
								pointer_type: pointer,
								alignment: alignment(ty),
								state_type: self.precision.state_type,
								state_zero: &state_zero,
								context: &pointers.context,
								rows: "%rows",
								channels: node.output.channels,
								length: node.output.length,
								width: normalize_width(node),
								span: normalize_span(node),
								weight,
								source: &pointers.source,
								mode,
								prefix: &prefix,
							},
							p,
							&delta_value,
							&output_value,
						);
						ir.push_str(&fragment.code);
						let source_pointer = format!("%{prefix}.source.adjoint.ptr");
						ir.push_str(&format!("{source_pointer} = getelementptr inbounds {ty}, {pointer} {source_adjoint}, i32 {p}\n", source_adjoint = pointers.source_adjoint));
						ir.push_str(&accumulate_owned(&source_pointer, &fragment.contribution, ty, pointer, &format!("{prefix}.owned")));
					})?;
					ir.push_str(barrier(backend));
					if node.parameters != 0 {
						// The weight gradient is one column per channel summed over every
						// row and position: each partition accumulates its own contiguous
						// run into its own scratch row, and one owner folds the rows in order.
						let count = checked_mul(self.rows, node.output.elements(), "normalize reverse count")?;
						let partitions = count.min(NATIVE_SCALAR_PARTITIONS).max(1);
						let (columns, offset) = (narrow(node.parameters, "normalization weight columns")?, narrow(node.offset, "normalization weight offset")?);
						let statistics = narrow(checked_mul(4, normalize_groups(node, self.rows)?, "normalization statistics")?, "normalization statistics")?;
						let scratch = format!("%{prefix}.scratch");
						let zero = native_literal(self.precision.model, ty, 0.0);
						ir.push_str(&format!("{scratch} = getelementptr inbounds {ty}, {pointer} {context}, i32 {statistics}\n", context = pointers.context));
						let name = "normalize.weight";
						let row = format!("%n{index}.{name}.partition.row");
						let weight_prefix = format!("{prefix}.weight");
						emit_partitioned_loop(
							&mut ir,
							index,
							name,
							PartitionedLoop { count, partitions, columns: node.parameters, value_type: ty, pointer_type: pointer, scratch: &scratch, zero: &zero, gradients: &[] },
							|ir, p| {
								let source_pointer = format!("%{weight_prefix}.source.ptr");
								let source_value = format!("%{weight_prefix}.source.value");
								let delta_pointer = format!("%{weight_prefix}.delta.ptr");
								let delta_value = format!("%{weight_prefix}.delta.value");
								ir.push_str(&format!("{source_pointer} = getelementptr inbounds {ty}, {pointer} {source}, i32 {p}\n{source_value} = load {ty}, {pointer} {source_pointer}, align {align}\n{delta_pointer} = getelementptr inbounds {ty}, {pointer} {delta}, i32 {p}\n{delta_value} = load {ty}, {pointer} {delta_pointer}, align {align}\n", source = pointers.source, delta = pointers.delta, align = alignment(ty)));
								// The forward fragment without its weight is the normalized input.
								let fragment = program_ir::emit_normalize(
									program_ir::NormalizeContext {
										value_type: ty,
										pointer_type: pointer,
										alignment: alignment(ty),
										source_value: &source_value,
										context: &pointers.context,
										rows: "%rows",
										channels: node.output.channels,
										length: node.output.length,
										width: normalize_width(node),
										span: normalize_span(node),
										weight: None,
										mode,
										prefix: &weight_prefix,
									},
									p,
								);
								ir.push_str(&fragment.code);
								// The partitions span the row capacity; rows past `%rows` hold stale
								// values and contribute nothing, and neither does a channel outside
								// the normalized span, which owns no scale column.
								let live = if normalize_span(node) < node.output.channels {
									format!(
										"%{weight_prefix}.live.row = icmp ult i32 %{weight_prefix}.normalize.row, %rows\n%{weight_prefix}.live = and i1 %{weight_prefix}.live.row, %{weight_prefix}.normalize.inside\n"
									)
								} else {
									format!("%{weight_prefix}.live = icmp ult i32 %{weight_prefix}.normalize.row, %rows\n")
								};
								ir.push_str(&format!("%{weight_prefix}.product = call {ty} @recipe.mul({ty} {delta_value}, {ty} {normalized})\n{live}%{weight_prefix}.contribution = select i1 %{weight_prefix}.live, {ty} %{weight_prefix}.product, {ty} {zero}\n%{weight_prefix}.column.channel = select i1 %{weight_prefix}.live, i32 %{weight_prefix}.normalize.channel, i32 0\n%{weight_prefix}.column = add i32 {row}, %{weight_prefix}.column.channel\n%{weight_prefix}.column.ptr = getelementptr inbounds {ty}, {pointer} {scratch}, i32 %{weight_prefix}.column\n%{weight_prefix}.column.value = load {ty}, {pointer} %{weight_prefix}.column.ptr, align {align}\n%{weight_prefix}.column.next = call {ty} @recipe.add({ty} %{weight_prefix}.column.value, {ty} %{weight_prefix}.contribution)\nstore {ty} %{weight_prefix}.column.next, {pointer} %{weight_prefix}.column.ptr, align {align}\n", normalized = fragment.value, align = alignment(ty)));
							},
						)?;
						ir.push_str(barrier(backend));
						ir.push_str(&format!(
							"call void @reduce_rows({pointer} {scratch}, {pointer} %gradient, i32 {partitions}, i32 {columns}, i32 {columns}, i32 0, i32 {offset}, i32 %threads)\n"
						));
						ir.push_str(barrier(backend));
					}
				}
			}
		}
		Ok(ir)
	}

	// Group statistics are reductions over the batch, like the loss, so they
	// accumulate in the state format and only the finished mean and scale are
	// encoded into the model format for the context arena. Batch groups span
	// every row, and neither their item count nor their running sums fit the
	// finite range of narrow model formats.
	fn emit_normalize_stats(&self, backend: Backend, index: usize, node: &Node, pointers: &ModelPointers, mode: program_ir::NormalizeMode) -> Result<String> {
		let pointer = pointer_type(backend);
		let ty = self.precision.model_type;
		let state_ty = self.precision.state_type;
		let prefix = format!("n{index}.normalize.stats");
		let elements = i32::try_from(node.output.elements()).map_err(|_| RecipeError::new("normalization element count exceeds i32"))?;
		let length = i32::try_from(node.output.length).map_err(|_| RecipeError::new("normalization length exceeds i32"))?;
		let channels = i32::try_from(node.output.channels).map_err(|_| RecipeError::new("normalization channels exceed i32"))?;
		let mut ir = String::new();
		let model_zero = native_literal(self.precision.model, ty, 0.0);
		let zero = native_literal(self.precision.state, state_ty, 0.0);
		let one = native_literal(self.precision.state, state_ty, 1.0);
		let epsilon = native_literal(self.precision.state, state_ty, node.argument[1]);
		let groups = format!("%{prefix}.groups");
		let items = format!("%{prefix}.items");
		let width = i32::try_from(normalize_width(node)).map_err(|_| RecipeError::new("normalization width exceeds i32"))?;
		let span = i32::try_from(normalize_span(node)).map_err(|_| RecipeError::new("normalization span exceeds i32"))?;
		let heads = span / width;
		match mode {
			program_ir::NormalizeMode::Batch => {
				ir.push_str(&format!("{items} = mul i32 %rows, {length}\n", length = length));
			}
			program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms | program_ir::NormalizeMode::L2 => {
				ir.push_str(&format!("{groups} = mul i32 %rows, {}\n{items} = add i32 0, {width}\n", length * heads));
			}
			program_ir::NormalizeMode::Evaluation => return Ok(ir),
		}
		let group_limit = match mode {
			program_ir::NormalizeMode::Batch => channels.to_string(),
			program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms | program_ir::NormalizeMode::L2 => groups.clone(),
			program_ir::NormalizeMode::Evaluation => unreachable!(),
		};
		let group = format!("%{prefix}.group");
		let emit_index = |code: &mut String, phase: &str, p: &str| {
			let row = format!("%{prefix}.{phase}.row");
			let position = format!("%{prefix}.{phase}.position");
			let row_base = format!("%{prefix}.{phase}.row.base");
			let channel_base = format!("%{prefix}.{phase}.channel.base");
			let local = format!("%{prefix}.{phase}.local");
			let value_index = format!("%{prefix}.{phase}.index");
			match mode {
				program_ir::NormalizeMode::Batch => {
					code.push_str(&format!("{row} = udiv i32 {p}, {length}\n{position} = urem i32 {p}, {length}\n{row_base} = mul i32 {row}, {elements}\n{channel_base} = mul i32 {group}, {length}\n{local} = add i32 {channel_base}, {position}\n{value_index} = add i32 {row_base}, {local}\n", p = p, length = length, elements = elements, group = group));
				}
				program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms | program_ir::NormalizeMode::L2 => {
					let row_local = format!("%{prefix}.{phase}.row.local");
					let head = format!("%{prefix}.{phase}.head");
					let head_base = format!("%{prefix}.{phase}.head.base");
					let channel = format!("%{prefix}.{phase}.channel");
					code.push_str(&format!("{row} = udiv i32 {group}, {span}\n{row_local} = urem i32 {group}, {span}\n{head} = udiv i32 {row_local}, {length}\n{position} = urem i32 {row_local}, {length}\n{head_base} = mul i32 {head}, {width}\n{channel} = add i32 {head_base}, {p}\n{channel_base} = mul i32 {channel}, {length}\n{row_base} = mul i32 {row}, {elements}\n{local} = add i32 {channel_base}, {position}\n{value_index} = add i32 {row_base}, {local}\n", span = length * heads));
				}
				program_ir::NormalizeMode::Evaluation => unreachable!(),
			}
		};
		ir.push_str(&format!("br label %{prefix}.entry\n{prefix}.entry:\nbr label %{prefix}.group.loop\n{prefix}.group.loop:\n{group} = phi i32 [ %tid, %{prefix}.entry ], [ %{prefix}.group.next, %{prefix}.store ]\n%{prefix}.group.more = icmp ult i32 {group}, {group_limit}\nbr i1 %{prefix}.group.more, label %{prefix}.mean.loop, label %{prefix}.done\n{prefix}.mean.loop:\n%{prefix}.mean.p = phi i32 [ 0, %{prefix}.group.loop ], [ %{prefix}.mean.next, %{prefix}.mean.step ]\n%{prefix}.mean.sum = phi {ty} [ {zero}, %{prefix}.group.loop ], [ %{prefix}.mean.sum.next, %{prefix}.mean.step ]\n%{prefix}.mean.more = icmp ult i32 %{prefix}.mean.p, {items}\nbr i1 %{prefix}.mean.more, label %{prefix}.mean.step, label %{prefix}.variance.loop\n{prefix}.mean.step:\n", group = group, group_limit = group_limit, ty = state_ty, zero = zero, items = items));
		emit_index(&mut ir, "mean", &format!("%{prefix}.mean.p"));
		ir.push_str(&format!("%{prefix}.mean.ptr = getelementptr inbounds {ty}, {pointer} {source}, i32 %{prefix}.mean.index\n%{prefix}.mean.model = load {ty}, {pointer} %{prefix}.mean.ptr, align {align}\n%{prefix}.mean.value = call {state_ty} @recipe.state.from.model({ty} %{prefix}.mean.model)\n%{prefix}.mean.sum.next = call {state_ty} @recipe.state.add({state_ty} %{prefix}.mean.sum, {state_ty} %{prefix}.mean.value)\n%{prefix}.mean.next = add i32 %{prefix}.mean.p, 1\nbr label %{prefix}.mean.loop\n{prefix}.variance.loop:\n%{prefix}.variance.p = phi i32 [ 0, %{prefix}.mean.loop ], [ %{prefix}.variance.next, %{prefix}.variance.step ]\n%{prefix}.variance.sum = phi {state_ty} [ {zero}, %{prefix}.mean.loop ], [ %{prefix}.variance.sum.next, %{prefix}.variance.step ]\n%{prefix}.items.value = call {state_ty} @recipe.state.from.u32(i32 {items})\n%{prefix}.mean = call {state_ty} @recipe.state.div({state_ty} %{prefix}.mean.sum, {state_ty} %{prefix}.items.value)\n%{prefix}.variance.more = icmp ult i32 %{prefix}.variance.p, {items}\nbr i1 %{prefix}.variance.more, label %{prefix}.variance.step, label %{prefix}.store\n{prefix}.variance.step:\n", pointer = pointer, source = pointers.source, ty = ty, state_ty = state_ty, zero = zero, items = items, align = alignment(ty)));
		emit_index(&mut ir, "variance", &format!("%{prefix}.variance.p"));
		ir.push_str(&format!("%{prefix}.variance.ptr = getelementptr inbounds {ty}, {pointer} {source}, i32 %{prefix}.variance.index\n%{prefix}.variance.model = load {ty}, {pointer} %{prefix}.variance.ptr, align {align}\n%{prefix}.variance.value = call {state_ty} @recipe.state.from.model({ty} %{prefix}.variance.model)\n%{prefix}.variance.centered = call {state_ty} @recipe.state.sub({state_ty} %{prefix}.variance.value, {state_ty} %{prefix}.mean)\n", pointer = pointer, source = pointers.source, ty = ty, state_ty = state_ty, align = alignment(ty)));
		let zero_mean = matches!(mode, program_ir::NormalizeMode::Rms | program_ir::NormalizeMode::L2);
		let difference = if zero_mean { format!("%{prefix}.variance.value") } else { format!("%{prefix}.variance.centered") };
		// L2 divides by the norm itself, floored at epsilon, instead of the root of
		// the epsilon-shifted mean square.
		let scale_code = if mode == program_ir::NormalizeMode::L2 {
			format!(
				"%{prefix}.norm = call {state_ty} @recipe.state.sqrt({state_ty} %{prefix}.variance.sum)\n%{prefix}.floored = call i1 @recipe.state.ogt({state_ty} %{prefix}.norm, {state_ty} {epsilon})\n%{prefix}.deviation = select i1 %{prefix}.floored, {state_ty} %{prefix}.norm, {state_ty} {epsilon}\n"
			)
		} else {
			format!(
				"%{prefix}.variance = call {state_ty} @recipe.state.div({state_ty} %{prefix}.variance.sum, {state_ty} %{prefix}.items.value)\n%{prefix}.adjusted = call {state_ty} @recipe.state.add({state_ty} %{prefix}.variance, {state_ty} {epsilon})\n%{prefix}.deviation = call {state_ty} @recipe.state.sqrt({state_ty} %{prefix}.adjusted)\n"
			)
		};
		ir.push_str(&format!("%{prefix}.variance.square = call {state_ty} @recipe.state.mul({state_ty} {difference}, {state_ty} {difference})\n%{prefix}.variance.sum.next = call {state_ty} @recipe.state.add({state_ty} %{prefix}.variance.sum, {state_ty} %{prefix}.variance.square)\n%{prefix}.variance.next = add i32 %{prefix}.variance.p, 1\nbr label %{prefix}.variance.loop\n{prefix}.store:\n{scale_code}%{prefix}.scale.state = call {state_ty} @recipe.state.div({state_ty} {one}, {state_ty} %{prefix}.deviation)\n%{prefix}.mean.stored = call {ty} @recipe.model.from.state({state_ty} %{prefix}.mean)\n%{prefix}.scale = call {ty} @recipe.model.from.state({state_ty} %{prefix}.scale.state)\n%{prefix}.mean.context.ptr = getelementptr inbounds {ty}, {pointer} {context}, i32 {group}\n%{prefix}.scale.index = add i32 {group_limit}, {group}\n%{prefix}.scale.ptr = getelementptr inbounds {ty}, {pointer} {context}, i32 %{prefix}.scale.index\n", pointer = pointer, context = pointers.context, ty = ty, state_ty = state_ty, one = one, group = group, group_limit = group_limit));
		let stored_mean = if zero_mean { model_zero.clone() } else { format!("%{prefix}.mean.stored") };
		ir.push_str(&format!("store {ty} {stored_mean}, {pointer} %{prefix}.mean.context.ptr, align {align}\nstore {ty} %{prefix}.scale, {pointer} %{prefix}.scale.ptr, align {align}\n%{prefix}.group.next = add i32 {group}, %threads\nbr label %{prefix}.group.loop\n{prefix}.done:\n", pointer = pointer, ty = ty, stored_mean = stored_mean, align = alignment(ty), group = group));
		Ok(ir)
	}

	/// The window a node writes, from the window its source wrote. A convolution
	/// finishes an output position once the whole window it reads has arrived, a
	/// pool once its window is full, and every other primitive reads one input
	/// position per output position. A node therefore extends by whole positions
	/// and never rewrites a position whose inputs are already present.
	fn emit_node_window(&self, index: usize, node: &Node, ir: &mut String) -> Result<NodeWindow> {
		let prefix = format!("n{index}");
		let (begin, end) = if node.source >= 0 { (format!("%n{}.begin", node.source), format!("%n{}.end", node.source)) } else { ("%begin".to_owned(), "%end".to_owned()) };
		let length = node.output.length;
		let kernel = if node.op == Primitive::Contraction { integer_argument(node.argument[0], "contraction kernel")? } else { 0 };
		match node.op {
			Primitive::Predictor => ir.push_str(&format!("%{prefix}.begin = add i32 0, 0\n%{prefix}.end = add i32 0, {length}\n")),
			Primitive::Pool => {
				let size = integer_argument(node.argument[0], "pool size")?;
				require(size > 0, "native pool size must be positive")?;
				ir.push_str(&format!(
					"%{prefix}.begin = udiv i32 {begin}, {size}\n%{prefix}.end.filled = add i32 {end}, {last}\n%{prefix}.end.rounded = udiv i32 %{prefix}.end.filled, {size}\n%{prefix}.end.over = icmp ugt i32 %{prefix}.end.rounded, {length}\n%{prefix}.end = select i1 %{prefix}.end.over, i32 {length}, i32 %{prefix}.end.rounded\n",
					last = size - 1
				));
			}
			Primitive::Contraction if kernel > 1 => {
				for (name, source) in [("begin", &begin), ("end", &end)] {
					ir.push_str(&format!(
						"%{prefix}.{name}.shifted = sub i32 {source}, {lag}\n%{prefix}.{name}.partial = icmp ult i32 {source}, {lag}\n%{prefix}.{name} = select i1 %{prefix}.{name}.partial, i32 0, i32 %{prefix}.{name}.shifted\n",
						lag = kernel - 1
					));
				}
			}
			_ => ir.push_str(&format!("%{prefix}.begin = add i32 {begin}, 0\n%{prefix}.end = add i32 {end}, 0\n")),
		}
		ir.push_str(&format!("%{prefix}.span = sub i32 %{prefix}.end, %{prefix}.begin\n"));
		Ok(NodeWindow { begin: format!("%{prefix}.begin"), span: format!("%{prefix}.span") })
	}

	fn emit_pointers(&self, backend: Backend, index: usize, plan: &NodePlan, reverse: bool, ir: &mut String) -> Result<ModelPointers> {
		let prefix = format!("n{index}");
		let source = if plan.node.source >= 0 { format!("%{prefix}.source") } else { "%samples".to_owned() };
		if plan.node.source >= 0 {
			let source = usize::try_from(plan.node.source).map_err(|_| RecipeError::new("native source node is invalid"))?;
			ir.push_str(&ptr_gep(backend, "values", self.layout.values[source], &format!("{prefix}.source")));
		}
		let second = if plan.node.second >= 0 {
			let second = usize::try_from(plan.node.second).map_err(|_| RecipeError::new("native second node is invalid"))?;
			ir.push_str(&ptr_gep(backend, "values", self.layout.values[second], &format!("{prefix}.second")));
			format!("%{prefix}.second")
		} else {
			source.clone()
		};
		let value = format!("%{prefix}.value");
		let context = format!("%{prefix}.context");
		let delta = format!("%{prefix}.delta");
		let weights = format!("%{prefix}.weights");
		ir.push_str(&ptr_gep(backend, "values", plan.value, &format!("{prefix}.value")));
		ir.push_str(&ptr_gep(backend, "contexts", plan.context, &format!("{prefix}.context")));
		if reverse {
			ir.push_str(&ptr_gep(backend, "adjoints", plan.adjoint, &format!("{prefix}.delta")));
		}
		ir.push_str(&ptr_gep(backend, "weights", plan.weight_offset, &format!("{prefix}.weights")));
		let source_adjoint = if reverse && plan.node.source >= 0 {
			let source = usize::try_from(plan.node.source).map_err(|_| RecipeError::new("native source adjoint node is invalid"))?;
			ir.push_str(&ptr_gep(backend, "adjoints", self.layout.adjoints[source], &format!("{prefix}.source.adjoint")));
			format!("%{prefix}.source.adjoint")
		} else {
			"%input_adjoint".to_owned()
		};
		let second_adjoint = if reverse && plan.node.second >= 0 {
			let second = usize::try_from(plan.node.second).map_err(|_| RecipeError::new("native second adjoint node is invalid"))?;
			ir.push_str(&ptr_gep(backend, "adjoints", self.layout.adjoints[second], &format!("{prefix}.second.adjoint")));
			format!("%{prefix}.second.adjoint")
		} else {
			source_adjoint.clone()
		};
		Ok(ModelPointers { source, second, value, context, delta, weights, source_adjoint, second_adjoint })
	}

	fn emit_native_quantization(&self, backend: Backend, format: &'static Quantization, native: NativeDequant) -> Result<String> {
		let (pointer, ty) = (pointer_type(backend), self.precision.model_type);
		let mut operations = NativeQuantOps { globals: String::new(), ir: String::new(), backend, precision: self.precision, next: 0 };
		require(!matches!(native, NativeDequant::Nf4), "NF4 native dequantization requires its model codebook")?;
		let result = native.decode(&mut operations);
		Ok(format!(
			"{globals}define internal {ty} @recipe_model_quantized_{name}({pointer} %matrix, i32 %row, i32 %column, i32 %columns) #1 {{\nentry:\n%blocks = udiv i32 %columns, {block}\n%row.base = mul i32 %row, %blocks\n%block.local = udiv i32 %column, {block}\n%block.index = add i32 %row.base, %block.local\n%block.offset = mul i32 %block.index, {stride}\n%block = getelementptr inbounds i8, {pointer} %matrix, i32 %block.offset\n%local.i32 = urem i32 %column, {block}\n%local = zext i32 %local.i32 to i64\n{body}ret {ty} {result}\n}}\n",
			globals = operations.globals,
			name = format.name,
			block = format.block,
			stride = format.stride,
			body = operations.ir
		))
	}

	fn emit_native_nf4(&self, backend: Backend, index: usize, stored: &StoredWeight) -> Result<String> {
		let (block, table, scales) = nf4_codebook(&stored.codebook, stored.count, stored.bytes.len())?;
		let (pointer, ty) = (pointer_type(backend), self.precision.model_type);
		let name = format!("q4_nf_n{index}");
		let table_name = format!("{name}_table");
		let scales_name = format!("{name}_scales");
		let mut operations = NativeQuantOps { globals: String::new(), ir: String::new(), backend, precision: self.precision, next: 0 };
		let result = dequant_nf4(&mut operations, block, &table_name, table, &scales_name, scales);
		Ok(format!(
			"{globals}define internal {ty} @recipe_model_quantized_{name}({pointer} %matrix, i32 %row, i32 %column, i32 %columns) #1 {{\nentry:\n%block = getelementptr inbounds i8, {pointer} %matrix, i32 0\n%local = zext i32 %column to i64\n{body}ret {ty} {result}\n}}\n",
			globals = operations.globals,
			body = operations.ir
		))
	}

	fn emit_quantized_decoders(&self, backend: Backend) -> Result<String> {
		let mut emitted = String::new();
		let mut seen = Vec::new();
		let mut tables = Vec::new();
		for (index, plan) in self.plans.iter().enumerate() {
			// A lookup's table decodes on the host, so the device needs no decoder for it.
			let Some(stored) = plan.stored.as_ref().filter(|_| plan.node.op != Primitive::Lookup) else { continue };
			let spec = stored.format.spec().ok_or_else(|| RecipeError::new(format!("native quantized format {} is unavailable", stored.format.0)))?;
			let format = spec.codec.quantization();
			let native = format.native;
			if matches!(native, NativeDequant::Nf4) {
				emitted.push_str(&self.emit_native_nf4(backend, index, stored)?);
				continue;
			}
			if seen.iter().any(|codec: &StorageCodec| *codec == spec.codec) {
				continue;
			}
			if let Some(table) = native.table() {
				if !tables.contains(&table.name()) {
					emitted.push_str(&table.definition());
					tables.push(table.name());
				}
			}
			emitted.push_str(&self.emit_native_quantization(backend, format, native)?);
			seen.push(spec.codec);
		}
		Ok(emitted)
	}

	/// Selects one packed node's decoder so a consuming kernel reads its stored representation.
	fn emit_weight_decode(&self, backend: Backend) -> Result<String> {
		let (pointer, ty) = (pointer_type(backend), self.precision.model_type);
		let (mut arms, mut bodies, mut q8_0_arms) = (String::new(), String::new(), String::new());
		for (index, plan) in self.plans.iter().enumerate() {
			let Some(stored) = plan.stored.as_ref().filter(|_| plan.packed) else { continue };
			let spec = stored.format.spec().ok_or_else(|| RecipeError::new(format!("native quantized format {} is unavailable", stored.format.0)))?;
			let format = spec.codec.quantization();
			if spec.codec == StorageCodec::Q8_0 && plan.node.op == Primitive::Contraction && plan.node.input.channels % spec.block == 0 {
				q8_0_arms.push_str(&format!("i32 {}, label %q8_0.yes\n", index + 1));
			}
			let (name, block) = match format.native {
				NativeDequant::Nf4 => (format!("{}_n{index}", format.name), nf4_codebook(&stored.codebook, stored.count, stored.bytes.len())?.0),
				_ => (format.name.to_owned(), spec.block),
			};
			let columns = i32::try_from(stored.count.div_ceil(block) * block).map_err(|_| RecipeError::new("native quantized block count exceeds i32"))?;
			arms.push_str(&format!("i32 {}, label %decode.n{index}\n", index + 1));
			bodies.push_str(&format!(
				"decode.n{index}:\n%decode.n{index}.value = call {ty} @recipe_model_quantized_{name}({pointer} %matrix, i32 0, i32 %index, i32 {columns})\nret {ty} %decode.n{index}.value\n"
			));
		}
		Ok(format!(
			"define internal {ty} @recipe.model.decode({pointer} %matrix, i32 %index, i32 %node) #1 {{\nentry:\nswitch i32 %node, label %decode.absent [\n{arms}]\n{bodies}decode.absent:\nunreachable\n}}\ndefine internal i1 @recipe.model.q8_0(i32 %node) #1 {{\nentry:\nswitch i32 %node, label %q8_0.no [\n{q8_0_arms}]\nq8_0.yes:\nret i1 true\nq8_0.no:\nret i1 false\n}}\n"
		))
	}

	fn emit_model_load(&self, backend: Backend) -> Result<String> {
		if self.storage_bytes == 0 {
			return Ok(String::new());
		}
		let pointer = pointer_type(backend);
		let ty = self.precision.model_type;
		let thread = match backend {
			Backend::Cpu => "call i32 @recipe.cpu.thread.id()".to_owned(),
			Backend::Amd | Backend::Nvidia => "call i32 @global_id()".to_owned(),
		};
		let kernel = match backend {
			Backend::Cpu => "",
			Backend::Amd => "protected amdgpu_kernel ",
			Backend::Nvidia => "protected ptx_kernel ",
		};
		let mut ir = format!(
			"define {kernel}void @recipe_model_load({pointer} %weights, {pointer} %storage, i32 %threads) #0 {{\nentry:\n%tid = {thread}\n",
			kernel = kernel,
			pointer = pointer,
			thread = thread
		);
		let mut predecessor = "entry".to_owned();
		for (index, plan) in self.plans.iter().enumerate() {
			let Some(stored) = arena_weight(&plan.node, &plan.stored).filter(|_| !plan.packed) else { continue };
			let spec = stored.format.spec().ok_or_else(|| RecipeError::new(format!("native quantized format {} is unavailable", stored.format.0)))?;
			let format = spec.codec.quantization();
			let native = format.native;
			let (name, block) = match native {
				NativeDequant::Nf4 => (format!("{}_n{index}", format.name), nf4_codebook(&stored.codebook, stored.count, stored.bytes.len())?.0),
				_ => (format.name.to_owned(), spec.block),
			};
			let count = i32::try_from(stored.count).map_err(|_| RecipeError::new("native quantized weight count exceeds i32"))?;
			let columns = i32::try_from(stored.count.div_ceil(block) * block).map_err(|_| RecipeError::new("native quantized block count exceeds i32"))?;
			let prefix = format!("load.n{index}");
			ir.push_str(&format!("br label %{prefix}.loop\n{prefix}.loop:\n%{prefix}.p = phi i32 [ %tid, %entry ], [ %{prefix}.next, %{prefix}.step ]\n%{prefix}.more = icmp ult i32 %{prefix}.p, {count}\nbr i1 %{prefix}.more, label %{prefix}.step, label %{prefix}.done\n{prefix}.step:\n%{prefix}.storage = getelementptr i8, {pointer} %storage, i64 {storage}\n%{prefix}.base = getelementptr i8, {pointer} %weights, i64 {weight}\n%{prefix}.weights = getelementptr {ty}, {pointer} %{prefix}.base, i32 %{prefix}.p\n%{prefix}.value = call {ty} @recipe_model_quantized_{name}({pointer} %{prefix}.storage, i32 0, i32 %{prefix}.p, i32 {columns})\nstore {ty} %{prefix}.value, {pointer} %{prefix}.weights, align {align}\n%{prefix}.next = add i32 %{prefix}.p, %threads\nbr label %{prefix}.loop\n{prefix}.done:\n", pointer = pointer, ty = ty, count = count, storage = plan.storage_offset, weight = plan.weight_offset, name = name, columns = columns, align = alignment(ty)).replace("%entry", &format!("%{predecessor}")));
			ir.push_str(barrier(backend));
			predecessor = format!("{prefix}.done");
		}
		ir.push_str("ret void\n}\n");
		Ok(ir)
	}

	pub(crate) fn emit(&self, backend: Backend, matrix: Option<NativeMatrix>, loss: Option<LossFunction>) -> Result<String> {
		let register_count = self.schedule.register_count;
		let q8_0 = StorageCodec::Q8_0.quantization();
		let q8_0_header = q8_0.stride.checked_sub(q8_0.block).ok_or_else(|| RecipeError::new("Q8_0 storage header is invalid"))?;
		let q8_0_max = (1_u16 << (q8_0.bits - 1)) - 1;
		let mut ir = backend_template(backend, self.precision, matrix)?
			.replace("RECIPE_WORKGROUP_SIZE", &self.schedule.block.to_string())
			.replace("RECIPE_REGISTER_M", &self.schedule.register_m.to_string())
			.replace("RECIPE_REGISTER_N", &self.schedule.register_n.to_string())
			.replace("RECIPE_REGISTER_COUNT", &register_count.to_string())
			.replace("RECIPE_FRAGMENT_K", &self.schedule.fragment_k.to_string())
			.replace("RECIPE_CHUNK_K", &self.schedule.chunk_k.to_string())
			.replace("RECIPE_CHUNK_VALUES", &self.schedule.chunk_values.to_string())
			.replace("RECIPE_CHUNK_BIAS_VALUES", &self.schedule.chunk_bias_values.to_string())
			.replace("RECIPE_SCRATCH_ROW_MASK", &(NATIVE_SCRATCH_ROW_VALUES - 1).to_string())
			.replace("RECIPE_SCRATCH_ROW_CLEAR", &(-(NATIVE_SCRATCH_ROW_VALUES as i64)).to_string())
			.replace("RECIPE_GRADIENT_SCRATCH_BASE", &self.schedule.scratch_base.to_string())
			.replace("RECIPE_Q8_0_BLOCK", &q8_0.block.to_string())
			.replace("RECIPE_Q8_0_STRIDE", &q8_0.stride.to_string())
			.replace("RECIPE_Q8_0_HEADER", &q8_0_header.to_string())
			.replace("RECIPE_Q8_0_MAX", &format!("{}.0", q8_0_max));
		ir = strip_definition(ir, "recipe.model.decode");
		ir = strip_definition(ir, "recipe.model.q8_0");
		let quantized_definitions = self.emit_quantized_decoders(backend)?;
		let weight_decode = self.emit_weight_decode(backend)?;
		let model_load = self.emit_model_load(backend)?;
		ir.push_str(&quantized_definitions);
		ir.push_str(&weight_decode);
		ir.push_str(&model_load);
		let pointer = pointer_type(backend);
		let model_ty = self.precision.model_type;
		let state_precision = self.precision.state;
		let state_ty = self.precision.state_type;
		let model_align = alignment(model_ty);
		let state_align = alignment(state_ty);
		let kernel = match backend {
			Backend::Cpu => "",
			Backend::Amd => "protected amdgpu_kernel ",
			Backend::Nvidia => "protected ptx_kernel ",
		};
		let thread = match backend {
			Backend::Cpu => "call i32 @recipe.cpu.thread.id()".to_owned(),
			Backend::Amd | Backend::Nvidia => "call i32 @global_id()".to_owned(),
		};
		let inference_forward = self.emit_fixed_primitives(backend, matrix.is_some(), false, false)?;
		let mut body = String::new();
		let forward_args = format!("{pointer} %samples, {pointer} %weights, {pointer} %values, {pointer} %contexts, i32 %rows, i32 %threads, i32 %begin, i32 %end");
		body.push_str(&format!("define internal void @recipe_model_inference_forward_body({forward_args}) #1 {{\nentry:\n%tid = {thread}\n"));
		body.push_str(&inference_forward);
		body.push_str("ret void\n}\n");
		if loss.is_some() {
			let training_forward = self.emit_fixed_primitives(backend, matrix.is_some(), false, true)?;
			body.push_str(&format!("define internal void @recipe_model_training_forward_body({forward_args}) #1 {{\nentry:\n%tid = {thread}\n"));
			body.push_str(&training_forward);
			body.push_str("ret void\n}\n");
		}
		body.push_str(&format!("define {kernel}void @recipe_model_forward({forward_args}) #0 {{\nentry:\ncall void @recipe_model_inference_forward_body({forward_args})\nret void\n}}\n"));
		if let Some(loss) = loss {
			let reverse = self.emit_fixed_primitives(backend, matrix.is_some(), true, false)?;
			let gradient_bytes = checked_mul(self.graph.parameters.len(), self.precision.model.bytes(), "native gradient clear bytes")?;
			let input_bytes = checked_mul(checked_mul(self.rows, self.graph.input.elements(), "native input clear elements")?, self.precision.model.bytes(), "native input clear bytes")?;
			let epoch_args = format!(
				"{pointer} %samples, {pointer} %targets, {pointer} %weights, {pointer} %frozen, {pointer} %moments, {pointer} %variances, {pointer} %gradient, {pointer} %metrics, {pointer} %input_adjoint, {pointer} %values, {pointer} %contexts, {pointer} %adjoints, i32 %rows, i32 %threads, {state_ty} %rate, {state_ty} %beta1, {state_ty} %beta2, {state_ty} %beta1.power, {state_ty} %beta2.power, {state_ty} %epsilon, {state_ty} %decay, i32 %run.gradient, i32 %run.optimizer"
			);
			body.push_str(&format!("define {kernel}void @recipe_model_epoch({epoch_args}) #0 {{\nentry:\n%tid = {thread}\n%epoch.gradient = icmp ne i32 %run.gradient, 0\n%epoch.optimizer = icmp ne i32 %run.optimizer, 0\nbr i1 %epoch.gradient, label %gradient.entry, label %optimizer.entry\ngradient.entry:\n"));
			body.push_str(&self.emit_clear_bytes(backend, "gradient", gradient_bytes, "gradient", "gradient.entry")?);
			body.push_str(&self.emit_clear_bytes(backend, "adjoints", self.layout.adjoints_bytes, "adjoints", "clear.gradient.done")?);
			body.push_str(&self.emit_clear_bytes(backend, "input_adjoint", input_bytes, "input", "clear.adjoints.done")?);
			body.push_str(barrier(backend));
			body.push_str(&format!(
				"\ncall void @recipe_model_training_forward_body({pointer} %samples, {pointer} %weights, {pointer} %values, {pointer} %contexts, i32 %rows, i32 %threads, i32 0, i32 {positions})\n",
				positions = graph_positions(&self.graph)
			));
			body.push('\n');
			body.push_str(&self.emit_loss_and_seed(backend, loss, model_ty, state_precision, state_ty, pointer, model_align, state_align)?);
			body.push_str(barrier(backend));
			body.push_str(&reverse);
			body.push_str("br i1 %epoch.optimizer, label %optimizer.entry, label %epoch.done\n");
			body.push_str(&self.emit_adamw(model_ty, state_precision, state_ty, pointer, model_align, state_align)?);
			body.push_str("br label %epoch.done\nepoch.done:\nret void\n}\n");
		}
		ir.push_str(&body);
		Ok(prune_internal_definitions(ir))
	}

	fn emit_loss_and_seed(
		&self, backend: Backend, loss: LossFunction, model_ty: &str, state_precision: Compute, state_ty: &str, pointer: &str, model_align: usize, state_align: usize,
	) -> Result<String> {
		let output = self.graph.output.elements();
		let items = checked_mul(self.rows, output, "native loss items")?;
		let last = self.plans.last().ok_or_else(|| RecipeError::new("native model has no output node"))?;
		let prediction_offset = last.value;
		let adjoint_offset = last.adjoint;
		let mut ir = String::new();
		let zero = native_literal(state_precision, state_ty, 0.0);
		ir.push_str(&format!("%prediction.base = getelementptr i8, {pointer} %values, i64 {prediction_offset}\n%prediction = bitcast {pointer} %prediction.base to {pointer}\n%metric.ptr = getelementptr {state_ty}, {pointer} %metrics, i32 0\n%loss.leader = icmp eq i32 %tid, 0\nbr i1 %loss.leader, label %loss.entry, label %loss.wait\nloss.entry:\n"));
		ir.push_str(&format!("%loss.items = call {state_ty} @recipe.state.from.u32(i32 {items})\n"));
		if loss.0 <= 1 {
			ir.push_str(&format!("%loss.normalizer = call {state_ty} @recipe.state.sqrt({state_ty} %loss.items)\n"));
		}
		ir.push_str(&format!("br label %loss.step\nloss.step:\n%loss.p = phi i32 [ 0, %loss.entry ], [ %loss.next, %loss.item ]\n%loss.mean = phi {state_ty} [ {zero}, %loss.entry ], [ %loss.mean.next, %loss.item ]\n%loss.more = icmp ult i32 %loss.p, {items}\nbr i1 %loss.more, label %loss.item, label %loss.store\nloss.item:\n"));
		let prediction = "%loss.prediction";
		let target = "%loss.target";
		let pred_ptr = "%loss.prediction.ptr";
		let target_ptr = "%loss.target.ptr";
		ir.push_str(&format!("{pred_ptr} = getelementptr {model_ty}, {pointer} %prediction, i32 %loss.p\n%loss.prediction.model = load {model_ty}, {pointer} {pred_ptr}, align {model_align}\n{prediction} = call {state_ty} @recipe.state.from.model({model_ty} %loss.prediction.model)\n{target_ptr} = getelementptr {model_ty}, {pointer} %targets, i32 %loss.p\n%loss.target.model = load {model_ty}, {pointer} {target_ptr}, align {model_align}\n{target} = call {state_ty} @recipe.state.from.model({model_ty} %loss.target.model)\n"));
		let threshold = loss_threshold(state_precision, state_ty)?;
		let loss_value = emit_loss_value(&mut ir, loss, state_precision, state_ty, prediction, target, &threshold)?;
		let contribution = if loss.0 <= 1 {
			loss_value
		} else {
			ir.push_str(&format!("%loss.contribution = call {state_ty} @recipe.state.div({state_ty} {loss_value}, {state_ty} %loss.items)\n"));
			"%loss.contribution".to_owned()
		};
		ir.push_str(&format!(
			"%loss.mean.next = call {state_ty} @recipe.state.add({state_ty} %loss.mean, {state_ty} {contribution})\n%loss.next = add i32 %loss.p, 1\nbr label %loss.step\nloss.store:\n"
		));
		if loss.0 == 1 {
			ir.push_str(&format!("%loss.value = call {state_ty} @recipe.state.sqrt({state_ty} %loss.mean)\n"));
		} else {
			ir.push_str(&format!("%loss.value = call {state_ty} @recipe.state.add({state_ty} %loss.mean, {state_ty} {zero})\n"));
		}
		ir.push_str(&format!("store {state_ty} %loss.value, {pointer} %metric.ptr, align {state_align}\nbr label %loss.wait\nloss.wait:\n"));
		let loss_value = if loss.0 == 1 {
			ir.push_str(barrier(backend));
			ir.push_str(&format!("%loss.value.shared = load {state_ty}, {pointer} %metric.ptr, align {state_align}\n"));
			"%loss.value.shared"
		} else {
			zero.as_str()
		};
		ir.push_str(&format!("%adjoint.base = getelementptr i8, {pointer} %adjoints, i64 {adjoint_offset}\n%adjoint = bitcast {pointer} %adjoint.base to {pointer}\nbr label %seed.loop\nseed.loop:\n%seed.p = phi i32 [ %tid, %loss.wait ], [ %seed.next, %seed.step ]\n%seed.more = icmp ult i32 %seed.p, {items}\nbr i1 %seed.more, label %seed.step, label %seed.done\nseed.step:\n%seed.pred.ptr = getelementptr {model_ty}, {pointer} %prediction, i32 %seed.p\n%seed.pred.model = load {model_ty}, {pointer} %seed.pred.ptr, align {model_align}\n%seed.pred = call {state_ty} @recipe.state.from.model({model_ty} %seed.pred.model)\n%seed.target.ptr = getelementptr {model_ty}, {pointer} %targets, i32 %seed.p\n%seed.target.model = load {model_ty}, {pointer} %seed.target.ptr, align {model_align}\n%seed.target = call {state_ty} @recipe.state.from.model({model_ty} %seed.target.model)\n",));
		let gradient = emit_loss_gradient(&mut ir, loss, state_precision, state_ty, "%seed.pred", "%seed.target", &threshold, loss_value, &format!("{items}"))?;
		ir.push_str(&format!("%seed.model = call {model_ty} @recipe.model.from.state({state_ty} {gradient})\n%seed.ptr = getelementptr {model_ty}, {pointer} %adjoint, i32 %seed.p\nstore {model_ty} %seed.model, {pointer} %seed.ptr, align {model_align}\n%seed.next = add i32 %seed.p, %threads\nbr label %seed.loop\nseed.done:\n"));
		Ok(ir)
	}

	fn emit_adamw(&self, model_ty: &str, state_precision: Compute, state_ty: &str, pointer: &str, model_align: usize, state_align: usize) -> Result<String> {
		let parameters = i32::try_from(self.graph.parameters.len()).map_err(|_| RecipeError::new("native parameter count exceeds i32"))?;
		let one = native_literal(state_precision, state_ty, 1.0);
		let mut ir = String::new();
		ir.push_str(&format!("optimizer.entry:\n%optimizer.base = add i32 0, %tid\nbr label %optimizer.loop\noptimizer.loop:\n%optimizer.p = phi i32 [ %optimizer.base, %optimizer.entry ], [ %optimizer.next, %optimizer.advance ]\n%optimizer.more = icmp ult i32 %optimizer.p, {parameters}\nbr i1 %optimizer.more, label %optimizer.step, label %optimizer.done\noptimizer.step:\n"));
		ir.push_str(&format!("%optimizer.frozen.ptr = getelementptr i8, {pointer} %frozen, i32 %optimizer.p\n%optimizer.gradient.ptr = getelementptr {model_ty}, {pointer} %gradient, i32 %optimizer.p\n%optimizer.moment.ptr = getelementptr {state_ty}, {pointer} %moments, i32 %optimizer.p\n%optimizer.variance.ptr = getelementptr {state_ty}, {pointer} %variances, i32 %optimizer.p\n%optimizer.weight.ptr = getelementptr {model_ty}, {pointer} %weights, i32 %optimizer.p\n"));
		ir.push_str(&format!("%optimizer.frozen.value = load i8, {pointer} %optimizer.frozen.ptr, align 1\n%optimizer.is.frozen = icmp ne i8 %optimizer.frozen.value, 0\nbr i1 %optimizer.is.frozen, label %optimizer.advance, label %optimizer.update\noptimizer.update:\n"));
		ir.push_str(&format!("%optimizer.gradient.model = load {model_ty}, {pointer} %optimizer.gradient.ptr, align {model_align}\n%optimizer.gradient.value = call {state_ty} @recipe.state.from.model({model_ty} %optimizer.gradient.model)\n%optimizer.moment.old = load {state_ty}, {pointer} %optimizer.moment.ptr, align {state_align}\n%optimizer.variance.old = load {state_ty}, {pointer} %optimizer.variance.ptr, align {state_align}\n%optimizer.weight.model = load {model_ty}, {pointer} %optimizer.weight.ptr, align {model_align}\n%optimizer.weight.value = call {state_ty} @recipe.state.from.model({model_ty} %optimizer.weight.model)\n"));
		append_binary(&mut ir, state_ty, "optimizer.one.beta1", "sub", &one, "%beta1");
		append_binary(&mut ir, state_ty, "optimizer.one.beta2", "sub", &one, "%beta2");
		append_binary(&mut ir, state_ty, "optimizer.moment.part", "mul", "%beta1", "%optimizer.moment.old");
		append_binary(&mut ir, state_ty, "optimizer.gradient.part", "mul", "%optimizer.one.beta1", "%optimizer.gradient.value");
		append_binary(&mut ir, state_ty, "optimizer.moment.new", "add", "%optimizer.moment.part", "%optimizer.gradient.part");
		append_binary(&mut ir, state_ty, "optimizer.gradient.square", "mul", "%optimizer.gradient.value", "%optimizer.gradient.value");
		append_binary(&mut ir, state_ty, "optimizer.variance.part", "mul", "%beta2", "%optimizer.variance.old");
		append_binary(&mut ir, state_ty, "optimizer.gradient.variance", "mul", "%optimizer.one.beta2", "%optimizer.gradient.square");
		append_binary(&mut ir, state_ty, "optimizer.variance.new", "add", "%optimizer.variance.part", "%optimizer.gradient.variance");
		ir.push_str(&format!(
			"store {state_ty} %optimizer.moment.new, {pointer} %optimizer.moment.ptr, align {state_align}\nstore {state_ty} %optimizer.variance.new, {pointer} %optimizer.variance.ptr, align {state_align}\n"
		));
		append_binary(&mut ir, state_ty, "optimizer.m.correct", "sub", &one, "%beta1.power");
		append_binary(&mut ir, state_ty, "optimizer.v.correct", "sub", &one, "%beta2.power");
		append_binary(&mut ir, state_ty, "optimizer.m.hat", "div", "%optimizer.moment.new", "%optimizer.m.correct");
		append_binary(&mut ir, state_ty, "optimizer.v.hat", "div", "%optimizer.variance.new", "%optimizer.v.correct");
		ir.push_str(&format!("%optimizer.root = call {state_ty} @recipe.state.sqrt({state_ty} %optimizer.v.hat)\n"));
		append_binary(&mut ir, state_ty, "optimizer.denominator", "add", "%optimizer.root", "%epsilon");
		append_binary(&mut ir, state_ty, "optimizer.direction", "div", "%optimizer.m.hat", "%optimizer.denominator");
		append_binary(&mut ir, state_ty, "optimizer.decay", "mul", "%decay", "%optimizer.weight.value");
		append_binary(&mut ir, state_ty, "optimizer.total", "add", "%optimizer.direction", "%optimizer.decay");
		append_binary(&mut ir, state_ty, "optimizer.change", "mul", "%rate", "%optimizer.total");
		append_binary(&mut ir, state_ty, "optimizer.next.state", "sub", "%optimizer.weight.value", "%optimizer.change");
		ir.push_str(&format!("%optimizer.next.weight = call {model_ty} @recipe.model.from.state({state_ty} %optimizer.next.state)\nstore {model_ty} %optimizer.next.weight, {pointer} %optimizer.weight.ptr, align {model_align}\nbr label %optimizer.advance\noptimizer.advance:\n%optimizer.next = add i32 %optimizer.p, %threads\nbr label %optimizer.loop\noptimizer.done:\n"));
		Ok(ir)
	}
	fn emit_clear_bytes(&self, backend: Backend, base: &str, bytes: usize, label: &str, from: &str) -> Result<String> {
		let count = i64::try_from(bytes).map_err(|_| RecipeError::new(format!("native {label} clear count exceeds i64")))?;
		let pointer = pointer_type(backend);
		let prefix = format!("clear.{label}");
		let mut ir = String::new();
		ir.push_str(&format!("%{prefix}.start = zext i32 %tid to i64\n%{prefix}.stride = zext i32 %threads to i64\nbr label %{prefix}.loop\n{prefix}.loop:\n%{prefix}.p = phi i64 [ %{prefix}.start, %{from} ], [ %{prefix}.next, %{prefix}.step ]\n%{prefix}.more = icmp ult i64 %{prefix}.p, {count}\nbr i1 %{prefix}.more, label %{prefix}.step, label %{prefix}.done\n{prefix}.step:\n%{prefix}.ptr = getelementptr i8, {pointer} %{base}, i64 %{prefix}.p\nstore i8 0, {pointer} %{prefix}.ptr, align 1\n%{prefix}.next = add i64 %{prefix}.p, %{prefix}.stride\nbr label %{prefix}.loop\n{prefix}.done:\n", base = base, from = from));
		Ok(ir)
	}
}

struct ModelPointers {
	source: String,
	second: String,
	value: String,
	context: String,
	delta: String,
	weights: String,
	source_adjoint: String,
	second_adjoint: String,
}

fn type_literal(ty: &str, value: f64) -> String {
	match ty {
		"double" => format!("0x{:016X}", value.to_bits()),
		"float" => format!("0x{:016X}", f64::from(value as f32).to_bits()),
		_ if value.fract() == 0.0 => (value as i64).to_string(),
		_ => value.to_string(),
	}
}

/// The selector arguments every attention kernel takes after its tiling.
fn attention_selectors(node: &Node, precision: &NativePrecision) -> Result<String> {
	Ok(format!(
		"i32 {kv}, i32 {index_heads}, i32 {index_width}, i32 {block}, i1 {gate}, {ty} {epsilon}",
		kv = integer_argument(node.argument[1], "attention key-value heads")?,
		index_heads = integer_argument(node.argument[5], "indexer heads")?,
		index_width = integer_argument(node.argument[6], "indexer width")?,
		block = integer_argument(node.argument[3], "indexer block")?,
		gate = node.argument[2] != 0.0,
		ty = precision.model_type,
		epsilon = native_literal(precision.model, precision.model_type, node.argument[7])
	))
}
fn native_literal(precision: Compute, ty: &str, value: f64) -> String {
	match ty {
		"double" => type_literal(ty, value),
		"float" => type_literal(ty, value),
		"half" => format!("0xH{:04X}", precision.pack(value)),
		_ => precision.pack(value).to_string(),
	}
}

fn normalize_mode(value: f64) -> Result<program_ir::NormalizeMode> {
	match integer_argument(value, "normalization mode")? {
		0 => Ok(program_ir::NormalizeMode::Batch),
		1 => Ok(program_ir::NormalizeMode::Layer),
		2 => Ok(program_ir::NormalizeMode::Rms),
		3 => Ok(program_ir::NormalizeMode::Evaluation),
		4 => Ok(program_ir::NormalizeMode::L2),
		_ => Err(RecipeError::new("normalization mode is unsupported")),
	}
}

/// Channels per normalization group: the node's declared width, or the whole row.
fn normalize_width(node: &Node) -> usize {
	if node.argument[2] == 0.0 { node.output.channels } else { node.argument[2] as usize }
}

/// Channels the node normalizes: its declared span, or the whole row. Attention
/// normalizes the leading query and key planes and passes the value plane through.
fn normalize_span(node: &Node) -> usize {
	if node.argument[3] == 0.0 { node.output.channels } else { node.argument[3] as usize }
}

/// The statistics arena holds four values per group for either group shape.
fn normalize_groups(node: &Node, rows: usize) -> Result<usize> {
	let heads = normalize_span(node) / normalize_width(node);
	match normalize_mode(node.argument[0])? {
		program_ir::NormalizeMode::Batch | program_ir::NormalizeMode::Evaluation => Ok(node.output.channels),
		program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms | program_ir::NormalizeMode::L2 => {
			checked_mul(checked_mul(rows, node.output.length, "row groups")?, heads, "head groups")
		}
	}
}

fn alignment(ty: &str) -> usize {
	match ty {
		"double" => 8,
		"float" | "i32" => 4,
		"i16" => 2,
		_ => 1,
	}
}

fn loss_threshold(precision: Compute, ty: &str) -> Result<String> {
	let value = env!("RECIPE_HUBER_THRESHOLD").parse::<f64>().map_err(|error| RecipeError::new(format!("invalid Huber threshold: {error}")))?;
	Ok(native_literal(precision, ty, value))
}

fn append_binary(ir: &mut String, ty: &str, name: &str, operation: &str, left: &str, right: &str) {
	ir.push_str(&format!("%{name} = call {ty} @recipe.state.{operation}({ty} {left}, {ty} {right})\n"));
}

fn emit_loss_value(ir: &mut String, loss: LossFunction, precision: Compute, ty: &str, prediction: &str, target: &str, threshold: &str) -> Result<String> {
	let literal = |value: f64| native_literal(precision, ty, value);
	let one = literal(1.0);
	append_binary(ir, ty, "loss.difference", "sub", prediction, target);
	match loss.0 {
		0 | 1 => {
			append_binary(ir, ty, "loss.scaled", "div", "%loss.difference", "%loss.normalizer");
			append_binary(ir, ty, "loss.square", "mul", "%loss.scaled", "%loss.scaled");
			Ok("%loss.square".to_owned())
		}
		2 => {
			append_binary(ir, ty, "loss.square", "mul", "%loss.difference", "%loss.difference");
			ir.push_str(&format!(
				"%loss.absolute = call {ty} @recipe.state.abs({ty} %loss.difference)\n%loss.small = call i1 @recipe.state.ole({ty} %loss.absolute, {ty} {threshold})\n",
				ty = ty
			));
			append_binary(ir, ty, "loss.half.square", "mul", "%loss.square", &literal(0.5));
			append_binary(ir, ty, "loss.half.threshold", "mul", threshold, &literal(0.5));
			append_binary(ir, ty, "loss.large.base", "sub", "%loss.absolute", "%loss.half.threshold");
			append_binary(ir, ty, "loss.large", "mul", threshold, "%loss.large.base");
			ir.push_str(&format!("%loss.huber = select i1 %loss.small, {ty} %loss.half.square, {ty} %loss.large\n", ty = ty));
			Ok("%loss.huber".to_owned())
		}
		3 => {
			ir.push_str(&format!("%loss.mae = call {ty} @recipe.state.abs({ty} %loss.difference)\n", ty = ty));
			Ok("%loss.mae".to_owned())
		}
		4 => {
			ir.push_str(&format!("%loss.probability.raw = call {ty} @recipe.state.sigmoid({ty} {prediction})\n%loss.probability.low = call i1 @recipe.state.olt({ty} %loss.probability.raw, {ty} {tiny})\n%loss.probability.floor = select i1 %loss.probability.low, {ty} {tiny}, {ty} %loss.probability.raw\n%loss.probability.high = call i1 @recipe.state.ogt({ty} %loss.probability.floor, {ty} {one_minus})\n%loss.probability = select i1 %loss.probability.high, {ty} {one_minus}, {ty} %loss.probability.floor\n%loss.log.probability = call {ty} @recipe.state.log({ty} %loss.probability)\n%loss.one.probability = call {ty} @recipe.state.sub({ty} {one}, {ty} %loss.probability)\n%loss.log.one.probability = call {ty} @recipe.state.log({ty} %loss.one.probability)\n%loss.first = call {ty} @recipe.state.mul({ty} {target}, {ty} %loss.log.probability)\n%loss.one.target = call {ty} @recipe.state.sub({ty} {one}, {ty} {target})\n%loss.second = call {ty} @recipe.state.mul({ty} %loss.one.target, {ty} %loss.log.one.probability)\n%loss.cross.sum = call {ty} @recipe.state.add({ty} %loss.first, {ty} %loss.second)\n%loss.cross = call {ty} @recipe.state.neg({ty} %loss.cross.sum)\n", ty = ty, tiny = literal(f64::EPSILON), one_minus = literal(precision.below_one(1.0 - f64::EPSILON)), target = target, one = one));
			Ok("%loss.cross".to_owned())
		}
		6 => {
			ir.push_str(&format!("%loss.probability = call {ty} @recipe.state.sigmoid({ty} {prediction})\n%loss.target.class = call i1 @recipe.state.oge({ty} {target}, {ty} {half})\n%loss.one.probability = call {ty} @recipe.state.sub({ty} {one}, {ty} %loss.probability)\n%loss.correct.raw = select i1 %loss.target.class, {ty} %loss.probability, {ty} %loss.one.probability\n%loss.correct.low = call i1 @recipe.state.olt({ty} %loss.correct.raw, {ty} {tiny})\n%loss.correct = select i1 %loss.correct.low, {ty} {tiny}, {ty} %loss.correct.raw\n%loss.incorrect = call {ty} @recipe.state.sub({ty} {one}, {ty} %loss.correct)\n%loss.incorrect.square = call {ty} @recipe.state.mul({ty} %loss.incorrect, {ty} %loss.incorrect)\n%loss.log.correct = call {ty} @recipe.state.log({ty} %loss.correct)\n%loss.focal.product = call {ty} @recipe.state.mul({ty} %loss.incorrect.square, {ty} %loss.log.correct)\n%loss.focal = call {ty} @recipe.state.neg({ty} %loss.focal.product)\n", ty = ty, target = target, one = one, half = literal(0.5), tiny = literal(f64::EPSILON)));
			Ok("%loss.focal".to_owned())
		}
		_ => Err(RecipeError::new(format!("native loss {} is unsupported", loss.0))),
	}
}

fn emit_loss_gradient(ir: &mut String, loss: LossFunction, precision: Compute, ty: &str, prediction: &str, target: &str, threshold: &str, loss_value: &str, rows: &str) -> Result<String> {
	let literal = |value: f64| native_literal(precision, ty, value);
	let zero = literal(0.0);
	let one = literal(1.0);
	let negative_one = literal(-1.0);
	let two = literal(2.0);
	let tiny = literal(f64::EPSILON);
	let half = literal(0.5);
	append_binary(ir, ty, "seed.difference", "sub", prediction, target);
	let rows_value = "%seed.rows";
	ir.push_str(&format!("{rows_value} = call {ty} @recipe.state.from.u32(i32 {rows})\n", rows_value = rows_value, ty = ty, rows = rows));
	match loss.0 {
		0 => {
			append_binary(ir, ty, "seed.twice", "add", "%seed.difference", "%seed.difference");
			append_binary(ir, ty, "seed.mse", "div", "%seed.twice", rows_value);
			Ok("%seed.mse".to_owned())
		}
		1 => {
			append_binary(ir, ty, "seed.rmse.denominator", "mul", rows_value, loss_value);
			ir.push_str(&format!("%seed.rmse.zero = call i1 @recipe.state.oeq({ty} {loss_value}, {ty} {zero})\n", ty = ty, loss_value = loss_value, zero = zero));
			append_binary(ir, ty, "seed.rmse.divided", "div", "%seed.difference", "%seed.rmse.denominator");
			ir.push_str(&format!("%seed.rmse = select i1 %seed.rmse.zero, {ty} {zero}, {ty} %seed.rmse.divided\n", ty = ty, zero = zero));
			Ok("%seed.rmse".to_owned())
		}
		2 => {
			ir.push_str(&format!("%seed.huber.negative.threshold = call {ty} @recipe.state.neg({ty} {threshold})\n%seed.huber.low = call i1 @recipe.state.olt({ty} %seed.difference, {ty} %seed.huber.negative.threshold)\n%seed.huber.high = call i1 @recipe.state.ogt({ty} %seed.difference, {ty} {threshold})\n%seed.huber.lower = select i1 %seed.huber.low, {ty} %seed.huber.negative.threshold, {ty} %seed.difference\n%seed.huber.clamped = select i1 %seed.huber.high, {ty} {threshold}, {ty} %seed.huber.lower\n", ty = ty, threshold = threshold));
			append_binary(ir, ty, "seed.huber", "div", "%seed.huber.clamped", rows_value);
			Ok("%seed.huber".to_owned())
		}
		3 => {
			ir.push_str(&format!("%seed.mae.negative = call i1 @recipe.state.olt({ty} %seed.difference, {ty} {zero})\n%seed.mae.positive = call i1 @recipe.state.ogt({ty} %seed.difference, {ty} {zero})\n%seed.mae.upper = select i1 %seed.mae.positive, {ty} {one}, {ty} {zero}\n%seed.mae.sign = select i1 %seed.mae.negative, {ty} {negative_one}, {ty} %seed.mae.upper\n", ty = ty, zero = zero, one = one, negative_one = negative_one));
			append_binary(ir, ty, "seed.mae", "div", "%seed.mae.sign", rows_value);
			Ok("%seed.mae".to_owned())
		}
		4 => {
			ir.push_str(&format!("%seed.probability = call {ty} @recipe.state.sigmoid({ty} {prediction})\n", ty = ty, prediction = prediction));
			append_binary(ir, ty, "seed.cross.difference", "sub", "%seed.probability", target);
			append_binary(ir, ty, "seed.cross", "div", "%seed.cross.difference", rows_value);
			Ok("%seed.cross".to_owned())
		}
		6 => {
			ir.push_str(&format!("%seed.probability = call {ty} @recipe.state.sigmoid({ty} {prediction})\n%seed.target.class = call i1 @recipe.state.oge({ty} {target}, {ty} {half})\n%seed.one.probability = call {ty} @recipe.state.sub({ty} {one}, {ty} %seed.probability)\n%seed.correct.raw = select i1 %seed.target.class, {ty} %seed.probability, {ty} %seed.one.probability\n%seed.correct.low = call i1 @recipe.state.olt({ty} %seed.correct.raw, {ty} {tiny})\n%seed.correct = select i1 %seed.correct.low, {ty} {tiny}, {ty} %seed.correct.raw\n%seed.incorrect = call {ty} @recipe.state.sub({ty} {one}, {ty} %seed.correct)\n%seed.log.correct = call {ty} @recipe.state.log({ty} %seed.correct)\n", ty = ty, prediction = prediction, target = target, half = half, one = one, tiny = tiny));
			append_binary(ir, ty, "seed.focal.first", "mul", &two, "%seed.incorrect");
			append_binary(ir, ty, "seed.focal.first.value", "mul", "%seed.focal.first", "%seed.log.correct");
			append_binary(ir, ty, "seed.focal.square", "mul", "%seed.incorrect", "%seed.incorrect");
			append_binary(ir, ty, "seed.focal.second", "div", "%seed.focal.square", "%seed.correct");
			append_binary(ir, ty, "seed.focal.by.correct", "sub", "%seed.focal.first.value", "%seed.focal.second");
			append_binary(ir, ty, "seed.focal.sigmoid.derivative", "mul", "%seed.probability", "%seed.one.probability");
			ir.push_str(&format!("%seed.focal.negative.direction = call {ty} @recipe.state.neg({ty} %seed.focal.sigmoid.derivative)\n%seed.focal.direction = select i1 %seed.target.class, {ty} %seed.focal.sigmoid.derivative, {ty} %seed.focal.negative.direction\n", ty = ty));
			append_binary(ir, ty, "seed.focal.chain", "mul", "%seed.focal.by.correct", "%seed.focal.direction");
			append_binary(ir, ty, "seed.focal", "div", "%seed.focal.chain", rows_value);
			Ok("%seed.focal".to_owned())
		}
		_ => Err(RecipeError::new(format!("native loss {} is unsupported", loss.0))),
	}
}

fn token_id(value: f64, vocabulary: f64) -> Result<i32> {
	require(value.fract() == 0.0 && value >= 0.0 && value < vocabulary, format!("token id {value} is outside the vocabulary of {vocabulary}"))?;
	Ok(value as i32)
}

/// The positions the first node consumes. A gather reads one token id per input
/// element; every other opening block reads the input sequence.
fn graph_positions(graph: &Graph) -> usize {
	if graph.nodes.first().is_some_and(|node| node.op == Primitive::Gather) { graph.input.elements() } else { graph.input.length }
}

/// The output positions one forward writes, named in the emitted IR.
struct NodeWindow {
	begin: String,
	span: String,
}

fn integer_argument(value: f64, role: &str) -> Result<i32> {
	require(value.is_finite() && value.fract() == 0.0 && value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX), format!("native {role} is not an integer"))?;
	Ok(value as i32)
}

/// Accumulate into an adjoint element with exactly one writing lane. Every
/// caller must own the destination element for the current barrier interval.
fn accumulate_owned(target: &str, value: &str, ty: &str, pointer: &str, prefix: &str) -> String {
	format!(
		"%{prefix}.prior = load {ty}, {pointer} {target}, align {align}\n%{prefix}.sum = call {ty} @recipe.add({ty} %{prefix}.prior, {ty} {value})\nstore {ty} %{prefix}.sum, {pointer} {target}, align {align}\n",
		align = alignment(ty)
	)
}

/// Contiguous pieces an elementwise reduction over a shared destination is cut
/// into. The count is `min(elements, this)`, so it follows the shape of the work
/// and never the width of the launch.
const NATIVE_SCALAR_PARTITIONS: usize = 4096;

struct PartitionedLoop<'a> {
	count: usize,
	partitions: usize,
	columns: usize,
	value_type: &'a str,
	pointer_type: &'a str,
	scratch: &'a str,
	zero: &'a str,
	gradients: &'a [(usize, String)],
}

/// Walk `count` elements as `partitions` contiguous runs, each summed in
/// ascending element order into its own scratch row. Partition `t` spans
/// `[t * q + min(t, r), (t + 1) * q + min(t + 1, r))` for the quotient `q` and
/// remainder `r` of the element count over the partition count, so both the
/// boundaries and the number of rows are fixed by the program.
fn emit_partitioned_loop(ir: &mut String, index: usize, name: &str, shape: PartitionedLoop<'_>, mut body: impl FnMut(&mut String, &str)) -> Result<()> {
	// The body owns the `n{index}.{name}` namespace, so every value this function
	// introduces sits under a suffix of its own.
	let prefix = format!("n{index}.{name}.partition");
	let PartitionedLoop { count, partitions, columns, value_type: ty, pointer_type: pointer, scratch, zero, gradients } = shape;
	require(partitions != 0 && columns != 0, "native partitioned loop is empty")?;
	require(gradients.iter().all(|(parameter, _)| *parameter < columns), "native partitioned loop parameter is out of range")?;
	let (whole, extra) = (narrow(count / partitions, "native partition span")?, narrow(count % partitions, "native partition remainder")?);
	let partitions = narrow(partitions, "native partition count")?;
	let columns = narrow(columns, "native partition columns")?;
	let align = alignment(ty);
	ir.push_str(&format!("br label %{prefix}.entry\n{prefix}.entry:\nbr label %{prefix}.loop\n{prefix}.loop:\n%{prefix}.t = phi i32 [ %tid, %{prefix}.entry ], [ %{prefix}.advance, %{prefix}.step ]\n%{prefix}.more = icmp ult i32 %{prefix}.t, {partitions}\nbr i1 %{prefix}.more, label %{prefix}.body, label %{prefix}.done\n{prefix}.body:\n"));
	ir.push_str(&format!("%{prefix}.t.plus = add i32 %{prefix}.t, 1\n%{prefix}.first.short = icmp ult i32 %{prefix}.t, {extra}\n%{prefix}.first.extra = select i1 %{prefix}.first.short, i32 %{prefix}.t, i32 {extra}\n%{prefix}.first.whole = mul i32 %{prefix}.t, {whole}\n%{prefix}.first = add i32 %{prefix}.first.whole, %{prefix}.first.extra\n"));
	ir.push_str(&format!("%{prefix}.limit.short = icmp ult i32 %{prefix}.t.plus, {extra}\n%{prefix}.limit.extra = select i1 %{prefix}.limit.short, i32 %{prefix}.t.plus, i32 {extra}\n%{prefix}.limit.whole = mul i32 %{prefix}.t.plus, {whole}\n%{prefix}.limit = add i32 %{prefix}.limit.whole, %{prefix}.limit.extra\n"));
	ir.push_str(&format!("%{prefix}.row = mul i32 %{prefix}.t, {columns}\n"));
	// A body with no fixed sums accumulates into its partition's scratch row at
	// `%{prefix}.row`, so the row starts at zero and keeps what the body left.
	let entry = if gradients.is_empty() {
		ir.push_str(&format!("br label %{prefix}.zero\n{prefix}.zero:\n%{prefix}.zero.c = phi i32 [ 0, %{prefix}.body ], [ %{prefix}.zero.next, %{prefix}.zero.step ]\n%{prefix}.zero.more = icmp ult i32 %{prefix}.zero.c, {columns}\nbr i1 %{prefix}.zero.more, label %{prefix}.zero.step, label %{prefix}.zeroed\n{prefix}.zero.step:\n%{prefix}.zero.index = add i32 %{prefix}.row, %{prefix}.zero.c\n%{prefix}.zero.ptr = getelementptr inbounds {ty}, {pointer} {scratch}, i32 %{prefix}.zero.index\nstore {ty} {zero}, {pointer} %{prefix}.zero.ptr, align {align}\n%{prefix}.zero.next = add i32 %{prefix}.zero.c, 1\nbr label %{prefix}.zero\n{prefix}.zeroed:\n"));
		"zeroed"
	} else {
		"body"
	};
	ir.push_str(&format!("br label %{prefix}.inner\n{prefix}.inner:\n%{prefix}.p = phi i32 [ %{prefix}.first, %{prefix}.{entry} ], [ %{prefix}.p.next, %{prefix}.fold ]\n"));
	for (parameter, _) in gradients {
		ir.push_str(&format!("%{prefix}.sum.{parameter} = phi {ty} [ {zero}, %{prefix}.{entry} ], [ %{prefix}.sum.{parameter}.next, %{prefix}.fold ]\n"));
	}
	ir.push_str(&format!("%{prefix}.inner.more = icmp ult i32 %{prefix}.p, %{prefix}.limit\nbr i1 %{prefix}.inner.more, label %{prefix}.inner.body, label %{prefix}.store\n{prefix}.inner.body:\n"));
	body(ir, &format!("%{prefix}.p"));
	ir.push_str(&format!("br label %{prefix}.fold\n{prefix}.fold:\n"));
	for (parameter, value) in gradients {
		ir.push_str(&format!("%{prefix}.sum.{parameter}.next = call {ty} @recipe.add({ty} %{prefix}.sum.{parameter}, {ty} {value})\n"));
	}
	ir.push_str(&format!("%{prefix}.p.next = add i32 %{prefix}.p, 1\nbr label %{prefix}.inner\n{prefix}.store:\n"));
	// Every column of the row is written, including the parameters this program
	// never touches, so the fold below never reads an uninitialised slot.
	for column in 0..if gradients.is_empty() { 0 } else { columns } {
		let stored = gradients.iter().find(|(parameter, _)| *parameter as i32 == column).map_or_else(|| zero.to_owned(), |(parameter, _)| format!("%{prefix}.sum.{parameter}"));
		ir.push_str(&format!("%{prefix}.index.{column} = add i32 %{prefix}.row, {column}\n%{prefix}.column.{column} = getelementptr inbounds {ty}, {pointer} {scratch}, i32 %{prefix}.index.{column}\nstore {ty} {stored}, {pointer} %{prefix}.column.{column}, align {align}\n"));
	}
	ir.push_str(&format!("br label %{prefix}.step\n{prefix}.step:\n%{prefix}.advance = add i32 %{prefix}.t, %threads\nbr label %{prefix}.loop\n{prefix}.done:\n"));
	Ok(())
}

/// Walk the elements of one window of output positions. The positions of a
/// channel are contiguous, so the window is a run per row and channel and the
/// loop index maps onto the element it owns.
fn emit_fixed_loop(ir: &mut String, index: usize, name: &str, rows: usize, shape: Shape, window: &NodeWindow, mut body: impl FnMut(&mut String, &str)) -> Result<()> {
	let prefix = format!("n{index}.{name}");
	let elements = narrow(checked_mul(shape.channels, shape.length, format!("native {name} row elements").as_str())?, "native loop row elements")?;
	let (channels, length) = (narrow(shape.channels, "native loop channels")?, narrow(shape.length, "native loop length")?);
	let rows = narrow(rows, "native loop rows")?;
	let (begin, span) = (&window.begin, &window.span);
	ir.push_str(&format!(
		"%{prefix}.at.plane = mul i32 {channels}, {span}\n%{prefix}.at.count = mul i32 {rows}, %{prefix}.at.plane\nbr label %{prefix}.entry\n{prefix}.entry:\nbr label %{prefix}.loop\n{prefix}.loop:\n%{prefix}.at.q = phi i32 [ %tid, %{prefix}.entry ], [ %{prefix}.at.next, %{prefix}.step ]\n%{prefix}.at.more = icmp ult i32 %{prefix}.at.q, %{prefix}.at.count\nbr i1 %{prefix}.at.more, label %{prefix}.body, label %{prefix}.done\n{prefix}.body:\n%{prefix}.at.row = udiv i32 %{prefix}.at.q, %{prefix}.at.plane\n%{prefix}.at.within = urem i32 %{prefix}.at.q, %{prefix}.at.plane\n%{prefix}.at.channel = udiv i32 %{prefix}.at.within, {span}\n%{prefix}.at.offset = urem i32 %{prefix}.at.within, {span}\n%{prefix}.at.position = add i32 %{prefix}.at.offset, {begin}\n%{prefix}.at.row.base = mul i32 %{prefix}.at.row, {elements}\n%{prefix}.at.channel.base = mul i32 %{prefix}.at.channel, {length}\n%{prefix}.at.local = add i32 %{prefix}.at.channel.base, %{prefix}.at.position\n%{prefix}.at.p = add i32 %{prefix}.at.row.base, %{prefix}.at.local\n"
	));
	body(ir, &format!("%{prefix}.at.p"));
	ir.push_str(&format!("br label %{prefix}.step\n{prefix}.step:\n%{prefix}.at.next = add i32 %{prefix}.at.q, %threads\nbr label %{prefix}.loop\n{prefix}.done:\n"));
	Ok(())
}

/// The delta rule arguments both directions share, and the context offset of the
/// per-pair decay partials that follow every other region.
struct DeltaShape {
	heads: i32,
	key_heads: i32,
	partials: i32,
	arguments: String,
}

/// A delta node's key and value extents. The queries and keys span the key
/// heads, the values and the recurrence output the value heads, and one thread
/// owns one row and value head, so its state is `key width` by `value width`.
fn delta_extent(node: &Node) -> Result<(i32, i32, i32, i32)> {
	let (heads, width) = (integer_argument(node.argument[0], "delta heads")?, integer_argument(node.argument[1], "delta width")?);
	let key_heads = integer_argument(node.argument[3], "delta key heads")?;
	let key_width = integer_argument(node.argument[4], "delta key width")?;
	Ok((if key_heads == 0 { heads } else { key_heads }, if key_width == 0 { width } else { key_width }, heads, width))
}

fn delta_shape(node: &Node, rows: usize) -> Result<DeltaShape> {
	let (key_heads, key_width, heads, width) = delta_extent(node)?;
	let chunk = integer_argument(node.argument[2], "delta chunk")?;
	let (pairs, state) = (checked_mul(rows, heads as usize, "delta pairs")?, checked_mul(key_width as usize, width as usize, "delta state")?);
	let chunks = node.output.length.div_ceil(chunk as usize);
	let spans = checked_add(chunks, checked_add(chunk as usize, 2, "delta live states")?, "delta state spans")?;
	let partials = narrow(
		checked_mul(pairs, checked_add(checked_mul(spans, state, "delta state span")?, checked_mul(2, width as usize, "delta vectors")?, "delta pair span")?, "delta partials")?,
		"delta partials",
	)?;
	let (length, count, blocks) = (narrow(node.output.length, "delta length")?, narrow(pairs, "delta pairs")?, narrow(chunks, "delta chunks")?);
	Ok(DeltaShape { heads, key_heads, partials, arguments: format!("i32 {key_heads}, i32 {key_width}, i32 {heads}, i32 {width}, i32 {length}, i32 {chunk}, i32 {blocks}, i32 {count}") })
}

/// List the positions routed to each expert, in ascending expert and position
/// order. A weight gradient walks one expert's list instead of every position.
fn emit_expert_buckets(ir: &mut String, backend: Backend, index: usize, rows: usize, node: &Node, pointers: &ModelPointers) -> Result<()> {
	let pairs = checked_mul(rows, node.output.length, "routed positions")?;
	// One list per expert: an `[experts, 1]` shape walked whole.
	let buckets = Shape { channels: integer_argument(node.argument[0], "routed experts")? as usize, length: 1 };
	let whole = NodeWindow { begin: "0".to_owned(), span: "1".to_owned() };
	emit_fixed_loop(ir, index, "expert.bucket", 1, buckets, &whole, |ir, p| {
		ir.push_str(&format!(
			"call void @moe_bucket_body( {pointer} {routing}, {pointer} {context}, i32 {p}, i32 {pairs}, i32 {length}, i32 {experts}, i32 {top} )\n",
			pointer = pointer_type(backend),
			routing = pointers.second,
			context = pointers.context,
			length = node.output.length,
			experts = node.argument[0],
			top = node.argument[1]
		));
	})?;
	ir.push_str(barrier(backend));
	Ok(())
}

static NATIVE_ARTIFACT_SERIAL: AtomicUsize = AtomicUsize::new(0);

struct NativeTemporaryFiles {
	paths: Vec<PathBuf>,
}

impl Drop for NativeTemporaryFiles {
	fn drop(&mut self) {
		for path in &self.paths {
			let _ = fs::remove_file(path);
		}
	}
}

fn native_artifact_directory(key: &str) -> Result<PathBuf> {
	require(!key.is_empty() && key != "." && key != ".." && !key.contains('/') && !key.contains('\\'), "native artifact key is not a single path component")?;
	let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).ok_or_else(|| RecipeError::new("home directory is absent"))?;
	Ok(PathBuf::from(home).join(".cache").join("recipe").join("native").join(key))
}

fn native_artifact_key(target: &BackendTarget, ir: &str) -> String {
	let mut hash = 14695981039346656037_u64;
	for part in [b"recipe-native-v2".as_slice(), native_target_label(target).as_bytes(), ir.as_bytes()] {
		for byte in (part.len() as u64).to_le_bytes().into_iter().chain(part.iter().copied()) {
			hash = (hash ^ u64::from(byte)).wrapping_mul(1099511628211)
		}
	}
	format!("recipe-native-{hash:016x}")
}

/// Run a compiler and return whatever it wrote to its diagnostic stream, which
/// is where the resource-usage remarks below arrive.
fn native_command(mut command: Command, role: &str, key: &str) -> Result<String> {
	debug(&format!("native compiler key={key} role={role} command={command:?}"))?;
	let output = command.output().map_err(|error| RecipeError::new(format!("cannot start {role}: {error}")))?;
	let diagnostic = String::from_utf8_lossy(&output.stderr).trim().to_owned();
	if output.status.success() {
		return Ok(diagnostic);
	}
	Err(RecipeError::new(format!("{role} failed: {diagnostic}")))
}

/// What the AMD backend reports about a compiled kernel. `occupancy` is the
/// number of waves the register allocation leaves resident per SIMD, so zero
/// means the kernel cannot be resident at the requested workgroup size and a
/// cooperative grid built from it would never complete.
#[derive(Clone, Debug)]
struct KernelResources {
	name: String,
	registers: u32,
	scalars: u32,
	occupancy: u32,
}

/// Parse `-Rpass-analysis=kernel-resource-usage` remarks. The remark text is a
/// compiler courtesy rather than a stable interface, so an unrecognised or
/// absent report yields no entries and simply leaves the check unexercised.
fn kernel_resources(diagnostic: &str) -> Vec<KernelResources> {
	// A remark reads "remark: <file>:<line>:<column>:     <label>: <value> [-Rpass...]",
	// so the value follows the last colon and the label precedes it.
	let mut found: Vec<KernelResources> = Vec::new();
	for line in diagnostic.lines().filter(|line| line.contains("kernel-resource-usage")) {
		let body = line.rsplit_once(" [-").map_or(line, |(head, _)| head);
		let Some((head, value)) = body.rsplit_once(':') else { continue };
		let (label, value) = (head.rsplit(':').next().unwrap_or_default().trim(), value.trim());
		match label {
			"Function Name" => found.push(KernelResources { name: value.to_owned(), registers: 0, scalars: 0, occupancy: 0 }),
			"VGPRs" => {
				if let Some(entry) = found.last_mut() {
					entry.registers = value.parse().unwrap_or(0)
				}
			}
			"TotalSGPRs" => {
				if let Some(entry) = found.last_mut() {
					entry.scalars = value.parse().unwrap_or(0)
				}
			}
			"Occupancy [waves/SIMD]" => {
				if let Some(entry) = found.last_mut() {
					entry.occupancy = value.parse().unwrap_or(0)
				}
			}
			_ => {}
		}
	}
	found
}

fn native_cpu_compiler() -> Result<&'static str> {
	option_env!("RECIPE_CPU_COMPILER").ok_or_else(|| RecipeError::new("CPU native compiler is unavailable"))
}

fn native_amd_compiler() -> Result<&'static str> {
	option_env!("RECIPE_HSA_COMPILER").ok_or_else(|| RecipeError::new("AMD native compiler is unavailable"))
}

fn native_nvidia_compiler() -> Result<&'static str> {
	option_env!("RECIPE_NV_COMPILER").ok_or_else(|| RecipeError::new("NVIDIA native compiler is unavailable"))
}

fn native_amd_library(name: &'static str) -> Result<&'static str> {
	option_env!("RECIPE_HSA_DEVICE_LIBRARY")
		.filter(|_| name == "device")
		.or(option_env!("RECIPE_HSA_CLOCK_LIBRARY").filter(|_| name == "clock"))
		.or(option_env!("RECIPE_HSA_ABI_LIBRARY").filter(|_| name == "abi"))
		.or(option_env!("RECIPE_HSA_FINITE_LIBRARY").filter(|_| name == "finite"))
		.or(option_env!("RECIPE_HSA_MATH_LIBRARY").filter(|_| name == "math"))
		.ok_or_else(|| RecipeError::new(format!("AMD native {name} library is unavailable")))
}

fn compile_native_artifact(target: &BackendTarget, source: &Path, output: &Path, bitcode: Option<&Path>, key: &str) -> Result<Vec<KernelResources>> {
	match target {
		BackendTarget::Cpu { target } => {
			let compiler = native_cpu_compiler()?;
			let (target, compiler_identity, _, _) = cpu_identity(target)?;
			let mut command = Command::new(compiler);
			command.args(["-target", target, "-march=native"]);
			if cpu_llvm_major(compiler_identity)? < LLVM_OPAQUE_POINTER_DEFAULT_MAJOR {
				command.args(["-mllvm", "-opaque-pointers=1"]);
			}
			if compiler_identity.contains(APPLE_CLANG_BROKEN_LICM_PROMOTION_PREFIX) {
				command.args(["-mllvm", "-disable-licm-promotion"]);
			}
			command.args(["-x", "ir", "-O2", "-fPIC", "-shared", "-o"]).arg(output).arg(source);
			native_command(command, "CPU LLVM IR compiler", key).map(|_| Vec::new())
		}
		BackendTarget::Amd { architecture } => {
			let compiler = native_amd_compiler()?;
			let mut command = Command::new(compiler);
			// The resource-usage remarks are the only route to the register
			// allocation of the compiled kernel, which decides whether the requested
			// workgroup can be resident at all.
			command.args(["-target", "amdgcn-amd-amdhsa"]).arg(format!("-mcpu={architecture}")).args(["-O3", "-nogpulib", "-Rpass-analysis=kernel-resource-usage"]);
			for name in ["device", "clock", "abi", "finite", "math"] {
				command.args(["-Xclang", "-mlink-builtin-bitcode", "-Xclang", native_amd_library(name)?]);
			}
			let library_directory = option_env!("RECIPE_HSA_DEVICE_LIBRARY_DIRECTORY").ok_or_else(|| RecipeError::new("AMD native device library directory is unavailable"))?;
			let isa = Path::new(library_directory).join(format!("oclc_isa_version_{}.bc", architecture.trim_start_matches("gfx")));
			command.args(["-Xclang", "-mlink-builtin-bitcode", "-Xclang"]).arg(isa).arg(source).arg("-o").arg(output);
			native_command(command, "AMD LLVM IR compiler", key).map(|diagnostic| kernel_resources(&diagnostic))
		}
		BackendTarget::Nvidia { architecture } => {
			let compiler = native_nvidia_compiler()?;
			let device = option_env!("RECIPE_NV_DEVICE_LIBRARY").ok_or_else(|| RecipeError::new("NVIDIA native device library is unavailable"))?;
			let bitcode = bitcode.ok_or_else(|| RecipeError::new("NVIDIA bitcode path is absent"))?;
			let ptx_version = option_env!("RECIPE_NV_PTX_VERSION").ok_or_else(|| RecipeError::new("NVIDIA PTX version is unavailable"))?;
			let mut llvm = Command::new(compiler);
			llvm.args(["-target", "nvptx64-nvidia-cuda"])
				.arg(format!("-march={architecture}"))
				.args(["-Xclang", "-target-feature", "-Xclang", ptx_version, "-O2", "-emit-llvm", "-c", "-x", "ir"])
				.arg(source.to_str().ok_or_else(|| RecipeError::new("native LLVM source path is not UTF-8"))?)
				.args(["-Xclang", "-mlink-builtin-bitcode", "-Xclang", device, "-o"])
				.arg(bitcode);
			native_command(llvm, "NVIDIA LLVM IR compiler", key)?;
			// Both stages take the pinned ISA: the bitcode step rejects any target newer than its own default, and the generator stamps the version into the artifact so the driver JIT loads it on every driver at or above that version.
			let generator = option_env!("RECIPE_NV_PTX_GENERATOR").ok_or_else(|| RecipeError::new("NVIDIA PTX generator is unavailable"))?;
			let mut llc = Command::new(generator);
			llc.args(["-march=nvptx64", &format!("-mcpu={architecture}"), &format!("-mattr={ptx_version}"), "-O2"]).arg(bitcode).args(["-o"]).arg(output);
			native_command(llc, "NVIDIA PTX generator", key)?;
			fs::read(output)
				.and_then(|mut image| {
					image.push(0);
					fs::write(output, &image)
				})
				.map(|_| Vec::new())
				.map_err(|error| RecipeError::new(format!("cannot terminate native PTX artifact: {error}")))
		}
	}
}

pub(crate) fn compile_model(target: &BackendTarget, graph: &Graph, precision: Compute, loss: Option<LossFunction>, rows: usize, schedule: NativeSchedule) -> Result<NativeArtifact> {
	target.validate()?;
	let model = NativeModelIr::from_graph(graph, rows, precision, schedule, loss.is_none())?;
	let matrix = match target {
		BackendTarget::Amd { architecture } if architecture.starts_with("gfx11") => Some(NativeMatrix::Gfx11),
		BackendTarget::Amd { architecture } if architecture.starts_with("gfx12") => Some(NativeMatrix::Gfx12),
		_ => None,
	}
	.filter(|_| model.schedule.matrix);
	let ir = model.emit(target.backend(), matrix, loss)?;
	let key = native_artifact_key(target, &ir);
	let directory = native_artifact_directory(&key)?;
	fs::create_dir_all(&directory).map_err(|error| RecipeError::new(format!("cannot create native artifact directory: {error}")))?;
	let path = directory.join(format!("artifact.{}", target.artifact_extension()));
	let cached = path.is_file();
	debug(&format!(
		"native artifact key={key} target={} arithmetic={} loss={} rows={rows} cache={} path={}",
		native_target_label(target).split(";features=").next().unwrap_or("unknown"),
		model.precision.model.label(),
		loss.map_or("none", |loss| loss.name()),
		if cached { "hit" } else { "miss" },
		path.display()
	))?;
	let artifact = if cached {
		fs::read(&path).map_err(|error| RecipeError::new(format!("cannot read native artifact {}: {error}", path.display())))?
	} else {
		let serial = NATIVE_ARTIFACT_SERIAL.fetch_add(1, Ordering::Relaxed);
		let stem = format!(".recipe-native-{}-{serial}", std::process::id());
		let source = directory.join(format!("{stem}.ll"));
		let output = directory.join(format!("{stem}.{}", target.artifact_extension()));
		let bitcode = (target.backend() == Backend::Nvidia).then(|| directory.join(format!("{stem}.bc")));
		let temporary = NativeTemporaryFiles { paths: std::iter::once(source.clone()).chain(std::iter::once(output.clone())).chain(bitcode.iter().cloned()).collect() };
		fs::write(&source, ir).map_err(|error| RecipeError::new(format!("cannot write native LLVM IR: {error}")))?;
		debug(&format!("native source key={key} path={}", source.display()))?;
		// The artifact is cached on disk, so this is the one moment the compiler's
		// view of the kernel exists. A cooperative grid requires every workgroup to
		// be resident at once, and the register allocation is what decides that.
		// The kernel declares a workgroup range up to the device maximum, so a
		// nonzero occupancy here holds for every workgroup width the schedule can
		// pick. It does not bound the tile: local memory is checked separately, and
		// the schedule still does not resize itself from these numbers.
		for kernel in compile_native_artifact(target, &source, &output, bitcode.as_deref(), &key)? {
			debug(&format!("native kernel {} registers={} scalars={} occupancy={} waves per SIMD", kernel.name, kernel.registers, kernel.scalars, kernel.occupancy))?;
			require(kernel.occupancy != 0, format!("native kernel {} cannot be resident at any workgroup width", kernel.name))?;
		}
		fs::rename(&output, &path).map_err(|error| RecipeError::new(format!("cannot publish native artifact {}: {error}", path.display())))?;
		drop(temporary);
		fs::read(&path).map_err(|error| RecipeError::new(format!("cannot read native artifact {}: {error}", path.display())))?
	};
	require(!artifact.is_empty(), format!("native artifact {} is empty", path.display()))?;
	Ok(NativeArtifact { backend: target.clone(), layout: model.layout.clone(), precision: model.precision, artifact, path, storage: model.storage(), training: loss.is_some() })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FloatLayout {
	sign: u8,
	exp: u8,
	man: u8,
}
impl FloatLayout {
	const fn new(sign: u8, exp: u8, man: u8) -> Self {
		Self { sign, exp, man }
	}
	const fn bias(self) -> u64 {
		(1u64 << (self.exp - 1)) - 1
	}
	const fn bits(self) -> u8 {
		self.sign + self.exp + self.man
	}
	fn pack(self, value: f64) -> u64 {
		let sign = value.to_bits() >> (u64::BITS - 1) << (self.exp + self.man);
		let exponent_limit = (1u64 << self.exp) - 1;
		let mantissa_limit = 1u64 << self.man;
		match value.classify() {
			std::num::FpCategory::Nan => sign | exponent_limit << self.man | 1u64 << (self.man - 1),
			std::num::FpCategory::Infinite => sign | exponent_limit << self.man,
			std::num::FpCategory::Zero => sign,
			std::num::FpCategory::Normal | std::num::FpCategory::Subnormal => {
				let magnitude = value.abs();
				let minimum_exponent = 1 - self.bias() as i64;
				match magnitude.log2().floor() as i64 {
					exponent if exponent < minimum_exponent => {
						let scale = power(minimum_exponent);
						let mantissa = (magnitude / scale * mantissa_limit as f64).round_ties_even() as u64;
						if mantissa == mantissa_limit { sign | 1u64 << self.man } else { sign | mantissa }
					}
					mut exponent => {
						let mut mantissa = ((magnitude / power(exponent) - 1.0) * mantissa_limit as f64).round_ties_even() as u64;
						if mantissa == mantissa_limit {
							mantissa = 0;
							exponent += 1
						}
						let stored_exponent = exponent + self.bias() as i64;
						if stored_exponent >= exponent_limit as i64 { sign | exponent_limit << self.man } else { sign | (stored_exponent as u64) << self.man | mantissa }
					}
				}
			}
		}
	}
	fn unpack(self, bits: u64) -> f64 {
		let negative = bits >> (self.exp + self.man) != 0;
		let exponent_limit = (1u64 << self.exp) - 1;
		let mantissa_limit = 1u64 << self.man;
		let exponent = bits >> self.man & exponent_limit;
		let mantissa = bits & (mantissa_limit - 1);
		let magnitude = match (exponent, mantissa) {
			(value, 0) if value == exponent_limit => f64::INFINITY,
			(value, _) if value == exponent_limit => f64::NAN,
			(0, 0) => 0.0,
			(0, value) => power(1 - self.bias() as i64) * value as f64 / mantissa_limit as f64,
			(value, man) => power(value as i64 - self.bias() as i64) * (1.0 + man as f64 / mantissa_limit as f64),
		};
		if negative { -magnitude } else { magnitude }
	}
}
fn power(exponent: i64) -> f64 {
	if exponent > f64::MAX_EXP as i64 {
		f64::INFINITY
	} else if exponent < f64::MIN_EXP as i64 - f64::MANTISSA_DIGITS as i64 {
		0.0
	} else {
		2.0f64.powi(exponent as i32)
	}
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
	const fn computed(exp: u8, man: u8) -> Self {
		Self { arithmetic: FloatLayout::new(1, exp, man), storage: Self::FP64.storage }
	}
	const fn native(sign: u8, exp: u8, man: u8) -> Self {
		let layout = FloatLayout::new(sign, exp, man);
		Self { arithmetic: layout, storage: layout }
	}
	const fn bytes(self) -> usize {
		self.storage.bits().div_ceil(8) as usize
	}
	fn pack(self, value: f64) -> u64 {
		self.storage.pack(self.arithmetic.unpack(self.arithmetic.pack(value)))
	}
	fn unpack(self, bits: u64) -> f64 {
		self.arithmetic.unpack(self.arithmetic.pack(self.storage.unpack(bits)))
	}
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
		let minimum = -(1i64 << (self.bits - 1));
		let maximum = (1i64 << (self.bits - 1)) - 1;
		(value.round_ties_even() as i64).clamp(minimum, maximum) as u64 & ((1u64 << self.bits) - 1)
	}
	fn unpack(self, bits: u64) -> f64 {
		((bits << (u64::BITS as u8 - self.bits)) as i64 >> (u64::BITS as u8 - self.bits)) as f64
	}
}
/// GGUF, the container open-weight models ship in: a little-endian header of
/// key-value metadata and tensor descriptors, then an aligned data section
/// whose tensors use the same GGML block layouts as Recipe's storage formats.
mod gguf {
	use super::{Binding, BoundNode, BoundWeight, Integer, Plane, RecipeError, Result, StorageFormat, StoredBytes, StoredWeight, checked_add, require, unfp16};
	use std::{
		path::{Path, PathBuf},
		sync::Arc,
	};

	const MAGIC: u32 = 0x4655_4747;
	const VERSION: u32 = 3;
	const DEFAULT_ALIGNMENT: u64 = 32;

	/// One metadata value, in the width the file declares.
	#[derive(Clone, Debug, PartialEq)]
	pub enum GgufValue {
		U8(u8),
		I8(i8),
		U16(u16),
		I16(i16),
		U32(u32),
		I32(i32),
		F32(f32),
		Bool(bool),
		String(String),
		Array(Vec<GgufValue>),
		U64(u64),
		I64(i64),
		F64(f64),
	}
	impl GgufValue {
		/// The value as a count or index, whichever integer width the file chose.
		pub fn integer(&self) -> Option<u64> {
			match *self {
				Self::U8(value) => Some(u64::from(value)),
				Self::U16(value) => Some(u64::from(value)),
				Self::U32(value) => Some(u64::from(value)),
				Self::U64(value) => Some(value),
				Self::I8(value) => u64::try_from(value).ok(),
				Self::I16(value) => u64::try_from(value).ok(),
				Self::I32(value) => u64::try_from(value).ok(),
				Self::I64(value) => u64::try_from(value).ok(),
				_ => None,
			}
		}
		pub fn text(&self) -> Option<&str> {
			match self {
				Self::String(value) => Some(value),
				_ => None,
			}
		}
		/// The value as a real number, whichever float width the file chose.
		pub fn float(&self) -> Option<f64> {
			match *self {
				Self::F32(value) => Some(f64::from(value)),
				Self::F64(value) => Some(value),
				_ => None,
			}
		}
	}

	/// One tensor descriptor: `kind` is the GGML type id and `offset` counts bytes
	/// from the start of its shard's data section.
	#[derive(Clone, Debug, PartialEq, Eq)]
	pub struct GgufTensor {
		pub name: String,
		pub shape: Vec<u64>,
		pub kind: u32,
		pub offset: usize,
		pub bytes: usize,
		shard: usize,
	}

	impl GgufTensor {
		/// The element count the shape declares.
		pub fn elements(&self) -> usize {
			self.shape.iter().product::<u64>() as usize
		}
		/// The `[k, n]` slice of a `[k, n, experts]` tensor at `index`. The slice is the
		/// expert's own blocks where the file is mapped, so reading one expert leaves
		/// the others untouched.
		pub fn expert(&self, index: usize) -> Result<Self> {
			require(self.shape.len() == 3, format!("tensor {} has {} dimensions; an expert slice takes a [k, n, experts] tensor", self.name, self.shape.len()))?;
			let experts = self.shape[2] as usize;
			require(index < experts, format!("tensor {} holds {experts} experts, so expert {index} is absent", self.name))?;
			let rows = self.shape[1] as usize;
			self.slice(self.shape[..2].to_vec(), index * rows, rows)
		}
		/// Whether the device block decoders read this tensor's layout, so it binds
		/// as a view of its mapping rather than as decoded parameters.
		pub fn blocked(&self) -> bool {
			layout(self.kind).is_ok_and(|(_, _, _, format)| format.is_some())
		}
		/// The `[k, count]` slice holding output rows `start .. start + count`.
		/// Rows are the tensor's slowest axis, so the slice is one contiguous run
		/// of the mapping and stays a view of it.
		pub fn rows(&self, start: usize, count: usize) -> Result<Self> {
			require(self.shape.len() >= 2, format!("tensor {} has {} dimensions; a row slice takes at least [k, n]", self.name, self.shape.len()))?;
			let rows = self.elements() / self.shape[0] as usize;
			let end = start.checked_add(count).ok_or_else(|| RecipeError::new(format!("tensor {} row slice overflows", self.name)))?;
			require(count != 0 && end <= rows, format!("tensor {} holds {rows} rows, so rows {start}..{end} are absent", self.name))?;
			self.slice(vec![self.shape[0], count as u64], start, count)
		}
		/// A view of `count` rows of `self.shape[0]` elements each, `before` rows in.
		fn slice(&self, shape: Vec<u64>, before: usize, count: usize) -> Result<Self> {
			let (_, block, stride, _) = layout(self.kind)?;
			let width = self.shape[0] as usize;
			let (skipped, elements) = (before * width, count * width);
			require(skipped % block == 0 && elements % block == 0, format!("tensor {} rows of {width} do not divide its {block}-element block, so a slice would cut one", self.name))?;
			Ok(Self { name: self.name.clone(), shape, kind: self.kind, offset: self.offset + skipped / block * stride, bytes: elements / block * stride, shard: self.shard })
		}
	}

	/// GGML type id to its name, elements per block, bytes per block, and the Recipe
	/// storage format that decodes the block, for the types that are block quantized.
	fn layout(kind: u32) -> Result<(&'static str, usize, usize, Option<StorageFormat>)> {
		let name = match kind {
			0 => return Ok(("F32", 1, 4, None)),
			1 => return Ok(("F16", 1, 2, None)),
			24 => return Ok(("I8", 1, 1, None)),
			25 => return Ok(("I16", 1, 2, None)),
			26 => return Ok(("I32", 1, 4, None)),
			27 => return Ok(("I64", 1, 8, None)),
			28 => return Ok(("F64", 1, 8, None)),
			30 => return Ok(("BF16", 1, 2, None)),
			2 => "q4_0",
			3 => "q4_1",
			6 => "q5_0",
			7 => "q5_1",
			8 => "q8_0",
			9 => "q8_1",
			10 => "q2k",
			11 => "q3k",
			12 => "q4k",
			13 => "q5k",
			14 => "q6k",
			15 => "q8k",
			16 => "iq2xxs",
			17 => "iq2xs",
			18 => "iq3xxs",
			19 => "iq1s",
			20 => "iq4nl",
			21 => "iq3s",
			22 => "iq2s",
			23 => "iq4xs",
			29 => "iq1m",
			other => return Err(RecipeError::new(format!("GGUF tensor type {other} is unsupported"))),
		};
		let spec = StorageFormat::named(name).and_then(|format| format.spec().map(|spec| (format, spec)));
		let (format, spec) = spec.ok_or_else(|| RecipeError::new(format!("GGUF type {name} has no Recipe storage layout")))?;
		Ok((name, spec.block, spec.stride, Some(format)))
	}

	/// A read-only view of one file: mapped by the kernel on unix, read into
	/// memory elsewhere. A stored weight holds a shard's mapping alive for as long
	/// as the tape reads its bytes.
	pub(super) struct Mapping {
		pointer: *const u8,
		length: usize,
		owned: Option<Vec<u8>>,
	}
	unsafe impl Send for Mapping {}
	unsafe impl Sync for Mapping {}
	impl Mapping {
		#[cfg(unix)]
		fn open(path: &Path) -> Result<Self> {
			use std::os::unix::io::AsRawFd;
			let file = std::fs::File::open(path).map_err(|error| RecipeError::new(format!("cannot open {}: {error}", path.display())))?;
			let length = usize::try_from(file.metadata().map_err(|error| RecipeError::new(format!("cannot inspect {}: {error}", path.display())))?.len())
				.map_err(|_| RecipeError::new("GGUF file exceeds the address space"))?;
			require(length != 0, format!("{} is empty", path.display()))?;
			let pointer = unsafe { super::mmap(std::ptr::null_mut(), length, 1, 2, file.as_raw_fd(), 0) };
			require(pointer as isize != -1, format!("cannot map {}: {}", path.display(), std::io::Error::last_os_error()))?;
			Ok(Self { pointer: pointer.cast(), length, owned: None })
		}
		#[cfg(not(unix))]
		fn open(path: &Path) -> Result<Self> {
			let owned = std::fs::read(path).map_err(|error| RecipeError::new(format!("cannot read {}: {error}", path.display())))?;
			require(!owned.is_empty(), format!("{} is empty", path.display()))?;
			Ok(Self { pointer: owned.as_ptr(), length: owned.len(), owned: Some(owned) })
		}
		pub(super) fn bytes(&self) -> &[u8] {
			unsafe { std::slice::from_raw_parts(self.pointer, self.length) }
		}
	}
	impl Drop for Mapping {
		fn drop(&mut self) {
			#[cfg(unix)]
			if self.owned.is_none() {
				unsafe { super::munmap(self.pointer.cast_mut().cast(), self.length) };
			}
		}
	}

	struct Reader<'a> {
		bytes: &'a [u8],
		at: usize,
		depth: u32,
	}
	impl<'a> Reader<'a> {
		fn take(&mut self, count: usize) -> Result<&'a [u8]> {
			let end = self.at.checked_add(count).filter(|end| *end <= self.bytes.len()).ok_or_else(|| RecipeError::new("GGUF header is truncated"))?;
			let slice = &self.bytes[self.at..end];
			self.at = end;
			Ok(slice)
		}
		fn u32(&mut self) -> Result<u32> {
			self.take(4).map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
		}
		fn u64(&mut self) -> Result<u64> {
			self.take(8).map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
		}
		fn string(&mut self) -> Result<String> {
			let length = usize::try_from(self.u64()?).map_err(|_| RecipeError::new("GGUF string length exceeds the address space"))?;
			String::from_utf8(self.take(length)?.to_vec()).map_err(|error| RecipeError::new(format!("GGUF string is not UTF-8: {error}")))
		}
		fn value(&mut self, kind: u32) -> Result<GgufValue> {
			Ok(match kind {
				0 => GgufValue::U8(self.take(1)?[0]),
				1 => GgufValue::I8(self.take(1)?[0] as i8),
				2 => GgufValue::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
				3 => GgufValue::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
				4 => GgufValue::U32(self.u32()?),
				5 => GgufValue::I32(self.u32()? as i32),
				6 => GgufValue::F32(f32::from_bits(self.u32()?)),
				7 => GgufValue::Bool(self.take(1)?[0] != 0),
				8 => GgufValue::String(self.string()?),
				9 => {
					let (kind, count) = (self.u32()?, self.u64()?);
					let count = usize::try_from(count).map_err(|_| RecipeError::new("GGUF array length exceeds the address space"))?;
					self.depth += 1;
					require(self.depth <= 64, "GGUF arrays nest deeper than 64 levels")?;
					let values = (0..count).map(|_| self.value(kind)).collect::<Result<Vec<_>>>()?;
					self.depth -= 1;
					GgufValue::Array(values)
				}
				10 => GgufValue::U64(self.u64()?),
				11 => GgufValue::I64(self.u64()? as i64),
				12 => GgufValue::F64(f64::from_bits(self.u64()?)),
				other => return Err(RecipeError::new(format!("GGUF value type {other} is unsupported"))),
			})
		}
	}

	#[derive(Clone)]
	struct Shard {
		mapping: Arc<Mapping>,
		data: usize,
	}

	/// One model: a single file or every shard of a split, opened by name.
	#[derive(Clone)]
	pub struct Gguf {
		shards: Vec<Shard>,
		metadata: Vec<(String, GgufValue)>,
		tensors: Vec<GgufTensor>,
	}
	impl Gguf {
		pub(super) fn open(path: &Path) -> Result<Self> {
			let first = Self::shard(path, 0)?;
			let count = first.1.iter().find(|(key, _)| key == "split.count").and_then(|(_, value)| value.integer()).unwrap_or(1);
			let mut shards = vec![first];
			for index in 1..count {
				shards.push(Self::shard(&sibling(path, index, count)?, index)?);
			}
			let (metadata, mut tensors) = (shards[0].1.clone(), Vec::new());
			let declared = metadata.iter().find(|(key, _)| key == "split.tensors.count").and_then(|(_, value)| value.integer());
			for (index, (_, _, shard_tensors)) in shards.iter_mut().enumerate() {
				tensors.extend(shard_tensors.drain(..).map(|mut tensor| {
					tensor.shard = index;
					tensor
				}));
			}
			if let Some(declared) = declared {
				require(declared == tensors.len() as u64, format!("GGUF split declares {declared} tensors and holds {}", tensors.len()))?;
			}
			Ok(Self { shards: shards.into_iter().map(|(shard, _, _)| shard).collect(), metadata, tensors })
		}
		/// Parses one file: its metadata, its tensors, and where its data begins.
		fn shard(path: &Path, index: u64) -> Result<(Shard, Vec<(String, GgufValue)>, Vec<GgufTensor>)> {
			let mapping = Mapping::open(path)?;
			let bytes = mapping.bytes();
			let mut reader = Reader { bytes, at: 0, depth: 0 };
			require(reader.u32()? == MAGIC, format!("{} is not a GGUF file", path.display()))?;
			let version = reader.u32()?;
			require(version == VERSION, format!("{} is GGUF version {version}; version {VERSION} is supported", path.display()))?;
			let (tensor_count, pair_count) = (reader.u64()?, reader.u64()?);
			let mut metadata = Vec::new();
			for _ in 0..pair_count {
				let key = reader.string()?;
				let kind = reader.u32()?;
				metadata.push((key, reader.value(kind)?));
			}
			let alignment = metadata.iter().find(|(key, _)| key == "general.alignment").and_then(|(_, value)| value.integer()).unwrap_or(DEFAULT_ALIGNMENT);
			require(alignment != 0 && alignment.is_power_of_two(), format!("{} declares alignment {alignment}", path.display()))?;
			if let Some(declared) = metadata.iter().find(|(key, _)| key == "split.no").and_then(|(_, value)| value.integer()) {
				require(declared == index, format!("{} is shard {declared}, expected shard {index}", path.display()))?;
			}
			let mut tensors = Vec::new();
			for _ in 0..tensor_count {
				let name = reader.string()?;
				let dimensions = reader.u32()?;
				let shape = (0..dimensions).map(|_| reader.u64()).collect::<Result<Vec<_>>>()?;
				let (kind, offset) = (reader.u32()?, reader.u64()?);
				let (_, block, stride, _) = layout(kind)?;
				let elements = shape.iter().try_fold(1_u64, |product, dimension| product.checked_mul(*dimension)).ok_or_else(|| RecipeError::new(format!("tensor {name} shape overflows")))?;
				require(elements % block as u64 == 0, format!("tensor {name} holds {elements} elements, not a multiple of its {block}-element block"))?;
				let bytes = usize::try_from(elements / block as u64 * stride as u64).map_err(|_| RecipeError::new(format!("tensor {name} exceeds the address space")))?;
				let offset = usize::try_from(offset).map_err(|_| RecipeError::new(format!("tensor {name} offset exceeds the address space")))?;
				tensors.push(GgufTensor { name, shape, kind, offset, bytes, shard: 0 });
			}
			let data = usize::try_from((reader.at as u64).div_ceil(alignment) * alignment).map_err(|_| RecipeError::new("GGUF data offset exceeds the address space"))?;
			for tensor in &tensors {
				let end = data.checked_add(tensor.offset).and_then(|start| start.checked_add(tensor.bytes));
				require(end.is_some_and(|end| end <= bytes.len()), format!("tensor {} runs past the end of {}", tensor.name, path.display()))?;
			}
			Ok((Shard { mapping: Arc::new(mapping), data }, metadata, tensors))
		}
		/// Every key-value pair of the first shard, in file order.
		pub fn metadata(&self) -> &[(String, GgufValue)] {
			&self.metadata
		}
		pub fn value(&self, key: &str) -> Option<&GgufValue> {
			self.metadata.iter().find(|(name, _)| name == key).map(|(_, value)| value)
		}
		/// Every tensor across every shard, in shard then file order.
		pub fn tensors(&self) -> &[GgufTensor] {
			&self.tensors
		}
		pub fn tensor(&self, name: &str) -> Option<&GgufTensor> {
			self.tensors.iter().find(|tensor| tensor.name == name)
		}
		pub(super) fn required(&self, key: &str) -> Result<&GgufValue> {
			self.value(key).ok_or_else(|| RecipeError::new(format!("{key} is absent")))
		}
		pub(super) fn integer_at(&self, key: &str) -> Result<usize> {
			let value = self.required(key)?.integer().ok_or_else(|| RecipeError::new(format!("{key} is not a nonnegative integer")))?;
			usize::try_from(value).map_err(|_| RecipeError::new(format!("{key} exceeds the address space")))
		}
		pub(super) fn float_at(&self, key: &str) -> Result<f64> {
			self.required(key)?.float().ok_or_else(|| RecipeError::new(format!("{key} is not a float")))
		}
		pub(super) fn integers_at(&self, key: &str) -> Result<Vec<u64>> {
			match self.required(key)? {
				GgufValue::Array(values) => {
					values.iter().map(|value| value.integer().ok_or_else(|| RecipeError::new(format!("{key} holds a value that is not a nonnegative integer")))).collect()
				}
				_ => Err(RecipeError::new(format!("{key} is not an array"))),
			}
		}
		pub(super) fn indices_at(&self, key: &str) -> Result<Vec<usize>> {
			self.integers_at(key)?.into_iter().map(|value| usize::try_from(value).map_err(|_| RecipeError::new(format!("{key} exceeds the address space")))).collect()
		}
		/// The tensor's bytes in the mapped file, in its GGML block layout.
		pub fn data(&self, tensor: &GgufTensor) -> &[u8] {
			let shard = &self.shards[tensor.shard];
			&shard.mapping.bytes()[shard.data + tensor.offset..shard.data + tensor.offset + tensor.bytes]
		}
		/// The byte-level BPE tokenizer the metadata describes.
		pub fn tokenizer(&self) -> super::Tokenizer {
			super::Tokenizer::from_gguf(self).unwrap_or_else(|error| panic!("{error}"))
		}
		/// The tensor's elements decoded to f64, dequantizing block formats through
		/// the same decoders the saved-model path uses.
		pub fn values(&self, tensor: &GgufTensor) -> Result<Vec<f64>> {
			decode(tensor, self.data(tensor), tensor.elements())
		}
		/// Row `index` of a tensor whose rows are its first dimension, decoded from
		/// that row's own bytes so a table larger than memory reads only the rows it
		/// addresses.
		pub fn row(&self, tensor: &GgufTensor, index: usize) -> Result<Vec<f64>> {
			let (_, block, stride, _) = layout(tensor.kind)?;
			let width = tensor.shape.first().copied().unwrap_or(0) as usize;
			require(width != 0 && width % block == 0, format!("tensor {} has rows of {width}, not whole blocks of {block}", tensor.name))?;
			let rows = tensor.elements() / width;
			require(index < rows, format!("tensor {} has {rows} rows, row {index} requested", tensor.name))?;
			let bytes = width / block * stride;
			decode(tensor, &self.data(tensor)[index * bytes..(index + 1) * bytes], width)
		}
		/// The tensor as the weight the tape binds: a view of its block bytes where the
		/// file is mapped, in the Recipe storage format the kernels already decode.
		pub(super) fn stored(&self, tensor: &GgufTensor) -> Result<StoredWeight> {
			let shard = &self.shards[tensor.shard];
			let bytes = StoredBytes::mapped(&shard.mapping, shard.data + tensor.offset, tensor.bytes);
			Ok(StoredWeight { format: block_format(tensor)?, count: tensor.elements(), bytes, codebook: Vec::new(), arithmetic: Vec::new() })
		}
		/// Resolves a plan against this file, one bound weight per entry. The views
		/// of one entry must share a layout: block-quantized views join as runs of
		/// their own mappings, so nothing is decoded or copied here, and views in
		/// an unblocked layout, or values the host rewrote, decode into values,
		/// and then every view of the entry decodes with them. Block-quantized
		/// views whose layouts differ are rejected by name before anything compiles.
		pub(super) fn bound(&self, plan: &Binding) -> Result<Vec<BoundNode>> {
			plan.nodes
				.iter()
				.enumerate()
				.map(|(entry, planes)| {
					let first = planes.first().ok_or_else(|| RecipeError::new(format!("plan entry {entry} names no tensor")))?;
					let views = planes.iter().map(Plane::mapped).collect::<Option<Vec<_>>>();
					let packed = views.as_ref().is_some_and(|views| views.iter().all(|view| view.blocked()));
					if let Some(views) = views.as_ref().filter(|_| packed)
						&& let Some(other) = views.iter().find(|plane| plane.kind != views[0].kind)
					{
						let name = |kind| layout(kind).map_or("an unsupported type", |(name, _, _, _)| name);
						return Err(RecipeError::new(format!("tensor {} is {} but tensor {} of the same node is {}", other.name, name(other.kind), views[0].name, name(views[0].kind))));
					}
					let elements = planes.iter().try_fold(0, |total, plane| checked_add(total, plane.elements(), "plan tensor elements"))?;
					let mut names = Vec::new();
					for plane in planes {
						if !names.contains(&plane.name()) {
							names.push(plane.name());
						}
					}
					let names = match (planes.len(), first) {
						(1, Plane::Mapped(tensor)) => format!("tensor {} {:?}", tensor.name, tensor.shape),
						(1, Plane::Owned { name, .. }) => format!("values {name}"),
						(count, _) => format!("{count} views of {}", names.join(", ")),
					};
					let weight = match views.filter(|_| packed) {
						Some(views) => {
							let parts = views.iter().map(|plane| self.stored(plane)).collect::<Result<Vec<_>>>()?;
							let format = parts[0].format;
							let bytes = StoredBytes::joined(parts.into_iter().map(|part| part.bytes).collect());
							BoundWeight::Stored(StoredWeight { format, count: elements, bytes, codebook: Vec::new(), arithmetic: Vec::new() })
						}
						None => {
							let mut values = Vec::with_capacity(elements);
							for plane in planes {
								match plane {
									Plane::Mapped(tensor) => values.extend(self.values(tensor)?),
									Plane::Owned { values: owned, .. } => values.extend_from_slice(owned),
								}
							}
							BoundWeight::Values(values)
						}
					};
					Ok(BoundNode { names, elements, weight })
				})
				.collect()
		}
	}

	/// The Recipe storage format a block-quantized tensor decodes through.
	pub(super) fn block_format(tensor: &GgufTensor) -> Result<StorageFormat> {
		let (name, _, _, format) = layout(tensor.kind)?;
		format.ok_or_else(|| RecipeError::new(format!("tensor {} is {name}, which the tape reads only as a block-quantized weight", tensor.name)))
	}

	/// `count` elements of a tensor decoded to f64 from `data`, the bytes holding them.
	fn decode(tensor: &GgufTensor, data: &[u8], count: usize) -> Result<Vec<f64>> {
		match tensor.kind {
			0 => Ok(data.chunks_exact(4).map(|bytes| f64::from(f32::from_le_bytes(bytes.try_into().unwrap()))).collect()),
			1 => Ok(data.chunks_exact(2).map(|bytes| f64::from(unfp16(u16::from_le_bytes(bytes.try_into().unwrap())))).collect()),
			30 => Ok(data.chunks_exact(2).map(|bytes| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16))).collect()),
			28 => Ok(data.chunks_exact(8).map(|bytes| f64::from_le_bytes(bytes.try_into().unwrap())).collect()),
			24 => Ok(data.iter().map(|byte| f64::from(*byte as i8)).collect()),
			25 => Ok(data.chunks_exact(2).map(|bytes| f64::from(i16::from_le_bytes(bytes.try_into().unwrap()))).collect()),
			26 => Ok(data.chunks_exact(4).map(|bytes| f64::from(i32::from_le_bytes(bytes.try_into().unwrap()))).collect()),
			27 => Ok(data.chunks_exact(8).map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()) as f64).collect()),
			_ => block_format(tensor)?.decompress(data, &[], count),
		}
	}

	/// The path of shard `index` of a split named like `model-00001-of-00004.gguf`.
	fn sibling(path: &Path, index: u64, count: u64) -> Result<PathBuf> {
		let name = path.file_name().and_then(|name| name.to_str()).ok_or_else(|| RecipeError::new("GGUF split path has no file name"))?;
		let stem = name.strip_suffix(".gguf").and_then(|stem| stem.rsplit_once("-of-")).and_then(|(head, _)| head.rsplit_once('-')).map(|(prefix, _)| prefix);
		let prefix = stem.ok_or_else(|| RecipeError::new(format!("{name} declares a split but is not named <prefix>-<number>-of-<count>.gguf")))?;
		Ok(path.with_file_name(format!("{prefix}-{:05}-of-{count:05}.gguf", index + 1)))
	}
}
pub use gguf::{Gguf, GgufTensor, GgufValue};
mod tokenizer {
	//! A byte-level BPE tokenizer built from GGUF metadata alone: the token
	//! table, the piece ranks, the pre-tokenizer family, the added tokens, the
	//! special ids, and the chat template.
	use super::{Gguf, GgufValue, RecipeError, Result, require};
	use std::collections::HashMap;

	/// The pre-tokenizer alternation that cuts text into words before merging.
	#[derive(Clone, Copy, PartialEq, Eq)]
	enum Family {
		Gpt2,
		Llama3,
		Qwen2,
	}

	fn letter(value: char) -> bool {
		value.is_alphabetic()
	}
	fn number(value: char) -> bool {
		value.is_numeric()
	}
	fn space(value: char) -> bool {
		value.is_whitespace()
	}
	fn newline(value: char) -> bool {
		value == '\r' || value == '\n'
	}
	fn other(value: char) -> bool {
		!space(value) && !letter(value) && !number(value)
	}
	fn run(chars: &[char], at: usize, class: fn(char) -> bool) -> usize {
		chars[at.min(chars.len())..].iter().take_while(|value| class(**value)).count()
	}
	/// `'s|'t|'re|'ve|'m|'ll|'d`, case-insensitive for the newer families.
	fn contraction(chars: &[char], at: usize, insensitive: bool) -> usize {
		if chars.get(at) != Some(&'\'') {
			return 0;
		}
		let same = |index: usize, expected: char| chars.get(at + index).is_some_and(|value| if insensitive { value.to_ascii_lowercase() == expected } else { *value == expected });
		["s", "t", "re", "ve", "m", "ll", "d"].iter().find(|suffix| suffix.chars().enumerate().all(|(index, expected)| same(1 + index, expected))).map_or(0, |suffix| 1 + suffix.len())
	}
	/// `\s+(?!\S)`: a whitespace run that leaves its last character to the
	/// following word unless it ends the text.
	fn trailing_space(chars: &[char], at: usize) -> usize {
		let count = run(chars, at, space);
		if count == 0 {
			0
		} else if at + count == chars.len() {
			count
		} else {
			count - 1
		}
	}

	impl Family {
		fn named(pre: &str) -> Result<Self> {
			Ok(match pre {
				"gpt-2" | "phi-2" | "jina-v1-en" | "jina-v2-es" | "jina-v2-de" | "jina-v2-code" | "roberta-bpe" | "gigachat" | "olmo" => Self::Gpt2,
				"llama3" | "llama-v3" | "llama-bpe" | "smaug-bpe" | "dbrx" => Self::Llama3,
				"qwen2" | "qwen35" | "deepseek-r1-qwen" | "stablelm2" => Self::Qwen2,
				other => return Err(RecipeError::new(format!("pre-tokenizer family {other:?} is not supported"))),
			})
		}
		/// The length in characters of the word that starts at `at`.
		fn word(self, chars: &[char], at: usize) -> usize {
			let space_prefix = usize::from(chars[at] == ' ');
			let prefixed = |class: fn(char) -> bool| {
				let count = run(chars, at + space_prefix, class);
				if count == 0 { 0 } else { space_prefix + count }
			};
			if self == Self::Gpt2 {
				let count = contraction(chars, at, false);
				if count != 0 {
					return count;
				}
				for class in [letter, number, other] {
					let count = prefixed(class);
					if count != 0 {
						return count;
					}
				}
			} else {
				let count = contraction(chars, at, true);
				if count != 0 {
					return count;
				}
				// [^\r\n\p{L}\p{N}]?\p{L}+
				let prefix = usize::from(!newline(chars[at]) && !letter(chars[at]) && !number(chars[at]));
				let count = run(chars, at + prefix, letter);
				if count != 0 {
					return prefix + count;
				}
				// \p{N}{1,3} for llama3, \p{N} for qwen2
				let count = run(chars, at, number).min(if self == Self::Llama3 { 3 } else { 1 });
				if count != 0 {
					return count;
				}
				// ?[^\s\p{L}\p{N}]+[\r\n]*
				let count = prefixed(other);
				if count != 0 {
					return count + run(chars, at + count, newline);
				}
				// \s*[\r\n]+
				let spaces = run(chars, at, space);
				if let Some(last) = (0..spaces).rev().find(|index| newline(chars[at + index])) {
					return last + 1;
				}
			}
			let count = trailing_space(chars, at);
			if count != 0 {
				return count;
			}
			run(chars, at, space).max(1)
		}
		fn split<'a>(self, text: &'a str) -> Vec<&'a str> {
			let (offsets, chars): (Vec<usize>, Vec<char>) = text.char_indices().unzip();
			let (mut at, mut words) = (0, Vec::new());
			while at < chars.len() {
				let next = at + self.word(&chars, at);
				words.push(&text[offsets[at]..offsets.get(next).copied().unwrap_or(text.len())]);
				at = next;
			}
			words
		}
	}

	/// The GPT-2 byte map: printable Latin-1 bytes stay themselves, the rest
	/// take the code points from U+0100 upward in byte order.
	fn byte_map() -> [char; 256] {
		let mut map = ['\0'; 256];
		let mut next = 0x100;
		for byte in 0..=255_u8 {
			let printable = (33..=126).contains(&byte) || (161..=172).contains(&byte) || byte >= 174;
			map[usize::from(byte)] = if printable {
				char::from(byte)
			} else {
				next += 1;
				char::from_u32(next - 1).unwrap()
			};
		}
		map
	}

	/// Where the vocabulary ranks its pieces: the merge list names each pair and
	/// its rank, while scores rank every piece on their own, best first.
	enum Ranks {
		Merges(HashMap<(u32, u32), (u32, u32)>),
		Scores(Vec<u32>),
	}

	/// Piece ranks from `tokenizer.ggml.scores`, the highest score merging first.
	fn ranked(scores: &[GgufValue], vocabulary: usize) -> Result<Vec<u32>> {
		require(scores.len() == vocabulary, format!("tokenizer.ggml.scores holds {} scores for {vocabulary} tokens", scores.len()))?;
		let scores = scores
			.iter()
			.map(|item| match *item {
				GgufValue::F32(score) => Ok(score),
				GgufValue::F64(score) => Ok(score as f32),
				_ => Err(RecipeError::new("tokenizer.ggml.scores holds a non-float")),
			})
			.collect::<Result<Vec<_>>>()?;
		let mut order = (0..scores.len()).collect::<Vec<_>>();
		order.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]).then(left.cmp(right)));
		let mut ranks = vec![0; scores.len()];
		for (rank, id) in order.into_iter().enumerate() {
			ranks[id] = rank as u32;
		}
		Ok(ranks)
	}

	pub struct Tokenizer {
		tokens: Vec<String>,
		ids: HashMap<String, u32>,
		ranks: Ranks,
		added: Vec<(String, u32)>,
		is_added: Vec<bool>,
		bytes: [u32; 256],
		byte_of: HashMap<char, u8>,
		family: Family,
		template: Option<String>,
		add_bos: bool,
		add_eos: bool,
		bos: Option<u32>,
		eos: Option<u32>,
		pad: Option<u32>,
	}

	impl Tokenizer {
		pub(super) fn from_gguf(model: &Gguf) -> Result<Self> {
			let text = |key: &str| model.value(key).and_then(GgufValue::text).ok_or_else(|| RecipeError::new(format!("{key} is absent")));
			let kind = text("tokenizer.ggml.model")?;
			require(kind == "gpt2", format!("tokenizer model {kind:?} is not byte-level BPE"))?;
			let family = Family::named(text("tokenizer.ggml.pre")?)?;
			let array = |key: &str| match model.value(key) {
				Some(GgufValue::Array(items)) => Ok(items.as_slice()),
				_ => Err(RecipeError::new(format!("{key} is absent"))),
			};
			let tokens = array("tokenizer.ggml.tokens")?
				.iter()
				.map(|item| item.text().map(str::to_owned).ok_or_else(|| RecipeError::new("tokenizer.ggml.tokens holds a non-string")))
				.collect::<Result<Vec<_>>>()?;
			let ids = tokens.iter().enumerate().map(|(id, token)| (token.clone(), id as u32)).collect::<HashMap<_, _>>();
			let id = |token: &str| ids.get(token).copied().ok_or_else(|| RecipeError::new(format!("token {token:?} is absent from the vocabulary")));
			let ranks = match (array("tokenizer.ggml.merges"), array("tokenizer.ggml.scores")) {
				(Ok(list), _) => {
					let mut merges = HashMap::new();
					for (rank, merge) in list.iter().enumerate() {
						let merge = merge.text().ok_or_else(|| RecipeError::new("tokenizer.ggml.merges holds a non-string"))?;
						let (left, right) = merge.split_once(' ').ok_or_else(|| RecipeError::new(format!("merge {merge:?} has no pair")))?;
						merges.insert((id(left)?, id(right)?), (rank as u32, id(&format!("{left}{right}"))?));
					}
					Ranks::Merges(merges)
				}
				(_, Ok(scores)) => Ranks::Scores(ranked(scores, tokens.len())?),
				_ => return Err(RecipeError::new("the vocabulary ranks its pieces by neither tokenizer.ggml.merges nor tokenizer.ggml.scores")),
			};
			let map = byte_map();
			let mut bytes = [0; 256];
			for (byte, symbol) in map.iter().enumerate() {
				bytes[byte] = id(&symbol.to_string())?;
			}
			// Control and user-defined tokens are the added tokens: `encode` matches
			// them whole and no merge ever spells one out of ordinary pieces.
			let types = array("tokenizer.ggml.token_type").ok();
			let kind = |index: usize| types.and_then(|types| types.get(index)).and_then(GgufValue::integer);
			let mut added = tokens.iter().enumerate().filter(|(index, _)| matches!(kind(*index), Some(3 | 4))).map(|(index, token)| (token.clone(), index as u32)).collect::<Vec<_>>();
			added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));
			let mut is_added = vec![false; tokens.len()];
			for (_, id) in &added {
				is_added[*id as usize] = true;
			}
			let special_id = |key: &str| model.value(key).and_then(GgufValue::integer).map(|value| value as u32);
			let flag = |key: &str| matches!(model.value(key), Some(GgufValue::Bool(true)));
			Ok(Self {
				byte_of: map.iter().enumerate().map(|(byte, symbol)| (*symbol, byte as u8)).collect(),
				ids,
				ranks,
				added,
				is_added,
				bytes,
				family,
				template: model.value("tokenizer.chat_template").and_then(GgufValue::text).map(str::to_owned),
				add_bos: flag("tokenizer.ggml.add_bos_token"),
				add_eos: flag("tokenizer.ggml.add_eos_token"),
				bos: special_id("tokenizer.ggml.bos_token_id"),
				eos: special_id("tokenizer.ggml.eos_token_id"),
				pad: special_id("tokenizer.ggml.padding_token_id"),
				tokens,
			})
		}
		pub fn bos(&self) -> Option<u32> {
			self.bos
		}
		pub fn eos(&self) -> Option<u32> {
			self.eos
		}
		pub fn pad(&self) -> Option<u32> {
			self.pad
		}
		pub fn vocabulary(&self) -> usize {
			self.tokens.len()
		}
		/// The token's text in the byte-map alphabet, as the vocabulary spells it.
		pub fn token(&self, id: u32) -> &str {
			&self.tokens[id as usize]
		}
		/// Token ids for `text`, framed by the beginning- and end-of-sequence ids
		/// when the model asks for them. Added tokens written out in the text are
		/// matched whole, longest first, before the pre-tokenizer sees the rest.
		pub fn encode(&self, text: &str) -> Vec<u32> {
			let mut output = self.bos.filter(|_| self.add_bos).into_iter().collect::<Vec<_>>();
			let (mut start, mut at) = (0, 0);
			while at < text.len() {
				if let Some((added, id)) = self.added.iter().find(|(added, _)| text[at..].starts_with(added.as_str())) {
					self.encode_plain(&text[start..at], &mut output);
					output.push(*id);
					at += added.len();
					start = at;
				} else {
					at += text[at..].chars().next().map_or(1, char::len_utf8);
				}
			}
			self.encode_plain(&text[start..], &mut output);
			output.extend(self.eos.filter(|_| self.add_eos));
			output
		}
		/// The rank and id of the piece that joins `left` to `right`, the lowest
		/// rank merging first. A merge never spells an added token.
		fn merge(&self, left: u32, right: u32) -> Option<(u32, u32)> {
			let (rank, merged) = match &self.ranks {
				Ranks::Merges(merges) => *merges.get(&(left, right))?,
				Ranks::Scores(ranks) => {
					let merged = *self.ids.get(&format!("{}{}", self.tokens[left as usize], self.tokens[right as usize]))?;
					(ranks[merged as usize], merged)
				}
			};
			(!self.is_added[merged as usize]).then_some((rank, merged))
		}
		fn encode_plain(&self, text: &str, output: &mut Vec<u32>) {
			for word in self.family.split(text) {
				let mut symbols = word.bytes().map(|byte| self.bytes[usize::from(byte)]).collect::<Vec<_>>();
				// Llama 3 vocabularies take a whole word that is already a token as is.
				if self.family == Family::Llama3
					&& let Some(id) = self.ids.get(&word.bytes().map(|byte| self.byte_of_token(byte)).collect::<String>()).filter(|id| !self.is_added[**id as usize])
				{
					output.push(*id);
					continue;
				}
				while let Some((index, merged)) = (0..symbols.len().saturating_sub(1))
					.filter_map(|index| self.merge(symbols[index], symbols[index + 1]).map(|(rank, merged)| (rank, index, merged)))
					.min()
					.map(|(_, index, merged)| (index, merged))
				{
					symbols[index] = merged;
					symbols.remove(index + 1);
				}
				output.extend(symbols);
			}
		}
		fn byte_of_token(&self, byte: u8) -> char {
			self.tokens[self.bytes[usize::from(byte)] as usize].chars().next().unwrap()
		}
		/// The text the ids spell. Bytes split across tokens rejoin before the
		/// UTF-8 decode, and a token outside the byte alphabet keeps its own text.
		pub fn decode(&self, ids: &[u32]) -> String {
			let mut bytes = Vec::new();
			for id in ids {
				for symbol in self.tokens[*id as usize].chars() {
					match self.byte_of.get(&symbol) {
						Some(byte) => bytes.push(*byte),
						None => bytes.extend(symbol.to_string().as_bytes()),
					}
				}
			}
			String::from_utf8_lossy(&bytes).into_owned()
		}
		/// `tokenizer.chat_template` rendered for `messages`, each a role and its
		/// content. `generation` sets `add_generation_prompt`, which the template
		/// reads to open the reply. Unsupported template syntax is an error.
		pub fn chat(&self, messages: &[(&str, &str)], generation: bool) -> String {
			self.prompt(messages, generation).unwrap_or_else(|error| panic!("{error}"))
		}
		fn prompt(&self, messages: &[(&str, &str)], generation: bool) -> Result<String> {
			let template = self.template.as_deref().ok_or_else(|| RecipeError::new("tokenizer.chat_template is absent"))?;
			let token = |id: Option<u32>| id.map_or(String::new(), |id| self.decode(&[id]));
			let scope = Scope { messages, generation, bos: token(self.bos), eos: token(self.eos), bound: None };
			let (pieces, mut out, mut at) = (parse(template)?, String::new(), 0);
			render(&pieces, &mut at, &mut out, &scope, true, &[])?;
			Ok(out)
		}
	}

	/// One piece of a chat template: literal text, a `{{ ... }}` substitution, or
	/// a `{% ... %}` statement.
	enum Piece {
		Text(String),
		Write(String),
		Tag(String),
	}

	fn push_text(pieces: &mut Vec<Piece>, text: &str, start: bool, end: bool) {
		let text = if start { text.trim_start() } else { text };
		let text = if end { text.trim_end() } else { text };
		if !text.is_empty() {
			pieces.push(Piece::Text(text.to_owned()));
		}
	}

	/// Splits a chat template into its pieces, applying the `-` controls that trim
	/// the whitespace around a tag.
	fn parse(template: &str) -> Result<Vec<Piece>> {
		let (mut pieces, mut rest, mut start) = (Vec::new(), template, false);
		while let Some(open) = rest.as_bytes().windows(2).position(|pair| pair == b"{{" || pair == b"{%") {
			let write = rest.as_bytes()[open + 1] == b'{';
			let close = if write { "}}" } else { "%}" };
			let body = &rest[open + 2..];
			let end = body.find(close).ok_or_else(|| RecipeError::new(format!("chat template leaves a {} tag unclosed", if write { "{{" } else { "{%" })))?;
			push_text(&mut pieces, &rest[..open], start, body.starts_with('-'));
			let source = body[..end].trim_matches('-').trim().to_owned();
			pieces.push(if write { Piece::Write(source) } else { Piece::Tag(source) });
			start = body[..end].ends_with('-');
			rest = &body[end + close.len()..];
		}
		push_text(&mut pieces, rest, start, false);
		Ok(pieces)
	}

	/// A value a chat template can name.
	enum Value<'a> {
		Text(String),
		Flag(bool),
		Messages(&'a [(&'a str, &'a str)]),
		Message(&'a (&'a str, &'a str)),
	}
	impl Value<'_> {
		fn truth(&self) -> bool {
			match self {
				Self::Text(text) => !text.is_empty(),
				Self::Flag(flag) => *flag,
				Self::Messages(messages) => !messages.is_empty(),
				Self::Message(_) => true,
			}
		}
		fn text(&self) -> Result<&str> {
			match self {
				Self::Text(text) => Ok(text),
				_ => Err(RecipeError::new("a chat template compares or writes a value that is not text")),
			}
		}
	}

	/// The names a chat template resolves: the conversation, the generation flag,
	/// the sequence tokens, and the message a `for` statement binds.
	#[derive(Clone)]
	struct Scope<'a> {
		messages: &'a [(&'a str, &'a str)],
		generation: bool,
		bos: String,
		eos: String,
		bound: Option<(String, &'a (&'a str, &'a str))>,
	}

	/// A literal, a name, or a name read through `['key']` and `.key` accessors.
	fn value<'a>(source: &str, scope: &Scope<'a>) -> Result<Value<'a>> {
		let source = source.trim();
		let quoted = |mark: char| source.strip_prefix(mark).and_then(|rest| rest.strip_suffix(mark));
		if let Some(text) = quoted('\'').or_else(|| quoted('"')) {
			return Ok(Value::Text(text.to_owned()));
		}
		let head = source.find(['[', '.']).unwrap_or(source.len());
		let name = source[..head].trim();
		let mut value = match name {
			"messages" => Value::Messages(scope.messages),
			"add_generation_prompt" => Value::Flag(scope.generation),
			"bos_token" => Value::Text(scope.bos.clone()),
			"eos_token" => Value::Text(scope.eos.clone()),
			_ => match &scope.bound {
				Some((bound, message)) if bound.as_str() == name => Value::Message(message),
				_ => return Err(RecipeError::new(format!("chat template names {name:?}, which this tokenizer does not define"))),
			},
		};
		let mut rest = &source[head..];
		while !rest.is_empty() {
			let (key, next) = match rest.strip_prefix('[') {
				Some(body) => {
					let end = body.find(']').ok_or_else(|| RecipeError::new("chat template leaves a [ accessor unclosed"))?;
					(body[..end].trim().trim_matches(['\'', '"']), &body[end + 1..])
				}
				None => {
					let body = &rest[1..];
					let end = body.find(['[', '.']).unwrap_or(body.len());
					(&body[..end], &body[end..])
				}
			};
			value = match (&value, key) {
				(Value::Message(message), "role") => Value::Text(message.0.to_owned()),
				(Value::Message(message), "content") => Value::Text(message.1.to_owned()),
				_ => return Err(RecipeError::new(format!("chat template reads {key:?} from a value that has no such field"))),
			};
			rest = next;
		}
		Ok(value)
	}

	/// `not a`, `a == b`, `a != b`, or the truth of one value.
	fn condition(source: &str, scope: &Scope) -> Result<bool> {
		if let Some(rest) = source.trim().strip_prefix("not ") {
			return Ok(!condition(rest, scope)?);
		}
		for (operator, equal) in [("==", true), ("!=", false)] {
			if let Some((left, right)) = source.split_once(operator) {
				return Ok((value(left, scope)?.text()? == value(right, scope)?.text()?) == equal);
			}
		}
		Ok(value(source, scope)?.truth())
	}

	/// Renders the pieces from `at` until one of `stop`, returning the tag that
	/// ended the block. A branch that is not taken still parses, with `emit` off,
	/// so the walk lands past its `endif` or `endfor`.
	fn render(pieces: &[Piece], at: &mut usize, out: &mut String, scope: &Scope, emit: bool, stop: &[&str]) -> Result<String> {
		while *at < pieces.len() {
			let piece = &pieces[*at];
			*at += 1;
			match piece {
				Piece::Text(text) if emit => out.push_str(text),
				Piece::Write(source) if emit => out.push_str(value(source, scope)?.text()?),
				Piece::Text(_) | Piece::Write(_) => {}
				Piece::Tag(source) => {
					let (head, rest) = source.split_once(' ').unwrap_or((source.as_str(), ""));
					if stop.contains(&head) {
						return Ok(source.clone());
					}
					match head {
						"if" => {
							// Each branch renders in turn; only the first whose condition
							// holds writes, and the others parse with `emit` off.
							let (mut branch, mut test, mut taken) = (head.to_owned(), rest.to_owned(), false);
							loop {
								let live = emit && !taken && (branch == "else" || condition(&test, scope)?);
								taken |= live;
								let ended = render(pieces, at, out, scope, live, &["elif", "else", "endif"])?;
								let (next, tail) = ended.split_once(' ').unwrap_or((ended.as_str(), ""));
								if next == "endif" {
									break;
								}
								(branch, test) = (next.to_owned(), tail.to_owned());
							}
						}
						"for" => {
							let (name, iterable) = rest.split_once(" in ").ok_or_else(|| RecipeError::new(format!("chat template for statement {rest:?} has no \"in\"")))?;
							let (name, body) = (name.trim().to_owned(), *at);
							let items = match emit {
								true => match value(iterable, scope)? {
									Value::Messages(messages) => messages,
									_ => return Err(RecipeError::new(format!("chat template iterates {:?}, which is not a list", iterable.trim()))),
								},
								false => &[][..],
							};
							for message in items {
								*at = body;
								let inner = Scope { bound: Some((name.clone(), message)), ..scope.clone() };
								render(pieces, at, out, &inner, true, &["endfor"])?;
							}
							if items.is_empty() {
								*at = body;
								render(pieces, at, out, scope, false, &["endfor"])?;
							}
						}
						_ => return Err(RecipeError::new(format!("chat template statement {head:?} is not supported"))),
					}
				}
			}
		}
		match stop.is_empty() {
			true => Ok(String::new()),
			false => Err(RecipeError::new(format!("chat template ends before its {}", stop.join(" or ")))),
		}
	}
}
pub use tokenizer::Tokenizer;
mod ngram {
	//! N-gram embeddings gathered on the host from a mapped GGUF table. Each token
	//! hashes itself with its predecessors into one row per head, the rows decode
	//! from their own bytes, and only those rows of a table far larger than device
	//! memory are ever read. The `ngram.*` keys describe a seeded hash into equal
	//! ranges; the `<arch>.ple.*` keys of a per-layer embedding describe the
	//! reference multiply-xor hash into named ranges. Both fill one description.
	use super::*;

	/// A token outside the context of a seeded table, before the sequence start or an end id.
	const ABSENT: u32 = u32::MAX;

	/// How one token addresses its rows. The context is the token and its `ngram - 1`
	/// predecessors, cut at an end id: a predecessor at or behind one, or before the
	/// sequence start, reads as `absent`. Every order `n` in `2..=ngram` hashes the
	/// leading `n` ids of the context for `per_order` heads, and head `h` lands in
	/// `offsets[h] + hash % vocabularies[h]`. The reference fold xors each id times
	/// its `multipliers` entry; a table with `seeds` folds the ids into its head's
	/// seed instead. `image` is the id a position without a token hashes as.
	#[derive(Clone, Debug, PartialEq, Eq)]
	pub struct RowHash {
		pub(crate) ngram: usize,
		pub(crate) per_order: usize,
		pub(crate) multipliers: Vec<u64>,
		pub(crate) seeds: Vec<u64>,
		pub(crate) offsets: Vec<u64>,
		pub(crate) vocabularies: Vec<u64>,
		pub(crate) ends: Vec<u32>,
		pub(crate) absent: u64,
		pub(crate) image: u64,
	}
	impl RowHash {
		/// Rows one token reads: `per_order` heads for every order.
		pub fn heads(&self) -> usize {
			self.ngram.saturating_sub(1) * self.per_order
		}
		/// Rows the highest head range reaches.
		pub fn rows(&self) -> usize {
			self.offsets.iter().zip(&self.vocabularies).map(|(offset, vocabulary)| offset.saturating_add(*vocabulary)).max().unwrap_or(0) as usize
		}
		pub(crate) fn validate(&self) -> Result<()> {
			let heads = self.heads();
			require(self.ngram >= 2 && self.per_order != 0, format!("a row hash of {} ids with {} heads per order addresses no row", self.ngram, self.per_order))?;
			require(self.seeds.is_empty() || self.seeds.len() == heads, format!("row hash holds {} seeds for {heads} heads", self.seeds.len()))?;
			require(!self.seeds.is_empty() || self.multipliers.len() == self.ngram, format!("row hash holds {} multipliers for {} ids", self.multipliers.len(), self.ngram))?;
			require(
				self.offsets.len() == heads && self.vocabularies.len() == heads,
				format!("row hash holds {} offsets and {} vocabularies for {heads} heads", self.offsets.len(), self.vocabularies.len()),
			)?;
			require(self.vocabularies.iter().all(|vocabulary| *vocabulary != 0), "row hash holds an empty head vocabulary")?;
			Ok(())
		}
		/// The row each head addresses for the token at `position`: the heads of
		/// order two first, then of every higher order.
		pub fn rows_at(&self, ids: &[u32], position: usize) -> Vec<usize> {
			let context = (0..self.ngram)
				.map(|back| if back <= position && !ids[position - back..position].iter().any(|id| self.ends.contains(id)) { u64::from(ids[position - back]) } else { self.absent })
				.collect::<Vec<_>>();
			let mut rows = Vec::with_capacity(self.heads());
			for order in 2..=self.ngram {
				let mixed = context[..order].iter().zip(&self.multipliers).fold(0, |mixed, (id, multiplier)| mixed ^ id.wrapping_mul(*multiplier));
				for head in (order - 2) * self.per_order..(order - 1) * self.per_order {
					let hash = match self.seeds.get(head) {
						Some(seed) => context[..order].iter().fold(*seed, |hash, id| {
							let mixed = (hash ^ id).wrapping_mul(0x9E37_79B9_7F4A_7C15);
							mixed ^ (mixed >> 32)
						}),
						None => mixed,
					};
					rows.push((self.offsets[head] + hash % self.vocabularies[head]) as usize);
				}
			}
			rows
		}
		/// The description as the words a graph carries beside its lookup node, so
		/// a tape addresses rows without the table's metadata: the counts, then every
		/// 64-bit value as its high and low halves, exactly.
		pub(crate) fn words(&self) -> Vec<f64> {
			let mut words = vec![self.ngram as f64, self.per_order as f64, self.multipliers.len() as f64, self.seeds.len() as f64, self.offsets.len() as f64, self.ends.len() as f64];
			for value in [self.absent, self.image].iter().chain(&self.multipliers).chain(&self.seeds).chain(&self.offsets).chain(&self.vocabularies) {
				words.extend([(value >> 32) as f64, (value & 0xFFFF_FFFF) as f64]);
			}
			words.extend(self.ends.iter().map(|end| f64::from(*end)));
			words
		}
		pub(crate) fn from_words(words: &[f64]) -> Result<Self> {
			let count = |index: usize| {
				words.get(index).copied().filter(|value| value.fract() == 0.0 && *value >= 0.0).map(|value| value as usize).ok_or_else(|| RecipeError::new("row hash words are invalid"))
			};
			let (ngram, per_order) = (count(0)?, count(1)?);
			let (multipliers, seeds, heads, ends) = (count(2)?, count(3)?, count(4)?, count(5)?);
			let mut at = 6;
			let mut wide = |count: usize| -> Result<Vec<u64>> {
				let values = words.get(at..at + 2 * count).ok_or_else(|| RecipeError::new("row hash words are incomplete"))?;
				at += 2 * count;
				Ok(values.chunks_exact(2).map(|halves| (halves[0] as u64) << 32 | halves[1] as u64).collect())
			};
			let fixed = wide(2)?;
			let (multipliers, seeds, offsets, vocabularies) = (wide(multipliers)?, wide(seeds)?, wide(heads)?, wide(heads)?);
			let ends = words.get(at..at + ends).ok_or_else(|| RecipeError::new("row hash words are incomplete"))?.iter().map(|end| *end as u32).collect();
			let hash = Self { ngram, per_order, multipliers, seeds, offsets, vocabularies, ends, absent: fixed[0], image: fixed[1] };
			hash.validate()?;
			Ok(hash)
		}
		/// The description as the fields of a saved model block.
		pub(crate) fn text(&self) -> String {
			let list = |values: &[u64]| values.iter().map(u64::to_string).collect::<Vec<_>>().join(":");
			let ends = self.ends.iter().map(u32::to_string).collect::<Vec<_>>().join(":");
			format!(
				"{},{},{},{},{},{},{},{},{ends}",
				self.ngram,
				self.per_order,
				self.absent,
				self.image,
				list(&self.multipliers),
				list(&self.seeds),
				list(&self.offsets),
				list(&self.vocabularies)
			)
		}
		pub(crate) fn parse<'f>(fields: &mut impl Iterator<Item = &'f str>) -> Result<Self> {
			let mut next = |role: &str| fields.next().ok_or_else(|| RecipeError::new(format!("row hash {role} is absent")));
			let value = |text: &str, role: &str| text.parse::<u64>().map_err(|error| RecipeError::new(format!("invalid row hash {role}: {error}")));
			let list = |text: &str, role: &str| text.split(':').filter(|item| !item.is_empty()).map(|item| value(item, role)).collect::<Result<Vec<_>>>();
			let (ngram, per_order) = (value(next("ids")?, "ids")? as usize, value(next("heads per order")?, "heads per order")? as usize);
			let (absent, image) = (value(next("absent id")?, "absent id")?, value(next("image id")?, "image id")?);
			let multipliers = list(next("multipliers")?, "multiplier")?;
			let seeds = list(next("seeds")?, "seed")?;
			let offsets = list(next("offsets")?, "offset")?;
			let vocabularies = list(next("vocabularies")?, "vocabulary")?;
			let ends = list(next("end ids")?, "end id")?.into_iter().map(|end| u32::try_from(end).map_err(|_| RecipeError::new("row hash end id exceeds u32"))).collect::<Result<Vec<_>>>()?;
			let hash = Self { ngram, per_order, multipliers, seeds, offsets, vocabularies, ends, absent, image };
			hash.validate()?;
			Ok(hash)
		}
	}

	/// Row `index` of a stored `[width, rows]` table, decoded from that row's own
	/// bytes: a raw table holds little-endian f64 values, a quantized one its blocks.
	pub(crate) fn table_row(table: &StoredWeight, width: usize, index: usize) -> Result<Vec<f64>> {
		require(width != 0 && index.checked_mul(width).is_some_and(|start| start + width <= table.count), format!("table of {} values has no row {index} of {width}", table.count))?;
		if table.format.0 == 0 {
			let bytes = table.bytes.slice(index * width * size_of::<f64>(), width * size_of::<f64>())?;
			return Ok(bytes.chunks_exact(size_of::<f64>()).map(|value| f64::from_le_bytes(value.try_into().unwrap())).collect());
		}
		let spec = table.format.spec().ok_or_else(|| table.format.unavailable())?;
		require(width % spec.block == 0, format!("table rows of {width} are not whole blocks of {}", spec.block))?;
		let row = width / spec.block * spec.stride;
		table.format.decompress(&table.bytes.slice(index * row, row)?, &table.codebook, width)
	}

	pub struct Ngram<'a> {
		model: &'a Gguf,
		table: GgufTensor,
		taps: Vec<f64>,
		hash: RowHash,
		layer: usize,
		kernel: usize,
		width: usize,
		rows: usize,
	}

	impl<'a> Ngram<'a> {
		pub(super) fn new(model: &'a Gguf) -> Result<Self> {
			let integer = |key: &str| model.value(key).and_then(GgufValue::integer);
			let named = |key: &str, fallback| model.value(key).and_then(GgufValue::text).unwrap_or(fallback);
			let architecture = model.value("general.architecture").and_then(GgufValue::text).unwrap_or("");
			let ple = |suffix: &str| format!("{architecture}.ple.{suffix}");
			let (name, hash, layer, kernel) = if let Some(heads) = integer("ngram.heads") {
				// The seeded form: every head owns an equal range of the table.
				let heads = heads as usize;
				let name = named("ngram.table", "ngram.table");
				let table = model.tensor(name).ok_or_else(|| RecipeError::new(format!("n-gram table {name:?} is absent")))?;
				require(table.shape.len() == 2 && heads != 0, format!("n-gram table {name:?} must be [width, rows] with at least one head"))?;
				let rows = table.shape[1] as usize;
				require(rows % (2 * heads) == 0, format!("n-gram table rows {rows} do not split into {} head ranges", 2 * heads))?;
				let seeds = match model.value("ngram.seeds") {
					Some(GgufValue::Array(items)) => {
						items.iter().map(|item| item.integer().ok_or_else(|| RecipeError::new("ngram.seeds holds a non-integer"))).collect::<Result<Vec<_>>>()?
					}
					_ => (1..=2 * heads as u64).collect(),
				};
				require(seeds.len() == 2 * heads, format!("ngram.seeds holds {} seeds for {} heads", seeds.len(), 2 * heads))?;
				let range = (rows / (2 * heads)) as u64;
				let hash = RowHash {
					ngram: 3,
					per_order: heads,
					multipliers: Vec::new(),
					seeds,
					offsets: (0..2 * heads as u64).map(|head| head * range).collect(),
					vocabularies: vec![range; 2 * heads],
					ends: integer("tokenizer.ggml.eos_token_id").map(|id| id as u32).into_iter().collect(),
					absent: u64::from(ABSENT),
					image: u64::from(ABSENT),
				};
				(name, hash, integer("ngram.layer").unwrap_or(0) as usize, integer("ngram.kernel").unwrap_or(1) as usize)
			} else if model.value(&ple("ngram_size")).is_some() {
				// The reference form: the metadata names every head's range and the
				// multipliers of the fold, and the end id stands in for a missing id.
				let ngram = model.integer_at(&ple("ngram_size"))?;
				let per_order = model.integer_at(&ple("heads_per_ngram"))?;
				let eos = model.integer_at(&ple("eos_token_id"))?;
				let hash = RowHash {
					ngram,
					per_order,
					multipliers: model.integers_at(&ple("layer_multipliers"))?,
					seeds: Vec::new(),
					offsets: model.integers_at(&ple("head_offsets"))?,
					vocabularies: model.integers_at(&ple("head_vocab_sizes"))?,
					ends: vec![u32::try_from(eos).map_err(|_| RecipeError::new(format!("{} exceeds u32", ple("eos_token_id"))))?],
					absent: eos as u64,
					image: integer(&ple("image_token_id")).unwrap_or(eos as u64),
				};
				let layer = model.value(&ple("layers")).map(|_| model.indices_at(&ple("layers"))).transpose()?.and_then(|layers| layers.first().copied()).unwrap_or(0);
				let kernel = model.value(&ple("conv_kernel")).map(|_| model.integer_at(&ple("conv_kernel"))).transpose()?.unwrap_or(1);
				("per_layer_token_embd.weight", hash, layer, kernel)
			} else {
				return Err(RecipeError::new(format!("ngram.heads and {} are absent", ple("ngram_size"))));
			};
			hash.validate()?;
			let table = model.tensor(name).ok_or_else(|| RecipeError::new(format!("n-gram table {name:?} is absent")))?.clone();
			require(table.shape.len() == 2, format!("n-gram table {name:?} must be [width, rows]"))?;
			let (width, rows) = (table.shape[0] as usize, table.shape[1] as usize);
			require(width != 0 && hash.rows() <= rows, format!("n-gram table {name:?} holds {rows} rows of {width}, the hash reaches row {}", hash.rows()))?;
			require(kernel != 0, "n-gram convolution kernel must be positive")?;
			let taps = match model.tensor(named("ngram.conv", "ngram.conv")) {
				Some(tensor) => model.values(tensor)?,
				None => Vec::new(),
			};
			Ok(Self { model, table, taps, hash, layer, kernel, width, rows })
		}
		/// Heads per n-gram order.
		pub fn heads(&self) -> usize {
			self.hash.per_order
		}
		/// Values per token: one row of the table width per head of every order.
		pub fn width(&self) -> usize {
			self.hash.heads() * self.width
		}
		/// The block the gathered vector is added to the stream before, or the layer
		/// a per-layer embedding block sits at.
		pub fn layer(&self) -> usize {
			self.layer
		}
		/// The depthwise convolution kernel of a per-layer embedding block.
		pub fn kernel(&self) -> usize {
			self.kernel
		}
		/// How the rows are addressed.
		pub fn hash(&self) -> &RowHash {
			&self.hash
		}
		/// The mapped table bytes one token reads, over every convolved position.
		pub fn bytes(&self) -> usize {
			self.hash.heads() * (self.table.bytes / self.rows) * self.taps.len().max(1)
		}
		/// The mapped table: its name, its rows, and the bytes of one row.
		pub fn table(&self) -> (&str, usize, usize) {
			(&self.table.name, self.rows, self.table.bytes / self.rows)
		}
		/// The row each head addresses for the token at `position`: the heads
		/// hashing the bigram first, then those hashing the trigram, and so on. A
		/// token before the sequence start or behind an end id is absent from the
		/// context.
		pub fn rows(&self, ids: &[u32], position: usize) -> Vec<usize> {
			self.hash.rows_at(ids, position)
		}
		/// The per-layer embedding block this table feeds: `heads` rows of `width`
		/// per token, a `kernel`-wide depthwise convolution dilated by the n-gram
		/// size, and the table's row count.
		pub(super) fn block(&self) -> PleBlock {
			PleBlock { heads: self.hash.heads(), width: self.width, rows: self.rows, kernel: self.kernel, dilation: self.hash.ngram, hash: self.hash.clone() }
		}
		/// The rows one token addresses, concatenated, decoded from their own bytes.
		fn gather(&self, ids: &[u32], position: usize) -> Result<Vec<f64>> {
			self.rows(ids, position).into_iter().try_fold(Vec::with_capacity(self.width()), |mut values, row| {
				values.extend(self.model.row(&self.table, row)?);
				Ok(values)
			})
		}
		/// The addressed rows of every token, concatenated: `ids.len() * width()` values.
		pub fn lookup(&self, ids: &[u32]) -> Result<Vec<f64>> {
			(0..ids.len()).try_fold(Vec::with_capacity(ids.len() * self.width()), |mut values, position| {
				values.extend(self.gather(ids, position)?);
				Ok(values)
			})
		}
		/// The vector the last token of `ids` adds to the stream: its gathered rows,
		/// or the `ngram.conv` taps applied across the positions ending at it.
		pub fn inject(&self, ids: &[u32]) -> Result<Vec<f64>> {
			let last = ids.len().checked_sub(1).ok_or_else(|| RecipeError::new("an n-gram injection reads at least one token"))?;
			if self.taps.is_empty() {
				return self.gather(ids, last);
			}
			let mut injected = vec![0.0; self.width()];
			for (tap, position) in self.taps.iter().rev().zip((0..=last).rev()) {
				for (value, row) in injected.iter_mut().zip(self.gather(ids, position)?) {
					*value += tap * row;
				}
			}
			Ok(injected)
		}
		/// Inference with the gathered vector added to the stream: the blocks before
		/// `ngram.layer` run on the selected device, the gather and the addition on
		/// the host that holds the table, and the blocks from it on the device again.
		pub fn infer(&self, path: impl AsRef<Path>, input: &[f64], ids: &[u32]) -> Vec<f64> {
			self.decode(path.as_ref(), input, ids).unwrap_or_else(|error| panic!("{error}"))
		}
		fn decode(&self, path: &Path, input: &[f64], ids: &[u32]) -> Result<Vec<f64>> {
			let path = resolve_path(path)?;
			let device = selected_gpu()?;
			let injected = self.inject(ids)?;
			bundle::run_infer(&path, input, |stored, samples| {
				let graph = materialize_saved_graph(stored, samples, device, Config::load()?)?;
				let (head, tail) = split_at_block(&graph, self.layer)?;
				let mut statistics = 0;
				let mut stream = match &head {
					Some(head) => forward_part(head, samples, samples, device, stored, &mut statistics)?,
					None => samples.to_vec(),
				};
				require(stream.len() == injected.len(), format!("block {} takes {} values, the n-gram table gathers {}", self.layer, stream.len(), injected.len()))?;
				for (value, added) in stream.iter_mut().zip(&injected) {
					*value += added;
				}
				forward_part(&tail, &stream, samples, device, stored, &mut statistics)
			})
		}
	}

	impl Gguf {
		/// The hashed row table the metadata describes, gathered on the host: the
		/// `ngram.*` keys or the `<arch>.ple.*` keys of a per-layer embedding.
		pub fn ngram(&self) -> Ngram<'_> {
			Ngram::new(self).unwrap_or_else(|error| panic!("{error}"))
		}
	}
}
pub use ngram::{Ngram, RowHash};
mod bundle {
	use super::*;
	use std::{collections::BTreeMap, io::Write as _, str::FromStr};
	const BUNDLE_HEADER: &str = "recipe-native-model";

	fn hex(value: &[u8]) -> String {
		value.iter().map(|byte| format!("{byte:02x}")).collect()
	}
	fn unhex(value: &str, role: &str) -> Result<Vec<u8>> {
		require(value.len() % 2 == 0, format!("{role} has an odd hexadecimal width"))?;
		(0..value.len()).step_by(2).map(|index| u8::from_str_radix(&value[index..index + 2], 16).map_err(|error| RecipeError::new(format!("invalid {role}: {error}")))).collect()
	}
	fn text(value: &str) -> String {
		hex(value.as_bytes())
	}
	fn untext(value: &str, role: &str) -> Result<String> {
		String::from_utf8(unhex(value, role)?).map_err(|error| RecipeError::new(format!("invalid {role}: {error}")))
	}
	fn bool_value(value: &str, role: &str) -> Result<bool> {
		match value {
			"0" => Ok(false),
			"1" => Ok(true),
			_ => Err(RecipeError::new(format!("invalid {role}"))),
		}
	}

	fn residual_text(value: &Residual) -> String {
		match value {
			Residual::Layer(width) => format!("layer,{width}"),
			Residual::Conv(filters, kernel) => format!("conv,{filters},{kernel}"),
			Residual::Activation(activation) => format!("activation,{}", *activation as u8),
		}
	}
	fn residual(value: &str) -> Result<Residual> {
		let mut fields = value.split(',');
		match fields.next().unwrap_or("") {
			"layer" => Ok(Residual::Layer(value_at(fields.next(), "residual layer width")?)),
			"conv" => Ok(Residual::Conv(value_at(fields.next(), "residual filters")?, value_at(fields.next(), "residual kernel")?)),
			"activation" => Ok(Residual::Activation(activation(value_at(fields.next(), "residual activation")?)?)),
			_ => Err(RecipeError::new(format!("invalid residual {value:?}"))),
		}
	}
	fn value_at<T: FromStr>(value: Option<&str>, role: &str) -> Result<T>
	where
		T::Err: fmt::Display,
	{
		value.ok_or_else(|| RecipeError::new(format!("{role} is absent")))?.parse().map_err(|error| RecipeError::new(format!("invalid {role}: {error}")))
	}
	fn scoring(value: u8) -> Result<Scoring> {
		match value {
			0 => Ok(Scoring::Softmax),
			1 => Ok(Scoring::Sigmoid),
			_ => Err(RecipeError::new(format!("invalid scoring {value}"))),
		}
	}
	fn activation(value: u8) -> Result<Activation> {
		match value {
			0 => Ok(Activation::Linear),
			1 => Ok(Activation::Cos),
			2 => Ok(Activation::Exp),
			3 => Ok(Activation::Log),
			4 => Ok(Activation::Ln),
			5 => Ok(Activation::Huber),
			6 => Ok(Activation::Tan),
			7 => Ok(Activation::Relu),
			8 => Ok(Activation::Leak),
			9 => Ok(Activation::Sigmoid),
			10 => Ok(Activation::Tanh),
			11 => Ok(Activation::Selu),
			12 => Ok(Activation::Gelu),
			13 => Ok(Activation::Silu),
			14 => Ok(Activation::Elu),
			15 => Ok(Activation::Prelu),
			_ => Err(RecipeError::new(format!("invalid activation {value}"))),
		}
	}
	fn operation_text(operation: &Operation) -> String {
		match operation {
			Operation::Layer(width) => format!("layer,{width}"),
			Operation::Conv(filters, kernel) => format!("conv,{filters},{kernel}"),
			Operation::Pool(size) => format!("pool,{size}"),
			Operation::Estimator(estimator) => format!("estimator,{},{}", estimator.name, estimator.param),
			Operation::Attention(attention) => {
				let (dims, base) = attention.rope.map_or((0, 0.0), |(dims, base)| (dims, f64::from_bits(base)));
				let index = attention.index.unwrap_or(Indexer::NONE);
				let (score_normalization, score_dims) = index.score.map_or((None, 0), |(normalization, dims)| (Some(normalization), dims));
				format!(
					"attn,{},{},{dims},{base},{},{},{},{},{},{},{},{},{score_dims}",
					attention.heads,
					attention.kv,
					index.heads,
					index.width,
					index.block,
					index.keep,
					u8::from(attention.gate),
					attention.width,
					index.tokens,
					normalization_text(score_normalization)
				)
			}
			Operation::Rnn(width) => format!("rnn,{width}"),
			Operation::Gru(width) => format!("gru,{width}"),
			Operation::Lstm(width) => format!("lstm,{width}"),
			Operation::Residual(parts) => format!("residual,{}", parts.iter().map(residual_text).collect::<Vec<_>>().join(";")),
			Operation::Moe(experts, top_k, hidden, activation, scoring, renormalize, shared) => {
				format!("moe,{experts},{top_k},{hidden},{},{},{},{}", *activation as u8, *scoring as u8, u8::from(*renormalize), u8::from(*shared))
			}
			Operation::Hyper(lanes, rank, blocks) => format!("hyper,{lanes},{rank},{}", blocks.iter().map(block_text).map(|block| text(&block)).collect::<Vec<_>>().join(";")),
			Operation::Perceptron(width) => format!("perc,{width}"),
			Operation::Embed(vocabulary, width) => format!("embed,{vocabulary},{width}"),
			Operation::Dconv(kernel) => format!("dconv,{kernel}"),
			Operation::Delta(delta) => format!("delta,{},{},{},{},{},{}", delta.heads, delta.kernel, delta.key_heads, delta.key_width, delta.value_width, delta.output),
			Operation::Ple(ple) => format!("ple,{},{},{},{},{},{}", ple.heads, ple.width, ple.rows, ple.kernel, ple.dilation, ple.hash.text()),
			Operation::Norm => "norm".to_owned(),
			Operation::Glu(hidden, activation) => format!("glu,{hidden},{}", *activation as u8),
		}
	}
	fn estimator(name: &str, param: usize) -> Result<Estimator> {
		let result = match name {
			"kmeans" => Estimator { fit: fit_kmeans, validate: cluster_estimator, param, name: "kmeans" },
			"knn" => Estimator { fit: fit_knn, validate: neighbor_estimator, param, name: "knn" },
			"svm" => Estimator { fit: fit_svm, validate: valid_estimator, param, name: "svm" },
			"forest" => Estimator { fit: fit_forest, validate: positive_estimator, param, name: "forest" },
			"bayes" => Estimator { fit: fit_bayes, validate: valid_estimator, param, name: "bayes" },
			"cbst" => Estimator { fit: fit_catboost, validate: valid_estimator, param, name: "cbst" },
			"xgbst" => Estimator { fit: fit_xgboost, validate: valid_estimator, param, name: "xgbst" },
			"lgbm" => Estimator { fit: fit_lightgbm, validate: valid_estimator, param, name: "lgbm" },
			_ => return Err(RecipeError::new(format!("invalid estimator {name:?}"))),
		};
		Ok(result)
	}
	fn operation(value: &str) -> Result<Operation> {
		let (name, rest) = value.split_once(',').unwrap_or((value, ""));
		let mut fields = rest.split(',');
		match name {
			"layer" => Ok(Operation::Layer(value_at(Some(rest), "layer width")?)),
			"conv" => Ok(Operation::Conv(value_at(fields.next(), "convolution filters")?, value_at(fields.next(), "convolution kernel")?)),
			"pool" => Ok(Operation::Pool(value_at(Some(rest), "pool size")?)),
			"estimator" => Ok(Operation::Estimator(estimator(fields.next().unwrap_or(""), value_at(fields.next(), "estimator parameter")?)?)),
			"attn" => {
				let heads = value_at(fields.next(), "attention heads")?;
				let kv = value_at(fields.next(), "attention key-value heads")?;
				let dims = value_at::<usize>(fields.next(), "rotary dimensions")?;
				let base = value_at::<f64>(fields.next(), "rotary base")?;
				let mut index = Indexer {
					heads: value_at(fields.next(), "indexer heads")?,
					width: value_at(fields.next(), "indexer width")?,
					block: value_at(fields.next(), "indexer block")?,
					keep: value_at(fields.next(), "indexer blocks kept")?,
					..Indexer::NONE
				};
				let gate = value_at::<u8>(fields.next(), "attention gate")? != 0;
				let width = fields.next().map(|field| value_at(Some(field), "attention head width")).transpose()?.unwrap_or(0);
				// The token budget and the scoring geometry follow the head width, so a
				// model saved without them reads as a block budget over raw planes.
				index.tokens = fields.next().map(|field| value_at(Some(field), "indexer token budget")).transpose()?.unwrap_or(0);
				let score_normalization = fields.next().map(|field| normalization(Some(field), "indexer scoring normalization")).transpose()?.flatten();
				let score_dims = fields.next().map(|field| value_at(Some(field), "indexer rotary dimensions")).transpose()?.unwrap_or(0);
				index.score = score_normalization.map(|normalization| (normalization, score_dims));
				Ok(Operation::Attention(AttentionBlock { heads, kv, width, rope: (dims != 0).then_some((dims, base.to_bits())), index: (index.block != 0).then_some(index), gate }))
			}
			"rnn" => Ok(Operation::Rnn(value_at(Some(rest), "RNN width")?)),
			"gru" => Ok(Operation::Gru(value_at(Some(rest), "GRU width")?)),
			"lstm" => Ok(Operation::Lstm(value_at(Some(rest), "LSTM width")?)),
			"residual" => Ok(Operation::Residual(if rest.is_empty() { Vec::new() } else { rest.split(';').map(residual).collect::<Result<Vec<_>>>()? })),
			"moe" => Ok(Operation::Moe(
				value_at(fields.next(), "MoE experts")?,
				value_at(fields.next(), "MoE top-k")?,
				value_at(fields.next(), "MoE expert width")?,
				activation(value_at(fields.next(), "MoE activation")?)?,
				scoring(value_at(fields.next(), "MoE scoring")?)?,
				bool_value(fields.next().unwrap_or(""), "MoE renormalization")?,
				bool_value(fields.next().unwrap_or(""), "MoE shared expert")?,
			)),
			"perc" => Ok(Operation::Perceptron(value_at(Some(rest), "perceptron width")?)),
			"embed" => Ok(Operation::Embed(value_at(fields.next(), "embedding vocabulary")?, value_at(fields.next(), "embedding width")?)),
			"hyper" => {
				let (lanes, rest) = rest.split_once(',').unwrap_or((rest, ""));
				let (rank, blocks) = rest.split_once(',').unwrap_or((rest, ""));
				Ok(Operation::Hyper(
					value_at(Some(lanes), "hyper-connection lanes")?,
					value_at(Some(rank), "hyper-connection rank")?,
					blocks.split(';').filter(|part| !part.is_empty()).map(|part| untext(part, "hyper-connection block").and_then(|part| block(&part))).collect::<Result<Vec<_>>>()?,
				))
			}
			"dconv" => Ok(Operation::Dconv(value_at(Some(rest), "depthwise convolution kernel")?)),
			"delta" => {
				let (heads, kernel) = (value_at(fields.next(), "delta heads")?, value_at(fields.next(), "delta kernel")?);
				// A bundle written before the extents were separable names neither, so
				// an absent field takes the extent from the stream, as the builder does.
				let mut extent = |role| fields.next().map(|field| value_at(Some(field), role)).transpose().map(|value| value.unwrap_or(0));
				let (key_heads, key_width) = (extent("delta key heads")?, extent("delta key width")?);
				let (value_width, output) = (extent("delta value width")?, extent("delta output width")?);
				Ok(Operation::Delta(DeltaBlock { heads, kernel, key_heads, key_width, value_width, output }))
			}
			"ple" => {
				let (heads, width) = (value_at(fields.next(), "per-layer embedding heads")?, value_at(fields.next(), "per-layer embedding width")?);
				let (rows, kernel) = (value_at(fields.next(), "per-layer embedding rows")?, value_at(fields.next(), "per-layer embedding kernel")?);
				let dilation = value_at(fields.next(), "per-layer embedding dilation")?;
				let hash = RowHash::parse(&mut fields)?;
				require(hash.heads() == heads, format!("per-layer embedding names {heads} heads, its hash addresses {}", hash.heads()))?;
				Ok(Operation::Ple(PleBlock { heads, width, rows, kernel, dilation, hash }))
			}
			"norm" => Ok(Operation::Norm),
			"glu" => Ok(Operation::Glu(value_at(fields.next(), "gated feed-forward width")?, activation(value_at(fields.next(), "gated feed-forward activation")?)?)),
			_ => Err(RecipeError::new(format!("invalid model operation {name:?}"))),
		}
	}
	fn normalization_text(normalization: Option<BlockNormalization>) -> u8 {
		normalization.map_or(0, |value| value as u8 + 1)
	}
	fn normalization(field: Option<&str>, role: &str) -> Result<Option<BlockNormalization>> {
		Ok(match value_at::<u8>(field, role)? {
			0 => None,
			1 => Some(BlockNormalization::Batch),
			2 => Some(BlockNormalization::Layer),
			3 => Some(BlockNormalization::Rms),
			4 => Some(BlockNormalization::L2),
			_ => return Err(RecipeError::new("invalid block normalization")),
		})
	}
	fn block_text(block: &Block) -> String {
		format!(
			"{}|{}|{}|{}|{}|{}|{}|{}",
			operation_text(&block.operation),
			block.activation as u8,
			normalization_text(block.normalization),
			block.quantization,
			u8::from(block.profile),
			normalization_text(block.qk),
			u8::from(block.frozen),
			u8::from(block.packed)
		)
	}
	fn block(value: &str) -> Result<Block> {
		let fields = value.split('|').collect::<Vec<_>>();
		require(fields.len() == 8, "semantic model block has the wrong width")?;
		Ok(Block {
			operation: operation(fields[0])?,
			activation: activation(value_at(Some(fields[1]), "block activation")?)?,
			normalization: normalization(Some(fields[2]), "block normalization")?,
			qk: normalization(Some(fields[5]), "block query and key normalization")?,
			quantization: value_at(Some(fields[3]), "block quantization")?,
			profile: bool_value(fields[4], "block quantization profile")?,
			frozen: bool_value(fields[6], "block frozen qualifier")?,
			packed: bool_value(fields[7], "block packed qualifier")?,
		})
	}
	fn model_text(model: &Model) -> Vec<String> {
		model.blocks.iter().map(block_text).collect()
	}
	fn model(blocks: Vec<Block>, loss: u8, quantization: u16, epsilon: f64) -> Result<Model> {
		require(!blocks.is_empty(), "semantic model has no blocks")?;
		require(matches!(loss, 0..=4 | 6), format!("saved model loss {loss} is unavailable"))?;
		Ok(Model { blocks, loss: LossFunction(loss), quantization, epsilon, frozen: false, packed: false })
	}
	#[derive(Clone)]
	pub(super) struct StoredGraph {
		pub graph: Graph,
		pub model: Model,
		pub precision: Compute,
		pub inputs: Vec<String>,
		pub outputs: Vec<String>,
		pub norm_mean: Vec<f64>,
		pub norm_scale: Vec<f64>,
		pub target_min: f64,
		pub target_span: f64,
		pub bn_stats: Vec<f64>,
		pub artifact: String,
	}
	#[derive(Clone)]
	pub(super) struct SemanticGraph {
		pub model: Model,
		pub precision: Compute,
		pub input: Shape,
		pub output: Shape,
		pub inputs: Vec<String>,
		pub outputs: Vec<String>,
		pub tensors: Vec<StoredWeight>,
		pub predictors: Vec<PredictorProgram>,
		pub frozen: Vec<u8>,
		pub state: TrainingState,
		pub norm_mean: Vec<f64>,
		pub norm_scale: Vec<f64>,
		pub target_min: f64,
		pub target_span: f64,
		pub bn_stats: Vec<f64>,
		pub artifact: String,
	}

	pub(super) fn raw_weight(values: &[f64]) -> StoredWeight {
		let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
		StoredWeight { format: StorageFormat(0), count: values.len(), bytes: bytes.into(), codebook: Vec::new(), arithmetic: values.to_vec() }
	}
	fn semantic_graph(stored: &StoredGraph) -> Result<SemanticGraph> {
		let graph = &stored.graph;
		let mut tensors = Vec::new();
		for (index, node) in graph.nodes.iter().enumerate() {
			let count = node.weights();
			if count == 0 {
				continue;
			}
			let values = graph.parameters.get(node.offset..node.offset + node.parameters).ok_or_else(|| RecipeError::new("model parameter span is invalid"))?;
			let encoded = graph.stored.get(index).and_then(Clone::clone).unwrap_or_else(|| raw_weight(values));
			require(encoded.count == count && encoded.arithmetic.len() == count, format!("model tensor {index} has the wrong shape"))?;
			tensors.push(encoded);
		}
		let mut predictors = Vec::new();
		for node in &graph.nodes {
			if node.op != Primitive::Predictor {
				continue;
			}
			let code =
				graph.programs.get(node.program_offset..node.program_offset + node.program_count * 2).ok_or_else(|| RecipeError::new("fitted estimator program span is invalid"))?.to_vec();
			predictors.push(PredictorProgram { code, locals: node.argument[0] as usize, stack: node.argument[1] as usize, table: vec![0.0; node.parameters], nearest: None });
		}
		Ok(SemanticGraph {
			model: stored.model.clone(),
			precision: stored.precision,
			input: graph.input,
			output: graph.output,
			inputs: stored.inputs.clone(),
			outputs: stored.outputs.clone(),
			tensors,
			predictors,
			frozen: graph.frozen.clone(),
			state: graph.state.clone(),
			norm_mean: stored.norm_mean.clone(),
			norm_scale: stored.norm_scale.clone(),
			target_min: stored.target_min,
			target_span: stored.target_span,
			bn_stats: stored.bn_stats.clone(),
			artifact: stored.artifact.clone(),
		})
	}
	fn same_model(a: &Model, b: &Model) -> bool {
		a.loss.0 == b.loss.0 && a.quantization == b.quantization && a.epsilon.to_bits() == b.epsilon.to_bits() && model_text(a) == model_text(b)
	}
	fn values<T: FromStr>(text: &str, role: &str) -> Result<Vec<T>>
	where
		T::Err: fmt::Display,
	{
		text.split_whitespace().map(|value| value.parse().map_err(|error| RecipeError::new(format!("invalid {role}: {error}")))).collect()
	}
	fn value<T: FromStr>(text: &str, role: &str) -> Result<T>
	where
		T::Err: fmt::Display,
	{
		text.parse().map_err(|error| RecipeError::new(format!("invalid {role}: {error}")))
	}
	fn precision(value: &str) -> Result<Compute> {
		let fields = value.split_whitespace().collect::<Vec<_>>();
		require(fields.len() == 5, "arithmetic format has the wrong width")?;
		let values = [
			self::value::<u8>(fields[1], "arithmetic bits")?,
			self::value::<u8>(fields[2], "arithmetic exponent")?,
			self::value::<u8>(fields[3], "arithmetic mantissa")?,
			self::value::<u8>(fields[4], "storage mantissa")?,
		];
		Compute::saved(fields[0], values).ok_or_else(|| RecipeError::new(format!("saved arithmetic format {} {} {} {} {} is unavailable", fields[0], values[0], values[1], values[2], values[3])))
	}
	#[derive(Default)]
	struct ModelParts {
		loss: Option<u8>,
		quantization: Option<u16>,
		epsilon: Option<f64>,
		blocks: Vec<Block>,
	}
	#[derive(Default)]
	struct SemanticBuilder {
		model: Option<ModelParts>,
		inputs: Vec<String>,
		outputs: Vec<String>,
		input: Option<Shape>,
		output: Option<Shape>,
		precision: Option<Compute>,
		tensors: Vec<StoredWeight>,
		predictors: Vec<PredictorProgram>,
		frozen: Vec<u8>,
		state: TrainingState,
		norm_mean: Vec<f64>,
		norm_scale: Vec<f64>,
		target_min: f64,
		target_span: f64,
		bn_stats: Vec<f64>,
		artifact: String,
	}
	impl SemanticBuilder {
		fn finish(self) -> Result<SemanticGraph> {
			let (input, output) =
				(self.input.ok_or_else(|| RecipeError::new("semantic model has no input shape"))?, self.output.ok_or_else(|| RecipeError::new("semantic model has no output shape"))?);
			let parts = self.model.ok_or_else(|| RecipeError::new("semantic model is absent"))?;
			let model = model(
				parts.blocks,
				parts.loss.ok_or_else(|| RecipeError::new("semantic model has no loss"))?,
				parts.quantization.ok_or_else(|| RecipeError::new("semantic model has no quantization"))?,
				// A bundle saved before models carried an epsilon was lowered with the Cargo default.
				parts.epsilon.map_or_else(default_epsilon, Ok)?,
			)?;
			require(self.inputs.len() == input.elements(), "semantic model input schema has the wrong width")?;
			require(self.outputs.len() == output.elements(), "semantic model output schema has the wrong width")?;
			require(
				self.norm_mean.len() == self.norm_scale.len() && (self.norm_mean.is_empty() || self.norm_mean.len() == self.inputs.len()),
				"semantic model normalization stats have the wrong width",
			)?;
			require(!self.artifact.is_empty(), "native artifact identity is absent")?;
			// An embed block saves its table as the gather's packed context, and a
			// per-layer embedding its host table, so each owns a tensor without a
			// parameter span for the frozen mask to cover.
			let tables = model.blocks.iter().filter_map(|block| match &block.operation {
				Operation::Embed(vocabulary, width) => Some(vocabulary.saturating_mul(*width)),
				Operation::Ple(ple) => Some(ple.table()),
				_ => None,
			});
			require(self.frozen.len().saturating_add(tables.sum::<usize>()) == self.tensors.iter().map(|tensor| tensor.count).sum::<usize>(), "semantic model frozen weights are incomplete")?;
			for (name, values) in [("moments", &self.state.moments), ("variances", &self.state.variances)] {
				require(values.is_empty() || values.len() == self.frozen.len(), format!("semantic model {name} are incomplete"))?;
			}
			let estimators = model.blocks.iter().filter(|block| matches!(block.operation, Operation::Estimator(_))).count();
			require(self.predictors.len() == estimators, "semantic model fitted estimator programs are incomplete")?;
			Ok(SemanticGraph {
				model,
				precision: self.precision.ok_or_else(|| RecipeError::new("semantic model has no arithmetic format"))?,
				input,
				output,
				inputs: self.inputs,
				outputs: self.outputs,
				tensors: self.tensors,
				predictors: self.predictors,
				frozen: self.frozen,
				state: self.state,
				norm_mean: self.norm_mean,
				norm_scale: self.norm_scale,
				target_min: self.target_min,
				target_span: self.target_span,
				bn_stats: self.bn_stats,
				artifact: self.artifact,
			})
		}
	}
	fn stored_weight(format: u16, count: usize, codebook: &str, encoded: &str) -> Result<StoredWeight> {
		let bytes = unhex(encoded, "semantic tensor bytes")?;
		let codebook = if codebook == "-" {
			Vec::new()
		} else {
			codebook.split(',').map(|value| value.parse::<f64>().map_err(|error| RecipeError::new(format!("invalid semantic codebook value: {error}")))).collect::<Result<Vec<_>>>()?
		};
		let arithmetic = if format == 0 {
			require(bytes.len() == count.checked_mul(std::mem::size_of::<f64>()).ok_or_else(|| RecipeError::new("semantic tensor size overflows"))?, "semantic raw tensor has the wrong size")?;
			bytes.chunks_exact(std::mem::size_of::<f64>()).map(|value| f64::from_le_bytes(value.try_into().unwrap())).collect()
		} else {
			StorageFormat(format).decompress(&bytes, &codebook, count)?
		};
		Ok(StoredWeight { format: StorageFormat(format), count, bytes: bytes.into(), codebook, arithmetic })
	}
	pub(super) fn load_semantic(path: &Path) -> Result<(DataSchema, Vec<SemanticGraph>)> {
		require(path.extension().and_then(|value| value.to_str()) == Some("ogdl"), "model path requires .ogdl")?;
		let document = fs::read_to_string(path).map_err(|error| RecipeError::new(format!("cannot read {}: {error}", path.display())))?;
		let (mut schema, mut graphs, mut current): (Option<DataSchema>, Vec<SemanticGraph>, Option<SemanticBuilder>) = (None, Vec::new(), None);
		for line in document.lines().map(str::trim) {
			if line.is_empty() {
				continue;
			}
			if line == BUNDLE_HEADER {
				continue;
			}
			if line == "schema" {
				require(schema.is_none() && current.is_none(), "semantic model has more than one schema")?;
				schema = Some(DataSchema::default());
				continue;
			}
			if line == "graph" {
				require(schema.is_some(), "semantic model has no schema")?;
				if let Some(builder) = current.take() {
					graphs.push(builder.finish()?)
				}
				current = Some(SemanticBuilder::default());
				continue;
			}
			let (kind, value) = line.split_once(' ').unwrap_or((line, ""));
			if current.is_none() {
				schema.as_mut().ok_or_else(|| RecipeError::new("semantic schema value precedes schema"))?.push((kind.to_owned(), value.to_owned()));
				continue;
			}
			let builder = current.as_mut().ok_or_else(|| RecipeError::new("semantic model value precedes graph"))?;
			match kind {
				"model" => {
					let fields = value.split_whitespace().collect::<Vec<_>>();
					require(matches!(fields.len(), 2 | 3), "semantic model header has the wrong width")?;
					require(builder.model.is_none(), "semantic graph has more than one model")?;
					builder.model = Some(ModelParts {
						loss: Some(value_at(fields.first().copied(), "semantic model loss")?),
						quantization: Some(value_at(fields.get(1).copied(), "semantic model quantization")?),
						epsilon: fields.get(2).map(|field| number("semantic model epsilon", field)).transpose()?,
						..ModelParts::default()
					});
				}
				"block" => {
					builder.model.as_mut().ok_or_else(|| RecipeError::new("semantic block precedes model"))?.blocks.push(block(value)?);
				}
				"arithmetic" => builder.precision = Some(precision(value)?),
				"in" => builder.inputs.push(untext(value, "model input")?),
				"out" => builder.outputs.push(untext(value, "model output")?),
				"shape" => {
					let shape = values::<usize>(value, "model shape")?;
					require(shape.len() == 4, "semantic model shape has the wrong width")?;
					builder.input = Some(Shape { channels: shape[0], length: shape[1] });
					builder.output = Some(Shape { channels: shape[2], length: shape[3] });
				}
				"tensor" => {
					let fields = value.split_whitespace().collect::<Vec<_>>();
					require(fields.len() == 4, "semantic tensor has the wrong width")?;
					builder.tensors.push(stored_weight(
						value_at(fields.first().copied(), "semantic tensor format")?,
						value_at(fields.get(1).copied(), "semantic tensor count")?,
						fields[2],
						fields[3],
					)?);
				}
				"predictor" => {
					let fields = values::<f64>(value, "fitted estimator program")?;
					require(fields.len() >= 3 && (fields.len() - 3) % 2 == 0, "fitted estimator program has the wrong width")?;
					let slot =
						|value: f64, role| usize::try_from(value as i64).ok().filter(|_| value.fract() == 0.0).ok_or_else(|| RecipeError::new(format!("invalid fitted estimator {role}")));
					builder.predictors.push(PredictorProgram {
						locals: slot(fields[0], "locals")?,
						stack: slot(fields[1], "stack")?,
						table: vec![0.0; slot(fields[2], "table width")?],
						code: fields[3..].to_vec(),
						nearest: None,
					});
				}
				"frozen" => builder.frozen = values(value, "frozen weight")?,
				"moments" => builder.state.moments = values(value, "Adam moment")?,
				"variances" => builder.state.variances = values(value, "Adam variance")?,
				"best_loss" => builder.state.best_loss = values(value, "best loss")?,
				"epoch" => builder.state.epoch = value.parse().map_err(|error| RecipeError::new(format!("invalid epoch: {error}")))?,
				"training_rows" => builder.state.training_rows = value.parse().map_err(|error| RecipeError::new(format!("invalid training rows: {error}")))?,
				"trained_samples" => builder.state.trained_samples = values(value, "trained sample identity")?,
				"norm_mean" => builder.norm_mean = values(value, "normalization mean")?,
				"norm_scale" => builder.norm_scale = values(value, "normalization scale")?,
				"target_min" => builder.target_min = value.parse().map_err(|error| RecipeError::new(format!("invalid target minimum: {error}")))?,
				"target_span" => builder.target_span = value.parse().map_err(|error| RecipeError::new(format!("invalid target span: {error}")))?,
				"bn_stats" => builder.bn_stats = values(value, "batch normalization statistics")?,
				"artifact" => builder.artifact = untext(value, "native artifact identity")?,
				_ => return Err(RecipeError::new(format!("invalid semantic model value: {line}"))),
			}
		}
		if let Some(builder) = current {
			graphs.push(builder.finish()?)
		}
		require(!graphs.is_empty(), "model has no graphs")?;
		Ok((schema.ok_or_else(|| RecipeError::new("semantic model has no schema"))?, graphs))
	}
	pub(super) fn save_semantic(path: &Path, schema: &DataSchema, graphs: &mut [StoredGraph]) -> Result<()> {
		let config = Config::load()?;
		let semantic = graphs
			.iter_mut()
			.map(|stored| {
				stored.graph.refresh_storage(config)?;
				semantic_graph(stored)
			})
			.collect::<Result<Vec<_>>>()?;
		save_semantic_graphs(path, schema, &semantic)
	}
	pub(super) fn save_semantic_graphs(path: &Path, schema: &DataSchema, graphs: &[SemanticGraph]) -> Result<()> {
		require(path.extension().and_then(|value| value.to_str()) == Some("ogdl"), "save requires an .ogdl model")?;
		require(!graphs.is_empty(), "model bundle has no graphs")?;
		fn field(document: &mut String, key: &str, value: &str) {
			document.push_str(&format!("        {key} {value}\n"));
		}
		let mut document = format!("{BUNDLE_HEADER}\n    schema\n");
		for (kind, value) in schema {
			document.push_str(&format!("        {kind} {value}\n"))
		}
		for semantic in graphs {
			document.push_str("    graph\n");
			field(&mut document, "model", &format!("{} {} {}", semantic.model.loss.0, semantic.model.quantization, semantic.model.epsilon));
			for block in &semantic.model.blocks {
				field(&mut document, "block", &block_text(block));
			}
			for name in &semantic.inputs {
				field(&mut document, "in", &text(name));
			}
			for name in &semantic.outputs {
				field(&mut document, "out", &text(name));
			}
			let (family, values) = semantic.precision.saved_fields();
			field(&mut document, "arithmetic", &format!("{family} {} {} {} {}", values[0], values[1], values[2], values[3]));
			field(&mut document, "shape", &format!("{} {} {} {}", semantic.input.channels, semantic.input.length, semantic.output.channels, semantic.output.length));
			for tensor in &semantic.tensors {
				let metadata = if tensor.codebook.is_empty() { "-".to_owned() } else { tensor.codebook.iter().map(ToString::to_string).collect::<Vec<_>>().join(",") };
				field(&mut document, "tensor", &format!("{} {} {metadata} {}", tensor.format.0, tensor.count, hex(&tensor.bytes.to_vec())));
			}
			for predictor in &semantic.predictors {
				field(&mut document, "predictor", &format!("{} {} {} {}", predictor.locals, predictor.stack, predictor.table.len(), join(&predictor.code)));
			}
			for (key, value) in [
				("frozen", join(&semantic.frozen)),
				("moments", join(&semantic.state.moments)),
				("variances", join(&semantic.state.variances)),
				("best_loss", join(&semantic.state.best_loss)),
				("epoch", semantic.state.epoch.to_string()),
				("training_rows", semantic.state.training_rows.to_string()),
				("trained_samples", join(&semantic.state.trained_samples)),
				("norm_mean", join(&semantic.norm_mean)),
				("norm_scale", join(&semantic.norm_scale)),
			] {
				field(&mut document, key, &value)
			}
			if semantic.target_span != 0.0 {
				field(&mut document, "target_min", &semantic.target_min.to_string());
				field(&mut document, "target_span", &semantic.target_span.to_string());
			}
			if !semantic.bn_stats.is_empty() {
				field(&mut document, "bn_stats", &join(&semantic.bn_stats))
			}
			require(!semantic.artifact.is_empty(), "native artifact identity is absent")?;
			field(&mut document, "artifact", &text(&semantic.artifact));
		}
		// Publish atomically through an exclusively created temporary sibling: the path always
		// holds one publisher's complete model, and concurrent publishers never share a file.
		let mut serial = 0;
		let (temporary, mut file) = loop {
			let candidate = path.with_extension(format!("ogdl.{}.{serial}.tmp", std::process::id()));
			match fs::File::create_new(&candidate) {
				Ok(file) => break (candidate, file),
				Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => serial += 1,
				Err(error) => return Err(RecipeError::new(format!("cannot write {}: {error}", candidate.display()))),
			}
		};
		let published =
			file.write_all(document.as_bytes()).and_then(|()| file.sync_all()).map_err(|error| RecipeError::new(format!("cannot write {}: {error}", temporary.display()))).and_then(|()| {
				drop(file);
				fs::rename(&temporary, path).map_err(|error| RecipeError::new(format!("cannot publish {}: {error}", path.display())))
			});
		if published.is_err() {
			fs::remove_file(&temporary).ok();
		}
		published
	}
	fn join<T: ToString>(values: &[T]) -> String {
		values.iter().map(ToString::to_string).collect::<Vec<_>>().join(" ")
	}
	fn same_structure(a: &SemanticGraph, b: &SemanticGraph) -> bool {
		a.precision == b.precision
			&& a.input == b.input
			&& a.output == b.output
			&& a.inputs == b.inputs
			&& a.outputs == b.outputs
			&& same_model(&a.model, &b.model)
			&& a.tensors.len() == b.tensors.len()
			&& a.tensors.iter().zip(&b.tensors).all(|(a, b)| a.format.0 == b.format.0 && a.count == b.count)
			&& a.frozen.len() == b.frozen.len()
	}
	pub(super) fn artifact_key(model: &Model, schema: &DataSchema, precision: Compute, graph: &Graph, target: &str) -> String {
		let mut hash = 0xcbf29ce484222325_u64;
		let mut feed = |value: &str| {
			for byte in value.as_bytes() {
				hash ^= u64::from(*byte);
				hash = hash.wrapping_mul(0x100000001b3);
			}
		};
		feed(BUNDLE_HEADER);
		for (kind, value) in schema {
			feed(&format!("{kind}:{value};"))
		}
		feed(target);
		feed(&format!("precision:{precision:?};"));
		feed(&format!("loss:{};quant:{};epsilon:{};blocks:{};", model.loss.0, model.quantization, model.epsilon.to_bits(), model_text(model).join("/")));
		for node in &graph.nodes {
			feed(&format!("node:{}:{}:{}:{};", node.offset, node.parameters, node.argument[8].to_bits(), node.output.elements()));
		}
		format!("recipe-native-{hash:016x}")
	}
	pub(super) fn restore(path: &Path, schema: &DataSchema, graphs: &mut [StoredGraph], identities: &[u64]) -> Result<()> {
		if !fs::exists(path).map_err(|error| RecipeError::new(format!("cannot inspect {}: {error}", path.display())))? {
			return save_semantic(path, schema, graphs);
		}
		let (stored_schema, stored) = load_semantic(path)?;
		let current = graphs.iter().map(semantic_graph).collect::<Result<Vec<_>>>()?;
		let matches = &stored_schema == schema && stored.len() == current.len() && stored.iter().zip(&current).all(|(a, b)| same_structure(a, b));
		if matches {
			for (current, saved) in graphs.iter_mut().zip(&stored) {
				let saved_boundary = saved.state.training_rows;
				let current_boundary = current.graph.state.training_rows;
				if saved_boundary != 0 {
					require(!saved.state.trained_samples.is_empty(), "resume rejected: saved model has no training membership identity")?;
					require(current_boundary <= identities.len(), "current training membership is incomplete")?;
					let trained = saved.state.trained_samples.iter().copied().collect::<BTreeSet<_>>();
					let overlap = identities[current_boundary..].iter().filter(|value| trained.contains(value)).count();
					require(
						overlap == 0,
						format!("resume rejected: {overlap} evaluation samples were previously trained, current boundary is {current_boundary} and saved boundary was {saved_boundary}"),
					)?;
				}
				let same = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits());
				require(
					same(&current.norm_mean, &saved.norm_mean)
						&& same(&current.norm_scale, &saved.norm_scale)
						&& current.target_min.to_bits() == saved.target_min.to_bits()
						&& current.target_span.to_bits() == saved.target_span.to_bits(),
					format!("resume rejected: fitted preprocessing differs, current boundary is {current_boundary} and saved boundary was {saved_boundary}"),
				)?;
			}
			for (current, saved) in graphs.iter_mut().zip(stored) {
				let current_training_rows = current.graph.state.training_rows;
				let mut tensor = 0;
				for (index, node) in current.graph.nodes.iter().enumerate() {
					if node.weights() == 0 {
						continue;
					}
					let encoded = saved.tensors.get(tensor).ok_or_else(|| RecipeError::new("saved semantic tensor is absent"))?;
					require(encoded.count == node.weights(), "saved semantic tensor has the wrong shape")?;
					current.graph.parameters[node.offset..node.offset + node.parameters].copy_from_slice(&encoded.arithmetic[..node.parameters]);
					if let Some(slot) = current.graph.stored.get_mut(index) {
						*slot = (encoded.format.0 != 0 || node.table()).then_some(encoded.clone())
					}
					tensor += 1;
				}
				require(tensor == saved.tensors.len(), "saved semantic tensors are incomplete")?;
				current.graph.state = saved.state;
				current.graph.frozen = saved.frozen;
				current.graph.state.training_rows = current_training_rows;
			}
			return Ok(());
		}
		eprint!("mismatch: overwrite {}? Y/n ", path.display());
		std::io::stderr().flush().map_err(|error| RecipeError::new(format!("cannot prompt: {error}")))?;
		let mut answer = String::new();
		let received = std::io::stdin().read_line(&mut answer).map_err(|error| RecipeError::new(format!("cannot read answer: {error}")))?;
		require(received != 0 && (answer.trim().is_empty() || answer.trim().eq_ignore_ascii_case("y")), "model mismatch not overwritten")?;
		save_semantic(path, schema, graphs)
	}
	pub(super) fn run_infer(path: &Path, input: &[f64], forward: impl FnMut(&SemanticGraph, &[f64]) -> Result<Vec<f64>>) -> Result<Vec<f64>> {
		let (_, graphs) = load_semantic(path)?;
		infer_graphs(&graphs, input, forward)
	}
	pub(super) fn infer_graphs(graphs: &[SemanticGraph], input: &[f64], mut forward: impl FnMut(&SemanticGraph, &[f64]) -> Result<Vec<f64>>) -> Result<Vec<f64>> {
		let first = graphs.first().ok_or_else(|| RecipeError::new("model has no graph"))?;
		require(input.len() == first.inputs.len(), format!("model input expected {} values, received {}", first.inputs.len(), input.len()))?;
		let mut values = first.inputs.iter().cloned().zip(input.iter().copied()).collect::<BTreeMap<_, _>>();
		let mut result = Vec::new();
		for stored in graphs {
			let mut samples = stored.inputs.iter().map(|name| values.get(name).copied().ok_or_else(|| RecipeError::new(format!("input {name:?} is absent")))).collect::<Result<Vec<_>>>()?;
			if !stored.norm_mean.is_empty() {
				require(stored.norm_mean.len() == samples.len(), format!("model normalization expected {} values, received {}", stored.norm_mean.len(), samples.len()))?;
				for (value, (mean, scale)) in samples.iter_mut().zip(stored.norm_mean.iter().zip(&stored.norm_scale)) {
					*value = (*value - mean) / scale;
				}
			}
			result = forward(stored, &samples)?;
			if stored.target_span > 0.0 {
				for value in &mut result {
					*value = stored.target_min + stored.target_span * logistic(*value);
				}
			}
			require(result.len() == stored.outputs.len(), format!("model output expected {} values, received {}", stored.outputs.len(), result.len()))?;
			for (name, value) in stored.outputs.iter().cloned().zip(result.iter().copied()) {
				values.insert(name, value);
			}
		}
		Ok(result)
	}
}
use std::{
	collections::{BTreeMap, BTreeSet, HashMap},
	error::Error,
	ffi::c_void,
	fmt, fs,
	io::{IsTerminal, Read, Write},
	mem::{size_of, size_of_val},
	path::{Path, PathBuf},
	process::Command,
	ptr,
	sync::{
		Arc, Mutex, OnceLock,
		atomic::{AtomicBool, AtomicU64, Ordering},
	},
	time::{Duration, Instant},
};
pub static recipe: Recipe = Recipe;
static RUN: AtomicU64 = AtomicU64::new(0);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
static INTERRUPT_CHECKPOINTED: AtomicBool = AtomicBool::new(false);
static DEBUG_LOG: OnceLock<std::io::Result<Mutex<fs::File>>> = OnceLock::new();
const DEBUG_LOG_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recipe.log");
const SIGINT: i32 = 2;
const INTERRUPTED_EXIT: i32 = 128 + SIGINT;
static SIGNAL: OnceLock<usize> = OnceLock::new();
extern "C" fn interrupt(_: i32) {
	if !INTERRUPTED.swap(true, Ordering::AcqRel) {
		let message = b"\ninterrupt received, finishing checkpoint\n";
		unsafe {
			write(2, message.as_ptr().cast(), message.len());
		}
	}
}
fn debug(message: &str) -> Result<()> {
	if std::env::var_os("RECIPE_DEBUG").is_none() {
		return Ok(());
	}
	let file = DEBUG_LOG
		.get_or_init(|| fs::OpenOptions::new().create(true).write(true).truncate(true).open(DEBUG_LOG_PATH).map(Mutex::new))
		.as_ref()
		.map_err(|error| RecipeError::new(format!("cannot open {DEBUG_LOG_PATH}: {error}")))?;
	let mut file = file.lock().map_err(|_| RecipeError::new("debug log lock is poisoned"))?;
	writeln!(file, "{message}").and_then(|_| file.flush()).map_err(|error| RecipeError::new(format!("cannot write {DEBUG_LOG_PATH}: {error}")))
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecipeError(String);
impl RecipeError {
	fn new(message: impl Into<String>) -> Self {
		Self(message.into())
	}
}
impl fmt::Display for RecipeError {
	fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
		formatter.write_str(&self.0)
	}
}
impl Error for RecipeError {}
pub type Result<T> = std::result::Result<T, RecipeError>;
type Ptr = *mut c_void;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
	Cpu,
	Amd,
	Nvidia,
}
pub struct Data {
	sources: Vec<String>,
	tests: Vec<String>,
	autoregressive: bool,
	target: Vec<String>,
	features: FeatureSelection,
	broadcast: bool,
	normalize: bool,
	split: f64,
	prepared: OnceLock<Result<Prepared>>,
}
enum FeatureSelection {
	All,
	Include(Vec<String>),
	Exclude(Vec<String>),
}
#[derive(Clone, Copy)]
pub struct Auto;
pub const auto: Auto = Auto;
const CHAR_IDS: [char; 100] = [
	'\t', '\n', ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C',
	'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
	'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '¦', '±', '€',
];
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Residual {
	Layer(usize),
	Conv(usize, usize),
	Activation(Activation),
}
pub const fn layer(width: usize) -> Residual {
	Residual::Layer(width)
}
pub const fn conv(filters: usize, kernel: usize) -> Residual {
	Residual::Conv(filters, kernel)
}
type FitFn = fn(usize, &Prepared, usize, Config) -> Result<Predictor>;
type ValidateFn = fn(usize, usize) -> Result<()>;
#[derive(Clone, Copy, Debug)]
struct Estimator {
	fit: FitFn,
	validate: ValidateFn,
	param: usize,
	name: &'static str,
}
impl PartialEq for Estimator {
	fn eq(&self, other: &Self) -> bool {
		self.param == other.param && self.name == other.name
	}
}
impl Eq for Estimator {}
/// Indexer that scores every key block and keeps the best `keep` blocks per
/// query. `heads` query projections and one shared key projection, both
/// `width` wide, compress `block` keys into one block score. A `tokens` budget
/// states the admission in keys instead and keeps the blocks that cover it.
/// `score` normalizes each indexer head with a trained scale and rotates its
/// leading dimensions before scoring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Indexer {
	heads: usize,
	width: usize,
	block: usize,
	keep: usize,
	tokens: usize,
	score: Option<(BlockNormalization, usize)>,
}
impl Indexer {
	const NONE: Self = Self { heads: 0, width: 0, block: 0, keep: 0, tokens: 0, score: None };
	/// Blocks a query keeps: the blocks that cover the token budget, or `keep`.
	fn admitted(self) -> usize {
		if self.tokens == 0 { self.keep } else { self.tokens.div_ceil(self.block) }
	}
}
/// One attention block: query heads, key-value heads, the head width, and the
/// rotary, indexer and output gate selectors. A zero width takes the head width
/// from the stream, so the heads exactly partition the block input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AttentionBlock {
	heads: usize,
	kv: usize,
	width: usize,
	rope: Option<(usize, u64)>,
	index: Option<Indexer>,
	gate: bool,
}
impl AttentionBlock {
	fn new(heads: usize) -> Self {
		Self { heads, kv: heads, width: 0, rope: None, index: None, gate: false }
	}
	/// The head width this block attends at, and the inner width its heads span.
	fn extent(self, channels: usize) -> Result<(usize, usize)> {
		require(self.heads != 0, "attention head partition is invalid")?;
		let width = match self.width {
			0 => {
				require(channels % self.heads == 0, "attention head partition is invalid")?;
				channels / self.heads
			}
			width => width,
		};
		Ok((width, checked_mul(self.heads, width, "attention inner width")?))
	}
}
/// One gated delta rule block: value heads, the convolution kernel, and the key
/// and value extents. A zero extent takes it from the stream, so the value heads
/// exactly partition the block input and the keys match the values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DeltaBlock {
	heads: usize,
	kernel: usize,
	key_heads: usize,
	key_width: usize,
	value_width: usize,
	output: usize,
}
impl DeltaBlock {
	fn new(heads: usize, kernel: usize) -> Self {
		Self { heads, kernel, key_heads: 0, key_width: 0, value_width: 0, output: 0 }
	}
	/// The key heads and width, the value width, and the output width, resolved
	/// against a block input of `channels`.
	fn extent(self, channels: usize) -> Result<(usize, usize, usize, usize)> {
		require(self.heads != 0, "delta head partition is invalid")?;
		let value = match self.value_width {
			0 => {
				require(channels % self.heads == 0, "delta head partition is invalid")?;
				channels / self.heads
			}
			width => width,
		};
		let keys = if self.key_heads == 0 { self.heads } else { self.key_heads };
		let key = if self.key_width == 0 { value } else { self.key_width };
		require(keys != 0 && self.heads % keys == 0, "delta key head partition is invalid")?;
		Ok((keys, key, value, if self.output == 0 { channels } else { self.output }))
	}
}
/// One per-layer embedding block: every token gathers `heads` rows of `width`
/// from a host-resident table of `rows` rows, addressed by `hash`, and the block
/// projects, gates, convolves and adds them into the stream it sits on. The
/// depthwise convolution is `kernel` wide and dilated by `dilation` positions.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PleBlock {
	heads: usize,
	width: usize,
	rows: usize,
	kernel: usize,
	dilation: usize,
	hash: RowHash,
}
impl PleBlock {
	/// Values the table holds: its rows of the head width.
	fn table(&self) -> usize {
		self.rows.saturating_mul(self.width)
	}
}
#[derive(Clone, Debug, PartialEq, Eq)]
enum Operation {
	Layer(usize),
	Conv(usize, usize),
	Pool(usize),
	Estimator(Estimator),
	Attention(AttentionBlock),
	Rnn(usize),
	Gru(usize),
	Lstm(usize),
	Residual(Vec<Residual>),
	Moe(usize, usize, usize, Activation, Scoring, bool, bool),
	Perceptron(usize),
	Embed(usize, usize),
	Hyper(usize, usize, Vec<Block>),
	Dconv(usize),
	Delta(DeltaBlock),
	Ple(PleBlock),
	/// A normalization that leads a model: the block's own normalization is the
	/// only thing it does, so the model input is normalized before its first block.
	Norm,
	/// A gated feed-forward: `down(activation(gate(x)) * up(x))` through `hidden`.
	Glu(usize, Activation),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Activation {
	Linear,
	Cos,
	Exp,
	Log,
	Ln,
	Huber,
	Tan,
	Relu,
	Leak,
	Sigmoid,
	Tanh,
	Selu,
	Gelu,
	Silu,
	Elu,
	Prelu,
}
/// How the router turns its scores into routing weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Scoring {
	Softmax,
	Sigmoid,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockNormalization {
	Batch,
	Layer,
	/// Root mean square over the row with one trainable scale per channel.
	Rms,
	/// The row divided by its Euclidean norm, floored at the normalization epsilon.
	L2,
}
impl BlockNormalization {
	/// The mode argument of the normalization node; evaluation holds 3.
	const fn mode(self) -> f64 {
		match self {
			Self::Batch => 0.0,
			Self::Layer => 1.0,
			Self::Rms => 2.0,
			Self::L2 => 4.0,
		}
	}
}
/// The normalization selectors with a declared identity: the batch, rms, and l2 markers
/// and the layer residual constructor. Any other selector is rejected instead of guessing a mode.
pub trait NormalizationSelector {
	fn normalization(self) -> BlockNormalization;
}
impl NormalizationSelector for Batch {
	fn normalization(self) -> BlockNormalization {
		BlockNormalization::Batch
	}
}
impl NormalizationSelector for Rms {
	fn normalization(self) -> BlockNormalization {
		BlockNormalization::Rms
	}
}
impl NormalizationSelector for L2 {
	fn normalization(self) -> BlockNormalization {
		BlockNormalization::L2
	}
}
impl<F: Fn(usize) -> Residual> NormalizationSelector for F {
	fn normalization(self) -> BlockNormalization {
		match self(0) {
			Residual::Layer(_) => BlockNormalization::Layer,
			_ => panic!("normalization selector must be batch, layer, rms, or l2"),
		}
	}
}
macro_rules! slots { ($(fn $name:ident = $value:ident),+ $(,)?) => {$(pub const fn $name() -> Residual {
	Residual::Activation(Activation::$value) })+}; }
pub mod atv {
	use super::{Activation, Residual};
	slots! {
	fn linear = Linear, fn cos = Cos, fn exp = Exp, fn log = Log, fn ln = Ln, fn huber = Huber,
	fn tan = Tan, fn relu = Relu, fn leak = Leak, fn sigmoid = Sigmoid, fn tanh = Tanh,
	fn selu = Selu, fn gelu = Gelu, fn silu = Silu, fn elu = Elu, fn prelu = Prelu, }
}
pub use atv::{cos, elu, exp, gelu, leak, linear, ln, log, prelu, relu, selu, sigmoid, silu, tan, tanh};
#[derive(Clone, Debug, PartialEq, Eq)]
struct Block {
	operation: Operation,
	activation: Activation,
	normalization: Option<BlockNormalization>,
	/// The per-head normalization of the attention queries and keys.
	qk: Option<BlockNormalization>,
	quantization: u16,
	profile: bool,
	frozen: bool,
	packed: bool,
}
#[derive(Clone)]
pub struct Model {
	blocks: Vec<Block>,
	loss: LossFunction,
	quantization: u16,
	/// The epsilon every normalization the model lowers is built with; saved with the model.
	epsilon: f64,
	frozen: bool,
	packed: bool,
}
macro_rules! operation_methods { ($(fn $method:ident($($argument:ident: $kind:ty),*) = $operation:expr;)+) => {
$(pub fn $method(&self, $($argument: $kind),*) -> Self { self.push($operation) })+ }; }
impl Model {
	fn push(&self, operation: Operation) -> Self {
		let mut model = self.clone();
		assert!(operation.weighted() || !(model.frozen || model.packed), "{} owns no weights to qualify", operation.name());
		model.blocks.push(Block {
			operation,
			activation: Activation::Linear,
			normalization: None,
			qk: None,
			quantization: model.quantization,
			profile: StorageFormat(model.quantization).selection().is_some(),
			frozen: model.frozen,
			packed: model.packed,
		});
		model.frozen = false;
		model.packed = false;
		model
	}
	/// Clones a model that a block suffix extends, so a pending qualifier cannot outlive its block.
	fn suffix(&self) -> Self {
		assert!(!(self.frozen || self.packed), "block qualifier requires a following block");
		self.clone()
	}
	pub fn frozen(&self) -> Self {
		let mut model = self.clone();
		assert!(!model.packed, "frozen must precede packed");
		model.frozen = true;
		model
	}
	pub fn packed(&self) -> Self {
		let mut model = self.clone();
		model.packed = true;
		model
	}
	pub fn activate(&self, activation: Activation) -> Self {
		let mut model = self.suffix();
		let block = model.blocks.last_mut().unwrap_or_else(|| panic!("activation requires a preceding block"));
		if block.normalization.is_some() {
			panic!("activation must precede normalization");
		}
		block.activation = activation;
		model
	}
	operation_methods! {
	fn layer(width: usize) = Operation::Layer(width);
	fn conv(filters: usize, kernel: usize) = Operation::Conv(filters, kernel);
	fn pool(size: usize) = Operation::Pool(size);
	fn kmeans(clusters: usize) = Operation::Estimator(Estimator { fit: fit_kmeans, validate: cluster_estimator, param: clusters, name: "kmeans" });
	fn knn(neighbors: usize) = Operation::Estimator(Estimator { fit: fit_knn, validate: neighbor_estimator, param: neighbors, name: "knn" });
	fn svm() = Operation::Estimator(Estimator { fit: fit_svm, validate: valid_estimator, param: 0, name: "svm" });
	fn forest(trees: usize) = Operation::Estimator(Estimator { fit: fit_forest, validate: positive_estimator, param: trees, name: "forest" });
	fn bayes() = Operation::Estimator(Estimator { fit: fit_bayes, validate: valid_estimator, param: 0, name: "bayes" });
	fn cbst() = Operation::Estimator(Estimator { fit: fit_catboost, validate: valid_estimator, param: 0, name: "cbst" });
	fn xgbst() = Operation::Estimator(Estimator { fit: fit_xgboost, validate: valid_estimator, param: 0, name: "xgbst" });
	fn lgbm() = Operation::Estimator(Estimator { fit: fit_lightgbm, validate: valid_estimator, param: 0, name: "lgbm" });
	fn attn(heads: usize) = Operation::Attention(AttentionBlock::new(heads));
	fn rnn(width: usize) = Operation::Rnn(width);
	fn gru(width: usize) = Operation::Gru(width);
	fn lstm(width: usize) = Operation::Lstm(width);
	fn perc(width: usize) = Operation::Perceptron(width);
	fn embed(vocabulary: usize, width: usize) = Operation::Embed(vocabulary, width);
	fn dconv(kernel: usize) = Operation::Dconv(kernel);
	fn delta(heads: usize, kernel: usize) = Operation::Delta(DeltaBlock::new(heads, kernel)); }
	pub fn res<const N: usize>(&self, parts: [Residual; N]) -> Self {
		self.push(Operation::Residual(parts.into()))
	}
	pub fn moe(&self, experts: usize, top_k: usize, hidden: usize, activation: Activation, scoring: Scoring, renormalize: bool, shared: bool) -> Self {
		self.push(Operation::Moe(experts, top_k, hidden, activation, scoring, renormalize, shared))
	}
	fn attention(&self, selector: &str, apply: impl FnOnce(&mut AttentionBlock)) -> Self {
		let mut model = self.suffix();
		let block = model.blocks.last_mut().unwrap_or_else(|| panic!("{selector} requires a preceding attn block"));
		match &mut block.operation {
			Operation::Attention(attention) => apply(attention),
			_ => panic!("{selector} requires a preceding attn block"),
		}
		model
	}
	fn delta_block(&self, selector: &str, apply: impl FnOnce(&mut DeltaBlock)) -> Self {
		let mut model = self.suffix();
		let block = model.blocks.last_mut().unwrap_or_else(|| panic!("{selector} requires a preceding delta block"));
		match &mut block.operation {
			Operation::Delta(delta) => apply(delta),
			_ => panic!("{selector} requires a preceding delta block"),
		}
		model
	}
	/// Key-value heads of the preceding `attn` block. Each key-value head serves
	/// `heads / kv` query heads.
	pub fn kv(&self, heads: usize) -> Self {
		self.attention("kv", |attention| attention.kv = heads)
	}
	/// Head width of the preceding `attn` block, so the heads need not partition
	/// the stream. The block attends over `heads * width` and its gate spans the
	/// same width before the output projection returns to the stream.
	pub fn head(&self, width: usize) -> Self {
		self.attention("head", |attention| attention.width = width)
	}
	/// Key and query heads of the preceding `delta` block, at `width` each. Every
	/// key head serves `heads / count` value heads.
	pub fn keys(&self, count: usize, width: usize) -> Self {
		self.delta_block("keys", |delta| (delta.key_heads, delta.key_width) = (count, width))
	}
	/// Value width per head of the preceding `delta` block, so its value heads
	/// need not partition the stream.
	pub fn values(&self, width: usize) -> Self {
		self.delta_block("values", |delta| delta.value_width = width)
	}
	/// Output width of the preceding `delta` block's closing projection.
	pub fn out(&self, width: usize) -> Self {
		self.delta_block("out", |delta| delta.output = width)
	}
	/// Rotary position embedding on the preceding `attn` block: the first `dims`
	/// channels of every query and key head rotate by their position at
	/// frequencies `base^(-2i/dims)`.
	pub fn rope(&self, dims: usize, base: f64) -> Self {
		self.attention("rope", |attention| attention.rope = Some((dims, base.to_bits())))
	}
	/// Sparse key selection on the preceding `attn` block. `heads` query
	/// projections and one key projection, each `width` wide, score every group
	/// of `block` keys, and each query attends to its best `keep` blocks.
	pub fn index(&self, heads: usize, width: usize, block: usize, keep: usize) -> Self {
		self.attention("index", |attention| attention.index = Some(Indexer { heads, width, block, keep, ..Indexer::NONE }))
	}
	/// Token budget of the preceding `index`: each query keeps the blocks that
	/// cover `tokens` keys, which is how a checkpoint's `attention.indexer.top_k`
	/// states its admission. A budget that covers the sequence is dense attention.
	pub fn budget(&self, tokens: usize) -> Self {
		self.indexer("budget", |index| index.tokens = tokens)
	}
	/// Trained scoring geometry of the preceding `index`: every indexer query and
	/// key head normalizes under `normalization` with its own trained scale, and
	/// its leading `dims` channels rotate at the block's `rope` base before the
	/// indexer scores. Zero `dims` leaves the planes unrotated.
	pub fn score(&self, normalization: impl NormalizationSelector, dims: usize) -> Self {
		let normalization = normalization.normalization();
		if !matches!(normalization, BlockNormalization::Rms | BlockNormalization::L2) {
			panic!("indexer scoring normalization must be rms or l2");
		}
		self.indexer("score", |index| index.score = Some((normalization, dims)))
	}
	fn indexer(&self, selector: &str, apply: impl FnOnce(&mut Indexer)) -> Self {
		self.attention(selector, |attention| match &mut attention.index {
			Some(index) => apply(index),
			None => panic!("{selector} requires a preceding index"),
		})
	}
	/// Sigmoid gate on the output of the preceding `attn` block, from its own
	/// projection of the block input.
	pub fn gate(&self) -> Self {
		self.attention("gate", |attention| attention.gate = true)
	}
	/// Hyper-connections: a stream of `lanes` copies of the width feeds `branch`
	/// through a gated read and takes its output back through gated writes.
	/// `rank` sizes the gate bottleneck; zero fixes every gate at one.
	pub fn hyper(&self, lanes: usize, rank: usize, branch: &Model) -> Self {
		assert!(!branch.blocks.is_empty(), "hyper-connection branch requires a block");
		self.push(Operation::Hyper(lanes, rank, branch.blocks.clone()))
	}
	/// Per-layer embedding: every token gathers `table`'s rows on the host, and the
	/// block projects, gates and convolves them into the stream it sits on, at
	/// whatever width the stream has there.
	pub fn ple(&self, table: &Ngram<'_>) -> Self {
		self.push(Operation::Ple(table.block()))
	}
	/// Normalizes the preceding block's output. Leading a model, it normalizes
	/// the model input before the first block, which is the pre-normalization
	/// of a residual branch when the model is one.
	pub fn norm(&self, normalization: impl NormalizationSelector) -> Self {
		let mut model = if self.blocks.is_empty() { self.push(Operation::Norm) } else { self.suffix() };
		let block = model.blocks.last_mut().unwrap_or_else(|| panic!("normalization requires a preceding block"));
		block.normalization = Some(normalization.normalization());
		model
	}
	/// A gated feed-forward: `down(activation(gate(x)) * up(x))` through `hidden`,
	/// returning to the block input width.
	pub fn glu(&self, hidden: usize, activation: Activation) -> Self {
		self.push(Operation::Glu(hidden, activation))
	}
	/// Normalizes each attention head's query and key rows after the projection.
	/// The value rows keep their projected magnitudes.
	pub fn qk(&self, normalization: impl NormalizationSelector) -> Self {
		let mut model = self.clone();
		let block = model.blocks.last_mut().unwrap_or_else(|| panic!("query and key normalization requires a preceding block"));
		let normalization = normalization.normalization();
		if !matches!(block.operation, Operation::Attention(_)) {
			panic!("query and key normalization requires an attention block");
		}
		if !matches!(normalization, BlockNormalization::Rms | BlockNormalization::L2) {
			panic!("query and key normalization must be rms or l2");
		}
		block.qk = Some(normalization);
		model
	}
	pub fn loss(&self, loss: LossFunction) -> Self {
		let mut model = self.clone();
		model.loss = loss;
		model
	}
	/// The epsilon every normalization of this model lowers with: the floor under an
	/// L2 norm and the shift under a batch, layer, or RMS variance, in every block,
	/// query and key normalization, delta rule, and hyper-connection gate. It is
	/// saved with the model, so a bundle reloads with the value it was trained with.
	pub fn epsilon(&self, value: f64) -> Self {
		assert!(value.is_finite() && value > 0.0, "normalization epsilon must be finite and positive");
		let mut model = self.clone();
		model.epsilon = value;
		model
	}
	pub fn quantize(&self, family: u16, bits: u8, variant: u16) -> Self {
		let mut model = self.clone();
		let format = family << 12 | variant << 8 | u16::from(bits);
		if let Some(block) = model.blocks.last_mut() {
			block.quantization = format;
			block.profile = StorageFormat(format).selection().is_some()
		} else {
			model.quantization = format
		}
		model
	}
	pub fn qi(&self, bits: u8) -> Qi {
		assert!([2, 3, 4, 5, 6, 8].contains(&bits), "qi bits must be 2, 3, 4, 5, 6, or 8");
		let q = |v| self.quantize(0, bits, v);
		Qi(q(0), q(1), QiSuffix { nf: q(2), k: Qk { model: q(3), s: q(4), m: q(5), l: q(6) } })
	}
	pub fn iq(&self, bits: u8) -> Iq {
		assert!((1..=4).contains(&bits), "iq bits must be 1 through 4");
		let q = |v| self.quantize(1, bits, v);
		Iq { xxs: q(1), xs: q(2), s: q(3), m: q(4), nl: q(5) }
	}
	fn description(&self, metrics: &[Metric]) -> String {
		let has = |value| metrics.iter().any(|metric| metric.0 == value);
		let selected = (has(5), has(6), has(7), has(9));
		let output = usize::from(matches!(self.blocks.last(), Some(Block { operation: Operation::Layer(1), activation: Activation::Linear, normalization: None, .. })));
		self.blocks
			.iter()
			.take(self.blocks.len() - output)
			.filter_map(|block| {
				let mut names = Vec::new();
				if selected.0 {
					names.push(block.operation.name().to_owned())
				}
				if selected.1 && block.activation != Activation::Linear {
					names.push(block.activation.name().to_owned())
				}
				if selected.2 {
					if let Some(name) = block.qk.map(BlockNormalization::name) {
						names.push(format!("qk-{name}"))
					}
					if let Some(name) = block.normalization.map(BlockNormalization::name) {
						names.push(name.to_owned())
					}
				}
				if selected.3 && block.quantization != 0 {
					names.push(quantization(block.quantization))
				}
				(!names.is_empty()).then(|| names.join("."))
			})
			.collect::<Vec<_>>()
			.join("/")
	}
}
fn quantization(code: u16) -> String {
	let (family, bits, variant) = (code >> 12, code as u8, usize::from(code >> 8 & 15));
	let variants: &[&str] = if family == 0 {
		&["_0", "_1", "_NF", "_K", "_K_S", "_K_M", "_K_L"]
	} else if family == 1 {
		&["", "_XXS", "_XS", "_S", "_M", "_NL"]
	} else {
		return format!("quantization code {code}");
	};
	variants.get(variant).map(|suffix| format!("{}{bits}{suffix}", if family == 0 { "Q" } else { "IQ" })).unwrap_or_else(|| format!("quantization code {code}"))
}
#[rustfmt::skip]
fn fp16(value: f32) -> u16 {
	let bits = value.to_bits();
	let sign = (bits >> 16 & 0x8000) as u16;
	let exponent = ((bits >> 23 & 0xff) as i32) - 112;
	let mantissa = bits & 0x7fffff;
	if exponent <= 0 {
		if exponent < -10 { return sign }
		let value = (mantissa | 0x800000) >> (1 - exponent);
		return sign | ((value + 0xfff + (value >> 13 & 1)) >> 13) as u16
	}
	if exponent >= 31 { return sign | 0x7c00 | u16::from(mantissa != 0) }
	let rounded = mantissa + 0xfff + (mantissa >> 13 & 1);
	if rounded & 0x800000 != 0 { return sign | ((exponent + 1).min(31) as u16) << 10 }
	sign | (exponent as u16) << 10 | (rounded >> 13) as u16
}
#[rustfmt::skip]
fn unfp16(value: u16) -> f32 {
	let sign = (u32::from(value) & 0x8000) << 16;
	let exponent = u32::from(value >> 10 & 31);
	let mantissa = u32::from(value & 1023);
	let bits = if exponent == 0 {
		if mantissa == 0 { sign } else {
			let shift = mantissa.leading_zeros() - 21;
			sign | (113 - shift) << 23 | (mantissa << (shift + 13) & 0x7fffff)
		}
	} else if exponent == 31 { sign | 0x7f800000 | mantissa << 13 }
	else { sign | (exponent + 112) << 23 | mantissa << 13 };
	f32::from_bits(bits)
}
fn put_half(output: &mut Vec<u8>, value: f32) {
	output.extend(fp16(value).to_le_bytes())
}
fn half(input: &[u8]) -> f32 {
	unfp16(u16::from_le_bytes([input[0], input[1]]))
}
fn qround(value: f32) -> f32 {
	(((value + 12582912.0).to_bits() as i32 & 0x007fffff) - 0x00400000) as f32
}
fn positive_max(values: &[f32]) -> f32 {
	values.iter().fold(0.0, |maximum, value| if *value > maximum { *value } else { maximum })
}
#[rustfmt::skip]
fn qkx2(values: &[f32], weights: &[f32], levels: i32, range: (f32, f32, usize), mad: bool, codes: &mut [u8]) -> (f32, f32) {
	let (mut minimum, mut maximum, mut sum_w, mut sum_x) = (values[0], values[0], weights[0], weights[0] * values[0]);
	for index in 1..values.len() { if values[index] < minimum { minimum = values[index] } if values[index] > maximum { maximum = values[index] } sum_w += weights[index]; sum_x += weights[index] * values[index] }
	if minimum > 0.0 { minimum = 0.0 }
	if maximum == minimum { codes.fill(0); return (0.0, -minimum) }
	let mut inverse = levels as f32 / (maximum - minimum); let mut scale = 1.0 / inverse; let mut best_error = 0.0;
	for index in 0..values.len() { codes[index] = qround(inverse * (values[index] - minimum)).max(0.0).min(levels as f32) as u8; let difference = scale * f32::from(codes[index]) + minimum - values[index]; best_error += weights[index] * if mad { difference.abs() } else { difference * difference } }
	let mut trial = vec![0_u8; values.len()];
	for step in 0..=range.2 {
		inverse = (range.0 + range.1 * step as f32 + levels as f32) / (maximum - minimum);
		let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0, 0.0, 0.0);
		for index in 0..values.len() { trial[index] = qround(inverse * (values[index] - minimum)).max(0.0).min(levels as f32) as u8; let code = f32::from(trial[index]); sum_l += weights[index] * code; sum_l2 += weights[index] * code * code; sum_xl += weights[index] * code * values[index] }
		let denominator = sum_w * sum_l2 - sum_l * sum_l;
		if denominator > 0.0 {
			let mut candidate_scale = (sum_w * sum_xl - sum_x * sum_l) / denominator;
			let mut candidate_minimum = (sum_l2 * sum_x - sum_l * sum_xl) / denominator;
			if candidate_minimum > 0.0 { candidate_minimum = 0.0; candidate_scale = sum_xl / sum_l2 }
			let mut error = 0.0; for index in 0..values.len() { let difference = candidate_scale * f32::from(trial[index]) + candidate_minimum - values[index]; error += weights[index] * if mad { difference.abs() } else { difference * difference } }
			if error < best_error { codes.copy_from_slice(&trial); best_error = error; scale = candidate_scale; minimum = candidate_minimum }
		}
	}
	(scale, -minimum)
}
#[rustfmt::skip]
fn q3(values: &[f32], codes: &mut [i8]) -> f32 {
	let (mut maximum, mut absolute) = (0.0_f32, 0.0_f32);
	for value in values { let candidate = value.abs(); if candidate > absolute { absolute = candidate; maximum = *value } }
	if absolute < 1.0e-15 { codes.fill(0); return 0.0 }
	let inverse = -4.0 / maximum;
	let (mut sum_lx, mut sum_l2) = (0.0, 0.0);
	for index in 0..values.len() { let code = qround(inverse * values[index]).max(-4.0).min(3.0); codes[index] = code as i8; let weight = values[index] * values[index]; sum_lx += weight * values[index] * code; sum_l2 += weight * code * code }
	for _ in 0..5 {
		let mut changed = 0;
		for index in 0..values.len() {
			let value = values[index]; let code = f32::from(codes[index]); let weight = value * value; let mut reduced_lx = sum_lx - weight * value * code;
			if reduced_lx > 0.0 { let mut reduced_l2 = sum_l2 - weight * code * code; let candidate = qround(value * reduced_l2 / reduced_lx).max(-4.0).min(3.0); if candidate != code { reduced_lx += weight * value * candidate; reduced_l2 += weight * candidate * candidate; if reduced_l2 > 0.0 && reduced_lx * reduced_lx * sum_l2 > sum_lx * sum_lx * reduced_l2 { codes[index] = candidate as i8; sum_lx = reduced_lx; sum_l2 = reduced_l2; changed += 1 } } }
		}
		if changed == 0 { break }
	}
	for code in codes { *code += 4 }
	if sum_l2 > 0.0 { sum_lx / sum_l2 } else { 0.0 }
}
#[rustfmt::skip]
fn qx(values: &[f32], levels: i32, codes: &mut [i8]) -> f32 {
	let (mut maximum, mut absolute) = (0.0_f32, 0.0_f32);
	for value in values { let candidate = value.abs(); if candidate > absolute { absolute = candidate; maximum = *value } }
	if absolute < 1.0e-15 { codes.fill(0); return 0.0 }
	let mut inverse = -(levels as f32) / maximum;
	let (mut sum_lx, mut sum_l2) = (0.0, 0.0);
	for index in 0..values.len() { let signed = qround(inverse * values[index]).max(-(levels as f32)).min((levels - 1) as f32); codes[index] = signed as i8 + levels as i8; let weight = values[index] * values[index]; sum_lx += weight * values[index] * signed; sum_l2 += weight * signed * signed }
	let mut scale = if sum_l2 == 0.0 { 0.0 } else { sum_lx / sum_l2 };
	let mut best = scale * sum_lx;
	for step in -9..=9 {
		if step == 0 { continue }
		inverse = -(levels as f32 + 0.1 * step as f32) / maximum;
		(sum_lx, sum_l2) = (0.0, 0.0);
		for value in values { let code = qround(inverse * value).max(-(levels as f32)).min((levels - 1) as f32); let weight = value * value; sum_lx += weight * value * code; sum_l2 += weight * code * code }
		if sum_l2 > 0.0 && sum_lx * sum_lx > best * sum_l2 {
			for (value, code) in values.iter().zip(codes.iter_mut()) { *code = qround(inverse * value).max(-(levels as f32)).min((levels - 1) as f32) as i8 + levels as i8 }
			scale = sum_lx / sum_l2; best = scale * sum_lx
		}
	}
	scale
}
fn k_scale(metadata: &[u8], block: usize) -> (u8, u8) {
	if block < 4 {
		(metadata[block] & 63, metadata[block + 4] & 63)
	} else {
		((metadata[block + 4] & 15) | (metadata[block - 4] >> 6) << 4, (metadata[block + 4] >> 4) | (metadata[block] >> 6) << 4)
	}
}
const IQ4: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const IQ3_XXS: [u16; 256] = [
	0, 2, 4, 9, 11, 15, 16, 18, 25, 34, 59, 61, 65, 67, 72, 74, 81, 85, 88, 90, 97, 108, 120, 128, 130, 132, 137, 144, 146, 153, 155, 159, 169, 175, 189, 193, 199, 200, 202, 213, 248, 267, 287,
	292, 303, 315, 317, 321, 327, 346, 362, 413, 436, 456, 460, 462, 483, 497, 513, 515, 520, 522, 529, 531, 536, 538, 540, 551, 552, 576, 578, 585, 592, 594, 641, 643, 648, 650, 657, 664, 698,
	704, 706, 720, 729, 742, 758, 769, 773, 808, 848, 852, 870, 889, 901, 978, 992, 1024, 1026, 1033, 1035, 1040, 1042, 1046, 1049, 1058, 1089, 1091, 1093, 1096, 1098, 1105, 1112, 1139, 1143, 1144,
	1152, 1154, 1161, 1167, 1168, 1170, 1183, 1184, 1197, 1217, 1224, 1228, 1272, 1276, 1309, 1323, 1347, 1367, 1377, 1404, 1473, 1475, 1486, 1509, 1537, 1544, 1546, 1553, 1555, 1576, 1589, 1594,
	1600, 1602, 1616, 1625, 1636, 1638, 1665, 1667, 1672, 1685, 1706, 1722, 1737, 1755, 1816, 1831, 1850, 1856, 1862, 1874, 1901, 1932, 1950, 1971, 2011, 2032, 2052, 2063, 2077, 2079, 2091, 2095,
	2172, 2192, 2207, 2208, 2224, 2230, 2247, 2277, 2308, 2345, 2356, 2389, 2403, 2424, 2501, 2504, 2506, 2520, 2570, 2593, 2616, 2624, 2630, 2646, 2669, 2700, 2714, 2746, 2754, 2795, 2824, 2835,
	2839, 2874, 2882, 2905, 2984, 3028, 3042, 3092, 3108, 3110, 3124, 3153, 3185, 3215, 3252, 3288, 3294, 3364, 3397, 3434, 3483, 3523, 3537, 3587, 3589, 3591, 3592, 3610, 3626, 3670, 3680, 3722,
	3749, 3754, 3776, 3789, 3803, 3824, 3857, 3873, 3904, 3906, 3924, 3992,
];
const IQ3_S: [u16; 512] = [
	0, 1, 2, 5, 7, 8, 9, 10, 12, 14, 16, 17, 21, 27, 32, 34, 37, 39, 41, 43, 48, 50, 57, 60, 63, 64, 65, 66, 68, 72, 73, 77, 80, 83, 87, 89, 93, 100, 113, 117, 122, 128, 129, 133, 135, 136, 139,
	142, 145, 149, 152, 156, 162, 165, 167, 169, 171, 184, 187, 195, 201, 205, 208, 210, 217, 219, 222, 228, 232, 234, 247, 249, 253, 256, 267, 271, 273, 276, 282, 288, 291, 297, 312, 322, 324,
	336, 338, 342, 347, 353, 357, 359, 374, 379, 390, 393, 395, 409, 426, 441, 448, 450, 452, 464, 466, 470, 475, 488, 492, 512, 513, 514, 516, 520, 521, 523, 525, 527, 528, 530, 537, 540, 542,
	556, 558, 561, 570, 576, 577, 579, 582, 584, 588, 593, 600, 603, 609, 616, 618, 632, 638, 640, 650, 653, 655, 656, 660, 666, 672, 675, 685, 688, 698, 705, 708, 711, 712, 715, 721, 727, 728,
	732, 737, 754, 760, 771, 773, 778, 780, 793, 795, 802, 806, 808, 812, 833, 840, 843, 849, 856, 858, 873, 912, 916, 919, 932, 934, 961, 963, 968, 970, 977, 989, 993, 1010, 1016, 1024, 1025,
	1027, 1029, 1031, 1032, 1034, 1036, 1038, 1041, 1043, 1047, 1048, 1050, 1057, 1059, 1061, 1064, 1066, 1079, 1080, 1083, 1085, 1088, 1090, 1096, 1099, 1103, 1106, 1109, 1113, 1116, 1122, 1129,
	1153, 1156, 1159, 1169, 1171, 1176, 1183, 1185, 1195, 1199, 1209, 1212, 1216, 1218, 1221, 1225, 1234, 1236, 1241, 1243, 1250, 1256, 1270, 1281, 1287, 1296, 1299, 1306, 1309, 1313, 1338, 1341,
	1348, 1353, 1362, 1375, 1376, 1387, 1400, 1408, 1410, 1415, 1425, 1453, 1457, 1477, 1481, 1494, 1496, 1507, 1512, 1538, 1545, 1547, 1549, 1551, 1554, 1561, 1563, 1565, 1570, 1572, 1575, 1577,
	1587, 1593, 1601, 1603, 1605, 1612, 1617, 1619, 1632, 1648, 1658, 1662, 1664, 1674, 1680, 1690, 1692, 1704, 1729, 1736, 1740, 1745, 1747, 1751, 1752, 1761, 1763, 1767, 1773, 1787, 1795, 1801,
	1806, 1810, 1817, 1834, 1840, 1844, 1857, 1864, 1866, 1877, 1882, 1892, 1902, 1915, 1934, 1953, 1985, 1987, 2000, 2002, 2013, 2048, 2052, 2058, 2064, 2068, 2071, 2074, 2081, 2088, 2104, 2114,
	2119, 2121, 2123, 2130, 2136, 2141, 2147, 2153, 2157, 2177, 2179, 2184, 2189, 2193, 2203, 2208, 2223, 2226, 2232, 2244, 2249, 2251, 2256, 2258, 2265, 2269, 2304, 2306, 2324, 2335, 2336, 2361,
	2373, 2375, 2385, 2418, 2443, 2460, 2480, 2504, 2509, 2520, 2531, 2537, 2562, 2568, 2572, 2578, 2592, 2596, 2599, 2602, 2614, 2620, 2625, 2627, 2629, 2634, 2641, 2650, 2682, 2688, 2697, 2707,
	2712, 2718, 2731, 2754, 2759, 2760, 2775, 2788, 2793, 2805, 2811, 2817, 2820, 2832, 2842, 2854, 2890, 2902, 2921, 2923, 2978, 3010, 3012, 3026, 3081, 3083, 3085, 3097, 3099, 3120, 3136, 3152,
	3159, 3188, 3210, 3228, 3234, 3245, 3250, 3256, 3264, 3276, 3281, 3296, 3349, 3363, 3378, 3392, 3395, 3420, 3440, 3461, 3488, 3529, 3531, 3584, 3588, 3591, 3600, 3602, 3614, 3616, 3628, 3634,
	3650, 3657, 3668, 3683, 3685, 3713, 3716, 3720, 3726, 3729, 3736, 3753, 3778, 3802, 3805, 3819, 3841, 3845, 3851, 3856, 3880, 3922, 3938, 3970, 3993, 4032,
];
const IQ2_XXS: [u16; 256] = [
	0, 2, 5, 8, 10, 17, 20, 32, 34, 40, 42, 65, 68, 80, 88, 97, 100, 128, 130, 138, 162, 257, 260, 272, 277, 320, 388, 408, 512, 514, 546, 642, 1025, 1028, 1040, 1057, 1060, 1088, 1090, 1096, 1120,
	1153, 1156, 1168, 1188, 1280, 1282, 1288, 1312, 1350, 1385, 1408, 1425, 1545, 1552, 1600, 1668, 1700, 2048, 2053, 2056, 2068, 2088, 2113, 2116, 2128, 2130, 2184, 2308, 2368, 2562, 2580, 4097,
	4100, 4112, 4129, 4160, 4192, 4228, 4240, 4245, 4352, 4360, 4384, 4432, 4442, 4480, 4644, 4677, 5120, 5128, 5152, 5157, 5193, 5248, 5400, 5474, 5632, 5654, 6145, 6148, 6160, 6208, 6273, 6400,
	6405, 6560, 6737, 8192, 8194, 8202, 8260, 8289, 8320, 8322, 8489, 8520, 8704, 8706, 9217, 9220, 9232, 9280, 9302, 9472, 9537, 9572, 9872, 10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408,
	16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516, 16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561, 17682, 17700, 17920,
	18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488, 20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100,
	22528, 22545, 22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874, 25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048,
	33088, 33297, 33793, 33796, 33808, 33813, 33856, 33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142, 37248, 37445, 37888, 37922,
	37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
];
const IQ2_XS: [u16; 512] = [
	0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70, 73, 80, 82, 85, 88, 97, 100, 128, 130, 133, 136, 145, 148, 153, 160, 257, 260, 262, 265, 272, 274, 277, 280, 282, 289, 292, 320, 322,
	325, 328, 337, 340, 352, 360, 385, 388, 400, 512, 514, 517, 520, 529, 532, 544, 577, 580, 592, 597, 640, 650, 1025, 1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1088, 1090, 1093, 1096,
	1105, 1108, 1110, 1120, 1153, 1156, 1168, 1280, 1282, 1285, 1288, 1297, 1300, 1312, 1345, 1348, 1360, 1377, 1408, 1537, 1540, 1552, 1574, 1600, 1602, 1668, 2048, 2050, 2053, 2056, 2058, 2065,
	2068, 2080, 2085, 2113, 2116, 2128, 2136, 2176, 2208, 2218, 2305, 2308, 2320, 2368, 2433, 2441, 2560, 2592, 2600, 2710, 2720, 4097, 4100, 4102, 4105, 4112, 4114, 4117, 4120, 4129, 4132, 4160,
	4162, 4165, 4168, 4177, 4180, 4192, 4202, 4225, 4228, 4240, 4352, 4354, 4357, 4360, 4369, 4372, 4384, 4417, 4420, 4432, 4480, 4500, 4502, 4609, 4612, 4614, 4624, 4672, 4704, 5120, 5122, 5125,
	5128, 5137, 5140, 5152, 5185, 5188, 5193, 5200, 5220, 5248, 5377, 5380, 5392, 5440, 5632, 5652, 5705, 6145, 6148, 6160, 6162, 6208, 6228, 6278, 6400, 6405, 6502, 6737, 6825, 8192, 8194, 8197,
	8200, 8202, 8209, 8212, 8224, 8257, 8260, 8272, 8320, 8352, 8449, 8452, 8464, 8512, 8520, 8549, 8704, 8738, 8832, 8872, 9217, 9220, 9232, 9257, 9280, 9472, 9537, 9554, 9625, 9729, 9754, 9894,
	10240, 10248, 10250, 10272, 10325, 10376, 10402, 10600, 10640, 10760, 10784, 10882, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16417, 16420, 16448, 16450, 16453,
	16456, 16458, 16465, 16468, 16480, 16485, 16513, 16516, 16528, 16640, 16642, 16645, 16648, 16657, 16660, 16672, 16705, 16708, 16720, 16768, 16773, 16802, 16897, 16900, 16912, 16914, 16937,
	16960, 17408, 17410, 17413, 17416, 17425, 17428, 17433, 17440, 17473, 17476, 17488, 17536, 17556, 17665, 17668, 17680, 17700, 17728, 17818, 17920, 17930, 17988, 18000, 18433, 18436, 18448,
	18496, 18501, 18516, 18530, 18688, 18705, 18756, 18768, 18793, 18948, 20480, 20482, 20485, 20488, 20497, 20500, 20512, 20520, 20545, 20548, 20560, 20608, 20737, 20740, 20752, 20757, 20800,
	20802, 20992, 21060, 21162, 21505, 21508, 21520, 21537, 21568, 21600, 21633, 21665, 21760, 21768, 21888, 21896, 22049, 22120, 22177, 22528, 22548, 22593, 22608, 22681, 22810, 22848, 22850,
	23173, 24577, 24580, 24592, 24640, 24660, 24674, 24710, 24745, 24832, 25124, 25162, 25234, 25600, 25622, 25872, 25920, 25925, 26020, 26625, 26730, 26917, 27142, 27220, 27234, 32768, 32770,
	32773, 32776, 32785, 32788, 32800, 32810, 32833, 32836, 32848, 32896, 32898, 32936, 32938, 33025, 33028, 33030, 33040, 33088, 33105, 33113, 33280, 33312, 33408, 33410, 33440, 33448, 33793,
	33796, 33808, 33810, 33813, 33856, 33888, 33929, 34048, 34116, 34213, 34328, 34410, 34816, 34824, 34853, 34906, 34944, 34946, 34984, 35078, 35362, 35456, 35464, 35478, 35496, 36865, 36868,
	36880, 36928, 36950, 36996, 37120, 37154, 37220, 37462, 37513, 37888, 37893, 37956, 37968, 37976, 38185, 38288, 38290, 38465, 38993, 39078, 39241, 39445, 39520, 40960, 40962, 40968, 40970,
	40992, 41002, 41120, 41297, 41305, 41382, 41472, 41474, 41480, 41514, 41600, 41632, 42048, 42133, 42597, 42648, 43018, 43040, 43042, 43048, 43168, 43176, 43268, 43396, 43398, 43560, 43562,
	43665, 43690,
];
const IQ1: [u16; 2048] = [
	0, 2, 5, 8, 10, 17, 21, 32, 34, 40, 42, 69, 81, 84, 86, 101, 128, 130, 136, 138, 149, 160, 162, 168, 170, 260, 261, 273, 276, 278, 281, 282, 293, 321, 326, 329, 338, 341, 346, 353, 356, 358,
	360, 389, 401, 404, 406, 421, 512, 514, 520, 522, 533, 544, 546, 552, 554, 581, 593, 601, 612, 617, 640, 642, 648, 650, 657, 661, 665, 672, 674, 680, 682, 1041, 1044, 1046, 1061, 1089, 1097,
	1109, 1114, 1124, 1125, 1169, 1177, 1189, 1281, 1284, 1285, 1286, 1301, 1304, 1306, 1321, 1344, 1349, 1354, 1360, 1361, 1364, 1365, 1366, 1369, 1376, 1378, 1381, 1384, 1386, 1409, 1425, 1429,
	1432, 1434, 1441, 1444, 1445, 1446, 1449, 1556, 1561, 1601, 1604, 1616, 1618, 1621, 1624, 1632, 1633, 1638, 1641, 1669, 1681, 1684, 1689, 2048, 2050, 2056, 2058, 2069, 2080, 2082, 2088, 2090,
	2117, 2129, 2134, 2149, 2176, 2178, 2184, 2186, 2197, 2208, 2210, 2216, 2218, 2309, 2321, 2324, 2329, 2340, 2341, 2369, 2384, 2385, 2389, 2401, 2404, 2409, 2449, 2452, 2454, 2457, 2469, 2560,
	2562, 2568, 2570, 2581, 2592, 2594, 2600, 2602, 2629, 2641, 2649, 2657, 2661, 2688, 2690, 2693, 2696, 2698, 2709, 2720, 2722, 2728, 2730, 4112, 4113, 4116, 4121, 4132, 4133, 4161, 4164, 4176,
	4181, 4184, 4193, 4196, 4197, 4201, 4241, 4244, 4246, 4257, 4261, 4353, 4356, 4358, 4361, 4368, 4370, 4373, 4376, 4385, 4388, 4393, 4421, 4426, 4432, 4433, 4434, 4436, 4437, 4438, 4441, 4448,
	4453, 4484, 4498, 4501, 4513, 4516, 4625, 4628, 4630, 4645, 4672, 4678, 4681, 4690, 4693, 4696, 4698, 4708, 4710, 4741, 4753, 4756, 4758, 4773, 5121, 5126, 5129, 5140, 5141, 5144, 5145, 5153,
	5158, 5185, 5189, 5190, 5192, 5194, 5201, 5204, 5205, 5206, 5209, 5218, 5221, 5224, 5252, 5257, 5264, 5268, 5269, 5272, 5273, 5274, 5281, 5284, 5285, 5289, 5378, 5381, 5386, 5393, 5396, 5397,
	5398, 5401, 5408, 5410, 5413, 5416, 5418, 5441, 5444, 5445, 5446, 5457, 5458, 5460, 5461, 5462, 5465, 5466, 5473, 5476, 5477, 5478, 5481, 5504, 5506, 5508, 5509, 5512, 5514, 5520, 5521, 5524,
	5525, 5526, 5529, 5530, 5536, 5538, 5541, 5633, 5636, 5637, 5638, 5653, 5654, 5656, 5658, 5665, 5670, 5696, 5698, 5700, 5701, 5704, 5706, 5713, 5717, 5718, 5720, 5721, 5729, 5732, 5733, 5736,
	5737, 5738, 5766, 5770, 5778, 5781, 5796, 5801, 6161, 6166, 6181, 6209, 6212, 6214, 6217, 6224, 6229, 6232, 6234, 6240, 6241, 6244, 6246, 6249, 6277, 6289, 6292, 6309, 6416, 6418, 6421, 6426,
	6433, 6437, 6466, 6468, 6469, 6472, 6481, 6484, 6485, 6486, 6489, 6490, 6496, 6501, 6506, 6537, 6545, 6546, 6549, 6552, 6561, 6566, 6569, 6665, 6678, 6692, 6694, 6724, 6726, 6729, 6736, 6738,
	6741, 6744, 6753, 6758, 6761, 6789, 6801, 6806, 6810, 8192, 8194, 8200, 8202, 8213, 8224, 8226, 8229, 8232, 8234, 8261, 8273, 8281, 8289, 8293, 8320, 8322, 8328, 8330, 8341, 8352, 8354, 8357,
	8360, 8362, 8453, 8465, 8468, 8473, 8485, 8514, 8516, 8521, 8533, 8536, 8538, 8545, 8548, 8549, 8550, 8581, 8592, 8598, 8601, 8613, 8705, 8712, 8714, 8721, 8725, 8736, 8738, 8744, 8746, 8773,
	8785, 8790, 8793, 8805, 8833, 8840, 8842, 8849, 8853, 8864, 8866, 8872, 8874, 9221, 9236, 9238, 9241, 9253, 9284, 9285, 9286, 9289, 9298, 9301, 9304, 9306, 9318, 9349, 9361, 9364, 9369, 9377,
	9381, 9481, 9493, 9505, 9513, 9536, 9541, 9544, 9553, 9556, 9557, 9561, 9570, 9573, 9576, 9609, 9616, 9620, 9621, 9624, 9626, 9633, 9636, 9638, 9641, 9733, 9744, 9746, 9753, 9765, 9793, 9801,
	9813, 9824, 9825, 9833, 9860, 9862, 9872, 9882, 10240, 10242, 10248, 10250, 10261, 10272, 10274, 10280, 10282, 10309, 10321, 10324, 10341, 10368, 10370, 10376, 10378, 10400, 10402, 10408,
	10410, 10505, 10513, 10516, 10521, 10533, 10566, 10569, 10578, 10581, 10593, 10596, 10598, 10601, 10629, 10640, 10646, 10649, 10660, 10661, 10752, 10754, 10760, 10762, 10784, 10786, 10792,
	10794, 10821, 10833, 10838, 10841, 10853, 10880, 10882, 10888, 10890, 10901, 10912, 10914, 10920, 10922, 16389, 16401, 16406, 16421, 16457, 16466, 16469, 16472, 16474, 16481, 16484, 16486,
	16532, 16537, 16545, 16550, 16640, 16641, 16644, 16646, 16649, 16658, 16661, 16662, 16664, 16666, 16673, 16678, 16681, 16709, 16712, 16714, 16721, 16724, 16725, 16726, 16729, 16730, 16741,
	16744, 16746, 16769, 16772, 16774, 16784, 16786, 16789, 16800, 16801, 16802, 16901, 16913, 16916, 16918, 16933, 16961, 16978, 16981, 16986, 16996, 17001, 17033, 17044, 17061, 17409, 17429,
	17433, 17449, 17477, 17480, 17482, 17489, 17492, 17493, 17494, 17505, 17506, 17509, 17512, 17514, 17537, 17542, 17545, 17552, 17554, 17557, 17568, 17569, 17577, 17665, 17666, 17669, 17674,
	17681, 17684, 17685, 17686, 17689, 17696, 17701, 17706, 17729, 17732, 17733, 17734, 17737, 17744, 17745, 17748, 17749, 17750, 17752, 17753, 17761, 17764, 17765, 17766, 17769, 17794, 17796,
	17797, 17800, 17809, 17812, 17813, 17814, 17817, 17818, 17829, 17832, 17834, 17921, 17925, 17929, 17940, 17941, 17944, 17946, 17953, 17956, 17961, 17984, 17986, 17989, 17992, 18000, 18001,
	18002, 18005, 18006, 18009, 18018, 18021, 18024, 18049, 18053, 18058, 18068, 18069, 18081, 18084, 18086, 18437, 18449, 18453, 18458, 18469, 18498, 18505, 18512, 18517, 18520, 18529, 18532,
	18534, 18537, 18565, 18577, 18580, 18582, 18585, 18597, 18689, 18693, 18694, 18698, 18704, 18708, 18709, 18712, 18721, 18724, 18726, 18752, 18757, 18762, 18769, 18770, 18772, 18773, 18774,
	18777, 18784, 18786, 18789, 18790, 18794, 18822, 18825, 18834, 18837, 18838, 18840, 18849, 18852, 18854, 18857, 18966, 19012, 19014, 19017, 19029, 19032, 19034, 19044, 19049, 19092, 19109,
	20481, 20484, 20485, 20486, 20489, 20498, 20501, 20506, 20513, 20516, 20521, 20544, 20549, 20552, 20561, 20564, 20565, 20566, 20569, 20581, 20584, 20614, 20617, 20629, 20632, 20640, 20641,
	20646, 20649, 20741, 20744, 20745, 20746, 20753, 20756, 20757, 20758, 20760, 20761, 20768, 20773, 20774, 20776, 20778, 20801, 20804, 20805, 20806, 20809, 20816, 20817, 20818, 20820, 20821,
	20822, 20824, 20825, 20826, 20833, 20836, 20837, 20838, 20841, 20866, 20869, 20881, 20884, 20885, 20886, 20889, 20896, 20901, 20906, 20993, 20998, 21010, 21013, 21018, 21025, 21028, 21058,
	21061, 21066, 21073, 21076, 21077, 21078, 21081, 21090, 21093, 21125, 21136, 21138, 21141, 21145, 21146, 21156, 21508, 21509, 21521, 21524, 21525, 21526, 21528, 21529, 21537, 21541, 21544,
	21546, 21569, 21572, 21573, 21574, 21577, 21578, 21584, 21585, 21588, 21589, 21590, 21592, 21593, 21594, 21601, 21602, 21604, 21605, 21606, 21609, 21632, 21640, 21642, 21649, 21652, 21653,
	21654, 21657, 21665, 21668, 21669, 21674, 21761, 21762, 21764, 21765, 21766, 21769, 21776, 21777, 21778, 21780, 21781, 21782, 21785, 21786, 21793, 21796, 21797, 21798, 21801, 21824, 21825,
	21826, 21828, 21829, 21830, 21832, 21833, 21840, 21841, 21842, 21844, 21845, 21846, 21848, 21849, 21850, 21856, 21857, 21860, 21861, 21862, 21864, 21865, 21866, 21889, 21892, 21893, 21897,
	21898, 21904, 21905, 21908, 21909, 21910, 21912, 21913, 21921, 21924, 21925, 21926, 21929, 22016, 22017, 22018, 22020, 22022, 22024, 22025, 22033, 22036, 22037, 22040, 22041, 22048, 22049,
	22050, 22052, 22053, 22054, 22056, 22057, 22081, 22085, 22086, 22088, 22089, 22090, 22096, 22097, 22098, 22100, 22101, 22102, 22104, 22105, 22106, 22113, 22116, 22117, 22121, 22146, 22149,
	22150, 22152, 22153, 22154, 22161, 22165, 22170, 22178, 22181, 22182, 22184, 22185, 22532, 22533, 22534, 22537, 22544, 22549, 22552, 22561, 22570, 22597, 22600, 22602, 22609, 22612, 22613,
	22614, 22616, 22617, 22624, 22626, 22628, 22629, 22658, 22665, 22672, 22674, 22677, 22680, 22689, 22697, 22785, 22786, 22789, 22794, 22801, 22804, 22805, 22806, 22809, 22821, 22849, 22852,
	22853, 22854, 22857, 22864, 22865, 22866, 22868, 22869, 22870, 22872, 22873, 22874, 22881, 22884, 22885, 22886, 22889, 22913, 22917, 22921, 22929, 22932, 22933, 22934, 22936, 22937, 22949,
	23044, 23048, 23061, 23066, 23072, 23077, 23078, 23081, 23109, 23112, 23113, 23121, 23125, 23126, 23128, 23129, 23138, 23141, 23144, 23146, 23169, 23178, 23186, 23189, 23190, 23192, 23194,
	23201, 24581, 24596, 24598, 24601, 24613, 24644, 24656, 24661, 24662, 24664, 24666, 24673, 24676, 24678, 24681, 24705, 24726, 24741, 24833, 24836, 24838, 24841, 24850, 24853, 24865, 24866,
	24870, 24873, 24901, 24905, 24913, 24917, 24918, 24921, 24933, 24934, 24938, 24964, 24970, 24978, 24981, 24993, 24998, 25001, 25105, 25110, 25113, 25152, 25153, 25158, 25173, 25174, 25176,
	25184, 25221, 25233, 25238, 25253, 25617, 25618, 25621, 25622, 25626, 25633, 25638, 25641, 25664, 25666, 25669, 25672, 25674, 25681, 25684, 25685, 25686, 25689, 25690, 25696, 25698, 25701,
	25732, 25733, 25737, 25744, 25746, 25748, 25749, 25750, 25752, 25754, 25761, 25764, 25769, 25861, 25864, 25866, 25873, 25877, 25878, 25881, 25924, 25925, 25926, 25929, 25936, 25937, 25940,
	25941, 25942, 25945, 25953, 25956, 25957, 25958, 25961, 25990, 25993, 25994, 26001, 26005, 26006, 26009, 26010, 26018, 26021, 26022, 26024, 26114, 26121, 26133, 26144, 26150, 26152, 26153,
	26176, 26181, 26184, 26186, 26193, 26196, 26197, 26198, 26200, 26202, 26208, 26213, 26216, 26240, 26242, 26245, 26250, 26260, 26262, 26264, 26265, 26272, 26276, 26278, 26282, 26646, 26649,
	26661, 26689, 26706, 26709, 26714, 26721, 26729, 26757, 26769, 26776, 26790, 26881, 26884, 26896, 26901, 26913, 26916, 26918, 26921, 26944, 26945, 26949, 26950, 26952, 26961, 26964, 26965,
	26966, 26969, 26976, 26981, 26986, 27010, 27012, 27018, 27029, 27041, 27044, 27045, 27049, 27153, 27158, 27160, 27201, 27204, 27209, 27216, 27221, 27224, 27226, 27236, 27237, 27241, 27270,
	27284, 27288, 27290, 27302, 32768, 32770, 32776, 32778, 32800, 32802, 32808, 32810, 32837, 32848, 32849, 32852, 32854, 32857, 32869, 32896, 32898, 32904, 32906, 32917, 32928, 32930, 32936,
	32938, 33029, 33041, 33044, 33046, 33049, 33061, 33089, 33092, 33097, 33104, 33106, 33109, 33110, 33112, 33113, 33124, 33126, 33129, 33157, 33161, 33172, 33174, 33177, 33189, 33280, 33282,
	33288, 33290, 33301, 33312, 33314, 33320, 33322, 33361, 33364, 33369, 33381, 33408, 33410, 33416, 33418, 33429, 33440, 33442, 33448, 33450, 33812, 33817, 33857, 33860, 33873, 33877, 33882,
	33889, 33892, 33897, 33940, 33945, 34049, 34057, 34066, 34069, 34074, 34086, 34089, 34112, 34113, 34117, 34120, 34129, 34132, 34133, 34134, 34137, 34138, 34149, 34150, 34152, 34154, 34177,
	34180, 34182, 34185, 34192, 34194, 34197, 34200, 34214, 34321, 34326, 34329, 34341, 34369, 34372, 34377, 34378, 34384, 34389, 34393, 34394, 34401, 34406, 34410, 34437, 34449, 34458, 34468,
	34816, 34818, 34824, 34826, 34837, 34848, 34850, 34856, 34858, 34881, 34885, 34897, 34900, 34905, 34917, 34921, 34944, 34946, 34952, 34954, 34965, 34976, 34978, 34984, 34986, 35077, 35078,
	35089, 35092, 35094, 35109, 35137, 35140, 35142, 35145, 35152, 35154, 35157, 35162, 35169, 35172, 35205, 35222, 35225, 35237, 35328, 35330, 35336, 35338, 35349, 35360, 35362, 35368, 35370,
	35397, 35409, 35412, 35414, 35456, 35458, 35464, 35466, 35477, 35488, 35490, 35496, 35498, 36869, 36881, 36886, 36888, 36889, 36901, 36929, 36934, 36937, 36949, 36952, 36954, 36969, 36970,
	36997, 37009, 37012, 37014, 37017, 37029, 37121, 37124, 37126, 37129, 37136, 37141, 37144, 37146, 37153, 37156, 37158, 37161, 37184, 37189, 37200, 37201, 37204, 37205, 37206, 37209, 37218,
	37221, 37252, 37254, 37266, 37269, 37272, 37281, 37284, 37286, 37289, 37381, 37393, 37396, 37401, 37413, 37444, 37446, 37449, 37456, 37458, 37461, 37464, 37478, 37481, 37509, 37524, 37526,
	37545, 37889, 37892, 37894, 37904, 37909, 37912, 37926, 37952, 37962, 37969, 37972, 37973, 37974, 37976, 37977, 37984, 37985, 37986, 37989, 38020, 38022, 38034, 38036, 38037, 38040, 38049,
	38057, 38144, 38149, 38152, 38154, 38160, 38161, 38164, 38165, 38166, 38169, 38177, 38181, 38185, 38186, 38209, 38212, 38213, 38214, 38217, 38224, 38225, 38226, 38228, 38229, 38230, 38232,
	38233, 38234, 38241, 38244, 38245, 38246, 38249, 38273, 38277, 38280, 38289, 38290, 38292, 38293, 38294, 38297, 38298, 38304, 38306, 38309, 38312, 38314, 38401, 38404, 38416, 38421, 38425,
	38432, 38438, 38441, 38469, 38472, 38473, 38481, 38482, 38485, 38486, 38489, 38501, 38504, 38530, 38532, 38537, 38538, 38546, 38548, 38549, 38564, 38566, 38569, 38917, 38934, 38937, 38949,
	38977, 38982, 38992, 38994, 38997, 38998, 39002, 39012, 39013, 39045, 39057, 39062, 39065, 39077, 39172, 39174, 39177, 39184, 39186, 39189, 39192, 39194, 39200, 39201, 39204, 39206, 39232,
	39234, 39237, 39240, 39242, 39249, 39252, 39253, 39254, 39257, 39266, 39269, 39270, 39274, 39297, 39300, 39312, 39314, 39317, 39322, 39329, 39334, 39429, 39445, 39461, 39492, 39494, 39497,
	39504, 39509, 39512, 39521, 39557, 39569, 39572, 39573, 39574, 40960, 40962, 40968, 40970, 40981, 40992, 40994, 41000, 41002, 41029, 41041, 41044, 41046, 41049, 41088, 41090, 41096, 41098,
	41109, 41120, 41122, 41128, 41130, 41221, 41225, 41233, 41236, 41238, 41241, 41242, 41286, 41289, 41297, 41301, 41304, 41306, 41313, 41316, 41349, 41360, 41362, 41366, 41369, 41474, 41480,
	41482, 41488, 41497, 41506, 41512, 41514, 41541, 41553, 41558, 41561, 41573, 41600, 41602, 41608, 41610, 41621, 41632, 41634, 41640, 41642, 42009, 42021, 42049, 42052, 42064, 42068, 42069,
	42072, 42074, 42081, 42085, 42086, 42088, 42089, 42117, 42246, 42249, 42256, 42258, 42261, 42264, 42278, 42281, 42306, 42309, 42321, 42324, 42325, 42326, 42329, 42341, 42346, 42369, 42372,
	42373, 42374, 42377, 42386, 42389, 42392, 42501, 42513, 42518, 42522, 42529, 42533, 42564, 42566, 42570, 42578, 42581, 42582, 42584, 42592, 42594, 42630, 42640, 42645, 42646, 42649, 42657,
	42660, 42662, 43008, 43010, 43016, 43018, 43040, 43042, 43048, 43050, 43089, 43092, 43094, 43097, 43136, 43138, 43144, 43146, 43157, 43168, 43170, 43176, 43178, 43269, 43284, 43289, 43297,
	43301, 43329, 43344, 43349, 43354, 43361, 43366, 43369, 43408, 43414, 43520, 43522, 43528, 43530, 43552, 43554, 43560, 43562, 43601, 43604, 43606, 43648, 43650, 43656, 43658, 43669, 43680,
	43682, 43688, 43690,
];
const IQ2_S: [u16; 1024] = [
	0, 2, 5, 8, 10, 17, 20, 22, 25, 32, 34, 37, 40, 65, 68, 70, 73, 80, 82, 85, 88, 97, 100, 102, 105, 128, 130, 133, 136, 145, 148, 160, 165, 170, 257, 260, 262, 265, 272, 274, 277, 280, 289, 292,
	320, 322, 325, 328, 337, 340, 342, 345, 352, 357, 360, 385, 388, 400, 402, 405, 417, 420, 512, 514, 517, 520, 529, 532, 544, 554, 577, 580, 582, 585, 592, 597, 640, 645, 650, 660, 674, 1025,
	1028, 1030, 1033, 1040, 1042, 1045, 1048, 1057, 1060, 1062, 1065, 1088, 1090, 1093, 1096, 1098, 1105, 1108, 1110, 1113, 1120, 1122, 1125, 1153, 1156, 1158, 1161, 1168, 1173, 1176, 1185, 1188,
	1280, 1282, 1285, 1288, 1290, 1297, 1300, 1302, 1305, 1312, 1317, 1320, 1345, 1348, 1350, 1353, 1360, 1362, 1365, 1368, 1377, 1380, 1408, 1410, 1413, 1416, 1425, 1428, 1440, 1537, 1540, 1542,
	1545, 1552, 1557, 1600, 1605, 1608, 1617, 1620, 1632, 1665, 1668, 1680, 2048, 2050, 2053, 2056, 2065, 2068, 2070, 2073, 2080, 2085, 2090, 2113, 2116, 2118, 2121, 2128, 2130, 2133, 2136, 2145,
	2148, 2176, 2181, 2196, 2218, 2305, 2308, 2320, 2322, 2325, 2328, 2337, 2368, 2373, 2376, 2385, 2388, 2400, 2433, 2448, 2560, 2577, 2580, 2594, 2600, 2602, 2640, 2713, 4097, 4100, 4102, 4105,
	4112, 4114, 4117, 4120, 4129, 4132, 4134, 4160, 4162, 4165, 4168, 4177, 4180, 4182, 4185, 4192, 4194, 4197, 4200, 4225, 4228, 4230, 4240, 4245, 4248, 4257, 4260, 4352, 4354, 4357, 4360, 4362,
	4369, 4372, 4374, 4377, 4384, 4386, 4389, 4392, 4417, 4420, 4422, 4425, 4432, 4434, 4437, 4440, 4449, 4452, 4480, 4482, 4485, 4488, 4497, 4500, 4609, 4612, 4617, 4624, 4629, 4641, 4644, 4672,
	4677, 4689, 4692, 4737, 4740, 4752, 5120, 5122, 5125, 5128, 5137, 5140, 5142, 5145, 5152, 5157, 5160, 5185, 5188, 5190, 5193, 5200, 5202, 5205, 5208, 5217, 5220, 5248, 5250, 5253, 5256, 5265,
	5268, 5280, 5377, 5380, 5382, 5385, 5392, 5394, 5397, 5400, 5409, 5412, 5440, 5442, 5445, 5448, 5457, 5460, 5472, 5505, 5508, 5520, 5632, 5637, 5640, 5649, 5652, 5664, 5697, 5700, 5712, 5760,
	5802, 6145, 6148, 6150, 6153, 6160, 6165, 6168, 6177, 6208, 6210, 6213, 6216, 6225, 6228, 6240, 6273, 6276, 6400, 6402, 6405, 6408, 6417, 6420, 6432, 6465, 6468, 6480, 6505, 6562, 6660, 6672,
	6720, 6742, 8192, 8194, 8197, 8200, 8209, 8212, 8214, 8217, 8224, 8229, 8234, 8257, 8260, 8272, 8274, 8277, 8292, 8320, 8330, 8340, 8362, 8449, 8452, 8464, 8466, 8469, 8481, 8512, 8514, 8517,
	8529, 8532, 8544, 8577, 8580, 8592, 8704, 8714, 8738, 8744, 8746, 8772, 8784, 8840, 8842, 8872, 9217, 9220, 9222, 9225, 9232, 9237, 9240, 9249, 9252, 9280, 9282, 9285, 9288, 9297, 9300, 9312,
	9345, 9348, 9360, 9472, 9477, 9480, 9489, 9492, 9504, 9537, 9540, 9552, 9574, 9600, 9729, 9732, 9744, 9792, 9817, 10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497, 10500,
	10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16410, 16417, 16420, 16422, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470,
	16473, 16480, 16482, 16485, 16513, 16516, 16528, 16533, 16536, 16545, 16548, 16640, 16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674, 16677, 16705, 16708, 16710, 16713, 16720,
	16722, 16725, 16728, 16737, 16740, 16768, 16770, 16773, 16776, 16785, 16788, 16800, 16897, 16900, 16912, 16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992, 17025, 17028,
	17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430, 17433, 17440, 17442, 17445, 17448, 17473, 17476, 17478, 17481, 17488, 17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544,
	17553, 17556, 17568, 17665, 17668, 17670, 17673, 17680, 17682, 17685, 17688, 17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760, 17770, 17793, 17796, 17808, 17920, 17922, 17925,
	17928, 17937, 17940, 17952, 17985, 17988, 18000, 18048, 18085, 18433, 18436, 18441, 18448, 18450, 18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516, 18528, 18564, 18576,
	18688, 18690, 18693, 18696, 18705, 18708, 18720, 18753, 18756, 18768, 18816, 18838, 18945, 18948, 18960, 19008, 20480, 20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517,
	20520, 20545, 20548, 20550, 20553, 20560, 20562, 20565, 20568, 20577, 20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740, 20742, 20745, 20752, 20754, 20757, 20760, 20769, 20772,
	20800, 20802, 20805, 20808, 20817, 20820, 20832, 20865, 20868, 20880, 20992, 20997, 21000, 21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508, 21510, 21513, 21520, 21522,
	21525, 21528, 21537, 21540, 21568, 21570, 21573, 21576, 21585, 21588, 21600, 21633, 21636, 21648, 21760, 21762, 21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017, 22020,
	22032, 22054, 22080, 22528, 22530, 22533, 22536, 22545, 22548, 22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800, 22848, 23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592,
	24594, 24597, 24600, 24609, 24612, 24640, 24645, 24648, 24657, 24660, 24672, 24708, 24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900, 24912, 24960, 24985, 25092, 25104,
	25152, 25174, 25249, 25600, 25605, 25608, 25617, 25620, 25632, 25665, 25668, 25680, 25728, 25857, 25860, 25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640, 26725, 26776,
	26880, 26922, 27202, 27297, 32768, 32770, 32773, 32776, 32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850, 32853, 32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033,
	33040, 33042, 33045, 33048, 33057, 33060, 33088, 33090, 33093, 33096, 33105, 33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300, 33345, 33348, 33360, 33793, 33796, 33798,
	33801, 33808, 33810, 33813, 33816, 33825, 33856, 33858, 33861, 33864, 33873, 33876, 33888, 33921, 33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113, 34116, 34128, 34176,
	34186, 34305, 34308, 34320, 34345, 34368, 34816, 34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076, 35136, 35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880,
	36882, 36885, 36888, 36900, 36928, 36930, 36933, 36936, 36945, 36948, 36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188, 37200, 37210, 37377, 37380, 37392, 37440, 37542,
	37888, 37890, 37893, 37896, 37905, 37908, 37920, 37953, 37956, 37968, 38016, 38038, 38145, 38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916, 38928, 38950, 38976, 39081,
	39168, 39241, 39250, 39568, 40960, 40965, 40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130, 41280, 41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640, 41985,
	41988, 42000, 42048, 42121, 42148, 42240, 42265, 42577, 43018, 43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656, 43690,
];
const IQ_NEIGHBOR_SHELLS: usize = 3;
struct IqNeighbors {
	exact: Vec<i32>,
	candidates: Vec<OnceLock<Vec<u16>>>,
}
struct IqGrid {
	points: &'static [u16],
	bits: usize,
	lanes: usize,
	shells: usize,
	neighbors: OnceLock<IqNeighbors>,
}
impl IqGrid {
	const fn new(points: &'static [u16], bits: usize, lanes: usize, shells: usize) -> Self {
		Self { points, bits, lanes, shells, neighbors: OnceLock::new() }
	}
	fn code(&self, index: usize, lane: usize) -> i8 {
		let mask = (1_u16 << self.bits) - 1;
		(self.points[index] >> (self.bits * lane) & mask) as i8
	}
	fn key(&self, levels: &[i8]) -> usize {
		levels.iter().enumerate().fold(0, |key, (lane, level)| key | (*level as usize) << (self.bits * lane))
	}
	fn distance(&self, point: u16, key: usize) -> i32 {
		let mask = (1_u16 << self.bits) - 1;
		(0..self.lanes)
			.map(|lane| {
				let difference = i32::from(point >> (self.bits * lane) & mask) - ((key >> (self.bits * lane) & usize::from(mask)) as i32);
				difference * difference
			})
			.sum()
	}
	fn neighbors(&self) -> &IqNeighbors {
		self.neighbors.get_or_init(|| {
			let keys = 1_usize << (self.bits * self.lanes);
			let mut exact = vec![-1_i32; keys];
			for (index, point) in self.points.iter().enumerate() {
				if exact[usize::from(*point)] < 0 {
					exact[usize::from(*point)] = index as i32
				}
			}
			IqNeighbors { exact, candidates: (0..keys).map(|_| OnceLock::new()).collect() }
		})
	}
	fn candidates(&self, key: usize) -> &[u16] {
		self.neighbors().candidates[key].get_or_init(|| {
			let mut nearest = [i32::MAX; IQ_NEIGHBOR_SHELLS];
			for point in self.points {
				let distance = self.distance(*point, key);
				if nearest[..self.shells].contains(&distance) {
					continue;
				}
				for position in 0..self.shells {
					if distance < nearest[position] {
						for slot in (position + 1..self.shells).rev() {
							nearest[slot] = nearest[slot - 1]
						}
						nearest[position] = distance;
						break;
					}
				}
			}
			let shells = if nearest[self.shells - 1] == i32::MAX { 1 } else { self.shells };
			let mut candidates = Vec::new();
			for shell in &nearest[..shells] {
				for (index, point) in self.points.iter().enumerate() {
					if self.distance(*point, key) == *shell {
						candidates.push(index as u16)
					}
				}
			}
			candidates
		})
	}
}
static IQ3_XXS_GRID: IqGrid = IqGrid::new(&IQ3_XXS, 3, 4, 2);
static IQ3_S_GRID: IqGrid = IqGrid::new(&IQ3_S, 3, 4, 2);
static IQ2_XXS_GRID: IqGrid = IqGrid::new(&IQ2_XXS, 2, 8, 2);
static IQ2_XS_GRID: IqGrid = IqGrid::new(&IQ2_XS, 2, 8, 2);
static IQ2_S_GRID: IqGrid = IqGrid::new(&IQ2_S, 2, 8, 1);
static IQ1_GRID: IqGrid = IqGrid::new(&IQ1, 2, 8, 3);
fn iq_nearest(grid: &IqGrid, levels: &mut [i8], values: &[f32], weights: &[f32], scale: f32) -> usize {
	let key = grid.key(levels);
	let exact = grid.neighbors().exact[key];
	if exact >= 0 {
		return exact as usize;
	}
	let index = grid
		.candidates(key)
		.iter()
		.map(|index| usize::from(*index))
		.min_by(|left, right| {
			let error = |index| {
				(0..grid.lanes)
					.map(|lane| {
						let difference = scale * f32::from(2 * grid.code(index, lane) + 1) - values[lane];
						weights[lane] * difference * difference
					})
					.sum::<f32>()
			};
			error(*left).total_cmp(&error(*right))
		})
		.unwrap();
	for lane in 0..grid.lanes {
		levels[lane] = grid.code(index, lane)
	}
	index
}
fn iq1_level(index: usize, lane: usize) -> i8 {
	IQ1_GRID.code(index, lane)
}
fn iq1_nearest(levels: &mut [i8], values: &[f32], weights: &[f32], scale: f32, shift: i8) -> usize {
	let key = IQ1_GRID.key(levels);
	let exact = IQ1_GRID.neighbors().exact[key];
	if exact >= 0 {
		return exact as usize;
	}
	let index = IQ1_GRID
		.candidates(key)
		.iter()
		.map(|index| usize::from(*index))
		.min_by(|left, right| {
			let error = |index| {
				(0..8).map(|lane| {
					let level = f32::from(iq1_level(index, lane)) - 1.0 + 0.125 * f32::from(shift);
					let difference = scale * level - values[lane];
					weights[lane] * difference * difference
				})
				.sum::<f32>()
			};
			error(*left).total_cmp(&error(*right))
		})
		.unwrap();
	for lane in 0..8 {
		levels[lane] = iq1_level(index, lane)
	}
	index
}
fn iq1_shift(medium: bool, pattern: i8, group: usize) -> i8 {
	if (!medium && pattern == 0) || (medium && if group == 0 { pattern < 2 } else { pattern % 2 == 0 }) { 1 } else { -1 }
}
#[rustfmt::skip] fn iq1(values:&[f32],importance:&[f32],medium:bool)->Vec<u8>{
	let mut output=Vec::new();for(chunk,values)in values.chunks(256).enumerate(){let value=|index|values.get(index).copied().unwrap_or(0.0);let importance=|index|importance.get(chunk*256+index).copied().unwrap_or(0.0);let sigma2=2.0*(0..256).map(|index|value(index)*value(index)).sum::<f32>()/256.0;let size=if medium{16}else{32};let blocks=256/size;let mut packed=vec![0_u8;if medium{56}else{48}];let mut scales=vec![0.0_f32;blocks];let mut patterns=vec![0_i8;blocks];let mut maximum=0.0_f32;
		for block in 0..blocks{let x=(0..size).map(|offset|value(block*size+offset)).collect::<Vec<_>>();let weights=(0..size).map(|offset|importance(block*size+offset)*(sigma2+x[offset]*x[offset]).sqrt()).collect::<Vec<_>>();let max=x.iter().map(|value|value.abs()).fold(0.0_f32,f32::max);let mut levels=vec![1_i8;size];if max<if medium{1.0e-7}else{1.0e-12}{continue}let mut pairs=x.iter().copied().enumerate().map(|(index,value)|(value,index)).collect::<Vec<_>>();pairs.sort_by(|left,right|left.0.total_cmp(&right.0));let(mut sumx,mut sumw)=(vec![0.0_f32;size+1],vec![0.0_f32;size+1]);for j in 0..size{let index=pairs[j].1;sumx[j+1]=sumx[j]+weights[index]*x[index];sumw[j+1]=sumw[j]+weights[index]}let(mut best,mut scale,mut split,mut pattern)=(f32::NEG_INFINITY,max,(0,0),-1_i8);
			for first in 0..=size{for second in first..=size{for candidate in if medium{&[0_i8,1,2,3][..]}else{&[0_i8,3][..]}{let(mut qx,mut q2)=(0.0_f32,0.0_f32);if medium{for(index,pair)in pairs.iter().enumerate(){let lane=pair.1;let level=if index<first{0.0}else if index<second{1.0}else{2.0};let q=level-1.0+0.125*f32::from(iq1_shift(true,*candidate,lane/8));qx+=weights[lane]*q*x[lane];q2+=weights[lane]*q*q}}else{let shift=iq1_shift(false,*candidate,0);let q=[-1.0+0.125*f32::from(shift),0.125*f32::from(shift),1.0+0.125*f32::from(shift)];qx=(sumx[first]-sumx[0])*q[0]+(sumx[second]-sumx[first])*q[1]+(sumx[size]-sumx[second])*q[2];q2=(sumw[first]-sumw[0])*q[0]*q[0]+(sumw[second]-sumw[first])*q[1]*q[1]+(sumw[size]-sumw[second])*q[2]*q[2]}if q2>0.0&&qx*qx>best*q2{scale=qx/q2;best=scale*qx;split=(first,second);pattern=*candidate}}}}if pattern<0{continue}for(index,pair)in pairs.iter().enumerate(){levels[pair.1]=if index<split.0{0}else if index<split.1{1}else{2}}if scale<0.0{for level in &mut levels{*level=2-*level}scale=-scale;pattern=3-pattern}
			let mut indices=vec![0_usize;size/8];let mut changed=false;for group in 0..size/8{let key=(0..8).fold(0_u16,|key,lane|key|(levels[group*8+lane]as u16)<<(2*lane));changed|=!IQ1.contains(&key);indices[group]=iq1_nearest(&mut levels[group*8..group*8+8],&x[group*8..group*8+8],&weights[group*8..group*8+8],scale,iq1_shift(medium,pattern,group))}if changed{let(mut qx,mut q2)=(0.0,0.0);for lane in 0..size{let level=f32::from(levels[lane])-1.0+0.125*f32::from(iq1_shift(medium,pattern,lane/8));qx+=weights[lane]*level*x[lane];q2+=weights[lane]*level*level}if qx>0.0&&q2>0.0{scale=qx/q2}}if medium{for group in 0..2{packed[block*2+group]=indices[group]as u8}packed[32+block]=((indices[0]>>8)as u8)|((indices[1]>>8)as u8)<<4|[0,128,8,136][pattern as usize]}else{let mut high=0_u16;for group in 0..4{packed[block*4+group]=indices[group]as u8;high|=((indices[group]>>8)as u16)<<(3*group)}packed[32+2*block..34+2*block].copy_from_slice(&high.to_le_bytes())}scales[block]=scale;patterns[block]=pattern;maximum=maximum.max(scale)}
		if maximum==0.0{if !medium{put_half(&mut output,0.0)}output.extend(packed);continue}let mut scale=maximum/15.0;if medium{let(mut qx,mut q2)=(0.0,0.0);for block in 0..16{let code=qround(0.5*(scales[block]/scale-1.0)).max(0.0).min(7.0)as u16;let word=block/4;let mut stored=u16::from_le_bytes(packed[48+2*word..50+2*word].try_into().unwrap());stored|=code<<(3*(block%4));packed[48+2*word..50+2*word].copy_from_slice(&stored.to_le_bytes());for lane in 0..16{let group=lane/8;let grid=usize::from(packed[2*block+group])|usize::from(packed[32+block]>>(4*group)&7)<<8;let level=(f32::from(iq1_level(grid,lane%8))-1.0+0.125*f32::from(iq1_shift(true,patterns[block],group)))*f32::from(2*code+1);let x=value(block*16+lane);let weight=importance(block*16+lane)*(sigma2+x*x).sqrt();qx+=weight*level*x;q2+=weight*level*level}}if q2>0.0{scale=qx/q2}
			let bits=fp16(scale*1.1125);for word in 0..4{let mut stored=u16::from_le_bytes(packed[48+2*word..50+2*word].try_into().unwrap());stored|=(bits>>(4*word)&15)<<12;packed[48+2*word..50+2*word].copy_from_slice(&stored.to_le_bytes())}output.extend(packed)}else{for block in 0..8{let code=qround(0.5*(scales[block]/scale-1.0)).max(0.0).min(7.0)as u16|u16::from(patterns[block]!=0)<<3;let mut high=u16::from_le_bytes(packed[32+2*block..34+2*block].try_into().unwrap());high|=code<<12;packed[32+2*block..34+2*block].copy_from_slice(&high.to_le_bytes())}put_half(&mut output,scale*1.125);output.extend(packed)}}output
}
fn qp_scale(values: &[f32], weights: &[f32], nmax: i8) -> f32 {
	let max = values.iter().copied().fold(0.0_f32, f32::max);
	if max < 1.0e-15 {
		return 0.0;
	}
	let mut inverse = f32::from(nmax) / max;
	let mut levels = values.iter().map(|value| qround(inverse * value).min(f32::from(nmax)) as i8).collect::<Vec<_>>();
	let error = |inverse: f32| {
		values.iter()
			.zip(weights)
			.map(|(value, weight)| {
				let level = qround(inverse * value).min(f32::from(nmax));
				let difference = value - level / inverse;
				weight * difference * difference
			})
			.sum::<f32>()
	};
	let mut best = error(inverse);
	for step in -4..=4 {
		if step == 0 {
			continue;
		}
		let trial = (f32::from(nmax) + 0.1 * step as f32) / max;
		let trial_error = error(trial);
		if trial_error < best {
			best = trial_error;
			inverse = trial
		}
	}
	let (mut qx, mut q2) = (0.0, 0.0);
	for lane in 0..values.len() {
		levels[lane] = qround(inverse * values[lane]).min(f32::from(nmax)) as i8;
		qx += weights[lane] * values[lane] * f32::from(levels[lane]);
		q2 += weights[lane] * f32::from(levels[lane]) * f32::from(levels[lane])
	}
	for _ in 0..5 {
		let mut changed = false;
		for lane in 0..values.len() {
			let level = f32::from(levels[lane]);
			let x = qx - weights[lane] * values[lane] * level;
			let q = q2 - weights[lane] * level * level;
			if x > 0.0 && q > 0.0 {
				let next = qround(values[lane] * q / x).min(f32::from(nmax)) as i8;
				if next != levels[lane] {
					let nx = x + weights[lane] * values[lane] * f32::from(next);
					let nq = q + weights[lane] * f32::from(next) * f32::from(next);
					if nx * nx * q2 > qx * qx * nq {
						levels[lane] = next;
						qx = nx;
						q2 = nq;
						changed = true
					}
				}
			}
		}
		if !changed {
			break;
		}
	}
	if q2 > 0.0 { qx / q2 } else { 0.0 }
}
#[rustfmt::skip] fn iq2_xxs(values:&[f32],importance:&[f32])->Vec<u8>{
	let mut output=Vec::new();for (chunk,values)in values.chunks(256).enumerate(){let value=|index|values.get(index).copied().unwrap_or(0.0);let importance=|index|importance.get(chunk*256+index).copied().unwrap_or(0.0);let sigma2=(0..256).map(|index|value(index)*value(index)).sum::<f32>()/256.0;let mut packed=[0_u8;64];let mut scales=[0.0_f32;8];let mut maximum=0.0_f32;
		for block in 0..8{let x=(0..32).map(|offset|value(block*32+offset)).collect::<Vec<_>>();let weights=(0..32).map(|offset|importance(block*32+offset)*(sigma2+x[offset]*x[offset]).sqrt()).collect::<Vec<_>>();let mut magnitudes=x.iter().map(|value|value.abs()).collect::<Vec<_>>();let mut signs=[0_u8;4];for group in 0..4{let mut flips=0;for lane in 0..8{if x[group*8+lane]<0.0{flips+=1;signs[group]|=1<<lane}}if flips%2!=0{let lane=(0..8).min_by(|a,b|(weights[group*8+*a]*x[group*8+*a]*x[group*8+*a]).total_cmp(&(weights[group*8+*b]*x[group*8+*b]*x[group*8+*b]))).unwrap();magnitudes[group*8+lane]=-magnitudes[group*8+lane];signs[group]^=1<<lane}signs[group]&=127}let max=magnitudes.iter().copied().fold(0.0_f32,f32::max);if max<1.0e-15{continue}
			let seed=qp_scale(&magnitudes,&weights,4);let effective=seed*3.0;if effective<=0.0{continue}let mut best=0.0_f32;let mut scale=seed;let mut levels=[0_i8;32];for step in -6..=6{let inverse=(5.0+0.1*step as f32)/effective;let trial_scale=inverse.recip();let mut trial=[0_i8;32];for group in 0..4{for lane in 0..8{trial[group*8+lane]=qround(0.5*(inverse*magnitudes[group*8+lane]-1.0)).max(0.0).min(2.0)as i8}iq_nearest(&IQ2_XXS_GRID,&mut trial[group*8..group*8+8],&magnitudes[group*8..group*8+8],&weights[group*8..group*8+8].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),trial_scale);}let(mut qx,mut q2)=(0.0,0.0);for lane in 0..32{let level=f32::from(2*trial[lane]+1);qx+=weights[lane]*magnitudes[lane]*level;q2+=weights[lane]*level*level}if q2>0.0&&qx*qx>best*q2{scale=qx/q2;best=scale*qx;levels=trial}}
			if scale>0.0{let inverse=scale.recip();for group in 0..4{for lane in 0..8{levels[group*8+lane]=qround(0.5*(inverse*magnitudes[group*8+lane]-1.0)).max(0.0).min(2.0)as i8}iq_nearest(&IQ2_XXS_GRID,&mut levels[group*8..group*8+8],&magnitudes[group*8..group*8+8],&weights[group*8..group*8+8].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),scale);}let(mut qx,mut q2)=(0.0,0.0);for lane in 0..32{let level=f32::from(2*levels[lane]+1);qx+=weights[lane]*magnitudes[lane]*level;q2+=weights[lane]*level*level}if q2>0.0{scale=qx/q2}}if scale<0.0{scale=-scale;for sign in &mut signs{*sign=(!*sign)&127}}
			for group in 0..4{packed[block*8+group]=iq_nearest(&IQ2_XXS_GRID,&mut levels[group*8..group*8+8],&magnitudes[group*8..group*8+8],&weights[group*8..group*8+8].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),scale)as u8}let word=u32::from(signs[0])|u32::from(signs[1])<<7|u32::from(signs[2])<<14|u32::from(signs[3])<<21;packed[block*8+4..block*8+8].copy_from_slice(&word.to_le_bytes());scales[block]=scale;maximum=maximum.max(scale)}
		if maximum==0.0{put_half(&mut output,0.0);output.extend(packed);continue}let scale=maximum/31.0;for block in 0..8{let code=qround(0.5*(scales[block]/scale-1.0)).max(0.0).min(15.0)as u32;let mut word=u32::from_le_bytes(packed[block*8+4..block*8+8].try_into().unwrap());word|=code<<28;packed[block*8+4..block*8+8].copy_from_slice(&word.to_le_bytes())}put_half(&mut output,scale);output.extend(packed)}output
}
#[rustfmt::skip] fn iq2_16(values:&[f32],importance:Option<&[f32]>,xs:bool)->Vec<u8>{
	let grid=if xs{&IQ2_XS_GRID}else{&IQ2_S_GRID};let mut output=Vec::new();for(chunk,values)in values.chunks(256).enumerate(){let value=|index|values.get(index).copied().unwrap_or(0.0);let importance=|index|importance.and_then(|values|values.get(chunk*256+index)).copied().unwrap_or(0.0);let sigma2=(if xs{1.0}else{2.0})*(0..256).map(|index|value(index)*value(index)).sum::<f32>()/256.0;let mut packed=vec![0_u8;if xs{72}else{80}];let mut scales=[0.0_f32;16];let mut maximum=0.0_f32;
		for block in 0..16{let x=(0..16).map(|offset|value(block*16+offset)).collect::<Vec<_>>();let weights=x.iter().enumerate().map(|(offset,value)|if xs{importance(block*16+offset)*(sigma2+value*value).sqrt()}else{0.25*sigma2+value*value}).collect::<Vec<_>>();let mut magnitudes=x.iter().map(|value|value.abs()).collect::<Vec<_>>();let mut signs=[0_u8;2];for group in 0..2{let mut flips=0;for lane in 0..8{if x[group*8+lane]<0.0{flips+=1;signs[group]|=1<<lane}}if xs&&flips%2!=0{let lane=(0..8).min_by(|a,b|(weights[group*8+*a]*x[group*8+*a]*x[group*8+*a]).total_cmp(&(weights[group*8+*b]*x[group*8+*b]*x[group*8+*b]))).unwrap();magnitudes[group*8+lane]=-magnitudes[group*8+lane];signs[group]^=1<<lane}if xs{signs[group]&=127}}let max=magnitudes.iter().copied().fold(0.0_f32,f32::max);if max<if xs{1.0e-15}else{1.0e-8}{continue}let mut best=0.0_f32;let mut scale=max/5.0;let mut levels=[0_i8;16];let mut on_grid=[true;2];
			for step in -9..=9{let inverse=(5.0+0.1*step as f32)/max;let trial_scale=inverse.recip();let mut trial=[0_i8;16];let mut trial_on=[true;2];for group in 0..2{for lane in 0..8{trial[group*8+lane]=qround(0.5*(inverse*magnitudes[group*8+lane]-1.0)).max(0.0).min(2.0)as i8}let key=(0..8).fold(0_u16,|key,lane|key|(trial[group*8+lane]as u16)<<(2*lane));trial_on[group]=grid.points.contains(&key);iq_nearest(grid,&mut trial[group*8..group*8+8],&magnitudes[group*8..group*8+8],&weights[group*8..group*8+8].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),trial_scale);}let(mut qx,mut q2)=(0.0,0.0);for lane in 0..16{let level=f32::from(2*trial[lane]+1);qx+=weights[lane]*magnitudes[lane]*level;q2+=weights[lane]*level*level}if q2>0.0&&qx*qx>best*q2{scale=qx/q2;best=scale*qx;levels=trial;on_grid=trial_on}}
			if on_grid.iter().any(|value|!*value)&&scale>0.0{let inverse=scale.recip();for group in 0..2{if on_grid[group]{continue}for lane in 0..8{levels[group*8+lane]=qround(0.5*(inverse*magnitudes[group*8+lane]-1.0)).max(0.0).min(2.0)as i8}iq_nearest(grid,&mut levels[group*8..group*8+8],&magnitudes[group*8..group*8+8],&weights[group*8..group*8+8].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),scale);}let(mut qx,mut q2)=(0.0,0.0);for lane in 0..16{let level=f32::from(2*levels[lane]+1);qx+=weights[lane]*magnitudes[lane]*level;q2+=weights[lane]*level*level}if q2>0.0{scale=qx/q2}}if scale<0.0{scale=-scale;for sign in &mut signs{*sign=if xs{(!*sign)&127}else{!*sign}}}
			for group in 0..2{let index=iq_nearest(grid,&mut levels[group*8..group*8+8],&magnitudes[group*8..group*8+8],&weights[group*8..group*8+8].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),scale);let slot=2*block+group;if xs{let word=index as u16|u16::from(signs[group])<<9;packed[2*slot..2*slot+2].copy_from_slice(&word.to_le_bytes())}else{packed[slot]=index as u8;packed[64+slot/4]|=((index>>8)as u8)<<(2*(slot%4));packed[32+slot]=signs[group]}}scales[block]=scale;maximum=maximum.max(scale)}
		if maximum==0.0{put_half(&mut output,0.0);output.extend(packed);continue}let scale=maximum/31.0;let offset=if xs{64}else{72};for block in 0..16{let code=qround(0.5*(scales[block]/scale-1.0)).max(0.0).min(15.0)as u8;packed[offset+block/2]|=code<<(block%2*4)}put_half(&mut output,scale*if xs{1.0}else{0.9875});output.extend(packed)}output
}
#[rustfmt::skip] fn iq3_xxs(values: &[f32]) -> Vec<u8> {
	let mut output = Vec::new(); for values in values.chunks(256) {
		let value = |index| values.get(index).copied().unwrap_or(0.0); let mut packed = [0_u8; 96]; let mut scales = [0.0_f32; 8]; let mut maximum = 0.0_f32;
		for block in 0..8 { let x = (0..32).map(|offset| value(block * 32 + offset)).collect::<Vec<_>>(); let weights = x.iter().map(|value| value * value).collect::<Vec<_>>(); let mut magnitudes = x.iter().map(|value| value.abs()).collect::<Vec<_>>(); let mut signs = [0_u8; 4];
			for group in 0..4 { let mut flips = 0; for lane in 0..8 { if x[group * 8 + lane] < 0.0 { flips += 1; signs[group] |= 1 << lane } } if flips % 2 != 0 { let lane = (0..8).min_by(|a,b| (weights[group*8+*a]*x[group*8+*a]*x[group*8+*a]).total_cmp(&(weights[group*8+*b]*x[group*8+*b]*x[group*8+*b]))).unwrap(); magnitudes[group*8+lane] = -magnitudes[group*8+lane]; signs[group] ^= 1 << lane } signs[group] &= 127 }
			let max = magnitudes.iter().copied().fold(0.0_f32, f32::max); if max < 1.0e-6 { continue }
			let mut best = 0.0_f32; let mut scale = max / 15.0; let mut levels = [0_i8; 32];
			for step in -15..=15 { let inverse = (15.0 + 0.2 * step as f32) / max; let trial_scale = inverse.recip(); let mut trial = [0_i8; 32]; for group in 0..8 { for lane in 0..4 { trial[group*4+lane] = qround(0.5*(inverse*magnitudes[group*4+lane]-1.0)).max(0.0).min(7.0) as i8 } iq_nearest(&IQ3_XXS_GRID, &mut trial[group*4..group*4+4], &magnitudes[group*4..group*4+4], &weights[group*4..group*4+4].iter().map(|value| value.sqrt()).collect::<Vec<_>>(), trial_scale); } let (mut qx, mut q2) = (0.0,0.0); for lane in 0..32 { let level = f32::from(2*trial[lane]+1); qx += weights[lane]*magnitudes[lane]*level; q2 += weights[lane]*level*level } if q2 > 0.0 && qx*qx > best*q2 { scale=qx/q2; best=scale*qx; levels=trial } }
			for group in 0..8 { packed[block*8+group] = iq_nearest(&IQ3_XXS_GRID, &mut levels[group*4..group*4+4], &magnitudes[group*4..group*4+4], &weights[group*4..group*4+4].iter().map(|value| value.sqrt()).collect::<Vec<_>>(), scale) as u8 }
			let word = u32::from(signs[0]) | u32::from(signs[1])<<7 | u32::from(signs[2])<<14 | u32::from(signs[3])<<21; packed[64+block*4..68+block*4].copy_from_slice(&word.to_le_bytes()); scales[block]=scale; maximum=maximum.max(scale)
		} if maximum == 0.0 { put_half(&mut output, 0.0); output.extend(packed); continue }
		let scale = maximum / 31.0; for block in 0..8 { let code=qround(0.5*(scales[block]/scale-1.0)).max(0.0).min(15.0) as u32; let mut word=u32::from_le_bytes(packed[64+block*4..68+block*4].try_into().unwrap()); word|=code<<28; packed[64+block*4..68+block*4].copy_from_slice(&word.to_le_bytes()) }
		put_half(&mut output, scale * 1.0125); output.extend(packed)
	} output
}
#[rustfmt::skip] fn iq3_s(values: &[f32]) -> Vec<u8> {
	let mut output=Vec::new(); for values in values.chunks(256) {
		let value=|index| values.get(index).copied().unwrap_or(0.0); let mut packed=[0_u8;108]; let mut scales=[0.0_f32;8]; let mut maximum=0.0_f32;
		for block in 0..8 { let x=(0..32).map(|offset| value(block*32+offset)).collect::<Vec<_>>(); let weights=x.iter().map(|value| value*value).collect::<Vec<_>>(); let magnitudes=x.iter().map(|value| value.abs()).collect::<Vec<_>>(); let max=magnitudes.iter().copied().fold(0.0_f32,f32::max); if max==0.0 {continue} let mut best=0.0_f32; let mut scale=max/15.0; let mut levels=[0_i8;32];
			for step in -9..=9 { let inverse=(15.0+0.2*step as f32)/max; let trial_scale=inverse.recip(); let mut trial=[0_i8;32]; for group in 0..8 { for lane in 0..4 {trial[group*4+lane]=qround(0.5*(inverse*magnitudes[group*4+lane]-1.0)).max(0.0).min(7.0) as i8} iq_nearest(&IQ3_S_GRID,&mut trial[group*4..group*4+4],&magnitudes[group*4..group*4+4],&weights[group*4..group*4+4].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),trial_scale); } let(mut qx,mut q2)=(0.0,0.0); for lane in 0..32 {let level=f32::from(2*trial[lane]+1);qx+=weights[lane]*magnitudes[lane]*level;q2+=weights[lane]*level*level} if q2>0.0&&qx*qx>best*q2 {scale=qx/q2;best=scale*qx;levels=trial} }
			for group in 0..8 {let index=iq_nearest(&IQ3_S_GRID,&mut levels[group*4..group*4+4],&magnitudes[group*4..group*4+4],&weights[group*4..group*4+4].iter().map(|value|value.sqrt()).collect::<Vec<_>>(),scale);packed[block*8+group]=index as u8;packed[64+(block*8+group)/8]|=((index>>8)as u8)<<((block*8+group)%8)} for group in 0..4 {packed[72+block*4+group]=(0..8).fold(0,|signs,lane|signs|u8::from(x[group*8+lane]<0.0)<<lane)} scales[block]=scale;maximum=maximum.max(scale)
		}
		if maximum==0.0 {put_half(&mut output,0.0);output.extend(packed);continue} let scale=maximum/31.0; for pair in 0..4 {let low=qround(0.5*(scales[pair*2]/scale-1.0)).max(0.0).min(15.0)as u8;let high=qround(0.5*(scales[pair*2+1]/scale-1.0)).max(0.0).min(15.0)as u8;packed[104+pair]=low|high<<4} put_half(&mut output,scale*1.033);output.extend(packed)
	} output
}
fn iq4_code(value: f32) -> u8 {
	IQ4.iter().enumerate().min_by(|left, right| (value - f32::from(*left.1)).abs().total_cmp(&(value - f32::from(*right.1)).abs())).unwrap().0 as u8
}
#[rustfmt::skip]
fn iq4_fit(values: &[f32], tries: i32) -> (f32, Vec<u8>) {
	let mut extreme = 0.0_f32;
	for value in values { if value.abs() > extreme.abs() { extreme = *value } }
	if extreme.abs() < 1.0e-15 { return (0.0, vec![0; values.len()]) }
	let initial = if tries > 0 { -extreme / f32::from(IQ4[0]) } else { extreme / f32::from(IQ4[0]) };
	let score = |inverse: f32| {
		values.iter().map(|value| { let level = f32::from(IQ4[usize::from(iq4_code(value * inverse))]);
			(value * value * level * value, value * value * level * level) }).fold((0.0, 0.0), |left, right| (left.0 + right.0, left.1 + right.1))
	};
	let (numerator, denominator) = score(initial.recip());
	let mut scale = if denominator > 0.0 { numerator / denominator } else { 0.0 };
	let mut best = scale * numerator;
	for attempt in -tries..=tries {
		let (numerator, denominator) = score((attempt as f32 + f32::from(IQ4[0])) / extreme);
		if denominator > 0.0 && numerator * numerator > best * denominator { scale = numerator / denominator; best = scale * numerator }
	}
	let inverse = if tries > 0 && scale != 0.0 { scale.recip() } else { initial.recip() };
	(scale, values.iter().map(|value| iq4_code(value * inverse)).collect())
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorageFormat(pub(crate) u16);

#[derive(Clone, Copy)]
enum NativeDequant {
	Nf4,
	Scalar(ScalarLayout),
	Q2K,
	Q3K,
	Q45K(u8),
	Q6K,
	Q8K,
	Iq4(Iq4Layout),
	Iq1(Iq1Layout),
	Iq(IqLayout),
}

impl NativeDequant {
	fn decode<Q: QuantOps>(self, operations: &mut Q) -> Q::Value {
		match self {
			Self::Nf4 => unreachable!("NF4 dequantization requires its model codebook"),
			Self::Scalar(layout) => quantized::dequant_scalar(operations, layout),
			Self::Q2K => quantized::dequant_q2k(operations),
			Self::Q3K => quantized::dequant_q3k(operations),
			Self::Q45K(man) => quantized::dequant_q45k(operations, man),
			Self::Q6K => quantized::dequant_q6k(operations),
			Self::Q8K => quantized::dequant_q8k(operations),
			Self::Iq4(layout) => quantized::dequant_iq4(operations, layout),
			Self::Iq1(layout) => quantized::dequant_iq1(operations, layout),
			Self::Iq(layout) => quantized::dequant_iq(operations, layout),
		}
	}

	fn table(self) -> Option<NativeQuantTable> {
		match self {
			Self::Iq4(layout) => Some(NativeQuantTable::Signed(layout.table_name, layout.table)),
			Self::Iq1(layout) => Some(NativeQuantTable::Unsigned(layout.table_name, layout.table)),
			Self::Iq(layout) => Some(NativeQuantTable::Unsigned(layout.table_name, layout.table)),
			_ => None,
		}
	}
}

#[derive(Clone, Copy)]
enum NativeQuantTable {
	Unsigned(&'static str, &'static [u16]),
	Signed(&'static str, &'static [i8]),
}

impl NativeQuantTable {
	fn name(self) -> &'static str {
		match self {
			Self::Unsigned(name, _) | Self::Signed(name, _) => name,
		}
	}

	fn definition(self) -> String {
		match self {
			Self::Unsigned(name, values) => format!(
				"@recipe_model_{name} = private unnamed_addr constant [{} x i16] [{}]\n",
				values.len(),
				values.iter().map(|value| format!("i16 {value}")).collect::<Vec<_>>().join(", ")
			),
			Self::Signed(name, values) => {
				format!("@recipe_model_{name} = private unnamed_addr constant [{} x i8] [{}]\n", values.len(), values.iter().map(|value| format!("i8 {value}")).collect::<Vec<_>>().join(", "))
			}
		}
	}
}

#[derive(Clone, Copy)]
enum Quantizer {
	Scalar { bits: u8, variant: u8 },
	Q2K,
	Q3K,
	Q45K { bits: u8 },
	Q6K,
	Q8K,
	Nf4,
	Iq4Nl,
	Iq4Xs,
	Iq2Xxs,
	Iq2 { importance: bool, xs: bool },
	Iq1 { medium: bool },
	Iq3Xxs,
	Iq3S,
}

#[derive(Clone, Copy)]
struct Quantization {
	codec: StorageCodec,
	family: u16,
	bits: u8,
	variants: &'static [u16],
	block: usize,
	stride: usize,
	name: &'static str,
	quantizer: Quantizer,
	native: NativeDequant,
}

macro_rules! quantizations {
	($( $codec:ident { code: ($family:literal, $bits:literal, [$($variant:literal),+]), block: $block:literal, stride: $stride:literal, name: $name:literal, quant: $quantizer:expr, native: Some($native:expr) } )+) => {
		#[derive(Clone, Copy, Debug, PartialEq, Eq)]
		pub(crate) enum StorageCodec { $($codec),+ }
		const QUANTIZATIONS: &[Quantization] = &[$(Quantization { codec: StorageCodec::$codec, family: $family, bits: $bits, variants: &[$($variant),+], block: $block, stride: $stride, name: $name, quantizer: $quantizer, native: $native }),+];
	};
}

quantizations! {
	Q4_0 { code: (0, 4, [0]), block: 32, stride: 18, name: "q4_0", quant: Quantizer::Scalar { bits: 4, variant: 0 }, native: Some(NativeDequant::Scalar(ScalarLayout { sign: 1, exp: 5, man: 4, variant: 0 })) }
	Q4_1 { code: (0, 4, [1]), block: 32, stride: 20, name: "q4_1", quant: Quantizer::Scalar { bits: 4, variant: 1 }, native: Some(NativeDequant::Scalar(ScalarLayout { sign: 1, exp: 5, man: 4, variant: 1 })) }
	Q5_0 { code: (0, 5, [0]), block: 32, stride: 22, name: "q5_0", quant: Quantizer::Scalar { bits: 5, variant: 0 }, native: Some(NativeDequant::Scalar(ScalarLayout { sign: 1, exp: 5, man: 5, variant: 0 })) }
	Q5_1 { code: (0, 5, [1]), block: 32, stride: 24, name: "q5_1", quant: Quantizer::Scalar { bits: 5, variant: 1 }, native: Some(NativeDequant::Scalar(ScalarLayout { sign: 1, exp: 5, man: 5, variant: 1 })) }
	Q8_0 { code: (0, 8, [0]), block: 32, stride: 34, name: "q8_0", quant: Quantizer::Scalar { bits: 8, variant: 0 }, native: Some(NativeDequant::Scalar(ScalarLayout { sign: 1, exp: 5, man: 8, variant: 0 })) }
	Q8_1 { code: (0, 8, [1]), block: 32, stride: 36, name: "q8_1", quant: Quantizer::Scalar { bits: 8, variant: 1 }, native: Some(NativeDequant::Scalar(ScalarLayout { sign: 1, exp: 5, man: 8, variant: 1 })) }
	NF4 { code: (0, 4, [2]), block: 0, stride: 0, name: "q4_nf", quant: Quantizer::Nf4, native: Some(NativeDequant::Nf4) }
	Q2K { code: (0, 2, [3]), block: 256, stride: 84, name: "q2k", quant: Quantizer::Q2K, native: Some(NativeDequant::Q2K) }
	Q3K { code: (0, 3, [3, 4, 5, 6]), block: 256, stride: 110, name: "q3k", quant: Quantizer::Q3K, native: Some(NativeDequant::Q3K) }
	Q4K { code: (0, 4, [3, 4, 5, 6]), block: 256, stride: 144, name: "q4k", quant: Quantizer::Q45K { bits: 4 }, native: Some(NativeDequant::Q45K(4)) }
	Q5K { code: (0, 5, [3, 4, 5, 6]), block: 256, stride: 176, name: "q5k", quant: Quantizer::Q45K { bits: 5 }, native: Some(NativeDequant::Q45K(5)) }
	Q6K { code: (0, 6, [3, 4, 5, 6]), block: 256, stride: 210, name: "q6k", quant: Quantizer::Q6K, native: Some(NativeDequant::Q6K) }
	Q8K { code: (0, 8, [3]), block: 256, stride: 292, name: "q8k", quant: Quantizer::Q8K, native: Some(NativeDequant::Q8K) }
	IQ4NL { code: (1, 4, [5]), block: 32, stride: 18, name: "iq4nl", quant: Quantizer::Iq4Nl, native: Some(NativeDequant::Iq4(Iq4Layout { sign: 1, exp: 1, man: 4, xs: false, table_name: "iq4", table: &IQ4 })) }
	IQ4XS { code: (1, 4, [2]), block: 256, stride: 136, name: "iq4xs", quant: Quantizer::Iq4Xs, native: Some(NativeDequant::Iq4(Iq4Layout { sign: 1, exp: 6, man: 4, xs: true, table_name: "iq4", table: &IQ4 })) }
	IQ3XXS { code: (1, 3, [1]), block: 256, stride: 98, name: "iq3xxs", quant: Quantizer::Iq3Xxs, native: Some(NativeDequant::Iq(IqLayout { man: 3, exp: 4, sign: 1, packing: IqPacking::Xxs, table_name: "iq3xxs", table: &IQ3_XXS })) }
	IQ2XXS { code: (1, 2, [1]), block: 256, stride: 66, name: "iq2xxs", quant: Quantizer::Iq2Xxs, native: Some(NativeDequant::Iq(IqLayout { man: 2, exp: 4, sign: 1, packing: IqPacking::Xxs, table_name: "iq2xxs", table: &IQ2_XXS })) }
	IQ2XS { code: (1, 2, [2]), block: 256, stride: 74, name: "iq2xs", quant: Quantizer::Iq2 { importance: true, xs: true }, native: Some(NativeDequant::Iq(IqLayout { man: 2, exp: 4, sign: 1, packing: IqPacking::Xs, table_name: "iq2xs", table: &IQ2_XS })) }
	IQ2S { code: (1, 2, [3]), block: 256, stride: 82, name: "iq2s", quant: Quantizer::Iq2 { importance: false, xs: false }, native: Some(NativeDequant::Iq(IqLayout { man: 2, exp: 4, sign: 1, packing: IqPacking::S, table_name: "iq2s", table: &IQ2_S })) }
	IQ1S { code: (1, 1, [3]), block: 256, stride: 50, name: "iq1s", quant: Quantizer::Iq1 { medium: false }, native: Some(NativeDequant::Iq1(Iq1Layout { man: 2, exp: 3, sign: 1, medium: false, table_name: "iq1", table: &IQ1 })) }
	IQ1M { code: (1, 1, [4]), block: 256, stride: 56, name: "iq1m", quant: Quantizer::Iq1 { medium: true }, native: Some(NativeDequant::Iq1(Iq1Layout { man: 2, exp: 3, sign: 1, medium: true, table_name: "iq1", table: &IQ1 })) }
	IQ3S { code: (1, 3, [3]), block: 256, stride: 110, name: "iq3s", quant: Quantizer::Iq3S, native: Some(NativeDequant::Iq(IqLayout { man: 3, exp: 4, sign: 1, packing: IqPacking::S, table_name: "iq3s", table: &IQ3_S })) }
}

fn nf4_codebook(codebook: &[f64], count: usize, bytes: usize) -> Result<(usize, &[f64], &[f64])> {
	let block_value = codebook.first().copied().unwrap_or(0.0);
	require(block_value.is_finite() && block_value.fract() == 0.0 && block_value >= 1.0 && block_value <= usize::MAX as f64, "NF4 block size is invalid")?;
	let block = block_value as usize;
	let scales = count.div_ceil(block);
	require(codebook.len() == 17 + scales && bytes == count.div_ceil(2), "NF4 weights are invalid")?;
	Ok((block, &codebook[1..17], &codebook[17..]))
}

impl StorageCodec {
	fn quantization(self) -> &'static Quantization {
		QUANTIZATIONS.iter().find(|format| format.codec == self).unwrap()
	}
	fn dequantize(self, data: &[u8], codebook: &[f64], count: usize) -> Result<Vec<f64>> {
		let format = self.quantization();
		let native = format.native;
		if matches!(native, NativeDequant::Nf4) {
			let (block, table, scales) = nf4_codebook(codebook, count, data.len())?;
			return Ok((0..count).map(|index| dequant_nf4(&mut HostQuantOps { bytes: data, index }, block, "nf4", table, "nf4_scales", scales)).collect());
		}
		decode_blocks(data, count, format.block, format.stride, "GGML quantized weights are invalid", |bytes, index| native.decode(&mut HostQuantOps { bytes, index }))
	}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct StorageSpec {
	pub(crate) codec: StorageCodec,
	pub(crate) block: usize,
	pub(crate) stride: usize,
}

/// A weight's block bytes: owned by the graph, or a view of the mapped file they
/// were read from, which the view keeps mapped.
#[derive(Clone)]
enum StoredSegment {
	Owned(Vec<u8>),
	Mapped(Arc<gguf::Mapping>, usize, usize),
}
impl std::ops::Deref for StoredSegment {
	type Target = [u8];
	fn deref(&self) -> &[u8] {
		match self {
			Self::Owned(bytes) => bytes,
			Self::Mapped(mapping, at, length) => &mapping.bytes()[*at..*at + *length],
		}
	}
}
/// One node's stored bytes as the ordered runs they are assembled from, each
/// either owned or a view of a mapped file. A weight drawn from several tensors
/// names one run per tensor, in the order the lowering lays its planes out, and
/// nothing is copied until an arena is written.
#[derive(Clone)]
pub(crate) struct StoredBytes(Arc<Vec<StoredSegment>>);
impl StoredBytes {
	fn mapped(mapping: &Arc<gguf::Mapping>, at: usize, length: usize) -> Self {
		Self(Arc::new(vec![StoredSegment::Mapped(mapping.clone(), at, length)]))
	}
	/// The runs of several weights, end to end, as one weight. A mapped run is
	/// shared with the weight it came from, so joining copies no bytes.
	fn joined(parts: Vec<Self>) -> Self {
		Self(Arc::new(parts.iter().flat_map(|part| part.0.iter().cloned()).collect()))
	}
	fn len(&self) -> usize {
		self.0.iter().map(|run| run.len()).sum()
	}
	fn runs(&self) -> impl Iterator<Item = &[u8]> {
		self.0.iter().map(|run| &**run)
	}
	/// The `length` bytes at `at`, gathered across the runs they fall in, so one
	/// row of a mapped table is read without touching the rest of it.
	fn slice(&self, at: usize, length: usize) -> Result<Vec<u8>> {
		require(at.checked_add(length).is_some_and(|end| end <= self.len()), format!("stored bytes {at}..{} exceed {} bytes", at.saturating_add(length), self.len()))?;
		let (mut out, mut skipped) = (Vec::with_capacity(length), 0);
		for run in self.runs() {
			let start = (at.max(skipped) - skipped).min(run.len());
			let end = ((at + length).saturating_sub(skipped)).min(run.len());
			if start < end {
				out.extend_from_slice(&run[start..end]);
			}
			skipped += run.len();
		}
		Ok(out)
	}
	/// Appends every run to `out`, which is the only point a mapped weight is copied.
	fn extend_into(&self, out: &mut Vec<u8>) {
		for run in self.runs() {
			out.extend_from_slice(run);
		}
	}
	/// Writes every run end to end over `out`, which spans exactly `len` bytes.
	fn copy_into(&self, out: &mut [u8]) {
		let mut at = 0;
		for run in self.runs() {
			out[at..at + run.len()].copy_from_slice(run);
			at += run.len();
		}
	}
	fn to_vec(&self) -> Vec<u8> {
		let mut out = Vec::with_capacity(self.len());
		self.extend_into(&mut out);
		out
	}
}
impl From<Vec<u8>> for StoredBytes {
	fn from(bytes: Vec<u8>) -> Self {
		Self(Arc::new(vec![StoredSegment::Owned(bytes)]))
	}
}

/// One node's parameters as they are stored. `arithmetic` is the host copy the
/// graph trains and saves; a weight the device decodes from `bytes` carries none.
#[derive(Clone)]
pub(crate) struct StoredWeight {
	pub(crate) format: StorageFormat,
	pub(crate) count: usize,
	pub(crate) bytes: StoredBytes,
	pub(crate) codebook: Vec<f64>,
	pub(crate) arithmetic: Vec<f64>,
}

/// What one parameterized node of a graph compiled over mapped tensors is
/// filled with: the names of its planes for messages, their element sum, and
/// the bytes or values they resolve to.
#[derive(Clone)]
struct BoundNode {
	names: String,
	elements: usize,
	weight: BoundWeight,
}
/// Block-quantized planes stay packed as runs of their own mappings, so the
/// node owns no arithmetic span and the tape decodes the file's blocks. Planes
/// in an unblocked layout, such as F32 or F16, decode once into the node's
/// span, which is the only representation those layouts have.
#[derive(Clone)]
enum BoundWeight {
	Stored(StoredWeight),
	Values(Vec<f64>),
}

impl StorageFormat {
	fn valid(self) -> bool {
		self.spec().is_some() || self.selection().is_some()
	}
	/// The format a storage name in the quantization table selects, at its first variant.
	pub(crate) fn named(name: &str) -> Option<Self> {
		QUANTIZATIONS.iter().find(|format| format.name == name).map(|format| Self(format.family << 12 | format.variants[0] << 8 | u16::from(format.bits)))
	}
	pub(crate) fn spec(self) -> Option<StorageSpec> {
		let (family, bits, variant) = (self.0 >> 12, self.bits(), self.0 >> 8 & 15);
		QUANTIZATIONS.iter().find(|format| format.family == family && format.bits == bits && format.variants.contains(&variant)).map(|format| StorageSpec {
			codec: format.codec,
			block: format.block,
			stride: format.stride,
		})
	}
	pub(crate) fn encode(self, arithmetic: &[f64], importance: &[f64], config: Config) -> Result<StoredWeight> {
		let (bytes, codebook) = self.compress(arithmetic, importance, config)?;
		Ok(StoredWeight { format: self, count: arithmetic.len(), bytes: bytes.into(), codebook, arithmetic: arithmetic.to_vec() })
	}
	fn unavailable(self) -> RecipeError {
		RecipeError::new(format!(
			"{} is unavailable; available GGML formats: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K, Q4_K_S, Q4_K_M, Q5_K, Q5_K_S, Q5_K_M, Q6_K, Q8_K, Q4_NF, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, IQ4_XS, and IQ4_NL",
			quantization(self.0)
		))
	}
	fn selection(self) -> Option<u16> {
		let (family, bits, variant) = (self.0 >> 12, self.bits(), self.0 >> 8 & 15);
		match (family, bits, variant) {
			(0, 3 | 4 | 5, 5) => Some(5),
			(0, 3 | 4 | 5, 4) => Some(4),
			(0, 3, 6) => Some(6),
			(1, 2, 4) | (1, 3, 2 | 4) => Some(variant),
			_ => None,
		}
	}
	fn tensor(self, role: u8, more: bool, output: bool) -> u16 {
		let (family, bits, style) = (self.0 >> 12, self.bits(), self.selection().unwrap());
		if output {
			return 3 << 8 | 6;
		}
		if family == 1 {
			return match (bits, style, role, more) {
				(2, 4, 2 | 3, _) | (2, 4, _, true) => 1 << 12 | 3 << 8 | 3,
				(3, 2, 0 | 1, _) | (3, 2, _, false) => 1 << 12 | 1 << 8 | 3,
				(3, 4, 2 | 3, _) | (3, 4, _, true) => 3 << 8 | 4,
				_ => 1 << 12 | 3 << 8 | u16::from(bits),
			};
		}
		let bits = match (bits, style, role) {
			(2, _, 2 | 3) => 3,
			(3, 5, 2) => 5,
			(3, 5, 3) => 4,
			(3, 6, 2 | 3) => 5,
			(4, 4, 2) => 5,
			(4, 5, 2) if more => 6,
			(5, 5, 2) if more => 6,
			_ => bits,
		};
		3 << 8 | u16::from(bits)
	}
}
trait Integer {
	fn compress(self, weights: &[f64], importance: &[f64], config: Config) -> Result<(Vec<u8>, Vec<f64>)>;
	fn decompress(self, data: &[u8], codebook: &[f64], count: usize) -> Result<Vec<f64>>;
	fn bits(self) -> u8;
}
fn decode_blocks(data: &[u8], count: usize, block: usize, stride: usize, error: &str, mut decode: impl FnMut(&[u8], usize) -> f64) -> Result<Vec<f64>> {
	require(data.len() >= count.div_ceil(block) * stride, error)?;
	let mut weights = Vec::with_capacity(count);
	for bytes in data.chunks_exact(stride) {
		let remaining = block.min(count - weights.len());
		weights.extend((0..remaining).map(|index| decode(bytes, index)));
	}
	Ok(weights)
}
impl Integer for StorageFormat {
	fn bits(self) -> u8 {
		self.0 as u8
	}
	fn compress(self, weights: &[f64], importance: &[f64], config: Config) -> Result<(Vec<u8>, Vec<f64>)> {
		let quantizer = self.spec().ok_or_else(|| self.unavailable())?.codec.quantization().quantizer;
		if let Quantizer::Scalar { bits, variant } = quantizer {
			let block = 32;
			let mut data = Vec::new();
			for values in weights.chunks(block) {
				let value = |index| values.get(index).copied().unwrap_or(0.0) as f32;
				let (minimum, maximum) = (0..block).map(value).fold((f32::INFINITY, f32::NEG_INFINITY), |(low, high), value| (low.min(value), high.max(value)));
				let extreme = (0..block).map(value).max_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap_or(0.0);
				let scale = match (bits, variant) {
					(8, _) => extreme.abs() / 127.0,
					(_, 0) => extreme / -(1_i32 << (bits - 1)) as f32,
					(_, 1) => (maximum - minimum) / ((1_u16 << bits) - 1) as f32,
					_ => unreachable!(),
				};
				let inverse = if scale == 0.0 { 0.0 } else { scale.recip() };
				put_half(&mut data, scale);
				if variant == 1 && bits != 8 {
					put_half(&mut data, minimum)
				}
				let (mut low, mut high) = ([0_u8; 32], [0_u8; 4]);
				let mut sum = 0_i32;
				for index in 0..block {
					let shifted = match (bits, variant) {
						(8, _) => (value(index) * inverse).round() + 128.0,
						(_, 0) => value(index) * inverse + (1_i32 << (bits - 1)) as f32 + 0.5,
						(_, 1) => (value(index) - minimum) * inverse + 0.5,
						_ => unreachable!(),
					};
					let code = shifted.max(0.0).min(f32::from((1_u16 << bits) - 1)) as u8;
					if bits == 4 || bits == 5 {
						low[index % 16] |= (code & 15) << (index / 16 * 4)
					}
					if bits == 5 {
						high[index / 8] |= (code >> 4) << (index % 8)
					}
					if bits == 8 {
						low[index] = code.wrapping_sub(128);
						sum += i32::from(i8::from_ne_bytes([low[index]]))
					}
				}
				if bits == 5 {
					data.extend(high)
				}
				if bits == 8 && variant == 1 {
					put_half(&mut data, scale * sum as f32)
				}
				data.extend_from_slice(
					&low[..match bits {
						4 | 5 => 16,
						8 => 32,
						_ => unreachable!(),
					}],
				);
			}
			return Ok((data, Vec::new()));
		}
		if matches!(quantizer, Quantizer::Q2K) {
			let mut data = Vec::new();
			for values in weights.chunks(256) {
				let values = (0..256).map(|index| values.get(index).copied().unwrap_or(0.0) as f32).collect::<Vec<_>>();
				let (mut codes, mut scales, mut minima) = ([0_u8; 256], [0.0_f32; 16], [0.0_f32; 16]);
				for block in 0..16 {
					let weights = values[block * 16..block * 16 + 16].iter().map(|value| value.abs()).collect::<Vec<_>>();
					(scales[block], minima[block]) = qkx2(&values[block * 16..block * 16 + 16], &weights, 3, (-0.5, 0.1, 15), true, &mut codes[block * 16..block * 16 + 16]);
				}
				let (max_scale, max_minimum) = (positive_max(&scales), positive_max(&minima));
				let (scale, minimum) = (max_scale / 15.0, max_minimum / 15.0);
				let (stored_scale, stored_minimum) = (unfp16(fp16(scale)), unfp16(fp16(minimum)));
				let mut packed_scales = [0_u8; 16];
				for block in 0..16 {
					let scale_code = if max_scale > 0.0 { qround(15.0 * scales[block] / max_scale) as u8 } else { 0 };
					let minimum_code = if max_minimum > 0.0 { qround(15.0 * minima[block] / max_minimum) as u8 } else { 0 };
					packed_scales[block] = scale_code | minimum_code << 4;
					let (d, m) = (stored_scale * f32::from(scale_code), stored_minimum * f32::from(minimum_code));
					if d != 0.0 {
						for offset in 0..16 {
							codes[block * 16 + offset] = qround((values[block * 16 + offset] + m) / d).max(0.0).min(3.0) as u8;
						}
					}
				}
				let mut packed = [0_u8; 64];
				for group in (0..256).step_by(128) {
					for offset in 0..32 {
						packed[group / 4 + offset] = codes[group + offset] | codes[group + offset + 32] << 2 | codes[group + offset + 64] << 4 | codes[group + offset + 96] << 6;
					}
				}
				data.extend(packed_scales);
				data.extend(packed);
				put_half(&mut data, scale);
				put_half(&mut data, minimum);
			}
			return Ok((data, Vec::new()));
		}
		if matches!(quantizer, Quantizer::Q3K) {
			let mut data = Vec::new();
			for values in weights.chunks(256) {
				let values = (0..256).map(|index| values.get(index).copied().unwrap_or(0.0) as f32).collect::<Vec<_>>();
				let (mut codes, mut block_scales) = ([0_i8; 256], [0.0_f32; 16]);
				let (mut maximum, mut extreme) = (0.0, 0.0);
				for block in 0..16 {
					block_scales[block] = q3(&values[block * 16..block * 16 + 16], &mut codes[block * 16..block * 16 + 16]);
					if block_scales[block].abs() > extreme {
						extreme = block_scales[block].abs();
						maximum = block_scales[block]
					}
				}
				let inverse = if maximum == 0.0 { 0.0 } else { -32.0 / maximum };
				let scale = if inverse == 0.0 { 0.0 } else { inverse.recip() };
				let stored_scale = unfp16(fp16(scale));
				let mut scales = [0_u8; 12];
				for block in 0..16 {
					let mut code = qround(inverse * block_scales[block]).max(-32.0).min(31.0) as i8 + 32;
					if block < 8 {
						scales[block] = code as u8 & 15
					} else {
						scales[block - 8] |= (code as u8 & 15) << 4
					}
					code >>= 4;
					scales[block % 4 + 8] |= (code as u8) << (2 * (block / 4));
					let signed =
						((scales[if block < 8 { block } else { block - 8 }] >> if block < 8 { 0 } else { 4 } & 15) | ((scales[8 + block % 4] >> (2 * (block / 4)) & 3) << 4)) as i8 - 32;
					let d = stored_scale * f32::from(signed);
					if d != 0.0 {
						for offset in 0..16 {
							codes[block * 16 + offset] = qround(values[block * 16 + offset] / d).max(-4.0).min(3.0) as i8 + 4;
						}
					}
				}
				let (mut high, mut low) = ([0_u8; 32], [0_u8; 64]);
				for index in 0..256 {
					let mut code = codes[index] as u8;
					if code > 3 {
						high[index % 32] |= 1 << (index / 32);
						code -= 4
					}
					low[index / 128 * 32 + index % 32] |= code << (index % 128 / 32 * 2);
				}
				data.extend(high);
				data.extend(low);
				data.extend(scales);
				put_half(&mut data, scale);
			}
			return Ok((data, Vec::new()));
		}
		if let Quantizer::Q45K { bits } = quantizer {
			let chunks = weights.chunks(256).collect::<Vec<_>>();
			let data = parallel_map(chunks.len(), |chunk| {
				let values = (0..256).map(|index| chunks[chunk].get(index).copied().unwrap_or(0.0) as f32).collect::<Vec<_>>();
				let mut data = Vec::new();
				let (mut codes, mut block_scales, mut minima) = ([0_u8; 256], [0.0_f32; 8], [0.0_f32; 8]);
				for block in 0..8 {
					let slice = &values[block * 32..block * 32 + 32];
					let magnitude = (slice.iter().map(|value| value * value).sum::<f32>() / 32.0).sqrt();
					let weights = slice.iter().map(|value| magnitude + value.abs()).collect::<Vec<_>>();
					let (levels, range) = if bits == 4 { (15, (-1.0, 0.1, 20)) } else { (31, (-0.5, 0.1, 15)) };
					(block_scales[block], minima[block]) = qkx2(slice, &weights, levels, range, false, &mut codes[block * 32..block * 32 + 32]);
				}
				let (maximum, max_minimum) = (positive_max(&block_scales), positive_max(&minima));
				let (scale, minimum) = (maximum / 63.0, max_minimum / 63.0);
				let (stored_scale, stored_minimum) = (unfp16(fp16(scale)), unfp16(fp16(minimum)));
				let mut metadata = [0_u8; 12];
				for block in 0..8 {
					let scale_code = if maximum > 0.0 { qround(63.0 * block_scales[block] / maximum).min(63.0) as u8 } else { 0 };
					let minimum_code = if max_minimum > 0.0 { qround(63.0 * minima[block] / max_minimum).min(63.0) as u8 } else { 0 };
					if block < 4 {
						metadata[block] = scale_code;
						metadata[block + 4] = minimum_code
					} else {
						metadata[block + 4] = scale_code & 15 | (minimum_code & 15) << 4;
						metadata[block - 4] |= scale_code >> 4 << 6;
						metadata[block] |= minimum_code >> 4 << 6
					}
				}
				for block in 0..8 {
					let (scale_code, minimum_code) = k_scale(&metadata, block);
					let (d, m) = (stored_scale * f32::from(scale_code), stored_minimum * f32::from(minimum_code));
					if d != 0.0 {
						for offset in 0..32 {
							codes[block * 32 + offset] = qround((values[block * 32 + offset] + m) / d).max(0.0).min(if bits == 4 { 15.0 } else { 31.0 }) as u8;
						}
					}
				}
				let (mut high, mut packed) = ([0_u8; 32], [0_u8; 128]);
				for group in (0..256).step_by(64) {
					for offset in 0..32 {
						packed[group / 2 + offset] = codes[group + offset] & 15 | (codes[group + offset + 32] & 15) << 4;
						high[offset] |= (codes[group + offset] >> 4) << (group / 32) | (codes[group + offset + 32] >> 4) << (group / 32 + 1)
					}
				}
				put_half(&mut data, scale);
				put_half(&mut data, minimum);
				data.extend(metadata);
				if bits == 5 {
					data.extend(high)
				}
				data.extend(packed);
				data
			})?;
			return Ok((data.concat(), Vec::new()));
		}
		if matches!(quantizer, Quantizer::Q6K) {
			let mut data = Vec::new();
			for values in weights.chunks(256) {
				let values = (0..256).map(|index| values.get(index).copied().unwrap_or(0.0) as f32).collect::<Vec<_>>();
				let (mut codes, mut block_scales) = ([0_i8; 256], [0.0_f32; 16]);
				let (mut maximum, mut extreme) = (0.0, 0.0);
				for block in 0..16 {
					block_scales[block] = qx(&values[block * 16..block * 16 + 16], 32, &mut codes[block * 16..block * 16 + 16]);
					if block_scales[block].abs() > extreme {
						extreme = block_scales[block].abs();
						maximum = block_scales[block]
					}
				}
				let inverse = if extreme < 1.0e-15 { 0.0 } else { -128.0 / maximum };
				let scale = if inverse == 0.0 { 0.0 } else { inverse.recip() };
				let stored_scale = unfp16(fp16(scale));
				let mut scales = [0_i8; 16];
				for block in 0..16 {
					scales[block] = qround(inverse * block_scales[block]).min(127.0) as i8;
					let d = stored_scale * f32::from(scales[block]);
					if d != 0.0 {
						for offset in 0..16 {
							codes[block * 16 + offset] = qround(values[block * 16 + offset] / d).max(-32.0).min(31.0) as i8 + 32;
						}
					}
				}
				let (mut low, mut high) = ([0_u8; 128], [0_u8; 64]);
				for group in (0..256).step_by(128) {
					for offset in 0..32 {
						let code = [codes[group + offset], codes[group + offset + 32], codes[group + offset + 64], codes[group + offset + 96]].map(|value| value as u8);
						low[group / 2 + offset] = code[0] & 15 | (code[2] & 15) << 4;
						low[group / 2 + offset + 32] = code[1] & 15 | (code[3] & 15) << 4;
						high[group / 4 + offset] = code[0] >> 4 | code[1] >> 4 << 2 | code[2] >> 4 << 4 | code[3] >> 4 << 6;
					}
				}
				data.extend(low);
				data.extend(high);
				data.extend(scales.map(|value| value as u8));
				put_half(&mut data, scale);
			}
			return Ok((data, Vec::new()));
		}
		if matches!(quantizer, Quantizer::Q8K) {
			let mut data = Vec::new();
			for values in weights.chunks(256) {
				let value = |index| values.get(index).copied().unwrap_or(0.0) as f32;
				let maximum = (0..256).map(value).max_by(|a, b| a.abs().total_cmp(&b.abs())).unwrap_or(0.0);
				let inverse = if maximum == 0.0 { 0.0 } else { -127.0 / maximum };
				let scale = if inverse == 0.0 { 0.0 } else { inverse.recip() };
				data.extend(scale.to_le_bytes());
				let codes = (0..256).map(|index| qround(inverse * value(index)).max(-128.0).min(127.0) as i8).collect::<Vec<_>>();
				data.extend(codes.iter().map(|code| *code as u8));
				for block in codes.chunks(16) {
					data.extend(block.iter().map(|code| i16::from(*code)).sum::<i16>().to_le_bytes())
				}
			}
			return Ok((data, Vec::new()));
		}
		if matches!(quantizer, Quantizer::Nf4) {
			const NF4: [f64; 16] = [
				-1.0,
				-0.6961928009986877,
				-0.5250730514526367,
				-0.39491748809814453,
				-0.28444138169288635,
				-0.18477343022823334,
				-0.09105003625154495,
				0.0,
				0.07958029955625534,
				0.16093020141124725,
				0.24611230194568634,
				0.33791524171829224,
				0.44070982933044434,
				0.5626170039176941,
				0.7229568362236023,
				1.0,
			];
			let mut metadata = vec![config.quantization_block as f64];
			metadata.extend(NF4);
			let mut data = vec![0_u8; weights.len().div_ceil(2)];
			for (block, values) in weights.chunks(config.quantization_block).enumerate() {
				let scale = values.iter().map(|value| value.abs()).max_by(f64::total_cmp).unwrap_or(0.0);
				metadata.push(scale);
				for (offset, weight) in values.iter().enumerate() {
					let index = block * config.quantization_block + offset;
					let code = nearest(std::slice::from_ref(&(if scale == 0.0 { 0.0 } else { weight / scale })), &NF4, 1).0 as u8;
					data[index / 2] |= code << (index % 2 * 4);
				}
			}
			return Ok((data, metadata));
		}
		if matches!(quantizer, Quantizer::Iq4Nl) {
			let mut data = Vec::new();
			for values in weights.chunks(32) {
				let values = (0..32).map(|index| values.get(index).copied().unwrap_or(0.0) as f32).collect::<Vec<_>>();
				let (scale, codes) = iq4_fit(&values, -1);
				put_half(&mut data, scale);
				for index in 0..16 {
					data.push(codes[index] | codes[index + 16] << 4)
				}
			}
			return Ok((data, Vec::new()));
		}
		if matches!(quantizer, Quantizer::Iq4Xs) {
			let mut data = Vec::new();
			for values in weights.chunks(256) {
				let values = (0..256).map(|index| values.get(index).copied().unwrap_or(0.0) as f32).collect::<Vec<_>>();
				let (mut scales, mut codes) = ([0.0_f32; 8], [0_u8; 256]);
				let (mut maximum, mut extreme) = (0.0, 0.0);
				for block in 0..8 {
					let (scale, fitted) = iq4_fit(&values[block * 32..block * 32 + 32], 7);
					scales[block] = scale;
					codes[block * 32..block * 32 + 32].copy_from_slice(&fitted);
					if scale.abs() > extreme {
						extreme = scale.abs();
						maximum = scale
					}
				}
				let scale = -maximum / 32.0;
				let stored_scale = unfp16(fp16(scale));
				let (mut high, mut low) = (0_u16, [0_u8; 4]);
				for block in 0..8 {
					let signed = if scale == 0.0 { 0 } else { qround(scales[block] / scale).max(-32.0).min(31.0) as i8 };
					let code = (signed + 32) as u8;
					low[block / 2] |= (code & 15) << (block % 2 * 4);
					high |= u16::from(code >> 4) << (block * 2);
					let d = stored_scale * f32::from(signed);
					let inverse = if d == 0.0 { 0.0 } else { d.recip() };
					for offset in 0..32 {
						codes[block * 32 + offset] = iq4_code(values[block * 32 + offset] * inverse)
					}
				}
				put_half(&mut data, scale);
				data.extend(high.to_le_bytes());
				data.extend(low);
				for block in 0..8 {
					for offset in 0..16 {
						data.push(codes[block * 32 + offset] | codes[block * 32 + offset + 16] << 4)
					}
				}
			}
			return Ok((data, Vec::new()));
		}
		if matches!(quantizer, Quantizer::Iq2Xxs) {
			require(
				importance.len() == weights.len() && importance.iter().all(|value| value.is_finite() && *value >= 0.0) && importance.iter().any(|value| *value > 0.0),
				"GGML IQ2_XXS requires trained importance weights",
			)?;
			return Ok((iq2_xxs(&weights.iter().map(|value| *value as f32).collect::<Vec<_>>(), &importance.iter().map(|value| *value as f32).collect::<Vec<_>>()), Vec::new()));
		}
		if matches!(quantizer, Quantizer::Iq2 { importance: true, xs: true }) {
			require(
				importance.len() == weights.len() && importance.iter().all(|value| value.is_finite() && *value >= 0.0) && importance.iter().any(|value| *value > 0.0),
				"GGML IQ2_XS requires trained importance weights",
			)?;
			return Ok((iq2_16(&weights.iter().map(|value| *value as f32).collect::<Vec<_>>(), Some(&importance.iter().map(|value| *value as f32).collect::<Vec<_>>()), true), Vec::new()));
		}
		if let Quantizer::Iq1 { medium } = quantizer {
			require(
				importance.len() == weights.len() && importance.iter().all(|value| value.is_finite() && *value >= 0.0) && importance.iter().any(|value| *value > 0.0),
				format!("GGML IQ1_{} requires trained importance weights", if medium { "M" } else { "S" }),
			)?;
			return Ok((iq1(&weights.iter().map(|value| *value as f32).collect::<Vec<_>>(), &importance.iter().map(|value| *value as f32).collect::<Vec<_>>(), medium), Vec::new()));
		}
		if matches!(quantizer, Quantizer::Iq3Xxs) {
			let weights = weights.iter().map(|value| *value as f32).collect::<Vec<_>>();
			let chunks = weights.chunks(256).collect::<Vec<_>>();
			return Ok((parallel_map(chunks.len(), |chunk| iq3_xxs(chunks[chunk]))?.concat(), Vec::new()));
		}
		if matches!(quantizer, Quantizer::Iq2 { importance: false, xs: false }) {
			return Ok((iq2_16(&weights.iter().map(|value| *value as f32).collect::<Vec<_>>(), None, false), Vec::new()));
		}
		if matches!(quantizer, Quantizer::Iq3S) {
			return Ok((iq3_s(&weights.iter().map(|value| *value as f32).collect::<Vec<_>>()), Vec::new()));
		}
		Err(self.unavailable())
	}
	fn decompress(self, data: &[u8], codebook: &[f64], count: usize) -> Result<Vec<f64>> {
		self.spec().ok_or_else(|| self.unavailable())?.codec.dequantize(data, codebook, count)
	}
}
pub struct Qi(pub Model, pub Model, QiSuffix);
#[doc(hidden)]
pub struct QiSuffix {
	pub nf: Model,
	pub k: Qk,
}
pub struct Qk {
	model: Model,
	pub s: Model,
	pub m: Model,
	pub l: Model,
}
pub struct Iq {
	pub xxs: Model,
	pub xs: Model,
	pub s: Model,
	pub m: Model,
	pub nl: Model,
}
impl std::ops::Deref for Qk {
	type Target = Model;
	fn deref(&self) -> &Model {
		&self.model
	}
}
impl std::ops::Deref for Qi {
	type Target = QiSuffix;
	fn deref(&self) -> &QiSuffix {
		&self.2
	}
}
impl Estimator {
	const fn name(&self) -> &'static str {
		self.name
	}
}
impl Operation {
	const fn name(&self) -> &'static str {
		match self {
			Self::Layer(_) => "layer",
			Self::Conv(..) => "conv",
			Self::Pool(_) => "pool",
			Self::Estimator(value) => value.name(),
			Self::Attention(..) => "attn",
			Self::Rnn(_) => "rnn",
			Self::Gru(_) => "gru",
			Self::Lstm(_) => "lstm",
			Self::Residual(_) => "residual",
			Self::Moe(..) => "moe",
			Self::Perceptron(_) => "perc",
			Self::Embed(..) => "embed",
			Self::Hyper(..) => "hyper",
			Self::Dconv(_) => "dconv",
			Self::Delta(..) => "delta",
			Self::Ple(_) => "ple",
			Self::Norm => "norm",
			Self::Glu(..) => "glu",
		}
	}
	/// Reports whether the operation owns weights that a qualifier can govern.
	fn weighted(&self) -> bool {
		let weighted_parts = |parts: &[Residual]| parts.iter().any(|part| !matches!(part, Residual::Activation(_)));
		match self {
			// An embedding table is the gather's context: never trained and always read packed.
			Self::Pool(_) | Self::Estimator(_) | Self::Embed(..) => false,
			Self::Residual(parts) => weighted_parts(parts),
			Self::Hyper(_, rank, blocks) => *rank != 0 || blocks.iter().any(|block| block.operation.weighted()),
			_ => true,
		}
	}
}
impl Activation {
	const fn name(self) -> &'static str {
		match self {
			Self::Linear => "linear",
			Self::Cos => "cos",
			Self::Exp => "exp",
			Self::Log => "log",
			Self::Ln => "ln",
			Self::Huber => "huber",
			Self::Tan => "tan",
			Self::Relu => "relu",
			Self::Leak => "leak",
			Self::Sigmoid => "sigmoid",
			Self::Tanh => "tanh",
			Self::Selu => "selu",
			Self::Gelu => "gelu",
			Self::Silu => "silu",
			Self::Elu => "elu",
			Self::Prelu => "prelu",
		}
	}
}
impl BlockNormalization {
	const fn name(self) -> &'static str {
		match self {
			Self::Batch => "bnorm",
			Self::Layer => "lnorm",
			Self::Rms => "rms",
			Self::L2 => "l2",
		}
	}
}
macro_rules! activations { ($(fn $method:ident = $activation:ident;)+) => {$(impl Model { pub fn $method(&self) -> Self {
self.activate(Activation::$activation) } })+}; }
activations! {
fn cos = Cos;
fn exp = Exp;
fn log = Log;
fn ln = Ln;
fn huber = Huber;
fn tan = Tan;
fn relu = Relu;
fn leak = Leak;
fn sigmoid = Sigmoid;
fn tanh = Tanh;
fn selu = Selu;
fn gelu = Gelu;
fn silu = Silu;
fn elu = Elu;
fn prelu = Prelu; }
pub struct Recipe;
pub struct Adamw;
#[derive(Clone, Copy)]
pub struct LossFunction(u8);
#[derive(Clone, Copy)]
pub struct Metric(u8);
pub struct ZScore;
pub type Normalization = fn(usize) -> Residual;
pub type Norm = Normalization;
pub type Loss = LossFunction;
pub const adamw: Adamw = Adamw;
pub const mse: LossFunction = LossFunction(0);
pub const rmse: LossFunction = LossFunction(1);
pub const huber: LossFunction = LossFunction(2);
pub const mae: LossFunction = LossFunction(3);
pub const bce: LossFunction = LossFunction(4);
// Width-one outputs make cross-entropy and binary cross-entropy the same computation: one identity.
pub const ce: LossFunction = LossFunction(4);
pub const focal: LossFunction = LossFunction(6);
pub const Run: Metric = Metric(0);
pub const Loss: Metric = Metric(1);
pub const R2: Metric = Metric(2);
pub const Time: Metric = Metric(3);
pub const Epoch: Metric = Metric(4);
pub const blck: Metric = Metric(5);
pub const atvn: Metric = Metric(6);
pub const norm: Metric = Metric(7);
pub const tok: Metric = Metric(8);
pub const quant: Metric = Metric(9);
pub const tile: Metric = Metric(10);
pub const all: [Metric; 10] = [Run, Time, Epoch, R2, Loss, blck, atvn, norm, quant, tile];
/// One metric or a set of them, so `.log(tile)` and `.log(all)` are the same call.
pub trait IntoMetrics {
	fn into_metrics(self) -> Vec<Metric>;
}
impl IntoMetrics for Metric {
	fn into_metrics(self) -> Vec<Metric> {
		vec![self]
	}
}
impl<const N: usize> IntoMetrics for [Metric; N] {
	fn into_metrics(self) -> Vec<Metric> {
		self.into()
	}
}
pub const z_score: ZScore = ZScore;
pub const batch: Batch = Batch;
#[derive(Clone, Copy, Debug)]
pub struct Batch;
pub const rms: Rms = Rms;
#[derive(Clone, Copy, Debug)]
pub struct Rms;
pub const l2: L2 = L2;
#[derive(Clone, Copy, Debug)]
pub struct L2;
impl LossFunction {
	const fn name(self) -> &'static str {
		match self.0 {
			0 => "mse",
			1 => "rmse",
			2 => "huber",
			3 => "mae",
			4 => "bce",
			6 => "focal",
			_ => unreachable!(),
		}
	}
	fn value(self, prediction: f64, target: f64, threshold: f64) -> f64 {
		let difference = prediction - target;
		let probability = logistic(prediction).clamp(f64::EPSILON, 1.0 - f64::EPSILON);
		match self.0 {
			0 | 1 => difference * difference,
			2 => {
				let absolute = difference.abs();
				if absolute <= threshold { 0.5 * difference * difference } else { threshold * (absolute - 0.5 * threshold) }
			}
			3 => difference.abs(),
			4 => -target * probability.ln() - (1.0 - target) * (1.0 - probability).ln(),
			6 => {
				let correct = if target >= 0.5 { probability } else { 1.0 - probability };
				-(1.0 - correct).powi(2) * correct.ln()
			}
			_ => f64::NAN,
		}
	}
}
impl Recipe {
	pub fn data<T: IntoDataSources>(&self, sources: T) -> Data {
		Data {
			sources: sources.into_data_sources(),
			tests: Vec::new(),
			autoregressive: T::AUTO,
			target: Vec::new(),
			features: FeatureSelection::All,
			broadcast: false,
			normalize: false,
			split: 1.0,
			prepared: OnceLock::new(),
		}
	}
	pub fn model(&self) -> Model {
		Model { blocks: Vec::new(), loss: mse, quantization: 0, epsilon: default_epsilon().unwrap_or_else(|error| panic!("{error}")), frozen: false, packed: false }
	}
	pub const fn train(&self) -> Train {
		Train { epochs: 1, learning_rate: 0.001, log_metrics: Vec::new(), stop: Some(1.0), resume: None, save: None, seed: None, precision: Compute::FP64 }
	}
}
impl Recipe {
	/// Opens a GGUF model, following every shard of a split, with its tensor data
	/// mapped rather than read.
	pub fn gguf(&self, path: impl AsRef<Path>) -> Gguf {
		let path = resolve_path(path).unwrap_or_else(|error| panic!("{error}"));
		Gguf::open(&path).unwrap_or_else(|error| panic!("{error}"))
	}
	pub fn infer(&self, path: impl AsRef<Path>, input: &[f64]) -> Vec<f64> {
		let path = resolve_path(path).unwrap_or_else(|error| panic!("{error}"));
		let device = selected_gpu().unwrap_or_else(|error| panic!("{error}"));
		let result = bundle::run_infer(&path, input, |stored, samples| {
			let config = Config::load()?;
			let graph = materialize_saved_graph(stored, samples, device, config)?;
			let tape = NativeTape::new(&graph, samples, samples, &[], device, stored.precision, None)?;
			tape.inject_bn_stats(&stored.bn_stats)?;
			tape.forward()?;
			tape.predictions()
		});
		result.unwrap_or_else(|error| panic!("{error}"))
	}
}
impl Gguf {
	/// Contracts `input` through the named tensor into `width` outputs: the
	/// one-node plan of `infer`, so the tensor's mapped bytes are the
	/// contraction's weight and the tape decodes the file's own blocks.
	pub fn contract(&self, name: &str, input: &[f64], width: usize) -> Vec<f64> {
		self.infer(&recipe.model().layer(width), &self.plan().named(self, name), input, input.len())
	}
	/// Contracts `input` through expert `index` of a `[k, n, experts]` tensor, over that
	/// expert's mapped blocks alone.
	pub fn expert(&self, name: &str, index: usize, input: &[f64], width: usize) -> Vec<f64> {
		let expert = self.named(name).expert(index).unwrap_or_else(|error| panic!("{error}"));
		self.infer(&recipe.model().layer(width), &self.plan().node(&[expert]), input, input.len())
	}
	/// Runs `blocks` over `input` with every parameterized node filled from
	/// `plan`, and returns the model's own output. `channels` names the input's
	/// channel axis, so the rest of its length is the sequence the blocks walk.
	/// The model compiles through the path a trained model takes; a bound node
	/// is an ordinary node whose weight is a view of the file, and a contraction
	/// bound to a weight without a bias row lowers and runs without one.
	pub fn infer(&self, blocks: &Model, plan: &Binding, input: &[f64], channels: usize) -> Vec<f64> {
		infer_gguf(self, blocks, plan, input, channels).unwrap_or_else(|error| panic!("{error}"))
	}
	/// Autoregressive decode over `blocks` bound from `plan`, as `recipe.decode`
	/// runs a saved model: one tape of `sequence` id positions holds every block's
	/// state, the prompt prefills it, and each step adds one id and forwards only
	/// the positions it reaches. The logits are the model's output after the last
	/// forward.
	pub fn decode(&self, blocks: &Model, plan: &Binding, sequence: usize, prompt: &[u32], sampler: &mut Sampler, stop: &[u32], budget: usize) -> Generation {
		decode_gguf(self, blocks, plan, sequence, prompt, sampler, stop, budget).unwrap_or_else(|error| panic!("{error}"))
	}
	/// An empty weight plan to fill from this model's tensors.
	pub fn plan(&self) -> Binding {
		Binding::default()
	}
	fn named(&self, name: &str) -> GgufTensor {
		self.tensor(name).unwrap_or_else(|| panic!("tensor {name} is absent")).clone()
	}
}
/// An ordered weight plan: the tensor views that fill each parameterized node
/// of a compiled model, one entry per node in the order the lowering pushes
/// them. An entry names as many views as the node's planes take, in the order
/// the lowering lays those planes out; a view is a whole tensor, one expert of a
/// `[k, n, experts]` tensor, or a run of output rows, and the views of one node
/// share a layout. A contraction whose views sum to its matrix binds no bias
/// row; one whose views sum to the matrix plus one output row binds the trailing
/// row as its bias.
#[derive(Clone, Default)]
pub struct Binding {
	pub(crate) nodes: Vec<Vec<Plane>>,
}
/// One plane of a node's weight: a view of a stored tensor, or values the host
/// rewrote once from a tensor the file stores in another parametrization, which
/// bind only as the node's values.
#[derive(Clone)]
pub(crate) enum Plane {
	Mapped(GgufTensor),
	Owned { name: String, values: Vec<f64> },
}
impl Plane {
	fn elements(&self) -> usize {
		match self {
			Self::Mapped(tensor) => tensor.elements(),
			Self::Owned { values, .. } => values.len(),
		}
	}
	fn name(&self) -> &str {
		match self {
			Self::Mapped(tensor) => &tensor.name,
			Self::Owned { name, .. } => name,
		}
	}
	fn mapped(&self) -> Option<&GgufTensor> {
		match self {
			Self::Mapped(tensor) => Some(tensor),
			Self::Owned { .. } => None,
		}
	}
}
impl Binding {
	/// The next parameterized node, filled from `planes` end to end.
	#[must_use]
	pub fn node(mut self, planes: &[GgufTensor]) -> Self {
		self.nodes.push(planes.iter().cloned().map(Plane::Mapped).collect());
		self
	}
	/// The next parameterized node, filled from one whole named tensor.
	#[must_use]
	pub fn named(self, model: &Gguf, name: &str) -> Self {
		let tensor = model.named(name);
		self.node(&[tensor])
	}
}
/// Compiles `blocks` over `input` with every parameterized node bound from
/// `plan`, and runs one forward. `channels` names the input's channel axis, so
/// the rest of its length is the sequence the blocks walk.
fn infer_gguf(model: &Gguf, blocks: &Model, plan: &Binding, input: &[f64], channels: usize) -> Result<Vec<f64>> {
	let (graph, device) = bound_graph(model, blocks, plan, input, channels)?;
	let tape = NativeTape::new(&graph, input, input, &[], device, Compute::FP64, None)?;
	tape.forward()?;
	tape.predictions()
}
fn decode_gguf(model: &Gguf, blocks: &Model, plan: &Binding, sequence: usize, prompt: &[u32], sampler: &mut Sampler, stop: &[u32], budget: usize) -> Result<Generation> {
	require(!prompt.is_empty(), "decode prompt is empty")?;
	require(checked_add(prompt.len(), budget, "decode length")? <= sequence, format!("decode of {} prompt ids and {budget} steps exceeds the sequence of {sequence}", prompt.len()))?;
	let mut samples = vec![0.0; sequence];
	for (slot, id) in samples.iter_mut().zip(prompt) {
		*slot = f64::from(*id);
	}
	let (graph, device) = bound_graph(model, blocks, plan, &samples, 1)?;
	let mut tape = NativeTape::new(&graph, &samples, &samples, &[], device, Compute::FP64, None)?;
	decode_steps(
		&mut tape,
		&mut samples,
		prompt,
		sampler,
		stop,
		budget,
		|_| Ok(()),
		|tape, samples, settled, reached| {
			for position in settled as usize..reached as usize {
				tape.write_sample(position, samples[position])?;
			}
			tape.forward_window(settled, reached)?;
			tape.predictions()
		},
	)
}
/// Compiles `blocks` over `input` and fills every weighted node from `plan`.
/// `channels` names the input's channel axis, so the rest of its length is the
/// sequence the blocks walk.
fn bound_graph(model: &Gguf, blocks: &Model, plan: &Binding, input: &[f64], channels: usize) -> Result<(Graph, &'static Gpu)> {
	require(channels != 0 && !input.is_empty() && input.len() % channels == 0, "the input is not a whole number of channel rows")?;
	let shape = Shape { channels, length: input.len() / channels };
	let (device, config) = (selected_gpu()?, Config::load()?);
	// A zero target width asks compile for the model's own output, so no
	// projection onto a target is appended to a bound graph.
	let data = Prepared {
		samples: input.to_vec(),
		targets: Vec::new(),
		target_width: 0,
		rows: 1,
		source_rows: 1,
		features: input.len(),
		schema: DataSchema::default(),
		sequence: Some((shape, shape)),
		target_categorical: false,
		norm_mean: Vec::new(),
		norm_scale: Vec::new(),
		identities: Vec::new(),
		fitted: Vec::new(),
		bound: Some(model.bound(plan)?),
	};
	let graph = compile(blocks, &data, &data.targets, 1, device, config, false)?;
	Ok((graph, device))
}
/// How an architecture pairs the channels its rotary embedding rotates. Recipe's
/// rope pairs each channel with the one half the rotated span away; an
/// architecture that pairs neighbouring channels binds its query and key rows in
/// the order that makes the two rotations agree.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RopePairs {
	Halves,
	Neighbours,
}
/// One row of the architecture table: the `general.architecture` names whose
/// standard metadata keys and tensor names map onto the same blocks, and the
/// convention those names leave implicit. Every dimension comes from the
/// `<architecture>.*` namespace and every weight from the `blk.<n>.*`,
/// `token_embd`, `output_norm` and `output` names, so a row adds no path of
/// its own.
struct Architecture {
	names: &'static [&'static str],
	rope: RopePairs,
}
const ARCHITECTURES: &[Architecture] = &[
	Architecture { names: &["llama"], rope: RopePairs::Neighbours },
	Architecture { names: &["qwen2", "qwen3", "qwen2moe", "qwen3moe"], rope: RopePairs::Halves },
	Architecture { names: &["qwen35", "qwen3next"], rope: RopePairs::Halves },
	Architecture { names: &["qwen4exp"], rope: RopePairs::Halves },
];
/// A model built from a GGUF file: the blocks its architecture metadata declares
/// and the plan that binds every weighted node to the file's tensors by name.
pub struct Bound {
	file: Gguf,
	model: Model,
	plan: Binding,
	blocks: usize,
	tensors: usize,
	vocabulary: usize,
}
impl Gguf {
	/// The model this file describes: `general.architecture` selects the row of
	/// the architecture table, the `<architecture>.*` namespace sizes every block,
	/// and each weighted node binds to the tensor of the standard name that
	/// holds its weight. A tensor no node reads, a node no tensor fills, or a
	/// tensor the file lacks is an error here, before any device is touched.
	pub fn model(&self) -> Bound {
		Builder::build(self).unwrap_or_else(|error| panic!("{error}"))
	}
}
impl Bound {
	/// The blocks the architecture declares, each an attention or delta block
	/// and its feed-forward, both on the residual stream.
	pub fn blocks(&self) -> usize {
		self.blocks
	}
	/// The stored tensors the plan reads, each counted once: as a mapped view, or
	/// as the values the host rewrote from it.
	pub fn tensors(&self) -> usize {
		self.tensors
	}
	pub fn vocabulary(&self) -> usize {
		self.vocabulary
	}
	/// The Recipe model the blocks lower through.
	pub fn model(&self) -> &Model {
		&self.model
	}
	/// The plan that fills the model's weighted nodes.
	pub fn plan(&self) -> &Binding {
		&self.plan
	}
	/// One forward over `ids`: the logit of every id at every position, laid out
	/// as `vocabulary` rows of one value per position.
	pub fn infer(&self, ids: &[u32]) -> Vec<f64> {
		let input = ids.iter().map(|id| f64::from(*id)).collect::<Vec<_>>();
		self.file.infer(&self.model, &self.plan, &input, 1)
	}
}
/// Walks the standard metadata and tensor names of one file, emitting the
/// Recipe blocks and, beside every block, the plan entries its weighted nodes
/// take, in the order the lowering pushes those nodes.
struct Builder<'a> {
	file: &'a Gguf,
	architecture: &'a str,
	rope: RopePairs,
	plan: Binding,
	consumed: std::collections::BTreeSet<String>,
}
/// The dimensions every row reads from the `<architecture>.*` namespace.
struct Dimensions {
	width: usize,
	heads: usize,
	kv: usize,
	head: usize,
	rope_dims: usize,
	rope_base: f64,
	/// Every `interval`th block attends and the rest are delta blocks; absent,
	/// every block attends.
	interval: Option<usize>,
	delta: Option<DeltaDims>,
	feed_forward: Option<usize>,
	experts: Option<ExpertDims>,
	hyper: Option<(usize, usize)>,
	indexer: Option<(usize, usize, usize)>,
	compression: Vec<usize>,
}
struct DeltaDims {
	heads: usize,
	key_heads: usize,
	state: usize,
	kernel: usize,
	inner: usize,
}
struct ExpertDims {
	count: usize,
	used: usize,
	hidden: usize,
	scoring: Scoring,
	renormalize: bool,
}
impl<'a> Builder<'a> {
	fn build(file: &'a Gguf) -> Result<Bound> {
		let architecture = file.required("general.architecture")?.text().ok_or_else(|| RecipeError::new("general.architecture is not a string"))?;
		let row = ARCHITECTURES.iter().find(|row| row.names.contains(&architecture)).ok_or_else(|| {
			let known = ARCHITECTURES.iter().flat_map(|row| row.names).copied().collect::<Vec<_>>().join(", ");
			RecipeError::new(format!("architecture {architecture:?} is not in the table; the table knows {known}"))
		})?;
		let mut builder = Self { file, architecture, rope: row.rope, plan: Binding::default(), consumed: std::collections::BTreeSet::new() };
		let dimensions = builder.dimensions()?;
		let blocks = builder.integer("block_count")?;
		let embedding = builder.tensor("token_embd.weight", "the embedding")?;
		require(
			embedding.shape.len() == 2 && embedding.shape[0] as usize == dimensions.width,
			format!("token_embd.weight has shape {:?}, not [{}, vocabulary]", embedding.shape, dimensions.width),
		)?;
		let vocabulary = embedding.shape[1] as usize;
		let format = gguf::block_format(&embedding).map_err(|_| RecipeError::new("token_embd.weight is not block quantized, and a gather reads packed rows"))?;
		let mut model = recipe.model();
		if builder.present("attention.layer_norm_rms_epsilon") {
			model = model.epsilon(file.float_at(&builder.key("attention.layer_norm_rms_epsilon"))?);
		}
		model = model.embed(vocabulary, dimensions.width);
		// The gather addresses rows of the file's own layout, so the block's
		// quantization is the tensor's format rather than a selection.
		let block = model.blocks.last_mut().ok_or_else(|| RecipeError::new("the embedding block is absent"))?;
		(block.quantization, block.profile) = (format.0, false);
		builder.mapped(vec![embedding.clone()]);
		let ple = if builder.present("ple.ngram_size") { Some(Ngram::new(file)?) } else { None };
		for layer in 0..blocks {
			if let Some(ple) = ple.as_ref().filter(|ple| ple.layer() == layer) {
				model = model.ple(ple);
				builder.ple(layer, ple, &dimensions)?;
			}
			let attends = dimensions.interval.is_none_or(|interval| (layer + 1) % interval == 0);
			let branch = builder.open(layer, "attn", &dimensions)?;
			let branch = if attends { builder.attention(branch, layer, &dimensions)? } else { builder.delta(branch, layer, &dimensions)? };
			model = builder.close(model, branch, &dimensions);
			let branch = builder.open(layer, "ffn", &dimensions)?;
			let branch = match &dimensions.experts {
				Some(experts) => builder.experts(branch, layer, experts, &dimensions)?,
				None => builder.feed_forward(branch, layer, &dimensions)?,
			};
			model = builder.close(model, branch, &dimensions);
		}
		if let Some(scale) = builder.optional("output_norm.weight") {
			model = model.norm(rms);
			builder.mapped(vec![scale]);
		} else if dimensions.hyper.is_some_and(|(_, rank)| rank != 0) {
			// The head mixer reads the stream through its own gates before the output.
			builder.whole("output_hc_norm.weight", "the head mixer normalization")?;
			builder.whole("output_hc_down.weight", "the head mixer read gate")?;
			builder.whole("output_hc_up.weight", "the head mixer read gate")?;
		}
		model = model.layer(vocabulary);
		// A file without an output tensor ties the output to the embedding.
		let output = match builder.optional("output.weight") {
			Some(output) => output,
			None => embedding,
		};
		builder.mapped(vec![output]);
		let unread = file.tensors().iter().filter(|tensor| !builder.consumed.contains(&tensor.name)).map(|tensor| tensor.name.as_str()).collect::<Vec<_>>();
		require(unread.is_empty(), format!("{} tensors are read by no node: {}", unread.len(), unread.join(", ")))?;
		let tensors = builder.consumed.len();
		Ok(Bound { file: file.clone(), model, plan: builder.plan, blocks, tensors, vocabulary })
	}
	fn key(&self, suffix: &str) -> String {
		format!("{}.{suffix}", self.architecture)
	}
	fn integer(&self, suffix: &str) -> Result<usize> {
		self.file.integer_at(&self.key(suffix))
	}
	fn integer_or(&self, suffix: &str, default: usize) -> Result<usize> {
		match self.file.value(&self.key(suffix)) {
			Some(_) => self.integer(suffix),
			None => Ok(default),
		}
	}
	fn present(&self, suffix: &str) -> bool {
		self.file.value(&self.key(suffix)).is_some()
	}
	fn dimensions(&self) -> Result<Dimensions> {
		let width = self.integer("embedding_length")?;
		let heads = self.integer("attention.head_count")?;
		let kv = self.integer_or("attention.head_count_kv", heads)?;
		require(heads != 0 && width % heads == 0 || self.present("attention.key_length"), format!("{} heads do not partition a stream of {width}", heads))?;
		let head = self.integer_or("attention.key_length", width / heads.max(1))?;
		let value = self.integer_or("attention.value_length", head)?;
		require(value == head, format!("attention keys are {head} wide and values {value}; the attention block takes one head width"))?;
		let rope_dims = self.integer_or("rope.dimension_count", head)?;
		let rope_base = if self.present("rope.freq_base") { self.file.float_at(&self.key("rope.freq_base"))? } else { 10000.0 };
		let interval = if self.present("full_attention_interval") { Some(self.integer("full_attention_interval")?) } else { None };
		let delta = match interval {
			Some(_) => Some(DeltaDims {
				heads: self.integer("ssm.time_step_rank")?,
				key_heads: self.integer("ssm.group_count")?,
				state: self.integer("ssm.state_size")?,
				kernel: self.integer("ssm.conv_kernel")?,
				inner: self.integer("ssm.inner_size")?,
			}),
			None => None,
		};
		let experts = match self.integer_or("expert_count", 0)? {
			0 => None,
			count => Some(ExpertDims {
				count,
				used: self.integer("expert_used_count")?,
				hidden: self.integer("expert_feed_forward_length")?,
				scoring: match self.integer_or("expert_gating_func", 1)? {
					1 => Scoring::Softmax,
					2 => Scoring::Sigmoid,
					other => return Err(RecipeError::new(format!("expert gating function {other} is unknown"))),
				},
				renormalize: match self.file.value(&self.key("expert_weights_norm")) {
					Some(GgufValue::Bool(value)) => *value,
					Some(_) => return Err(RecipeError::new("expert_weights_norm is not a boolean")),
					None => true,
				},
			}),
		};
		let feed_forward = if self.present("feed_forward_length") { Some(self.integer("feed_forward_length")?) } else { None };
		require(experts.is_some() || feed_forward.is_some(), "the architecture names neither a feed-forward width nor experts")?;
		let hyper = if self.present("hyper_connection.count") { Some((self.integer("hyper_connection.count")?, self.integer("hyper_connection.low_rank")?)) } else { None };
		let indexer = if self.present("attention.indexer.head_count") {
			Some((self.integer("attention.indexer.head_count")?, self.integer("attention.indexer.key_length")?, self.integer("attention.indexer.top_k")?))
		} else {
			None
		};
		let compression = if indexer.is_some() { self.file.indices_at(&self.key("attention.compress_ratios"))? } else { Vec::new() };
		Ok(Dimensions { width, heads, kv, head, rope_dims, rope_base, interval, delta, feed_forward, experts, hyper, indexer, compression })
	}
	/// The named tensor, which `role` reads, marked as read.
	fn tensor(&mut self, name: &str, role: &str) -> Result<GgufTensor> {
		let tensor = self.file.tensor(name).ok_or_else(|| RecipeError::new(format!("tensor {name} is absent; {role} reads it")))?.clone();
		self.consumed.insert(name.to_owned());
		Ok(tensor)
	}
	fn optional(&mut self, name: &str) -> Option<GgufTensor> {
		let tensor = self.file.tensor(name)?.clone();
		self.consumed.insert(name.to_owned());
		Some(tensor)
	}
	/// The next parameterized node, filled from mapped views.
	fn mapped(&mut self, planes: Vec<GgufTensor>) {
		self.slot(planes.into_iter().map(Plane::Mapped).collect());
	}
	fn slot(&mut self, planes: Vec<Plane>) {
		self.plan.nodes.push(planes);
	}
	/// The next weighted node, filled from one whole named tensor.
	fn whole(&mut self, name: &str, role: &str) -> Result<()> {
		let tensor = self.tensor(name, role)?;
		self.mapped(vec![tensor]);
		Ok(())
	}
	/// A projection `[inputs, outputs]` of the given name, checked against the
	/// widths the node contracts over.
	fn projection(&mut self, name: &str, role: &str, inputs: usize, outputs: usize) -> Result<GgufTensor> {
		let tensor = self.tensor(name, role)?;
		require(
			tensor.shape.len() == 2 && tensor.shape[0] as usize == inputs && tensor.shape[1] as usize == outputs,
			format!("{name} has shape {:?}; {role} contracts {inputs} inputs into {outputs} outputs", tensor.shape),
		)?;
		Ok(tensor)
	}
	/// A normalization scale of `width` values, repeated over `groups` groups of
	/// the span it normalizes, in the order `order` reads each group's channels.
	fn scale(&mut self, name: &str, role: &str, width: usize, groups: usize, order: &[usize]) -> Result<Vec<Plane>> {
		let tensor = self.tensor(name, role)?;
		require(tensor.elements() == width, format!("{name} holds {} values; {role} scales {width} channels", tensor.elements()))?;
		if order.iter().enumerate().all(|(index, channel)| index == *channel) {
			return Ok(vec![Plane::Mapped(tensor); groups]);
		}
		let values = self.file.values(&tensor)?;
		let permuted = order.iter().map(|channel| values[*channel]).collect::<Vec<_>>();
		Ok(vec![Plane::Owned { name: format!("{name} (paired)"), values: permuted }; groups])
	}
	/// The order Recipe reads one head's rows in: the channels the rotation pairs
	/// as neighbours moved into halves, the rest in place.
	fn head_order(&self, width: usize, dims: usize) -> Vec<usize> {
		match self.rope {
			RopePairs::Halves => (0..width).collect(),
			RopePairs::Neighbours => (0..dims).step_by(2).chain((1..dims).step_by(2)).chain(dims..width).collect(),
		}
	}
	/// The rows of one head at `base`, as one view when they are read in place
	/// and as one view per row otherwise.
	fn head_rows(tensor: &GgufTensor, base: usize, order: &[usize]) -> Result<Vec<GgufTensor>> {
		if order.iter().enumerate().all(|(index, channel)| index == *channel) {
			return Ok(vec![tensor.rows(base, order.len())?]);
		}
		order.iter().map(|channel| tensor.rows(base + channel, 1)).collect()
	}
	/// One attention block and the plan of its projection, its query and key
	/// scales, and its output projection.
	fn attention(&mut self, branch: Model, layer: usize, dimensions: &Dimensions) -> Result<Model> {
		let Dimensions { width, heads, kv, head, rope_dims, rope_base, .. } = *dimensions;
		let name = |suffix: &str| format!("blk.{layer}.{suffix}");
		let role = format!("block {layer} attention");
		let query = self.tensor(&name("attn_q.weight"), &role)?;
		require(query.shape.len() == 2 && query.shape[0] as usize == width, format!("{} has shape {:?}; {role} contracts {width} inputs", query.name, query.shape))?;
		let gated = match query.shape[1] as usize {
			outputs if outputs == heads * head => false,
			outputs if outputs == 2 * heads * head => true,
			outputs => return Err(RecipeError::new(format!("{} projects {outputs} outputs; {heads} heads of {head} take {} or, gated, {}", query.name, heads * head, 2 * heads * head))),
		};
		let key = self.projection(&name("attn_k.weight"), &role, width, kv * head)?;
		let value = self.projection(&name("attn_v.weight"), &role, width, kv * head)?;
		let order = self.head_order(head, rope_dims);
		let stride = if gated { 2 * head } else { head };
		let mut planes = Vec::new();
		for index in 0..heads {
			planes.extend(Self::head_rows(&query, index * stride, &order)?);
		}
		for index in 0..kv {
			planes.extend(Self::head_rows(&key, index * head, &order)?);
		}
		planes.push(value);
		let mut block = branch.attn(heads).kv(kv).head(head);
		if let Some((index_heads, index_width, top_k)) = dimensions.indexer {
			let block_size = dimensions.compression.get(layer).copied().filter(|ratio| *ratio != 0).unwrap_or(1);
			planes.push(self.projection(&name("indexer.q_proj.weight"), &role, width, index_heads * index_width)?);
			planes.push(self.projection(&name("indexer.k_proj.weight"), &role, width, index_width)?);
			block = block.index(index_heads, index_width, block_size, top_k.div_ceil(block_size).max(1));
		}
		if gated {
			for index in 0..heads {
				planes.push(query.rows(index * stride + head, head)?);
			}
			block = block.gate();
		}
		let normalized = self.file.tensor(&name("attn_q_norm.weight")).is_some();
		if normalized {
			block = block.qk(rms);
		}
		block = block.rope(rope_dims, rope_base);
		self.mapped(planes);
		if normalized {
			let mut scales = self.scale(&name("attn_q_norm.weight"), &role, head, heads, &order)?;
			scales.extend(self.scale(&name("attn_k_norm.weight"), &role, head, kv, &order)?);
			self.slot(scales);
		}
		let output = self.projection(&name("attn_output.weight"), &role, heads * head, width)?;
		self.mapped(vec![output]);
		Ok(block)
	}
	/// One gated delta rule block and the plan of its gate projection, its
	/// query-key-value projection, its convolution taps, its decay, its output
	/// scale, its output gate and its output projection.
	fn delta(&mut self, branch: Model, layer: usize, dimensions: &Dimensions) -> Result<Model> {
		let width = dimensions.width;
		let DeltaDims { heads, key_heads, state, kernel, inner } = *dimensions.delta.as_ref().ok_or_else(|| RecipeError::new("the architecture declares delta blocks without ssm dimensions"))?;
		require(inner == heads * state, format!("ssm.inner_size {inner} is not {heads} value heads of {state}"))?;
		let name = |suffix: &str| format!("blk.{layer}.{suffix}");
		let role = format!("block {layer} delta");
		let alpha = self.projection(&name("ssm_alpha.weight"), &role, width, heads)?;
		let beta = self.projection(&name("ssm_beta.weight"), &role, width, heads)?;
		let mut gates = vec![Plane::Mapped(alpha), Plane::Mapped(beta)];
		// The decay bias offsets the alpha half; the beta half has none, so the
		// bias row the node binds ends with zeros there.
		if let Some(bias) = self.optional(&name("ssm_dt.bias")) {
			require(bias.elements() == heads, format!("{} holds {} values; {role} offsets {heads} decay gates", bias.name, bias.elements()))?;
			gates.push(Plane::Mapped(bias));
			gates.push(Plane::Owned { name: name("ssm_beta.bias (zero)"), values: vec![0.0; heads] });
		}
		self.slot(gates);
		let conv_width = 2 * key_heads * state + inner;
		let qkv = self.projection(&name("attn_qkv.weight"), &role, width, conv_width)?;
		self.mapped(vec![qkv]);
		let taps = self.tensor(&name("ssm_conv1d.weight"), &role)?;
		require(
			taps.shape.len() == 2 && taps.shape[0] as usize == kernel && taps.shape[1] as usize == conv_width,
			format!("{} has shape {:?}; {role} convolves {conv_width} channels with {kernel} taps", taps.name, taps.shape),
		)?;
		self.mapped(vec![taps]);
		// The file stores the decay as `-exp(A)`; the delta node takes `A`.
		let decay = self.tensor(&name("ssm_a"), &role)?;
		let values = self.file.values(&decay)?;
		require(values.len() == heads && values.iter().all(|value| *value < 0.0), format!("{} holds {} values; {role} takes {heads} negative decays", decay.name, values.len()))?;
		self.slot(vec![Plane::Owned { name: format!("{} (ln(-a))", decay.name), values: values.iter().map(|value| (-value).ln()).collect() }]);
		let order = (0..state).collect::<Vec<_>>();
		let scales = self.scale(&name("ssm_norm.weight"), &role, state, heads, &order)?;
		self.slot(scales);
		let gate = self.projection(&name("attn_gate.weight"), &role, width, inner)?;
		self.mapped(vec![gate]);
		let output = self.projection(&name("ssm_out.weight"), &role, inner, width)?;
		self.mapped(vec![output]);
		Ok(branch.delta(heads, kernel).keys(key_heads, state).values(state).out(width))
	}
	/// One gated feed-forward and the plan of its gate, up and down projections.
	fn feed_forward(&mut self, branch: Model, layer: usize, dimensions: &Dimensions) -> Result<Model> {
		let width = dimensions.width;
		let hidden = dimensions.feed_forward.ok_or_else(|| RecipeError::new("the architecture names no feed-forward width"))?;
		let name = |suffix: &str| format!("blk.{layer}.{suffix}");
		let role = format!("block {layer} feed-forward");
		for (suffix, inputs, outputs) in [("ffn_gate.weight", width, hidden), ("ffn_up.weight", width, hidden), ("ffn_down.weight", hidden, width)] {
			let tensor = self.projection(&name(suffix), &role, inputs, outputs)?;
			self.mapped(vec![tensor]);
		}
		Ok(branch.glu(hidden, Activation::Silu))
	}
	/// One mixture of experts and the plan of its router, its expert tables and
	/// its shared expert.
	fn experts(&mut self, branch: Model, layer: usize, experts: &ExpertDims, dimensions: &Dimensions) -> Result<Model> {
		let width = dimensions.width;
		let ExpertDims { count, used, hidden, scoring, renormalize } = *experts;
		let name = |suffix: &str| format!("blk.{layer}.{suffix}");
		let role = format!("block {layer} experts");
		let router = self.projection(&name("ffn_gate_inp.weight"), &role, width, count)?;
		self.mapped(vec![router]);
		for (suffix, inputs, outputs) in [("ffn_gate_exps.weight", width, hidden), ("ffn_up_exps.weight", width, hidden), ("ffn_down_exps.weight", hidden, width)] {
			let table = self.tensor(&name(suffix), &role)?;
			require(
				table.shape.len() == 3 && table.shape[0] as usize == inputs && table.shape[1] as usize == outputs && table.shape[2] as usize == count,
				format!("{} has shape {:?}; {role} holds {count} experts of [{inputs}, {outputs}]", table.name, table.shape),
			)?;
			self.mapped(vec![table]);
		}
		let shared = self.file.tensor(&name("ffn_gate_shexp.weight")).is_some();
		if shared {
			let shared_role = format!("block {layer} shared expert");
			// The per-position gate is the first weighted node in the shared path.
			let gate = self.tensor(&name("ffn_gate_inp_shexp.weight"), &shared_role)?;
			require(gate.elements() == width, format!("{} holds {} values; {shared_role} gate takes {width}", gate.name, gate.elements()))?;
			self.mapped(vec![gate]);
			for (suffix, inputs, outputs) in [("ffn_gate_shexp.weight", width, hidden), ("ffn_up_shexp.weight", width, hidden), ("ffn_down_shexp.weight", hidden, width)] {
				let tensor = self.projection(&name(suffix), &shared_role, inputs, outputs)?;
				self.mapped(vec![tensor]);
			}
		}
		Ok(branch.moe(count, used, hidden, Activation::Silu, scoring, renormalize, shared))
	}
	/// One per-layer embedding and the plan of its host table, key and value
	/// projections, grouped normalization scales, and dilated depthwise taps.
	fn ple(&mut self, layer: usize, ple: &Ngram<'_>, dimensions: &Dimensions) -> Result<()> {
		let role = format!("block {layer} per-layer embedding");
		let name = |suffix: &str| format!("blk.{layer}.{suffix}");
		let (table_name, _, _) = ple.table();
		let table = self.tensor(table_name, &role)?;
		self.mapped(vec![table]);
		let lanes = dimensions.hyper.map_or(1, |(lanes, _)| lanes);
		let stream = checked_mul(lanes, dimensions.width, "per-layer embedding stream")?;
		let gathered = ple.width();
		let key = self.projection(&name("ple_key.weight"), &role, gathered, stream)?;
		self.mapped(vec![key]);
		self.whole(&name("ple_norm_key.weight"), &role)?;
		self.whole(&name("ple_norm_query.weight"), &role)?;
		let value = self.projection(&name("ple_value.weight"), &role, gathered, dimensions.width)?;
		self.mapped(vec![value]);
		self.whole(&name("ple_norm_conv.weight"), &role)?;
		let taps = self.tensor(&name("ple_conv1d.weight"), &role)?;
		require(
			taps.shape.len() == 2 && taps.shape[0] as usize == ple.kernel() && taps.shape[1] as usize == stream,
			format!("{} has shape {:?}; {role} convolves {stream} channels with {} taps", taps.name, taps.shape, ple.kernel()),
		)?;
		self.mapped(vec![taps]);
		Ok(())
	}
	/// Opens the `part` branch of block `layer` on the residual stream. Under the
	/// hyper-connection mixer the architecture declares, the mixer's gates bind
	/// ahead of the branch and the branch starts empty; on a plain residual the
	/// branch starts with its pre-normalization, whose scale binds first.
	fn open(&mut self, layer: usize, part: &str, dimensions: &Dimensions) -> Result<Model> {
		let name = |suffix: &str| format!("blk.{layer}.{suffix}");
		match dimensions.hyper {
			Some((lanes, rank)) => {
				if rank != 0 {
					let role = format!("block {layer} {part} mixer");
					let stream = lanes * dimensions.width;
					self.whole(&name(&format!("hc_{part}_norm.weight")), &role)?;
					let down = self.projection(&name(&format!("hc_{part}_down.weight")), &role, stream, rank)?;
					let up = self.projection(&name(&format!("hc_{part}_up.weight")), &role, rank, stream)?;
					let inject = self.projection(&name(&format!("hc_{part}_inject.weight")), &role, stream, lanes)?;
					self.mapped(vec![down]);
					self.mapped(vec![up]);
					self.mapped(vec![inject]);
				}
				Ok(recipe.model())
			}
			None => {
				let role = format!("block {layer} {part} pre-normalization");
				let scale = match part {
					"attn" => self.tensor(&name("attn_norm.weight"), &role)?,
					_ => match self.optional(&name("ffn_norm.weight")) {
						Some(scale) => scale,
						None => self.tensor(&name("post_attention_norm.weight"), &role)?,
					},
				};
				require(scale.elements() == dimensions.width, format!("{} holds {} values; {role} scales {} channels", scale.name, scale.elements(), dimensions.width))?;
				self.mapped(vec![scale]);
				Ok(recipe.model().norm(rms))
			}
		}
	}
	/// Closes a branch: the mixer's lanes and rank under hyper-connections, and
	/// one ungated lane, the plain residual, otherwise.
	fn close(&self, model: Model, branch: Model, dimensions: &Dimensions) -> Model {
		let (lanes, rank) = dimensions.hyper.unwrap_or((1, 0));
		model.hyper(lanes, rank, &branch)
	}
}
/// One id at a time from a model's logits: a repetition penalty over the
/// recent ids, then top-k, top-p, min-p, temperature, and a seeded draw.
/// Temperature zero is greedy.
pub struct Sampler {
	temperature: f64,
	top_k: usize,
	top_p: f64,
	min_p: f64,
	penalty: f64,
	window: usize,
	state: u64,
}
impl Sampler {
	pub fn temperature(mut self, value: f64) -> Self {
		self.temperature = value;
		self
	}
	/// Keep the `count` highest logits; zero keeps every id.
	pub fn top_k(mut self, count: usize) -> Self {
		self.top_k = count;
		self
	}
	/// Keep the smallest set of ids whose probability sums to at least `mass`.
	pub fn top_p(mut self, mass: f64) -> Self {
		self.top_p = mass;
		self
	}
	/// Drop ids whose probability is below `ratio` times the highest.
	pub fn min_p(mut self, ratio: f64) -> Self {
		self.min_p = ratio;
		self
	}
	/// Divide positive logits of the last `window` ids by `penalty`, and
	/// multiply negative ones.
	pub fn repeat(mut self, penalty: f64, window: usize) -> Self {
		(self.penalty, self.window) = (penalty, window);
		self
	}
	pub fn seed(mut self, seed: u64) -> Self {
		self.state = seed;
		self
	}
	pub fn sample(&mut self, logits: &[f64], history: &[u32]) -> u32 {
		assert!(!logits.is_empty(), "sampler received no logits");
		let mut candidates = logits.iter().copied().enumerate().map(|(id, logit)| (id as u32, logit)).collect::<Vec<_>>();
		for id in history.iter().rev().take(self.window) {
			let logit = &mut candidates[*id as usize].1;
			*logit = if *logit > 0.0 { *logit / self.penalty } else { *logit * self.penalty };
		}
		candidates.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
		if self.temperature <= 0.0 {
			return candidates[0].0;
		}
		if self.top_k != 0 {
			candidates.truncate(self.top_k);
		}
		let probabilities = |candidates: &[(u32, f64)], temperature: f64| {
			let weights = candidates.iter().map(|(_, logit)| ((logit - candidates[0].1) / temperature).exp()).collect::<Vec<_>>();
			let total = weights.iter().sum::<f64>();
			weights.into_iter().map(|weight| weight / total).collect::<Vec<_>>()
		};
		let mass = probabilities(&candidates, 1.0);
		let mut kept = 1;
		let mut cumulative = mass[0];
		while kept < candidates.len() && cumulative < self.top_p && mass[kept] >= self.min_p * mass[0] {
			cumulative += mass[kept];
			kept += 1;
		}
		candidates.truncate(kept);
		let mass = probabilities(&candidates, self.temperature);
		self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		let mut draw = (self.state >> 11) as f64 / (1_u64 << 53) as f64;
		for (candidate, probability) in candidates.iter().zip(&mass) {
			if draw < *probability {
				return candidate.0;
			}
			draw -= probability;
		}
		candidates.last().unwrap().0
	}
}
/// What `decode` produced: the prompt followed by the generated ids, the logits
/// of the last forward, the seconds of the prefill, and the seconds of each
/// later step.
pub struct Generation {
	pub ids: Vec<u32>,
	pub logits: Vec<f64>,
	pub prefill_seconds: f64,
	pub step_seconds: Vec<f64>,
}
impl Recipe {
	pub fn sampler(&self) -> Sampler {
		Sampler { temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0, penalty: 1.0, window: 64, state: 0x9E37_79B9_7F4A_7C15 }
	}
	/// Autoregressive decode over a saved model on the primary device: the
	/// model placed as one range, decoded by [`Placed::decode`] over one tape.
	pub fn decode(&self, path: impl AsRef<Path>, prompt: &[u32], sampler: &mut Sampler, stop: &[u32], budget: usize) -> Generation {
		self.place_primary(path).try_decode(prompt, sampler, stop, budget, |_| Ok(())).unwrap_or_else(|error| panic!("{error}"))
	}
	/// Answer `requests` decodes over HTTP on the primary device, as
	/// [`Placed::serve`] does over a placement.
	pub fn serve(&self, path: impl AsRef<Path>, address: &str, requests: usize) {
		self.place_primary(path).serve(address, requests);
	}
	/// Place a saved model across the selected devices: `split` gives every
	/// device, in `--device` order, the number of blocks it takes, and an empty
	/// split is measured from the free memory of each device.
	pub fn place(&self, path: impl AsRef<Path>, split: &[usize]) -> Placed {
		selected_gpus().and_then(|devices| place_model(path.as_ref(), split, devices)).unwrap_or_else(|error| panic!("{error}"))
	}
	/// The model placed as one range on the primary device.
	fn place_primary(&self, path: impl AsRef<Path>) -> Placed {
		selected_gpus().and_then(|devices| place_model(path.as_ref(), &[], &devices[..1])).unwrap_or_else(|error| panic!("{error}"))
	}
}
fn request_field<'a>(query: &'a str, name: &str) -> Option<&'a str> {
	query.split('&').find_map(|pair| pair.strip_prefix(name)?.strip_prefix('=')).filter(|value| !value.is_empty())
}
fn request_number<T: std::str::FromStr>(query: &str, name: &str) -> Result<Option<T>> {
	request_field(query, name).map(|value| value.parse().map_err(|_| RecipeError::new(format!("request {name} is {value:?}, which is not a number")))).transpose()
}
fn request_ids(query: &str, name: &str) -> Result<Vec<u32>> {
	request_field(query, name)
		.map_or_else(|| Ok(Vec::new()), |value| value.split(',').map(|id| id.parse().map_err(|_| RecipeError::new(format!("request {name} holds {id:?}, which is not an id")))).collect())
}
fn try_serve(placed: &Placed, address: &str, requests: usize) -> Result<()> {
	use std::io::Write as _;
	let listener = std::net::TcpListener::bind(address).map_err(|error| RecipeError::new(format!("cannot serve decode on {address}: {error}")))?;
	for _ in 0..requests {
		let mut stream = listener.accept().map_err(|error| RecipeError::new(format!("cannot accept a decode request: {error}")))?.0;
		if let Err(error) = serve_decode(placed, &mut stream) {
			let body = error.to_string();
			let answer = format!("HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
			stream.write_all(answer.as_bytes()).map_err(|error| RecipeError::new(format!("cannot answer a decode request: {error}")))?;
		}
	}
	Ok(())
}
fn serve_decode(placed: &Placed, stream: &mut std::net::TcpStream) -> Result<()> {
	use std::io::{Read as _, Write as _};
	let mut head = Vec::new();
	let mut byte = [0_u8; 1];
	while !head.ends_with(b"\r\n\r\n") {
		require(head.len() < 8192, "decode request head is longer than 8192 bytes")?;
		let read = stream.read(&mut byte).map_err(|error| RecipeError::new(format!("cannot read a decode request: {error}")))?;
		require(read == 1, "decode request ended before its head")?;
		head.push(byte[0]);
	}
	let head = String::from_utf8_lossy(&head).into_owned();
	let target = head.split_whitespace().nth(1).ok_or_else(|| RecipeError::new("decode request names no target"))?;
	let query = target.split_once('?').map_or("", |(_, query)| query);
	let prompt = request_ids(query, "ids")?;
	let stop = request_ids(query, "stop")?;
	let budget = request_number(query, "budget")?.unwrap_or(16);
	// An absent field keeps the sampler's own default rather than restating it.
	let mut sampler = recipe.sampler();
	if let Some(value) = request_number(query, "temperature")? {
		sampler = sampler.temperature(value);
	}
	if let Some(value) = request_number(query, "top_k")? {
		sampler = sampler.top_k(value);
	}
	if let Some(value) = request_number(query, "top_p")? {
		sampler = sampler.top_p(value);
	}
	if let Some(value) = request_number(query, "min_p")? {
		sampler = sampler.min_p(value);
	}
	if let Some(value) = request_number(query, "penalty")? {
		sampler = sampler.repeat(value, request_number(query, "penalty_window")?.unwrap_or(64));
	}
	if let Some(value) = request_number(query, "seed")? {
		sampler = sampler.seed(value);
	}
	let write = |stream: &mut std::net::TcpStream, bytes: &[u8]| {
		stream.write_all(bytes).and_then(|()| stream.flush()).map_err(|error| RecipeError::new(format!("cannot answer a decode request: {error}")))
	};
	write(stream, b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n")?;
	placed.try_decode(&prompt, &mut sampler, &stop, budget, |id| {
		let chunk = format!("{id}\n");
		write(stream, format!("{:x}\r\n{chunk}\r\n", chunk.len()).as_bytes())
	})?;
	write(stream, b"0\r\n\r\n")
}
/// One id at a time over a tape whose input is a sequence of ids. The prefill
/// runs the prompt, every step adds one id and forwards the positions it reaches
/// through `logits`, and `samples` holds the whole sequence as the ids settle.
fn decode_steps(
	tape: &mut NativeTape, samples: &mut [f64], prompt: &[u32], sampler: &mut Sampler, stop: &[u32], budget: usize, mut emit: impl FnMut(u32) -> Result<()>,
	mut logits: impl FnMut(&mut NativeTape, &[f64], u32, u32) -> Result<Vec<f64>>,
) -> Result<Generation> {
	let mut generation = Generation { ids: prompt.to_vec(), logits: Vec::new(), prefill_seconds: 0.0, step_seconds: Vec::new() };
	let mut settled = 0;
	for step in 0..=budget {
		let reached = narrow(generation.ids.len(), "decode position")? as u32;
		let started = std::time::Instant::now();
		let logits = logits(tape, samples, settled, reached)?;
		let seconds = started.elapsed().as_secs_f64();
		if step == 0 {
			generation.prefill_seconds = seconds;
		} else {
			generation.step_seconds.push(seconds);
		}
		settled = reached;
		generation.logits = logits;
		if step == budget {
			break;
		}
		let id = sampler.sample(&generation.logits, &generation.ids);
		emit(id)?;
		samples[generation.ids.len()] = f64::from(id);
		generation.ids.push(id);
		if stop.contains(&id) {
			break;
		}
	}
	Ok(generation)
}
/// A saved model placed across the selected devices: contiguous block ranges,
/// each held by one persistent tape on its own device, run in sequence with
/// the stream moved at every hop. The tapes are the model's state, so a decode
/// extends them one position at a time.
pub struct Placed {
	graphs: Vec<bundle::SemanticGraph>,
	split: Vec<usize>,
	/// One tape per range of every graph, on the range's device.
	tapes: Vec<Vec<NativeTape>>,
	resident: Vec<usize>,
	moved: usize,
}
/// The saved statistics a batch normalization carries into inference, as the
/// node index and the channels it normalizes.
fn carried_state(nodes: &[Node]) -> impl Iterator<Item = (usize, usize)> + '_ {
	nodes.iter().enumerate().filter_map(|(index, node)| (node.op == Primitive::Normalize && node.argument[0] == 0.0).then_some((index, node.output.channels)))
}
/// The bytes a device holds for the nodes `range` of an inference graph, as
/// the tape lays them out: each node's packed or decoded weights, and its
/// value and context arenas for one row, which keep every position's
/// activations, attention keys and values, recurrent state, convolution tail
/// and batch normalization statistics.
fn resident_bytes(graph: &Graph, range: std::ops::Range<usize>, precision: Compute) -> Result<usize> {
	let element = precision.bytes();
	let mut bytes = 0;
	for index in range {
		let node = &graph.nodes[index];
		let weights = match packed_weight(graph, index, true) {
			Some(weight) => weight.bytes.len(),
			None => checked_mul(node.parameters, element, "resident weights")?,
		};
		for span in [align(weights, element)?, align(graph_rows_buffer(node.output, 1, element)?, element.max(8))?, align(node_context(node, 1, element)?, element.max(8))?] {
			bytes = checked_add(bytes, span, "resident bytes")?;
		}
	}
	Ok(bytes)
}
/// The input row a range starting at node `start` holds: the model's ids for
/// a gather, else the stream the previous node leaves.
fn input_row_bytes(graph: &Graph, start: usize, element: usize) -> Result<usize> {
	let (shape, element) = match start {
		0 if graph.nodes.first().is_some_and(|node| node.op == Primitive::Gather) => (graph.input, size_of::<i32>()),
		0 => (graph.input, element),
		_ => (graph.nodes[start - 1].output, element),
	};
	checked_mul(shape.elements(), element, "input row bytes")
}
/// Whether a device boundary before node `start` cuts a connection into a later
/// node: a residual reaching back over it, or the model input.
fn cuts_connection(graph: &Graph, start: usize) -> bool {
	graph.nodes[start..].iter().flat_map(|node| [(node.source, 1), (node.second, 0)]).any(|(index, first)| match usize::try_from(index) {
		Ok(from) => from + first < start,
		Err(_) => index == -1 && start != 0,
	})
}
/// Blocks per device measured from the free memory of each: a block joins the
/// current device while its resident bytes, weights and state, fit and the
/// boundary before it cuts no connection, so the device listed last takes the
/// tail.
fn measured_split(graph: &Graph, precision: Compute, devices: &[&'static Gpu]) -> Result<Vec<usize>> {
	let reserve = natural("placement launch reserve bytes", env!("RECIPE_PLACEMENT_LAUNCH_RESERVE_BYTES"))? as u64;
	let available = |device: &&'static Gpu| device.free_bytes().map(|free| free.saturating_sub(reserve));
	let mut starts = Vec::new();
	for (index, node) in graph.nodes.iter().enumerate() {
		if index == 0 || node.block_index != graph.nodes[index - 1].block_index {
			starts.push(index)
		}
	}
	let (mut split, mut taken, mut free) = (Vec::new(), 0, available(&devices[0])?);
	for (block, &start) in starts.iter().enumerate() {
		let end = starts.get(block + 1).copied().unwrap_or(graph.nodes.len());
		let resident = resident_bytes(graph, start..end, precision)? as u64;
		if taken != 0 && resident > free && split.len() + 1 < devices.len() && !cuts_connection(graph, start) {
			split.push(taken);
			(taken, free) = (0, available(&devices[split.len()])?);
		}
		if taken == 0 {
			free = free.saturating_sub(input_row_bytes(graph, start, precision.bytes())? as u64);
		}
		free = free.saturating_sub(resident);
		taken += 1;
	}
	require(taken != 0, "model has no block")?;
	split.push(taken);
	Ok(split)
}
/// The nodes of the blocks each device takes, rebased so every part is a graph
/// of its own whose input is the previous part's output.
fn split_graph(graph: &Graph, split: &[usize]) -> Result<Vec<Graph>> {
	let (mut parts, mut start, mut first_block) = (Vec::new(), 0, 0);
	for (device, blocks) in split.iter().enumerate() {
		first_block += blocks;
		let end = if device + 1 == split.len() { graph.nodes.len() } else { graph.nodes.iter().position(|node| node.block_index >= first_block).unwrap_or(graph.nodes.len()) };
		require(end > start, format!("device {device} takes no blocks"))?;
		require(!cuts_connection(graph, start), format!("the split cuts a connection into block {}", graph.nodes[start].block_index))?;
		let base = graph.nodes[start].offset;
		let rebase = |index: i32| {
			if index >= start as i32 {
				index - start as i32
			} else if index == -2 {
				-2
			} else {
				-1
			}
		};
		let nodes = graph.nodes[start..end]
			.iter()
			.map(|node| {
				require(node.op != Primitive::Predictor, "estimator blocks cannot be placed across devices")?;
				Ok(Node { source: rebase(node.source), second: rebase(node.second), offset: node.offset - base, ..node.clone() })
			})
			.collect::<Result<Vec<_>>>()?;
		let last = &graph.nodes[end - 1];
		let parameters = last.offset + last.parameters;
		parts.push(Graph {
			nodes,
			parameters: graph.parameters[base..parameters].to_vec(),
			frozen: graph.frozen[base..parameters].to_vec(),
			programs: graph.programs.clone(),
			lanes: graph.lanes,
			rank: graph.rank,
			stored: graph.stored[start..end].to_vec(),
			input: if start == 0 { graph.input } else { graph.nodes[start - 1].output },
			output: last.output,
			source: (end - start) as i32 - 1,
			state: TrainingState::default(),
			block_index: last.block_index,
			block_kind: last.block_kind,
			block_frozen: false,
			block_packed: false,
			bound: None,
			bound_values: Vec::new(),
			epsilon: graph.epsilon,
		});
		start = end;
	}
	Ok(parts)
}
/// The runs of a one-row arena that positions `begin..end` of every channel
/// occupy: the whole arena when the window spans it, else one run per channel,
/// as a row holds each channel's positions together.
fn window_runs(shape: Shape, begin: u32, end: u32) -> Vec<(usize, usize)> {
	let (begin, end) = (begin as usize, end as usize);
	if begin == 0 && end == shape.length { vec![(0, shape.elements())] } else { (0..shape.channels).map(|channel| (channel * shape.length + begin, end - begin)).collect() }
}
/// Place a saved model over `devices`: every range gets its tape, created once
/// on its device with the batch normalization statistics its blocks carry.
fn place_model(path: &Path, split: &[usize], devices: &'static [&'static Gpu]) -> Result<Placed> {
	let path = resolve_path(path)?;
	let (_, graphs) = bundle::load_semantic(&path)?;
	let (mut split, mut tapes, mut resident, mut moved) = (split.to_vec(), Vec::new(), vec![0; devices.len()], 0);
	for stored in &graphs {
		let graph = materialize_saved_graph(stored, &vec![0.0; stored.input.elements()], devices[0], Config::load()?)?;
		if split.is_empty() {
			split = measured_split(&graph, stored.precision, devices)?;
		}
		let blocks = stored.model.blocks.len();
		require(split.len() <= devices.len(), format!("the split names {} devices but {} are selected", split.len(), devices.len()))?;
		require(split.iter().sum::<usize>() == blocks, format!("the split places {} blocks but the model has {blocks}", split.iter().sum::<usize>()))?;
		let (mut ranges, mut statistics) = (Vec::new(), 0);
		let tokens = vec![0.0; stored.input.elements()];
		for (index, (part, device)) in split_graph(&graph, &split)?.iter().zip(devices).enumerate() {
			let tape = range_tape(part, &vec![0.0; part.input.elements()], &tokens, device, stored, &mut statistics)?;
			resident[index] += tape.resident_bytes();
			if index + 1 < split.len() {
				moved += part.output.channels * stored.precision.bytes();
			}
			ranges.push(tape);
		}
		tapes.push(ranges);
	}
	Ok(Placed { graphs, split, tapes, resident, moved })
}
impl Placed {
	pub fn infer(&self, input: &[f64]) -> Vec<f64> {
		let mut graph = 0;
		bundle::infer_graphs(&self.graphs, input, |_, samples| {
			let tapes = &self.tapes[graph];
			graph += 1;
			self.forward_window(tapes, samples, 0, tapes.first().map_or(0, |tape| tape.positions))
		})
		.unwrap_or_else(|error| panic!("{error}"))
	}
	/// Autoregressive decode over the placed model, whose input is a sequence
	/// of ids and whose output is one logit per id. The tapes hold the state of
	/// every block for the whole decode: the prefill fills them from the prompt
	/// and each step extends them by the one position the new id reaches, so no
	/// step runs a position the decode has already settled. The decode ends at
	/// a `stop` id, after `budget` ids, or when the ids fill the model's
	/// sequence.
	pub fn decode(&self, prompt: &[u32], sampler: &mut Sampler, stop: &[u32], budget: usize) -> Generation {
		self.try_decode(prompt, sampler, stop, budget, |_| Ok(())).unwrap_or_else(|error| panic!("{error}"))
	}
	/// Answer `requests` decodes over HTTP and return. A request names its prompt
	/// in the target, as `GET /decode?ids=3,1,4&budget=16&stop=2&temperature=0.8&seed=7`,
	/// and the answer sends each id as its own chunk as the decode reaches it.
	pub fn serve(&self, address: &str, requests: usize) {
		try_serve(self, address, requests).unwrap_or_else(|error| panic!("{error}"));
	}
	/// The blocks each device takes, in `--device` order.
	pub fn split(&self) -> &[usize] {
		&self.split
	}
	/// Bytes each device holds, in `--device` order: its ranges' weights,
	/// input rows, and the arenas that keep their state.
	pub fn resident_bytes(&self) -> &[usize] {
		&self.resident
	}
	/// Bytes one token moves across all hops.
	pub fn moved_bytes(&self) -> usize {
		self.moved
	}
	fn try_decode(&self, prompt: &[u32], sampler: &mut Sampler, stop: &[u32], budget: usize, mut emit: impl FnMut(u32) -> Result<()>) -> Result<Generation> {
		let (stored, tapes) = match (self.graphs.as_slice(), self.tapes.as_slice()) {
			([only], [tapes]) => (only, tapes.as_slice()),
			_ => return Err(RecipeError::new(format!("decode expects a model of one graph, this model has {}", self.graphs.len()))),
		};
		let sequence = stored.inputs.len();
		require(!prompt.is_empty(), "decode prompt is empty")?;
		require(stored.input.channels == 1 || stored.input.length == 1, "decode expects one input value per position")?;
		require(checked_add(prompt.len(), budget, "decode length")? <= sequence, format!("decode of {} prompt ids and {budget} steps exceeds the model sequence of {sequence}", prompt.len()))?;
		let mut samples = vec![0.0; sequence];
		for (slot, id) in samples.iter_mut().zip(prompt) {
			*slot = f64::from(*id);
		}
		let mut generation = Generation { ids: prompt.to_vec(), logits: Vec::new(), prefill_seconds: 0.0, step_seconds: Vec::new() };
		let mut settled = 0;
		for step in 0..=budget {
			let reached = narrow(generation.ids.len(), "decode position")? as u32;
			let started = std::time::Instant::now();
			let logits = bundle::infer_graphs(&self.graphs, &samples, |_, prepared| self.forward_window(tapes, prepared, settled, reached))?;
			let seconds = started.elapsed().as_secs_f64();
			if step == 0 {
				generation.prefill_seconds = seconds;
			} else {
				generation.step_seconds.push(seconds);
			}
			settled = reached;
			generation.logits = logits;
			if step == budget {
				break;
			}
			let id = sampler.sample(&generation.logits, &generation.ids);
			emit(id)?;
			samples[generation.ids.len()] = f64::from(id);
			generation.ids.push(id);
			if stop.contains(&id) {
				break;
			}
		}
		Ok(generation)
	}
	/// Run the input positions `begin..end` through every range in order. A
	/// window from position zero starts a new sequence. The window's rows of
	/// `samples` enter the first range, each range writes the positions the
	/// window reaches and keeps them as its state, and only the window's rows of
	/// the stream hop to the next device. Returns the last range's output.
	fn forward_window(&self, tapes: &[NativeTape], samples: &[f64], begin: u32, end: u32) -> Result<Vec<f64>> {
		let (Some(first), Some(last)) = (tapes.first(), tapes.last()) else { return Err(RecipeError::new("placement has no range")) };
		if begin == 0 {
			tapes.iter().try_for_each(NativeTape::reset_sequence)?;
		}
		let token_window = samples.get(begin as usize..end as usize).ok_or_else(|| RecipeError::new("token window is outside the model input"))?;
		tapes.iter().try_for_each(|tape| tape.write_tokens(begin as usize, token_window))?;
		for (start, count) in first.input_runs(begin, end) {
			first.write_samples(start, samples.get(start..start + count).ok_or_else(|| RecipeError::new("input window is outside the model input"))?)?;
		}
		let (mut begin, mut end) = (begin, end);
		for (index, tape) in tapes.iter().enumerate() {
			tape.forward_window(begin, end)?;
			let Some(next) = tapes.get(index + 1) else { break };
			(begin, end) = tape.output_window(begin, end)?;
			for (start, count) in window_runs(tape.output, begin, end) {
				next.write_samples(start, &tape.output(start, count)?)?;
			}
		}
		last.predictions()
	}
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Shape {
	channels: usize,
	length: usize,
}
impl Shape {
	fn elements(self) -> usize {
		self.channels * self.length
	}
}
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
enum Primitive {
	Contraction = 0,
	Pool = 2,
	Attention = 4,
	Scan = 5,
	Elementwise = 6,
	Normalize = 8,
	Predictor = 9,
	Gather = 10,
	Rope = 11,
	Expand = 12,
	Read = 13,
	Outer = 14,
	TopK = 15,
	ExpertIn = 16,
	Dconv = 17,
	Delta = 18,
	ExpertOut = 19,
	/// Rows gathered on the host from a table the device never holds.
	Lookup = 20,
	/// Every group of channels summed into one.
	Fold = 21,
}
struct ScalarProgram(Vec<f64>);
impl ScalarProgram {
	fn op(&mut self, opcode: ScalarOpcode, left: f64, right: f64) -> f64 {
		let result = (self.0.len() / 3) as f64;
		self.0.extend([opcode as i32 as f64, left, right]);
		result
	}
	fn constant(&mut self, value: f64) -> f64 {
		self.op(ScalarOpcode::Constant, value, 0.0)
	}
	// Both branches are always evaluated as straight-line SSA, so the untaken
	// branch must be dropped by selection, never by arithmetic masking: a
	// multiply blend turns an infinite untaken value into 0*inf = NaN.
	fn select(&mut self, condition: f64, value: f64) -> f64 {
		self.op(ScalarOpcode::Select, condition, value)
	}
	fn choose(&mut self, condition: f64, yes: f64, no: f64) -> f64 {
		let one = self.constant(1.0);
		let inv = self.op(ScalarOpcode::Subtract, one, condition);
		let (a, b) = (self.select(condition, yes), self.select(inv, no));
		self.op(ScalarOpcode::Add, a, b)
	}
	fn unary(&mut self, opcode: ScalarOpcode, value: f64) -> f64 {
		self.op(opcode, value, 0.0)
	}
}
impl Node {
	// A gather owns no trainable parameters: its table is the packed context the
	// kernel reads, so its tensor spans the vocabulary instead of a parameter span.
	// A lookup's table stays on the host and spans its rows of the head width.
	fn weights(&self) -> usize {
		match self.op {
			Primitive::Gather => (self.argument[0] as usize).saturating_mul(self.output.channels),
			Primitive::Lookup => (self.argument[2] as usize).saturating_mul(self.argument[1] as usize),
			_ => self.parameters,
		}
	}
	/// Whether the node's weights are a table it reads rows of rather than a parameter span.
	fn table(&self) -> bool {
		matches!(self.op, Primitive::Gather | Primitive::Lookup)
	}
	fn identity(&self, index: usize) -> String {
		let prim = match self.op {
			Primitive::Contraction => "Contraction",
			Primitive::Pool => "Pool",
			Primitive::Attention => "Attention",
			Primitive::Scan => "Scan",
			Primitive::Elementwise => "Elementwise",
			Primitive::Normalize => "Normalize",
			Primitive::Predictor => "Predictor",
			Primitive::Gather => "Gather",
			Primitive::Rope => "Rope",
			Primitive::Expand => "Expand",
			Primitive::Read => "Read",
			Primitive::Outer => "Outer",
			Primitive::Dconv => "Dconv",
			Primitive::Delta => "Delta",
			Primitive::TopK => "TopK",
			Primitive::ExpertIn => "ExpertIn",
			Primitive::ExpertOut => "ExpertOut",
			Primitive::Lookup => "Lookup",
			Primitive::Fold => "Fold",
		};
		format!(
			"block {} {}, node {} {}, input {}x{}, output {}x{}, offset={} count={}, source={}",
			self.block_index, self.block_kind, index, prim, self.input.channels, self.input.length, self.output.channels, self.output.length, self.offset, self.parameters, self.source
		)
	}
}
#[derive(Clone)]
struct Node {
	op: Primitive,
	source: i32,
	second: i32,
	input: Shape,
	output: Shape,
	offset: usize,
	parameters: usize,
	argument: [f64; 9],
	program_offset: usize,
	program_count: usize,
	block_index: usize,
	block_kind: &'static str,
	frozen: bool,
	packed: bool,
}
#[derive(Clone, Default)]
struct TrainingState {
	moments: Vec<f64>,
	variances: Vec<f64>,
	best_loss: Vec<f64>,
	trained_samples: Vec<u64>,
	epoch: usize,
	training_rows: usize,
}
#[derive(Clone)]
struct Graph {
	nodes: Vec<Node>,
	parameters: Vec<f64>,
	frozen: Vec<u8>,
	programs: Vec<f64>,
	stored: Vec<Option<StoredWeight>>,
	input: Shape,
	output: Shape,
	source: i32,
	state: TrainingState,
	block_index: usize,
	block_kind: &'static str,
	lanes: usize,
	rank: usize,
	block_frozen: bool,
	block_packed: bool,
	/// The weights still to bind while a graph compiles over mapped tensors:
	/// each parameterized node takes the front entry as it is pushed.
	bound: Option<std::collections::VecDeque<BoundNode>>,
	/// Bound values by node, written into their spans once every lowering has
	/// set its own initial parameters.
	bound_values: Vec<(usize, Vec<f64>)>,
	/// The model's normalization epsilon, which every lowered normalization reads.
	epsilon: f64,
}
impl Graph {
	fn new(shape: Shape, epsilon: f64) -> Self {
		Self {
			epsilon,
			nodes: Vec::new(),
			parameters: Vec::new(),
			frozen: Vec::new(),
			programs: Vec::new(),
			stored: Vec::new(),
			input: shape,
			output: shape,
			source: -1,
			lanes: 0,
			rank: 0,
			state: TrainingState::default(),
			block_index: 0,
			block_kind: "",
			block_frozen: false,
			block_packed: false,
			bound: None,
			bound_values: Vec::new(),
		}
	}
	fn refresh_storage(&mut self, config: Config) -> Result<()> {
		encode_graph_storage(self, config)
	}
}
fn encode_graph_storage(graph: &mut Graph, config: Config) -> Result<()> {
	require(graph.stored.len() == graph.nodes.len(), "model graph storage spans are incomplete")?;
	for (index, node) in graph.nodes.iter().enumerate() {
		// A weight bound from a mapped file has no arithmetic to encode from and
		// stays the bytes it was bound to.
		if graph.stored[index].as_ref().is_some_and(|stored| stored.arithmetic.is_empty()) {
			continue;
		}
		// A table owns no parameter span, so it is the tensor it was drawn into, and
		// one that carries no quantization stays as drawn.
		let drawn = graph.stored[index].take().filter(|_| node.table()).map(|stored| stored.arithmetic);
		let weights = drawn.as_deref().unwrap_or(&graph.parameters[node.offset..node.offset + node.parameters]);
		if weights.is_empty() || node.argument[8] == 0.0 {
			if let Some(arithmetic) = drawn {
				graph.stored[index] = Some(bundle::raw_weight(&arithmetic));
			}
			continue;
		}
		let format = StorageFormat(node.argument[8] as u16);
		require(format.spec().is_some(), format.unavailable().to_string())?;
		// A node that never received gradient, like an unrouted expert, has an all-zero
		// variance slice: it carries no importance signal, so weight it uniformly.
		let importance = graph
			.state
			.variances
			.get(node.offset..node.offset + node.parameters)
			.filter(|values| values.len() == weights.len() && values.iter().any(|value| *value > 0.0))
			.map_or_else(|| vec![1.0; weights.len()], |values| values.to_vec());
		graph.stored[index] = Some(format.encode(weights, &importance, config)?);
	}
	Ok(())
}
fn sequential_operation(operation: &Operation) -> bool {
	match operation {
		Operation::Conv(..) | Operation::Pool(..) | Operation::Attention(..) | Operation::Dconv(..) | Operation::Delta(..) | Operation::Ple(..) => true,
		Operation::Residual(parts) => parts.iter().any(|part| matches!(part, Residual::Conv(..))),
		Operation::Hyper(_, _, blocks) => blocks.iter().any(|block| sequential_operation(&block.operation)),
		_ => false,
	}
}
fn compile(model: &Model, data: &Prepared, targets: &[f64], rows: usize, gpu: &'static Gpu, config: Config, initialize: bool) -> Result<Graph> {
	require(!model.blocks.is_empty(), "model must contain a block")?;
	if let Some(format) = model.blocks.iter().map(|block| StorageFormat(block.quantization)).find(|format| format.0 != 0 && !format.valid()) {
		return Err(format.unavailable());
	}
	let sequence = data.sequence.map(|(sequence, attention)| if matches!(model.blocks[0].operation, Operation::Attention(_)) { attention } else { sequence });
	// A sequential block anywhere in the model needs the sequence axis, including inside a residual or hyper branch.
	let sequential = model.blocks.iter().any(|block| sequential_operation(&block.operation));
	let shape = if sequential { sequence.unwrap_or(Shape { channels: 1, length: data.features }) } else { Shape { channels: data.features, length: 1 } };
	let mut graph = Graph::new(shape, model.epsilon);
	graph.bound = data.bound.clone().map(std::collections::VecDeque::from);
	for (index, block) in model.blocks.iter().enumerate() {
		graph.block_index = index;
		graph.block_kind = block.operation.name();
		graph.block_frozen = block.frozen;
		graph.block_packed = block.packed;
		lower_block(&mut graph, block, model.blocks.len(), data, targets, rows, gpu, config)?;
	}
	graph.block_frozen = false;
	graph.block_packed = false;
	if graph.lanes != 0 {
		lower_collapse(&mut graph, config)?;
	}
	let mut output_profile = model.blocks.last().filter(|block| block.profile).map(|block| StorageFormat(block.quantization));
	// A model whose last block already emits one value per target needs no projection; the
	// channel and length are checked separately because a matching element count can still
	// be the wrong shape for the projection's bias.
	if data.target_width != 0 && (graph.output.channels != data.target_width || graph.output.length != 1) {
		let length = graph.output.length;
		lower_conv(&mut graph, data.target_width, length)?;
		if model.quantization != 0 {
			graph.nodes.last_mut().unwrap().argument[8] = f64::from(model.quantization)
		}
		output_profile = StorageFormat(model.quantization).selection().map(|_| StorageFormat(model.quantization));
	}
	if let Some(left) = graph.bound.take().filter(|plan| !plan.is_empty()) {
		return Err(RecipeError::new(format!("the plan names {} more weights than the model has parameterized nodes, starting with {}", left.len(), left[0].names)));
	}
	for (index, values) in std::mem::take(&mut graph.bound_values) {
		let (offset, parameters) = (graph.nodes[index].offset, graph.nodes[index].parameters);
		graph.parameters[offset..offset + parameters].copy_from_slice(&values);
	}
	if let Some(format) = output_profile
		&& let Some(node) = graph.nodes.iter_mut().rev().find(|node| node.op != Primitive::Predictor && node.weights() != 0 && node.block_index + 1 == model.blocks.len())
	{
		node.argument[8] = f64::from(format.tensor(0, false, true))
	}
	if initialize {
		initialize_graph(&mut graph, config);
		if let Some(offset) = output_bias_offset(&graph) {
			let mean = data.targets[..rows].iter().sum::<f64>() / rows as f64;
			graph.parameters[offset] = mean;
		}
	}
	// A frozen block keeps its initialized weights, so the mask lands after initialization.
	for (offset, parameters) in graph.nodes.iter().filter(|node| node.frozen).map(|node| (node.offset, node.parameters)).collect::<Vec<_>>() {
		graph.frozen[offset..offset + parameters].fill(1);
	}
	encode_graph_storage(&mut graph, config)?;
	Ok(graph)
}
fn materialize_saved_graph(saved: &bundle::SemanticGraph, samples: &[f64], gpu: &'static Gpu, config: Config) -> Result<Graph> {
	let prepared = Prepared {
		samples: samples.to_vec(),
		targets: vec![0.0; saved.output.elements()],
		target_width: saved.output.elements().max(1),
		rows: 1,
		source_rows: 1,
		features: saved.input.elements(),
		schema: DataSchema::default(),
		sequence: (saved.input.length > 1).then_some((saved.input, saved.input)),
		target_categorical: false,
		norm_mean: saved.norm_mean.clone(),
		norm_scale: saved.norm_scale.clone(),
		identities: Vec::new(),
		fitted: saved.predictors.clone(),
		bound: None,
	};
	let mut graph = compile(&saved.model, &prepared, &prepared.targets, 1, gpu, config, false)?;
	require(graph.input == saved.input, "saved semantic input shape does not match the compiled model")?;
	require(graph.output == saved.output, "saved semantic output shape does not match the compiled model")?;
	require(graph.nodes.iter().map(Node::weights).sum::<usize>() == saved.tensors.iter().map(|tensor| tensor.count).sum::<usize>(), "saved semantic weights do not match the compiled model")?;
	let mut tensor = 0;
	for (index, node) in graph.nodes.iter().enumerate() {
		if node.weights() == 0 {
			continue;
		}
		let encoded = saved.tensors.get(tensor).ok_or_else(|| RecipeError::new("saved semantic tensor is absent"))?;
		require(encoded.count == node.weights(), "saved semantic tensor has the wrong shape")?;
		graph.parameters[node.offset..node.offset + node.parameters].copy_from_slice(&encoded.arithmetic[..node.parameters]);
		if let Some(slot) = graph.stored.get_mut(index) {
			*slot = (encoded.format.0 != 0 || node.table()).then_some(encoded.clone())
		}
		tensor += 1;
	}
	require(tensor == saved.tensors.len(), "saved semantic tensors are incomplete")?;
	graph.frozen = saved.frozen.clone();
	graph.state = saved.state.clone();
	Ok(graph)
}
/// The graph either side of a block boundary: the nodes before `block`, when it
/// is not the first, and the nodes from it on, whose reads of the boundary node
/// become reads of the stream the host hands back.
fn split_at_block(graph: &Graph, block: usize) -> Result<(Option<Graph>, Graph)> {
	let at = graph.nodes.iter().position(|node| node.block_index >= block).unwrap_or(graph.nodes.len());
	require(at < graph.nodes.len(), format!("the model has no block {block} to inject before"))?;
	let boundary = at as i32 - 1;
	let rebase = |reference: i32| match reference {
		_ if reference == boundary => Ok(-1),
		-2 => Ok(-2),
		_ if reference > boundary => Ok(reference - at as i32),
		_ => Err(RecipeError::new(format!("block {block} reads node {reference}, which an earlier block holds"))),
	};
	let mut tail = graph.clone();
	tail.nodes = graph.nodes[at..].to_vec();
	tail.stored = graph.stored[at..].to_vec();
	for node in &mut tail.nodes {
		node.source = rebase(node.source)?;
		node.second = rebase(node.second)?;
	}
	tail.input = tail.nodes[0].input;
	tail.source = tail.nodes.len() as i32 - 1;
	let head = (at != 0).then(|| {
		let mut head = graph.clone();
		head.nodes.truncate(at);
		head.stored.truncate(at);
		head.output = graph.nodes[at - 1].output;
		head.source = boundary;
		head
	});
	Ok((head, tail))
}
/// The tape of one part of a split graph on its device, holding the batch
/// normalization statistics that follow the parts already made.
fn range_tape(graph: &Graph, samples: &[f64], tokens: &[f64], gpu: &'static Gpu, stored: &bundle::SemanticGraph, statistics: &mut usize) -> Result<NativeTape> {
	let tape = NativeTape::new(graph, samples, tokens, &[], gpu, stored.precision, None)?;
	let count = tape.batch_normalizations.iter().map(|(_, channels)| 2 * channels).sum::<usize>();
	tape.inject_bn_stats(stored.bn_stats.get(*statistics..*statistics + count).ok_or_else(|| RecipeError::new("saved batch normalization statistics are incomplete"))?)?;
	*statistics += count;
	Ok(tape)
}
/// One part of a split graph run forward on its device.
fn forward_part(graph: &Graph, samples: &[f64], tokens: &[f64], gpu: &'static Gpu, stored: &bundle::SemanticGraph, statistics: &mut usize) -> Result<Vec<f64>> {
	let tape = range_tape(graph, samples, tokens, gpu, stored, statistics)?;
	tape.forward()?;
	tape.predictions()
}
fn append_graph(graph: &mut Graph, mut part: Graph) -> Result<i32> {
	let source = graph.source;
	let (node_base, weight_base) = (narrow(graph.nodes.len(), "model graph nodes")?, graph.parameters.len());
	let program_base = graph.programs.len();
	for node in &mut part.nodes {
		node.source = if node.source < 0 { source } else { node.source + node_base };
		if node.second >= 0 {
			node.second += node_base
		}
		node.offset = checked_add(node.offset, weight_base, "model weight offset")?;
		if node.program_count != 0 {
			node.program_offset = checked_add(node.program_offset, program_base, "model program offset")?;
		}
	}
	graph.parameters.extend(part.parameters);
	graph.frozen.extend(part.frozen);
	graph.programs.extend(part.programs);
	graph.stored.extend(part.stored);
	graph.nodes.extend(part.nodes);
	graph.output = part.output;
	graph.source = narrow(graph.nodes.len(), "model graph nodes")? - 1;
	Ok(graph.source)
}
fn lower_block(graph: &mut Graph, block: &Block, total: usize, data: &Prepared, targets: &[f64], rows: usize, gpu: &'static Gpu, config: Config) -> Result<()> {
	// A per-layer embedding adds into the lanes-wide stream itself, so it keeps
	// the stream open as a hyper-connection block does.
	if graph.lanes != 0 && !matches!(block.operation, Operation::Hyper(..) | Operation::Ple(..)) {
		lower_collapse(graph, config)?;
	}
	let skip = graph.source;
	let first = graph.nodes.len();
	match &block.operation {
		Operation::Layer(width) | Operation::Perceptron(width) => lower_project(graph, *width)?,
		Operation::Conv(f, k) => lower_conv(graph, *f, *k)?,
		Operation::Pool(size) => lower_pool(graph, *size)?,
		Operation::Embed(vocabulary, width) => lower_embed(graph, *vocabulary, *width)?,
		Operation::Dconv(kernel) => lower_dconv(graph, *kernel, 1)?,
		Operation::Delta(delta) => lower_delta(graph, *delta, config)?,
		Operation::Ple(ple) => lower_ple(graph, ple, config)?,
		Operation::Attention(attention) => lower_attention(graph, *attention, block.qk)?,
		Operation::Rnn(width) => lower_scan(graph, *width, 1)?,
		Operation::Gru(width) => lower_scan(graph, *width, 3)?,
		Operation::Lstm(width) => lower_scan(graph, *width, 4)?,
		Operation::Residual(parts) => lower_residual(graph, parts, skip, config)?,
		Operation::Moe(experts, top_k, hidden, activation, scoring, renormalize, shared) => lower_moe(graph, *experts, *top_k, *hidden, *activation, *scoring, *renormalize, *shared, config)?,
		Operation::Hyper(lanes, rank, blocks) => lower_hyper(graph, *lanes, *rank, blocks, total, data, targets, rows, gpu, config)?,
		Operation::Norm => require(block.normalization.is_some(), "a leading normalization block names no normalization")?,
		Operation::Glu(hidden, activation) => lower_glu(graph, *hidden, *activation, config)?,
		Operation::Estimator(estimator) => {
			initialize_graph(graph, config);
			lower_estimator(graph, estimator, data, targets, rows, gpu, config)?
		}
	}
	if block.activation != Activation::Linear {
		lower_activation(graph, block.activation, config)?;
	}
	if let Some(normalization) = block.normalization {
		let channels = graph.output.channels;
		lower_normalize(graph, normalization, channels, channels)?;
	}
	if block.quantization != 0 {
		let more = graph.block_index < total / 8 || graph.block_index >= 7 * total / 8 || (graph.block_index - total / 8) % 3 == 2;
		let mut parameter = 0;
		for node in &mut graph.nodes[first..] {
			if !matches!(node.op, Primitive::Predictor | Primitive::Normalize) && node.weights() != 0 {
				let role = if block.operation.name() == "attn" { parameter } else { 0 };
				node.argument[8] = f64::from(if block.profile { StorageFormat(block.quantization).tensor(role, more, false) } else { block.quantization });
				parameter += 1
			}
		}
	}
	let elements = checked_mul(rows, graph.output.elements(), "node batch")?;
	narrow(elements, "GPU node batch")?;
	Ok(())
}
fn push_node(graph: &mut Graph, op: Primitive, output: Shape, parameters: usize, argument: [f64; 9], second: i32) -> Result<()> {
	let (source, offset, index) = (graph.source, graph.parameters.len(), graph.nodes.len());
	let mut node = Node {
		op,
		source,
		second,
		input: graph.output,
		output,
		offset,
		parameters,
		argument,
		program_offset: 0,
		program_count: 0,
		block_index: graph.block_index,
		block_kind: graph.block_kind,
		frozen: graph.block_frozen,
		packed: graph.block_packed,
	};
	// A graph compiled over mapped tensors fills each parameterized node from
	// the next plan entry. Block bytes stay packed and the node reserves no
	// arithmetic span; unblocked values wait until every lowering has written
	// its own initial parameters, then land in the span.
	let weights = node.weights();
	let stored = match graph.bound.as_mut().filter(|_| weights != 0).map(std::collections::VecDeque::pop_front) {
		None => None,
		Some(None) => return Err(RecipeError::new(format!("{} takes {weights} values but the plan names no tensor for it", node.identity(index)))),
		Some(Some(bound)) => {
			require(bound.elements == weights, format!("{} hold {} values; {} takes {weights}", bound.names, bound.elements, node.identity(index)))?;
			match bound.weight {
				BoundWeight::Stored(weight) => {
					node.packed = true;
					Some(weight)
				}
				// A table has no parameter span: its rows decode from the bound
				// bytes, on the device or on the host, so it binds only packed.
				BoundWeight::Values(_) if node.table() => {
					return Err(RecipeError::new(format!("{} bind {}, a table that reads its rows only from a block-quantized layout", bound.names, node.identity(index))));
				}
				BoundWeight::Values(values) => {
					graph.bound_values.push((index, values));
					None
				}
			}
		}
	};
	if stored.is_none() {
		graph.parameters.resize(checked_add(offset, parameters, "model parameters")?, 0.0);
		graph.frozen.resize(graph.parameters.len(), 0);
	}
	graph.nodes.push(node);
	graph.stored.push(stored);
	graph.output = output;
	graph.source = graph.nodes.len() as i32 - 1;
	Ok(())
}
fn push_program(graph: &mut Graph, second: i32, initial: &[f64], program: ScalarProgram) -> Result<()> {
	let (program_offset, program_count) = (graph.programs.len(), program.0.len() / 3);
	graph.programs.extend(program.0);
	let arguments = arguments(0.0, 0.0);
	push_node(graph, Primitive::Elementwise, graph.output, initial.len(), arguments, second)?;
	let node = graph.nodes.last_mut().ok_or_else(|| RecipeError::new("scalar program node is absent"))?;
	graph.parameters[node.offset..node.offset + initial.len()].copy_from_slice(initial);
	node.program_offset = program_offset;
	node.program_count = program_count;
	Ok(())
}
fn push_predictor(graph: &mut Graph, program: PredictorProgram) -> Result<()> {
	let (program_offset, program_count) = (graph.programs.len(), program.code.len() / 2);
	graph.programs.extend(program.code);
	push_node(graph, Primitive::Predictor, Shape { channels: 1, length: 1 }, program.table.len(), arguments(program.locals as f64, program.stack as f64), -2)?;
	let node = graph.nodes.last_mut().ok_or_else(|| RecipeError::new("predictor node is absent"))?;
	node.program_offset = program_offset;
	node.program_count = program_count;
	let (offset, parameters) = (node.offset, node.parameters);
	graph.parameters[offset..offset + parameters].copy_from_slice(&program.table);
	graph.frozen[offset..offset + parameters].fill(1);
	Ok(())
}
fn lower_activation(graph: &mut Graph, activation: Activation, config: Config) -> Result<()> {
	if activation == Activation::Relu {
		let source = graph.source;
		let last = graph.nodes.len() as i32 - 1;
		if let Some(node) = graph.nodes.last_mut()
			&& source == last
			&& node.op == Primitive::Contraction
			&& node.argument[1] == 0.0
		{
			node.argument[1] = 1.0;
			return Ok(());
		}
	}
	let (mut program, x) = (ScalarProgram(Vec::new()), -1.0);
	let (zero, one) = (program.constant(0.0), program.constant(1.0));
	let positive = program.op(ScalarOpcode::Greater, x, zero);
	let constant = |program: &mut ScalarProgram, value| program.constant(value);
	let result = match activation {
		Activation::Cos => program.unary(ScalarOpcode::Cos, x),
		Activation::Exp => program.unary(ScalarOpcode::Exp, x),
		Activation::Log | Activation::Ln => {
			let absolute = program.unary(ScalarOpcode::Absolute, x);
			let shifted = program.op(ScalarOpcode::Add, one, absolute);
			let magnitude = program.unary(ScalarOpcode::Log, shifted);
			let negative = program.op(ScalarOpcode::Subtract, zero, magnitude);
			let signed = program.choose(positive, magnitude, negative);
			if activation == Activation::Log {
				let base = constant(&mut program, std::f64::consts::LN_10);
				program.op(ScalarOpcode::Divide, signed, base)
			} else {
				signed
			}
		}
		Activation::Huber => {
			let threshold = constant(&mut program, config.activation[7]);
			let absolute = program.unary(ScalarOpcode::Absolute, x);
			let large = program.op(ScalarOpcode::Greater, absolute, threshold);
			let square = program.op(ScalarOpcode::Multiply, x, x);
			let half = constant(&mut program, 0.5);
			let small = program.op(ScalarOpcode::Multiply, half, square);
			let half_threshold = program.op(ScalarOpcode::Multiply, half, threshold);
			let excess = program.op(ScalarOpcode::Subtract, absolute, half_threshold);
			let tail = program.op(ScalarOpcode::Multiply, threshold, excess);
			program.choose(large, tail, small)
		}
		Activation::Tan => {
			let sine = program.unary(ScalarOpcode::Sin, x);
			let cosine = program.unary(ScalarOpcode::Cos, x);
			program.op(ScalarOpcode::Divide, sine, cosine)
		}
		Activation::Relu => program.select(positive, x),
		Activation::Leak | Activation::Elu | Activation::Selu | Activation::Prelu => {
			let negative = match activation {
				Activation::Leak => {
					let slope = constant(&mut program, config.activation[0]);
					program.op(ScalarOpcode::Multiply, slope, x)
				}
				Activation::Prelu => {
					let slope = program.op(ScalarOpcode::Parameter, 0.0, 0.0);
					program.op(ScalarOpcode::Multiply, slope, x)
				}
				_ => {
					// choose only selects this branch for x <= 0, but exp still runs on
					// the full range: mask its argument through the same predicate so a
					// large positive x cannot overflow exp in the untaken branch.
					let inverse = program.op(ScalarOpcode::Subtract, one, positive);
					let masked = program.select(inverse, x);
					let exponential = program.unary(ScalarOpcode::Exp, masked);
					let shifted = program.op(ScalarOpcode::Subtract, exponential, one);
					let alpha = constant(&mut program, config.activation[usize::from(activation == Activation::Selu) + 2]);
					program.op(ScalarOpcode::Multiply, alpha, shifted)
				}
			};
			let selected = program.choose(positive, x, negative);
			if activation == Activation::Selu {
				let scale = constant(&mut program, config.activation[4]);
				program.op(ScalarOpcode::Multiply, scale, selected)
			} else {
				selected
			}
		}
		Activation::Sigmoid | Activation::Silu => {
			let half = constant(&mut program, 0.5);
			let half_x = program.op(ScalarOpcode::Multiply, half, x);
			let curved = program.unary(ScalarOpcode::Tanh, half_x);
			let shifted = program.op(ScalarOpcode::Add, curved, one);
			let sigmoid = program.op(ScalarOpcode::Multiply, half, shifted);
			if activation == Activation::Silu { program.op(ScalarOpcode::Multiply, x, sigmoid) } else { sigmoid }
		}
		Activation::Tanh => program.unary(ScalarOpcode::Tanh, x),
		Activation::Gelu => {
			let square = program.op(ScalarOpcode::Multiply, x, x);
			let cube = program.op(ScalarOpcode::Multiply, square, x);
			let cubic = constant(&mut program, config.activation[6]);
			let curved = program.op(ScalarOpcode::Multiply, cubic, cube);
			let sum = program.op(ScalarOpcode::Add, x, curved);
			let scale = constant(&mut program, config.activation[5]);
			let argument = program.op(ScalarOpcode::Multiply, scale, sum);
			let tanh = program.unary(ScalarOpcode::Tanh, argument);
			let shifted = program.op(ScalarOpcode::Add, one, tanh);
			let half = constant(&mut program, 0.5);
			let half_x = program.op(ScalarOpcode::Multiply, half, x);
			program.op(ScalarOpcode::Multiply, half_x, shifted)
		}
		Activation::Linear => unreachable!(),
	};
	let initial = if activation == Activation::Prelu { &config.activation[1..2] } else { &[] };
	debug_assert_eq!(result as usize + 1, program.0.len() / 3);
	push_program(graph, -2, initial, program)
}
/// Whether the contraction the graph is about to push carries a bias row. A
/// trained contraction does when its lowering owns one (`bias`); a gate that
/// never carries one takes the matrix alone. A graph compiled over mapped
/// tensors reads the answer off the views bound to the node: a sum equal to the
/// matrix binds no bias row, a sum one output row larger binds the trailing row
/// as the bias of a node that owns one, and any other sum is rejected here,
/// before anything runs, naming the views, the node and the accepted spans.
fn contraction_bias(graph: &Graph, matrix: usize, width: usize, bias: bool) -> Result<bool> {
	let biased = checked_add(matrix, width, "contraction bias")?;
	match graph.bound.as_ref().and_then(std::collections::VecDeque::front) {
		None => Ok(bias),
		Some(bound) if bound.elements == matrix => Ok(false),
		Some(bound) if bias && bound.elements == biased => Ok(true),
		Some(bound) => Err(RecipeError::new(format!(
			"{} hold {} values; block {} {} node {} contracts {} inputs onto {width} outputs and takes {matrix} values{}",
			bound.names,
			bound.elements,
			graph.block_index,
			graph.block_kind,
			graph.nodes.len(),
			matrix / width,
			if bias { format!(" without a bias row or {biased} with one") } else { String::new() }
		))),
	}
}
/// A contraction's arguments: the convolution kernel, or zero for a projection;
/// the fused ReLU flag, which activation lowering sets; and whether the node
/// carries no bias row, which the kernels read to skip the bias term and which
/// `output_bias_offset` follows.
fn contraction_arguments(kernel: usize, bias: bool) -> [f64; 9] {
	[kernel as f64, 0.0, f64::from(u8::from(!bias)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
/// A projection of the graph output onto `channels`, with a bias row unless the
/// weight it binds to carries none, so its parameter span is then the matrix.
fn lower_project(graph: &mut Graph, channels: usize) -> Result<()> {
	lower_contraction(graph, channels, true)
}
/// A contraction of the graph output onto `channels` whose lowering owns a bias
/// row when `bias` is set; a gate lowered without one spans the matrix alone.
fn lower_contraction(graph: &mut Graph, channels: usize, bias: bool) -> Result<()> {
	require(channels != 0, "layer width must be positive")?;
	let matrix = checked_mul(graph.output.channels, channels, "projection matrix")?;
	let bias = contraction_bias(graph, matrix, channels, bias)?;
	let parameters = if bias { checked_add(matrix, channels, "projection bias")? } else { matrix };
	let output = Shape { channels, length: graph.output.length };
	push_node(graph, Primitive::Contraction, output, parameters, contraction_arguments(0, bias), -2)
}
fn lower_conv(graph: &mut Graph, filters: usize, kernel: usize) -> Result<()> {
	require(filters != 0 && kernel != 0, "convolution dimensions must be positive")?;
	require(kernel <= graph.output.length, "convolution kernel exceeds sequence length")?;
	let matrix = checked_mul(filters, checked_mul(graph.output.channels, kernel, "convolution window")?, "conv matrix")?;
	let bias = contraction_bias(graph, matrix, filters, true)?;
	let parameters = if bias { checked_add(matrix, filters, "conv bias")? } else { matrix };
	let output = Shape { channels: filters, length: graph.output.length - kernel + 1 };
	push_node(graph, Primitive::Contraction, output, parameters, contraction_arguments(kernel, bias), -2)
}
/// The bias row of the last contraction that carries one.
fn output_bias_offset(graph: &Graph) -> Option<usize> {
	graph.nodes.iter().rev().find(|node| node.op == Primitive::Contraction && node.argument[2] == 0.0).map(|node| node.offset + node.parameters - node.output.channels)
}
fn lower_pool(graph: &mut Graph, size: usize) -> Result<()> {
	require(size != 0, "pool window must be positive")?;
	let output = Shape { channels: graph.output.channels, length: graph.output.length.div_ceil(size) };
	push_node(graph, Primitive::Pool, output, 0, arguments(size as f64, 0.0), -2)
}
fn lower_embed(graph: &mut Graph, vocabulary: usize, width: usize) -> Result<()> {
	require(vocabulary != 0 && width != 0, "embedding dimensions must be positive")?;
	require(graph.nodes.is_empty(), "embedding must be the first block: it reads token ids from the model input")?;
	checked_mul(vocabulary, width, "embedding table")?;
	let output = Shape { channels: width, length: graph.output.elements() };
	// The table is the packed context the gather reads, so the node owns no
	// trainable parameters and no run holds a second copy of it.
	push_node(graph, Primitive::Gather, output, 0, arguments(vocabulary as f64, width as f64), -2)
}
/// A causal depthwise convolution keeps the shape: every channel mixes its own
/// last `kernel` positions with one tap each, left-padded with zeros. The taps
/// sit `dilation` positions apart, so position `t` reads `t - j * dilation`.
fn lower_dconv(graph: &mut Graph, kernel: usize, dilation: usize) -> Result<()> {
	require(kernel != 0 && dilation != 0, "depthwise convolution kernel and dilation must be positive")?;
	push_node(graph, Primitive::Dconv, graph.output, checked_mul(graph.output.channels, kernel, "depthwise taps")?, arguments(kernel as f64, dilation as f64), -2)
}
/// A per-layer embedding. The rows one token addresses are gathered on the host
/// and staged for the device, which projects them to a key and a value without
/// a bias. The key and the stream take grouped root-mean-square norms over one
/// lane with a scale over every lane; their per-lane dot product over the root of
/// the lane width, through a signed square root and a sigmoid, gates the value
/// broadcast over the lanes. A third grouped norm, a causal depthwise convolution
/// dilated by the n-gram size and a SiLU form the second term, and both add into
/// the stream, which keeps its width.
fn lower_ple(graph: &mut Graph, ple: &PleBlock, config: Config) -> Result<()> {
	let (stream, shape) = (graph.source, graph.output);
	require(stream >= 0, "a per-layer embedding follows the block whose stream it adds into")?;
	require(ple.heads != 0 && ple.width != 0 && ple.kernel != 0 && ple.dilation != 0, "per-layer embedding dimensions must be positive")?;
	ple.hash.validate()?;
	require(ple.hash.heads() == ple.heads, format!("per-layer embedding names {} heads, its hash addresses {}", ple.heads, ple.hash.heads()))?;
	require(ple.hash.rows() <= ple.rows, format!("per-layer embedding table holds {} rows, its hash reaches row {}", ple.rows, ple.hash.rows()))?;
	let lanes = graph.lanes.max(1);
	require(shape.channels != 0 && shape.channels % lanes == 0, format!("per-layer embedding stream of {} does not split into {lanes} lanes", shape.channels))?;
	let channels = shape.channels / lanes;
	let gathered = Shape { channels: checked_mul(ple.heads, ple.width, "per-layer embedding width")?, length: shape.length };
	// The lookup reads the ids on the host, so it names no device source; the
	// hash rides beside the node as program words.
	let (program_offset, words) = (graph.programs.len(), ple.hash.words());
	let program_count = words.len().div_ceil(3);
	graph.programs.extend(words);
	graph.programs.resize(program_offset + program_count * 3, 0.0);
	reset(graph, -2, Shape { channels: 1, length: shape.length });
	let argument = [ple.heads as f64, ple.width as f64, ple.rows as f64, ple.kernel as f64, ple.dilation as f64, 0.0, 0.0, 0.0, 0.0];
	push_node(graph, Primitive::Lookup, gathered, 0, argument, -2)?;
	if let Some(node) = graph.nodes.last_mut() {
		(node.program_offset, node.program_count) = (program_offset, program_count);
	}
	let rows = graph.source;
	lower_project(graph, shape.channels)?;
	lower_normalize(graph, BlockNormalization::Rms, channels, shape.channels)?;
	let key = graph.source;
	reset(graph, stream, shape);
	lower_normalize(graph, BlockNormalization::Rms, channels, shape.channels)?;
	let query = graph.source;
	binary(graph, key, query, shape, ScalarOpcode::Multiply)?;
	push_node(graph, Primitive::Fold, Shape { channels: lanes, length: shape.length }, 0, arguments(channels as f64, 0.0), -2)?;
	// gate = sigmoid(sign(s) * sqrt(max(|s|, 1e-6))) for s the scaled dot product.
	let (mut program, x) = (ScalarProgram(Vec::new()), -1.0);
	let scale = program.constant(1.0 / (channels as f64).sqrt());
	let s = program.op(ScalarOpcode::Multiply, x, scale);
	let (zero, one, half) = (program.constant(0.0), program.constant(1.0), program.constant(0.5));
	let magnitude = program.unary(ScalarOpcode::Absolute, s);
	let floor = program.constant(1e-6);
	let above = program.op(ScalarOpcode::Greater, magnitude, floor);
	let clamped = program.choose(above, magnitude, floor);
	let log = program.unary(ScalarOpcode::Log, clamped);
	let half_log = program.op(ScalarOpcode::Multiply, half, log);
	let root = program.unary(ScalarOpcode::Exp, half_log);
	let positive = program.op(ScalarOpcode::Greater, s, zero);
	let negative = program.op(ScalarOpcode::Greater, zero, s);
	let sign = program.op(ScalarOpcode::Subtract, positive, negative);
	let signed = program.op(ScalarOpcode::Multiply, sign, root);
	let negated = program.op(ScalarOpcode::Subtract, zero, signed);
	let exponential = program.unary(ScalarOpcode::Exp, negated);
	let denominator = program.op(ScalarOpcode::Add, one, exponential);
	program.op(ScalarOpcode::Divide, one, denominator);
	push_program(graph, -2, &[], program)?;
	let gate = graph.source;
	reset(graph, rows, gathered);
	lower_project(graph, channels)?;
	push_node(graph, Primitive::Outer, shape, 0, arguments(lanes as f64, 0.0), gate)?;
	let gated = graph.source;
	lower_normalize(graph, BlockNormalization::Rms, channels, shape.channels)?;
	lower_dconv(graph, ple.kernel, ple.dilation)?;
	lower_activation(graph, Activation::Silu, config)?;
	let convolved = graph.source;
	let added = binary(graph, gated, convolved, shape, ScalarOpcode::Add)?;
	binary(graph, stream, added, shape, ScalarOpcode::Add).map(drop)
}
/// A gated delta rule carries one `width` by `width` state per head. One projection
/// feeds the causal depthwise convolution over the concatenated query, key and value
/// stream, a second carries the decay and write gate pre-activations, and the
/// recurrence reads one value per head. The queries and keys take a per-head unit
/// length, the output a per-head root mean square and the gate built from a third
/// projection, and the output projection closes the block.
fn lower_delta(graph: &mut Graph, delta: DeltaBlock, config: Config) -> Result<()> {
	let (source, input) = (graph.source, graph.output);
	let (heads, kernel) = (delta.heads, delta.kernel);
	let (key_heads, key_width, value_width, output) = delta.extent(input.channels)?;
	// Queries and keys span the key heads; values, the recurrence output and the
	// gate span the value heads. Both are independent of the stream the block sits on.
	let keys = checked_mul(key_heads, key_width, "delta key width")?;
	let inner = checked_mul(heads, value_width, "delta value width")?;
	let recurrent = Shape { channels: inner, length: input.length };
	let chunk = natural("delta chunk", env!("RECIPE_DELTA_CHUNK"))?;
	lower_project(graph, checked_mul(2, heads, "delta gate width")?)?;
	let gates = graph.source;
	reset(graph, source, input);
	lower_project(graph, checked_add(checked_mul(2, keys, "delta query and key width")?, inner, "delta projection width")?)?;
	lower_dconv(graph, kernel, 1)?;
	// The projection lays the queries and keys out ahead of the values, so the
	// normalized span stops at the value plane and each key head owns one group.
	lower_normalize(graph, BlockNormalization::L2, key_width, checked_mul(2, keys, "delta query and key span")?)?;
	let argument = [heads as f64, value_width as f64, chunk as f64, key_heads as f64, key_width as f64, 0.0, 0.0, 0.0, 0.0];
	push_node(graph, Primitive::Delta, recurrent, heads, argument, gates)?;
	lower_normalize(graph, BlockNormalization::Rms, value_width, inner)?;
	let normalized = graph.source;
	reset(graph, source, input);
	lower_project(graph, inner)?;
	let (gate, shape) = activation(graph, graph.source, graph.output, Activation::Sigmoid, config)?;
	binary(graph, normalized, gate, shape, ScalarOpcode::Multiply)?;
	lower_project(graph, output)
}
/// Lowers one attention block into its projection, the optional query and key
/// normalization and rotary nodes, the attention node and the output
/// projection. The projection carries the query, key and value planes, then
/// the indexer planes, then the gate plane.
fn lower_attention(graph: &mut Graph, attention: AttentionBlock, qk: Option<BlockNormalization>) -> Result<()> {
	let AttentionBlock { heads, kv, rope, index, gate, .. } = attention;
	let (input, source) = (graph.output, graph.source);
	// The heads attend at their own width, so the inner width the block spans is
	// independent of the stream the output projection returns to.
	let (width, inner) = attention.extent(input.channels)?;
	require(kv != 0 && kv <= heads && heads % kv == 0, "attention key-value head partition is invalid")?;
	let pairs = checked_mul(width, checked_add(heads, checked_mul(2, kv, "attention key-value planes")?, "attention projection heads")?, "attention QKV projection width")?;
	let gated = if gate { inner } else { 0 };
	lower_project(graph, checked_add(pairs, gated, "attention projection width")?)?;
	if let Some(normalization) = qk {
		// The projection lays the queries and keys out ahead of the values, so the
		// normalized span stops at the value plane and each head owns one group.
		lower_normalize(graph, normalization, width, checked_mul(width, checked_add(heads, kv, "attention query and key heads")?, "attention query and key span")?)?;
	}
	if let Some((dims, base)) = rope {
		require(dims != 0 && dims % 2 == 0 && dims <= width, "rotary dimensions must be even and at most the head width")?;
		require(f64::from_bits(base) > 1.0, "rotary base must exceed one")?;
		let rotated = checked_mul(width, checked_add(heads, kv, "rotary head partition")?, "rotary width")?;
		push_node(graph, Primitive::Rope, graph.output, 0, [dims as f64, f64::from_bits(base), width as f64, rotated as f64, 0.0, 0.0, 0.0, 0.0, 0.0], -2)?;
	}
	let (main, main_shape) = (graph.source, graph.output);
	// The indexer is its own projection of the block input, so a checkpoint binds
	// its indexer tensors as their own node in whatever layout they hold, and the
	// normalization and rotary the model trained sit on that projection as graph
	// nodes: every indexer head normalizes under `.score`, or to unit length
	// without it, before the attention node scores plain dot products.
	let indexer = index.unwrap_or(Indexer::NONE);
	let mut side = -2;
	if let Some(index) = index {
		require(index.heads != 0 && index.width != 0, "indexer projection must be positive")?;
		require(index.block != 0 && index.admitted() != 0, "indexer selection must be positive")?;
		reset(graph, source, input);
		lower_project(graph, checked_mul(index.width, checked_add(index.heads, 1, "indexer projection heads")?, "indexer projection width")?)?;
		let (normalization, dims) = index.score.unwrap_or((BlockNormalization::L2, 0));
		let planes = graph.output.channels;
		lower_normalize(graph, normalization, index.width, planes)?;
		if dims != 0 {
			let (_, base) = rope.ok_or_else(|| RecipeError::new("indexer rotary dimensions require the block's rope base"))?;
			require(dims % 2 == 0 && dims <= index.width, "indexer rotary dimensions must be even and at most the indexer width")?;
			push_node(graph, Primitive::Rope, graph.output, 0, [dims as f64, f64::from_bits(base), index.width as f64, planes as f64, 0.0, 0.0, 0.0, 0.0, 0.0], -2)?;
		}
		side = graph.source;
		reset(graph, main, main_shape);
	}
	let epsilon = graph.epsilon;
	let argument = [heads as f64, kv as f64, f64::from(u8::from(gate)), indexer.block as f64, indexer.admitted() as f64, indexer.heads as f64, indexer.width as f64, epsilon, 0.0];
	push_node(graph, Primitive::Attention, Shape { channels: inner, length: input.length }, 0, argument, side)?;
	lower_project(graph, input.channels)
}
/// Pushes a normalization over the graph output. A per-row mode splits the leading
/// `span` channels into groups of `width`; the rest pass through untouched.
fn lower_normalize(graph: &mut Graph, normalization: BlockNormalization, width: usize, span: usize) -> Result<()> {
	let epsilon = graph.epsilon;
	// RMS carries one trainable scale per normalized channel, starting at identity.
	let parameters = if normalization == BlockNormalization::Rms { span } else { 0 };
	let output = graph.output;
	push_node(graph, Primitive::Normalize, output, parameters, [normalization.mode(), epsilon, width as f64, span as f64, 0.0, 0.0, 0.0, 0.0, 0.0], -2)?;
	let offset = graph.parameters.len() - parameters;
	graph.parameters[offset..].fill(1.0);
	Ok(())
}
/// Key blocks the indexer scores for a sequence of `length` positions.
fn attention_blocks(node: &Node) -> usize {
	let block = node.argument[3] as usize;
	if block == 0 { 0 } else { node.output.length.div_ceil(block) }
}
fn reset(graph: &mut Graph, source: i32, shape: Shape) {
	graph.source = source;
	graph.output = shape;
}
fn program(graph: &mut Graph, first: i32, second: i32, shape: Shape, initial: &[f64], program: ScalarProgram) -> Result<i32> {
	reset(graph, first, shape);
	push_program(graph, second, initial, program)?;
	Ok(graph.source)
}
fn binary(graph: &mut Graph, first: i32, second: i32, shape: Shape, opcode: ScalarOpcode) -> Result<i32> {
	let mut scalar = ScalarProgram(Vec::new());
	scalar.op(opcode, -1.0, -2.0);
	program(graph, first, second, shape, &[], scalar)
}
/// Apply `value` to `source`, or pass it through when linear, and name the result.
fn activation(graph: &mut Graph, source: i32, shape: Shape, value: Activation, config: Config) -> Result<(i32, Shape)> {
	reset(graph, source, shape);
	if value != Activation::Linear {
		lower_activation(graph, value, config)?;
	}
	Ok((graph.source, graph.output))
}
/// Lower one gated feed-forward: `down(act(gate(x)) * up(x))`.
fn lower_gated(graph: &mut Graph, gate: i32, up: i32, shape: Shape, activation: Activation, config: Config) -> Result<i32> {
	reset(graph, gate, shape);
	if activation != Activation::Linear {
		lower_activation(graph, activation, config)?;
	}
	let activated = graph.source;
	binary(graph, activated, up, shape, ScalarOpcode::Multiply)
}
/// Runs the gated feed-forward of the experts `routing` names over `input`, the
/// graph output at `source`. `routing` is a `[experts, length]` plane of per-position
/// weights: a nonzero entry takes that expert for the position, and the position's
/// output is the sum of its `top_k` experts under those weights.
fn lower_experts(graph: &mut Graph, source: i32, input: Shape, routing: i32, experts: usize, top_k: usize, hidden: usize, activation: Activation, config: Config) -> Result<()> {
	let routed = Shape { channels: checked_mul(top_k, hidden, "moe routed width")?, length: input.length };
	let table = checked_mul(experts, checked_mul(hidden, input.channels, "moe expert matrix")?, "moe expert table")?;
	let dispatch = [experts as f64, top_k as f64, hidden as f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
	reset(graph, source, input);
	push_node(graph, Primitive::ExpertIn, routed, table, dispatch, routing)?;
	let gate = graph.source;
	reset(graph, source, input);
	push_node(graph, Primitive::ExpertIn, routed, table, dispatch, routing)?;
	let up = graph.source;
	let product = lower_gated(graph, gate, up, routed, activation, config)?;
	reset(graph, product, routed);
	push_node(graph, Primitive::ExpertOut, input, table, dispatch, routing)
}
/// One gated feed-forward over the block input: the gate and up projections,
/// their activated product, and the down projection back to the input width.
fn lower_glu(graph: &mut Graph, hidden: usize, activation: Activation, config: Config) -> Result<()> {
	require(hidden != 0, "gated feed-forward width must be positive")?;
	let (source, input) = (graph.source, graph.output);
	let wide = Shape { channels: hidden, length: input.length };
	lower_project(graph, hidden)?;
	let gate = graph.source;
	reset(graph, source, input);
	lower_project(graph, hidden)?;
	let up = graph.source;
	let product = lower_gated(graph, gate, up, wide, activation, config)?;
	reset(graph, product, wide);
	lower_project(graph, input.channels)
}
fn lower_moe(graph: &mut Graph, experts: usize, top_k: usize, hidden: usize, activation: Activation, scoring: Scoring, renormalize: bool, shared: bool, config: Config) -> Result<()> {
	require(experts != 0, "moe requires an expert")?;
	require(top_k != 0 && top_k <= experts, "moe top-k is invalid")?;
	require(hidden != 0, "moe expert width must be positive")?;
	let (source, input) = (graph.source, graph.output);
	// One router scores every expert per position. The top-k weights name the
	// experts whose gated feed-forward runs, so a position costs top-k of them.
	lower_project(graph, experts)?;
	push_node(graph, Primitive::TopK, graph.output, 0, [top_k as f64, f64::from(scoring as u8), f64::from(u8::from(renormalize)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], -2)?;
	let routing = graph.source;
	lower_experts(graph, source, input, routing, experts, top_k, hidden, activation, config)?;
	if !shared {
		return Ok(());
	}
	// The shared expert is one more expert that every position takes. Its routing
	// weight is the sigmoid of a `[width]` gate over the position, with no bias,
	// so the dispatch that runs the routed experts runs it under that per-position
	// value and its gradient reaches the gate and the input through the same adjoints.
	let dispatched = graph.source;
	reset(graph, source, input);
	// The gate is the `[width]` vector alone, trained or bound, so a view of it
	// must hold exactly that many values.
	push_node(graph, Primitive::Contraction, Shape { channels: 1, length: input.length }, input.channels, contraction_arguments(0, false), -2)?;
	lower_activation(graph, Activation::Sigmoid, config)?;
	let gate = graph.source;
	lower_experts(graph, source, input, gate, 1, 1, hidden, activation, config)?;
	let gated = graph.source;
	binary(graph, dispatched, gated, input, ScalarOpcode::Add)?;
	Ok(())
}
fn lower_scan(graph: &mut Graph, channels: usize, gates: usize) -> Result<()> {
	require(channels != 0, "recurrent width must be positive")?;
	let (input, state) = (checked_mul(graph.output.channels, channels, "scan input matrix")?, checked_mul(channels, channels, "scan state matrix")?);
	let stride = checked_add(checked_add(input, state, "scan gate")?, channels, "scan bias")?;
	let output = Shape { channels, length: graph.output.length };
	push_node(graph, Primitive::Scan, output, checked_mul(gates, stride, "scan parameters")?, arguments(gates as f64, 0.0), -2)
}
fn lower_residual(graph: &mut Graph, parts: &[Residual], skip: i32, config: Config) -> Result<()> {
	let shape = graph.output;
	require(!parts.is_empty(), "residual branch must contain an operation")?;
	for part in parts {
		match part {
			Residual::Layer(width) => lower_project(graph, *width)?,
			Residual::Conv(filters, kernel) => lower_conv(graph, *filters, *kernel)?,
			Residual::Activation(activation) => lower_activation(graph, *activation, config)?,
		}
	}
	require(graph.output.channels == shape.channels && graph.output.length == shape.length, "residual shape mismatch")?;
	let mut program = ScalarProgram(Vec::new());
	program.op(ScalarOpcode::Add, -1.0, -2.0);
	push_program(graph, skip, &[], program)
}
fn lower_hyper(graph: &mut Graph, lanes: usize, rank: usize, blocks: &[Block], total: usize, data: &Prepared, targets: &[f64], rows: usize, gpu: &'static Gpu, config: Config) -> Result<()> {
	require(lanes != 0 && !blocks.is_empty(), "hyper-connections need at least one lane and one block")?;
	if graph.lanes == 0 {
		let shape = graph.output;
		push_node(graph, Primitive::Expand, Shape { channels: checked_mul(shape.channels, lanes, "hyper-connection stream")?, length: shape.length }, 0, arguments(lanes as f64, 0.0), -2)?;
		graph.lanes = lanes;
	}
	require(graph.lanes == lanes, format!("hyper-connections with {lanes} lanes follow a stream of {}", graph.lanes))?;
	graph.rank = rank;
	let (stream, shape) = (graph.source, graph.output);
	let width = shape.channels / lanes;
	let (source, read, write) = lower_gates(graph, lanes, rank, true, config)?;
	reset(graph, source, shape);
	push_node(graph, Primitive::Read, Shape { channels: width, length: shape.length }, 0, arguments(lanes as f64, 0.0), read)?;
	let (outer_frozen, outer_packed, outer_kind) = (graph.block_frozen, graph.block_packed, graph.block_kind);
	graph.lanes = 0;
	for block in blocks {
		graph.block_frozen = block.frozen;
		graph.block_packed = block.packed;
		graph.block_kind = block.operation.name();
		lower_block(graph, block, total, data, targets, rows, gpu, config)?;
	}
	if graph.lanes != 0 {
		lower_collapse(graph, config)?;
	}
	(graph.block_frozen, graph.block_packed, graph.block_kind, graph.lanes) = (outer_frozen, outer_packed, outer_kind, lanes);
	require(graph.output.channels == width && graph.output.length == shape.length, "hyper-connection branch shape mismatch")?;
	push_node(graph, Primitive::Outer, shape, 0, arguments(lanes as f64, 0.0), write)?;
	let mut program = ScalarProgram(Vec::new());
	program.op(ScalarOpcode::Add, -1.0, -2.0);
	push_program(graph, stream, &[], program)
}
/// The mixer gates from the stream: per-lane RMS statistics under one trainable
/// scale over the whole stream give `xn`; the read gate is
/// `sigmoid(W_up · silu(W_down · xn / lanes))` over the stream, and the write
/// gate is `2 sigmoid(W_inject · xn / lanes)` per lane, so a zero injection is
/// the plain residual. No projection carries a bias. Returns the node the read
/// consumes, the read gate, and the write gate. With `rank` zero no node is
/// added, every gate is one, and the read takes the raw stream.
fn lower_gates(graph: &mut Graph, lanes: usize, rank: usize, write: bool, config: Config) -> Result<(i32, i32, i32)> {
	let (stream, shape) = (graph.source, graph.output);
	if rank == 0 {
		return Ok((stream, -2, -2));
	}
	lower_normalize(graph, BlockNormalization::Rms, shape.channels / lanes, shape.channels)?;
	let normalized = graph.source;
	lower_contraction(graph, rank, false)?;
	lower_scale(graph, 1.0 / lanes as f64)?;
	lower_activation(graph, Activation::Silu, config)?;
	lower_contraction(graph, shape.channels, false)?;
	lower_activation(graph, Activation::Sigmoid, config)?;
	let read = graph.source;
	if !write {
		return Ok((normalized, read, -2));
	}
	reset(graph, normalized, shape);
	lower_contraction(graph, lanes, false)?;
	lower_scale(graph, 1.0 / lanes as f64)?;
	lower_activation(graph, Activation::Sigmoid, config)?;
	lower_scale(graph, 2.0)?;
	Ok((normalized, read, graph.source))
}
/// Multiplies the graph output by `factor`.
fn lower_scale(graph: &mut Graph, factor: f64) -> Result<()> {
	let mut program = ScalarProgram(Vec::new());
	let factor = program.constant(factor);
	program.op(ScalarOpcode::Multiply, -1.0, factor);
	push_program(graph, -2, &[], program)
}
/// The head read: the stream collapses to the mean of its lanes under its own
/// read gate.
fn lower_collapse(graph: &mut Graph, config: Config) -> Result<()> {
	let (lanes, rank, shape) = (graph.lanes, graph.rank, graph.output);
	let (source, read, _) = lower_gates(graph, lanes, rank, false, config)?;
	reset(graph, source, shape);
	push_node(graph, Primitive::Read, Shape { channels: shape.channels / lanes, length: shape.length }, 0, arguments(lanes as f64, 0.0), read)?;
	graph.lanes = 0;
	Ok(())
}
fn lower_estimator(graph: &mut Graph, estimator: &Estimator, data: &Prepared, targets: &[f64], rows: usize, gpu: &'static Gpu, config: Config) -> Result<()> {
	let (source, input) = (graph.source, graph.output);
	let restored = data.fitted.get(graph.nodes.iter().filter(|node| node.op == Primitive::Predictor).count()).cloned();
	let (predictor, surrogate) = if let Some(program) = restored {
		let blank = Prepared {
			samples: vec![0.0; input.elements()],
			targets: vec![0.0],
			target_width: 1,
			rows: 1,
			source_rows: 1,
			features: input.elements(),
			schema: DataSchema::default(),
			sequence: None,
			target_categorical: false,
			norm_mean: Vec::new(),
			norm_scale: Vec::new(),
			identities: Vec::new(),
			fitted: Vec::new(),
			bound: None,
		};
		let mut surrogate = compile(&surrogate_model(config.surrogate_width), &blank, &blank.targets, 1, gpu, config, false)?;
		surrogate.frozen.fill(1);
		(program, surrogate)
	} else {
		(estimator.validate)(estimator.param, rows)?;
		let inputs = graph_inputs(graph, &data.samples, &data.targets, rows, gpu, config.precision)?;
		let prepared = Prepared {
			samples: inputs.clone(),
			targets: targets[..rows].to_vec(),
			target_width: 1,
			rows,
			source_rows: rows,
			features: input.elements(),
			schema: DataSchema::default(),
			sequence: None,
			target_categorical: data.target_categorical,
			norm_mean: Vec::new(),
			norm_scale: Vec::new(),
			identities: Vec::new(),
			fitted: Vec::new(),
			bound: None,
		};
		let fitted = estimator.fit(&prepared, rows, config)?;
		let targets = predict_rows(&fitted, &inputs, input.elements())?;
		(fitted.program, fit_surrogate(input, &inputs, &targets, config.surrogate_width, gpu, config)?)
	};
	reset(graph, source, input);
	push_predictor(graph, predictor)?;
	let real = graph.source;
	reset(graph, source, input);
	let surrogate = append_graph(graph, surrogate)?;
	let mut rat = ScalarProgram(Vec::new());
	rat.op(ScalarOpcode::StraightThrough, -1.0, -2.0);
	program(graph, real, surrogate, Shape { channels: 1, length: 1 }, &[], rat).map(drop)
}
fn next_weight(state: &mut u64, scale: f64) -> f64 {
	*state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
	((*state >> 11) as f64 / ((1_u64 << 53) as f64) * 2.0 - 1.0) * scale
}
fn initialize_graph(graph: &mut Graph, config: Config) {
	let mut state = config.random_seed as u64;
	for (position, node) in graph.nodes.iter().enumerate() {
		// Scalar programs and normalizations set their own initial parameters.
		if matches!(node.op, Primitive::Elementwise | Primitive::Normalize) {
			continue;
		}
		// An embedding row feeds the next block with its width, not with the
		// vocabulary. An expert table holds every expert slice, but one output sums
		// over a single slice: the fan-in is that slice, not the whole table.
		let span = match node.op {
			Primitive::Gather => node.output.channels,
			Primitive::ExpertIn => node.input.channels,
			Primitive::ExpertOut => node.argument[2] as usize,
			_ => node.parameters / node.output.channels.max(1),
		};
		let fan_in = span.max(1) as f64;
		let scale = config.initial / fan_in.sqrt();
		// A table is the node's context rather than a trainable span, so it is
		// drawn once into the node's tensor and no optimizer step can write it back.
		if node.table() {
			if graph.stored[position].is_none() {
				let arithmetic = (0..node.weights()).map(|_| next_weight(&mut state, scale)).collect::<Vec<_>>();
				graph.stored[position] = Some(StoredWeight { format: StorageFormat(0), count: arithmetic.len(), bytes: Vec::new().into(), codebook: Vec::new(), arithmetic });
			}
			continue;
		}
		for index in node.offset..node.offset + node.parameters {
			if graph.frozen[index] == 0 {
				graph.parameters[index] = next_weight(&mut state, scale);
			}
		}
		if node.op == Primitive::Contraction && node.argument[2] == 0.0 {
			graph.parameters[node.offset + node.parameters - node.output.channels..node.offset + node.parameters].fill(0.0);
		}
		// Depthwise taps open at the identity: the current position keeps its value
		// and the earlier taps start at zero, so the stream the convolution mixes
		// reaches the next node with the magnitude it arrived with.
		if node.op == Primitive::Dconv {
			let kernel = node.argument[0] as usize;
			graph.parameters[node.offset..node.offset + node.parameters].fill(0.0);
			for channel in 0..node.output.channels {
				graph.parameters[node.offset + channel * kernel + kernel - 1] = 1.0;
			}
		}
		if node.op == Primitive::Scan {
			let channels = node.output.channels;
			let input_matrix = node.input.channels * channels;
			let state_matrix = channels * channels;
			let stride = input_matrix + state_matrix + channels;
			for gate in 0..node.argument[0] as usize {
				graph.parameters[node.offset + gate * stride + input_matrix + state_matrix..node.offset + (gate + 1) * stride].fill(0.0);
			}
			if node.argument[0] as usize == 4 {
				graph.parameters[node.offset + stride + input_matrix + state_matrix..node.offset + stride * 2].fill(1.0);
			}
		}
	}
}
fn arguments(first: f64, second: f64) -> [f64; 9] {
	[first, second, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
fn checked_add(left: usize, right: usize, role: &str) -> Result<usize> {
	left.checked_add(right).ok_or_else(|| RecipeError::new(format!("{role} overflows")))
}
fn checked_mul(left: usize, right: usize, role: &str) -> Result<usize> {
	left.checked_mul(right).ok_or_else(|| RecipeError::new(format!("{role} overflows")))
}
fn require(condition: bool, message: impl Into<String>) -> Result<()> {
	condition.then_some(()).ok_or_else(|| RecipeError::new(message))
}
fn logistic(value: f64) -> f64 {
	1.0 / (1.0 + (-value).exp())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
struct Tile {
	m: u32,
	n: u32,
	k: u32,
}
#[derive(Clone)]
struct NativeSchedule {
	matrix: bool,
	block: u32,
	tile: Tile,
	register_m: u32,
	register_n: u32,
	register_count: u32,
	fragment_k: u32,
	chunk_k: u32,
	chunk_values: u32,
	chunk_bias_values: u32,
	scratch_base: i32,
	shared_values: u32,
	contractions: Vec<Option<NativeContractionTiles>>,
	attention: Vec<Option<Tile>>,
}
#[derive(Clone, Copy, Debug)]
struct NativeContractionTiles {
	forward: Tile,
	gradient: Tile,
	previous: Tile,
	gradient_shape: Tile,
	parameters: usize,
}
#[derive(Clone, Copy)]
struct NativeContractionShapes {
	forward: Tile,
	gradient: Tile,
	previous: Tile,
	parameters: usize,
}
/// The permitted placement policy: `false` trains on the local device, `true`
/// forces every selected device, and `"auto"` takes the candidate route with the lowest predicted complete epoch.
#[derive(Clone, Copy, PartialEq)]
enum MultiDevice {
	Local,
	Forced,
	Auto,
}
#[derive(Clone, Copy)]
struct Config {
	multi_device: MultiDevice,
	kmeans_iterations: usize,
	svm_iterations: usize,
	svm_rate: f64,
	svm_regularization: f64,
	svm_epsilon: f64,
	tree_depth: usize,
	tree_min_rows: usize,
	forest_feature_fraction: f64,
	bayes_prior_precision: f64,
	bayes_noise_variance: f64,
	bayes_variance_epsilon: f64,
	boost_iterations: usize,
	boost_rate: f64,
	catboost_prior: f64,
	catboost_borders: usize,
	xgboost_regularization: f64,
	xgboost_min_gain: f64,
	lightgbm_bins: usize,
	lightgbm_leaves: usize,
	quantization_block: usize,
	surrogate_epochs: usize,
	surrogate_width: usize,
	surrogate_rate: f64,
	initial: f64,
	beta1: f64,
	beta2: f64,
	epsilon: f64,
	decay: f64,
	progress_refresh_hz: usize,
	random_seed: usize,
	activation: [f64; 8],
	precision: Compute,
}
impl Config {
	fn load() -> Result<Self> {
		Ok(Self {
			multi_device: match env!("RECIPE_MULTI_DEVICE") {
				"false" => MultiDevice::Local,
				"true" => MultiDevice::Forced,
				"auto" => MultiDevice::Auto,
				value => return Err(RecipeError::new(format!("multi-device must be false, true, or \"auto\", not {value:?}"))),
			},
			kmeans_iterations: natural("kmeans iterations", env!("RECIPE_KMEANS_ITERATIONS"))?,
			svm_iterations: natural("SVM iterations", env!("RECIPE_SVM_ITERATIONS"))?,
			svm_rate: number("SVM learning rate", env!("RECIPE_SVM_LEARNING_RATE"))?,
			svm_regularization: number("SVM regularization", env!("RECIPE_SVM_REGULARIZATION"))?,
			svm_epsilon: number("SVM epsilon", env!("RECIPE_SVM_EPSILON"))?,
			tree_depth: natural("tree depth", env!("RECIPE_TREE_DEPTH"))?,
			tree_min_rows: natural("tree minimum rows", env!("RECIPE_TREE_MIN_ROWS"))?,
			forest_feature_fraction: fraction("forest feature fraction", env!("RECIPE_FOREST_FEATURE_FRACTION"))?,
			bayes_prior_precision: number("Bayes prior precision", env!("RECIPE_BAYES_PRIOR_PRECISION"))?,
			bayes_noise_variance: number("Bayes noise variance", env!("RECIPE_BAYES_NOISE_VARIANCE"))?,
			bayes_variance_epsilon: number("Bayes variance epsilon", env!("RECIPE_BAYES_VARIANCE_EPSILON"))?,
			boost_iterations: natural("boost iterations", env!("RECIPE_BOOST_ITERATIONS"))?,
			boost_rate: fraction("boost learning rate", env!("RECIPE_BOOST_LEARNING_RATE"))?,
			catboost_prior: number("CatBoost ordered prior", env!("RECIPE_CATBOOST_ORDERED_PRIOR"))?,
			catboost_borders: natural("CatBoost border count", env!("RECIPE_CATBOOST_BORDER_COUNT"))?,
			xgboost_regularization: number("XGBoost L2 regularization", env!("RECIPE_XGBOOST_L2_REGULARIZATION"))?,
			xgboost_min_gain: number("XGBoost minimum gain", env!("RECIPE_XGBOOST_MINIMUM_GAIN"))?,
			lightgbm_bins: natural("LightGBM histogram bins", env!("RECIPE_LIGHTGBM_HISTOGRAM_BINS"))?,
			lightgbm_leaves: natural("LightGBM leaves", env!("RECIPE_LIGHTGBM_LEAVES"))?,
			quantization_block: natural("quantization block weights", env!("RECIPE_QUANTIZATION_BLOCK_WEIGHTS"))?,
			surrogate_epochs: natural("surrogate epochs", env!("RECIPE_SURROGATE_EPOCHS"))?,
			surrogate_width: natural("surrogate width", env!("RECIPE_SURROGATE_WIDTH"))?,
			surrogate_rate: number("surrogate rate", env!("RECIPE_SURROGATE_RATE"))?,
			progress_refresh_hz: natural("progress refresh Hz", env!("RECIPE_PROGRESS_REFRESH_HZ"))?,
			random_seed: natural("random seed", env!("RECIPE_RANDOM_SEED"))?,
			initial: number("initial weight", env!("RECIPE_TRAIN_INITIAL_WEIGHT"))?,
			beta1: number("AdamW beta1", env!("RECIPE_ADAMW_BETA1"))?,
			beta2: number("AdamW beta2", env!("RECIPE_ADAMW_BETA2"))?,
			epsilon: number("AdamW epsilon", env!("RECIPE_ADAMW_EPSILON"))?,
			decay: number("AdamW weight decay", env!("RECIPE_ADAMW_WEIGHT_DECAY"))?,
			activation: [
				number("leak slope", env!("RECIPE_LEAK_SLOPE"))?,
				number("PReLU slope", env!("RECIPE_PRELU_SLOPE"))?,
				number("ELU alpha", env!("RECIPE_ELU_ALPHA"))?,
				number("SELU alpha", env!("RECIPE_SELU_ALPHA"))?,
				number("SELU scale", env!("RECIPE_SELU_SCALE"))?,
				number("GELU scale", env!("RECIPE_GELU_SCALE"))?,
				number("GELU cubic", env!("RECIPE_GELU_CUBIC"))?,
				number("Huber threshold", env!("RECIPE_HUBER_THRESHOLD"))?,
			],
			precision: Compute::FP64,
		})
	}
}
/// The normalization epsilon a model starts with: the Cargo default, which is also
/// the value every bundle saved before models carried their own was lowered with.
fn default_epsilon() -> Result<f64> {
	number("normalization epsilon", env!("RECIPE_NORMALIZATION_EPSILON"))
}
fn number(name: &str, text: &str) -> Result<f64> {
	let value = text.parse::<f64>().map_err(|error| RecipeError::new(format!("invalid {name}: {error}")))?;
	(value.is_finite() && value > 0.0).then_some(value).ok_or_else(|| RecipeError::new(format!("{name} must be finite and positive")))
}
fn fraction(name: &str, text: &str) -> Result<f64> {
	let value = number(name, text)?;
	require(value <= 1.0, format!("{name} must not exceed one")).map(|_| value)
}
fn natural(name: &str, text: &str) -> Result<usize> {
	let value = count(name, text)?;
	require(value != 0, format!("{name} must be positive")).map(|_| value)
}
fn count(name: &str, text: &str) -> Result<usize> {
	text.parse::<usize>().map_err(|error| RecipeError::new(format!("invalid {name}: {error}")))
}
fn stored_graph(graph: &Graph, model: &Model, data: &Data, scale: Option<TargetScale>, precision: Compute, target: &str) -> bundle::StoredGraph {
	let inputs = (0..graph.input.elements()).map(|index| format!("input{index}")).collect();
	// Every selected target column is an output, in the order the user declared them.
	let outputs = if data.autoregressive {
		vec!["char-id".to_owned()]
	} else if data.target.is_empty() {
		vec!["target".to_owned()]
	} else {
		data.target.clone()
	};
	let (norm_mean, norm_scale) = match data.prepared.get() {
		Some(Ok(prepared)) => (prepared.norm_mean.clone(), prepared.norm_scale.clone()),
		_ => (Vec::new(), Vec::new()),
	};
	let (target_min, target_span) = scale.map_or((0.0, 0.0), |s| (s.minimum, s.span));
	let schema = data.prepared.get().and_then(|prepared| prepared.as_ref().ok()).map_or_else(DataSchema::default, |prepared| prepared.schema.clone());
	let artifact = bundle::artifact_key(model, &schema, precision, graph, target);
	bundle::StoredGraph { graph: graph.clone(), model: model.clone(), precision, inputs, outputs, norm_mean, norm_scale, target_min, target_span, bn_stats: Vec::new(), artifact }
}
/// A host-gathered table of one lookup node: the rows every token addresses are
/// decoded on the host and staged in the node's context as [row][position][channel]
/// for the positions a forward writes.
struct HostLookup {
	context: usize,
	hash: RowHash,
	table: StoredWeight,
	width: usize,
	length: usize,
}
struct NativeTape {
	program: NativeProgram,
	precision: NativePrecision,
	values: Buffer,
	contexts: Buffer,
	context_resets: Vec<(usize, usize)>,
	lookups: Vec<HostLookup>,
	tokens: Mutex<Vec<f64>>,
	adjoints: Buffer,
	batch_normalizations: Vec<(usize, usize)>,
	samples: Buffer,
	input_adjoint: Buffer,
	targets: Buffer,
	weights: Buffer,
	frozen: Buffer,
	moments: Buffer,
	variances: Buffer,
	gradient: Buffer,
	metrics: Buffer,
	best_loss: [f64; 4],
	rows: u32,
	parameters: usize,
	step: u32,
	input: Shape,
	output: Shape,
	nodes: Vec<Node>,
	capacity: usize,
	positions: u32,
	vocabulary: f64,
}
macro_rules! ptrs { ($($e:expr),* $(,)?) => { [$(&$e as *const _ as Ptr),*] } }

#[derive(Clone, Copy, Debug)]
enum EpochOperation {
	Full,
	Gradient,
	Optimizer,
}

impl EpochOperation {
	fn gradient(self) -> bool {
		matches!(self, Self::Full | Self::Gradient)
	}
	fn optimizer(self) -> bool {
		matches!(self, Self::Full | Self::Optimizer)
	}
}

impl NativeTape {
	/// `tokens` are the model's ids, one per position of every row, which the
	/// lookups of this graph gather rows for; a whole graph reads its own
	/// samples, and a part of a split graph reads the ids its stream came from.
	fn new(graph: &Graph, samples: &[f64], tokens: &[f64], targets: &[f64], gpu: &'static Gpu, precision: Compute, loss: Option<LossFunction>) -> Result<Self> {
		let input = graph.input.elements();
		require(input != 0 && !samples.is_empty() && samples.len() % input == 0, format!("model input batch expected a nonempty multiple of {input} values, received {}", samples.len()))?;
		let rows = samples.len() / input;
		let output = graph.output.elements();
		require(targets.is_empty() || targets.len() == rows * output, format!("target batch expected 0 or {} values, received {}", rows * output, targets.len()))?;
		let inference = loss.is_none();
		let program = gpu.native_program(graph, rows, precision, loss)?;
		let (precision, layout, parameters) = (program.artifact.precision, program.artifact.layout.clone(), graph.parameters.len());
		// Only the epoch entrypoint reads the optimizer state, the gradient and
		// the adjoints, so an inference tape holds none of them on the device.
		let training = program.artifact.training;
		let state_values = if training { parameters.max(1) } else { 1 };
		let zeros = vec![0.0; state_values];
		let gradient_bytes = if training { checked_mul(program.gradient_values.max(1), precision.model.bytes(), "native gradient allocation")? } else { 1 };
		require(graph.state.moments.is_empty() || graph.state.moments.len() == parameters, "saved optimizer moments have the wrong shape")?;
		require(graph.state.variances.is_empty() || graph.state.variances.len() == parameters, "saved optimizer variances have the wrong shape")?;
		require(graph.frozen.is_empty() || graph.frozen.len() == parameters, "frozen parameters have the wrong shape")?;
		let moments = if training && !graph.state.moments.is_empty() { graph.state.moments.clone() } else { zeros.clone() };
		let variances = if training && !graph.state.variances.is_empty() { graph.state.variances.clone() } else { zeros.clone() };
		let frozen = if training && !graph.frozen.is_empty() { graph.frozen.clone() } else { vec![0_u8; state_values] };
		let batch_normalizations = carried_state(&graph.nodes).collect();
		let best_loss = if graph.state.best_loss.is_empty() {
			[f64::INFINITY, f64::NAN, f64::NAN, f64::INFINITY]
		} else {
			graph.state.best_loss.as_slice().try_into().map_err(|_| RecipeError::new("saved loss state is invalid"))?
		};
		let step = narrow(graph.state.epoch, "optimizer epoch")? as u32;
		let target_buffer = if targets.is_empty() { vec![0.0] } else { targets.to_vec() };
		let adjoints_bytes = if training { layout.adjoints_bytes.max(1) } else { 1 };
		let input_adjoint_bytes = if training { checked_mul(samples.len(), precision.model.bytes(), "native input adjoint allocation")?.max(1) } else { 1 };
		// A graph that starts with a gather reads its input as i32 token ids.
		let vocabulary = graph.nodes.first().filter(|node| node.op == Primitive::Gather).map_or(0.0, |node| node.argument[0]);
		let token_count = tokens.len();
		let tokens = Mutex::new(tokens.to_vec());
		let samples = if vocabulary > 0.0 {
			Buffer::upload(gpu, &samples.iter().map(|id| token_id(*id, vocabulary)).collect::<Result<Vec<_>>>()?)?
		} else {
			Buffer::upload_float(gpu, samples, precision.model)?
		};
		let weights = Buffer::upload(gpu, &native_weight_bytes(graph, precision.model, inference)?)?;
		if program.model_load.is_some() {
			require(!program.artifact.storage.is_empty(), "native model-load storage is empty")?;
			let storage = Buffer::upload(gpu, &program.artifact.storage)?;
			let threads = program.dispatch(NativeEntry::ModelLoad)?.geometry.threads()?;
			let mut call = ptrs![weights.pointer, storage.pointer, threads];
			program.launch_model_load(&mut call)?;
			gpu.synchronize()?;
		} else {
			require(program.artifact.storage.is_empty(), "native artifact storage has no model-load entrypoint")?;
		}
		// The packed embedding table is the gather's context, so it reaches the
		// device whole once and the kernel then reads only the rows it addresses.
		let mut contexts = vec![0_u8; layout.contexts_bytes.max(1)];
		// A lookup's table never leaves the host: the tape keeps it with its hash
		// and stages the rows of every forward window from the ids it holds.
		let (mut lookups, positions) = (Vec::new(), graph_positions(graph));
		for (index, node) in graph.nodes.iter().enumerate() {
			if node.op == Primitive::Gather {
				let table = graph.stored.get(index).and_then(Option::as_ref).ok_or_else(|| RecipeError::new("embedding table is absent"))?;
				let offset = layout.contexts[index];
				let end = checked_add(offset, table.bytes.len(), "embedding table context")?;
				require(end <= contexts.len(), "embedding table exceeds its context arena")?;
				table.bytes.copy_into(&mut contexts[offset..end]);
			}
			if node.op == Primitive::Lookup {
				let table = graph.stored.get(index).and_then(Option::as_ref).ok_or_else(|| RecipeError::new("per-layer embedding table is absent"))?.clone();
				let words = graph.programs.get(node.program_offset..node.program_offset + node.program_count * 3).ok_or_else(|| RecipeError::new("per-layer embedding hash is absent"))?;
				require(node.output.length == positions, format!("per-layer embedding reads {} positions of {positions} ids", node.output.length))?;
				require(token_count == rows * positions, format!("per-layer embedding reads {} ids for {rows} rows of {positions} positions, received {token_count}", rows * positions))?;
				lookups.push(HostLookup { context: layout.contexts[index], hash: RowHash::from_words(words)?, table, width: node.argument[1] as usize, length: node.output.length });
			}
		}
		// A new sequence clears every mutable context span. Packed embedding
		// tables and saved evaluation statistics remain unchanged.
		let mut context_resets = Vec::<(usize, usize)>::new();
		for (index, node) in graph.nodes.iter().enumerate() {
			let persistent = node.op == Primitive::Gather || (node.op == Primitive::Normalize && normalize_mode(node.argument[0])? == program_ir::NormalizeMode::Evaluation);
			if persistent {
				continue;
			}
			let start = layout.contexts[index];
			let end = layout.contexts.get(index + 1).copied().unwrap_or(layout.contexts_bytes);
			if let Some((_, prior_end)) = context_resets.last_mut().filter(|(_, prior_end)| *prior_end == start) {
				*prior_end = end;
			} else {
				context_resets.push((start, end));
			}
		}
		let tape = Self {
			program,
			precision,
			values: Buffer::upload(gpu, &vec![0_u8; layout.values_bytes.max(1)])?,
			contexts: Buffer::upload(gpu, &contexts)?,
			context_resets,
			lookups,
			tokens,
			adjoints: Buffer { runtime: gpu, pointer: gpu.allocate(adjoints_bytes)?, bytes: adjoints_bytes },
			batch_normalizations,
			samples,
			input_adjoint: Buffer { runtime: gpu, pointer: gpu.allocate(input_adjoint_bytes)?, bytes: input_adjoint_bytes },
			targets: Buffer::upload_float(gpu, &target_buffer, precision.model)?,
			weights,
			frozen: Buffer::upload(gpu, &frozen)?,
			moments: Buffer::upload_float(gpu, &moments, precision.state)?,
			variances: Buffer::upload_float(gpu, &variances, precision.state)?,
			gradient: Buffer::upload(gpu, &vec![0_u8; gradient_bytes])?,
			metrics: Buffer::upload_float(gpu, &[0.0], precision.state)?,
			best_loss,
			rows: narrow(rows, "native rows")? as u32,
			parameters,
			step,
			input: graph.input,
			output: graph.output,
			nodes: graph.nodes.clone(),
			capacity: rows,
			positions: narrow(positions, "native input positions")? as u32,
			vocabulary,
		};
		tape.stage_lookups(0, tape.positions)?;
		Ok(tape)
	}
	fn forward(&self) -> Result<()> {
		self.forward_window(0, self.positions)
	}
	/// The id one input value names.
	fn token(&self, value: f64) -> Result<u32> {
		if self.vocabulary > 0.0 {
			return token_id(value, self.vocabulary).map(|id| id as u32);
		}
		require(value.fract() == 0.0 && value >= 0.0 && value <= f64::from(u32::MAX), format!("input value {value} is not a token id"))?;
		Ok(value as u32)
	}
	/// Gather every lookup's rows for the positions `begin..end` of every row on
	/// the host and write them into the staging context the device reads.
	fn stage_lookups(&self, begin: u32, end: u32) -> Result<()> {
		let (begin, end, bytes) = (begin as usize, end as usize, self.precision.model.bytes());
		let tokens = self.tokens.lock().map_err(|_| RecipeError::new("token state is poisoned"))?;
		for lookup in &self.lookups {
			let channels = lookup.hash.heads() * lookup.width;
			for row in 0..self.rows as usize {
				let ids = tokens[row * lookup.length..(row + 1) * lookup.length].iter().map(|value| self.token(*value)).collect::<Result<Vec<_>>>()?;
				let mut staged = Vec::with_capacity((end - begin) * channels);
				for position in begin..end {
					for index in lookup.hash.rows_at(&ids, position) {
						staged.extend(ngram::table_row(&lookup.table, lookup.width, index)?);
					}
				}
				let slot = checked_mul(checked_add(checked_mul(row, lookup.length, "lookup row")?, begin, "lookup slot")?, channels, "lookup staging offset")?;
				self.contexts.write_float_bytes(checked_add(lookup.context, checked_mul(slot, bytes, "lookup staging bytes")?, "lookup staging")?, &staged, self.precision.model)?;
			}
		}
		Ok(())
	}
	/// Write the output positions that the input positions before `end` reach and
	/// that the input positions before `begin` did not. The arenas keep every
	/// earlier position, so a step reads the attention keys and values, the
	/// recurrent state, and the convolution tail that earlier calls left.
	fn forward_window(&self, begin: u32, end: u32) -> Result<()> {
		require(begin <= end && end <= self.positions, format!("forward window {begin}..{end} is outside the {} input positions", self.positions))?;
		self.stage_lookups(begin, end)?;
		let threads = self.program.forward.geometry.threads()?;
		let rows = self.rows;
		let thread_count = threads;
		let mut call = ptrs![self.samples.pointer, self.weights.pointer, self.values.pointer, self.contexts.pointer, rows, thread_count, begin, end];
		self.program.launch_forward(&mut call).map_err(|error| RecipeError::new(format!("forward: {error}")))?;
		Ok(())
	}
	/// The runs of the one-row input that the input positions `begin..end`
	/// occupy: one run of ids for a graph that starts with a gather, and one run
	/// per channel otherwise, as the arenas hold a row channel by channel.
	fn input_runs(&self, begin: u32, end: u32) -> Vec<(usize, usize)> {
		if self.vocabulary > 0.0 { vec![(begin as usize, (end - begin) as usize)] } else { window_runs(self.input, begin, end) }
	}
	/// Replace a run of the input values from element `first`. A graph that
	/// starts with a gather holds its input as i32 token ids.
	fn write_samples(&self, first: usize, values: &[f64]) -> Result<()> {
		if self.vocabulary > 0.0 {
			let ids = values.iter().map(|value| token_id(*value, self.vocabulary).map(i32::to_ne_bytes)).collect::<Result<Vec<_>>>()?;
			self.samples.write_bytes(checked_mul(first, size_of::<i32>(), "token offset")?, ids.as_flattened())
		} else {
			self.samples.write_float_bytes(checked_mul(first, self.precision.model.bytes(), "sample offset")?, values, self.precision.model)
		}
	}
	fn write_tokens(&self, first: usize, values: &[f64]) -> Result<()> {
		let mut tokens = self.tokens.lock().map_err(|_| RecipeError::new("token state is poisoned"))?;
		let end = checked_add(first, values.len(), "token write")?;
		let target = tokens.get_mut(first..end).ok_or_else(|| RecipeError::new("token write exceeds the sequence"))?;
		target.copy_from_slice(values);
		Ok(())
	}
	fn write_sample(&self, position: usize, value: f64) -> Result<()> {
		self.write_tokens(position, &[value])?;
		self.write_samples(position, &[value])
	}
	/// The output positions the input positions `begin..end` reach: the window
	/// every node derives from its source, as the emitted forward derives it,
	/// carried to the last node.
	fn output_window(&self, begin: u32, end: u32) -> Result<(u32, u32)> {
		let mut windows = Vec::with_capacity(self.nodes.len());
		for node in &self.nodes {
			let (begin, end) = usize::try_from(node.source).map_or((begin, end), |source| windows[source]);
			let length = narrow(node.output.length, "node length")? as u32;
			windows.push(match node.op {
				Primitive::Predictor => (0, length),
				Primitive::Pool => {
					let size = integer_argument(node.argument[0], "pool size")? as u32;
					require(size > 0, "native pool size must be positive")?;
					(begin / size, end.div_ceil(size).min(length))
				}
				Primitive::Contraction if node.argument[0] > 1.0 => {
					let lag = integer_argument(node.argument[0], "contraction kernel")? as u32 - 1;
					(begin.saturating_sub(lag), end.saturating_sub(lag))
				}
				_ => (begin, end),
			});
		}
		windows.last().copied().ok_or_else(|| RecipeError::new("native model has no node"))
	}
	/// Restore the token, input, and mutable arenas before a new sequence, while
	/// retaining packed tables and saved evaluation statistics.
	fn reset_sequence(&self) -> Result<()> {
		self.tokens.lock().map_err(|_| RecipeError::new("token state is poisoned"))?.fill(0.0);
		self.samples.clear()?;
		self.values.clear()?;
		self.context_resets.iter().try_for_each(|&(start, end)| self.contexts.clear_range(start, end - start))
	}
	/// The bytes this tape holds on its device: the weights, the input row, and
	/// the value and context arenas that keep every position's activations,
	/// attention keys and values, recurrent state, convolution tail and batch
	/// normalization statistics.
	fn resident_bytes(&self) -> usize {
		self.weights.bytes + self.samples.bytes + self.values.bytes + self.contexts.bytes
	}
	fn inject_bn_stats(&self, stats: &[f64]) -> Result<()> {
		let expected = self.batch_normalizations.iter().map(|(_, channels)| 2 * channels).sum::<usize>();
		require(stats.len() == expected, format!("batch normalization expected {expected} saved statistics, received {}", stats.len()))?;
		let mut offset = 0;
		for &(node, channels) in &self.batch_normalizations {
			let end = offset + 2 * channels;
			self.contexts.write_float_bytes(self.program.artifact.layout.contexts[node], &stats[offset..end], self.precision.model)?;
			offset = end;
		}
		Ok(())
	}
	fn extract_bn_stats(&self) -> Result<Vec<f64>> {
		let mut stats = Vec::new();
		for &(node, channels) in &self.batch_normalizations {
			stats.extend(self.contexts.download_float_bytes(self.program.artifact.layout.contexts[node], 2 * channels, self.precision.model)?);
		}
		Ok(stats)
	}
	fn predictions(&self) -> Result<Vec<f64>> {
		self.output(0, self.capacity * self.output.elements())
	}
	/// A run of `count` output values from element `first` of the output arena.
	fn output(&self, first: usize, count: usize) -> Result<Vec<f64>> {
		let arena = *self.program.artifact.layout.values.last().ok_or_else(|| RecipeError::new("native model has no output arena"))?;
		let offset = checked_add(arena, checked_mul(first, self.precision.model.bytes(), "output offset")?, "output arena offset")?;
		let values = self.values.download_float_bytes(offset, count, self.precision.model)?;
		require(values.iter().all(|value| value.is_finite()), format!("device {} produced a nonfinite prediction", self.program.gpu.name)).map(|_| values)
	}
	fn epoch_launch(&mut self, rate: f64, config: Config, operation: EpochOperation) -> Result<()> {
		require(self.step != 0, "optimizer epoch is absent")?;
		let threads = self.program.dispatch(NativeEntry::Epoch)?.geometry.threads()?;
		let rows = self.rows;
		let thread_count = threads;
		let beta1 = self.precision.state.below_one(config.beta1);
		let beta2 = self.precision.state.below_one(config.beta2);
		let epsilon = self.precision.state.optimizer_epsilon(config.epsilon);
		let beta1_power = beta1.powi(self.step as i32);
		let beta2_power = beta2.powi(self.step as i32);
		let decay = config.decay;
		let encoded = [rate, beta1, beta2, beta1_power, beta2_power, epsilon, decay].map(|value| self.precision.state.pack(value));
		let run_gradient = u32::from(operation.gradient());
		let run_optimizer = u32::from(operation.optimizer());
		let mut call = ptrs![
			self.samples.pointer,
			self.targets.pointer,
			self.weights.pointer,
			self.frozen.pointer,
			self.moments.pointer,
			self.variances.pointer,
			self.gradient.pointer,
			self.metrics.pointer,
			self.input_adjoint.pointer,
			self.values.pointer,
			self.contexts.pointer,
			self.adjoints.pointer,
			rows,
			thread_count,
			encoded[0],
			encoded[1],
			encoded[2],
			encoded[3],
			encoded[4],
			encoded[5],
			encoded[6],
			run_gradient,
			run_optimizer
		];
		debug(&format!("epoch {} {operation:?} launch", self.step))?;
		self.program.launch_epoch(&mut call).map_err(|error| RecipeError::new(format!("training epoch: {error}")))?;
		debug(&format!("epoch {} {operation:?} launch complete", self.step))?;
		Ok(())
	}
	fn objective(&self) -> Result<f64> {
		let objective = self.metrics.download_float(1, self.precision.state)?[0];
		debug(&format!("epoch {} metric complete", self.step))?;
		Ok(objective)
	}
	fn full_epoch(&mut self, rate: f64, config: Config) -> Result<f64> {
		self.epoch_launch(rate, config, EpochOperation::Full)?;
		self.objective()
	}
	/// Computes this shard's loss and reduced parameter gradient without
	/// changing optimizer state or model weights.
	fn gradient_launch(&mut self, rate: f64, config: Config) -> Result<f64> {
		self.epoch_launch(rate, config, EpochOperation::Gradient)?;
		self.objective()
	}
	/// Applies the emitted AdamW operation to the gradient already stored on
	/// this tape's device through the same model epoch entrypoint.
	fn optimizer_launch(&mut self, rate: f64, config: Config) -> Result<()> {
		self.epoch_launch(rate, config, EpochOperation::Optimizer)
	}
	fn advance(&mut self) -> Result<()> {
		self.step = self.step.checked_add(1).ok_or_else(|| RecipeError::new("optimizer epoch overflows"))?;
		Ok(())
	}
	fn weights(&self) -> Result<Vec<f64>> {
		self.weights.download_float(self.parameters, self.precision.model)
	}
	/// The reduced full-shard parameter gradient the last epoch dispatch left
	/// on the device.
	fn download_gradient(&self) -> Result<Vec<f64>> {
		self.gradient.download_float(self.parameters, self.precision.model)
	}
	fn upload_gradient(&self, gradient: &[f64]) -> Result<()> {
		self.gradient.write_float_bytes(0, gradient, self.precision.model)
	}
	fn upload_weights(&self, weights: &[f64]) -> Result<()> {
		self.weights.write_float_bytes(0, weights, self.precision.model)
	}
	fn optimizer_state(&self) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
		Ok((self.weights()?, self.moments.download_float(self.parameters, self.precision.state)?, self.variances.download_float(self.parameters, self.precision.state)?))
	}
	fn capture(&self, graph: &mut Graph) -> Result<()> {
		let (weights, moments, variances) = self.optimizer_state()?;
		graph.parameters = weights;
		graph.state.moments = moments;
		graph.state.variances = variances;
		graph.state.epoch = self.step as usize;
		graph.state.best_loss = self.best_loss.to_vec();
		Ok(())
	}
	fn tile(&self) -> Tile {
		self.program.tile
	}
	/// The dispatched contraction schedule, in model execution order: one
	/// forward/gradient/previous group per contraction node, collapsed to a single
	/// extent only when every group agrees.
	fn schedule(&self) -> String {
		let extents = self
			.program
			.contractions
			.iter()
			.flatten()
			.flat_map(|node| [node.forward, node.gradient, node.previous])
			.map(|extent| format!("{}x{}x{}", extent.m, extent.n, extent.k))
			.collect::<Vec<_>>();
		if extents.windows(2).all(|pair| pair[0] == pair[1]) {
			return extents.first().cloned().unwrap_or_default();
		}
		self.program
			.contractions
			.iter()
			.flatten()
			.map(|node| [node.forward, node.gradient, node.previous].map(|extent| format!("{}x{}x{}", extent.m, extent.n, extent.k)).join("/"))
			.collect::<Vec<_>>()
			.join(" ")
	}
	fn device_label(&self) -> Result<String> {
		device_label(self.program.gpu)
	}
}
fn device_label(gpu: &Gpu) -> Result<String> {
	if gpu.name.contains(':') { Ok(gpu.name.clone()) } else { Ok(format!("{}:{}", local_host()?, gpu.name)) }
}
/// Tracks the running best loss and decides whether this epoch triggers a
/// checkpoint, updating the four-slot loss state in place.
fn observe_loss(best_loss: &mut [f64; 4], loss: f64, tolerance: f64) -> bool {
	let (old_best, last, armed, saved) = (best_loss[0], best_loss[1], best_loss[2].is_finite(), best_loss[3]);
	let best = if loss < old_best { loss } else { old_best };
	let trigger = armed && loss > last * (2.0 - tolerance) && tolerance > 0.0;
	best_loss[0] = best;
	best_loss[1] = loss;
	best_loss[2] = if trigger {
		f64::NAN
	} else if last.is_finite() && last < saved && loss < saved {
		best
	} else {
		best_loss[2]
	};
	if trigger {
		best_loss[3] = best;
	}
	trigger
}
/// One measured direction of a topology link.
#[derive(Clone, Copy)]
struct TransferCost {
	latency: Duration,
	bandwidth: f64,
}
impl TransferCost {
	fn seconds(self, bytes: usize) -> f64 {
		self.latency.as_secs_f64() + bytes as f64 / self.bandwidth
	}
}
/// The measured behavior of one device: the two transfer directions between it and the coordinating
/// host, the gradient work it retires each second, and the fixed cost of one dispatch on it.
#[derive(Clone, Copy)]
struct Link {
	to_host: TransferCost,
	from_host: TransferCost,
	work: f64,
	overhead: f64,
}
fn measure_link(gpu: &'static Gpu, config: Config) -> Result<Link> {
	let probe_bytes = parse_natural(env!("RECIPE_TOPOLOGY_PROBE_BYTES"), "topology probe bytes must be a positive integer");
	let mut scratch = vec![0_u8; probe_bytes];
	let pointer = gpu.upload(0, scratch.as_ptr().cast(), probe_bytes)?;
	let measured = (|| {
		gpu.synchronize()?;
		let started = Instant::now();
		gpu.download(scratch.as_mut_ptr().cast(), pointer, 1)?;
		gpu.synchronize()?;
		let to_host_latency = started.elapsed();
		let started = Instant::now();
		gpu.download(scratch.as_mut_ptr().cast(), pointer, probe_bytes)?;
		gpu.synchronize()?;
		let to_host_bandwidth = probe_bytes as f64 / started.elapsed().as_secs_f64().max(f64::EPSILON);
		let started = Instant::now();
		gpu.upload(pointer, scratch.as_ptr().cast(), 1)?;
		gpu.synchronize()?;
		let from_host_latency = started.elapsed();
		let started = Instant::now();
		gpu.upload(pointer, scratch.as_ptr().cast(), probe_bytes)?;
		gpu.synchronize()?;
		let from_host_bandwidth = probe_bytes as f64 / started.elapsed().as_secs_f64().max(f64::EPSILON);
		let (work, overhead) = calibrate(gpu, config)?;
		Ok(Link {
			to_host: TransferCost { latency: to_host_latency, bandwidth: to_host_bandwidth },
			from_host: TransferCost { latency: from_host_latency, bandwidth: from_host_bandwidth },
			work,
			overhead,
		})
	})();
	gpu.free(pointer);
	measured
}
static LINKS: OnceLock<Result<Vec<Link>>> = OnceLock::new();
#[derive(Clone, Copy)]
struct Transfer {
	from: usize,
	to: usize,
	bytes: usize,
	cost: TransferCost,
}
impl Transfer {
	fn seconds(self) -> f64 {
		self.cost.seconds(self.bytes)
	}
}
/// The route one training run takes: the row share of every shard, the movement its fused epoch performs, and the
/// complete epoch predicted for it from computation, transfers, synchronization, and persistent-state movement.
struct Placement {
	shares: Vec<f64>,
	gradient_to_host: Vec<Transfer>,
	gradient_to_primary: Transfer,
	weights_to_host: Transfer,
	weights_from_host: Vec<Transfer>,
	loss: LossFunction,
	predicted: [f64; 4],
}
impl Placement {
	fn movements(&self) -> impl Iterator<Item = &Transfer> {
		self.gradient_to_host.iter().chain([&self.gradient_to_primary, &self.weights_to_host]).chain(&self.weights_from_host)
	}
	fn seconds(&self) -> f64 {
		self.predicted.iter().sum()
	}
}
/// The planned gradient work of one fused epoch over `rows` rows: every
/// contraction's forward, gradient, and previous-adjoint tile, every node's
/// elementwise traffic, and the loss reduction. The one optimizer update the
/// leading device applies is priced once, separately, by `optimizer_work`.
fn gradient_work(graph: &Graph, rows: usize) -> Result<f64> {
	let tiles = native_contraction_shapes(graph, rows)?
		.iter()
		.flatten()
		.flat_map(|shapes| [shapes.forward, shapes.gradient, shapes.previous])
		.map(|extent| 2.0 * f64::from(extent.m) * f64::from(extent.n) * f64::from(extent.k))
		.sum::<f64>();
	let elementwise = graph.nodes.iter().map(|node| 8.0 * rows as f64 * node.output.elements() as f64).sum::<f64>();
	Ok(8.0 * checked_mul(rows, graph.output.elements(), "predicted loss reduction")? as f64 + tiles + elementwise)
}
fn optimizer_work(graph: &Graph) -> f64 {
	16.0 * graph.parameters.len() as f64
}
/// Measures one device through the configured surrogate workload. The optimizer
/// dispatch isolates fixed cost, and the gradient dispatch measures planned work
/// without allocating, forwarding, training, or dispatching the placed model.
fn calibrate(gpu: &'static Gpu, config: Config) -> Result<(f64, f64)> {
	let workers = if matches!(&gpu.driver, Driver::Cpu) { cpu_worker_threads()? as usize } else { 1 };
	let rows = checked_mul(checked_mul(config.surrogate_epochs, config.surrogate_width, "surrogate rows")?, workers, "parallel surrogate rows")?;
	let features = config.surrogate_width;
	let samples = (0..rows * features).map(|value| ((value % 17) as f64 - 8.0) / 8.0).collect::<Vec<_>>();
	let targets = (0..rows).map(|value| ((value % 5) as f64 - 2.0) / 2.0).collect::<Vec<_>>();
	let prepared = Prepared {
		samples: samples.clone(),
		targets: targets.clone(),
		target_width: 1,
		rows,
		source_rows: rows,
		features,
		schema: DataSchema::default(),
		sequence: None,
		target_categorical: false,
		norm_mean: Vec::new(),
		norm_scale: Vec::new(),
		identities: Vec::new(),
		fitted: Vec::new(),
		bound: None,
	};
	let graph = compile(&surrogate_model(config.surrogate_width), &prepared, &targets, rows, gpu, config, true)?;
	let mut tape = NativeTape::new(&graph, &samples, &samples, &targets, gpu, config.precision, Some(mse))?;
	let timed = |tape: &mut NativeTape, gradient: bool| -> Result<f64> {
		tape.advance()?;
		let started = Instant::now();
		if gradient {
			tape.full_epoch(config.surrogate_rate, config)?;
		} else {
			tape.optimizer_launch(config.surrogate_rate, config)?;
		}
		gpu.synchronize()?;
		Ok(started.elapsed().as_secs_f64())
	};
	timed(&mut tape, true)?;
	let overhead = timed(&mut tape, false)?;
	let epoch = timed(&mut tape, true)?;
	let gradient = epoch - overhead;
	require(gradient.is_finite() && gradient > 0.0, "surrogate gradient time must be finite and positive")?;
	Ok(((gradient_work(&graph, rows)? / gradient).max(1.0), overhead))
}
/// Plans one candidate route from the workload and storage plan already established for this run: the row share of
/// every shard, the movement list its fused epoch performs, and the complete epoch that movement and each device's
/// measured behavior predict.
fn plan_route(route: &[usize], links: &[Link], graph: &Graph, rows: usize, bytes: usize, loss: LossFunction, policy: MultiDevice) -> Result<(Vec<usize>, Placement)> {
	let total = route.iter().map(|device| if policy == MultiDevice::Auto { 1.0 } else { links[*device].work }).sum::<f64>();
	let mut counts = route.iter().map(|device| ((rows as f64 * if policy == MultiDevice::Auto { 1.0 } else { links[*device].work } / total) as usize).max(1)).collect::<Vec<_>>();
	counts[0] += rows - counts.iter().sum::<usize>();
	let (gradient_to_host, weights_from_host) = (
		route.iter().enumerate().map(|(shard, device)| Transfer { from: shard + 1, to: 0, bytes, cost: links[*device].to_host }).collect::<Vec<_>>(),
		route.iter().enumerate().skip(1).map(|(shard, device)| Transfer { from: 0, to: shard + 1, bytes, cost: links[*device].from_host }).collect::<Vec<_>>(),
	);
	let placement = Placement {
		shares: counts.iter().map(|count| *count as f64 / rows as f64).collect(),
		gradient_to_host,
		gradient_to_primary: Transfer { from: 0, to: 1, bytes, cost: links[route[0]].from_host },
		weights_to_host: Transfer { from: 1, to: 0, bytes, cost: links[route[0]].to_host },
		weights_from_host,
		loss,
		predicted: [0.0; 4],
	};
	let bandwidth = |transfer: &Transfer| transfer.bytes as f64 / transfer.cost.bandwidth;
	// Shards compute their gradients concurrently, so the slowest shard sets the
	// route's gradient time, and the leading device adds the one optimizer update.
	let computation = route.iter().zip(&counts).map(|(device, count)| Ok(gradient_work(graph, *count)? / links[*device].work)).collect::<Result<Vec<_>>>()?.into_iter().fold(0.0, f64::max)
		+ optimizer_work(graph) / links[route[0]].work;
	let transfers = placement.gradient_to_host.iter().map(bandwidth).sum::<f64>() + bandwidth(&placement.gradient_to_primary);
	let movement = bandwidth(&placement.weights_to_host) + placement.weights_from_host.iter().map(bandwidth).sum::<f64>();
	let synchronization = route.iter().map(|device| links[*device].overhead).sum::<f64>() + placement.movements().map(|transfer| transfer.cost.latency.as_secs_f64()).sum::<f64>();
	Ok((counts, Placement { predicted: [computation, transfers, synchronization, movement], ..placement }))
}
/// Selects the route this run trains on. `multi-device = false` keeps the
/// local device, `true` forces every selected device, and `"auto"` predicts the
/// complete epoch of every valid candidate route from the established workload,
/// the storage plan, and measured device behavior, then takes the lowest. No
/// policy allocates or dispatches the model being placed to decide.
fn select_route(gpus: &'static [&'static Gpu], graph: &Graph, rows: usize, precision: Compute, loss: LossFunction, config: Config) -> Result<(Vec<usize>, Vec<usize>, Placement)> {
	let bytes = checked_mul(graph.parameters.len().max(1), precision.bytes(), "topology transfer bytes")?;
	let links = LINKS.get_or_init(|| gpus.iter().map(|gpu| measure_link(gpu, config)).collect()).as_ref().map_err(Clone::clone)?;
	for (gpu, link) in gpus.iter().zip(links) {
		eprintln!(
			"measured {} {:.6e} work/s {:.9}s/dispatch to-host {:.1} MB/s {:.0?} from-host {:.1} MB/s {:.0?}",
			device_label(gpu)?,
			link.work,
			link.overhead,
			link.to_host.bandwidth / 1e6,
			link.to_host.latency,
			link.from_host.bandwidth / 1e6,
			link.from_host.latency
		);
	}
	let candidates: Vec<Vec<usize>> = match config.multi_device {
		MultiDevice::Local => vec![vec![0]],
		MultiDevice::Forced => vec![(0..gpus.len()).collect()],
		MultiDevice::Auto => (1..1_u64 << gpus.len()).map(|mask| (0..gpus.len()).filter(|device| mask >> device & 1 == 1).collect()).collect(),
	};
	let mut best: Option<(Vec<usize>, Vec<usize>, Placement)> = None;
	for mut route in candidates.into_iter().filter(|route| route.len() <= rows) {
		// The fastest device leads the route and applies the one update.
		route.sort_by(|left, right| links[*right].work.total_cmp(&links[*left].work).then(left.cmp(right)));
		let (counts, placement) = plan_route(&route, &links, graph, rows, bytes, loss, config.multi_device)?;
		let [computation, transfers, synchronization, movement] = placement.predicted;
		eprintln!(
			"route {} rows {} predicted epoch {:.9}s = computation {computation:.9} + transfers {transfers:.9} + synchronization {synchronization:.9} + persistent-state {movement:.9}",
			route.iter().map(|device| device_label(gpus[*device])).collect::<Result<Vec<_>>>()?.join(","),
			counts.iter().map(usize::to_string).collect::<Vec<_>>().join(","),
			placement.seconds()
		);
		best = if best.as_ref().is_none_or(|previous| placement.seconds() < previous.2.seconds()) { Some((route, counts, placement)) } else { best };
	}
	best.ok_or_else(|| RecipeError::new("no candidate route fits this workload"))
}
/// A training tape placed across the selected device topology. One device
/// trains through the same gradient and optimizer entrypoints. Across several
/// devices the rows shard contiguously, every device computes a gradient, the
/// primary device applies the one emitted optimizer, and its weights broadcast.
struct DeviceTape {
	shards: Vec<NativeTape>,
	placement: Placement,
}
impl DeviceTape {
	fn new(graph: &Graph, samples: &[f64], targets: &[f64], gpus: &'static [&'static Gpu], precision: Compute, loss: LossFunction, config: Config) -> Result<Self> {
		let (input, output) = (graph.input.elements(), graph.output.elements());
		require(input != 0 && samples.len() % input == 0, "model input batch is not a whole number of rows")?;
		let rows = samples.len() / input;
		require(!targets.is_empty(), "training requires targets")?;
		require(
			gpus.len() == 1 || !graph.nodes.iter().any(|node| node.op == Primitive::Normalize && node.argument[0] == 0.0),
			"batch normalization computes whole-batch statistics, so this model trains on one device",
		)?;
		let (route, counts, placement) = select_route(gpus, graph, rows, precision, loss, config)?;
		eprintln!("selected route {} predicted epoch {:.9}s", route.iter().map(|device| device_label(gpus[*device])).collect::<Result<Vec<_>>>()?.join(","), placement.seconds());
		let (mut shards, mut start) = (Vec::new(), 0);
		for (device, count) in route.iter().zip(&counts) {
			let end = start + count;
			shards.push(NativeTape::new(
				graph,
				&samples[start * input..end * input],
				&samples[start * input..end * input],
				&targets[start * output..end * output],
				gpus[*device],
				precision,
				Some(loss),
			)?);
			start = end;
		}
		Ok(Self { shards, placement })
	}
	fn forward(&mut self) -> Result<()> {
		self.shards.iter().try_for_each(NativeTape::forward)
	}
	fn predictions(&self) -> Result<Vec<f64>> {
		let mut predictions = Vec::new();
		for shard in &self.shards {
			predictions.extend(shard.predictions()?);
		}
		Ok(predictions)
	}
	fn inject_bn_stats(&self, stats: &[f64]) -> Result<()> {
		if self.shards.len() > 1 {
			return require(stats.is_empty(), "batch normalization statistics cannot place across devices");
		}
		self.shards[0].inject_bn_stats(stats)
	}
	fn extract_bn_stats(&self) -> Result<Vec<f64>> {
		if self.shards.len() > 1 {
			return Ok(Vec::new());
		}
		self.shards[0].extract_bn_stats()
	}
	fn advance(&mut self) -> Result<()> {
		self.shards.iter_mut().try_for_each(NativeTape::advance)
	}
	fn step(&self) -> u32 {
		self.shards[0].step
	}
	fn best_loss(&self) -> [f64; 4] {
		self.shards[0].best_loss
	}
	fn tile(&self) -> Tile {
		self.shards[0].tile()
	}
	fn schedule(&self) -> String {
		self.shards[0].schedule()
	}
	/// The one fused epoch every policy runs: each shard computes its gradient
	/// concurrently, the leading device applies the one emitted optimizer to the
	/// aggregate, and the updated persistent weights return to every shard.
	fn epoch(&mut self, rate: f64, tolerance: f64, config: Config) -> Result<(f64, bool)> {
		if self.shards.len() == 1 {
			let loss = self.shards[0].full_epoch(rate, config)?;
			let checkpoint_requested = observe_loss(&mut self.shards[0].best_loss, loss, tolerance);
			return Ok((loss, checkpoint_requested));
		}
		let placement = &self.placement;
		let shards = &mut self.shards;
		let measured = std::thread::scope(|scope| {
			let dispatched = shards.iter_mut().zip(&placement.gradient_to_host).map(|(shard, transfer)| {
				let transfer = *transfer;
				scope.spawn(move || -> Result<(f64, Vec<f64>)> {
					require(transfer.to == 0, "gradient transfer must end on the coordinating host")?;
					let objective = shard.gradient_launch(rate, config)?;
					Ok((objective, shard.download_gradient()?))
				})
			});
			dispatched.collect::<Vec<_>>().into_iter().map(|shard| shard.join().map_err(|_| RecipeError::new("device epoch panicked"))?).collect::<Result<Vec<_>>>()
		})?;
		let root_metric = placement.loss.0 == 1;
		let loss = if root_metric {
			measured.iter().zip(&placement.shares).map(|((objective, _), share)| share * objective * objective).sum::<f64>().sqrt()
		} else {
			measured.iter().zip(&placement.shares).map(|((objective, _), share)| share * objective).sum()
		};
		let parameters = self.shards[0].parameters;
		let mut gradient = vec![0.0; parameters];
		for ((objective, shard_gradient), share) in measured.iter().zip(&placement.shares) {
			// The RMSE seed divides by the shard-local loss, so restoring the
			// whole-batch gradient rescales each shard by its loss ratio.
			let scale = share * if root_metric { if loss == 0.0 { 0.0 } else { objective / loss } } else { 1.0 };
			for (total, partial) in gradient.iter_mut().zip(shard_gradient) {
				*total += scale * partial;
			}
		}
		require(
			placement.gradient_to_primary.from == 0 && placement.gradient_to_primary.to == 1 && placement.weights_to_host.from == 1 && placement.weights_to_host.to == 0,
			"the aggregate gradient and the updated weights must cross the coordinating host",
		)?;
		self.shards[0].upload_gradient(&gradient)?;
		self.shards[0].optimizer_launch(rate, config)?;
		let weights = self.shards[0].weights()?;
		for (shard, transfer) in self.shards.iter().skip(1).zip(&placement.weights_from_host) {
			require(transfer.from == 0, "weight transfer must originate on the coordinating host")?;
			shard.upload_weights(&weights)?;
		}
		let checkpoint_requested = observe_loss(&mut self.shards[0].best_loss, loss, tolerance);
		Ok((loss, checkpoint_requested))
	}
	fn weights(&self) -> Result<Vec<f64>> {
		self.shards[0].weights()
	}
	fn capture(&self, graph: &mut Graph) -> Result<()> {
		self.shards[0].capture(graph)
	}
	/// Reports the executing route and every movement its fused epoch performs, in the order the epoch performs them.
	fn print_devices(&self, graph: &Graph) -> Result<()> {
		for (index, node) in graph.nodes.iter().enumerate() {
			if node.op == Primitive::Gather {
				let (layout, row) = embedding_row(node)?;
				let table = checked_mul(integer_argument(node.argument[0], "embedding vocabulary")? as usize, row, "embedding table bytes")?;
				eprintln!("movement gather {} {row} bytes per token of a {table} byte table", layout.name);
			}
			if node.op == Primitive::Lookup
				&& let Some(table) = graph.stored.get(index).and_then(Option::as_ref)
			{
				let (heads, rows) = (node.argument[0] as usize, (node.argument[2] as usize).max(1));
				eprintln!("movement lookup {heads} rows of {} bytes per token from a {} byte host table", table.bytes.len() / rows, table.bytes.len());
			}
		}
		for (shard, share) in self.shards.iter().zip(&self.placement.shares) {
			eprintln!("{}.{} rows {} share {share:.6}", shard.device_label()?, shard.precision.model.label(), shard.rows);
		}
		for (index, movement) in self.placement.movements().enumerate() {
			let kind = ["gradient", "aggregate", "weights"][(index >= self.shards.len()) as usize + (index > self.shards.len()) as usize];
			eprintln!(
				"movement {kind} {}>{} {} bytes {:.0?} {:.1} MB/s {:.6} ms",
				movement.from,
				movement.to,
				movement.bytes,
				movement.cost.latency,
				movement.cost.bandwidth / 1e6,
				movement.seconds() * 1e3
			);
		}
		Ok(())
	}
}
#[derive(Clone, Copy)]
enum CheckpointStatus {
	Saved,
	Kept,
}
fn checkpoint(path: &Path, schema: &DataSchema, stored: &mut bundle::StoredGraph, tape: &DeviceTape) -> Result<CheckpointStatus> {
	if let Ok((_, saved)) = bundle::load_semantic(path) {
		if saved.first().and_then(|g| g.state.best_loss.first().copied()).is_some_and(|v| v <= tape.best_loss()[0]) {
			return Ok(CheckpointStatus::Kept);
		}
	}
	tape.capture(&mut stored.graph)?;
	bundle::save_semantic(path, schema, std::slice::from_mut(stored))?;
	Ok(CheckpointStatus::Saved)
}
fn structural(value: f64) -> Result<i32> {
	require(value.is_finite() && value.fract() == 0.0 && value >= f64::from(i32::MIN) && value <= f64::from(i32::MAX), "node structural argument is invalid").map(|_| value as i32)
}
fn graph_rows_buffer(shape: Shape, rows: usize, element: usize) -> Result<usize> {
	checked_mul(checked_mul(rows, shape.elements(), "node elements")?, element, "node bytes")
}
// An embedding row is addressed inside the packed table, so the row must span
// whole blocks and the layout must keep one row's blocks together.
fn embedding_row(node: &Node) -> Result<(&'static Quantization, usize)> {
	let format = StorageFormat(node.argument[8] as u16);
	let layout = format.spec().ok_or_else(|| RecipeError::new("embedding table must be stored in a quantization"))?.codec.quantization();
	require(!matches!(layout.native, NativeDequant::Nf4), format!("embedding table cannot use {}: its codebook addresses the whole tensor", layout.name))?;
	require(node.output.channels % layout.block == 0, format!("embedding width {} must be a multiple of the {} block of {}", node.output.channels, layout.name, layout.block))?;
	checked_mul(node.output.channels / layout.block, layout.stride, "embedding row bytes").map(|row| (layout, row))
}
// A table is decoded one row at a time, from the context arena or on the host,
// so it is absent from the storage the model-load kernel expands into weights.
fn arena_weight<'a>(node: &Node, stored: &'a Option<StoredWeight>) -> Option<&'a StoredWeight> {
	stored.as_ref().filter(|_| !node.table())
}
fn node_context(node: &Node, rows: usize, element: usize) -> Result<usize> {
	let elements = match node.op {
		// One scratch row per reduction partition, holding this node's trainable
		// scalars. Programs without trainable scalars reduce nothing and take the
		// minimum allocation below.
		Primitive::Elementwise => checked_mul(checked_mul(rows, node.output.elements(), "program batch")?.min(NATIVE_SCALAR_PARTITIONS), node.parameters, "scalar gradient partials")?,
		Primitive::Predictor => checked_mul(checked_add(node.argument[0] as usize, node.argument[1] as usize, "predictor workspace")?, rows, "predictor batch")?,
		// Softmax statistics, then the indexer block representatives, then one
		// row of block scores and one admission flag per block per query, then
		// one block score gradient per attention head per query.
		Primitive::Attention => {
			let queries = checked_mul(rows, node.output.length, "attention statistics rows")?;
			let statistics = checked_mul(checked_mul(queries, node.argument[0] as usize, "attention statistics heads")?, 2, "attention statistics")?;
			let blocks = attention_blocks(node);
			let representatives = checked_mul(checked_mul(rows, blocks, "indexer block rows")?, node.argument[6] as usize, "indexer representatives")?;
			let scores = checked_mul(queries, checked_mul(blocks, 2, "indexer score row")?, "indexer scores")?;
			let derivatives = checked_mul(checked_mul(queries, node.argument[0] as usize, "indexer derivative heads")?, blocks, "indexer derivatives")?;
			checked_add(statistics, checked_add(representatives, checked_add(scores, derivatives, "indexer gradient context")?, "indexer context")?, "attention context")?
		}
		Primitive::Scan => {
			let (state_count, gates) = (checked_mul(rows, node.output.elements(), "scan batch")?, node.argument[0] as usize);
			let states = checked_mul(2 * gates + 1, state_count, "scan states")?;
			let gradients = checked_mul(rows, node.parameters, "scan gradients")?;
			checked_add(states, checked_add(gradients, 2 * rows * node.output.channels, "scan scratch")?, "scan")?
		}
		// One thread owns one row and head: the chunk entry states, the live state,
		// the chunk the reverse pass replays, the state adjoint, the readout error
		// and key weight vectors, and one decay partial.
		Primitive::Delta => {
			let (_, key_width, heads, width) = delta_extent(node).map(|(a, b, c, d)| (a as usize, b as usize, c as usize, d as usize))?;
			let (chunk, state) = ((node.argument[2] as usize).max(1), checked_mul(key_width, width, "delta state")?);
			let spans = checked_add(node.output.length.div_ceil(chunk), checked_add(chunk, 2, "delta live states")?, "delta state spans")?;
			let pair = checked_add(checked_mul(spans, state, "delta state span")?, checked_add(checked_mul(2, width, "delta vectors")?, 1, "delta decay partial")?, "delta pair context")?;
			checked_mul(checked_mul(rows, heads, "delta pairs")?, pair, "delta context")?
		}
		Primitive::Pool => return checked_mul(checked_mul(rows, node.output.elements(), "pool context")?, size_of::<u64>(), "pool context bytes"),
		// The packed embedding table is the node's persistent state: the gather
		// decodes rows out of it and never expands it into the weights.
		Primitive::Gather => return checked_mul(integer_argument(node.argument[0], "embedding vocabulary")? as usize, embedding_row(node)?.1, "embedding table bytes"),
		// The rows the host gathers for every position, staged as [row][position][channel].
		Primitive::Lookup => checked_mul(rows, node.output.elements(), "lookup staging")?,
		// One count per expert, then every routed position of every expert in
		// ascending expert and position order.
		Primitive::ExpertIn | Primitive::ExpertOut => {
			let entries = checked_mul(checked_mul(rows, node.output.length, "routed positions")?, node.argument[1] as usize, "routed slots")?;
			return checked_mul(checked_add(node.argument[0] as usize, entries, "expert bucket")?, size_of::<u32>(), "expert bucket bytes");
		}
		Primitive::Normalize => {
			let statistics = checked_mul(4, normalize_groups(node, rows)?, "normalization context")?;
			let partials = checked_mul(checked_mul(rows, node.output.elements(), "normalization batch")?.min(NATIVE_SCALAR_PARTITIONS), node.parameters, "normalization weight partials")?;
			checked_add(statistics, partials, "normalization")?
		}
		_ => 1,
	};
	checked_mul(elements.max(1), element, "context bytes")
}
fn narrow(value: usize, role: &str) -> Result<i32> {
	i32::try_from(value).map_err(|_| RecipeError::new(format!("{role} exceeds i32")))
}
struct Buffer {
	runtime: &'static Gpu,
	pointer: u64,
	bytes: usize,
}
impl Buffer {
	fn upload<T>(runtime: &'static Gpu, values: &[T]) -> Result<Self> {
		let bytes = size_of_val(values);
		Ok(Self { runtime, pointer: runtime.upload(0, values.as_ptr().cast(), bytes)?, bytes })
	}
	fn upload_float(runtime: &'static Gpu, values: &[f64], precision: Compute) -> Result<Self> {
		Self::upload(runtime, &encode_floats(values, precision))
	}
	fn write_float_bytes(&self, offset: usize, values: &[f64], precision: Compute) -> Result<()> {
		let bytes = precision.bytes();
		let encoded = values.iter().flat_map(|value| precision.pack(*value).to_le_bytes().into_iter().take(bytes)).collect::<Vec<_>>();
		self.write_bytes(offset, &encoded)
	}
	fn write_bytes(&self, offset: usize, values: &[u8]) -> Result<()> {
		require(checked_add(offset, values.len(), "GPU byte write")? <= self.bytes, "GPU byte write exceeds buffer")?;
		self.runtime.upload(self.pointer + offset as u64, values.as_ptr().cast(), values.len()).map(|_| ())
	}
	fn clear(&self) -> Result<()> {
		self.clear_range(0, self.bytes)
	}
	fn clear_range(&self, offset: usize, bytes: usize) -> Result<()> {
		require(checked_add(offset, bytes, "GPU byte clear")? <= self.bytes, "GPU byte clear exceeds buffer")?;
		self.runtime.clear(self.pointer + offset as u64, bytes)
	}
	fn download<T: Copy + Default>(&self, count: usize) -> Result<Vec<T>> {
		self.download_range(0, count)
	}
	fn download_range<T: Copy + Default>(&self, offset: usize, count: usize) -> Result<Vec<T>> {
		let start = checked_mul(offset, size_of::<T>(), "GPU read offset")?;
		let mut values = std::iter::repeat_n(T::default(), count).collect::<Vec<_>>();
		require(checked_add(start, size_of_val(&*values), "GPU read")? <= self.bytes, "GPU read exceeds buffer")?;
		self.runtime.synchronize()?;
		self.runtime.download(values.as_mut_ptr().cast(), self.pointer + start as u64, size_of_val(&*values))?;
		Ok(values)
	}
	fn download_float(&self, count: usize, precision: Compute) -> Result<Vec<f64>> {
		let bytes = precision.bytes();
		self.download::<u8>(checked_mul(count, bytes, "GPU float download")?).map(|encoded| {
			encoded
				.chunks_exact(bytes)
				.map(|chunk| {
					let mut bits = [0u8; 8];
					bits[..bytes].copy_from_slice(chunk);
					precision.unpack(u64::from_le_bytes(bits))
				})
				.collect()
		})
	}
	fn download_float_bytes(&self, offset: usize, count: usize, precision: Compute) -> Result<Vec<f64>> {
		let bytes = precision.bytes();
		let encoded = self.download_range::<u8>(offset, checked_mul(count, bytes, "GPU float byte download")?)?;
		Ok(encoded
			.chunks_exact(bytes)
			.map(|chunk| {
				let mut bits = [0_u8; 8];
				bits[..bytes].copy_from_slice(chunk);
				precision.unpack(u64::from_le_bytes(bits))
			})
			.collect())
	}
}
impl Drop for Buffer {
	fn drop(&mut self) {
		self.runtime.free(self.pointer);
	}
}
#[derive(Clone, Copy)]
struct Kernel {
	object: u64,
	shared: u32,
	element: u8,
	#[cfg(amd)]
	kernarg: usize,
	#[cfg(amd)]
	private: u32,
	layout: &'static [u8],
}
#[derive(Clone, Copy)]
struct Dispatch {
	kernel: Kernel,
	geometry: Geometry,
}
type NativeForward = unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, i32, i32, i32, i32);
type NativeModelLoad = unsafe extern "C" fn(Ptr, Ptr, i32);
type NativeCpuThread = unsafe extern "C" fn(i32, Ptr, Ptr);
type NativeEpochF64 = unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, i32, i32, f64, f64, f64, f64, f64, f64, f64, i32, i32);
type NativeEpochF32 = unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, i32, i32, f32, f32, f32, f32, f32, f32, f32, i32, i32);
type NativeEpochF16 = unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, i32, i32, i16, i16, i16, i16, i16, i16, i16, i32, i32);
type NativeEpochF8 = unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, Ptr, i32, i32, i8, i8, i8, i8, i8, i8, i8, i32, i32);

#[derive(Clone, Copy)]
enum NativeCpuEpoch {
	F64(NativeEpochF64),
	F32(NativeEpochF32),
	F16(NativeEpochF16),
	F8(NativeEpochF8),
}

#[cfg(unix)]
struct NativeCpuProgram {
	_library: Library,
	thread: NativeCpuThread,
	forward: NativeForward,
	epoch: Option<NativeCpuEpoch>,
	model_load: Option<NativeModelLoad>,
}

#[cfg(amd)]
struct HsaReader {
	handle: u64,
	destroy: unsafe extern "C" fn(u64) -> i32,
}

#[cfg(amd)]
impl Drop for HsaReader {
	fn drop(&mut self) {
		if self.handle != 0 {
			unsafe { (self.destroy)(self.handle) };
		}
	}
}

#[cfg(amd)]
struct HsaExecutable {
	handle: u64,
	destroy: unsafe extern "C" fn(u64) -> i32,
}

#[cfg(amd)]
impl Drop for HsaExecutable {
	fn drop(&mut self) {
		if self.handle != 0 {
			unsafe { (self.destroy)(self.handle) };
		}
	}
}

#[cfg(amd)]
struct NativeHsaProgram {
	executable: HsaExecutable,
	kernarg: usize,
	kernarg_size: usize,
	grid_sync: usize,
	free: unsafe extern "C" fn(Ptr) -> i32,
}

#[cfg(amd)]
const HSA_IMPLICIT_ARGUMENT_ALIGNMENT: usize = 8;
#[cfg(amd)]
const HSA_IMPLICIT_ARGUMENT_BYTES: usize = 256;
#[cfg(amd)]
const HSA_MULTIGRID_SYNC_POINTER_OFFSET: usize = 88;
#[cfg(amd)]
const HSA_GRID_SYNC_ALIGNMENT: usize = 8;
#[cfg(amd)]
const HSA_GRID_SYNC_BYTES: usize = 48;
#[cfg(amd)]
const HSA_GRID_SYNC_GROUPS_OFFSET: usize = 40;

#[cfg(nvidia)]
struct NativeCudaProgram {
	module: usize,
	unload: unsafe extern "C" fn(Ptr) -> i32,
}

#[cfg(nvidia)]
impl Drop for NativeCudaProgram {
	fn drop(&mut self) {
		if self.module != 0 {
			unsafe { (self.unload)(self.module as Ptr) };
		}
	}
}

enum NativeBackend {
	#[cfg(unix)]
	Cpu(NativeCpuProgram),
	#[cfg(amd)]
	Amd(NativeHsaProgram),
	#[cfg(nvidia)]
	Nvidia(NativeCudaProgram),
	Remote,
}

struct NativeProgram {
	gpu: &'static Gpu,
	artifact: NativeArtifact,
	backend: NativeBackend,
	forward: Dispatch,
	epoch: Option<Dispatch>,
	model_load: Option<Dispatch>,
	tile: Tile,
	contractions: Vec<Option<NativeContractionTiles>>,
	shared_values: u32,
	reduction_values: u32,
	gradient_values: usize,
}

#[cfg(amd)]
impl Drop for NativeHsaProgram {
	fn drop(&mut self) {
		if self.kernarg != 0 {
			unsafe { (self.free)(self.kernarg as Ptr) };
		}
	}
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum NativeEntry {
	Forward = 0,
	Epoch = 1,
	ModelLoad = 2,
}

fn native_symbol(name: &str) -> Vec<u8> {
	let mut bytes = name.as_bytes().to_vec();
	bytes.push(0);
	bytes
}

fn native_artifact_contract(artifact: &NativeArtifact) -> Result<()> {
	require(matches!(artifact.precision.model.bytes(), 1 | 2 | 4 | 8) && matches!(artifact.precision.state.bytes(), 4 | 8), "native artifact precision composition is unsupported")?;
	require(!artifact.artifact.is_empty(), "native artifact is empty")?;
	Ok(())
}

impl Kernel {
	const fn remote(shared: u32, element: u8, layout: &'static [u8]) -> Self {
		Self {
			object: 0,
			shared,
			element,
			#[cfg(amd)]
			kernarg: 0,
			#[cfg(amd)]
			private: 0,
			layout,
		}
	}
}
#[cfg(nvidia)]
struct Cuda {
	_runtime: std::sync::Arc<Library>,
	context: Ptr,
	set: unsafe extern "C" fn(Ptr) -> i32,
	allocate: unsafe extern "C" fn(*mut u64, usize) -> i32,
	free: unsafe extern "C" fn(u64) -> i32,
	upload: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
	download: unsafe extern "C" fn(Ptr, u64, usize) -> i32,
	clear: unsafe extern "C" fn(u64, u8, usize) -> i32,
	memory_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32,
	synchronize: unsafe extern "C" fn() -> i32,
	launch: unsafe extern "C" fn(usize, u32, u32, u32, u32, u32, u32, u32, Ptr, *mut Ptr, *mut Ptr) -> i32,
	load: unsafe extern "C" fn(*mut Ptr, *const c_void) -> i32,
	unload: unsafe extern "C" fn(Ptr) -> i32,
	function: unsafe extern "C" fn(*mut usize, Ptr, *const u8) -> i32,
	function_attribute: unsafe extern "C" fn(*mut i32, i32, usize) -> i32,
	occupancy: unsafe extern "C" fn(*mut i32, usize, i32, usize) -> i32,
	cus: u32,
	wave: u32,
	workgroup: u32,
	block_lds: u32,
	sm_lds: u32,
	registers: u32,
	threads: u32,
}
#[cfg(nvidia)]
impl Kernel {
	const fn cuda(object: usize, shared: u32, element: u8, layout: &'static [u8]) -> Self {
		Self {
			object: object as u64,
			shared,
			element,
			#[cfg(amd)]
			kernarg: 0,
			#[cfg(amd)]
			private: 0,
			layout,
		}
	}
}
#[cfg(amd)]
#[allow(dead_code)]
struct Hsa {
	_runtime: std::sync::Arc<Library>,
	reader_create: unsafe extern "C" fn(*const c_void, usize, *mut u64) -> i32,
	reader_destroy: unsafe extern "C" fn(u64) -> i32,
	executable_create: unsafe extern "C" fn(i32, i32, Ptr, *mut u64) -> i32,
	executable_destroy: unsafe extern "C" fn(u64) -> i32,
	executable_load: unsafe extern "C" fn(u64, u64, u64, Ptr, Ptr) -> i32,
	executable_freeze: unsafe extern "C" fn(u64, Ptr) -> i32,
	symbol: HsaSymbol,
	symbol_info: HsaSymbolInfo,
	info: HsaInfo,
	allocate: unsafe extern "C" fn(u64, usize, u32, *mut Ptr) -> i32,
	free: unsafe extern "C" fn(Ptr) -> i32,
	allow: unsafe extern "C" fn(u32, *const u64, *const u32, *const c_void) -> i32,
	copy: unsafe extern "C" fn(Ptr, *const c_void, usize) -> i32,
	clear: unsafe extern "C" fn(Ptr, u32, usize) -> i32,
	store: unsafe extern "C" fn(u64, i64),
	wait: unsafe extern "C" fn(u64, i32, i64, u64, i32) -> i64,
	write: unsafe extern "C" fn(*const HsaQueue, u64) -> u64,
	queue: Ptr,
	signal: u64,
	cpu_agent: u64,
	vram_pool: u64,
	kernarg_pool: u64,
	agent: u64,
	cus: u32,
	wave: u32,
	workgroup: u32,
	lds: u32,
}
const REMOTE_ALLOCATE: u8 = 1;
const REMOTE_FREE: u8 = 2;
const REMOTE_UPLOAD: u8 = 3;
const REMOTE_DOWNLOAD: u8 = 4;
const REMOTE_SYNCHRONIZE: u8 = 5;
const REMOTE_LOAD: u8 = 6;
const REMOTE_LAUNCH: u8 = 7;
const REMOTE_MEMORY: u8 = 8;
const REMOTE_CLEAR: u8 = 9;
struct Wire<R: Read, W: Write> {
	input: std::io::BufReader<R>,
	output: std::io::BufWriter<W>,
	role: &'static str,
}
impl<R: Read, W: Write> Wire<R, W> {
	fn read_error<T>(role: &str, error: std::io::Error) -> Result<T> {
		Err(RecipeError::new(format!("{role} channel: {error}")))
	}
	fn write_u8(&mut self, value: u8) -> Result<()> {
		self.output.write_all(&[value]).map_err(|error| RecipeError::new(format!("{} channel: {error}", self.role)))
	}
	fn write_u32(&mut self, value: u32) -> Result<()> {
		self.output.write_all(&value.to_le_bytes()).map_err(|error| RecipeError::new(format!("{} channel: {error}", self.role)))
	}
	fn write_u64(&mut self, value: u64) -> Result<()> {
		self.output.write_all(&value.to_le_bytes()).map_err(|error| RecipeError::new(format!("{} channel: {error}", self.role)))
	}
	fn write_bytes(&mut self, data: &[u8]) -> Result<()> {
		self.output.write_all(data).map_err(|error| RecipeError::new(format!("{} channel: {error}", self.role)))
	}
	fn flush(&mut self) -> Result<()> {
		self.output.flush().map_err(|error| RecipeError::new(format!("{} channel: {error}", self.role)))
	}
	fn read_u8(&mut self) -> Result<u8> {
		let mut bytes = [0; 1];
		self.input.read_exact(&mut bytes).or_else(|error| Self::read_error(self.role, error))?;
		Ok(bytes[0])
	}
	fn read_u32(&mut self) -> Result<u32> {
		let mut bytes = [0; 4];
		self.input.read_exact(&mut bytes).or_else(|error| Self::read_error(self.role, error))?;
		Ok(u32::from_le_bytes(bytes))
	}
	fn read_u64(&mut self) -> Result<u64> {
		let mut bytes = [0; 8];
		self.input.read_exact(&mut bytes).or_else(|error| Self::read_error(self.role, error))?;
		Ok(u64::from_le_bytes(bytes))
	}
	fn read_into(&mut self, buffer: &mut [u8]) -> Result<()> {
		self.input.read_exact(buffer).or_else(|error| Self::read_error(self.role, error))
	}
	/// Reads a status byte; a nonzero status carries the worker's error message.
	fn read_status(&mut self, action: &str) -> Result<()> {
		if self.read_u8()? == 0 {
			return Ok(());
		}
		let length = self.read_u32()? as usize;
		let mut message = vec![0_u8; length.min(4096)];
		self.read_into(&mut message)?;
		for _ in message.len()..length {
			self.read_u8()?;
		}
		Err(RecipeError::new(format!("remote {action}: {}", String::from_utf8_lossy(&message))))
	}
	fn status(&mut self, result: &Result<()>) -> Result<()> {
		match result {
			Ok(()) => self.write_u8(0),
			Err(error) => {
				let message = error.to_string();
				self.write_u8(1)?;
				self.write_u32(message.len() as u32)?;
				self.write_bytes(message.as_bytes())
			}
		}
	}
}
type RemoteChannel = Wire<std::process::ChildStdout, std::process::ChildStdin>;
struct Remote {
	channel: Mutex<RemoteChannel>,
	wave: u32,
}
enum Driver {
	Cpu,
	#[cfg(amd)]
	Hsa(Hsa),
	#[cfg(nvidia)]
	Cuda(Cuda),
	Remote(Remote),
}
#[allow(dead_code)]
struct Gpu {
	name: String,
	backend: Backend,
	native_target: BackendTarget,
	driver: Driver,
	memory: u64,
	shared_limit: u32,
	dispatch: Mutex<()>,
}
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}
fn native_target_label(target: &BackendTarget) -> &str {
	match target {
		BackendTarget::Cpu { target } => target.as_str(),
		BackendTarget::Amd { architecture } => architecture.as_str(),
		BackendTarget::Nvidia { architecture } => architecture.as_str(),
	}
}
#[cfg(amd)]
#[repr(C)]
struct HsaQueue {
	kind: u32,
	features: u32,
	base: Ptr,
	doorbell: u64,
	size: u32,
	reserved: u32,
	id: u64,
}
#[cfg(amd)]
#[repr(C)]
struct HsaPacket {
	header: u16,
	setup: u16,
	workgroup_x: u16,
	workgroup_y: u16,
	workgroup_z: u16,
	reserved0: u16,
	grid_x: u32,
	grid_y: u32,
	grid_z: u32,
	private: u32,
	group: u32,
	object: u64,
	kernarg: Ptr,
	reserved1: u64,
	completion: u64,
}
#[cfg(nvidia)]
type NvQuery = unsafe extern "C" fn(*mut i32, i32, i32) -> i32;
#[cfg(any(unix, all(nvidia, windows)))]
struct Library(usize);
#[cfg(any(unix, all(nvidia, windows)))]
impl Library {
	fn open(name: &str) -> Result<Self> {
		let name = format!("{name}\0");
		let handle = unsafe { dlopen(name.as_ptr().cast(), 2) };
		require(!handle.is_null(), format!("cannot load {name:?}"))?;
		Ok(Self(handle as usize))
	}
	fn function<F: Copy>(&self, name: &[u8]) -> Result<F> {
		let pointer = unsafe { dlsym(self.0 as Ptr, name.as_ptr().cast()) };
		require(!pointer.is_null(), format!("runtime symbol {:?} is absent", name))?;
		Ok(unsafe { std::mem::transmute_copy(&pointer) })
	}
}

#[cfg(any(unix, all(nvidia, windows)))]
impl Drop for Library {
	fn drop(&mut self) {
		unsafe {
			#[cfg(unix)]
			dlclose(self.0 as Ptr);
			#[cfg(all(nvidia, windows))]
			FreeLibrary(self.0 as Ptr);
		}
	}
}

#[cfg(unix)]
fn load_native_cpu(artifact: &NativeArtifact) -> Result<NativeCpuProgram> {
	let path = artifact.path.to_str().ok_or_else(|| RecipeError::new("CPU native artifact path is not UTF-8"))?;
	let library = Library::open(path)?;
	let thread = library.function::<NativeCpuThread>(&native_symbol(NATIVE_CPU_THREAD_SYMBOL))?;
	let forward = library.function::<NativeForward>(&native_symbol(NATIVE_FORWARD_SYMBOL))?;
	let epoch = || -> Result<NativeCpuEpoch> {
		match artifact.precision.state.bytes() {
			8 => library.function::<NativeEpochF64>(&native_symbol(NATIVE_EPOCH_SYMBOL)).map(NativeCpuEpoch::F64),
			4 => library.function::<NativeEpochF32>(&native_symbol(NATIVE_EPOCH_SYMBOL)).map(NativeCpuEpoch::F32),
			2 => library.function::<NativeEpochF16>(&native_symbol(NATIVE_EPOCH_SYMBOL)).map(NativeCpuEpoch::F16),
			1 => library.function::<NativeEpochF8>(&native_symbol(NATIVE_EPOCH_SYMBOL)).map(NativeCpuEpoch::F8),
			_ => Err(RecipeError::new("native CPU precision width is invalid")),
		}
	};
	let epoch = artifact.training.then(epoch).transpose()?;
	let model_load = (!artifact.storage.is_empty()).then(|| library.function::<NativeModelLoad>(&native_symbol(NATIVE_MODEL_LOAD_SYMBOL))).transpose()?;
	Ok(NativeCpuProgram { _library: library, thread, forward, epoch, model_load })
}

#[cfg(any(amd, nvidia))]
fn driver_status(backend: Backend, status: i32, action: &str) -> Result<()> {
	(status == 0).then_some(()).ok_or_else(|| RecipeError::new(format!("{backend:?} {action} failed: {status}")))
}
impl Gpu {
	#[cfg(any(amd, nvidia))]
	fn status(&self, status: i32, action: &str) -> Result<()> {
		driver_status(self.backend, status, action).map_err(|error| RecipeError::new(format!("device {} {:?}: {error}", self.name, self.backend)))
	}
	fn activate(&self) -> Result<()> {
		match &self.driver {
			Driver::Cpu | Driver::Remote(_) => Ok(()),
			#[cfg(nvidia)]
			Driver::Cuda(driver) => self.status(unsafe { (driver.set)(driver.context) }, "context"),
			#[cfg(amd)]
			Driver::Hsa(_) => Ok(()),
		}
	}
	fn native_program(&'static self, graph: &Graph, rows: usize, precision: Compute, loss: Option<LossFunction>) -> Result<NativeProgram> {
		let vector_waves = if matches!(&self.driver, Driver::Cpu) {
			1
		} else {
			narrow(natural("contraction resident waves per workgroup", env!("RECIPE_CONTRACTION_RESIDENT_WAVES_PER_WORKGROUP"))?, "contraction resident waves per workgroup")? as u32
		};
		let shared_values = if matches!(&self.driver, Driver::Cpu) {
			narrow(natural("CPU contraction shared values", env!("RECIPE_CONTRACTION_CPU_SHARED_VALUES"))?, "CPU contraction shared values")? as u32
		} else {
			self.shared_limit / precision.bytes() as u32
		};
		let shapes = native_contraction_shapes(graph, rows)?;
		let mut limits = Tile { m: 1, n: 1, k: 1 };
		let mut dominant = None;
		for (index, shape) in shapes.iter().enumerate().filter_map(|(index, shape)| shape.map(|shape| (index, shape))) {
			for direction in [shape.forward, shape.gradient, shape.previous] {
				limits.m = limits.m.max(direction.m);
				limits.n = limits.n.max(direction.n);
				limits.k = limits.k.max(direction.k);
			}
			let work = checked_mul(checked_mul(shape.gradient.m as usize, shape.gradient.n as usize, "native contraction output work")?, shape.gradient.k as usize, "native contraction work")?;
			if dominant.is_none_or(|(_, best)| work > best) {
				dominant = Some((index, work))
			}
		}
		let wave = match &self.driver {
			Driver::Cpu => 1,
			#[cfg(amd)]
			Driver::Hsa(driver) => driver.wave,
			#[cfg(nvidia)]
			Driver::Cuda(driver) => driver.wave,
			Driver::Remote(remote) => remote.wave,
		};
		let dominant_shape = dominant.and_then(|(index, _)| shapes[index]).map_or(limits, |shape| shape.gradient);
		let fragment_k = narrow(natural("contraction fragment K", env!("RECIPE_CONTRACTION_FRAGMENT_K"))?, "contraction fragment K")? as u32;
		let aligned_attention = graph.nodes.iter().filter(|node| node.op == Primitive::Attention).try_fold(true, |aligned, node| {
			let heads = integer_argument(node.argument[0], "attention heads")?;
			require(heads != 0, "attention heads are empty")?;
			// The matrix path covers dense attention with tied heads, no gate and
			// no indexer.
			let plain = node.argument[1] == node.argument[0] && node.argument[2] == 0.0 && node.argument[3] == 0.0;
			Ok::<_, RecipeError>(aligned && plain && node.output.channels / heads as usize % fragment_k as usize == 0)
		})?;
		let matrix = matches!(&self.native_target, BackendTarget::Amd { architecture } if architecture.starts_with("gfx11") || architecture.starts_with("gfx12"))
			&& [Compute::FP16, Compute::BF16, Compute::INT8, Compute::INT4].contains(&precision)
			&& dominant_shape.m >= fragment_k
			&& dominant_shape.n >= fragment_k
			&& dominant_shape.k >= fragment_k
			&& aligned_attention;
		let matrix_waves =
			narrow(natural("contraction matrix maximum waves per workgroup", env!("RECIPE_CONTRACTION_MATRIX_MAX_WAVES_PER_WORKGROUP"))?, "contraction matrix maximum waves per workgroup")?
				as u32;
		let waves_per_workgroup = if matrix { matrix_waves.min(dominant_shape.m.div_ceil(fragment_k)).max(1) } else { vector_waves };
		// The reduction chunk is a multiple of the staging fragment so a chunk
		// boundary never falls inside a vector staging load.
		let chunk_k = narrow(natural("contraction chunk K", env!("RECIPE_CONTRACTION_CHUNK_K"))?, "contraction chunk K")? as u32;
		require(chunk_k % fragment_k == 0, "contraction chunk K must be a multiple of the staging fragment")?;
		let register_m = (narrow(natural("contraction register M", env!("RECIPE_CONTRACTION_REGISTER_M"))?, "contraction register M")? as u32).min(limits.m);
		let waves = if self.backend == Backend::Amd { waves_per_workgroup } else { 1 };
		let block = wave.checked_mul(waves).ok_or_else(|| RecipeError::new("native contraction workgroup overflows"))?;
		let register_n = (narrow(natural("contraction register N", env!("RECIPE_CONTRACTION_REGISTER_N"))?, "contraction register N")? as u32).min(limits.n).min((self.shared_limit
			/ precision.bytes() as u32
			/ block / register_m
			.checked_add(1)
			.ok_or_else(|| RecipeError::new("native contraction register width overflows"))?)
		.max(1));
		// A cooperative grid deadlocks unless every workgroup is resident, so the
		// tile must leave local memory unclaimed for the waves that share a compute
		// unit. Local memory is allocated per workgroup rather than per wave, so
		// this divisor is a margin and not the exact resource equation: it is the
		// wave count because that is the multiple by which the workgroup was
		// widened. Claiming the whole local store deadlocks even at one wave,
		// because the kernel's own fixed allocation shares the same store.
		let shared_budget = shared_values / waves;
		// Chunk partials keep the arithmetic width while the tile allocation is
		// counted in model elements, so a narrow model needs proportionally more
		// elements per partial value.
		let ratio = narrow(NativePrecision::new(precision)?.state.bytes().div_ceil(precision.bytes()), "native contraction state ratio")? as u32;
		let mut extent = native_contraction_tile(dominant_shape, register_m, register_n, block, shared_budget, chunk_k, ratio, matrix)?;
		let contractions = shapes
			.iter()
			.map(|shape| {
				shape.map(|shape| {
					Ok(NativeContractionTiles {
						forward: native_contraction_tile(shape.forward, register_m, register_n, block, shared_budget, chunk_k, ratio, matrix)?,
						gradient: native_contraction_tile(shape.gradient, register_m, register_n, block, shared_budget, chunk_k, ratio, matrix)?,
						previous: native_contraction_tile(shape.previous, register_m, register_n, block, shared_budget, chunk_k, ratio, matrix)?,
						gradient_shape: shape.gradient,
						parameters: shape.parameters,
					})
				})
				.transpose()
			})
			.collect::<Result<Vec<_>>>()?;
		extent = dominant.and_then(|(index, _)| contractions[index]).map_or(extent, |contraction| contraction.gradient);
		let contraction_shared_values = contractions
			.iter()
			.flatten()
			.flat_map(|contraction| [contraction.forward, contraction.gradient, contraction.previous])
			.map(|extent| native_contraction_shared_values(extent, register_m, register_n, block, chunk_k, ratio, matrix))
			.collect::<Result<Vec<_>>>()?
			.into_iter()
			.max()
			.unwrap_or(1);
		let attention_query_tile = narrow(natural("attention query tile", env!("RECIPE_ATTENTION_QUERY_TILE"))?, "attention query tile")? as u32;
		let attention = native_attention_tiles(graph, shared_budget, attention_query_tile)?;
		let attention_shared_values = attention
			.iter()
			.enumerate()
			.filter_map(|(index, extent)| extent.map(|extent| native_attention_shared_values(extent, extent.m as usize == graph.nodes[index].output.length)))
			.collect::<Result<Vec<_>>>()?
			.into_iter()
			.max()
			.unwrap_or(1);
		let shared_values = contraction_shared_values.max(attention_shared_values);
		let register_count = register_m.checked_mul(register_n).ok_or_else(|| RecipeError::new("native contraction register tile overflows"))?;
		let register_values =
			register_count.checked_add(register_n).and_then(|values| values.checked_mul(ratio)).ok_or_else(|| RecipeError::new("native contraction register reduction overflows"))?;
		// The private chunk buffer holds every partial a single lane can own at
		// once. A full tile with one k lane folds locally and reuses one slot; the
		// exchange only ever runs with at least two k lanes, and tails only shrink
		// the output lanes and so grow the k lanes, so the full-tile lane count
		// bounds how many chunks a lane can hold.
		let mut owned = 1_u32;
		for extent in contractions.iter().flatten().flat_map(|contraction| [contraction.forward, contraction.gradient, contraction.previous]) {
			let output_lanes = (extent.m / register_m).max(1).checked_mul((extent.n / register_n).max(1)).ok_or_else(|| RecipeError::new("native contraction lane count overflows"))?;
			let k_lanes = (block / output_lanes).max(2);
			owned = owned.max(extent.k.div_ceil(chunk_k).div_ceil(k_lanes));
		}
		let chunk_values = owned.checked_mul(register_count).ok_or_else(|| RecipeError::new("native contraction chunk buffer overflows"))?;
		let chunk_bias_values = owned.checked_mul(register_n).ok_or_else(|| RecipeError::new("native contraction chunk bias buffer overflows"))?;
		let scratch_base = narrow(graph.parameters.len().next_multiple_of(NATIVE_SCRATCH_ROW_VALUES), "native gradient scratch base")?;
		debug(&format!("native schedule block={block} waves={waves} registers={register_count} shared={shared_values} contractions={contractions:?} attention={attention:?}"))?;
		let schedule = NativeSchedule {
			matrix,
			block,
			tile: extent,
			register_m,
			register_n,
			register_count,
			fragment_k,
			chunk_k,
			chunk_values,
			chunk_bias_values,
			scratch_base,
			shared_values,
			contractions,
			attention,
		};
		let artifact = compile_model(&self.native_target, graph, precision, loss, rows, schedule.clone())?;
		let program = NativeProgram::load(self, artifact, graph, schedule, register_values, waves)?;
		let fixed = [Some(program.forward), program.epoch, program.model_load].into_iter().flatten().map(|dispatch| dispatch.kernel.shared).max().unwrap_or(0);
		let required = fixed
			.checked_add(shared_values.max(program.reduction_values).checked_mul(precision.bytes() as u32).ok_or_else(|| RecipeError::new("native model shared memory overflows"))?)
			.ok_or_else(|| RecipeError::new("native model shared memory overflows"))?;
		require(required <= self.shared_limit, "native model exceeds resident device shared memory")?;
		Ok(program)
	}
	fn allocate(&self, bytes: usize) -> Result<u64> {
		self.activate()?;
		unsafe {
			match &self.driver {
				Driver::Cpu => {
					let size = checked_add(bytes.max(1), size_of::<usize>(), "CPU allocation")?;
					let layout = std::alloc::Layout::from_size_align(size, 8).map_err(|error| RecipeError::new(format!("CPU allocation layout is invalid: {error}")))?;
					let base = std::alloc::alloc_zeroed(layout);
					require(!base.is_null(), "CPU allocation failed")?;
					base.cast::<usize>().write(size);
					Ok(base.add(size_of::<usize>()) as u64)
				}
				#[cfg(nvidia)]
				Driver::Cuda(driver) => {
					let mut pointer = 0;
					self.status((driver.allocate)(&mut pointer, bytes), "allocation")?;
					Ok(pointer)
				}
				#[cfg(amd)]
				Driver::Hsa(driver) => {
					let mut pointer = ptr::null_mut();
					self.status((driver.allocate)(driver.vram_pool, bytes, 0, &mut pointer), "allocation")?;
					self.status((driver.allow)(1, &driver.cpu_agent, ptr::null(), pointer), "CPU allocation access")?;
					Ok(pointer as u64)
				}
				Driver::Remote(remote) => {
					let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
					channel.write_u8(REMOTE_ALLOCATE)?;
					channel.write_u64(bytes as u64)?;
					channel.flush()?;
					channel.read_status("allocation")?;
					channel.read_u64()
				}
			}
		}
	}
	#[cfg_attr(not(any(amd, nvidia)), allow(unused_unsafe))]
	fn free(&self, pointer: u64) {
		unsafe {
			match &self.driver {
				Driver::Cpu => {
					let base = (pointer as *mut u8).sub(size_of::<usize>());
					let size = base.cast::<usize>().read();
					std::alloc::dealloc(base, std::alloc::Layout::from_size_align_unchecked(size, 8))
				}
				#[cfg(nvidia)]
				Driver::Cuda(driver) => {
					(driver.set)(driver.context);
					(driver.free)(pointer);
				}
				#[cfg(amd)]
				Driver::Hsa(driver) => {
					(driver.free)(pointer as Ptr);
				}
				Driver::Remote(remote) => {
					if let Ok(mut channel) = remote.channel.lock() {
						channel.write_u8(REMOTE_FREE).and_then(|_| channel.write_u64(pointer)).and_then(|_| channel.flush()).ok();
					}
				}
			}
		}
	}
	#[cfg_attr(not(any(amd, nvidia)), allow(unused_unsafe))]
	fn upload(&self, dst: u64, src: *const c_void, bytes: usize) -> Result<u64> {
		self.activate()?;
		let dst = if dst == 0 { self.allocate(bytes)? } else { dst };
		unsafe {
			match &self.driver {
				Driver::Cpu => {
					ptr::copy_nonoverlapping(src.cast::<u8>(), dst as *mut u8, bytes);
					Ok(dst)
				}
				#[cfg(nvidia)]
				Driver::Cuda(driver) => self.status((driver.upload)(dst, src, bytes), "upload").map(|_| dst),
				#[cfg(amd)]
				Driver::Hsa(driver) => self.status((driver.copy)(dst as Ptr, src, bytes), "upload").map(|_| dst),
				Driver::Remote(remote) => {
					let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
					channel.write_u8(REMOTE_UPLOAD)?;
					channel.write_u64(dst)?;
					channel.write_u64(bytes as u64)?;
					channel.write_bytes(std::slice::from_raw_parts(src.cast::<u8>(), bytes))?;
					channel.flush()?;
					channel.read_status("upload").map(|_| dst)
				}
			}
		}
	}
	#[cfg_attr(not(any(amd, nvidia)), allow(unused_unsafe))]
	fn clear(&self, pointer: u64, bytes: usize) -> Result<()> {
		self.activate()?;
		unsafe {
			match &self.driver {
				Driver::Cpu => {
					ptr::write_bytes(pointer as *mut u8, 0, bytes);
					Ok(())
				}
				#[cfg(nvidia)]
				Driver::Cuda(driver) => self.status((driver.clear)(pointer, 0, bytes), "clear"),
				#[cfg(amd)]
				Driver::Hsa(driver) => {
					let words = bytes / size_of::<u32>();
					self.status((driver.clear)(pointer as Ptr, 0, words), "clear")?;
					let tail = bytes - words * size_of::<u32>();
					if tail != 0 {
						let zero = [0_u8; size_of::<u32>() - 1];
						self.status((driver.copy)((pointer as usize + words * size_of::<u32>()) as Ptr, zero.as_ptr().cast(), tail), "clear")?;
					}
					Ok(())
				}
				Driver::Remote(remote) => {
					let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
					channel.write_u8(REMOTE_CLEAR)?;
					channel.write_u64(pointer)?;
					channel.write_u64(bytes as u64)?;
					channel.flush()?;
					channel.read_status("clear")
				}
			}
		}
	}
	#[cfg_attr(not(any(amd, nvidia)), allow(unused_unsafe))]
	fn download(&self, dst: Ptr, src: u64, bytes: usize) -> Result<()> {
		self.activate()?;
		unsafe {
			match &self.driver {
				Driver::Cpu => {
					ptr::copy_nonoverlapping(src as *const u8, dst.cast::<u8>(), bytes);
					Ok(())
				}
				#[cfg(nvidia)]
				Driver::Cuda(cuda) => self.status((cuda.download)(dst, src, bytes), "download"),
				#[cfg(amd)]
				Driver::Hsa(driver) => self.status((driver.copy)(dst, src as *const c_void, bytes), "download"),
				Driver::Remote(remote) => {
					let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
					channel.write_u8(REMOTE_DOWNLOAD)?;
					channel.write_u64(src)?;
					channel.write_u64(bytes as u64)?;
					channel.flush()?;
					channel.read_status("download")?;
					channel.read_into(std::slice::from_raw_parts_mut(dst.cast::<u8>(), bytes))
				}
			}
		}
	}
	/// The memory the device has free now. The host is not bounded by device
	/// memory, so the CPU reports its whole capacity.
	#[cfg_attr(not(any(amd, nvidia)), allow(unused_unsafe))]
	fn free_bytes(&self) -> Result<u64> {
		self.activate()?;
		unsafe {
			match &self.driver {
				Driver::Cpu => Ok(self.memory),
				#[cfg(nvidia)]
				Driver::Cuda(driver) => {
					let (mut free, mut total) = (0, 0);
					self.status((driver.memory_info)(&mut free, &mut total), "free memory")?;
					Ok(free as u64)
				}
				#[cfg(amd)]
				Driver::Hsa(driver) => {
					let mut free = 0_u64;
					self.status((driver.info)(driver.agent, 0xA011, (&mut free as *mut u64).cast()), "free memory")?;
					Ok(free)
				}
				Driver::Remote(remote) => {
					let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
					channel.write_u8(REMOTE_MEMORY)?;
					channel.flush()?;
					channel.read_status("free memory")?;
					channel.read_u64()
				}
			}
		}
	}
	#[cfg_attr(not(any(amd, nvidia)), allow(unused_unsafe))]
	fn synchronize(&self) -> Result<()> {
		self.activate()?;
		unsafe {
			match &self.driver {
				Driver::Cpu => Ok(()),
				#[cfg(nvidia)]
				Driver::Cuda(driver) => self.status((driver.synchronize)(), "synchronization"),
				#[cfg(amd)]
				Driver::Hsa(driver) => require((driver.wait)(driver.signal, 0, 0, u64::MAX, 1) == 0, "AMD synchronization failed"),
				Driver::Remote(remote) => {
					let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
					channel.write_u8(REMOTE_SYNCHRONIZE)?;
					channel.flush()?;
					channel.read_status("synchronization")
				}
			}
		}
	}
}
static DEVICES: OnceLock<Result<Vec<Gpu>>> = OnceLock::new();
fn cpu_worker_threads() -> Result<u32> {
	let limit = count("CPU worker threads", env!("RECIPE_CPU_WORKER_THREADS"))?;
	let available = std::thread::available_parallelism().map_err(|error| RecipeError::new(format!("cannot read available CPU parallelism: {error}")))?.get();
	u32::try_from(if limit == 0 { available } else { available.min(limit) }).map_err(|_| RecipeError::new("CPU worker threads exceed u32"))
}
fn cpu_device() -> Result<Gpu> {
	Ok(Gpu { name: "cpu".to_owned(), backend: Backend::Cpu, native_target: native_cpu_target()?, driver: Driver::Cpu, memory: u64::MAX, shared_limit: u32::MAX, dispatch: Mutex::new(()) })
}
/// The local device names `RECIPE_DEVICE` selects, without this host's prefix.
/// `None` selects the whole machine, so an unnamed run still sees every device.
fn device_selection() -> Result<Option<Vec<String>>> {
	let Ok(selection) = std::env::var("RECIPE_DEVICE") else { return Ok(None) };
	let prefix = format!("{}:", local_host()?);
	Ok(Some(selection.split(',').map(|name| name.strip_prefix(&prefix).unwrap_or(name).to_owned()).collect()))
}
fn devices() -> Result<&'static [Gpu]> {
	DEVICES
		.get_or_init(|| {
			if std::env::var_os("RECIPE_FORCE_CPU").is_some() {
				return cpu_device().map(|gpu| vec![gpu]);
			}
			let selection = device_selection()?;
			let mut found = Vec::new();
			let mut errors = Vec::new();
			for load in [load_amd as fn(Option<&[String]>) -> Result<Vec<Gpu>>, load_nvidia] {
				match load(selection.as_deref()) {
					Ok(mut devices) => found.append(&mut devices),
					Err(error) => errors.push(error.to_string()),
				}
			}
			// A selection can name only another host, so an empty local accelerator
			// list is an error only when no selection explains it.
			if found.is_empty() && cfg!(any(amd, nvidia)) && selection.is_none() {
				return Err(RecipeError::new(errors.join("; ")));
			}
			// The CPU is always selectable, after the accelerators, so a placement
			// can end on the host.
			found.push(cpu_device()?);
			Ok(found)
		})
		.as_ref()
		.map(Vec::as_slice)
		.map_err(Clone::clone)
}
fn device(name: Option<&str>) -> Result<&'static Gpu> {
	let found = devices()?;
	if let Some(name) = name {
		return found.iter().find(|gpu| gpu.name == name).ok_or_else(|| RecipeError::new(format!("GPU {name:?} is absent")));
	}
	require(found.iter().filter(|gpu| !matches!(gpu.backend, Backend::Cpu)).count() <= 1, "multiple GPUs require named selection")?;
	Ok(&found[0])
}
fn local_host() -> Result<String> {
	let output = Command::new("hostname").output().map_err(|error| RecipeError::new(format!("cannot read hostname: {error}")))?;
	require(output.status.success(), "cannot read hostname")?;
	let host = String::from_utf8(output.stdout).map_err(|error| RecipeError::new(format!("cannot read hostname: {error}")))?;
	Ok(host.trim().to_owned())
}
static SELECTED: OnceLock<Result<Vec<&'static Gpu>>> = OnceLock::new();
/// Resolves the `RECIPE_DEVICE` selection to the ordered device list. Each
/// comma-separated name is a local device (`amd0`, `engi:amd0`) or a device on
/// a reachable host (`benji:nv0`); the first name is the primary device.
fn selected_gpus() -> Result<&'static [&'static Gpu]> {
	SELECTED
		.get_or_init(|| {
			let Some(selection) = std::env::var("RECIPE_DEVICE").ok() else { return device(None).map(|gpu| vec![gpu]) };
			let (host, local_only) = (local_host()?, Config::load()?.multi_device == MultiDevice::Local);
			let mut selected = Vec::new();
			// `multi-device = false` trains on the local device, so a wider
			// selection never connects to, allocates on, or executes on another.
			for name in selection.split(',').take(if local_only { 1 } else { usize::MAX }) {
				let gpu = match devices()?.iter().find(|gpu| gpu.name == name || format!("{host}:{}", gpu.name) == name) {
					Some(gpu) => gpu,
					None => match name.split_once(':') {
						Some((remote, device)) if remote != host && !local_only => connect_remote(remote, device, name)?,
						_ => return Err(RecipeError::new(format!("GPU {name:?} is absent"))),
					},
				};
				require(!selected.iter().any(|previous: &&Gpu| ptr::eq(*previous, gpu)), format!("GPU {name:?} is selected twice"))?;
				selected.push(gpu);
			}
			require(!selected.is_empty(), "RECIPE_DEVICE selects no device")?;
			Ok(selected)
		})
		.as_ref()
		.map(Vec::as_slice)
		.map_err(Clone::clone)
}
fn selected_gpu() -> Result<&'static Gpu> {
	selected_gpus().map(|gpus| gpus[0])
}
struct RemoteDirectory {
	host: String,
	path: String,
}
impl Drop for RemoteDirectory {
	fn drop(&mut self) {
		Command::new("ssh").args(["-o", "BatchMode=yes", &self.host, &format!("rm -rf -- {}", self.path)]).status().ok();
	}
}
fn command_output(command: &mut Command, action: &str) -> Result<Vec<u8>> {
	let output = command.output().map_err(|error| RecipeError::new(format!("cannot {action}: {error}")))?;
	require(output.status.success(), format!("cannot {action}: {}", String::from_utf8_lossy(&output.stderr)))?;
	Ok(output.stdout)
}
fn remote_directory(host: &str) -> Result<RemoteDirectory> {
	let mut command = Command::new("ssh");
	command.args(["-o", "BatchMode=yes", host, "umask 077; mktemp -d /tmp/recipe.XXXXXXXX"]);
	let output = command_output(&mut command, &format!("create a private worker directory on {host}"))?;
	let path = String::from_utf8(output).map_err(|error| RecipeError::new(format!("worker directory from {host} is invalid: {error}")))?.trim().to_owned();
	require(
		path.starts_with("/tmp/recipe.") && path.len() == "/tmp/recipe.".len() + 8 && path.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"/._-".contains(&byte)),
		format!("worker directory from {host} is unsafe: {path:?}"),
	)?;
	Ok(RemoteDirectory { host: host.to_owned(), path })
}
/// Uploads this build's `recipe` binary to the remote host, starts it as a
/// device worker over SSH, and wraps the probed device as a local `Gpu` whose
/// driver speaks the worker protocol.
fn connect_remote(host: &str, device_name: &str, canonical: &str) -> Result<&'static Gpu> {
	static REMOTES: Mutex<Vec<&'static Gpu>> = Mutex::new(Vec::new());
	for (kind, value) in [("host", host), ("device", device_name)] {
		require(!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte)), format!("remote {kind} name is unsafe: {value:?}"))?;
	}
	let mut remotes = REMOTES.lock().map_err(|_| RecipeError::new("remote registry is poisoned"))?;
	if let Some(gpu) = remotes.iter().find(|gpu| gpu.name == canonical) {
		return Ok(gpu);
	}
	let binary = std::env::var_os("RECIPE_BINARY").map(PathBuf::from).ok_or_else(|| RecipeError::new(format!("GPU {canonical:?} requires the recipe launcher to reach host {host:?}")))?;
	require(binary.is_file(), format!("recipe binary is absent at {}", binary.display()))?;
	let directory = remote_directory(host)?;
	let remote_path = format!("{}/recipe", directory.path);
	let copy = Command::new("scp")
		.args(["-q", "-o", "BatchMode=yes"])
		.arg(&binary)
		.arg(format!("{host}:{remote_path}"))
		.status()
		.map_err(|error| RecipeError::new(format!("cannot copy the worker to {host}: {error}")))?;
	require(copy.success(), format!("cannot copy the worker to {host}: {copy}"))?;
	let mut local_hash = Command::new("sha256sum");
	local_hash.arg(&binary);
	let local_hash = command_output(&mut local_hash, "hash the local worker")?;
	let mut remote_hash = Command::new("ssh");
	remote_hash.args(["-o", "BatchMode=yes", host, &format!("sha256sum {remote_path}")]);
	let remote_hash = command_output(&mut remote_hash, &format!("hash the copied worker on {host}"))?;
	require(local_hash.split(|byte| byte.is_ascii_whitespace()).next() == remote_hash.split(|byte| byte.is_ascii_whitespace()).next(), format!("copied worker hash differs on {host}"))?;
	let mut child = Command::new("ssh")
		.args(["-o", "BatchMode=yes", host, &format!("{remote_path} --worker {device_name}")])
		.stdin(std::process::Stdio::piped())
		.stdout(std::process::Stdio::piped())
		.spawn()
		.map_err(|error| RecipeError::new(format!("cannot start the worker on {host}: {error}")))?;
	let input = child.stdin.take().ok_or_else(|| RecipeError::new("remote worker stdin is absent"))?;
	let output = child.stdout.take().ok_or_else(|| RecipeError::new("remote worker stdout is absent"))?;
	std::thread::spawn(move || child.wait());
	let mut channel = RemoteChannel { input: std::io::BufReader::new(output), output: std::io::BufWriter::new(input), role: "remote" };
	channel.read_status(&format!("worker on {host}"))?;
	let backend = match channel.read_u8()? {
		1 => Backend::Amd,
		2 => Backend::Nvidia,
		byte => return Err(RecipeError::new(format!("remote worker reported unknown backend {byte}"))),
	};
	let mut architecture = vec![0_u8; channel.read_u8()? as usize];
	channel.read_into(&mut architecture)?;
	let architecture = String::from_utf8(architecture).map_err(|error| RecipeError::new(format!("remote architecture is invalid: {error}")))?;
	let memory = channel.read_u64()?;
	let shared_limit = channel.read_u32()?;
	let wave = channel.read_u32()?;
	drop(directory);
	let native_target = match backend {
		Backend::Amd => BackendTarget::Amd { architecture },
		_ => BackendTarget::Nvidia { architecture },
	};
	let gpu = Box::leak(Box::new(Gpu {
		name: canonical.to_owned(),
		backend,
		native_target,
		driver: Driver::Remote(Remote { channel: Mutex::new(channel), wave }),
		memory,
		shared_limit,
		dispatch: Mutex::new(()),
	}));
	remotes.push(gpu);
	Ok(gpu)
}
#[cfg(amd)]
type HsaInfo = unsafe extern "C" fn(u64, i32, Ptr) -> i32;
#[cfg(amd)]
struct HsaQuery {
	info: HsaInfo,
	attribute: i32,
	expected: u32,
	secondary: i32,
	mask: u32,
	found: u64,
}
#[cfg(amd)]
extern "C" fn collect_hsa(handle: u64, pointer: Ptr) -> i32 {
	unsafe {
		let query = &mut *pointer.cast::<HsaQuery>();
		let mut value = 0;
		let mut status = (query.info)(handle, query.attribute, (&mut value as *mut u32).cast());
		if status != 0 || value != query.expected {
			return status;
		}
		if query.secondary >= 0 {
			status = (query.info)(handle, query.secondary, (&mut value as *mut u32).cast());
			if status != 0 || value & query.mask == 0 {
				return status;
			}
		}
		if query.found == 0 {
			query.found = handle;
		}
		0
	}
}
#[cfg(amd)]
struct HsaGpuQuery {
	info: HsaInfo,
	found: Vec<u64>,
}
#[cfg(amd)]
extern "C" fn collect_discrete_hsa(handle: u64, pointer: Ptr) -> i32 {
	unsafe {
		let query = &mut *pointer.cast::<HsaGpuQuery>();
		let mut device = 0_u32;
		let mut status = (query.info)(handle, 17, (&mut device as *mut u32).cast());
		if status != 0 || device != 1 {
			return status;
		}
		let mut properties = 0_u64;
		status = (query.info)(handle, 0xA114, (&mut properties as *mut u64).cast());
		if status != 0 || properties & 1 != 0 {
			return status;
		}
		query.found.push(handle);
		0
	}
}
#[cfg(amd)]
type HsaSymbol = unsafe extern "C" fn(u64, *const u8, *const u64, *mut u64) -> i32;
#[cfg(amd)]
type HsaSymbolInfo = unsafe extern "C" fn(u64, i32, Ptr) -> i32;
#[cfg(amd)]
unsafe fn hsa_kernel(symbol: HsaSymbol, info: HsaSymbolInfo, executable: u64, agent: u64, name: &std::ffi::CStr, element: u8, layout: &'static [u8]) -> Result<Kernel> {
	let mut handle = 0;
	driver_status(Backend::Amd, unsafe { symbol(executable, name.as_ptr().cast(), &agent, &mut handle) }, "kernel lookup")?;
	let mut kernel = Kernel { object: 0, shared: 0, element, kernarg: 0, private: 0, layout };
	for (attribute, output) in [
		(22, (&mut kernel.object as *mut u64).cast()),
		(11, (&mut kernel.kernarg as *mut usize).cast()),
		(13, (&mut kernel.shared as *mut u32).cast()),
		(14, (&mut kernel.private as *mut u32).cast()),
	] {
		driver_status(Backend::Amd, unsafe { info(handle, attribute, output) }, "kernel metadata")?;
	}
	Ok(kernel)
}
#[cfg(amd)]
fn kfd_property(text: &str, name: &str) -> Result<u32> {
	text.lines()
		.find_map(|line| line.split_once(' ').filter(|value| value.0 == name))
		.ok_or_else(|| RecipeError::new(format!("KFD property {name:?} is absent")))?
		.1
		.parse::<u32>()
		.map_err(|error| RecipeError::new(format!("KFD property {name:?} is invalid: {error}")))
}
#[cfg(amd)]
impl Hsa {
	unsafe fn native_dispatch(&self, executable: u64, element: u8, waves: u32, name: &str, layout: &'static [u8]) -> Result<Dispatch> {
		unsafe {
			let name = std::ffi::CString::new(format!("{name}.kd")).map_err(|error| RecipeError::new(format!("AMD native symbol is invalid: {error}")))?;
			let kernel = hsa_kernel(self.symbol, self.symbol_info, executable, self.agent, &name, element, layout)?;
			let geometry = amd(self.cus, self.wave, self.workgroup, self.lds, waves, Resources { shared: kernel.shared, max_block: self.workgroup })?;
			Ok(Dispatch { kernel, geometry })
		}
	}

	unsafe fn load_native(
		&self, bytes: &[u8], element: u8, epoch_layout: &'static [u8], training: bool, has_storage: bool, waves: u32,
	) -> Result<(NativeHsaProgram, Dispatch, Option<Dispatch>, Option<Dispatch>)> {
		unsafe {
			require(!bytes.is_empty(), "native AMD artifact is empty")?;
			let mut reader = HsaReader { handle: 0, destroy: self.reader_destroy };
			let mut executable = HsaExecutable { handle: 0, destroy: self.executable_destroy };
			driver_status(Backend::Amd, (self.reader_create)(bytes.as_ptr().cast(), bytes.len(), &mut reader.handle), "native code-object reader")?;
			driver_status(Backend::Amd, (self.executable_create)(1, 0, ptr::null_mut(), &mut executable.handle), "native executable creation")?;
			driver_status(Backend::Amd, (self.executable_load)(executable.handle, self.agent, reader.handle, ptr::null_mut(), ptr::null_mut()), "native code-object load")?;
			driver_status(Backend::Amd, (self.executable_freeze)(executable.handle, ptr::null_mut()), "native executable freeze")?;
			let forward = self.native_dispatch(executable.handle, element, waves, NATIVE_FORWARD_SYMBOL, NATIVE_FORWARD_LAYOUT)?;
			let epoch = training.then(|| self.native_dispatch(executable.handle, element, waves, NATIVE_EPOCH_SYMBOL, epoch_layout)).transpose()?;
			let model_load = has_storage.then(|| self.native_dispatch(executable.handle, element, waves, NATIVE_MODEL_LOAD_SYMBOL, NATIVE_MODEL_LOAD_LAYOUT)).transpose()?;
			let kernarg_size = [Some(forward), epoch, model_load].into_iter().flatten().map(|dispatch| dispatch.kernel.kernarg).max().unwrap_or(0);
			let grid_sync = kernarg_size.next_multiple_of(HSA_GRID_SYNC_ALIGNMENT);
			let allocation_size = grid_sync.checked_add(HSA_GRID_SYNC_BYTES).ok_or_else(|| RecipeError::new("native AMD KERNARG allocation overflows"))?;
			let mut kernarg = ptr::null_mut();
			driver_status(Backend::Amd, (self.allocate)(self.kernarg_pool, allocation_size, 0, &mut kernarg), "native KERNARG allocation")?;
			driver_status(Backend::Amd, (self.allow)(1, &self.agent, ptr::null(), kernarg), "native GPU KERNARG access")?;
			Ok((NativeHsaProgram { executable, kernarg: kernarg as usize, kernarg_size, grid_sync: kernarg.add(grid_sync) as usize, free: self.free }, forward, epoch, model_load))
		}
	}
}
#[cfg(nvidia)]
impl Cuda {
	unsafe fn native_dispatch(&self, module: Ptr, name: &str, element: u8, layout: &'static [u8], waves: u32, shared_values: u32, register_values: u32) -> Result<Dispatch> {
		unsafe {
			let name = std::ffi::CString::new(name).map_err(|error| RecipeError::new(format!("NVIDIA native symbol is invalid: {error}")))?;
			let mut object = 0;
			driver_status(Backend::Nvidia, (self.function)(&mut object, module, name.as_ptr().cast()), "native symbol lookup")?;
			let (mut max_block, mut shared, mut used_registers) = (0, 0, 0);
			for (kind, output, action) in [(0, &mut max_block, "native workgroup query"), (1, &mut shared, "native shared-memory query"), (4, &mut used_registers, "native register query")] {
				driver_status(Backend::Nvidia, (self.function_attribute)(output, kind, object), action)?;
			}
			require(max_block > 0 && shared >= 0 && used_registers > 0, "NVIDIA native symbol resources are invalid")?;
			let register_wave = (used_registers as u32).checked_mul(self.wave).ok_or_else(|| RecipeError::new("NVIDIA native register count overflows"))?;
			require((self.registers / register_wave).min(self.threads / self.wave) != 0, "NVIDIA native symbol has no resident wave")?;
			let resources = Resources { shared: shared as u32, max_block: max_block as u32 };
			// The schedule sized every tile and the reduction buffer for its own workgroup, so the dispatch must use that width and not the wider one the register budget would allow.
			let geometry = nvidia(self.cus, self.wave, self.workgroup, self.block_lds, self.sm_lds, waves, resources)?;
			let mut active = 0;
			// The grid is one workgroup per SM and the barrier only completes once every one of them
			// is resident, so the occupancy question has to be asked about the launch this dispatch
			// really makes. With no dynamic shared memory it answers a question nobody goes on to ask.
			let values = shared_values.max(geometry.block.checked_mul(register_values).ok_or_else(|| RecipeError::new("NVIDIA native reduction buffer overflows"))?);
			let dynamic = values.checked_mul(u32::from(element)).ok_or_else(|| RecipeError::new("NVIDIA native shared memory overflows"))?;
			driver_status(Backend::Nvidia, (self.occupancy)(&mut active, object, geometry.block as i32, dynamic as usize), "native occupancy query")?;
			require(active > 0, "NVIDIA native symbol has no resident workgroup")?;
			Ok(Dispatch { kernel: Kernel::cuda(object, resources.shared, element, layout), geometry })
		}
	}

	unsafe fn load_native(
		&self, bytes: &[u8], element: u8, epoch_layout: &'static [u8], training: bool, has_storage: bool, waves: u32, shared_values: u32, register_values: u32,
	) -> Result<(NativeCudaProgram, Dispatch, Option<Dispatch>, Option<Dispatch>)> {
		unsafe {
			driver_status(Backend::Nvidia, (self.set)(self.context), "native context")?;
			let mut module = ptr::null_mut();
			driver_status(Backend::Nvidia, (self.load)(&mut module, bytes.as_ptr().cast()), "native cubin load")?;
			let program = NativeCudaProgram { module: module as usize, unload: self.unload };
			let forward = self.native_dispatch(program.module as Ptr, NATIVE_FORWARD_SYMBOL, element, NATIVE_FORWARD_LAYOUT, waves, shared_values, register_values)?;
			let epoch = training.then(|| self.native_dispatch(program.module as Ptr, NATIVE_EPOCH_SYMBOL, element, epoch_layout, waves, shared_values, register_values)).transpose()?;
			let model_load = has_storage.then(|| self.native_dispatch(program.module as Ptr, NATIVE_MODEL_LOAD_SYMBOL, element, NATIVE_MODEL_LOAD_LAYOUT, waves, 0, 0)).transpose()?;
			Ok((program, forward, epoch, model_load))
		}
	}
}
unsafe fn native_cpu_pointer(arguments: &[Ptr], index: usize) -> Ptr {
	unsafe { *arguments[index].cast::<u64>() as Ptr }
}

unsafe fn native_cpu_value<T: Copy>(arguments: &[Ptr], index: usize) -> T {
	unsafe { *arguments[index].cast::<T>() }
}

#[cfg(unix)]
unsafe extern "C" fn native_cpu_barrier(context: Ptr) {
	unsafe { &*context.cast::<std::sync::Barrier>() }.wait();
}

#[cfg(unix)]
unsafe fn launch_native_cpu_entry(forward: NativeForward, epoch: Option<NativeCpuEpoch>, model_load: Option<NativeModelLoad>, entry: NativeEntry, arguments: &[Ptr]) -> Result<()> {
	unsafe {
		match entry {
			NativeEntry::Forward => {
				require(arguments.len() == NATIVE_FORWARD_LAYOUT.len(), "native CPU forward argument count is invalid")?;
				forward(
					native_cpu_pointer(arguments, 0),
					native_cpu_pointer(arguments, 1),
					native_cpu_pointer(arguments, 2),
					native_cpu_pointer(arguments, 3),
					native_cpu_value(arguments, 4),
					native_cpu_value(arguments, 5),
					native_cpu_value(arguments, 6),
					native_cpu_value(arguments, 7),
				);
			}
			NativeEntry::Epoch => {
				require(arguments.len() == NATIVE_EPOCH_LAYOUT_FP64.len(), "native CPU epoch argument count is invalid")?;
				let pointers = (0..12).map(|index| native_cpu_pointer(arguments, index)).collect::<Vec<_>>();
				macro_rules! launch {
					($function:expr) => {
						$function(
							pointers[0],
							pointers[1],
							pointers[2],
							pointers[3],
							pointers[4],
							pointers[5],
							pointers[6],
							pointers[7],
							pointers[8],
							pointers[9],
							pointers[10],
							pointers[11],
							native_cpu_value(arguments, 12),
							native_cpu_value(arguments, 13),
							native_cpu_value(arguments, 14),
							native_cpu_value(arguments, 15),
							native_cpu_value(arguments, 16),
							native_cpu_value(arguments, 17),
							native_cpu_value(arguments, 18),
							native_cpu_value(arguments, 19),
							native_cpu_value(arguments, 20),
							native_cpu_value(arguments, 21),
							native_cpu_value(arguments, 22),
						)
					};
				}
				match epoch.ok_or_else(|| RecipeError::new("native epoch symbol is absent"))? {
					NativeCpuEpoch::F64(function) => launch!(function),
					NativeCpuEpoch::F32(function) => launch!(function),
					NativeCpuEpoch::F16(function) => launch!(function),
					NativeCpuEpoch::F8(function) => launch!(function),
				}
			}
			NativeEntry::ModelLoad => {
				require(arguments.len() == NATIVE_MODEL_LOAD_LAYOUT.len(), "native CPU model-load argument count is invalid")?;
				let function = model_load.ok_or_else(|| RecipeError::new("native model-load symbol is absent"))?;
				function(native_cpu_pointer(arguments, 0), native_cpu_pointer(arguments, 1), native_cpu_value(arguments, 2));
			}
		}
		Ok(())
	}
}

#[cfg(unix)]
unsafe fn launch_native_cpu(cpu: &NativeCpuProgram, entry: NativeEntry, arguments: &[Ptr], threads: u32) -> Result<()> {
	require(threads != 0, "native CPU worker count is empty")?;
	let slots = arguments.iter().map(|argument| *argument as usize).collect::<Vec<_>>();
	let barrier = std::sync::Barrier::new(threads as usize);
	let context = ptr::from_ref(&barrier) as usize;
	let wait = native_cpu_barrier as *const () as usize;
	let (thread, forward, epoch, model_load) = (cpu.thread, cpu.forward, cpu.epoch, cpu.model_load);
	std::thread::scope(|scope| {
		let workers = (0..threads)
			.map(|thread_id| {
				let slots = &slots;
				scope.spawn(move || -> Result<()> {
					let thread_id = i32::try_from(thread_id).map_err(|_| RecipeError::new("native CPU worker ID exceeds i32"))?;
					let arguments = slots.iter().map(|slot| *slot as Ptr).collect::<Vec<_>>();
					unsafe {
						thread(thread_id, context as Ptr, wait as Ptr);
						launch_native_cpu_entry(forward, epoch, model_load, entry, &arguments)
					}
				})
			})
			.collect::<Vec<_>>();
		for worker in workers {
			worker.join().map_err(|_| RecipeError::new("native CPU worker panicked"))??;
		}
		Ok(())
	})
}

impl NativeProgram {
	fn load(gpu: &'static Gpu, artifact: NativeArtifact, graph: &Graph, schedule: NativeSchedule, register_values: u32, waves: u32) -> Result<Self> {
		native_artifact_contract(&artifact)?;
		require(artifact.backend.backend() == gpu.backend, format!("native artifact backend {:?} does not match device {:?}", artifact.backend.backend(), gpu.backend))?;
		let element = u8::try_from(artifact.precision.model.bytes()).map_err(|_| RecipeError::new("native precision width is invalid"))?;
		let (backend, forward, epoch, model_load) = match &gpu.driver {
			Driver::Cpu => {
				#[cfg(unix)]
				{
					let cpu = load_native_cpu(&artifact)?;
					let geometry = Geometry { groups: cpu_worker_threads()?, block: 1 };
					let forward = Dispatch { kernel: Kernel::remote(0, artifact.precision.model.bytes() as u8, NATIVE_FORWARD_LAYOUT), geometry };
					let epoch = artifact.training.then_some(Dispatch { kernel: Kernel::remote(0, artifact.precision.model.bytes() as u8, artifact.precision.epoch_layout), geometry });
					let model_load =
						(!artifact.storage.is_empty()).then_some(Dispatch { kernel: Kernel::remote(0, artifact.precision.model.bytes() as u8, NATIVE_MODEL_LOAD_LAYOUT), geometry });
					(NativeBackend::Cpu(cpu), forward, epoch, model_load)
				}
				#[cfg(not(unix))]
				return Err(RecipeError::new("CPU native artifact loading requires POSIX dynamic loading"));
			}
			#[cfg(amd)]
			Driver::Hsa(driver) => {
				let (program, forward, epoch, model_load) =
					unsafe { driver.load_native(&artifact.artifact, element, artifact.precision.epoch_layout, artifact.training, !artifact.storage.is_empty(), waves)? };
				(NativeBackend::Amd(program), forward, epoch, model_load)
			}
			#[cfg(nvidia)]
			Driver::Cuda(driver) => {
				let (program, forward, epoch, model_load) = unsafe {
					driver.load_native(
						&artifact.artifact,
						element,
						artifact.precision.epoch_layout,
						artifact.training,
						!artifact.storage.is_empty(),
						waves,
						schedule.shared_values,
						register_values,
					)?
				};
				(NativeBackend::Nvidia(program), forward, epoch, model_load)
			}
			Driver::Remote(remote) => {
				let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
				channel.write_u8(REMOTE_LOAD)?;
				channel.write_u64(artifact.artifact.len() as u64)?;
				channel.write_bytes(&artifact.artifact)?;
				channel.write_u32(waves)?;
				channel.write_u32(schedule.shared_values)?;
				channel.write_u32(register_values)?;
				channel.write_u8(element)?;
				channel.write_u8(u8::from(artifact.training))?;
				channel.write_u8(u8::from(artifact.precision.epoch_layout == NATIVE_EPOCH_LAYOUT_FP64))?;
				channel.write_u8(u8::from(!artifact.storage.is_empty()))?;
				channel.flush()?;
				channel.read_status("artifact load")?;
				let mut read_dispatch = |layout: &'static [u8]| -> Result<Dispatch> {
					let shared = channel.read_u32()?;
					let groups = channel.read_u32()?;
					let block = channel.read_u32()?;
					Ok(Dispatch { kernel: Kernel::remote(shared, element, layout), geometry: Geometry { groups, block } })
				};
				let forward = read_dispatch(NATIVE_FORWARD_LAYOUT)?;
				let epoch = artifact.training.then(|| read_dispatch(artifact.precision.epoch_layout)).transpose()?;
				let model_load = (!artifact.storage.is_empty()).then(|| read_dispatch(NATIVE_MODEL_LOAD_LAYOUT)).transpose()?;
				(NativeBackend::Remote, forward, epoch, model_load)
			}
		};
		let entrypoints = [Some(NATIVE_FORWARD_SYMBOL), epoch.map(|_| NATIVE_EPOCH_SYMBOL), model_load.map(|_| NATIVE_MODEL_LOAD_SYMBOL)].into_iter().flatten().collect::<Vec<_>>().join(",");
		debug(&format!(
			"native load key={} path={} entrypoints={entrypoints}",
			artifact.path.parent().and_then(Path::file_name).and_then(|key| key.to_str()).unwrap_or("unknown"),
			artifact.path.display()
		))?;
		let block = forward.geometry.block.max(epoch.map_or(0, |dispatch| dispatch.geometry.block));
		let reduction_values = block.checked_mul(register_values).ok_or_else(|| RecipeError::new("native contraction lane reduction overflows"))?;
		let gradient_values = native_gradient_values(graph.parameters.len(), &schedule.contractions)?;
		Ok(Self {
			gpu,
			artifact,
			backend,
			forward,
			epoch,
			model_load,
			tile: schedule.tile,
			contractions: schedule.contractions,
			shared_values: schedule.shared_values,
			reduction_values,
			gradient_values,
		})
	}

	fn dispatch(&self, entry: NativeEntry) -> Result<Dispatch> {
		match entry {
			NativeEntry::Forward => Ok(self.forward),
			NativeEntry::Epoch => self.epoch.ok_or_else(|| RecipeError::new("native epoch symbol is absent")),
			NativeEntry::ModelLoad => self.model_load.ok_or_else(|| RecipeError::new("native model-load symbol is absent")),
		}
	}

	fn launch_forward(&self, arguments: &mut [Ptr]) -> Result<()> {
		self.launch(NativeEntry::Forward, arguments, self.forward.geometry.threads()?)
	}

	fn launch_epoch(&self, arguments: &mut [Ptr]) -> Result<()> {
		let dispatch = self.dispatch(NativeEntry::Epoch)?;
		self.launch(NativeEntry::Epoch, arguments, dispatch.geometry.threads()?)
	}

	fn launch_model_load(&self, arguments: &mut [Ptr]) -> Result<()> {
		let dispatch = self.dispatch(NativeEntry::ModelLoad)?;
		self.launch(NativeEntry::ModelLoad, arguments, dispatch.geometry.threads()?)
	}

	fn launch(&self, entry: NativeEntry, arguments: &mut [Ptr], threads: u32) -> Result<()> {
		let gpu = self.gpu;
		require(!INTERRUPTED.load(Ordering::Acquire), "interrupted before native dispatch")?;
		let dispatch = self.dispatch(entry)?;
		require(arguments.len() == dispatch.kernel.layout.len(), "native argument count is invalid")?;
		gpu.activate()?;
		let values = if matches!(entry, NativeEntry::ModelLoad) { 0 } else { self.shared_values.max(self.reduction_values) };
		let dynamic = values.checked_mul(u32::from(dispatch.kernel.element)).ok_or_else(|| RecipeError::new("native shared memory size overflows"))?;
		let shared = dispatch.kernel.shared.checked_add(dynamic).ok_or_else(|| RecipeError::new("native shared memory size overflows"))?;
		require(shared <= gpu.shared_limit, "native shared memory exceeds device limit")?;
		let _guard = gpu.dispatch.lock().map_err(|_| RecipeError::new("GPU dispatch lock is poisoned"))?;
		unsafe { launch_backend(gpu, &self.backend, &dispatch, entry, arguments, threads, dynamic, shared) }
	}
}

/// Dispatches one loaded entrypoint on the device that loaded it. The caller
/// holds the device dispatch lock and has already validated the argument list
/// and shared-memory budget.
unsafe fn launch_backend(gpu: &Gpu, backend: &NativeBackend, dispatch: &Dispatch, entry: NativeEntry, arguments: &mut [Ptr], threads: u32, dynamic: u32, shared: u32) -> Result<()> {
	let block = dispatch.geometry.block;
	unsafe {
		match (backend, &gpu.driver) {
			#[cfg(unix)]
			(NativeBackend::Cpu(cpu), Driver::Cpu) => launch_native_cpu(cpu, entry, arguments, threads),
			#[cfg(amd)]
			(NativeBackend::Amd(program), Driver::Hsa(driver)) => {
				require(program.executable.handle != 0, "native AMD executable is absent")?;
				let kernarg = program.kernarg as Ptr;
				ptr::write_bytes(kernarg.cast::<u8>(), 0, program.kernarg_size);
				let mut offset = 0_usize;
				for (argument, kind) in arguments.iter().zip(dispatch.kernel.layout) {
					let bytes = usize::from(*kind - b'0');
					offset = offset.next_multiple_of(bytes);
					ptr::copy_nonoverlapping((*argument).cast::<u8>(), kernarg.cast::<u8>().add(offset), bytes);
					offset += bytes;
				}
				let implicit = offset.next_multiple_of(HSA_IMPLICIT_ARGUMENT_ALIGNMENT);
				let implicit_bytes = dispatch
					.kernel
					.kernarg
					.checked_sub(implicit)
					.ok_or_else(|| RecipeError::new(format!("native HSA KERNARG metadata {} is shorter than its {implicit}-byte explicit layout", dispatch.kernel.kernarg)))?;
				require(
					matches!(implicit_bytes, 0 | HSA_IMPLICIT_ARGUMENT_BYTES) && dispatch.kernel.kernarg <= program.kernarg_size,
					format!(
						"native HSA KERNARG layout is invalid: entry={entry:?} metadata={} explicit={offset} implicit={implicit} allocation={} layout={:?}",
						dispatch.kernel.kernarg, program.kernarg_size, dispatch.kernel.layout
					),
				)?;
				let groups = threads
					.checked_div(block)
					.filter(|groups| groups.saturating_mul(block) == threads && *groups <= u32::from(u16::MAX))
					.ok_or_else(|| RecipeError::new("native AMD grid size is invalid"))?;
				let grid_sync = program.grid_sync as Ptr;
				if std::env::var_os("RECIPE_DEBUG").is_some() {
					debug(&format!("AMD grid sync before reset {:?}", std::slice::from_raw_parts(grid_sync.cast::<u32>(), HSA_GRID_SYNC_BYTES / size_of::<u32>())))?;
				}
				ptr::write_bytes(grid_sync.cast::<u8>(), 0, HSA_GRID_SYNC_BYTES);
				grid_sync.cast::<u8>().add(HSA_GRID_SYNC_GROUPS_OFFSET).cast::<u32>().write(groups);
				if implicit_bytes != 0 {
					kernarg.cast::<u8>().add(implicit + HSA_MULTIGRID_SYNC_POINTER_OFFSET).cast::<u64>().write(program.grid_sync as u64);
				}
				(driver.store)(driver.signal, 1);
				let queue = &mut *(driver.queue as *mut HsaQueue);
				let index = (driver.write)(queue, 1);
				let packet = queue.base.cast::<HsaPacket>().add(index as usize & (queue.size as usize - 1));
				packet.write(HsaPacket {
					header: 1,
					setup: 1,
					workgroup_x: block as u16,
					workgroup_y: 1,
					workgroup_z: 1,
					reserved0: 0,
					grid_x: threads,
					grid_y: 1,
					grid_z: 1,
					private: dispatch.kernel.private,
					group: shared,
					object: dispatch.kernel.object,
					kernarg,
					reserved1: 0,
					completion: driver.signal,
				});
				std::sync::atomic::fence(Ordering::Release);
				let header = &*(&mut (*packet).header as *mut u16 as *mut std::sync::atomic::AtomicU16);
				header.store(2 | 2 << 9 | 2 << 11, Ordering::Release);
				(driver.store)(queue.doorbell, index as i64);
				debug("AMD dispatch submitted")?;
				let completed = (driver.wait)(driver.signal, 0, 0, u64::MAX, 1);
				debug(&format!("AMD dispatch completed with signal {completed}"))?;
				require(completed == 0, "native AMD dispatch failed")
			}
			#[cfg(nvidia)]
			(NativeBackend::Nvidia(program), Driver::Cuda(driver)) => {
				require(program.module != 0, "native NVIDIA module is absent")?;
				let stream = ptr::null_mut();
				driver_status(
					Backend::Nvidia,
					(driver.launch)(dispatch.kernel.object as usize, threads / block, 1, 1, block, 1, 1, dynamic, stream, arguments.as_mut_ptr(), ptr::null_mut()),
					"native dispatch",
				)
			}
			(NativeBackend::Remote, Driver::Remote(remote)) => {
				let entry = match entry {
					NativeEntry::Forward => 0_u8,
					NativeEntry::Epoch => 1,
					NativeEntry::ModelLoad => 2,
				};
				let mut channel = remote.channel.lock().map_err(|_| RecipeError::new("remote channel is poisoned"))?;
				channel.write_u8(REMOTE_LAUNCH)?;
				channel.write_u8(entry)?;
				for (argument, kind) in arguments.iter().zip(dispatch.kernel.layout) {
					let bytes = usize::from(*kind - b'0');
					let mut data = [0_u8; 8];
					ptr::copy_nonoverlapping((*argument).cast::<u8>(), data.as_mut_ptr(), bytes);
					channel.write_bytes(&data[..bytes])?;
				}
				channel.flush()?;
				channel.read_status("dispatch")
			}
			_ => Err(RecipeError::new("native program backend changed after loading")),
		}
	}
}

fn load_amd(_selection: Option<&[String]>) -> Result<Vec<Gpu>> {
	#[cfg(not(amd))]
	return Err(RecipeError::new("AMD support is not compiled into this build"));
	#[cfg(amd)]
	unsafe {
		let runtime = std::sync::Arc::new(Library::open(env!("RECIPE_HSA_RUNTIME"))?);
		let init: unsafe extern "C" fn() -> i32 = runtime.function(b"hsa_init\0")?;
		let iterate: unsafe extern "C" fn(extern "C" fn(u64, Ptr) -> i32, Ptr) -> i32 = runtime.function(b"hsa_iterate_agents\0")?;
		let info: HsaInfo = runtime.function(b"hsa_agent_get_info\0")?;
		let check = |s, a| driver_status(Backend::Amd, s, a);
		check(init(), "initialization")?;
		let mut cpu = HsaQuery { info, attribute: 17, expected: 0, secondary: -1, mask: 0, found: 0 };
		let mut gpu = HsaGpuQuery { info, found: Vec::new() };
		check(iterate(collect_hsa, (&mut cpu as *mut HsaQuery).cast()), "CPU agent")?;
		check(iterate(collect_discrete_hsa, (&mut gpu as *mut HsaGpuQuery).cast()), "GPU agent")?;
		require(cpu.found != 0 && !gpu.found.is_empty(), "AMD CPU or discrete GPU agent is absent")?;
		gpu.found
			.into_iter()
			.enumerate()
			.filter(|(index, _)| _selection.is_none_or(|names| names.contains(&format!("amd{index}"))))
			.map(|(index, agent)| load_amd_gpu(&runtime, info, cpu.found, agent, index))
			.collect()
	}
}
#[cfg(amd)]
fn load_amd_gpu(runtime: &std::sync::Arc<Library>, info: HsaInfo, cpu_agent: u64, agent: u64, index: usize) -> Result<Gpu> {
	unsafe {
		let pool_info: HsaInfo = runtime.function(b"hsa_amd_memory_pool_get_info\0")?;
		let pool_iterate: unsafe extern "C" fn(u64, extern "C" fn(u64, Ptr) -> i32, Ptr) -> i32 = runtime.function(b"hsa_amd_agent_iterate_memory_pools\0")?;
		let check = |s, a| driver_status(Backend::Amd, s, a);
		let mut vram = HsaQuery { info: pool_info, attribute: 0, expected: 0, secondary: 1, mask: 4, found: 0 };
		let mut kernarg = HsaQuery { info: pool_info, attribute: 0, expected: 0, secondary: 1, mask: 1, found: 0 };
		check(pool_iterate(agent, collect_hsa, (&mut vram as *mut HsaQuery).cast()), "VRAM pools")?;
		check(pool_iterate(cpu_agent, collect_hsa, (&mut kernarg as *mut HsaQuery).cast()), "KERNARG pools")?;
		require(vram.found != 0 && kernarg.found != 0, "AMD VRAM or KERNARG pool is absent")?;
		let mut memory = 0_usize;
		check(pool_info(vram.found, 2, (&mut memory as *mut usize).cast()), "VRAM size")?;
		let (mut wave, mut workgroup, mut available, mut node, mut cus) = (0_u32, 0_u32, 0_u32, 0_u32, 0_u32);
		for (attribute, output, action) in [
			(6, (&mut wave as *mut u32).cast(), "wave query"),
			(8, (&mut workgroup as *mut u32).cast(), "workgroup query"),
			(0xA002, (&mut available as *mut u32).cast(), "CU query"),
			(0xA004, (&mut node as *mut u32).cast(), "KFD node query"),
			(0xA014, (&mut cus as *mut u32).cast(), "cooperative CU query"),
		] {
			check(info(agent, attribute, output), action)?;
		}
		require(cus <= available, "AMD cooperative CU count exceeds available CUs")?;
		let path = format!("/sys/class/kfd/kfd/topology/nodes/{node}/properties");
		let properties = fs::read_to_string(&path).map_err(|error| RecipeError::new(format!("cannot read {path}: {error}")))?;
		let gfx = kfd_property(&properties, "gfx_target_version")?;
		let target = format!("gfx{}{}{}", gfx / 10000, gfx / 100 % 100, gfx % 100);
		let native_target = BackendTarget::Amd { architecture: target.clone() };
		let reader_create: unsafe extern "C" fn(*const c_void, usize, *mut u64) -> i32 = runtime.function(b"hsa_code_object_reader_create_from_memory\0")?;
		let reader_destroy: unsafe extern "C" fn(u64) -> i32 = runtime.function(b"hsa_code_object_reader_destroy\0")?;
		let executable_create: unsafe extern "C" fn(i32, i32, Ptr, *mut u64) -> i32 = runtime.function(b"hsa_executable_create_alt\0")?;
		let executable_destroy: unsafe extern "C" fn(u64) -> i32 = runtime.function(b"hsa_executable_destroy\0")?;
		let executable_load: unsafe extern "C" fn(u64, u64, u64, Ptr, Ptr) -> i32 = runtime.function(b"hsa_executable_load_agent_code_object\0")?;
		let executable_freeze: unsafe extern "C" fn(u64, Ptr) -> i32 = runtime.function(b"hsa_executable_freeze\0")?;
		let symbol: HsaSymbol = runtime.function(b"hsa_executable_get_symbol_by_name\0")?;
		let symbol_info: HsaSymbolInfo = runtime.function(b"hsa_executable_symbol_get_info\0")?;
		let lds = kfd_property(&properties, "lds_size_in_kb")?.checked_mul(1024).ok_or_else(|| RecipeError::new("AMD LDS size overflows"))?;
		let queue_create: unsafe extern "C" fn(u64, u32, u32, Ptr, Ptr, u32, u32, *mut Ptr) -> i32 = runtime.function(b"hsa_queue_create\0")?;
		let signal_create: unsafe extern "C" fn(i64, u32, *const u64, *mut u64) -> i32 = runtime.function(b"hsa_signal_create\0")?;
		let allocate: unsafe extern "C" fn(u64, usize, u32, *mut Ptr) -> i32 = runtime.function(b"hsa_amd_memory_pool_allocate\0")?;
		let allow: unsafe extern "C" fn(u32, *const u64, *const u32, *const c_void) -> i32 = runtime.function(b"hsa_amd_agents_allow_access\0")?;
		let (mut queue, mut completion) = (ptr::null_mut(), 0);
		driver_status(Backend::Amd, queue_create(agent, 256, 2, ptr::null_mut(), ptr::null_mut(), u32::MAX, u32::MAX, &mut queue), "queue creation")?;
		check(signal_create(0, 0, ptr::null(), &mut completion), "signal creation")?;
		let hsa = Hsa {
			_runtime: runtime.clone(),
			reader_create,
			reader_destroy,
			executable_create,
			executable_destroy,
			executable_load,
			executable_freeze,
			symbol,
			symbol_info,
			info,
			allocate,
			allow,
			queue,
			cpu_agent,
			free: runtime.function(b"hsa_amd_memory_pool_free\0")?,
			copy: runtime.function(b"hsa_memory_copy\0")?,
			clear: runtime.function(b"hsa_amd_memory_fill\0")?,
			store: runtime.function(b"hsa_signal_store_screlease\0")?,
			wait: runtime.function(b"hsa_signal_wait_scacquire\0")?,
			write: runtime.function(b"hsa_queue_add_write_index_scacq_screl\0")?,
			signal: completion,
			vram_pool: vram.found,
			kernarg_pool: kernarg.found,
			agent,
			cus,
			wave,
			workgroup,
			lds,
		};
		Ok(Gpu { name: format!("amd{index}"), backend: Backend::Amd, native_target, driver: Driver::Hsa(hsa), memory: memory as u64, shared_limit: lds, dispatch: Mutex::new(()) })
	}
}
fn load_nvidia(_selection: Option<&[String]>) -> Result<Vec<Gpu>> {
	#[cfg(not(nvidia))]
	return Err(RecipeError::new("NVIDIA support is not compiled into this build"));
	#[cfg(nvidia)]
	unsafe {
		const MAX_BLOCK: i32 = 1;
		const BLOCK_LDS: i32 = 8;
		const WAVE: i32 = 10;
		const CUS: i32 = 16;
		const INTEGRATED: i32 = 18;
		const THREADS_PER_SM: i32 = 39;
		const SM_LDS: i32 = 81;
		const REGISTERS_PER_SM: i32 = 82;
		const COMPUTE_MAJOR: i32 = 75;
		const COMPUTE_MINOR: i32 = 76;
		let runtime = std::sync::Arc::new(Library::open(if cfg!(windows) { "nvcuda.dll" } else { env!("RECIPE_NV_RUNTIME") })?);
		let init: unsafe extern "C" fn(u32) -> i32 = runtime.function(b"cuInit\0")?;
		let count_devices: unsafe extern "C" fn(*mut i32) -> i32 = runtime.function(b"cuDeviceGetCount\0")?;
		let get_device: unsafe extern "C" fn(*mut i32, i32) -> i32 = runtime.function(b"cuDeviceGet\0")?;
		let attribute: NvQuery = runtime.function(b"cuDeviceGetAttribute\0")?;
		let total: unsafe extern "C" fn(*mut usize, i32) -> i32 = runtime.function(b"cuDeviceTotalMem_v2\0")?;
		let create: unsafe extern "C" fn(*mut Ptr, u32, i32) -> i32 = runtime.function(b"cuCtxCreate_v2\0")?;
		let load: unsafe extern "C" fn(*mut Ptr, *const c_void) -> i32 = runtime.function(b"cuModuleLoadData\0")?;
		let unload: unsafe extern "C" fn(Ptr) -> i32 = runtime.function(b"cuModuleUnload\0")?;
		let function: unsafe extern "C" fn(*mut usize, Ptr, *const u8) -> i32 = runtime.function(b"cuModuleGetFunction\0")?;
		let function_attribute: unsafe extern "C" fn(*mut i32, i32, usize) -> i32 = runtime.function(b"cuFuncGetAttribute\0")?;
		let occupancy: unsafe extern "C" fn(*mut i32, usize, i32, usize) -> i32 = runtime.function(b"cuOccupancyMaxActiveBlocksPerMultiprocessor\0")?;
		let check = |s, a| driver_status(Backend::Nvidia, s, a);
		let mut count = 0;
		check(init(0), "initialization")?;
		check(count_devices(&mut count), "device enumeration")?;
		let load_device = |device, index| -> Result<Gpu> {
			let check = |s, a| driver_status(Backend::Nvidia, s, a);
			let mut context = ptr::null_mut();
			let (mut cus, mut wave, mut workgroup, mut block_lds, mut sm_lds, mut registers, mut threads, mut compute_major, mut compute_minor) = (0, 0, 0, 0, 0, 0, 0, 0, 0);
			let mut memory = 0;
			check(total(&mut memory, device), "VRAM size")?;
			for (kind, output, action) in [
				(CUS, &mut cus, "SM query"),
				(WAVE, &mut wave, "warp query"),
				(MAX_BLOCK, &mut workgroup, "workgroup query"),
				(BLOCK_LDS, &mut block_lds, "workgroup LDS query"),
				(SM_LDS, &mut sm_lds, "SM LDS query"),
				(REGISTERS_PER_SM, &mut registers, "register query"),
				(THREADS_PER_SM, &mut threads, "resident thread query"),
				(COMPUTE_MAJOR, &mut compute_major, "compute capability major query"),
				(COMPUTE_MINOR, &mut compute_minor, "compute capability minor query"),
			] {
				check(attribute(output, kind, device), action)?;
			}
			require(compute_major > 0 && compute_minor >= 0, "Nvidia compute capability is invalid")?;
			let native_target = BackendTarget::Nvidia { architecture: format!("sm_{compute_major}{compute_minor}") };
			check(create(&mut context, 0, device), "context creation")?;
			let cuda = Cuda {
				_runtime: runtime.clone(),
				context,
				set: runtime.function(b"cuCtxSetCurrent\0")?,
				allocate: runtime.function(b"cuMemAlloc_v2\0")?,
				free: runtime.function(b"cuMemFree_v2\0")?,
				upload: runtime.function(b"cuMemcpyHtoD_v2\0")?,
				download: runtime.function(b"cuMemcpyDtoH_v2\0")?,
				clear: runtime.function(b"cuMemsetD8_v2\0")?,
				memory_info: runtime.function(b"cuMemGetInfo_v2\0")?,
				synchronize: runtime.function(b"cuCtxSynchronize\0")?,
				launch: runtime.function(b"cuLaunchKernel\0")?,
				load,
				unload,
				function,
				function_attribute,
				occupancy,
				cus: cus as u32,
				wave: wave as u32,
				workgroup: workgroup as u32,
				block_lds: block_lds as u32,
				sm_lds: sm_lds as u32,
				registers: registers as u32,
				threads: threads as u32,
			};
			Ok(Gpu {
				name: format!("nv{index}"),
				backend: Backend::Nvidia,
				native_target,
				driver: Driver::Cuda(cuda),
				memory: memory as u64,
				shared_limit: (block_lds as u32).min(sm_lds as u32),
				dispatch: Mutex::new(()),
			})
		};
		let mut discrete = Vec::new();
		for ordinal in 0..count {
			let (mut gpu, mut integrated) = (0, 0);
			check(get_device(&mut gpu, ordinal), "device enumeration")?;
			check(attribute(&mut integrated, INTEGRATED, gpu), "device probe")?;
			if integrated == 0 {
				discrete.push(gpu)
			}
		}
		require(!discrete.is_empty(), "Nvidia has no discrete GPU")?;
		discrete.into_iter().enumerate().filter(|(index, _)| _selection.is_none_or(|names| names.contains(&format!("nv{index}")))).map(|(index, device)| load_device(device, index)).collect()
	}
}
type WorkerWire = Wire<std::io::Stdin, std::io::Stdout>;
struct WorkerProgram {
	backend: NativeBackend,
	dispatches: [Option<Dispatch>; 3],
	shared_values: u32,
	reduction_values: u32,
}
/// Serves one local device to a remote Recipe process over stdin/stdout: the
/// transport half of a cross-host topology link. Commands mirror the `Gpu`
/// verbs plus artifact load and entrypoint dispatch, so a driving process can
/// place work on this host's device exactly as on a local one.
pub fn worker_serve(name: &str) -> Result<()> {
	let mut wire = WorkerWire { input: std::io::BufReader::new(std::io::stdin()), output: std::io::BufWriter::new(std::io::stdout()), role: "worker" };
	let probe = device(Some(name)).and_then(|gpu| {
		let (backend, wave) = match &gpu.driver {
			#[cfg(amd)]
			Driver::Hsa(driver) => (1_u8, driver.wave),
			#[cfg(nvidia)]
			Driver::Cuda(driver) => (2_u8, driver.wave),
			_ => return Err(RecipeError::new(format!("device {name:?} is not a local GPU"))),
		};
		Ok((gpu, backend, wave))
	});
	let (gpu, backend, wave) = match probe {
		Ok(probe) => probe,
		Err(error) => {
			wire.status(&Err(error.clone()))?;
			wire.flush()?;
			return Err(error);
		}
	};
	wire.status(&Ok(()))?;
	let architecture = native_target_label(&gpu.native_target);
	wire.write_bytes(&[backend, architecture.len() as u8])?;
	wire.write_bytes(architecture.as_bytes())?;
	wire.write_bytes(&gpu.memory.to_le_bytes())?;
	wire.write_u32(gpu.shared_limit)?;
	wire.write_u32(wave)?;
	wire.flush()?;
	let mut program: Option<WorkerProgram> = None;
	loop {
		let mut command = [0_u8; 1];
		match wire.input.read_exact(&mut command) {
			Ok(()) => {}
			Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
			Err(error) => return WorkerWire::read_error("worker", error),
		}
		match command[0] {
			REMOTE_ALLOCATE => {
				let bytes = wire.read_u64()? as usize;
				let allocated = gpu.allocate(bytes);
				wire.status(&allocated.as_ref().map(|_| ()).map_err(Clone::clone))?;
				if let Ok(pointer) = allocated {
					wire.write_bytes(&pointer.to_le_bytes())?;
				}
			}
			REMOTE_FREE => {
				let pointer = wire.read_u64()?;
				gpu.free(pointer);
			}
			REMOTE_UPLOAD => {
				let pointer = wire.read_u64()?;
				let bytes = wire.read_u64()? as usize;
				let mut data = vec![0_u8; bytes];
				wire.read_into(&mut data)?;
				wire.status(&gpu.upload(pointer, data.as_ptr().cast(), bytes).map(|_| ()))?;
			}
			REMOTE_DOWNLOAD => {
				let pointer = wire.read_u64()?;
				let bytes = wire.read_u64()? as usize;
				let mut data = vec![0_u8; bytes];
				let downloaded = gpu.download(data.as_mut_ptr().cast(), pointer, bytes);
				wire.status(&downloaded)?;
				if downloaded.is_ok() {
					wire.write_bytes(&data)?;
				}
			}
			REMOTE_SYNCHRONIZE => wire.status(&gpu.synchronize())?,
			REMOTE_MEMORY => {
				let free = gpu.free_bytes();
				wire.status(&free.as_ref().map(|_| ()).map_err(Clone::clone))?;
				if let Ok(bytes) = free {
					wire.write_bytes(&bytes.to_le_bytes())?;
				}
			}
			REMOTE_CLEAR => {
				let pointer = wire.read_u64()?;
				let bytes = wire.read_u64()? as usize;
				wire.status(&gpu.clear(pointer, bytes))?;
			}
			REMOTE_LOAD => {
				let bytes = wire.read_u64()? as usize;
				let mut artifact = vec![0_u8; bytes];
				wire.read_into(&mut artifact)?;
				let waves = wire.read_u32()?;
				let shared_values = wire.read_u32()?;
				let register_values = wire.read_u32()?;
				let element = wire.read_u8()?;
				let training = wire.read_u8()? != 0;
				let epoch_layout: &'static [u8] = if wire.read_u8()? != 0 { NATIVE_EPOCH_LAYOUT_FP64 } else { NATIVE_EPOCH_LAYOUT_FP32 };
				let has_storage = wire.read_u8()? != 0;
				let loaded: Result<(NativeBackend, Dispatch, Option<Dispatch>, Option<Dispatch>)> = match &gpu.driver {
					#[cfg(amd)]
					Driver::Hsa(driver) => unsafe { driver.load_native(&artifact, element, epoch_layout, training, has_storage, waves) }
						.map(|(program, forward, epoch, model_load)| (NativeBackend::Amd(program), forward, epoch, model_load)),
					#[cfg(nvidia)]
					Driver::Cuda(driver) => unsafe { driver.load_native(&artifact, element, epoch_layout, training, has_storage, waves, shared_values, register_values) }
						.map(|(program, forward, epoch, model_load)| (NativeBackend::Nvidia(program), forward, epoch, model_load)),
					_ => Err(RecipeError::new("worker device driver is not native")),
				};
				wire.status(&loaded.as_ref().map(|_| ()).map_err(Clone::clone))?;
				if let Ok((backend, forward, epoch, model_load)) = loaded {
					let block = forward.geometry.block.max(epoch.map_or(0, |dispatch| dispatch.geometry.block));
					let reduction_values = block.checked_mul(register_values).ok_or_else(|| RecipeError::new("native contraction lane reduction overflows"))?;
					for dispatch in [Some(forward), epoch, model_load].into_iter().flatten() {
						wire.write_u32(dispatch.kernel.shared)?;
						wire.write_u32(dispatch.geometry.groups)?;
						wire.write_u32(dispatch.geometry.block)?;
					}
					program = Some(WorkerProgram { backend, dispatches: [Some(forward), epoch, model_load], shared_values, reduction_values });
				}
			}
			REMOTE_LAUNCH => {
				let entry = match wire.read_u8()? {
					0 => NativeEntry::Forward,
					1 => NativeEntry::Epoch,
					2 => NativeEntry::ModelLoad,
					byte => return Err(RecipeError::new(format!("worker received unknown entrypoint {byte}"))),
				};
				let launched = program.as_ref().ok_or_else(|| RecipeError::new("worker has no loaded program")).and_then(|program| {
					let dispatch = program.dispatches[entry as usize].ok_or_else(|| RecipeError::new("worker entrypoint is absent"))?;
					let mut slots = [0_u64; 32];
					for (slot, kind) in slots.iter_mut().zip(dispatch.kernel.layout) {
						let bytes = usize::from(*kind - b'0');
						let mut data = [0_u8; 8];
						wire.read_into(&mut data[..bytes])?;
						*slot = u64::from_le_bytes(data);
					}
					let mut arguments = slots[..dispatch.kernel.layout.len()].iter().map(|slot| slot as *const u64 as Ptr).collect::<Vec<_>>();
					let values = if matches!(entry, NativeEntry::ModelLoad) { 0 } else { program.shared_values.max(program.reduction_values) };
					let dynamic = values.checked_mul(u32::from(dispatch.kernel.element)).ok_or_else(|| RecipeError::new("native shared memory size overflows"))?;
					let shared = dispatch.kernel.shared.checked_add(dynamic).ok_or_else(|| RecipeError::new("native shared memory size overflows"))?;
					require(shared <= gpu.shared_limit, "native shared memory exceeds device limit")?;
					gpu.activate()?;
					let _guard = gpu.dispatch.lock().map_err(|_| RecipeError::new("GPU dispatch lock is poisoned"))?;
					unsafe { launch_backend(gpu, &program.backend, &dispatch, entry, &mut arguments, dispatch.geometry.threads()?, dynamic, shared) }
				});
				wire.status(&launched)?;
			}
			byte => return Err(RecipeError::new(format!("worker received unknown command {byte}"))),
		}
		wire.flush()?;
	}
}
#[cfg(all(unix, not(windows)))]
#[link(name = "dl")]
unsafe extern "C" {
	fn dlopen(name: *const std::ffi::c_char, flags: i32) -> Ptr;
	fn dlsym(handle: Ptr, name: *const std::ffi::c_char) -> Ptr;
	fn dlclose(handle: Ptr) -> i32;
	fn mmap(address: Ptr, length: usize, protection: i32, flags: i32, descriptor: i32, offset: i64) -> Ptr;
	fn munmap(address: Ptr, length: usize) -> i32;
}
#[cfg(all(nvidia, windows))]
unsafe fn dlopen(name: *const std::ffi::c_char, _: i32) -> Ptr {
	unsafe { LoadLibraryA(name) }
}
#[cfg(all(nvidia, windows))]
unsafe fn dlsym(handle: Ptr, name: *const std::ffi::c_char) -> Ptr {
	unsafe { GetProcAddress(handle, name) }
}
#[cfg(all(nvidia, windows))]
#[link(name = "kernel32")]
unsafe extern "system" {
	fn LoadLibraryA(name: *const std::ffi::c_char) -> Ptr;
	fn GetProcAddress(handle: Ptr, name: *const std::ffi::c_char) -> Ptr;
	fn FreeLibrary(handle: Ptr) -> i32;
}
unsafe extern "C" {
	fn signal(number: i32, handler: extern "C" fn(i32)) -> usize;
	#[cfg_attr(windows, link_name = "_write")]
	fn write(file: i32, bytes: *const c_void, length: usize) -> isize;
}
fn distance(left: &[f64], right: &[f64]) -> f64 {
	left.iter().zip(right).map(|(a, b)| (a - b).powi(2)).sum()
}
fn nearest(query: &[f64], state: &[f64], features: usize) -> (usize, f64) {
	state.chunks_exact(features).enumerate().map(|(index, row)| (index, distance(query, row))).min_by(|left, right| left.1.total_cmp(&right.1)).unwrap_or((0, f64::INFINITY))
}
fn graph_inputs(graph: &Graph, samples: &[f64], targets: &[f64], rows: usize, gpu: &'static Gpu, precision: Compute) -> Result<Vec<f64>> {
	let input_count = checked_mul(rows, graph.input.elements(), "estimator input slice")?;
	if graph.nodes.is_empty() {
		return Ok(samples[..rows * graph.output.elements()].to_vec());
	}
	let _ = targets;
	let tape = NativeTape::new(graph, &samples[..input_count], &samples[..input_count], &[], gpu, precision, None)?;
	tape.forward()?;
	tape.predictions()
}
fn surrogate_model(hidden: usize) -> Model {
	recipe.model().layer(hidden).tanh().layer(1)
}
fn fit_surrogate(input: Shape, samples: &[f64], targets: &[f64], hidden: usize, gpu: &'static Gpu, config: Config) -> Result<Graph> {
	require(!targets.is_empty(), "surrogate requires teacher outputs")?;
	let sample_count = checked_mul(targets.len(), input.elements(), "surrogate samples")?;
	require(samples.len() == sample_count, "surrogate sample batch is invalid")?;
	let model = surrogate_model(hidden);
	let prepared = Prepared {
		samples: samples.to_vec(),
		targets: targets.to_vec(),
		target_width: 1,
		rows: targets.len(),
		source_rows: targets.len(),
		features: input.elements(),
		schema: DataSchema::default(),
		sequence: None,
		target_categorical: false,
		norm_mean: Vec::new(),
		norm_scale: Vec::new(),
		identities: Vec::new(),
		fitted: Vec::new(),
		bound: None,
	};
	let mut graph = compile(&model, &prepared, targets, prepared.rows, gpu, config, true)?;
	let mut tape = NativeTape::new(&graph, samples, samples, targets, gpu, config.precision, Some(mse))?;
	for _ in 0..config.surrogate_epochs {
		tape.advance()?;
		tape.full_epoch(config.surrogate_rate, config)?;
	}
	tape.capture(&mut graph)?;
	graph.frozen.fill(1);
	Ok(graph)
}
#[derive(Clone)]
struct NearestNode {
	minimum: u32,
	start: u32,
	end: u32,
	split: Option<(usize, f64, Box<NearestNode>, Box<NearestNode>)>,
}
#[derive(Clone)]
struct NearestIndex {
	features: usize,
	permutation: Vec<u32>,
	root: NearestNode,
}
impl NearestIndex {
	fn build(samples: &[f64], features: usize, rows: usize) -> Self {
		let mut permutation = (0..rows as u32).collect::<Vec<_>>();
		let root = Self::partition(samples, features, &mut permutation, 0);
		Self { features, permutation, root }
	}
	fn partition(samples: &[f64], features: usize, permutation: &mut [u32], start: u32) -> NearestNode {
		let minimum = permutation.iter().copied().min().unwrap_or(0);
		let end = start + permutation.len() as u32;
		if permutation.len() <= 16 {
			return NearestNode { minimum, start, end, split: None };
		}
		let mut widest = (f64::NEG_INFINITY, 0);
		for feature in 0..features {
			let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
			for &row in permutation.iter() {
				let value = samples[row as usize * features + feature];
				(low, high) = (low.min(value), high.max(value));
			}
			widest = if high - low > widest.0 { (high - low, feature) } else { widest };
		}
		let (dimension, middle) = (widest.1, permutation.len() / 2);
		permutation.select_nth_unstable_by(middle, |&a, &b| samples[a as usize * features + dimension].total_cmp(&samples[b as usize * features + dimension]).then(a.cmp(&b)));
		let threshold = samples[permutation[middle] as usize * features + dimension];
		let (left, right) = permutation.split_at_mut(middle);
		let left = Box::new(Self::partition(samples, features, left, start));
		let right = Box::new(Self::partition(samples, features, right, start + middle as u32));
		NearestNode { minimum, start, end, split: Some((dimension, threshold, left, right)) }
	}
	fn nearest(&self, node: &NearestNode, samples: &[f64], query: &[f64], row: usize, count: usize, exclude: bool, best: &mut Vec<(f64, u32)>) {
		let Some((dimension, threshold, left, right)) = &node.split else {
			for &candidate in &self.permutation[node.start as usize..node.end as usize] {
				if exclude && candidate as usize == row {
					continue;
				}
				let base = candidate as usize * self.features;
				let measured = distance(query, &samples[base..base + self.features]);
				let keeps = best.len() < count || best.last().is_some_and(|&(kept, index)| measured < kept || (measured == kept && candidate < index));
				if measured < f64::MAX && keeps {
					let position = best.iter().position(|&(kept, index)| kept > measured || (kept == measured && index > candidate)).unwrap_or(best.len());
					best.insert(position, (measured, candidate));
					best.truncate(count);
				}
			}
			return;
		};
		// On an equal coordinate the left child holds the lower row indices, so it is
		// searched first to settle tie-breaks before the far bound is consulted.
		let (near, far) = if query[*dimension] <= *threshold { (left, right) } else { (right, left) };
		self.nearest(near, samples, query, row, count, exclude, best);
		// The squared split coordinate gap lower-bounds every far-side distance, so the
		// far subtree is skipped only when no far row can displace or tie a kept one.
		if best.len() == count {
			if let Some(&(kept, index)) = best.last() {
				let gap = (query[*dimension] - threshold).powi(2);
				if gap > kept || (gap == kept && far.minimum > index) {
					return;
				}
			}
		}
		self.nearest(far, samples, query, row, count, exclude, best);
	}
}
#[derive(Clone)]
struct PredictorProgram {
	code: Vec<f64>,
	locals: usize,
	stack: usize,
	table: Vec<f64>,
	nearest: Option<NearestIndex>,
}
impl PredictorProgram {
	fn evaluate(&self, row: usize, query: &[f64]) -> Result<f64> {
		let mut locals = vec![0.0; self.locals];
		let mut stack = Vec::with_capacity(self.stack);
		let pop = |stack: &mut Vec<f64>| stack.pop().ok_or_else(|| RecipeError::new("predictor stack underflows"));
		for instruction in self.code.chunks_exact(2) {
			let opcode = structural(instruction[0])?;
			let slot = || structural(instruction[1]).and_then(|value| usize::try_from(value).map_err(|_| RecipeError::new("predictor index is negative")));
			match opcode {
				value if value == PredictorOpcode::Feature as i32 => stack.push(*query.get(slot()?).ok_or_else(|| RecipeError::new("predictor feature is absent"))?),
				value if value == PredictorOpcode::Row as i32 => stack.push(row as f64),
				value if value == PredictorOpcode::Constant as i32 => stack.push(instruction[1]),
				value if value == PredictorOpcode::Load as i32 => stack.push(*locals.get(slot()?).ok_or_else(|| RecipeError::new("predictor local is absent"))?),
				value if value == PredictorOpcode::Store as i32 => {
					let value = pop(&mut stack)?;
					*locals.get_mut(slot()?).ok_or_else(|| RecipeError::new("predictor local is absent"))? = value
				}
				value if value == PredictorOpcode::Duplicate as i32 => stack.push(*stack.last().ok_or_else(|| RecipeError::new("predictor stack underflows"))?),
				value if value == PredictorOpcode::Add as i32 => {
					let right = pop(&mut stack)?;
					let left = pop(&mut stack)?;
					stack.push(left + right)
				}
				value if value == PredictorOpcode::Subtract as i32 => {
					let right = pop(&mut stack)?;
					let left = pop(&mut stack)?;
					stack.push(left - right)
				}
				value if value == PredictorOpcode::Multiply as i32 => {
					let right = pop(&mut stack)?;
					let left = pop(&mut stack)?;
					stack.push(left * right)
				}
				value if value == PredictorOpcode::Divide as i32 => {
					let right = pop(&mut stack)?;
					let left = pop(&mut stack)?;
					stack.push(left / right)
				}
				value if value == PredictorOpcode::Greater as i32 => {
					let right = pop(&mut stack)?;
					let left = pop(&mut stack)?;
					stack.push(f64::from(left > right))
				}
				value if value == PredictorOpcode::Choose as i32 => {
					let no = pop(&mut stack)?;
					let yes = pop(&mut stack)?;
					let condition = pop(&mut stack)?;
					stack.push(if condition != 0.0 { yes } else { no })
				}
				value if value == PredictorOpcode::Nearest as i32 => {
					let count = instruction[1].abs() as usize;
					let exclude = instruction[1] < 0.0;
					let features = query.len();
					let rows = self
						.table
						.len()
						.checked_div(features + 1)
						.filter(|rows| rows * (features + 1) == self.table.len())
						.ok_or_else(|| RecipeError::new("nearest table width is invalid"))?;
					let (samples, targets) = self.table.split_at(rows * features);
					// An index is built beside the table it searches and neither is replaced afterwards, so a
					// program that carries an absent or differently shaped index rebuilds from the table it holds
					// now rather than answering from one that never saw these values.
					let rebuilt =
						self.nearest.as_ref().filter(|index| index.features == features && index.permutation.len() == rows).is_none().then(|| NearestIndex::build(samples, features, rows));
					let index = rebuilt.as_ref().or(self.nearest.as_ref()).ok_or_else(|| RecipeError::new("nearest index is absent"))?;
					let mut best = Vec::with_capacity(count);
					index.nearest(&index.root, samples, query, row, count, exclude, &mut best);
					stack.push((0..count).map(|slot| best.get(slot).map_or(0.0, |&(_, candidate)| targets[candidate as usize])).sum::<f64>() / count as f64)
				}
				value if value == PredictorOpcode::Affine as i32 => {
					require(self.table.len() == 3 * query.len() && !query.is_empty(), "affine table width is invalid")?;
					let (means, rest) = self.table.split_at(query.len());
					let (scales, weights) = rest.split_at(query.len());
					stack.push(query.iter().zip(means).zip(scales).zip(weights).map(|(((value, mean), scale), weight)| (value - mean) * scale * weight).sum())
				}
				value if value == PredictorOpcode::Gaussian as i32 => {
					let features = query.len();
					let width = 2 * features + 2;
					let classes = self
						.table
						.len()
						.checked_div(width)
						.filter(|classes| classes * width == self.table.len() && *classes != 0 && features != 0)
						.ok_or_else(|| RecipeError::new("gaussian table width is invalid"))?;
					let (means, rest) = self.table.split_at(classes * features);
					let (scales, rest) = rest.split_at(classes * features);
					let (bases, labels) = rest.split_at(classes);
					let mut best = (f64::MIN, labels[0]);
					for class in 0..classes {
						let score = query
							.iter()
							.zip(&means[class * features..])
							.zip(&scales[class * features..])
							.fold(bases[class], |sum, ((value, mean), scale)| sum + (value - mean) * (value - mean) * scale);
						if score > best.0 {
							best = (score, labels[class])
						}
					}
					stack.push(best.1)
				}
				_ => return Err(RecipeError::new(format!("invalid predictor opcode {opcode}"))),
			}
		}
		require(stack.len() == 1, "predictor stack has the wrong final depth")?;
		finite_prediction(stack[0])
	}
}
fn finite_prediction(value: f64) -> Result<f64> {
	require(value.is_finite(), "predictor produced a nonfinite value").map(|_| value)
}
struct PredictorBuilder {
	code: Vec<f64>,
	locals: usize,
	depth: usize,
	stack: usize,
	table: Vec<f64>,
	index: Option<NearestIndex>,
}
impl PredictorBuilder {
	fn new() -> Self {
		Self { code: Vec::new(), locals: 0, depth: 0, stack: 0, table: Vec::new(), index: None }
	}
	fn nearest(mut self, count: usize, exclude: bool, features: usize, table: Vec<f64>) -> Result<PredictorProgram> {
		(self.index, self.table) = (Some(NearestIndex::build(&table, features, table.len() / (features + 1))), table);
		self.push(PredictorOpcode::Nearest, if exclude { -(count as f64) } else { count as f64 });
		self.finish()
	}
	fn affine(&mut self, table: Vec<f64>) {
		(self.index, self.table) = (None, table);
		self.push(PredictorOpcode::Affine, 0.0);
	}
	fn gaussian(&mut self, table: Vec<f64>) {
		(self.index, self.table) = (None, table);
		self.push(PredictorOpcode::Gaussian, 0.0);
	}
	fn emit(&mut self, opcode: PredictorOpcode, argument: f64) {
		self.code.extend([opcode as i32 as f64, argument])
	}
	fn push(&mut self, opcode: PredictorOpcode, argument: f64) {
		self.emit(opcode, argument);
		self.depth += 1;
		self.stack = self.stack.max(self.depth)
	}
	fn feature(&mut self, index: usize) {
		self.push(PredictorOpcode::Feature, index as f64)
	}
	fn constant(&mut self, value: f64) {
		self.push(PredictorOpcode::Constant, value)
	}
	fn binary(&mut self, opcode: PredictorOpcode) {
		self.emit(opcode, 0.0);
		self.depth -= 1
	}
	fn choose(&mut self) {
		self.emit(PredictorOpcode::Choose, 0.0);
		self.depth -= 2
	}
	fn finish(self) -> Result<PredictorProgram> {
		require(self.depth == 1 && self.stack != 0 && self.code.len() % 2 == 0, "predictor program is invalid")?;
		require(self.code.chunks_exact(2).all(|instruction| instruction[0].is_finite() && instruction[1].is_finite()), "predictor program contains a nonfinite value")?;
		Ok(PredictorProgram { code: self.code, locals: self.locals, stack: self.stack, table: self.table, nearest: self.index })
	}
}
struct Predictor {
	program: PredictorProgram,
	predict: Box<dyn Fn(usize, &[f64]) -> Result<f64> + Send + Sync>,
}
impl Predictor {
	fn new(mut program: PredictorProgram) -> Self {
		let evaluator = PredictorProgram { nearest: program.nearest.take(), code: program.code.clone(), locals: program.locals, stack: program.stack, table: program.table.clone() };
		Self { program, predict: Box::new(move |row, query| evaluator.evaluate(row, query)) }
	}
	// The fitted model answers teacher queries directly, so labeling never interprets the lowered program.
	fn fitted(program: PredictorProgram, teacher: impl Fn(&[f64]) -> f64 + Send + Sync + 'static) -> Self {
		Self { program, predict: Box::new(move |_, query| finite_prediction(teacher(query))) }
	}
}
#[derive(Clone)]
enum TreeNode {
	Leaf(f64),
	Split { feature: usize, threshold: f64, left: Box<TreeNode>, right: Box<TreeNode> },
}
fn tree_mean(rows: &[usize], targets: &[f64]) -> f64 {
	rows.iter().map(|&row| targets[row]).sum::<f64>() / rows.len() as f64
}
fn tree_error(rows: &[usize], targets: &[f64]) -> f64 {
	let mean = tree_mean(rows, targets);
	rows.iter().map(|&row| (targets[row] - mean).powi(2)).sum()
}
fn fit_tree(samples: &[f64], targets: &[f64], features: usize, rows: &[usize], depth: usize, candidates: &[usize], minimum: usize) -> TreeNode {
	if depth == 0 || rows.len() < 2 * minimum {
		return TreeNode::Leaf(tree_mean(rows, targets));
	}
	let mut best = None;
	for &feature in candidates {
		let mut ordered = rows.iter().map(|&row| (samples[row * features + feature], row)).collect::<Vec<_>>();
		ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
		let (mut left_sum, mut left_square) = (0.0, 0.0);
		let (total_sum, total_square) = ordered.iter().map(|value| targets[value.1]).fold((0.0, 0.0), |(sum, square), value| (sum + value, square + value * value));
		for split in 1..ordered.len() {
			let target = targets[ordered[split - 1].1];
			left_sum += target;
			left_square += target * target;
			if split < minimum || ordered.len() - split < minimum || ordered[split - 1].0 >= ordered[split].0 {
				continue;
			}
			let right_sum = total_sum - left_sum;
			let right_square = total_square - left_square;
			let error = left_square - left_sum * left_sum / split as f64 + right_square - right_sum * right_sum / (ordered.len() - split) as f64;
			if best.as_ref().is_none_or(|value: &(f64, usize, f64)| error < value.0) {
				best = Some((error, feature, (ordered[split - 1].0 + ordered[split].0) * 0.5));
			}
		}
	}
	let Some((error, feature, threshold)) = best else { return TreeNode::Leaf(tree_mean(rows, targets)) };
	if error >= tree_error(rows, targets) {
		return TreeNode::Leaf(tree_mean(rows, targets));
	}
	let (left, right): (Vec<_>, Vec<_>) = rows.iter().copied().partition(|row| samples[row * features + feature] < threshold);
	TreeNode::Split {
		feature,
		threshold,
		left: Box::new(fit_tree(samples, targets, features, &left, depth - 1, candidates, minimum)),
		right: Box::new(fit_tree(samples, targets, features, &right, depth - 1, candidates, minimum)),
	}
}
fn emit_tree(node: &TreeNode, program: &mut PredictorBuilder) {
	match node {
		TreeNode::Leaf(value) => program.constant(*value),
		TreeNode::Split { feature, threshold, left, right } => {
			program.constant(*threshold);
			program.feature(*feature);
			program.binary(PredictorOpcode::Greater);
			emit_tree(left, program);
			emit_tree(right, program);
			program.choose();
		}
	}
}
fn next_random(state: &mut u64) -> u64 {
	*state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
	*state
}
fn valid_estimator(_: usize, _: usize) -> Result<()> {
	Ok(())
}
fn positive_estimator(value: usize, _: usize) -> Result<()> {
	require(value != 0, format!("estimator count {value} is invalid"))
}
fn cluster_estimator(value: usize, rows: usize) -> Result<()> {
	require(value != 0 && value <= rows, format!("kmeans cluster count {value} is invalid for {rows} training rows"))
}
fn neighbor_estimator(value: usize, rows: usize) -> Result<()> {
	require(value != 0 && value < rows, format!("knn neighbor count {value} is invalid for {rows} training rows"))
}
fn fit_svm(_: usize, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
	require(rows != 0 && data.features != 0, "SVM requires training rows and features")?;
	let mut means = vec![0.0; data.features];
	for sample in data.samples[..rows * data.features].chunks_exact(data.features) {
		for (mean, value) in means.iter_mut().zip(sample) {
			*mean += value / rows as f64
		}
	}
	let mut inverse = vec![0.0; data.features];
	for sample in data.samples[..rows * data.features].chunks_exact(data.features) {
		for ((variance, value), mean) in inverse.iter_mut().zip(sample).zip(&means) {
			*variance += (value - mean).powi(2) / rows as f64
		}
	}
	// Regularize the variance like normalize_samples does: an exact-zero guard
	// misses float-residue variances on numerically constant features, whose
	// unbounded inverses cannot survive the model's storage format.
	let epsilon = number("normalization epsilon", env!("RECIPE_NORMALIZATION_EPSILON"))?;
	for value in &mut inverse {
		*value = (*value + epsilon).sqrt().recip()
	}
	let mut weights = vec![0.0; data.features];
	let mut bias = data.targets[..rows].iter().sum::<f64>() / rows as f64;
	for _ in 0..config.svm_iterations {
		let mut gradient = weights.iter().map(|weight| config.svm_regularization * weight).collect::<Vec<_>>();
		let mut bias_gradient = 0.0;
		for (sample, target) in data.samples[..rows * data.features].chunks_exact(data.features).zip(&data.targets[..rows]) {
			let prediction = bias + weights.iter().zip(sample).zip(&means).zip(&inverse).map(|(((weight, value), mean), scale)| weight * (value - mean) * scale).sum::<f64>();
			let error = prediction - target;
			let direction = if error > config.svm_epsilon {
				1.0
			} else if error < -config.svm_epsilon {
				-1.0
			} else {
				0.0
			};
			bias_gradient += direction / rows as f64;
			for (((value, mean), scale), value_gradient) in sample.iter().zip(&means).zip(&inverse).zip(&mut gradient) {
				*value_gradient += direction * (value - mean) * scale / rows as f64
			}
		}
		bias -= config.svm_rate * bias_gradient;
		for (weight, gradient) in weights.iter_mut().zip(gradient) {
			*weight -= config.svm_rate * gradient
		}
	}
	// The fitted model lives in the predictor table as three feature-length
	// planes (means, scales, weights), so the emitted program is a fixed-size
	// feature loop instead of straight-line code that grows with the feature
	// count, and each storage block quantizes values of one magnitude family.
	let mut table = Vec::with_capacity(3 * data.features);
	table.extend(means);
	table.extend(inverse);
	table.extend(weights);
	let mut program = PredictorBuilder::new();
	program.constant(bias);
	program.affine(table);
	program.binary(PredictorOpcode::Add);
	Ok(Predictor::new(program.finish()?))
}
fn fit_forest(trees: usize, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
	require(rows >= config.tree_min_rows && data.features != 0, "forest requires enough training rows and features")?;
	let feature_count = ((data.features as f64 * config.forest_feature_fraction).ceil() as usize).clamp(1, data.features);
	let mut state = config.random_seed as u64;
	let mut forest = Vec::with_capacity(trees);
	for _ in 0..trees {
		let sampled = (0..rows).map(|_| next_random(&mut state) as usize % rows).collect::<Vec<_>>();
		let mut candidates = (0..data.features).collect::<Vec<_>>();
		for index in (1..candidates.len()).rev() {
			candidates.swap(index, next_random(&mut state) as usize % (index + 1));
		}
		candidates.truncate(feature_count);
		forest.push(fit_tree(&data.samples, &data.targets, data.features, &sampled, config.tree_depth, &candidates, config.tree_min_rows));
	}
	let mut program = PredictorBuilder::new();
	program.constant(0.0);
	for tree in &forest {
		emit_tree(tree, &mut program);
		program.binary(PredictorOpcode::Add);
	}
	program.constant(trees as f64);
	program.binary(PredictorOpcode::Divide);
	Ok(Predictor::fitted(program.finish()?, move |sample| forest.iter().fold(0.0, |sum, tree| sum + tree_predict(tree, sample)) / trees as f64))
}
fn solve_linear(mut matrix: Vec<f64>, mut values: Vec<f64>, epsilon: f64) -> Result<Vec<f64>> {
	let width = values.len();
	require(matrix.len() == width * width && width != 0, "linear system shape is invalid")?;
	for column in 0..width {
		let pivot = (column..width)
			.max_by(|left, right| matrix[*left * width + column].abs().total_cmp(&matrix[*right * width + column].abs()))
			.ok_or_else(|| RecipeError::new("linear system has no pivot"))?;
		require(matrix[pivot * width + column].abs() > epsilon, "linear system is singular")?;
		for entry in 0..width {
			matrix.swap(column * width + entry, pivot * width + entry)
		}
		values.swap(column, pivot);
		let scale = matrix[column * width + column];
		for entry in column..width {
			matrix[column * width + entry] /= scale
		}
		values[column] /= scale;
		for row in 0..width {
			if row == column {
				continue;
			}
			let factor = matrix[row * width + column];
			for entry in column..width {
				matrix[row * width + entry] -= factor * matrix[column * width + entry]
			}
			values[row] -= factor * values[column];
		}
	}
	require(values.iter().all(|value| value.is_finite()), "linear system produced a nonfinite solution").map(|_| values)
}
fn fit_bayes(_: usize, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
	require(rows != 0 && data.features != 0, "Bayes requires training rows and features")?;
	if !data.target_categorical {
		let mut means = vec![0.0; data.features];
		let target_mean = data.targets[..rows].iter().sum::<f64>() / rows as f64;
		for sample in data.samples[..rows * data.features].chunks_exact(data.features) {
			for (mean, value) in means.iter_mut().zip(sample) {
				*mean += value / rows as f64
			}
		}
		// The covariance is symmetric and the noise variance is invariant, so only
		// the upper triangle accumulates and the noise divides each sum once.
		let mut matrix = vec![0.0; data.features * data.features];
		let mut values = vec![0.0; data.features];
		let mut centered = vec![0.0; data.features];
		for (sample, target) in data.samples[..rows * data.features].chunks_exact(data.features).zip(&data.targets[..rows]) {
			for (centered, (value, mean)) in centered.iter_mut().zip(sample.iter().zip(&means)) {
				*centered = value - mean
			}
			for left in 0..data.features {
				values[left] += centered[left] * (target - target_mean);
				for right in left..data.features {
					matrix[left * data.features + right] += centered[left] * centered[right]
				}
			}
		}
		for left in 0..data.features {
			values[left] /= config.bayes_noise_variance;
			for right in left..data.features {
				let entry = matrix[left * data.features + right] / config.bayes_noise_variance;
				matrix[left * data.features + right] = entry;
				matrix[right * data.features + left] = entry;
			}
			matrix[left * data.features + left] += config.bayes_prior_precision
		}
		let weights = solve_linear(matrix, values, config.bayes_variance_epsilon)?;
		let mut table = vec![1.0; 2 * data.features];
		table[..data.features].copy_from_slice(&means);
		table.extend(weights);
		let mut program = PredictorBuilder::new();
		program.constant(target_mean);
		program.affine(table);
		program.binary(PredictorOpcode::Add);
		return Ok(Predictor::new(program.finish()?));
	}
	let mut classes = data.targets[..rows].to_vec();
	classes.sort_by(f64::total_cmp);
	classes.dedup_by(|left, right| left.to_bits() == right.to_bits());
	require(!classes.is_empty(), "Bayes has no target class")?;
	let (scales, bases, labels) = (classes.len() * data.features, 2 * classes.len() * data.features, 2 * classes.len() * data.features + classes.len());
	let mut table = vec![0.0; classes.len() * (2 * data.features + 2)];
	for (index, &class) in classes.iter().enumerate() {
		let members = data.targets[..rows].iter().enumerate().filter_map(|(row, target)| (target.to_bits() == class.to_bits()).then_some(row)).collect::<Vec<_>>();
		let mut means = vec![0.0; data.features];
		for &row in &members {
			for feature in 0..data.features {
				means[feature] += data.samples[row * data.features + feature] / members.len() as f64
			}
		}
		let mut variance = vec![config.bayes_variance_epsilon; data.features];
		for &row in &members {
			for feature in 0..data.features {
				variance[feature] += (data.samples[row * data.features + feature] - means[feature]).powi(2) / members.len() as f64
			}
		}
		table[bases + index] = (members.len() as f64 / rows as f64).ln() - 0.5 * variance.iter().map(|value| value.ln()).sum::<f64>();
		table[labels + index] = class;
		for (feature, variance) in variance.into_iter().enumerate() {
			table[scales + index * data.features + feature] = -0.5 * variance.recip()
		}
		table[index * data.features..(index + 1) * data.features].copy_from_slice(&means);
	}
	let mut program = PredictorBuilder::new();
	program.gaussian(table);
	Ok(Predictor::new(program.finish()?))
}
fn tree_predict(tree: &TreeNode, sample: &[f64]) -> f64 {
	match tree {
		TreeNode::Leaf(value) => *value,
		TreeNode::Split { feature, threshold, left, right } => tree_predict(if sample[*feature] < *threshold { left } else { right }, sample),
	}
}
fn boosted_predictor(base: f64, trees: &[TreeNode], rate: f64) -> Result<PredictorProgram> {
	let mut program = PredictorBuilder::new();
	program.constant(base);
	for tree in trees {
		emit_tree(tree, &mut program);
		program.constant(rate);
		program.binary(PredictorOpcode::Multiply);
		program.binary(PredictorOpcode::Add);
	}
	program.finish()
}
fn xgboost_leaf(rows: &[usize], gradients: &[f64], regularization: f64) -> f64 {
	-rows.iter().map(|&row| gradients[row]).sum::<f64>() / (rows.len() as f64 + regularization)
}
fn fit_xgboost_tree(samples: &[f64], gradients: &[f64], features: usize, rows: &[usize], depth: usize, minimum: usize, regularization: f64, minimum_gain: f64) -> TreeNode {
	if depth == 0 || rows.len() < 2 * minimum {
		return TreeNode::Leaf(xgboost_leaf(rows, gradients, regularization));
	}
	let total = rows.iter().map(|&row| gradients[row]).sum::<f64>();
	let parent = total * total / (rows.len() as f64 + regularization);
	let mut best = None;
	for feature in 0..features {
		let mut ordered = rows.iter().map(|&row| (samples[row * features + feature], row)).collect::<Vec<_>>();
		ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
		let mut left = 0.0;
		for split in 1..ordered.len() {
			left += gradients[ordered[split - 1].1];
			if split < minimum || ordered.len() - split < minimum || ordered[split - 1].0 >= ordered[split].0 {
				continue;
			}
			let right = total - left;
			let gain = 0.5 * (left * left / (split as f64 + regularization) + right * right / ((ordered.len() - split) as f64 + regularization) - parent);
			if gain > minimum_gain && best.as_ref().is_none_or(|value: &(f64, usize, f64)| gain > value.0) {
				best = Some((gain, feature, (ordered[split - 1].0 + ordered[split].0) * 0.5))
			}
		}
	}
	let Some((_, feature, threshold)) = best else { return TreeNode::Leaf(xgboost_leaf(rows, gradients, regularization)) };
	let (left, right): (Vec<_>, Vec<_>) = rows.iter().copied().partition(|row| samples[row * features + feature] < threshold);
	TreeNode::Split {
		feature,
		threshold,
		left: Box::new(fit_xgboost_tree(samples, gradients, features, &left, depth - 1, minimum, regularization, minimum_gain)),
		right: Box::new(fit_xgboost_tree(samples, gradients, features, &right, depth - 1, minimum, regularization, minimum_gain)),
	}
}
fn fit_xgboost(_: usize, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
	require(rows >= config.tree_min_rows && data.features != 0, "XGBoost requires enough training rows and features")?;
	let base = data.targets[..rows].iter().sum::<f64>() / rows as f64;
	let mut predictions = vec![base; rows];
	let indices = (0..rows).collect::<Vec<_>>();
	let mut trees = Vec::with_capacity(config.boost_iterations);
	for _ in 0..config.boost_iterations {
		let gradients = predictions.iter().zip(&data.targets[..rows]).map(|(prediction, target)| prediction - target).collect::<Vec<_>>();
		let tree = fit_xgboost_tree(&data.samples, &gradients, data.features, &indices, config.tree_depth, config.tree_min_rows, config.xgboost_regularization, config.xgboost_min_gain);
		for (row, sample) in data.samples[..rows * data.features].chunks_exact(data.features).enumerate() {
			predictions[row] += config.boost_rate * tree_predict(&tree, sample)
		}
		trees.push(tree);
	}
	Ok(Predictor::fitted(boosted_predictor(base, &trees, config.boost_rate)?, move |sample| trees.iter().fold(base, |value, tree| value + config.boost_rate * tree_predict(tree, sample))))
}
struct LightNode {
	rows: Vec<usize>,
	value: f64,
	split: Option<(usize, f64, usize, usize)>,
}
fn lightgbm_split(samples: &[f64], residuals: &[f64], features: usize, rows: &[usize], bins: usize, minimum: usize) -> Option<(f64, usize, f64, Vec<usize>, Vec<usize>)> {
	let total = rows.iter().map(|&row| residuals[row]).sum::<f64>();
	let square = rows.iter().map(|&row| residuals[row].powi(2)).sum::<f64>();
	let parent = square - total * total / rows.len() as f64;
	let mut best = None;
	for feature in 0..features {
		let minimum_value = rows.iter().map(|&row| samples[row * features + feature]).fold(f64::INFINITY, f64::min);
		let maximum_value = rows.iter().map(|&row| samples[row * features + feature]).fold(f64::NEG_INFINITY, f64::max);
		if minimum_value >= maximum_value {
			continue;
		}
		let width = (maximum_value - minimum_value) / bins as f64;
		let mut counts = vec![0_usize; bins];
		let mut sums = vec![0.0; bins];
		let mut squares = vec![0.0; bins];
		for &row in rows {
			let bin = (((samples[row * features + feature] - minimum_value) / width).floor() as usize).min(bins - 1);
			counts[bin] += 1;
			sums[bin] += residuals[row];
			squares[bin] += residuals[row].powi(2);
		}
		let (mut left_count, mut left_sum, mut left_square) = (0, 0.0, 0.0);
		for bin in 0..bins - 1 {
			left_count += counts[bin];
			left_sum += sums[bin];
			left_square += squares[bin];
			let right_count = rows.len() - left_count;
			if left_count < minimum || right_count < minimum {
				continue;
			}
			let right_sum = total - left_sum;
			let right_square = square - left_square;
			let gain = parent - (left_square - left_sum * left_sum / left_count as f64) - (right_square - right_sum * right_sum / right_count as f64);
			let threshold = minimum_value + width * (bin + 1) as f64;
			if gain > 0.0 && best.as_ref().is_none_or(|value: &(f64, usize, f64)| gain > value.0) {
				best = Some((gain, feature, threshold))
			}
		}
	}
	best.map(|(gain, feature, threshold)| {
		let (left, right) = rows.iter().copied().partition(|row| samples[row * features + feature] < threshold);
		(gain, feature, threshold, left, right)
	})
}
fn materialize_lightgbm(nodes: &[LightNode], index: usize) -> TreeNode {
	match nodes[index].split {
		Some((feature, threshold, left, right)) => TreeNode::Split { feature, threshold, left: Box::new(materialize_lightgbm(nodes, left)), right: Box::new(materialize_lightgbm(nodes, right)) },
		None => TreeNode::Leaf(nodes[index].value),
	}
}
fn fit_lightgbm(_: usize, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
	require(config.lightgbm_bins >= 2, "LightGBM histogram bins must be at least two")?;
	require(config.lightgbm_leaves >= 2 && rows >= config.tree_min_rows && data.features != 0, "LightGBM requires at least two leaves and enough training rows and features")?;
	let base = data.targets[..rows].iter().sum::<f64>() / rows as f64;
	let mut predictions = vec![base; rows];
	let mut trees = Vec::with_capacity(config.boost_iterations);
	for _ in 0..config.boost_iterations {
		let residuals = data.targets[..rows].iter().zip(&predictions).map(|(target, prediction)| target - prediction).collect::<Vec<_>>();
		let indices = (0..rows).collect::<Vec<_>>();
		let mut nodes = vec![LightNode { value: tree_mean(&indices, &residuals), rows: indices, split: None }];
		for _ in 1..config.lightgbm_leaves {
			let selected = nodes
				.iter()
				.enumerate()
				.filter(|(_, node)| node.split.is_none())
				.filter_map(|(index, node)| lightgbm_split(&data.samples, &residuals, data.features, &node.rows, config.lightgbm_bins, config.tree_min_rows).map(|split| (index, split)))
				.max_by(|left, right| left.1.0.total_cmp(&right.1.0));
			let Some((index, (_, feature, threshold, left, right))) = selected else { break };
			let left_index = nodes.len();
			nodes.push(LightNode { value: tree_mean(&left, &residuals), rows: left, split: None });
			let right_index = nodes.len();
			nodes.push(LightNode { value: tree_mean(&right, &residuals), rows: right, split: None });
			nodes[index].split = Some((feature, threshold, left_index, right_index));
		}
		let tree = materialize_lightgbm(&nodes, 0);
		for (row, sample) in data.samples[..rows * data.features].chunks_exact(data.features).enumerate() {
			predictions[row] += config.boost_rate * tree_predict(&tree, sample)
		}
		trees.push(tree);
	}
	boosted_predictor(base, &trees, config.boost_rate).map(Predictor::new)
}
/// Candidate split thresholds for one feature of a CatBoost fit. CatBoost
/// quantizes every feature into a bounded border set before growing trees, so
/// the ordered error scan visits `count` candidates per feature instead of one
/// per distinct value, which on a continuous feature would make the scan
/// quadratic in the row count. A feature with at most `count` distinct
/// midpoints keeps all of them. `bins[row]` is the number of thresholds the
/// row's value does not fall left of, so the row sits left of threshold
/// `index` exactly when `bins[row] <= index`.
struct CatboostBorders {
	thresholds: Vec<f64>,
	bins: Vec<usize>,
}
fn catboost_borders(samples: &[f64], features: usize, rows: usize, count: usize) -> Vec<CatboostBorders> {
	(0..features)
		.map(|feature| {
			let mut values = (0..rows).map(|row| samples[row * features + feature]).collect::<Vec<_>>();
			values.sort_by(f64::total_cmp);
			values.dedup_by(|left, right| left.to_bits() == right.to_bits());
			let midpoints = values.len().saturating_sub(1);
			// Rank-spaced positions keep every candidate when the feature is small
			// and pick evenly spread interior quantiles when it is not.
			let thresholds = (0..midpoints.min(count))
				.map(|index| {
					let position = if midpoints <= count { index } else { (index + 1) * midpoints / (count + 1) };
					(values[position] + values[position + 1]) * 0.5
				})
				.collect::<Vec<f64>>();
			let bins = (0..rows).map(|row| thresholds.partition_point(|threshold| !(samples[row * features + feature] < *threshold))).collect();
			CatboostBorders { thresholds, bins }
		})
		.collect()
}
fn ordered_split(borders: &[CatboostBorders], residuals: &[f64], permutation: &[usize], codes: &[usize], level: usize, prior: f64, minimum: usize) -> Result<Option<(usize, f64)>> {
	let Some(groups) = 1_usize.checked_shl((level + 1) as u32) else { return Ok(None) };
	// Each feature's candidate scan is independent; the reduction keeps the first
	// strict minimum in feature order, matching the sequential scan.
	Ok(parallel_map(borders.len(), |feature| {
		let candidates = &borders[feature];
		let (mut counts, mut sums, mut best) = (vec![0_usize; groups], vec![0.0; groups], None);
		for (index, &threshold) in candidates.thresholds.iter().enumerate() {
			counts.fill(0);
			sums.fill(0.0);
			let mut error = 0.0;
			for &row in permutation {
				let group = codes[row] | usize::from(candidates.bins[row] <= index) << level;
				error += (residuals[row] - sums[group] / (counts[group] as f64 + prior)).powi(2);
				sums[group] += residuals[row];
				counts[group] += 1;
			}
			if counts.iter().filter(|count| **count != 0).all(|count| *count >= minimum) && best.as_ref().is_none_or(|value: &(f64, f64)| error < value.0) {
				best = Some((error, threshold))
			}
		}
		best
	})?
	.into_iter()
	.enumerate()
	.filter_map(|(feature, best)| best.map(|(error, threshold)| (error, feature, threshold)))
	.reduce(|best, candidate| if candidate.0 < best.0 { candidate } else { best })
	.map(|(_, feature, threshold)| (feature, threshold)))
}
fn oblivious_tree(splits: &[(usize, f64)], leaves: &[f64], level: usize, code: usize) -> TreeNode {
	if level == splits.len() {
		return TreeNode::Leaf(leaves[code]);
	}
	let (feature, threshold) = splits[level];
	TreeNode::Split { feature, threshold, left: Box::new(oblivious_tree(splits, leaves, level + 1, code | 1 << level)), right: Box::new(oblivious_tree(splits, leaves, level + 1, code)) }
}
fn fit_catboost(_: usize, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
	require(rows >= config.tree_min_rows && data.features != 0, "CatBoost requires enough training rows and features")?;
	require(config.tree_depth < usize::BITS as usize, "CatBoost tree depth is too large")?;
	let borders = catboost_borders(&data.samples, data.features, rows, config.catboost_borders);
	let base = data.targets[..rows].iter().sum::<f64>() / rows as f64;
	let mut predictions = vec![base; rows];
	let mut state = config.random_seed as u64;
	let mut trees = Vec::with_capacity(config.boost_iterations);
	for _ in 0..config.boost_iterations {
		let residuals = data.targets[..rows].iter().zip(&predictions).map(|(target, prediction)| target - prediction).collect::<Vec<_>>();
		let mut permutation = (0..rows).collect::<Vec<_>>();
		for index in (1..permutation.len()).rev() {
			permutation.swap(index, next_random(&mut state) as usize % (index + 1))
		}
		let mut codes = vec![0_usize; rows];
		let mut splits = Vec::with_capacity(config.tree_depth);
		for level in 0..config.tree_depth {
			let Some(split) = ordered_split(&borders, &residuals, &permutation, &codes, level, config.catboost_prior, config.tree_min_rows)? else { break };
			for row in 0..rows {
				codes[row] |= usize::from(data.samples[row * data.features + split.0] < split.1) << level
			}
			splits.push(split);
		}
		let leaf_count = 1_usize << splits.len();
		let mut sums = vec![0.0; leaf_count];
		let mut counts = vec![0_usize; leaf_count];
		for &row in &permutation {
			sums[codes[row]] += residuals[row];
			counts[codes[row]] += 1
		}
		let leaves = sums.into_iter().zip(counts).map(|(sum, count)| sum / (count as f64 + config.catboost_prior)).collect::<Vec<_>>();
		let tree = oblivious_tree(&splits, &leaves, 0, 0);
		for (row, sample) in data.samples[..rows * data.features].chunks_exact(data.features).enumerate() {
			predictions[row] += config.boost_rate * tree_predict(&tree, sample)
		}
		trees.push(tree);
	}
	boosted_predictor(base, &trees, config.boost_rate).map(Predictor::new)
}
fn cluster(data: &[f64], width: usize, clusters: usize, iterations: usize, importance: Option<&[f64]>) -> Result<(Vec<f64>, Vec<usize>)> {
	let rows = data.len() / width;
	require(width != 0 && clusters != 0 && clusters <= rows, "kmeans cluster count is invalid")?;
	let (mut centers, mut assignments, mut distances) = (data[..clusters * width].to_vec(), vec![0; rows], vec![0.0; rows]);
	for _ in 0..iterations {
		for (row, sample) in data.chunks_exact(width).enumerate() {
			let selected = nearest(sample, &centers, width);
			assignments[row] = selected.0;
			distances[row] = selected.1;
		}
		for group in 0..clusters {
			let members = assignments.iter().enumerate().filter(|value| *value.1 == group).map(|value| value.0).collect::<Vec<_>>();
			if members.is_empty() {
				let worst = distances.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|value| value.0).ok_or_else(|| RecipeError::new("kmeans has no training row"))?;
				centers[group * width..(group + 1) * width].copy_from_slice(&data[worst * width..(worst + 1) * width]);
				distances[worst] = -1.0;
			} else {
				for feature in 0..width {
					let total = members.iter().map(|&row| importance.map_or(1.0, |weights| weights[row])).sum::<f64>();
					centers[group * width + feature] = members.iter().map(|&row| data[row * width + feature] * importance.map_or(1.0, |weights| weights[row])).sum::<f64>() / total;
				}
			}
		}
	}
	Ok((centers, assignments))
}
fn fit_kmeans(clusters: usize, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
	// Assigning a row to its closest centre is the one-neighbour case of the nearest
	// table, whose lowering is a loop. Emitting the distance per centre per feature
	// instead unrolls the whole comparison into the kernel.
	let (mut table, _) = cluster(&data.samples[..rows * data.features], data.features, clusters, config.kmeans_iterations, None)?;
	let groups = table.len() / data.features.max(1);
	table.extend((0..groups).map(|group| group as f64));
	Ok(Predictor::new(PredictorBuilder::new().nearest(1, false, data.features, table)?))
}
/// Runs the work over disjoint index ranges, one worker per configured CPU worker
/// thread, and returns the results in index order. A worker panic resumes on the caller.
fn parallel_map<R: Send>(count: usize, work: impl Fn(usize) -> R + Sync) -> Result<Vec<R>> {
	let span = count.div_ceil(cpu_worker_threads()? as usize).max(1);
	let mut results = Vec::with_capacity(count);
	std::thread::scope(|scope| {
		let handles = (0..count)
			.step_by(span)
			.map(|start| {
				let work = &work;
				scope.spawn(move || (start..(start + span).min(count)).map(work).collect::<Vec<_>>())
			})
			.collect::<Vec<_>>();
		for handle in handles {
			results.extend(handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)))
		}
	});
	Ok(results)
}
/// Evaluates the immutable teacher once for each prepared sample.
fn predict_rows(teacher: &Predictor, inputs: &[f64], features: usize) -> Result<Vec<f64>> {
	require(features != 0, "teacher prediction has no features")?;
	parallel_map(inputs.len() / features, |row| (teacher.predict)(row, &inputs[row * features..row * features + features]))?.into_iter().collect()
}
fn fit_knn(count: usize, data: &Prepared, rows: usize, _: Config) -> Result<Predictor> {
	require(count != 0 && count < rows, "knn neighbor count is invalid")?;
	let (mut seen, sample) = (HashMap::new(), |row| &data.samples[row * data.features..(row + 1) * data.features]);
	let table = |kept: &[usize]| kept.iter().flat_map(|&r| sample(r).iter().copied()).chain(kept.iter().map(|&r| data.targets[r])).collect::<Vec<_>>();
	// The teacher labels each training row leave one out over every row, while the lowered program searches the compacted rows.
	let teacher = PredictorBuilder::new().nearest(count, true, data.features, table(&(0..rows).collect::<Vec<_>>()))?;
	let kept = (0..rows).filter(|&r| *seen.entry(sample(r).iter().map(|&x| x.to_bits()).collect::<Vec<_>>()).and_modify(|n| *n += 1).or_insert(1) <= count).collect::<Vec<_>>();
	Ok(Predictor { program: PredictorBuilder::new().nearest(count, false, data.features, table(&kept))?, predict: Box::new(move |row, query| teacher.evaluate(row, query)) })
}
impl Estimator {
	fn fit(&self, data: &Prepared, rows: usize, config: Config) -> Result<Predictor> {
		(self.fit)(self.param, data, rows, config)
	}
}
fn native_contraction_shapes(graph: &Graph, rows: usize) -> Result<Vec<Option<NativeContractionShapes>>> {
	graph.nodes
		.iter()
		.map(|node| {
			let dimensions = match node.op {
				Primitive::Contraction => {
					let span = integer_argument(node.argument[0], "native contraction kernel")?.max(1) as usize;
					let window = checked_mul(node.input.channels, span, "native contraction window")?;
					let output_rows = checked_mul(rows, node.output.length, "native contraction output rows")?;
					let input_rows = checked_mul(rows, node.input.length, "native contraction input rows")?;
					let previous_terms = checked_mul(node.output.channels, span, "native contraction previous terms")?;
					Some(((output_rows, node.output.channels, window), (window, node.output.channels, output_rows), (input_rows, node.input.channels, previous_terms), node.parameters))
				}
				Primitive::Scan => {
					let rows = checked_mul(rows, node.input.length, "native scan projection rows")?;
					let parameters = checked_mul(node.input.channels, node.output.channels, "native scan projection parameters")?;
					Some((
						(rows, node.output.channels, node.input.channels),
						(node.input.channels, node.output.channels, rows),
						(rows, node.input.channels, node.output.channels),
						parameters,
					))
				}
				_ => None,
			};
			dimensions
				.map(|(forward, gradient, previous, parameters)| {
					let extent =
						|(m, n, k), role| Ok(Tile { m: narrow(m, &format!("{role} M"))? as u32, n: narrow(n, &format!("{role} N"))? as u32, k: narrow(k, &format!("{role} K"))? as u32 });
					Ok(NativeContractionShapes {
						forward: extent(forward, "native forward contraction")?,
						gradient: extent(gradient, "native gradient contraction")?,
						previous: extent(previous, "native previous contraction")?,
						parameters,
					})
				})
				.transpose()
		})
		.collect()
}
fn native_attention_shared_values(extent: Tile, full: bool) -> Result<u32> {
	let queries = extent.m.checked_mul(extent.k).ok_or_else(|| RecipeError::new("native attention query tile overflows"))?;
	let keys = extent.n.checked_mul(extent.k).ok_or_else(|| RecipeError::new("native attention key tile overflows"))?;
	let pairs = extent.m.checked_mul(extent.n).ok_or_else(|| RecipeError::new("native attention pair tile overflows"))?;
	let forward = queries
		.checked_mul(2)
		.and_then(|values| keys.checked_mul(2).and_then(|keys| values.checked_add(keys)))
		.and_then(|values| pairs.checked_mul(2).and_then(|pairs| values.checked_add(pairs)))
		.and_then(|values| extent.m.checked_mul(3).and_then(|statistics| values.checked_add(statistics)));
	let query_gradient = queries
		.checked_mul(3)
		.and_then(|values| keys.checked_mul(2).and_then(|keys| values.checked_add(keys)))
		.and_then(|values| pairs.checked_mul(2).and_then(|pairs| values.checked_add(pairs)))
		.and_then(|values| values.checked_add(extent.m));
	let key_value_gradient = queries
		.checked_mul(2)
		.and_then(|values| keys.checked_mul(4).and_then(|keys| values.checked_add(keys)))
		.and_then(|values| pairs.checked_mul(2).and_then(|pairs| values.checked_add(pairs)))
		.and_then(|values| values.checked_add(extent.m));
	let matrix_pairs = extent.m.checked_mul(extent.m);
	let matrix =
		queries.checked_mul(4).and_then(|values| matrix_pairs.and_then(|pairs| pairs.checked_mul(2)).and_then(|pairs| values.checked_add(pairs))).and_then(|values| values.checked_add(extent.m));
	forward
		.zip(query_gradient)
		.zip(key_value_gradient)
		.zip(matrix)
		.map(|(((forward, query_gradient), key_value_gradient), matrix)| forward.max(query_gradient).max(key_value_gradient).max(if full { matrix } else { 0 }))
		.ok_or_else(|| RecipeError::new("native attention shared values overflow"))
}
fn native_attention_tile(length: u32, width: u32, shared_values: u32, query_tile: u32) -> Result<Tile> {
	require(length != 0 && width != 0 && shared_values != 0 && query_tile != 0, "native attention tile inputs are empty")?;
	let mut queries = length.min(query_tile);
	loop {
		let query_values = queries.checked_mul(width).ok_or_else(|| RecipeError::new("native attention tile overflows"))?;
		let forward_fixed = query_values
			.checked_mul(2)
			.and_then(|values| queries.checked_mul(3).and_then(|statistics| values.checked_add(statistics)))
			.ok_or_else(|| RecipeError::new("native attention tile overflows"))?;
		let forward_per_key = width.checked_add(queries).and_then(|values| values.checked_mul(2)).ok_or_else(|| RecipeError::new("native attention tile overflows"))?;
		let query_gradient_fixed = query_values.checked_mul(3).and_then(|values| values.checked_add(queries)).ok_or_else(|| RecipeError::new("native attention tile overflows"))?;
		let query_gradient_per_key =
			width.checked_mul(2).and_then(|values| queries.checked_mul(2).and_then(|queries| values.checked_add(queries))).ok_or_else(|| RecipeError::new("native attention tile overflows"))?;
		let key_value_gradient_fixed = query_values.checked_mul(2).and_then(|values| values.checked_add(queries)).ok_or_else(|| RecipeError::new("native attention tile overflows"))?;
		let key_value_gradient_per_key =
			width.checked_mul(4).and_then(|values| queries.checked_mul(2).and_then(|queries| values.checked_add(queries))).ok_or_else(|| RecipeError::new("native attention tile overflows"))?;
		let keys = (shared_values.saturating_sub(forward_fixed) / forward_per_key)
			.min(shared_values.saturating_sub(query_gradient_fixed) / query_gradient_per_key)
			.min(shared_values.saturating_sub(key_value_gradient_fixed) / key_value_gradient_per_key);
		let pairs = queries.checked_mul(queries).ok_or_else(|| RecipeError::new("native attention matrix tile overflows"))?;
		let matrix = query_values
			.checked_mul(4)
			.and_then(|values| pairs.checked_mul(2).and_then(|pairs| values.checked_add(pairs)))
			.and_then(|values| values.checked_add(queries))
			.ok_or_else(|| RecipeError::new("native attention matrix tile overflows"))?;
		if keys != 0 && (queries != length || matrix <= shared_values) {
			return Ok(Tile { m: queries, n: keys.min(length), k: width });
		}
		queries = queries.checked_sub(1).filter(|value| *value != 0).ok_or_else(|| RecipeError::new("native attention tile does not fit the device"))?;
	}
}
fn native_attention_tiles(graph: &Graph, shared_values: u32, query_tile: u32) -> Result<Vec<Option<Tile>>> {
	graph.nodes
		.iter()
		.map(|node| {
			if node.op != Primitive::Attention {
				return Ok(None);
			}
			let heads = integer_argument(node.argument[0], "native attention heads")? as u32;
			let channels = narrow(node.output.channels, "native attention channels")? as u32;
			let length = narrow(node.output.length, "native attention length")? as u32;
			require(channels % heads == 0, "native attention head partition is invalid")?;
			native_attention_tile(length, channels / heads, shared_values, query_tile).map(Some)
		})
		.collect()
}
fn native_tiles(total: usize, width: u32, role: &str) -> Result<usize> {
	let width = width as usize;
	require(total != 0 && width != 0, format!("{role} is empty"))?;
	checked_add(total / width, usize::from(total % width != 0), role)
}
/// Chunk partials that the k lanes of one job exchange, in model-sized elements
/// per chunk. A job whose output fills the workgroup has a single k lane and
/// folds its own chunks locally, so the exchange only ever happens with at most
/// half the workgroup holding output positions; the region is sized for that
/// worst case. `ratio` converts state-typed partial values into model-sized
/// elements, because the allocation is counted in model elements while the
/// partials keep the arithmetic width.
fn native_contraction_partial_per_chunk(m: u32, n: u32, register_m: u32, register_n: u32, block: u32, ratio: u32) -> Result<u32> {
	let output_lanes = (m / register_m).max(1).checked_mul((n / register_n).max(1)).ok_or_else(|| RecipeError::new("native contraction lane count overflows"))?;
	let exchange_lanes = output_lanes.min((block / 2).max(1));
	let registers = register_m.checked_mul(register_n).ok_or_else(|| RecipeError::new("native contraction register tile overflows"))?;
	let sums = exchange_lanes.checked_mul(registers).ok_or_else(|| RecipeError::new("native contraction partial region overflows"))?;
	let biases = (n / register_n).max(1).checked_mul(register_n).ok_or_else(|| RecipeError::new("native contraction partial region overflows"))?;
	sums.checked_add(biases).and_then(|values| values.checked_mul(ratio)).ok_or_else(|| RecipeError::new("native contraction partial region overflows"))
}
/// The tile's local memory serves two phases in turn: the staged operands, and
/// then the chunk partials the k lanes exchange after a barrier. Both live in
/// the same allocation, so K is bounded by whichever phase is larger, and a tile
/// too wide to stage a whole chunk narrows its M lanes until one fits.
fn native_contraction_tile(limits: Tile, register_m: u32, register_n: u32, block: u32, shared_values: u32, fragment: u32, ratio: u32, matrix: bool) -> Result<Tile> {
	require(register_m != 0 && register_n != 0 && block != 0 && fragment != 0 && ratio != 0, "native contraction tile inputs are empty")?;
	if matrix {
		let waves = block / 32;
		require(waves != 0, "native matrix contraction has no wave")?;
		let m = waves.checked_mul(16).ok_or_else(|| RecipeError::new("native matrix contraction M tile overflows"))?;
		let n = (block / 2).max(32);
		let width = m.checked_add(n).ok_or_else(|| RecipeError::new("native matrix contraction tile width overflows"))?;
		let room = shared_values / width;
		let capacity = room - room % fragment;
		let required = limits.k.div_ceil(fragment).checked_mul(fragment).ok_or_else(|| RecipeError::new("native matrix contraction K tile overflows"))?;
		let k = required.min(capacity);
		require(k != 0, "native matrix contraction tile does not fit the device")?;
		return Ok(Tile { m, n, k });
	}
	let mut lane_n = limits.n.div_ceil(register_n).min(block.isqrt().max(1));
	let widest_m = limits.m.div_ceil(register_m);
	let mut lane_m = widest_m.min(block / lane_n);
	loop {
		let m = lane_m.checked_mul(register_m).ok_or_else(|| RecipeError::new("native contraction M tile overflows"))?;
		let n = lane_n.checked_mul(register_n).ok_or_else(|| RecipeError::new("native contraction N tile overflows"))?;
		let width = m.checked_add(n).ok_or_else(|| RecipeError::new("native contraction tile width overflows"))?;
		let staging_k = shared_values / width;
		let partial_per_chunk = native_contraction_partial_per_chunk(m, n, register_m, register_n, block, ratio)?;
		let partial_k = (shared_values / partial_per_chunk).checked_mul(fragment).ok_or_else(|| RecipeError::new("native contraction partial K overflows"))?;
		let room = staging_k.min(partial_k);
		// Reduction chunks are RECIPE_FRAGMENT_K elements of K, aligned to the
		// start of the walk. A multi-tile walk must therefore stage whole chunks,
		// so the tile is rounded down to a chunk multiple; a walk that fits in one
		// staged tile has no interior tile boundary and may keep its exact length.
		if room >= limits.k {
			return Ok(Tile { m, n, k: limits.k });
		}
		if room >= fragment {
			return Ok(Tile { m, n, k: room - room % fragment });
		}
		// The staged width is M plus N, so narrowing M alone never fits a chunk once N
		// fills the budget by itself. The N lanes narrow once the M lanes are spent.
		match lane_m.checked_sub(1).filter(|lanes| *lanes != 0) {
			Some(lanes) => lane_m = lanes,
			None => {
				lane_n = lane_n.checked_sub(1).filter(|lanes| *lanes != 0).ok_or_else(|| RecipeError::new("native contraction tile does not fit the device"))?;
				lane_m = widest_m.min(block / lane_n)
			}
		}
	}
}
fn native_contraction_shared_values(extent: Tile, register_m: u32, register_n: u32, block: u32, fragment: u32, ratio: u32, matrix: bool) -> Result<u32> {
	let staging = extent.m.checked_add(extent.n).and_then(|width| width.checked_mul(extent.k)).ok_or_else(|| RecipeError::new("native contraction shared values overflow"))?;
	if matrix {
		return Ok(staging);
	}
	let partials = extent
		.k
		.div_ceil(fragment)
		.checked_mul(native_contraction_partial_per_chunk(extent.m, extent.n, register_m, register_n, block, ratio)?)
		.ok_or_else(|| RecipeError::new("native contraction partial region overflows"))?;
	Ok(staging.max(partials))
}
/// Values per split-K scratch row. Rows are written by separate workgroups, so
/// each row starts on a machine-word boundary for every supported element width.
const NATIVE_SCRATCH_ROW_VALUES: usize = 4;
/// A reverse K extent is cut into one contiguous partition per span of elements,
/// capped at the partition limit. The count and the boundaries are a function of
/// the extent and these two constants alone, so the summation order does not
/// follow the tile, the workgroup width, or the number of compute units, while a
/// long K still spreads across enough workgroups to cover the device when the
/// output produces few jobs.
const NATIVE_SPLIT_SPAN: usize = parse_natural(env!("RECIPE_CONTRACTION_SPLIT_SPAN"), "contraction split span must be a positive integer");
const NATIVE_MATRIX_SPLIT_SPAN: usize = parse_natural(env!("RECIPE_CONTRACTION_MATRIX_SPLIT_SPAN"), "contraction matrix split span must be a positive integer");
const NATIVE_K_PARTITIONS: usize = parse_natural(env!("RECIPE_CONTRACTION_K_PARTITIONS"), "contraction K partitions must be a positive integer");
const fn parse_natural(text: &str, role: &'static str) -> usize {
	let text = text.as_bytes();
	let (mut value, mut index) = (0_usize, 0);
	while index < text.len() {
		assert!(text[index].is_ascii_digit(), "{}", role);
		value = value * 10 + (text[index] - b'0') as usize;
		index += 1;
	}
	assert!(value != 0, "{}", role);
	value
}

fn native_gradient_values(parameters: usize, contractions: &[Option<NativeContractionTiles>]) -> Result<usize> {
	let mut scratch = 0_usize;
	for contraction in contractions {
		let Some(contraction) = contraction else { continue };
		// The allocation covers either scalar or matrix partitioning. A backend
		// using the larger span leaves the extra rows untouched.
		let extent = contraction.gradient_shape.k as usize;
		let splits = extent.div_ceil(NATIVE_SPLIT_SPAN.min(NATIVE_MATRIX_SPLIT_SPAN)).min(NATIVE_K_PARTITIONS).max(1);
		let jobs = checked_mul(
			native_tiles(contraction.gradient_shape.m as usize, contraction.gradient.m, "native split-K M tiles")?,
			native_tiles(contraction.gradient_shape.n as usize, contraction.gradient.n, "native split-K N tiles")?,
			"native split-K jobs",
		)?;
		narrow(checked_mul(jobs, splits, "native split-K tasks")?, "native split-K tasks")?;
		if splits > 1 {
			scratch = scratch.max(checked_mul(splits, contraction.parameters.next_multiple_of(NATIVE_SCRATCH_ROW_VALUES), "native split-K scratch")?);
		}
	}
	// Row zero has to start on the same boundary as every later row, so the base
	// is the aligned parameter count rather than the raw one. The optimiser still
	// walks only the parameters themselves.
	let base = parameters.next_multiple_of(NATIVE_SCRATCH_ROW_VALUES);
	let values = checked_add(base, scratch, "native gradient and split-K scratch")?;
	narrow(values, "native gradient and split-K scratch")?;
	Ok(values)
}
#[cfg(any(amd, nvidia))]
#[derive(Clone, Copy)]
struct Resources {
	pub shared: u32,
	pub max_block: u32,
}
#[derive(Clone, Copy)]
struct Geometry {
	pub groups: u32,
	pub block: u32,
}
impl Geometry {
	pub fn threads(self) -> Result<u32> {
		self.groups.checked_mul(self.block).filter(|value| *value != 0).ok_or_else(|| RecipeError::new("GPU launch size overflows"))
	}
}
#[cfg(any(amd, nvidia))]
fn geometry(cus: u32, wave: u32, workgroup: u32, lds: u32, groups_per_cu: u32, resources: Resources) -> Result<Geometry> {
	require(wave != 0 && wave <= workgroup && wave <= resources.max_block, "GPU wave exceeds kernel workgroup")?;
	let waves = groups_per_cu.min(workgroup / wave).min(resources.max_block / wave);
	require(waves != 0, "GPU has no resident wave")?;
	let block = waves.checked_mul(wave).ok_or_else(|| RecipeError::new("GPU workgroup size overflows"))?;
	require(resources.shared <= lds, "GPU tile exceeds local memory")?;
	Ok(Geometry { groups: cus, block })
}
#[cfg(amd)]
fn amd(cus: u32, wave: u32, workgroup: u32, lds: u32, waves: u32, resources: Resources) -> Result<Geometry> {
	let block = wave.checked_mul(waves).ok_or_else(|| RecipeError::new("AMD workgroup size overflows"))?;
	require(wave != 0 && block <= workgroup && block <= resources.max_block && waves != 0, "AMD workgroup geometry is invalid")?;
	require(resources.shared <= lds, "AMD workgroup exceeds local memory")?;
	Ok(Geometry { groups: cus, block })
}
#[cfg(nvidia)]
fn nvidia(cus: u32, wave: u32, workgroup: u32, block_lds: u32, sm_lds: u32, waves_per_cu: u32, resources: Resources) -> Result<Geometry> {
	require(resources.shared <= block_lds, "Nvidia tile exceeds workgroup shared memory")?;
	geometry(cus, wave, workgroup, sm_lds, waves_per_cu, resources)
}
pub trait IntoDataSources {
	const AUTO: bool = false;
	fn into_data_sources(self) -> Vec<String>;
}
impl IntoDataSources for Auto {
	const AUTO: bool = true;
	fn into_data_sources(self) -> Vec<String> {
		Vec::new()
	}
}
impl IntoDataSources for &str {
	fn into_data_sources(self) -> Vec<String> {
		vec![self.to_owned()]
	}
}
impl IntoDataSources for String {
	fn into_data_sources(self) -> Vec<String> {
		vec![self]
	}
}
impl<T: Into<String>, const N: usize> IntoDataSources for [T; N] {
	fn into_data_sources(self) -> Vec<String> {
		self.into_iter().map(Into::into).collect()
	}
}
impl<T: Into<String>> IntoDataSources for Vec<T> {
	fn into_data_sources(self) -> Vec<String> {
		self.into_iter().map(Into::into).collect()
	}
}
impl<T: Clone + Into<String>> IntoDataSources for &[T] {
	fn into_data_sources(self) -> Vec<String> {
		self.iter().cloned().map(Into::into).collect()
	}
}
impl Data {
	pub fn target(mut self, target: impl IntoDataSources) -> Self {
		self.target = target.into_data_sources();
		self
	}
	pub fn include(mut self, names: impl IntoDataSources) -> Self {
		assert!(!matches!(&self.features, FeatureSelection::Exclude(_)), "include and exclude are mutually exclusive");
		self.features = FeatureSelection::Include(names.into_data_sources());
		self
	}
	pub fn exclude(mut self, names: impl IntoDataSources) -> Self {
		assert!(!matches!(&self.features, FeatureSelection::Include(_)), "include and exclude are mutually exclusive");
		self.features = FeatureSelection::Exclude(names.into_data_sources());
		self
	}
	pub fn test(mut self, sources: impl IntoDataSources) -> Self {
		self.tests = sources.into_data_sources();
		self
	}
	pub fn set(mut self, source: impl Into<String>) -> Self {
		self.sources.push(source.into());
		self
	}
	pub const fn broadcast(mut self) -> Self {
		self.broadcast = true;
		self
	}
	pub const fn norm(mut self, _: ZScore) -> Self {
		self.normalize = true;
		self
	}
	pub const fn split(mut self, fraction: f64) -> Self {
		self.split = fraction;
		self
	}
}
type DataSchema = Vec<(String, String)>;
struct Prepared {
	samples: Vec<f64>,
	/// One row's targets are contiguous, so the buffer is `rows * target_width` long.
	targets: Vec<f64>,
	target_width: usize,
	rows: usize,
	source_rows: usize,
	features: usize,
	schema: DataSchema,
	sequence: Option<(Shape, Shape)>,
	target_categorical: bool,
	norm_mean: Vec<f64>,
	norm_scale: Vec<f64>,
	identities: Vec<u64>,
	fitted: Vec<PredictorProgram>,
	/// The weights each parameterized node binds to, in lowering order, when the
	/// graph compiles over mapped tensors instead of training.
	bound: Option<Vec<BoundNode>>,
}
struct Table {
	name: String,
	headers: Vec<String>,
	rows: Vec<Vec<String>>,
	/// Row-major image values are channel-major when each image row is one channel.
	attention: Option<Shape>,
}
enum FeatureType {
	Numeric,
	Categorical(Vec<String>),
	Text(usize),
}
fn prepare(data: &Data) -> Result<&Prepared> {
	match data.prepared.get_or_init(|| prepare_data(data)) {
		Ok(prepared) => Ok(prepared),
		Err(error) => Err(error.clone()),
	}
}
fn column_match(name: &str, table: &Table, header: &str, column: usize) -> bool {
	name == header
		|| name == format!("{}.{}", table.name, header)
		|| name == format!("col{}", column + 1)
		|| name == format!("{}.col{}", table.name, column + 1)
		|| header.strip_suffix(name).is_some_and(|prefix| prefix.ends_with('.'))
		|| header.rsplit_once('.').is_some_and(|(base, row)| row.parse::<usize>().is_ok() && (base == name || base.strip_suffix(name).is_some_and(|prefix| prefix.ends_with('.'))))
}
impl FeatureSelection {
	fn selects(&self, table: &Table, header: &str, column: usize) -> bool {
		match self {
			Self::All => true,
			Self::Include(names) => names.iter().any(|name| column_match(name, table, header, column)),
			Self::Exclude(names) => !names.iter().any(|name| column_match(name, table, header, column)),
		}
	}
}
fn load_tables(data: &Data, sources: &[String]) -> Result<(Vec<Table>, Vec<PathBuf>)> {
	let mut paths = Vec::new();
	for source in sources {
		collect_files(&resolve_path(source)?, &mut paths)?;
	}
	for path in &mut paths {
		*path = fs::canonicalize(&*path).map_err(|error| RecipeError::new(format!("cannot resolve {}: {error}", path.display())))?
	}
	paths.sort();
	paths.dedup();
	// A ZIP source contributes its entries, not itself: the container is not a
	// table or a sample, and its entries take virtual paths anchored at the
	// archive's own path, so the directory-layout rules that already interpret
	// a real class-subfolder tree interpret an archived one identically.
	let mut files = Vec::new();
	for path in &paths {
		let bytes = fs::read(path).map_err(|error| RecipeError::new(format!("cannot read {}: {error}", path.display())))?;
		if path.extension().and_then(|value| value.to_str()).is_some_and(is_archive) {
			for (entry, contents) in zip_entries(&bytes).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))? {
				files.push((path.join(entry), contents));
			}
		} else {
			files.push((path.clone(), bytes));
		}
	}
	let mut grouped = Vec::new();
	for (path, bytes) in &files {
		if !path.extension().and_then(|value| value.to_str()).is_some_and(is_table) {
			continue;
		}
		let directory = path.parent().unwrap_or_else(|| Path::new("")).to_owned();
		for table in decode_tables(path, bytes)? {
			grouped.push((directory.clone(), table));
		}
	}
	if let Some(table) = directory_samples(data, sources, &files, &grouped)? {
		return Ok((vec![table], paths));
	}
	let mut tables = merge_captures(grouped, &data.target)?;
	tables = merge_partitions(tables, &data.target, &data.features)?;
	require(!tables.is_empty(), "data source contains no supported table")?;
	if tables.len() > 1 {
		let rows = tables.iter().map(|table| table.rows.len()).max().unwrap_or(0);
		let aligned = rows != 0 && tables.iter().all(|table| table.rows.len() == rows);
		require(aligned || data.broadcast, "multiple tables require explicit .broadcast() alignment")?;
		if data.broadcast {
			for table in &mut tables {
				let count = table.rows.len();
				require(count != 0 && rows % count == 0, format!("table {:?} expected a nonzero row count dividing {rows}, received {count}", table.name))?;
				if count != rows {
					table.rows = table.rows.iter().cloned().cycle().take(rows).collect()
				}
			}
		}
	}
	Ok((tables, paths))
}
/// One interpretation of directory layout for sample trees whose target is not a table
/// column: flat sidecar-labeled samples, class-labeled subdirectories, and paired
/// subdirectories. Each file is read once; text samples contribute their content and image
/// samples their decoded pixels. Anything else falls through to the table flow.
fn directory_samples(data: &Data, sources: &[String], files: &[(PathBuf, Vec<u8>)], parsed: &[(PathBuf, Table)]) -> Result<Option<Table>> {
	let [source] = sources else { return Ok(None) };
	let [target] = data.target.as_slice() else { return Ok(None) };
	let sample = |path: &Path| path.extension().and_then(|value| value.to_str()).is_some_and(|extension| is_table(extension) || is_image(extension) || is_document(extension));
	let samples = files.iter().filter(|(path, _)| sample(path)).collect::<Vec<_>>();
	if samples.is_empty() {
		return Ok(None);
	}
	let root = fs::canonicalize(resolve_path(source)?).map_err(|error| RecipeError::new(format!("cannot resolve {source}: {error}")))?;
	let name = root.file_name().and_then(|value| value.to_str()).unwrap_or("data").to_owned();
	let stem = |path: &Path| path.file_stem().and_then(|value| value.to_str()).unwrap_or("").to_owned();
	// Flat sidecar samples: every file directly under the root, labeled by a .label sibling.
	if samples.iter().all(|(path, _)| path.parent() == Some(root.as_path())) {
		let sidecar = |path: &Path| files.iter().find(|(candidate, _)| *candidate == path.with_extension("label"));
		if samples.iter().all(|(path, _)| sidecar(path).is_some()) {
			let mut builder = SampleTableBuilder::new(target.clone());
			for (path, bytes) in &samples {
				let (_, label) = sidecar(path).unwrap();
				builder.push(path, bytes, sample_text(path, label)?)?;
			}
			return builder.finish(name).map(Some);
		}
		// Name-labeled samples: `<target>__<name>` carries the label in the file name
		// rather than in a sibling or a directory. Every file under the root has to be
		// one labeled sample, so a tree whose samples are only partly recognized never
		// trains on the recognized part alone.
		const SEPARATOR: &str = "__";
		let labels = samples.iter().map(|(path, _)| stem(path).split_once(SEPARATOR).map(|(label, _)| label.to_owned())).collect::<Option<Vec<_>>>();
		if let Some(labels) = labels.filter(|labels| samples.len() == files.len() && labels.iter().collect::<BTreeSet<_>>().len() > 1) {
			let mut builder = SampleTableBuilder::new(target.clone());
			for ((path, bytes), label) in samples.iter().zip(labels) {
				builder.push(path, bytes, label)?;
			}
			return builder.finish(name).map(Some);
		}
		return Ok(None);
	}
	// One level of subdirectories under the root.
	if !samples.iter().all(|(path, _)| path.parent().and_then(Path::parent) == Some(root.as_path())) {
		return Ok(None);
	}
	let mut directories = BTreeMap::<String, Vec<&(PathBuf, Vec<u8>)>>::new();
	for entry in &samples {
		let directory =
			entry.0.parent().and_then(Path::file_name).and_then(|value| value.to_str()).ok_or_else(|| RecipeError::new(format!("sample directory of {} is unreadable", entry.0.display())))?;
		directories.entry(directory.to_owned()).or_default().push(entry);
	}
	if directories.len() < 2 {
		return Ok(None);
	}
	// Paired subdirectories: identical sample stems in every directory, one directory named
	// for the requested target. Each stem is one sample and each directory one column group.
	let singular = |directory: &str| directory.strip_suffix('s').filter(|value| !value.is_empty()).unwrap_or(directory).to_owned();
	let stems = directories.values().map(|entries| entries.iter().map(|(path, _)| stem(path)).collect::<BTreeSet<_>>()).collect::<Vec<_>>();
	let aligned = stems.windows(2).all(|pair| pair[0] == pair[1]);
	let paired_target = directories.keys().any(|directory| singular(directory) == *target);
	let sample_target = stems[0].contains(target);
	require(!(aligned && paired_target && sample_target), format!("target {target:?} names both a paired directory and a per-sample file"))?;
	// Per-sample layouts use directories as rows; paired layouts use directories as columns.
	if aligned && (sample_target || paired_target) {
		let mut columns = Vec::new();
		if sample_target {
			for column in &stems[0] {
				columns.push((column.clone(), directories.values().map(|entries| entries.iter().copied().find(|(path, _)| stem(path) == *column).unwrap()).collect()));
			}
		} else {
			for (directory, entries) in &directories {
				let mut entries = entries.clone();
				entries.sort_by_key(|(path, _)| stem(path));
				columns.push((singular(directory), entries));
			}
		}
		let mut headers = Vec::new();
		let mut attention = Some(Shape { channels: 0, length: 0 });
		let mut rows = vec![Vec::new(); columns[0].1.len()];
		for (column, entries) in &columns {
			let mut kind = None;
			for (row, (path, bytes)) in entries.iter().enumerate() {
				let (shape, values) = sample_values(path, bytes)?;
				require(!sample_target || column != target || shape.is_none(), format!("per-sample target files {column:?} hold images, not values"))?;
				let current = (shape, values.len());
				require(*kind.get_or_insert(current) == current, format!("sample {} expected {:?}, received {current:?}", path.display(), kind.unwrap()))?;
				rows[row].extend(values);
			}
			let (shape, width) = kind.unwrap_or((None, 0));
			if column != target
				&& let Some(previous) = attention
			{
				attention = shape
					.filter(|shape| previous.channels == 0 || previous.length == shape.channels)
					.map(|shape| Shape { channels: previous.channels + shape.length, length: shape.channels })
			}
			headers.extend((1..=width).map(|index| if width == 1 { column.clone() } else { format!("{column}.{index}") }));
		}
		return Ok(Some(Table { name, headers, rows, attention: attention.filter(|shape| shape.channels != 0) }));
	}
	if aligned {
		return Ok(None);
	}
	// Class subdirectories: differing sample stems, the directory name is the target value.
	if parsed.iter().any(|(_, table)| target_column(table, target).is_some()) {
		return Ok(None);
	}
	let mut builder = SampleTableBuilder::new(target.clone());
	for (directory, entries) in &directories {
		for (path, bytes) in entries {
			builder.push(path, bytes, directory.clone())?;
		}
	}
	builder.finish(name).map(Some)
}
/// Rows of one sample table: each sample contributes its content columns plus the target.
struct SampleTableBuilder {
	target: String,
	shape: Option<Shape>,
	headers: Vec<String>,
	rows: Vec<Vec<String>>,
}
impl SampleTableBuilder {
	fn new(target: String) -> Self {
		Self { target, shape: None, headers: Vec::new(), rows: Vec::new() }
	}
	fn push(&mut self, path: &Path, bytes: &[u8], target: String) -> Result<()> {
		let (shape, values) = sample_values(path, bytes)?;
		if self.headers.is_empty() {
			self.shape = shape;
			let name = if shape.is_some() { "pixel" } else { "content" };
			self.headers = (1..=values.len()).map(|index| if values.len() == 1 { name.to_owned() } else { format!("{name}.{index}") }).collect();
			self.headers.push(self.target.clone());
		}
		let (expected, received) = ((self.shape, self.headers.len() - 1), (shape, values.len()));
		require(expected == received, format!("sample {} expected {expected:?}, received {received:?}", path.display()))?;
		let mut row = values;
		row.push(target);
		self.rows.push(row);
		Ok(())
	}
	fn finish(self, name: String) -> Result<Table> {
		Ok(Table { name, headers: self.headers, rows: self.rows, attention: self.shape.map(|shape| Shape { channels: shape.length, length: shape.channels }) })
	}
}
fn sample_text(path: &Path, bytes: &[u8]) -> Result<String> {
	Ok(str::from_utf8(bytes).map_err(|error| RecipeError::new(format!("sample {} is not UTF-8: {error}", path.display())))?.trim().to_owned())
}
fn sample_values(path: &Path, bytes: &[u8]) -> Result<(Option<Shape>, Vec<String>)> {
	if !path.extension().and_then(|value| value.to_str()).is_some_and(is_image) {
		return Ok((None, vec![sample_text(path, bytes)?]));
	}
	let jpeg = path.extension().and_then(|value| value.to_str()).is_some_and(|value| matches!(value.to_ascii_lowercase().as_str(), "jpg" | "jpeg"));
	let decoded = if jpeg { jpeg_pixels(bytes) } else { png_pixels(bytes) };
	let (width, height, channels, pixels) = decoded.map_err(|error| RecipeError::new(format!("image {}: {error}", path.display())))?;
	Ok((Some(Shape { channels: checked_mul(width, channels, "image row width")?, length: height }), pixels.iter().map(|value| value.to_string()).collect()))
}
fn is_image(extension: &str) -> bool {
	matches!(extension.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg")
}
/// Document formats that carry sample text but never decode as tables, so they only
/// count as samples inside a recognized directory layout.
fn is_document(extension: &str) -> bool {
	matches!(extension.to_ascii_lowercase().as_str(), "md" | "html" | "htm")
}
/// Decoded baseline JFIF pixels: 8-bit precision, Huffman entropy coding, and the
/// libjpeg fixed-point inverse DCT and color conversion so pixels match its output.
fn jpeg_pixels(bytes: &[u8]) -> Result<(usize, usize, usize, Vec<u8>)> {
	const ZIGZAG: [usize; 64] = [
		0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
		59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
	];
	let truncated = || RecipeError::new("JPEG stream is truncated");
	require(bytes.get(..2) == Some(&[0xff, 0xd8]), "JPEG signature is absent")?;
	let mut quantization = [[0_u16; 64]; 4];
	let mut huffman: [[Option<(Vec<u8>, Vec<u8>)>; 4]; 2] = Default::default();
	let (mut frame, mut restart_interval) = (None, 0_usize);
	let mut offset = 2;
	let scan;
	loop {
		require(bytes.get(offset) == Some(&0xff), "JPEG marker is invalid")?;
		let marker = *bytes.get(offset + 1).ok_or_else(truncated)?;
		let length = usize::from(u16::from_be_bytes(bytes.get(offset + 2..offset + 4).ok_or_else(truncated)?.try_into().unwrap()));
		let body = bytes.get(offset + 4..offset + 2 + length).ok_or_else(truncated)?;
		match marker {
			0xdb => {
				let mut position = 0;
				while position < body.len() {
					let (precision, table) = (body[position] >> 4, usize::from(body[position] & 15));
					require(precision == 0 && table < 4, "JPEG quantization table is unsupported")?;
					for index in 0..64 {
						quantization[table][ZIGZAG[index]] = u16::from(*body.get(position + 1 + index).ok_or_else(truncated)?);
					}
					position += 65;
				}
			}
			0xc4 => {
				let mut position = 0;
				while position < body.len() {
					let (class, table) = (usize::from(body[position] >> 4), usize::from(body[position] & 15));
					require(class < 2 && table < 4, "JPEG Huffman table is unsupported")?;
					let counts = body.get(position + 1..position + 17).ok_or_else(truncated)?.to_vec();
					let total = counts.iter().map(|&count| usize::from(count)).sum::<usize>();
					let symbols = body.get(position + 17..position + 17 + total).ok_or_else(truncated)?.to_vec();
					huffman[class][table] = Some((counts, symbols));
					position += 17 + total;
				}
			}
			0xc0 => {
				let (height, width, components) =
					(usize::from(u16::from_be_bytes(body[1..3].try_into().unwrap())), usize::from(u16::from_be_bytes(body[3..5].try_into().unwrap())), usize::from(body[5]));
				require(body[0] == 8, "JPEG precision is unsupported")?;
				require(matches!(components, 1 | 3), format!("JPEG component count {components} is unsupported"))?;
				let mut layout = Vec::new();
				for component in 0..components {
					let (sampling, table) = (body[7 + 3 * component], usize::from(body[8 + 3 * component]));
					require(sampling == 0x11, "JPEG chroma subsampling is unsupported")?;
					layout.push(table);
				}
				frame = Some((width, height, layout));
			}
			0xc1..=0xcf if marker != 0xc4 && marker != 0xc8 && marker != 0xcc => return Err(RecipeError::new(format!("JPEG frame type {marker:#x} is unsupported"))),
			0xdd => restart_interval = usize::from(u16::from_be_bytes(body[..2].try_into().unwrap())),
			0xda => {
				let components = usize::from(body[0]);
				let mut tables = Vec::new();
				for component in 0..components {
					tables.push((usize::from(body[2 + 2 * component] >> 4), usize::from(body[2 + 2 * component] & 15)));
				}
				scan = (tables, offset + 2 + length);
				break;
			}
			_ => {}
		}
		offset += 2 + length;
	}
	let (width, height, layout) = frame.ok_or_else(|| RecipeError::new("JPEG frame header is absent"))?;
	let (scan_tables, mut position) = scan;
	require(scan_tables.len() == layout.len(), "JPEG scan does not cover the frame components")?;
	// Entropy-coded segment with byte stuffing and restart markers.
	struct Entropy<'a> {
		bytes: &'a [u8],
		position: usize,
		bits: u32,
		count: u32,
	}
	impl Entropy<'_> {
		fn bit(&mut self) -> Result<u32> {
			if self.count == 0 {
				let byte = *self.bytes.get(self.position).ok_or_else(|| RecipeError::new("JPEG entropy data is truncated"))?;
				self.position += 1;
				if byte == 0xff {
					let stuffed = *self.bytes.get(self.position).ok_or_else(|| RecipeError::new("JPEG entropy data is truncated"))?;
					require(stuffed == 0, "JPEG marker interrupts entropy data")?;
					self.position += 1;
				}
				self.bits = u32::from(byte);
				self.count = 8;
			}
			self.count -= 1;
			Ok(self.bits >> self.count & 1)
		}
		fn receive(&mut self, length: u32) -> Result<i32> {
			let mut value = 0_i32;
			for _ in 0..length {
				value = value << 1 | self.bit()? as i32;
			}
			Ok(value)
		}
		fn decode(&mut self, table: &(Vec<u8>, Vec<u8>)) -> Result<u8> {
			let (mut code, mut first, mut index) = (0_u32, 0_u32, 0_u32);
			for length in 0..16 {
				code = code << 1 | self.bit()?;
				let count = u32::from(table.0[length]);
				if code < first + count {
					return Ok(table.1[(index + code - first) as usize]);
				}
				index += count;
				first = (first + count) << 1;
			}
			Err(RecipeError::new("JPEG Huffman code is invalid"))
		}
	}
	fn extend(value: i32, length: u32) -> i32 {
		if length != 0 && value < 1 << (length - 1) { value - (1 << length) + 1 } else { value }
	}
	// libjpeg jpeg_idct_islow: 13-bit fixed point, two passes, descale rounding.
	fn idct(block: &[i32; 64], quantum: &[u16; 64]) -> [u8; 64] {
		let mut workspace = [0_i32; 64];
		for column in 0..8 {
			let at = |row: usize| block[row * 8 + column] * i32::from(quantum[row * 8 + column]);
			if (1..8).all(|row| at(row) == 0) {
				let value = at(0) << 2;
				for row in 0..8 {
					workspace[row * 8 + column] = value;
				}
				continue;
			}
			let (z2, z3) = (at(2), at(6));
			let z1 = (z2 + z3) * 4433;
			let tmp2 = z1 + z3 * -15137;
			let tmp3 = z1 + z2 * 6270;
			let (tmp0, tmp1) = ((at(0) + at(4)) << 13, (at(0) - at(4)) << 13);
			let (t10, t13, t11, t12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
			let (o0, o1, o2, o3) = (at(7), at(5), at(3), at(1));
			let (z1, z2, z3, z4) = (o0 + o3, o1 + o2, o0 + o2, o1 + o3);
			let z5 = (z3 + z4) * 9633;
			let (mut t0, mut t1, mut t2, mut t3) = (o0 * 2446, o1 * 16819, o2 * 25172, o3 * 12299);
			let (z1, z2) = (z1 * -7373, z2 * -20995);
			let z3 = z3 * -16069 + z5;
			let z4 = z4 * -3196 + z5;
			t0 += z1 + z3;
			t1 += z2 + z4;
			t2 += z2 + z3;
			t3 += z1 + z4;
			workspace[column] = t10 + t3 + 1024 >> 11;
			workspace[56 + column] = t10 - t3 + 1024 >> 11;
			workspace[8 + column] = t11 + t2 + 1024 >> 11;
			workspace[48 + column] = t11 - t2 + 1024 >> 11;
			workspace[16 + column] = t12 + t1 + 1024 >> 11;
			workspace[40 + column] = t12 - t1 + 1024 >> 11;
			workspace[24 + column] = t13 + t0 + 1024 >> 11;
			workspace[32 + column] = t13 - t0 + 1024 >> 11;
		}
		let mut output = [0_u8; 64];
		let clamp = |value: i32| value.clamp(0, 255) as u8;
		for row in 0..8 {
			let at = |column: usize| workspace[row * 8 + column];
			let (z2, z3) = (at(2), at(6));
			let z1 = (z2 + z3) * 4433;
			let tmp2 = z1 + z3 * -15137;
			let tmp3 = z1 + z2 * 6270;
			let (tmp0, tmp1) = ((at(0) + at(4)) << 13, (at(0) - at(4)) << 13);
			let (t10, t13, t11, t12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
			let (o0, o1, o2, o3) = (at(7), at(5), at(3), at(1));
			let (z1, z2, z3, z4) = (o0 + o3, o1 + o2, o0 + o2, o1 + o3);
			let z5 = (z3 + z4) * 9633;
			let (mut t0, mut t1, mut t2, mut t3) = (o0 * 2446, o1 * 16819, o2 * 25172, o3 * 12299);
			let (z1, z2) = (z1 * -7373, z2 * -20995);
			let z3 = z3 * -16069 + z5;
			let z4 = z4 * -3196 + z5;
			t0 += z1 + z3;
			t1 += z2 + z4;
			t2 += z2 + z3;
			t3 += z1 + z4;
			output[row * 8] = clamp((t10 + t3 + (1 << 17) >> 18) + 128);
			output[row * 8 + 7] = clamp((t10 - t3 + (1 << 17) >> 18) + 128);
			output[row * 8 + 1] = clamp((t11 + t2 + (1 << 17) >> 18) + 128);
			output[row * 8 + 6] = clamp((t11 - t2 + (1 << 17) >> 18) + 128);
			output[row * 8 + 2] = clamp((t12 + t1 + (1 << 17) >> 18) + 128);
			output[row * 8 + 5] = clamp((t12 - t1 + (1 << 17) >> 18) + 128);
			output[row * 8 + 3] = clamp((t13 + t0 + (1 << 17) >> 18) + 128);
			output[row * 8 + 4] = clamp((t13 - t0 + (1 << 17) >> 18) + 128);
		}
		output
	}
	let components = layout.len();
	let (blocks_x, blocks_y) = (width.div_ceil(8), height.div_ceil(8));
	let mut planes = vec![vec![0_u8; blocks_x * blocks_y * 64]; components];
	let mut entropy = Entropy { bytes, position, bits: 0, count: 0 };
	let mut predictions = vec![0_i32; components];
	let mut units = 0_usize;
	for block_y in 0..blocks_y {
		for block_x in 0..blocks_x {
			if restart_interval != 0 && units == restart_interval {
				entropy.count = 0;
				require(bytes.get(entropy.position) == Some(&0xff) && bytes.get(entropy.position + 1).is_some_and(|marker| (0xd0..=0xd7).contains(marker)), "JPEG restart marker is absent")?;
				entropy.position += 2;
				predictions.fill(0);
				units = 0;
			}
			for component in 0..components {
				let (dc_table, ac_table) = scan_tables[component];
				let dc = huffman[0][dc_table].as_ref().ok_or_else(|| RecipeError::new("JPEG DC table is absent"))?;
				let ac = huffman[1][ac_table].as_ref().ok_or_else(|| RecipeError::new("JPEG AC table is absent"))?;
				let mut block = [0_i32; 64];
				let length = u32::from(entropy.decode(dc)?);
				predictions[component] += extend(entropy.receive(length)?, length);
				block[0] = predictions[component];
				let mut index = 1;
				while index < 64 {
					let symbol = entropy.decode(ac)?;
					let (run, length) = (usize::from(symbol >> 4), u32::from(symbol & 15));
					if length == 0 {
						if run == 15 {
							index += 16;
							continue;
						}
						break;
					}
					index += run;
					require(index < 64, "JPEG coefficient index overflows")?;
					block[ZIGZAG[index]] = extend(entropy.receive(length)?, length);
					index += 1;
				}
				let decoded = idct(&block, &quantization[layout[component]]);
				let plane = &mut planes[component];
				for row in 0..8 {
					for column in 0..8 {
						plane[(block_y * 8 + row) * blocks_x * 8 + block_x * 8 + column] = decoded[row * 8 + column];
					}
				}
			}
			units += 1;
		}
	}
	position = entropy.position;
	let _ = position;
	let mut pixels = vec![0_u8; width * height * components];
	if components == 1 {
		for row in 0..height {
			for column in 0..width {
				pixels[row * width + column] = planes[0][row * blocks_x * 8 + column];
			}
		}
	} else {
		// libjpeg ycc_rgb_convert: 16-bit fixed-point coefficients with one-half rounding.
		let fix = |value: f64| (value * 65536.0 + 0.5) as i64;
		for row in 0..height {
			for column in 0..width {
				let index = row * blocks_x * 8 + column;
				let (y, cb, cr) = (i64::from(planes[0][index]), i64::from(planes[1][index]) - 128, i64::from(planes[2][index]) - 128);
				let clamp = |value: i64| value.clamp(0, 255) as u8;
				let red = y + (fix(1.40200) * cr + 32768 >> 16);
				let green = y + (-fix(0.34414) * cb - fix(0.71414) * cr + 32768 >> 16);
				let blue = y + (fix(1.77200) * cb + 32768 >> 16);
				let out = (row * width + column) * 3;
				pixels[out] = clamp(red);
				pixels[out + 1] = clamp(green);
				pixels[out + 2] = clamp(blue);
			}
		}
	}
	Ok((width, height, components, pixels))
}
/// Decoded 8-bit PNG pixels: grayscale or RGB, no interlacing, all five scanline filters.
fn png_pixels(bytes: &[u8]) -> Result<(usize, usize, usize, Vec<u8>)> {
	require(bytes.get(..8) == Some(&b"\x89PNG\r\n\x1a\n"[..]), "PNG signature is absent")?;
	let (mut offset, mut header, mut compressed) = (8, None, Vec::new());
	while offset + 8 <= bytes.len() {
		let length = u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
		let kind = &bytes[offset + 4..offset + 8];
		let body = bytes.get(offset + 8..offset + 8 + length).ok_or_else(|| RecipeError::new("PNG chunk is truncated"))?;
		match kind {
			b"IHDR" => {
				require(body.len() == 13, "PNG header has the wrong size")?;
				let width = u32::from_be_bytes(body[..4].try_into().unwrap()) as usize;
				let height = u32::from_be_bytes(body[4..8].try_into().unwrap()) as usize;
				let (depth, color, interlace) = (body[8], body[9], body[12]);
				require(depth == 8, format!("PNG bit depth {depth} is unsupported"))?;
				require(interlace == 0, "PNG interlacing is unsupported")?;
				let channels = match color {
					0 => 1,
					2 => 3,
					color => return Err(RecipeError::new(format!("PNG color type {color} is unsupported"))),
				};
				header = Some((width, height, channels));
			}
			b"IDAT" => compressed.extend_from_slice(body),
			b"IEND" => break,
			_ => {}
		}
		offset += 12 + length;
	}
	let (width, height, channels) = header.ok_or_else(|| RecipeError::new("PNG header is absent"))?;
	let raw = zlib_inflate(&compressed)?;
	let stride = checked_mul(width, channels, "PNG scanline")?;
	require(raw.len() == checked_mul(height, stride + 1, "PNG image")?, "PNG data has the wrong size")?;
	let mut pixels = vec![0_u8; height * stride];
	for row in 0..height {
		let filter = raw[row * (stride + 1)];
		let line = &raw[row * (stride + 1) + 1..(row + 1) * (stride + 1)];
		for column in 0..stride {
			let left = if column >= channels { pixels[row * stride + column - channels] } else { 0 };
			let above = if row > 0 { pixels[(row - 1) * stride + column] } else { 0 };
			let corner = if row > 0 && column >= channels { pixels[(row - 1) * stride + column - channels] } else { 0 };
			let predictor = match filter {
				0 => 0,
				1 => left,
				2 => above,
				3 => ((u16::from(left) + u16::from(above)) / 2) as u8,
				4 => {
					let estimate = i32::from(left) + i32::from(above) - i32::from(corner);
					let (da, db, dc) = ((estimate - i32::from(left)).abs(), (estimate - i32::from(above)).abs(), (estimate - i32::from(corner)).abs());
					if da <= db && da <= dc {
						left
					} else if db <= dc {
						above
					} else {
						corner
					}
				}
				filter => return Err(RecipeError::new(format!("PNG filter {filter} is unsupported"))),
			};
			pixels[row * stride + column] = line[column].wrapping_add(predictor);
		}
	}
	Ok((width, height, channels, pixels))
}
fn prepare_data(data: &Data) -> Result<Prepared> {
	let (mut tables, sources) = load_tables(data, &data.sources)?;
	let source_table_rows = tables.first().map_or(0, |table| table.rows.len());
	if !data.tests.is_empty() {
		let (tests, test_sources) = load_tables(data, &data.tests)?;
		require(!sources.iter().any(|source| test_sources.binary_search(source).is_ok()), "training and test data must use separate files")?;
		require(tables.len() == tests.len(), "test data table count differs from training data")?;
		for (table, test) in tables.iter_mut().zip(tests) {
			require(table.headers == test.headers && table.attention == test.attention, format!("test table {:?} differs from training table {:?}", test.name, table.name))?;
			table.rows.extend(test.rows);
		}
	}
	if data.autoregressive {
		require(data.tests.is_empty(), "autoregressive test data is unsupported")?;
		return prepare_autoregression(data, &tables);
	}
	let mut selected = Vec::new();
	for name in &data.target {
		let mut matches = Vec::new();
		for (table, value) in tables.iter().enumerate() {
			for (column, header) in value.headers.iter().enumerate() {
				if column_match(name, value, header, column) {
					matches.push((table, column));
				}
			}
		}
		if matches.len() != 1 {
			let grouped = !matches.is_empty()
				&& matches.iter().all(|(table, column)| tables[*table].headers[*column].rsplit_once('.').is_some_and(|(base, suffix)| base == name && suffix.parse::<usize>().is_ok()));
			require(grouped, format!("target {name:?} must identify exactly one feature or a numbered group"))?;
			selected.extend(matches);
			continue;
		}
		selected.push(matches[0]);
	}
	let table_index = selected.first().map_or(0, |target| target.0);
	let row_count = tables[table_index].rows.len();
	for table in &tables {
		require(table.rows.len() == row_count, format!("table {:?} expected {row_count} positionally aligned rows, received {}", table.name, table.rows.len()))?
	}
	let mut columns = Vec::new();
	for (table, value) in tables.iter().enumerate() {
		for (column, header) in value.headers.iter().enumerate() {
			if !selected.contains(&(table, column)) && data.features.selects(value, header, column) {
				columns.push((table, column, infer_feature(value, column, source_table_rows)));
			}
		}
	}
	let features = columns.iter().map(|column| column.2.width()).sum();
	let mut sequence_widths = BTreeMap::new();
	let repeated = columns.iter().all(|column| {
		tables[column.0].headers[column.1].rsplit_once('.').and_then(|value| value.1.parse::<usize>().ok().map(|row| *sequence_widths.entry(row).or_insert(0) += column.2.width())).is_some()
	});
	let sequence = (repeated && sequence_widths.len() > 1 && sequence_widths.keys().copied().eq(1..=sequence_widths.len()) && sequence_widths.values().all(|width| *width == sequence_widths[&1]))
		.then(|| Shape { channels: sequence_widths[&1], length: sequence_widths.len() });
	let attention = tables.iter().filter_map(|table| table.attention).find(|shape| shape.elements() == features);
	let shapes = sequence.map(|sequence| (sequence, attention.unwrap_or(sequence)));
	require(features != 0, "dataset has no training features")?;
	let target_categories = selected.iter().map(|target| categories(&tables[target.0], target.1, source_table_rows)).collect::<Vec<_>>();
	let target_categorical =
		selected.iter().any(|target| tables[target.0].rows.iter().take(source_table_rows).filter_map(|row| row.get(target.1)).any(|value| !value.is_empty() && value.parse::<f64>().is_err()));
	let target_width = selected.len().max(1);
	let mut samples = Vec::new();
	let mut targets = Vec::new();
	let mut source_rows = 0;
	let mut missing = vec![0_usize; columns.len()];
	for row in 0..row_count {
		if row == source_table_rows {
			source_rows = targets.len() / target_width
		}
		let mut encoded = Vec::with_capacity(features);
		let valid = columns.iter().all(|column| tables[column.0].rows[row].get(column.1).is_some_and(|value| encode(value, &column.2, &mut encoded)));
		if valid {
			if let Some(shape) = sequence {
				let mut ordered = Vec::with_capacity(features);
				for channel in 0..shape.channels {
					for position in 0..shape.length {
						ordered.push(encoded[position * shape.channels + channel]);
					}
				}
				encoded = ordered;
			}
		}
		if valid && selected.is_empty() {
			samples.extend_from_slice(&encoded);
			targets.push(0.0);
			for (count, column) in missing.iter_mut().zip(&columns) {
				*count += usize::from(tables[column.0].rows[row][column.1].is_empty());
			}
		} else if valid {
			// One row is one sample whose target is the vector of its selected columns. A row
			// missing any of them contributes nothing, exactly as a missing scalar target did.
			let row_targets = selected
				.iter()
				.zip(&target_categories)
				.map(|(target, categories)| {
					let value = tables[target.0].rows[row].get(target.1);
					value.and_then(|value| value.parse::<f64>().ok())
						.or_else(|| value.and_then(|value| categories.iter().position(|category| category == value)).map(|value| value as f64))
						.filter(|target| target.is_finite())
				})
				.collect::<Option<Vec<_>>>();
			if let Some(row_targets) = row_targets {
				samples.extend_from_slice(&encoded);
				targets.extend(row_targets);
				for (count, column) in missing.iter_mut().zip(&columns) {
					*count += usize::from(tables[column.0].rows[row][column.1].is_empty());
				}
			}
		}
	}
	if source_table_rows == row_count {
		source_rows = targets.len() / target_width
	}
	for (column, count) in columns.iter().zip(missing).filter(|value| value.1 != 0) {
		let percentage = count as f64 * 100.0 / row_count as f64;
		let precision = 4_usize.max((-percentage.log10()).ceil().max(0.0) as usize);
		eprintln!("imputed {}.{}: {percentage:.precision$}%", tables[column.0].name, tables[column.0].headers[column.1]);
	}
	let schema = columns
		.iter()
		.map(|column| ("feature".to_owned(), format!("{} {}.{}", column.2.width(), tables[column.0].name, tables[column.0].headers[column.1])))
		.chain(data.target.iter().cloned().map(|target| ("target".to_owned(), target)))
		.collect();
	finish_prepared(data, samples, targets, target_width, source_rows, features, shapes, target_categorical, schema)
}
fn prepare_autoregression(data: &Data, tables: &[Table]) -> Result<Prepared> {
	let mut sequences = Vec::new();
	for table in tables {
		for column in 0..table.headers.len() {
			if matches!(infer_feature(table, column, table.rows.len()), FeatureType::Numeric) {
				continue;
			}
			for (row, values) in table.rows.iter().enumerate() {
				let text = values.get(column).cloned().unwrap_or_default();
				let chars = text
					.chars()
					.map(|character| CHAR_IDS.iter().position(|value| *value == character).ok_or_else(|| RecipeError::new(format!("unsupported character {character:?} in row {}", row + 1))))
					.collect::<Result<Vec<_>>>()?;
				if !chars.is_empty() {
					sequences.push(chars)
				}
			}
		}
	}
	let length = sequences.iter().map(|sequence| sequence.len().saturating_sub(1)).max().unwrap_or(0);
	require(length != 0, "autoregression requires a string containing at least two characters")?;
	let features = checked_mul(CHAR_IDS.len(), length, "autoregression input width")?;
	let mut samples = Vec::new();
	let mut targets = Vec::new();
	for sequence in &sequences {
		for prefix in 1..sequence.len() {
			let mut sample = vec![0.0; features];
			for (position, id) in sequence[..prefix].iter().copied().enumerate() {
				sample[id * length + position] = 1.0
			}
			samples.extend(sample);
			targets.push(sequence[prefix] as f64)
		}
	}
	let schema = CHAR_IDS.iter().map(|character| ("character".to_owned(), format!("U+{:04X}", *character as u32))).collect();
	let source_rows = targets.len();
	let sequence = Shape { channels: CHAR_IDS.len(), length };
	finish_prepared(data, samples, targets, 1, source_rows, features, Some((sequence, sequence)), true, schema)
}
fn finish_prepared(
	data: &Data, mut samples: Vec<f64>, mut targets: Vec<f64>, target_width: usize, source_rows: usize, features: usize, sequence: Option<(Shape, Shape)>, target_categorical: bool,
	schema: DataSchema,
) -> Result<Prepared> {
	require(target_width != 0 && targets.len() % target_width == 0, "target vector width does not divide the target buffer")?;
	let rows = targets.len() / target_width;
	require(source_rows != 0 && source_rows <= rows, "dataset has no complete training rows")?;
	// Sources may repeat a row verbatim; each copy is its own sample, so its identity mixes
	// in how many identical rows precede it in source order, which the seed never changes.
	let mut occurrences = BTreeMap::new();
	let mut identities = samples
		.chunks_exact(features)
		.zip(targets.chunks_exact(target_width))
		.map(|(sample, target)| {
			let content = target[1..].iter().fold(sample_identity(sample, target[0]), |hash, value| (hash ^ value.to_bits()).wrapping_mul(1099511628211));
			let occurrence = occurrences.entry(content).and_modify(|count| *count += 1_u64).or_insert(0);
			occurrence.to_le_bytes().iter().fold(content, |hash, byte| (hash ^ u64::from(*byte)).wrapping_mul(1099511628211))
		})
		.collect();
	shuffle(&mut samples, &mut targets, &mut identities, features, source_rows, target_width)?;
	let (norm_mean, norm_scale) = if data.normalize {
		normalize_samples(&mut samples, features, ((source_rows as f64) * data.split).floor() as usize)?
	} else {
		impute_missing(&mut samples);
		(Vec::new(), Vec::new())
	};
	Ok(Prepared { samples, targets, target_width, rows, source_rows, features, schema, sequence, target_categorical, norm_mean, norm_scale, identities, fitted: Vec::new(), bound: None })
}
fn normalize_samples(samples: &mut [f64], features: usize, fit: usize) -> Result<(Vec<f64>, Vec<f64>)> {
	require(fit != 0, "split must retain normalization rows")?;
	let epsilon = number("normalization epsilon", env!("RECIPE_NORMALIZATION_EPSILON"))?;
	let (mut means, mut scales) = (Vec::with_capacity(features), Vec::with_capacity(features));
	for column in 0..features {
		let valid = (0..fit).filter(|&row| samples[row * features + column].is_finite()).collect::<Vec<_>>();
		let count = valid.len().max(1) as f64;
		let mean = valid.iter().map(|&row| samples[row * features + column]).sum::<f64>() / count;
		let variance = valid.iter().map(|&row| (samples[row * features + column] - mean).powi(2)).sum::<f64>() / count;
		let scale = (variance + epsilon).sqrt();
		for row in 0..samples.len() / features {
			let value = &mut samples[row * features + column];
			*value = (if value.is_finite() { *value } else { mean } - mean) / scale;
		}
		means.push(mean);
		scales.push(scale);
	}
	Ok((means, scales))
}
fn impute_missing(samples: &mut [f64]) {
	for value in samples.iter_mut() {
		if !value.is_finite() {
			*value = 0.0
		}
	}
}
fn sample_identity(sample: &[f64], target: f64) -> u64 {
	const OFFSET: u64 = 14695981039346656037;
	const PRIME: u64 = 1099511628211;
	// Feed the hash bytewise: word-wide mixing leaves the low hash bits untouched by the
	// all-zero low mantissa bits of small integer values, collapsing the identity space.
	sample.iter().copied().chain(std::iter::once(target)).flat_map(|value| value.to_bits().to_le_bytes()).fold(OFFSET, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(PRIME))
}
fn is_table(extension: &str) -> bool {
	matches!(extension.to_ascii_lowercase().as_str(), "csv" | "tsv" | "txt" | "data" | "dat" | "all-data" | "jsonl" | "json" | "npz" | "sqlite" | "sqlite3" | "db" | "h5" | "hdf5" | "xml")
}
fn is_archive(extension: &str) -> bool {
	matches!(extension.to_ascii_lowercase().as_str(), "zip")
}
fn resolve_path(path: impl AsRef<Path>) -> Result<PathBuf> {
	let path = path.as_ref();
	let mut components = path.components();
	if components.next().is_some_and(|component| component.as_os_str() == "~") {
		let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).ok_or_else(|| RecipeError::new("home directory is absent"))?;
		return Ok(PathBuf::from(home).join(components.as_path()));
	}
	Ok(path.to_owned())
}
fn collect_files(path: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
	let metadata = fs::metadata(path).map_err(|error| RecipeError::new(format!("cannot inspect {}: {error}", path.display())))?;
	if metadata.is_file() {
		files.push(path.to_owned());
		return Ok(());
	}
	let mut children = fs::read_dir(path)
		.map_err(|error| RecipeError::new(format!("cannot read {}: {error}", path.display())))?
		.collect::<std::io::Result<Vec<_>>>()
		.map_err(|error| RecipeError::new(format!("cannot read {}: {error}", path.display())))?;
	children.sort_by_key(fs::DirEntry::path);
	for child in children {
		collect_files(&child.path(), files)?;
	}
	Ok(())
}
fn target_column(table: &Table, name: &str) -> Option<usize> {
	table.headers.iter().enumerate().position(|(column, header)| column_match(name, table, header, column))
}
fn merge_captures(tables: Vec<(PathBuf, Table)>, targets: &[String]) -> Result<Vec<Table>> {
	let mut groups = BTreeMap::<PathBuf, Vec<Table>>::new();
	for (directory, table) in tables {
		groups.entry(directory).or_default().push(table);
	}
	let valid = |group: &[Table]| {
		group.len() > 1
			&& targets.iter().all(|target| {
				group.iter().filter(|table| target_column(table, target).is_some()).count() == 1
					&& group.iter().find(|table| target_column(table, target).is_some()).is_some_and(|table| table.rows.len() == 1)
			})
	};
	if targets.is_empty() || groups.values().all(|group| !valid(group)) {
		let mut tables = Vec::new();
		for mut group in groups.into_values() {
			// Tables in one directory align onto each other only when they contribute different columns. Same columns means more rows of one table, and broadcasting those duplicates records instead of widening them.
			if group.iter().any(|table| table.headers != group[0].headers) {
				let rows = group.iter().map(|table| table.rows.len()).max().unwrap_or(0);
				for table in &mut group {
					let count = table.rows.len();
					require(count != 0 && rows % count == 0, format!("table {:?} expected a nonzero row count dividing {rows}, received {count}", table.name))?;
					if count != rows {
						table.rows = table.rows.iter().cloned().cycle().take(rows).collect()
					}
				}
			}
			tables.extend(group);
		}
		return Ok(tables);
	}
	groups.retain(|_, group| group.iter().all(|table| !table.rows.is_empty()));
	require(!groups.is_empty(), "data source contains no usable captures")?;
	let mut captures = groups.into_values().collect::<Vec<_>>();
	let key = |table: &Table| (table.headers.join("\0"), table.rows.len());
	for capture in &mut captures {
		capture.sort_by_key(&key);
	}
	let schemas = captures[0].iter().map(|table| (table.headers.clone(), table.rows.len())).collect::<Vec<_>>();
	for (capture_index, capture) in captures.iter().enumerate() {
		require(capture.len() == schemas.len(), format!("capture {capture_index} expected {} tables, received {}", schemas.len(), capture.len()))?;
		for (table_index, (table, schema)) in capture.iter().zip(&schemas).enumerate() {
			require(
				table.headers == schema.0 && table.rows.len() == schema.1,
				format!(
					"capture {capture_index} table {table_index} expected {} columns and {} rows, received {} columns and {} rows",
					schema.0.len(),
					schema.1,
					table.headers.len(),
					table.rows.len()
				),
			)?
		}
	}
	let names = (0..schemas.len())
		.map(|index| {
			let name = &captures[0][index].name;
			if captures.iter().all(|capture| capture[index].name == *name) { name.clone() } else { format!("table{}", index + 1) }
		})
		.collect::<Vec<_>>();
	let mut headers = Vec::new();
	for (table, name) in captures[0].iter().zip(&names) {
		for row in 0..table.rows.len() {
			for header in &table.headers {
				if targets.contains(header) {
					headers.push(header.clone());
				} else if table.rows.len() == 1 {
					headers.push(format!("{name}.{header}"));
				} else {
					headers.push(format!("{name}.{header}.{}", row + 1));
				}
			}
		}
	}
	let mut rows = Vec::with_capacity(captures.len());
	for capture in captures {
		let row = capture.into_iter().flat_map(|table| table.rows.into_iter().flatten()).collect::<Vec<_>>();
		require(row.len() == headers.len(), "capture value width differs")?;
		rows.push(row);
	}
	Ok(vec![Table { name: "data".to_owned(), headers, rows, attention: None }])
}
fn merge_partitions(mut tables: Vec<Table>, targets: &[String], features: &FeatureSelection) -> Result<Vec<Table>> {
	if targets.is_empty() || targets.iter().any(|target| target.contains('.')) {
		return Ok(tables);
	}
	let members = tables.iter().enumerate().filter_map(|(index, table)| targets.iter().all(|target| target_column(table, target).is_some()).then_some(index)).collect::<Vec<_>>();
	if members.len() < 2 {
		return Ok(tables);
	}
	let mut headers = Vec::new();
	for &index in &members {
		for header in &tables[index].headers {
			if !headers.contains(header) {
				headers.push(header.clone())
			}
		}
	}
	let union = Table { name: "data".to_owned(), headers: headers.clone(), rows: Vec::new(), attention: None };
	for &index in &members {
		for (column, header) in headers.iter().enumerate() {
			let ignored = targets.iter().any(|name| column_match(name, &union, header, column)) || !features.selects(&union, header, column);
			require(ignored || tables[index].headers.contains(header), format!("feature {header:?} is absent from partition {:?}", tables[index].name))?;
		}
	}
	let mut rows = Vec::new();
	for index in members {
		let positions = tables[index].headers.iter().map(|header| headers.iter().position(|value| value == header).unwrap()).collect::<Vec<_>>();
		for row in std::mem::take(&mut tables[index].rows) {
			let mut merged = std::iter::repeat_with(String::new).take(headers.len()).collect::<Vec<_>>();
			for (column, value) in row.into_iter().enumerate() {
				merged[positions[column]] = value;
			}
			rows.push(merged);
		}
	}
	let name = "data".to_owned();
	Ok(vec![Table { name, headers, rows, attention: None }])
}
/// Decode one source file into its tables, dispatching on the container format.
fn decode_tables(path: &Path, bytes: &[u8]) -> Result<Vec<Table>> {
	let name = path.file_stem().and_then(|value| value.to_str()).unwrap_or("data").to_owned();
	match path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref() {
		Some("jsonl") => {
			let text = str::from_utf8(bytes).map_err(|error| RecipeError::new(format!("dataset {} is not UTF-8: {error}", path.display())))?;
			let records = text
				.lines()
				.map(str::trim)
				.filter(|line| !line.is_empty())
				.map(|line| {
					let (value, rest) = json_value(line)?;
					require(rest.trim().is_empty(), format!("JSONL record has trailing content {:?}", rest.trim()))?;
					Ok(value)
				})
				.collect::<Result<Vec<_>>>()
				.map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?;
			Ok(vec![json_records_table(name, &records).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?])
		}
		Some("json") => {
			let text = str::from_utf8(bytes).map_err(|error| RecipeError::new(format!("dataset {} is not UTF-8: {error}", path.display())))?;
			let records = json_array(text).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?;
			Ok(vec![json_records_table(name, &records).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?])
		}
		Some("npz") => {
			let mut columns = Vec::new();
			for (entry, contents) in zip_entries(bytes).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))? {
				let array = entry.strip_suffix(".npy").unwrap_or(&entry).to_owned();
				columns.extend(npy_columns(&array, &contents).map_err(|error| RecipeError::new(format!("dataset {} entry {entry}: {error}", path.display())))?);
			}
			Ok(vec![array_table(name, columns).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?])
		}
		Some("sqlite" | "sqlite3" | "db") => sqlite_tables(bytes).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display()))),
		Some("xml") => {
			let text = str::from_utf8(bytes).map_err(|error| RecipeError::new(format!("dataset {} is not UTF-8: {error}", path.display())))?;
			let records = xml_records(text).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?;
			Ok(vec![json_records_table(name, &records).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?])
		}
		Some("h5" | "hdf5") => {
			let columns = hdf5_columns(bytes).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?;
			Ok(vec![array_table(name, columns).map_err(|error| RecipeError::new(format!("dataset {}: {error}", path.display())))?])
		}
		_ => parse_table(path, bytes).map(|(table, _)| vec![table]),
	}
}
/// Every user table of a SQLite database, walked from its rowid b-trees.
fn sqlite_tables(bytes: &[u8]) -> Result<Vec<Table>> {
	require(bytes.get(..16) == Some(b"SQLite format 3\0"), "SQLite header is absent")?;
	let page_size = match u16::from_be_bytes(bytes[16..18].try_into().unwrap()) as usize {
		1 => 65536,
		size => size,
	};
	let mut schema = Vec::new();
	sqlite_rows(bytes, page_size, 1, &mut schema)?;
	let mut tables = Vec::new();
	for row in schema {
		let [kind, name, _, root, sql] = row.as_slice() else { return Err(RecipeError::new("SQLite schema row has the wrong width")) };
		if kind != "table" || name.starts_with("sqlite_") {
			continue;
		}
		let root = root.parse::<usize>().map_err(|error| RecipeError::new(format!("invalid SQLite root page: {error}")))?;
		let columns =
			sql.split_once('(').map(|(_, rest)| rest.rsplit_once(')').map_or(rest, |(inner, _)| inner)).ok_or_else(|| RecipeError::new(format!("SQLite table {name:?} has no column list")))?;
		let headers = columns.split(',').map(|column| column.trim().split_whitespace().next().unwrap_or("").trim_matches(['"', '\'', '`', '[', ']']).to_owned()).collect::<Vec<_>>();
		require(headers.iter().all(|header| !header.is_empty()), format!("SQLite table {name:?} has an unreadable column list"))?;
		let mut rows = Vec::new();
		sqlite_rows(bytes, page_size, root, &mut rows)?;
		for row in &mut rows {
			require(row.len() <= headers.len(), format!("SQLite table {name:?} row exceeds {} columns", headers.len()))?;
			row.resize_with(headers.len(), String::new);
		}
		tables.push(Table { name: name.clone(), headers, rows, attention: None });
	}
	require(!tables.is_empty(), "SQLite database has no tables")?;
	Ok(tables)
}
/// In-order rowid b-tree walk appending each leaf record's decoded values.
fn sqlite_rows(bytes: &[u8], page_size: usize, page: usize, rows: &mut Vec<Vec<String>>) -> Result<()> {
	let start = checked_mul(page - 1, page_size, "SQLite page offset")?;
	let header = start + if page == 1 { 100 } else { 0 };
	let contents = bytes.get(start..start + page_size).ok_or_else(|| RecipeError::new(format!("SQLite page {page} is truncated")))?;
	let kind = *bytes.get(header).ok_or_else(|| RecipeError::new(format!("SQLite page {page} is truncated")))?;
	let cells = u16::from_be_bytes(bytes[header + 3..header + 5].try_into().unwrap()) as usize;
	let pointers = header + if kind == 5 { 12 } else { 8 };
	for cell in 0..cells {
		let pointer = u16::from_be_bytes(bytes[pointers + cell * 2..pointers + cell * 2 + 2].try_into().unwrap()) as usize;
		let mut offset = start + pointer;
		match kind {
			5 => {
				let child = u32::from_be_bytes(bytes.get(offset..offset + 4).ok_or_else(|| RecipeError::new("SQLite interior cell is truncated"))?.try_into().unwrap()) as usize;
				sqlite_rows(bytes, page_size, child, rows)?;
			}
			13 => {
				let (payload, _) = sqlite_varint(bytes, &mut offset)?;
				let _ = sqlite_varint(bytes, &mut offset)?;
				let usable = page_size - 35;
				require((payload as usize) <= usable, format!("SQLite page {page} overflows; overflow pages are unsupported"))?;
				rows.push(sqlite_record(bytes.get(offset..offset + payload as usize).ok_or_else(|| RecipeError::new("SQLite record is truncated"))?)?);
			}
			_ => return Err(RecipeError::new(format!("SQLite page type {kind} is unsupported"))),
		}
	}
	if kind == 5 {
		let right = u32::from_be_bytes(bytes[header + 8..header + 12].try_into().unwrap()) as usize;
		sqlite_rows(bytes, page_size, right, rows)?;
	}
	let _ = contents;
	Ok(())
}
fn sqlite_varint(bytes: &[u8], offset: &mut usize) -> Result<(i64, usize)> {
	let mut value = 0_i64;
	for length in 1..=9 {
		let byte = *bytes.get(*offset).ok_or_else(|| RecipeError::new("SQLite varint is truncated"))?;
		*offset += 1;
		if length == 9 {
			value = value << 8 | i64::from(byte);
			return Ok((value, length));
		}
		value = value << 7 | i64::from(byte & 0x7f);
		if byte & 0x80 == 0 {
			return Ok((value, length));
		}
	}
	unreachable!()
}
/// Decode one SQLite record into per-column text values.
fn sqlite_record(record: &[u8]) -> Result<Vec<String>> {
	let mut offset = 0;
	let (header, _) = sqlite_varint(record, &mut offset)?;
	let mut serials = Vec::new();
	while offset < header as usize {
		serials.push(sqlite_varint(record, &mut offset)?.0);
	}
	let mut body = header as usize;
	let mut values = Vec::with_capacity(serials.len());
	for serial in serials {
		let mut integer = |width: usize| -> Result<i64> {
			let mut value = if record.get(body).is_some_and(|byte| byte & 0x80 != 0) { -1_i64 } else { 0 };
			for _ in 0..width {
				value = value << 8 | i64::from(*record.get(body).ok_or_else(|| RecipeError::new("SQLite value is truncated"))?);
				body += 1;
			}
			Ok(value)
		};
		values.push(match serial {
			0 => String::new(),
			1 => integer(1)?.to_string(),
			2 => integer(2)?.to_string(),
			3 => integer(3)?.to_string(),
			4 => integer(4)?.to_string(),
			5 => integer(6)?.to_string(),
			6 => integer(8)?.to_string(),
			7 => {
				let value = f64::from_bits(integer(8)? as u64);
				value.to_string()
			}
			8 => "0".to_owned(),
			9 => "1".to_owned(),
			serial if serial >= 13 && serial % 2 == 1 => {
				let length = (serial as usize - 13) / 2;
				let text = record.get(body..body + length).ok_or_else(|| RecipeError::new("SQLite text is truncated"))?;
				body += length;
				String::from_utf8(text.to_vec()).map_err(|error| RecipeError::new(format!("SQLite text is not UTF-8: {error}")))?
			}
			serial => return Err(RecipeError::new(format!("SQLite serial type {serial} is unsupported"))),
		});
	}
	Ok(values)
}
/// Raw DEFLATE decompression (RFC 1951): stored, fixed, and dynamic Huffman blocks.
fn inflate(bytes: &[u8]) -> Result<Vec<u8>> {
	struct Bits<'a> {
		bytes: &'a [u8],
		position: usize,
	}
	impl Bits<'_> {
		fn bit(&mut self) -> Result<u64> {
			let byte = *self.bytes.get(self.position / 8).ok_or_else(|| RecipeError::new("DEFLATE stream is truncated"))?;
			let bit = u64::from(byte >> (self.position % 8) & 1);
			self.position += 1;
			Ok(bit)
		}
		fn bits(&mut self, count: u32) -> Result<u64> {
			let mut value = 0;
			for index in 0..count {
				value |= self.bit()? << index;
			}
			Ok(value)
		}
	}
	struct Huffman {
		counts: [u16; 16],
		symbols: Vec<u16>,
	}
	impl Huffman {
		fn new(lengths: &[u8]) -> Result<Self> {
			let mut counts = [0_u16; 16];
			for &length in lengths {
				require(length < 16, "DEFLATE code length exceeds 15")?;
				counts[length as usize] += 1;
			}
			counts[0] = 0;
			let mut offsets = [0_u16; 16];
			for length in 1..16 {
				offsets[length] = offsets[length - 1] + counts[length - 1];
			}
			let mut symbols = vec![0_u16; lengths.iter().filter(|length| **length != 0).count()];
			for (symbol, &length) in lengths.iter().enumerate() {
				if length != 0 {
					symbols[offsets[length as usize] as usize] = symbol as u16;
					offsets[length as usize] += 1;
				}
			}
			Ok(Self { counts, symbols })
		}
		fn decode(&self, bits: &mut Bits) -> Result<u16> {
			let (mut code, mut first, mut index) = (0_u32, 0_u32, 0_u32);
			for length in 1..16 {
				code |= bits.bit()? as u32;
				let count = u32::from(self.counts[length]);
				if code < first + count {
					return Ok(self.symbols[(index + code - first) as usize]);
				}
				index += count;
				first = (first + count) << 1;
				code <<= 1;
			}
			Err(RecipeError::new("DEFLATE code is invalid"))
		}
	}
	const LENGTH_BASE: [u16; 29] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
	const LENGTH_EXTRA: [u32; 29] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
	const DISTANCE_BASE: [u16; 30] = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
	const DISTANCE_EXTRA: [u32; 30] = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
	let mut bits = Bits { bytes, position: 0 };
	let mut output = Vec::new();
	loop {
		let last = bits.bit()?;
		match bits.bits(2)? {
			0 => {
				bits.position = bits.position.div_ceil(8) * 8;
				let start = bits.position / 8;
				let length = u16::from_le_bytes(bytes.get(start..start + 2).ok_or_else(|| RecipeError::new("DEFLATE stream is truncated"))?.try_into().unwrap()) as usize;
				output.extend_from_slice(bytes.get(start + 4..start + 4 + length).ok_or_else(|| RecipeError::new("DEFLATE stream is truncated"))?);
				bits.position = (start + 4 + length) * 8;
			}
			kind @ (1 | 2) => {
				let (literals, distances) = if kind == 1 {
					let mut lengths = [8_u8; 288];
					lengths[144..256].fill(9);
					lengths[256..280].fill(7);
					(Huffman::new(&lengths)?, Huffman::new(&[5; 30])?)
				} else {
					const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
					let literal_count = bits.bits(5)? as usize + 257;
					let distance_count = bits.bits(5)? as usize + 1;
					let code_count = bits.bits(4)? as usize + 4;
					let mut code_lengths = [0_u8; 19];
					for index in 0..code_count {
						code_lengths[ORDER[index]] = bits.bits(3)? as u8;
					}
					let codes = Huffman::new(&code_lengths)?;
					let mut lengths = vec![0_u8; literal_count + distance_count];
					let mut index = 0;
					while index < lengths.len() {
						match codes.decode(&mut bits)? {
							16 => {
								let previous = *lengths.get(index.wrapping_sub(1)).ok_or_else(|| RecipeError::new("DEFLATE repeat has no previous length"))?;
								for _ in 0..bits.bits(2)? + 3 {
									require(index < lengths.len(), "DEFLATE code lengths overflow")?;
									lengths[index] = previous;
									index += 1;
								}
							}
							17 => index += bits.bits(3)? as usize + 3,
							18 => index += bits.bits(7)? as usize + 11,
							length => {
								lengths[index] = length as u8;
								index += 1;
							}
						}
					}
					(Huffman::new(&lengths[..literal_count])?, Huffman::new(&lengths[literal_count..])?)
				};
				loop {
					match literals.decode(&mut bits)? {
						symbol if symbol < 256 => output.push(symbol as u8),
						256 => break,
						symbol => {
							let entry = symbol as usize - 257;
							require(entry < LENGTH_BASE.len(), "DEFLATE length code is invalid")?;
							let length = LENGTH_BASE[entry] as usize + bits.bits(LENGTH_EXTRA[entry])? as usize;
							let code = distances.decode(&mut bits)? as usize;
							require(code < DISTANCE_BASE.len(), "DEFLATE distance code is invalid")?;
							let distance = DISTANCE_BASE[code] as usize + bits.bits(DISTANCE_EXTRA[code])? as usize;
							require(distance <= output.len(), "DEFLATE distance exceeds output")?;
							for _ in 0..length {
								output.push(output[output.len() - distance]);
							}
						}
					}
				}
			}
			_ => return Err(RecipeError::new("DEFLATE block type is invalid")),
		}
		if last == 1 {
			return Ok(output);
		}
	}
}
/// zlib envelope: header check, DEFLATE body, Adler-32 verification.
fn zlib_inflate(bytes: &[u8]) -> Result<Vec<u8>> {
	require(bytes.len() > 6 && bytes[0] & 0xf == 8 && (u16::from(bytes[0]) << 8 | u16::from(bytes[1])) % 31 == 0 && bytes[1] & 0x20 == 0, "zlib header is invalid")?;
	let output = inflate(&bytes[2..bytes.len() - 4])?;
	let (mut low, mut high) = (1_u32, 0_u32);
	for byte in &output {
		low = (low + u32::from(*byte)) % 65521;
		high = (high + low) % 65521;
	}
	let expected = u32::from_be_bytes(bytes[bytes.len() - 4..].try_into().unwrap());
	require(high << 16 | low == expected, "zlib checksum mismatch")?;
	Ok(output)
}
/// Named columns from every dataset of an HDF5 file (version-0 superblock, symbol-table
/// groups, version-1 object headers, contiguous or chunked layouts, optional deflate filter).
fn hdf5_columns(bytes: &[u8]) -> Result<Vec<(String, usize, Vec<f64>)>> {
	let read16 = |offset: usize| bytes.get(offset..offset + 2).map(|value| u16::from_le_bytes(value.try_into().unwrap()) as usize);
	let read32 = |offset: usize| bytes.get(offset..offset + 4).map(|value| u32::from_le_bytes(value.try_into().unwrap()) as usize);
	let read64 = |offset: usize| bytes.get(offset..offset + 8).map(|value| u64::from_le_bytes(value.try_into().unwrap()) as usize);
	let truncated = || RecipeError::new("HDF5 file is truncated");
	require(bytes.get(..8) == Some(&b"\x89HDF\r\n\x1a\n"[..]), "HDF5 signature is absent")?;
	require(bytes.get(8) == Some(&0), "HDF5 superblock version is unsupported")?;
	require(bytes.get(13) == Some(&8) && bytes.get(14) == Some(&8), "HDF5 offset or length size is unsupported")?;
	let root_header = read64(0x40).ok_or_else(truncated)?;
	fn messages(bytes: &[u8], output: &mut Vec<(u16, usize, usize)>, remaining: &mut usize, start: usize, end: usize) -> Result<()> {
		let mut offset = start;
		while offset + 8 <= end && *remaining != 0 {
			let kind = u16::from_le_bytes(bytes.get(offset..offset + 2).ok_or_else(|| RecipeError::new("HDF5 message is truncated"))?.try_into().unwrap());
			let size = u16::from_le_bytes(bytes[offset + 2..offset + 4].try_into().unwrap()) as usize;
			let body = offset + 8;
			*remaining -= 1;
			if kind == 0x10 {
				let continuation = u64::from_le_bytes(bytes.get(body..body + 8).ok_or_else(|| RecipeError::new("HDF5 continuation is truncated"))?.try_into().unwrap()) as usize;
				let length = u64::from_le_bytes(bytes[body + 8..body + 16].try_into().unwrap()) as usize;
				messages(bytes, output, remaining, continuation, continuation + length)?;
			} else {
				output.push((kind, body, size));
			}
			offset = body + size;
		}
		Ok(())
	}
	let object_messages = |header: usize| -> Result<Vec<(u16, usize, usize)>> {
		require(bytes.get(header) == Some(&1), "HDF5 object header version is unsupported")?;
		let mut count = read16(header + 2).ok_or_else(truncated)?;
		let size = read32(header + 8).ok_or_else(truncated)?;
		let mut output = Vec::new();
		messages(bytes, &mut output, &mut count, header + 16, header + 16 + size)?;
		Ok(output)
	};
	let (mut btree, mut heap) = (None, None);
	for (kind, body, _) in object_messages(root_header)? {
		if kind == 0x11 {
			btree = read64(body);
			heap = read64(body + 8);
		}
	}
	let (btree, heap) = (btree.ok_or_else(|| RecipeError::new("HDF5 root group has no symbol table"))?, heap.ok_or_else(truncated)?);
	require(bytes.get(heap..heap + 4) == Some(&b"HEAP"[..]), "HDF5 local heap is invalid")?;
	let heap_data = read64(heap + 24).ok_or_else(truncated)?;
	let mut datasets = Vec::new();
	let mut group_nodes = vec![btree];
	while let Some(node) = group_nodes.pop() {
		require(bytes.get(node..node + 4) == Some(&b"TREE"[..]), "HDF5 group b-tree is invalid")?;
		let (level, entries) = (bytes[node + 5], read16(node + 6).ok_or_else(truncated)?);
		let mut offset = node + 24 + 8;
		for _ in 0..entries {
			let child = read64(offset).ok_or_else(truncated)?;
			offset += 16;
			if level > 0 {
				group_nodes.push(child);
				continue;
			}
			require(bytes.get(child..child + 4) == Some(&b"SNOD"[..]), "HDF5 symbol table node is invalid")?;
			let symbols = read16(child + 6).ok_or_else(truncated)?;
			for symbol in 0..symbols {
				let entry = child + 8 + symbol * 40;
				let name_offset = heap_data + read64(entry).ok_or_else(truncated)?;
				let terminator = bytes.get(name_offset..).and_then(|tail| tail.iter().position(|byte| *byte == 0)).ok_or_else(truncated)?;
				let dataset_name =
					String::from_utf8(bytes[name_offset..name_offset + terminator].to_vec()).map_err(|error| RecipeError::new(format!("HDF5 dataset name is not UTF-8: {error}")))?;
				datasets.push((dataset_name, read64(entry + 8).ok_or_else(truncated)?));
			}
		}
	}
	let mut columns = Vec::new();
	for (dataset, header) in datasets {
		let (mut dims, mut chunk_dims, mut address, mut contiguous_size, mut deflated, mut element, mut float, mut signed) = (Vec::new(), Vec::new(), None, 0, false, 0_usize, false, false);
		for (kind, body, size) in object_messages(header)? {
			match kind {
				1 => {
					let rank = bytes[body + 1] as usize;
					dims = (0..rank).map(|index| read64(body + 8 + 8 * index).ok_or_else(truncated)).collect::<Result<Vec<_>>>()?;
				}
				3 => {
					let class = bytes[body] & 0xf;
					require(bytes[body] >> 4 <= 1, "HDF5 datatype version is unsupported")?;
					require(class <= 1, format!("HDF5 datatype class {class} of dataset {dataset:?} is unsupported"))?;
					require(bytes[body + 1] & 1 == 0, format!("HDF5 dataset {dataset:?} is not little-endian"))?;
					float = class == 1;
					signed = class == 0 && bytes[body + 1] & 8 != 0;
					element = read32(body + 4).ok_or_else(truncated)?;
				}
				8 => {
					require(bytes[body] == 3, "HDF5 data layout version is unsupported")?;
					match bytes[body + 1] {
						1 => {
							address = read64(body + 2);
							contiguous_size = read64(body + 10).ok_or_else(truncated)?;
						}
						2 => {
							let rank = bytes[body + 2] as usize;
							address = read64(body + 3);
							chunk_dims = (0..rank.checked_sub(1).ok_or_else(truncated)?).map(|index| read32(body + 11 + 4 * index).ok_or_else(truncated)).collect::<Result<Vec<_>>>()?;
						}
						class => return Err(RecipeError::new(format!("HDF5 data layout class {class} is unsupported"))),
					}
				}
				11 => {
					deflated = bytes.get(body..body + size).ok_or_else(truncated)?.windows(7).any(|window| window == b"deflate");
					require(deflated, format!("HDF5 dataset {dataset:?} uses an unsupported filter"))?;
				}
				_ => {}
			}
		}
		require(!dims.is_empty() && element != 0, format!("HDF5 dataset {dataset:?} has no shape or type"))?;
		require(matches!((float, element), (false, 1 | 2 | 4 | 8) | (true, 4 | 8)), format!("HDF5 dataset {dataset:?} element size {element} is unsupported"))?;
		let count = dims.iter().try_fold(1_usize, |product, dimension| product.checked_mul(*dimension)).ok_or_else(|| RecipeError::new("HDF5 dataset size overflows"))?;
		let mut raw = vec![0_u8; checked_mul(count, element, "HDF5 dataset bytes")?];
		let address = address.ok_or_else(|| RecipeError::new(format!("HDF5 dataset {dataset:?} has no data address")))?;
		if chunk_dims.is_empty() {
			require(contiguous_size == raw.len(), format!("HDF5 dataset {dataset:?} has the wrong contiguous size"))?;
			raw.copy_from_slice(bytes.get(address..address + contiguous_size).ok_or_else(truncated)?);
		} else {
			require(chunk_dims.len() == dims.len(), format!("HDF5 dataset {dataset:?} chunk rank differs from its shape"))?;
			let mut chunk_nodes = vec![address];
			let key_length = 8 + 8 * (chunk_dims.len() + 1);
			while let Some(node) = chunk_nodes.pop() {
				require(bytes.get(node..node + 4) == Some(&b"TREE"[..]) && bytes.get(node + 4) == Some(&1), "HDF5 chunk b-tree is invalid")?;
				let (level, entries) = (bytes[node + 5], read16(node + 6).ok_or_else(truncated)?);
				let mut offset = node + 24;
				for _ in 0..entries {
					let compressed = read32(offset).ok_or_else(truncated)?;
					let mask = read32(offset + 4).ok_or_else(truncated)?;
					let starts = (0..chunk_dims.len()).map(|index| read64(offset + 8 + 8 * index).ok_or_else(truncated)).collect::<Result<Vec<_>>>()?;
					let child = read64(offset + key_length).ok_or_else(truncated)?;
					offset += key_length + 8;
					if level > 0 {
						chunk_nodes.push(child);
						continue;
					}
					require(mask == 0, format!("HDF5 dataset {dataset:?} chunk filter mask is unsupported"))?;
					let chunk = bytes.get(child..child + compressed).ok_or_else(truncated)?;
					let chunk = if deflated { zlib_inflate(chunk)? } else { chunk.to_vec() };
					let chunk_count = chunk_dims.iter().product::<usize>();
					require(chunk.len() == chunk_count * element, format!("HDF5 dataset {dataset:?} chunk has the wrong size"))?;
					for local in 0..chunk_count {
						let (mut remainder, mut inside) = (local, true);
						let mut coordinates = vec![0_usize; chunk_dims.len()];
						for axis in (0..chunk_dims.len()).rev() {
							coordinates[axis] = starts[axis] + remainder % chunk_dims[axis];
							remainder /= chunk_dims[axis];
							inside &= coordinates[axis] < dims[axis];
						}
						if !inside {
							continue;
						}
						let mut index = 0;
						for axis in 0..chunk_dims.len() {
							index = index * dims[axis] + coordinates[axis];
						}
						raw[index * element..(index + 1) * element].copy_from_slice(&chunk[local * element..(local + 1) * element]);
					}
				}
			}
		}
		let decode = |value: &[u8]| -> f64 {
			match (float, signed, element) {
				(true, _, 4) => f64::from(f32::from_le_bytes(value.try_into().unwrap())),
				(true, _, 8) => f64::from_le_bytes(value.try_into().unwrap()),
				(false, true, 1) => f64::from(value[0] as i8),
				(false, true, 2) => f64::from(i16::from_le_bytes(value.try_into().unwrap())),
				(false, true, 4) => f64::from(i32::from_le_bytes(value.try_into().unwrap())),
				(false, true, 8) => i64::from_le_bytes(value.try_into().unwrap()) as f64,
				(false, false, 1) => f64::from(value[0]),
				(false, false, 2) => f64::from(u16::from_le_bytes(value.try_into().unwrap())),
				(false, false, 4) => f64::from(u32::from_le_bytes(value.try_into().unwrap())),
				(false, false, 8) => u64::from_le_bytes(value.try_into().unwrap()) as f64,
				_ => unreachable!(),
			}
		};
		let decoded = raw.chunks_exact(element).map(decode).collect::<Vec<_>>();
		let rows = dims[0];
		let width = decoded.len() / rows.max(1);
		for column in 0..width {
			let header = if width == 1 { dataset.clone() } else { format!("{dataset}.{}", column + 1) };
			columns.push((header, rows, (0..rows).map(|row| decoded[row * width + column]).collect()));
		}
	}
	require(!columns.is_empty(), "HDF5 file has no datasets")?;
	Ok(columns)
}
/// The stored entries of a ZIP archive, resolved through the central directory.
fn zip_entries(bytes: &[u8]) -> Result<Vec<(String, Vec<u8>)>> {
	let read16 = |offset: usize| bytes.get(offset..offset + 2).map(|value| u16::from_le_bytes(value.try_into().unwrap()) as usize);
	let read32 = |offset: usize| bytes.get(offset..offset + 4).map(|value| u32::from_le_bytes(value.try_into().unwrap()) as usize);
	let tail = bytes.len().checked_sub(22).ok_or_else(|| RecipeError::new("ZIP archive is truncated"))?;
	let end = (0..=tail.min(65535))
		.map(|back| tail - back)
		.find(|&offset| bytes[offset..offset + 4] == [0x50, 0x4b, 0x05, 0x06])
		.ok_or_else(|| RecipeError::new("ZIP end of central directory is absent"))?;
	let count = read16(end + 10).ok_or_else(|| RecipeError::new("ZIP archive is truncated"))?;
	let mut offset = read32(end + 16).ok_or_else(|| RecipeError::new("ZIP archive is truncated"))?;
	let mut entries = Vec::new();
	for _ in 0..count {
		require(bytes.get(offset..offset + 4) == Some(&[0x50, 0x4b, 0x01, 0x02]), "ZIP central directory entry is invalid")?;
		let (method, size, name_length, extra, comment) = (read16(offset + 10), read32(offset + 24), read16(offset + 28), read16(offset + 30), read16(offset + 32));
		let (method, size) = (method.ok_or_else(|| RecipeError::new("ZIP archive is truncated"))?, size.ok_or_else(|| RecipeError::new("ZIP archive is truncated"))?);
		let local = read32(offset + 42).ok_or_else(|| RecipeError::new("ZIP archive is truncated"))?;
		let name = String::from_utf8(bytes.get(offset + 46..offset + 46 + name_length.unwrap_or(0)).ok_or_else(|| RecipeError::new("ZIP archive is truncated"))?.to_vec())
			.map_err(|error| RecipeError::new(format!("ZIP entry name is not UTF-8: {error}")))?;
		require(bytes.get(local..local + 4) == Some(&[0x50, 0x4b, 0x03, 0x04]), "ZIP local header is invalid")?;
		let (local_name, local_extra) = (read16(local + 26).unwrap_or(0), read16(local + 28).unwrap_or(0));
		let start = local + 30 + local_name + local_extra;
		require(method == 0, format!("ZIP entry {name:?} uses unsupported compression method {method}"))?;
		let contents = bytes.get(start..start + size).ok_or_else(|| RecipeError::new(format!("ZIP entry {name:?} is truncated")))?.to_vec();
		if !name.ends_with('/') {
			entries.push((name, contents));
		}
		offset += 46 + name_length.unwrap_or(0) + extra.unwrap_or(0) + comment.unwrap_or(0);
	}
	require(!entries.is_empty(), "ZIP archive has no entries")?;
	Ok(entries)
}
/// One named column group from an NPY array: trailing dimensions flatten to `name.1..name.k` columns.
fn npy_columns(name: &str, bytes: &[u8]) -> Result<Vec<(String, usize, Vec<f64>)>> {
	require(bytes.get(..6) == Some(b"\x93NUMPY"), "NPY magic is absent")?;
	let header_length = match bytes.get(6) {
		Some(1) => bytes.get(8..10).map(|value| u16::from_le_bytes(value.try_into().unwrap()) as usize + 10),
		Some(2 | 3) => bytes.get(8..12).map(|value| u32::from_le_bytes(value.try_into().unwrap()) as usize + 12),
		_ => None,
	}
	.ok_or_else(|| RecipeError::new("NPY header is invalid"))?;
	let header = str::from_utf8(bytes.get(if bytes[6] == 1 { 10 } else { 12 }..header_length).ok_or_else(|| RecipeError::new("NPY header is truncated"))?)
		.map_err(|error| RecipeError::new(format!("NPY header is not UTF-8: {error}")))?;
	let field = |key: &str| header.split(key).nth(1).and_then(|rest| rest.split(':').nth(1)).map(str::trim_start);
	let descr = field("'descr'").and_then(|value| value.split('\'').nth(1)).ok_or_else(|| RecipeError::new("NPY descr is absent"))?.to_owned();
	require(field("'fortran_order'").is_some_and(|value| value.starts_with("False")), "NPY fortran order is unsupported")?;
	let shape_text = field("'shape'").and_then(|value| value.split(')').next()).and_then(|value| value.split('(').nth(1)).ok_or_else(|| RecipeError::new("NPY shape is absent"))?;
	let shape = shape_text
		.split(',')
		.map(str::trim)
		.filter(|value| !value.is_empty())
		.map(|value| value.parse::<usize>().map_err(|error| RecipeError::new(format!("invalid NPY dimension: {error}"))))
		.collect::<Result<Vec<_>>>()?;
	let rows = shape.first().copied().unwrap_or(1);
	let width = shape.iter().skip(1).product::<usize>().max(1);
	let element = descr.as_bytes().last().and_then(|digit| char::from(*digit).to_digit(10)).ok_or_else(|| RecipeError::new(format!("NPY dtype {descr:?} is unsupported")))? as usize;
	let count = checked_mul(rows, width, "NPY elements")?;
	let values = bytes.get(header_length..header_length + count * element).ok_or_else(|| RecipeError::new("NPY data is truncated"))?;
	let kind = &descr[descr.len() - 2..];
	let decode = |value: &[u8]| -> Result<f64> {
		Ok(match (kind, element) {
			("f4", 4) => f64::from(f32::from_le_bytes(value.try_into().unwrap())),
			("f8", 8) => f64::from_le_bytes(value.try_into().unwrap()),
			("i1", 1) => f64::from(value[0] as i8),
			("i2", 2) => f64::from(i16::from_le_bytes(value.try_into().unwrap())),
			("i4", 4) => f64::from(i32::from_le_bytes(value.try_into().unwrap())),
			("i8", 8) => i64::from_le_bytes(value.try_into().unwrap()) as f64,
			("u1", 1) => f64::from(value[0]),
			("u2", 2) => f64::from(u16::from_le_bytes(value.try_into().unwrap())),
			("u4", 4) => f64::from(u32::from_le_bytes(value.try_into().unwrap())),
			("u8", 8) => u64::from_le_bytes(value.try_into().unwrap()) as f64,
			_ => return Err(RecipeError::new(format!("NPY dtype {descr:?} is unsupported"))),
		})
	};
	require(matches!(descr.as_bytes().first(), Some(b'<' | b'|')) || element == 1, format!("NPY dtype {descr:?} is unsupported"))?;
	let decoded = values.chunks_exact(element).map(decode).collect::<Result<Vec<_>>>()?;
	let mut columns = Vec::with_capacity(width);
	for column in 0..width {
		let header = if width == 1 { name.to_owned() } else { format!("{name}.{}", column + 1) };
		columns.push((header, rows, (0..rows).map(|row| decoded[row * width + column]).collect()));
	}
	Ok(columns)
}
/// One table from named numeric columns; every column must agree on the row count.
fn array_table(name: String, columns: Vec<(String, usize, Vec<f64>)>) -> Result<Table> {
	require(!columns.is_empty(), "array source has no columns")?;
	let rows = columns[0].1;
	for (header, count, _) in &columns {
		require(*count == rows, format!("array column {header:?} expected {rows} rows, received {count}"))?;
	}
	let headers = columns.iter().map(|(header, _, _)| header.clone()).collect();
	let table_rows = (0..rows).map(|row| columns.iter().map(|(_, _, values)| values[row].to_string()).collect()).collect();
	Ok(Table { name, headers, rows: table_rows, attention: None })
}
/// The records of a top-level JSON array.
fn json_array(text: &str) -> Result<Vec<JsonValue>> {
	let mut rest = text.trim_start().strip_prefix('[').ok_or_else(|| RecipeError::new("JSON records expect a top-level array"))?.trim_start();
	let mut values = Vec::new();
	loop {
		if let Some(after) = rest.strip_prefix(']') {
			require(after.trim().is_empty(), "JSON records have trailing content")?;
			return Ok(values);
		}
		if !values.is_empty() {
			rest = rest.strip_prefix(',').ok_or_else(|| RecipeError::new("JSON array expects a comma"))?.trim_start();
		}
		let (value, remaining) = json_value(rest)?;
		values.push(value);
		rest = remaining.trim_start();
	}
}
enum JsonValue {
	Null,
	Bool(bool),
	Number(String),
	Text(String),
	Array,
	Object(Vec<(String, JsonValue)>),
}
impl JsonValue {
	fn scalar(&self) -> Option<String> {
		match self {
			Self::Null => Some(String::new()),
			Self::Bool(value) => Some(value.to_string()),
			Self::Number(value) | Self::Text(value) => Some(value.clone()),
			Self::Array | Self::Object(_) => None,
		}
	}
}
/// Parse one JSON value from the start of `text`, returning it with the unconsumed remainder.
fn json_value(text: &str) -> Result<(JsonValue, &str)> {
	let text = text.trim_start();
	let mut characters = text.char_indices();
	match characters.next().map(|(_, character)| character) {
		Some('n') => Ok((JsonValue::Null, text.strip_prefix("null").ok_or_else(|| RecipeError::new("invalid JSON literal"))?)),
		Some('t') => Ok((JsonValue::Bool(true), text.strip_prefix("true").ok_or_else(|| RecipeError::new("invalid JSON literal"))?)),
		Some('f') => Ok((JsonValue::Bool(false), text.strip_prefix("false").ok_or_else(|| RecipeError::new("invalid JSON literal"))?)),
		Some('"') => {
			let (value, rest) = json_string(text)?;
			Ok((JsonValue::Text(value), rest))
		}
		Some('[') => {
			let mut rest = text[1..].trim_start();
			let mut values = 0;
			loop {
				if let Some(after) = rest.strip_prefix(']') {
					return Ok((JsonValue::Array, after));
				}
				if values != 0 {
					rest = rest.strip_prefix(',').ok_or_else(|| RecipeError::new("JSON array expects a comma"))?.trim_start();
				}
				let (_, remaining) = json_value(rest)?;
				values += 1;
				rest = remaining.trim_start();
			}
		}
		Some('{') => {
			let mut rest = text[1..].trim_start();
			let mut fields = Vec::new();
			loop {
				if let Some(after) = rest.strip_prefix('}') {
					return Ok((JsonValue::Object(fields), after));
				}
				if !fields.is_empty() {
					rest = rest.strip_prefix(',').ok_or_else(|| RecipeError::new("JSON object expects a comma"))?.trim_start();
				}
				let (key, remaining) = json_string(rest)?;
				rest = remaining.trim_start().strip_prefix(':').ok_or_else(|| RecipeError::new("JSON object expects a colon"))?;
				let (value, remaining) = json_value(rest)?;
				fields.push((key, value));
				rest = remaining.trim_start();
			}
		}
		Some(character) if character == '-' || character.is_ascii_digit() => {
			let end = text.find(|character: char| !matches!(character, '0'..='9' | '-' | '+' | '.' | 'e' | 'E')).unwrap_or(text.len());
			let (number, rest) = text.split_at(end);
			number.parse::<f64>().map_err(|error| RecipeError::new(format!("invalid JSON number {number:?}: {error}")))?;
			Ok((JsonValue::Number(number.to_owned()), rest))
		}
		_ => Err(RecipeError::new("invalid JSON value")),
	}
}
fn json_string(text: &str) -> Result<(String, &str)> {
	let mut characters = text.strip_prefix('"').ok_or_else(|| RecipeError::new("JSON expects a string"))?.char_indices();
	let mut value = String::new();
	while let Some((index, character)) = characters.next() {
		match character {
			'"' => return Ok((value, &text[index + 2..])),
			'\\' => match characters.next().map(|(_, escape)| escape) {
				Some('"') => value.push('"'),
				Some('\\') => value.push('\\'),
				Some('/') => value.push('/'),
				Some('b') => value.push('\u{8}'),
				Some('f') => value.push('\u{c}'),
				Some('n') => value.push('\n'),
				Some('r') => value.push('\r'),
				Some('t') => value.push('\t'),
				Some('u') => {
					let unit = |characters: &mut std::str::CharIndices| -> Result<u32> {
						let digits =
							(0..4).map(|_| characters.next().map(|(_, digit)| digit).ok_or_else(|| RecipeError::new("JSON unicode escape is truncated"))).collect::<Result<String>>()?;
						u32::from_str_radix(&digits, 16).map_err(|error| RecipeError::new(format!("invalid JSON unicode escape: {error}")))
					};
					let code = match unit(&mut characters)? {
						high @ 0xd800..=0xdbff => {
							require(
								characters.next().map(|(_, escape)| escape) == Some('\\') && characters.next().map(|(_, escape)| escape) == Some('u'),
								"JSON high surrogate expects a paired low surrogate",
							)?;
							let low = unit(&mut characters)?;
							require((0xdc00..=0xdfff).contains(&low), "JSON high surrogate expects a paired low surrogate")?;
							0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
						}
						low @ 0xdc00..=0xdfff => return Err(RecipeError::new(format!("JSON low surrogate {low:#x} has no preceding high surrogate"))),
						code => code,
					};
					value.push(char::from_u32(code).ok_or_else(|| RecipeError::new("invalid JSON unicode escape"))?);
				}
				_ => return Err(RecipeError::new("invalid JSON escape")),
			},
			character => value.push(character),
		}
	}
	Err(RecipeError::new("JSON string is unterminated"))
}
/// Flat records from an XML document: each first-level child of the root is one record and
/// each of its child elements is one text field, expressed as the shared record objects.
fn xml_records(text: &str) -> Result<Vec<JsonValue>> {
	fn tag(rest: &str) -> Result<(&str, &str)> {
		let rest = rest.strip_prefix('<').ok_or_else(|| RecipeError::new("XML expects an element"))?;
		let end = rest.find('>').ok_or_else(|| RecipeError::new("XML tag is unterminated"))?;
		let name = rest[..end].split_whitespace().next().unwrap_or("").trim_end_matches('/');
		require(!name.is_empty(), "XML tag has no name")?;
		Ok((name, &rest[end + 1..]))
	}
	fn unescape(value: &str) -> Result<String> {
		let mut output = String::with_capacity(value.len());
		let mut rest = value;
		while let Some(position) = rest.find('&') {
			output.push_str(&rest[..position]);
			let entity = rest[position + 1..].split(';').next().ok_or_else(|| RecipeError::new("XML entity is unterminated"))?;
			match entity {
				"amp" => output.push('&'),
				"lt" => output.push('<'),
				"gt" => output.push('>'),
				"quot" => output.push('"'),
				"apos" => output.push('\''),
				entity => {
					let code = entity
						.strip_prefix("#x")
						.map(|digits| u32::from_str_radix(digits, 16))
						.or_else(|| entity.strip_prefix('#').map(|digits| digits.parse()))
						.ok_or_else(|| RecipeError::new(format!("XML entity {entity:?} is unsupported")))?
						.map_err(|error| RecipeError::new(format!("invalid XML entity: {error}")))?;
					output.push(char::from_u32(code).ok_or_else(|| RecipeError::new("XML entity is out of range"))?);
				}
			}
			rest = &rest[position + entity.len() + 2..];
		}
		output.push_str(rest);
		Ok(output)
	}
	let mut rest = text.trim_start();
	if let Some(after) = rest.strip_prefix("<?") {
		rest = after.split_once("?>").ok_or_else(|| RecipeError::new("XML declaration is unterminated"))?.1.trim_start();
	}
	let (root, mut rest) = tag(rest)?;
	let mut records = Vec::new();
	loop {
		rest = rest.trim_start();
		if let Some(after) = rest.strip_prefix(&format!("</{root}>")) {
			require(after.trim().is_empty(), "XML document has trailing content")?;
			return Ok(records);
		}
		let (record, mut inner) = tag(rest)?;
		let mut fields = Vec::new();
		loop {
			inner = inner.trim_start();
			if let Some(after) = inner.strip_prefix(&format!("</{record}>")) {
				rest = after;
				break;
			}
			let (field, after) = tag(inner)?;
			let close = format!("</{field}>");
			let end = after.find(&close).ok_or_else(|| RecipeError::new(format!("XML field {field:?} is unterminated")))?;
			let value = &after[..end];
			require(!value.contains('<'), format!("XML field {field:?} nests elements"))?;
			fields.push((field.to_owned(), JsonValue::Text(unescape(value)?)));
			inner = &after[end + close.len()..];
		}
		records.push(JsonValue::Object(fields));
	}
}
/// One table from flat JSON records: the ordered union of keys becomes the header row.
fn json_records_table(name: String, records: &[JsonValue]) -> Result<Table> {
	let mut headers = Vec::<String>::new();
	for record in records {
		let JsonValue::Object(fields) = record else { return Err(RecipeError::new("JSON record is not an object")) };
		for (key, _) in fields {
			if !headers.contains(key) {
				headers.push(key.clone());
			}
		}
	}
	require(!headers.is_empty(), "JSON records have no fields")?;
	let mut rows = Vec::with_capacity(records.len());
	for record in records {
		let JsonValue::Object(fields) = record else { unreachable!() };
		let mut row = std::iter::repeat_with(String::new).take(headers.len()).collect::<Vec<_>>();
		for (key, value) in fields {
			let column = headers.iter().position(|header| header == key).unwrap();
			row[column] = value.scalar().ok_or_else(|| RecipeError::new(format!("JSON record field {key:?} is not a scalar")))?;
		}
		rows.push(row);
	}
	Ok(Table { name, headers, rows, attention: None })
}
fn parse_table(path: &Path, bytes: &[u8]) -> Result<(Table, usize)> {
	// The delimiter splits every record into the same number of fields. First-line frequency does not identify it: one incidental comma in a line of prose is not a second column.
	let (_, mut rows, blank) = [b'\t', b';', b','].into_iter().try_fold((0, Vec::new(), 0), |widest, delimiter| {
		let (rows, blank) = records(bytes, delimiter)?;
		let width = rows.first().map_or(0, Vec::len);
		let rectangle = if rows.iter().all(|row| row.len() == width) { width } else { 0 };
		Ok(if rectangle >= widest.0 { (rectangle, rows, blank) } else { widest })
	})?;
	require(!rows.is_empty(), format!("dataset {} is empty", path.display()))?;
	let first = rows.remove(0);
	let numeric = |value: &String| value.parse::<f64>().is_ok();
	let headerless = first.iter().all(numeric) || rows.is_empty();
	// A headerless table carries its label in the conventional final position, so that
	// column's authoritative name is "target" and the earlier columns take positional
	// names. Positional colN forms still match every column through column_match.
	let headers = if headerless { (1..=first.len()).map(|column| if column == first.len() { "target".to_owned() } else { format!("col{column}") }).collect() } else { first.clone() };
	if headerless {
		rows.insert(0, first);
	}
	let width = headers.len();
	let malformed = rows.iter().filter(|row| row.len() != width).count();
	require(malformed == 0, format!("dataset {} has {malformed} rows differing from the expected {width} fields", path.display()))?;
	let name = path.file_stem().and_then(|value| value.to_str()).unwrap_or("data").to_owned();
	Ok((Table { name, headers, rows, attention: None }, blank))
}
fn records(bytes: &[u8], delimiter: u8) -> Result<(Vec<Vec<String>>, usize)> {
	let (mut rows, mut row, mut field, mut quoted, mut blank) = (Vec::new(), Vec::new(), Vec::new(), false, 0);
	let mut index = 0;
	while index < bytes.len() {
		let byte = bytes[index];
		if byte == b'"' {
			if quoted && bytes.get(index + 1) == Some(&b'"') {
				field.push(byte);
				index += 1;
			} else {
				quoted = !quoted;
			}
		} else if byte == delimiter && !quoted {
			row.push(String::from_utf8(field).map_err(|_| RecipeError::new("feature is not UTF-8"))?);
			field = Vec::new();
		} else if byte == b'\n' && !quoted {
			let value = String::from_utf8(field).map_err(|_| RecipeError::new("feature is not UTF-8"))?;
			row.push(value.trim_end_matches('\r').to_owned());
			field = Vec::new();
			// One rule decides whether an assembled record carries data, so a record of blank fields is padding wherever it ends.
			if row.iter().any(|value| !value.trim().is_empty()) {
				rows.push(row)
			} else {
				blank += 1
			}
			row = Vec::new();
		} else {
			field.push(byte);
		}
		index += 1;
	}
	require(!quoted, "unterminated quoted feature")?;
	if !field.is_empty() || !row.is_empty() {
		row.push(String::from_utf8(field).map_err(|_| RecipeError::new("feature is not UTF-8"))?);
		if row.iter().any(|value| !value.trim().is_empty()) { rows.push(row) } else { blank += 1 }
	}
	Ok((rows, blank))
}
fn categories(table: &Table, column: usize, rows: usize) -> Vec<String> {
	table.rows.iter().take(rows).filter_map(|row| row.get(column)).filter(|value| !value.is_empty()).cloned().collect::<BTreeSet<_>>().into_iter().collect()
}
fn infer_feature(table: &Table, column: usize, rows: usize) -> FeatureType {
	let values = table.rows.iter().take(rows).filter_map(|row| row.get(column)).filter(|value| !value.is_empty()).collect::<Vec<_>>();
	if !values.is_empty() && values.iter().all(|value| value.parse::<f64>().is_ok()) {
		return FeatureType::Numeric;
	}
	let categories = categories(table, column, rows);
	let categorical_ratio = env!("RECIPE_CATEGORICAL_RATIO").parse::<f64>().expect("categorical ratio must be numeric");
	if categories.len() as f64 / values.len().max(1) as f64 <= categorical_ratio {
		FeatureType::Categorical(categories)
	} else {
		FeatureType::Text(values.iter().map(|value| value.len()).max().unwrap_or(0))
	}
}
impl FeatureType {
	fn width(&self) -> usize {
		match self {
			Self::Numeric => 1,
			Self::Categorical(values) => values.len(),
			Self::Text(width) => *width,
		}
	}
}
fn encode(value: &str, kind: &FeatureType, output: &mut Vec<f64>) -> bool {
	if value.is_empty() {
		output.resize(output.len() + kind.width(), f64::NAN);
		return true;
	}
	match kind {
		FeatureType::Numeric => value.parse::<f64>().is_ok_and(|value| {
			output.push(value);
			value.is_finite()
		}),
		FeatureType::Categorical(categories) => {
			let found = categories.iter().position(|category| category == value);
			output.extend((0..categories.len()).map(|index| f64::from(found == Some(index))));
			found.is_some()
		}
		FeatureType::Text(width) => {
			output.extend(value.bytes().map(f64::from).chain(std::iter::repeat(0.0)).take(*width));
			value.len() <= *width
		}
	}
}
fn shuffle(samples: &mut Vec<f64>, targets: &mut Vec<f64>, identities: &mut Vec<u64>, features: usize, source_rows: usize, target_width: usize) -> Result<()> {
	let mut seed = env!("RECIPE_RANDOM_SEED").parse::<u64>().map_err(|error| RecipeError::new(format!("invalid random seed: {error}")))?;
	let rows = targets.len() / target_width;
	let mut order = Vec::with_capacity(rows);
	for (start, end) in [(0, source_rows), (source_rows, rows)] {
		let mut partition = (start..end).collect::<Vec<_>>();
		for index in (1..partition.len()).rev() {
			seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
			partition.swap(index, (seed as usize) % (index + 1));
		}
		order.extend(partition);
	}
	let old_samples = std::mem::take(samples);
	let old_targets = std::mem::take(targets);
	let old_identities = std::mem::take(identities);
	for row in order {
		samples.extend_from_slice(&old_samples[row * features..(row + 1) * features]);
		targets.extend_from_slice(&old_targets[row * target_width..(row + 1) * target_width]);
		identities.push(old_identities[row]);
	}
	Ok(())
}
pub struct Train {
	epochs: usize,
	learning_rate: f64,
	log_metrics: Vec<Metric>,
	stop: Option<f64>,
	resume: Option<PathBuf>,
	save: Option<PathBuf>,
	seed: Option<usize>,
	precision: Compute,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Compute {
	F(FloatFormat),
	Fp(FloatFormat),
	Int(IntFormat),
	Bf(FloatFormat),
	Tf(FloatFormat),
}
impl Compute {
	const FP8: Self = Self::Fp(FloatFormat::FP8);
	const FP16: Self = Self::Fp(FloatFormat::FP16);
	const FP32: Self = Self::Fp(FloatFormat::FP32);
	const FP64: Self = Self::Fp(FloatFormat::FP64);
	const INT1: Self = Self::Int(IntFormat::INT1);
	const INT4: Self = Self::Int(IntFormat::INT4);
	const INT8: Self = Self::Int(IntFormat::INT8);
	const BF16: Self = Self::Bf(FloatFormat::BF16);
	const TF32: Self = Self::Tf(FloatFormat::TF32);
	const fn bytes(self) -> usize {
		match self {
			Self::F(_) => FloatFormat::FP64.bytes(),
			Self::Fp(value) | Self::Bf(value) | Self::Tf(value) => value.bytes(),
			Self::Int(value) => value.bytes(),
		}
	}
	fn pack(self, value: f64) -> u64 {
		match self {
			Self::F(format) | Self::Fp(format) | Self::Bf(format) | Self::Tf(format) => format.pack(value),
			Self::Int(format) => format.pack(value),
		}
	}
	fn unpack(self, bits: u64) -> f64 {
		match self {
			Self::F(format) | Self::Fp(format) | Self::Bf(format) | Self::Tf(format) => format.unpack(bits),
			Self::Int(format) => format.unpack(bits),
		}
	}
	fn optimizer_epsilon(self, value: f64) -> f64 {
		let rounded = self.unpack(self.pack(value));
		match self {
			Self::F(format) | Self::Fp(format) | Self::Bf(format) | Self::Tf(format) if rounded == 0.0 => format.arithmetic.unpack(1u64 << format.arithmetic.man),
			_ => rounded,
		}
	}
	fn below_one(self, value: f64) -> f64 {
		let rounded = self.unpack(self.pack(value));
		match self {
			Self::F(format) | Self::Fp(format) | Self::Bf(format) | Self::Tf(format) if rounded >= 1.0 => format.arithmetic.unpack(format.arithmetic.pack(1.0) - 1),
			_ => rounded,
		}
	}
	fn saved(family: &str, values: [u8; 4]) -> Option<Self> {
		let [bits, exp, man, storage_man] = values;
		match family {
			"f" if bits == FloatFormat::FP64.storage.bits() && storage_man == FloatFormat::FP64.storage.man && exp != 0 && man != 0 && u16::from(exp) + u16::from(man) < 64 => {
				Some(Self::F(FloatFormat::computed(exp, man)))
			}
			"fp" => [Self::FP8, Self::FP16, Self::FP32, Self::FP64].into_iter().find(|format| format.saved_fields().1 == values),
			"int" => [Self::INT1, Self::INT4, Self::INT8].into_iter().find(|format| format.saved_fields().1 == values),
			"bf" if values == Self::BF16.saved_fields().1 => Some(Self::BF16),
			"tf" if values == Self::TF32.saved_fields().1 => Some(Self::TF32),
			_ => None,
		}
	}
	fn saved_fields(self) -> (&'static str, [u8; 4]) {
		match self {
			Self::F(value) => ("f", [value.storage.bits(), value.arithmetic.exp, value.arithmetic.man, value.storage.man]),
			Self::Fp(value) => ("fp", [value.storage.bits(), value.arithmetic.exp, value.arithmetic.man, value.storage.man]),
			Self::Int(value) => ("int", [value.bits, 0, 0, 0]),
			Self::Bf(value) => ("bf", [value.storage.bits(), value.arithmetic.exp, value.arithmetic.man, value.storage.man]),
			Self::Tf(value) => ("tf", [value.storage.bits(), value.arithmetic.exp, value.arithmetic.man, value.storage.man]),
		}
	}
	fn label(self) -> String {
		match self {
			Self::F(value) => format!("f({},{})", value.arithmetic.exp, value.arithmetic.man),
			Self::Fp(value) => format!("fp{}", value.storage.bits()),
			Self::Int(value) => format!("int{}", value.bits),
			Self::Bf(value) => format!("bf{}", value.storage.bits()),
			Self::Tf(value) => format!("tf{}", value.storage.bits()),
		}
	}
}
impl Train {
	fn arithmetic(mut self, format: Compute) -> Self {
		self.precision = format;
		self
	}
	pub fn f(self, exp: u8, man: u8) -> Self {
		assert!(exp != 0 && man != 0 && u16::from(exp) + u16::from(man) < 64, "f requires a representation no wider than 64 bits");
		self.arithmetic(Compute::F(FloatFormat::computed(exp, man)))
	}
	pub fn fp(self, bits: u8) -> Self {
		let format = match bits {
			8 => Compute::FP8,
			16 => Compute::FP16,
			32 => Compute::FP32,
			64 => Compute::FP64,
			_ => panic!("fp bits must be 8, 16, 32, or 64"),
		};
		self.arithmetic(format)
	}
	pub fn int(self, bits: u8) -> Self {
		let format = match bits {
			1 => Compute::INT1,
			4 => Compute::INT4,
			8 => Compute::INT8,
			_ => panic!("int bits must be 1, 4, or 8"),
		};
		self.arithmetic(format)
	}
	pub fn bf(self, bits: u8) -> Self {
		assert_eq!(bits, 16, "bf bits must be 16");
		self.arithmetic(Compute::BF16)
	}
	pub fn tf(self, bits: u8) -> Self {
		assert_eq!(bits, 32, "tf bits must be 32");
		self.arithmetic(Compute::TF32)
	}
	pub const fn seed(mut self, value: usize) -> Self {
		self.seed = Some(value);
		self
	}
	pub const fn stop(mut self, value: f64) -> Self {
		self.stop = if value == 0.0 { None } else { Some(value) };
		self
	}
	pub const fn optimizer(self, _: Adamw) -> Self {
		self
	}
	pub const fn epochs(mut self, value: usize) -> Self {
		self.epochs = value;
		self
	}
	pub const fn lr(mut self, value: f64) -> Self {
		self.learning_rate = value;
		self
	}
	pub fn log(mut self, metrics: impl IntoMetrics) -> Self {
		self.log_metrics = metrics.into_metrics();
		self
	}
	// Save and resume use the same file.
	pub fn save(mut self, path: impl AsRef<Path>) -> Self {
		self.save = Some(resolve_path(path).unwrap_or_else(|error| panic!("{error}")));
		self
	}
	pub fn resume(mut self, path: impl AsRef<Path>) -> Self {
		self.resume = Some(resolve_path(path).unwrap_or_else(|error| panic!("{error}")));
		self
	}
	fn execute(&self, model: &Model, data: &Data, evaluation: bool) -> TrainingReport {
		SIGNAL.get_or_init(|| unsafe { signal(SIGINT, interrupt) });
		INTERRUPT_CHECKPOINTED.store(false, Ordering::Release);
		if INTERRUPTED.load(Ordering::Acquire) {
			std::process::exit(INTERRUPTED_EXIT);
		}
		self.try_run(model, data, evaluation).unwrap_or_else(|error| {
			if INTERRUPTED.load(Ordering::Acquire) {
				std::process::exit(INTERRUPTED_EXIT)
			}
			panic!("{error}")
		})
	}
	pub fn run(&self, model: &Model, data: &Data) -> TrainingReport {
		let evaluation = data.split < 1.0 || !data.tests.is_empty();
		let report = self.execute(model, data, evaluation);
		if evaluation {
			self.print_evaluation(model, &report)
		}
		report
	}
	fn try_run(&self, model: &Model, data: &Data, evaluation: bool) -> Result<TrainingReport> {
		let started = Instant::now();
		let prepared = prepare(data)?;
		let training_rows = ((prepared.source_rows as f64) * data.split).floor() as usize;
		require(training_rows != 0 && training_rows <= prepared.source_rows, "split must select training rows")?;
		let (gpus, mut config) = (selected_gpus()?, Config::load()?);
		let gpu = gpus[0];
		let precision = self.precision;
		config.precision = precision;
		if let Some(seed) = self.seed {
			config.random_seed = seed;
		}
		let probability = model.loss.0 >= 4;
		let training_values = training_rows * prepared.target_width;
		let scale = probability.then(|| TargetScale::fit(&prepared.targets[..training_values]));
		let target_values = prepared.targets.iter().map(|target| scale.map_or(*target, |scale| scale.encode(*target))).collect::<Vec<_>>();
		let (run, mut graph) = (RUN.fetch_add(1, Ordering::Relaxed) + 1, compile(model, prepared, &target_values, training_rows, gpu, config, true)?);
		graph.state.training_rows = training_rows;
		if let Some(scale) = scale
			&& let Some(offset) = output_bias_offset(&graph)
		{
			for channel in 0..prepared.target_width {
				let mean = target_values[..training_values].iter().skip(channel).step_by(prepared.target_width).sum::<f64>() / training_rows as f64;
				graph.parameters[offset + channel] = scale.logit(mean);
			}
		}
		graph.refresh_storage(config)?;
		let mut stored = stored_graph(&graph, model, data, scale, precision, native_target_label(&gpu.native_target));
		require(stored.graph.output.elements() == prepared.target_width, format!("model output width must be {}", prepared.target_width))?;
		if let Some(path) = &self.resume {
			bundle::restore(path, &prepared.schema, std::slice::from_mut(&mut stored), &prepared.identities)?;
		}
		stored.graph.state.trained_samples.extend_from_slice(&prepared.identities[..training_rows]);
		stored.graph.state.trained_samples.sort_unstable();
		stored.graph.state.trained_samples.dedup();
		let (samples, targets) = (&prepared.samples[..training_rows * prepared.features], &target_values[..training_values]);
		let mut tape = DeviceTape::new(&stored.graph, samples, targets, gpus, config.precision, model.loss, config)?;
		self.finish_dispatch(
			if stored.bn_stats.is_empty() { tape.forward() } else { tape.inject_bn_stats(&stored.bn_stats).and_then(|_| tape.forward()) },
			&mut stored,
			&prepared.schema,
			&tape,
			None,
		)?;
		tape.print_devices(&stored.graph)?;
		stored.bn_stats = tape.extract_bn_stats()?;
		let initial_predictions = tape.predictions()?;
		let initial_loss = model_loss(&initial_predictions, targets, model.loss, config.activation[7]);
		let tolerance = self.stop.unwrap_or(0.0);
		let report_r2 = self.log_metrics.iter().any(|metric| metric.0 == R2.0);
		let mut epoch_seconds = 0.0;
		require(tolerance.is_finite() && (0.0..=1.0).contains(&tolerance), "stop must be between zero and one")?;
		for _ in 0..self.epochs {
			if INTERRUPTED.load(Ordering::Acquire) {
				self.finish_dispatch::<()>(Err(RecipeError::new("interrupted")), &mut stored, &prepared.schema, &tape, None).ok();
				break;
			}
			tape.advance()?;
			let epoch = tape.step() as usize;
			// Read once per epoch from the dispatched schedule, so a schedule change appears on the next line.
			let schedule = tape.schedule();
			let ((loss, checkpoint, predictions), seconds, live) = self.live_epoch(model, run, epoch, self.epochs, config, &schedule, || {
				let dispatched = tape.epoch(self.learning_rate, tolerance, config);
				let ((loss, checkpoint_requested), checkpoint) = self.finish_dispatch(dispatched, &mut stored, &prepared.schema, &tape, None)?;
				if checkpoint_requested {
					stored.bn_stats = tape.extract_bn_stats()?
				}
				let (_, persisted) = self.finish_dispatch(Ok(()), &mut stored, &prepared.schema, &tape, checkpoint_requested.then_some(()))?;
				let checkpoint = checkpoint.or(persisted);
				let predictions = if report_r2 { tape.predictions()? } else { Vec::new() };
				let (_, persisted) = self.finish_dispatch(Ok(()), &mut stored, &prepared.schema, &tape, None)?;
				Ok((loss, checkpoint.or(persisted), predictions))
			})?;
			epoch_seconds += seconds;
			self.print(model, run, epoch, self.epochs, loss, targets, &predictions, seconds, checkpoint, live, &schedule)?;
			if INTERRUPTED.load(Ordering::Acquire) {
				std::process::exit(INTERRUPTED_EXIT)
			}
		}
		stored.bn_stats = tape.extract_bn_stats()?;
		tape.inject_bn_stats(&stored.bn_stats)?;
		self.finish_dispatch(tape.forward(), &mut stored, &prepared.schema, &tape, None)?;
		let raw_predictions = tape.predictions()?;
		let mut final_loss = model_loss(&raw_predictions, targets, model.loss, config.activation[7]);
		let mut predictions = raw_predictions.iter().map(|value| scale.map_or(*value, |scale| scale.decode(*value))).collect::<Vec<_>>();
		let mut evaluated = Vec::new();
		if evaluation && data.autoregressive {
			let mut graph = stored.graph.clone();
			graph.parameters = tape.weights()?;
			predictions.clear();
			let mut raw_outputs = Vec::new();
			let stream = self.log_metrics.iter().any(|metric| metric.0 == tok.0);
			for sample in prepared.samples.chunks_exact(prepared.features) {
				let validation = NativeTape::new(&graph, sample, sample, &[], gpu, config.precision, None)?;
				validation.inject_bn_stats(&stored.bn_stats)?;
				validation.forward()?;
				let raw = validation.predictions()?;
				require(raw.len() == 1, "autoregressive forward must produce one char ID")?;
				raw_outputs.push(raw[0]);
				let prediction = scale.map_or(raw[0], |scale| scale.decode(raw[0]));
				predictions.push(prediction);
				if stream {
					eprint!("{}", predicted_char(prediction)?);
					std::io::Write::flush(&mut std::io::stderr()).map_err(|error| RecipeError::new(format!("cannot flush token stream: {error}")))?
				}
			}
			if stream {
				eprintln!()
			}
			// Evaluation loss lives in the training representation and covers only held-out rows;
			// decoding is for the user-facing predictions, r2, and tokens.
			final_loss = model_loss(&raw_outputs[training_rows..], &target_values[training_rows..], model.loss, config.activation[7]);
		} else if training_rows < prepared.rows {
			let mut graph = stored.graph.clone();
			graph.parameters = tape.weights()?;
			let (start, validation_targets) = (training_rows * prepared.features, &target_values[training_values..]);
			let validation = NativeTape::new(&graph, &prepared.samples[start..], &prepared.samples[start..], validation_targets, gpu, config.precision, None)?;
			validation.inject_bn_stats(&stored.bn_stats)?;
			validation.forward()?;
			let raw = validation.predictions()?;
			final_loss = model_loss(&raw, validation_targets, model.loss, config.activation[7]);
			evaluated = raw.into_iter().map(|value| scale.map_or(value, |scale| scale.decode(value))).collect();
		}
		let r2 = if training_rows == prepared.rows {
			coefficient(&prepared.targets, &predictions)
		} else if evaluation && data.autoregressive {
			coefficient(&prepared.targets[training_rows..], &predictions[training_rows..])
		} else {
			coefficient(&prepared.targets[training_values..], &evaluated)
		};
		if !evaluated.is_empty() {
			predictions = evaluated
		}
		self.finish_dispatch(Ok(()), &mut stored, &prepared.schema, &tape, Some(()))?;
		Ok(TrainingReport {
			initial_loss,
			final_loss,
			initial_predictions,
			predictions,
			r2,
			tile: tape.tile(),
			schedule: tape.schedule(),
			run,
			epoch: tape.step() as usize,
			seconds: started.elapsed().as_secs_f64(),
			epoch_seconds,
		})
	}
	fn finish_dispatch<T>(&self, result: Result<T>, stored: &mut bundle::StoredGraph, schema: &DataSchema, tape: &DeviceTape, request: Option<()>) -> Result<(T, Option<CheckpointStatus>)> {
		let request = if INTERRUPTED.load(Ordering::Acquire) && !INTERRUPT_CHECKPOINTED.swap(true, Ordering::AcqRel) { Some(()) } else { request.filter(|_| !INTERRUPTED.load(Ordering::Acquire)) };
		let checkpoint = if request.is_some()
			&& let Some(path) = &self.save
		{
			Some(checkpoint(path, schema, stored, tape)?)
		} else {
			None
		};
		result.map(|value| (value, checkpoint))
	}
	fn print(
		&self, model: &Model, run: u64, epoch: usize, epochs: usize, loss: f64, targets: &[f64], predictions: &[f64], seconds: f64, checkpoint: Option<CheckpointStatus>, live: bool,
		schedule: &str,
	) -> Result<()> {
		if self.log_metrics.is_empty() {
			return Ok(());
		}
		let r2 = self.log_metrics.iter().any(|metric| metric.0 == R2.0).then(|| coefficient(targets, predictions));
		Self::write_progress(
			&Self::metric_line(
				model.loss.name(),
				&model.description(&self.log_metrics),
				&self.log_metrics,
				epochs,
				schedule,
				Metrics { run, epoch, loss: Some(loss), r2, seconds, checkpoint, evaluation: false },
			),
			live,
			true,
		)
	}
	fn print_evaluation(&self, model: &Model, report: &TrainingReport) {
		let defaults = [Loss, R2];
		let metrics = if self.log_metrics.is_empty() { &defaults[..] } else { &self.log_metrics };
		Self::write_progress(
			&Self::metric_line(
				model.loss.name(),
				&model.description(metrics),
				metrics,
				self.epochs,
				&report.schedule,
				Metrics { run: report.run, epoch: report.epoch, loss: Some(report.final_loss), r2: Some(report.r2), seconds: report.seconds, checkpoint: None, evaluation: true },
			),
			false,
			true,
		)
		.unwrap_or_else(|error| panic!("{error}"))
	}
	fn metric_line(loss: &str, topology: &str, metrics: &[Metric], epochs: usize, schedule: &str, measurement: Metrics) -> String {
		let time = measurement.seconds * 1000.0;
		let mut values = Vec::new();
		let mut topology_printed = false;
		for metric in metrics {
			let value = match metric.0 {
				0 => format!("{} \x1b[38\x3b2\x3b242\x3b40\x3b60m{}\x1b[0m", if measurement.evaluation { "eval" } else { "run" }, measurement.run),
				1 => format!("{loss} \x1b[38\x3b2\x3b0\x3b174\x3b107m{}\x1b[0m", measurement.loss.map_or_else(|| format!("{:>6}", "..."), |value| format!("{value:.4}"))),
				2 => format!("r2 \x1b[38\x3b2\x3b39\x3b125\x3b255m{}\x1b[0m", measurement.r2.map_or_else(|| format!("{:>7}", "..."), |value| format!("{value:>7.4}"))),
				3 => {
					if measurement.seconds < 1.0 {
						format!("time \x1b[38\x3b2\x3b255\x3b194\x3b0m{time:>7.3} ms\x1b[0m")
					} else {
						format!("time \x1b[38\x3b2\x3b255\x3b194\x3b0m{:>8.4} s\x1b[0m", measurement.seconds)
					}
				}
				4 => format!("epoch \x1b[38\x3b2\x3b135\x3b90\x3b251m{:>width$}\x1b[0m", measurement.epoch, width = epochs.max(1).ilog10() as usize + 1),
				5..=7 | 9 if !topology_printed && !topology.is_empty() => {
					topology_printed = true;
					topology.to_owned()
				}
				5..=7 | 9 => continue,
				8 => continue,
				10 if schedule.is_empty() => continue,
				10 => format!("tile \x1b[38\x3b2\x3b135\x3b90\x3b251m{schedule}\x1b[0m"),
				_ => unreachable!(),
			};
			values.push(value);
		}
		if let Some(checkpoint) = measurement.checkpoint {
			values.push(match checkpoint {
				CheckpointStatus::Saved => "\x1b[1\x3b32m← checkpoint\x1b[0m",
				CheckpointStatus::Kept => "kept",
			}
			.to_owned())
		}
		values.join("  ")
	}
	fn write_progress(line: &str, replace: bool, complete: bool) -> Result<()> {
		if line.is_empty() && !replace {
			return Ok(());
		}
		let mut frame = if line.is_empty() {
			"\r\x1b[2K\x1b[?7h".to_owned()
		} else if replace {
			format!("\r\x1b[2K{}", if complete { "\x1b[?7h" } else { "" })
		} else if complete {
			String::new()
		} else {
			"\x1b[?7l\r\x1b[2K".to_owned()
		};
		frame.push_str(line);
		if complete {
			frame.push('\n')
		}
		let mut output = std::io::stderr().lock();
		output.write_all(frame.as_bytes()).and_then(|_| output.flush()).map_err(|error| RecipeError::new(format!("cannot write epoch progress: {error}")))
	}
	fn live_epoch<T>(&self, model: &Model, run: u64, epoch: usize, epochs: usize, config: Config, schedule: &str, action: impl FnOnce() -> Result<T>) -> Result<(T, f64, bool)> {
		let started = Instant::now();
		let partial = Metrics { run, epoch, loss: None, r2: None, seconds: 0.0, checkpoint: None, evaluation: false };
		let line = Self::metric_line(model.loss.name(), &model.description(&self.log_metrics), &self.log_metrics, epochs, schedule, partial);
		let live = !line.is_empty() && std::io::stderr().is_terminal();
		if !live {
			return action().map(|value| (value, started.elapsed().as_secs_f64(), false));
		}
		Self::write_progress(&line, false, false)?;
		let (stop, wait) = std::sync::mpsc::channel();
		let (metrics, loss, topology, schedule) = (self.log_metrics.clone(), model.loss.name(), model.description(&self.log_metrics), schedule.to_owned());
		let updates = std::thread::spawn(move || -> Result<bool> {
			let mut row = false;
			loop {
				match wait.recv_timeout(Duration::from_secs(1).div_f64(config.progress_refresh_hz as f64)) {
					Ok(()) | Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
						if INTERRUPTED.load(Ordering::Acquire) && !row {
							Self::write_progress(
								&Self::metric_line(loss, &topology, &metrics, epochs, &schedule, Metrics { seconds: started.elapsed().as_secs_f64(), ..partial }),
								false,
								false,
							)?
						};
						return Ok(row || INTERRUPTED.load(Ordering::Acquire));
					}
					Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
						let interrupted = INTERRUPTED.load(Ordering::Acquire);
						Self::write_progress(
							&Self::metric_line(loss, &topology, &metrics, epochs, &schedule, Metrics { seconds: started.elapsed().as_secs_f64(), ..partial }),
							row || !interrupted,
							false,
						)?;
						row |= interrupted
					}
				}
			}
		});
		let result = action();
		let _ = stop.send(());
		updates.join().map_err(|_| RecipeError::new("epoch progress panicked"))??;
		let value = match result {
			Ok(value) => value,
			Err(error) => {
				let _ = Self::write_progress("", true, false);
				return Err(error);
			}
		};
		Ok((value, started.elapsed().as_secs_f64(), true))
	}
}
#[derive(Clone, Copy)]
struct Metrics {
	run: u64,
	epoch: usize,
	loss: Option<f64>,
	r2: Option<f64>,
	seconds: f64,
	checkpoint: Option<CheckpointStatus>,
	evaluation: bool,
}
pub struct TrainingReport {
	initial_loss: f64,
	final_loss: f64,
	initial_predictions: Vec<f64>,
	predictions: Vec<f64>,
	r2: f64,
	tile: Tile,
	schedule: String,
	run: u64,
	epoch: usize,
	seconds: f64,
	epoch_seconds: f64,
}
impl TrainingReport {
	pub const fn initial_loss(&self) -> f64 {
		self.initial_loss
	}
	pub const fn final_loss(&self) -> f64 {
		self.final_loss
	}
	pub fn initial_predictions(&self) -> &[f64] {
		&self.initial_predictions
	}
	pub fn predictions(&self) -> &[f64] {
		&self.predictions
	}
	pub const fn r2(&self) -> f64 {
		self.r2
	}
	pub const fn tile(&self) -> [u32; 3] {
		[self.tile.m, self.tile.n, self.tile.k]
	}
	pub const fn epoch_seconds(&self) -> f64 {
		self.epoch_seconds
	}
}
#[derive(Clone, Copy)]
struct TargetScale {
	minimum: f64,
	span: f64,
}
impl TargetScale {
	fn fit(targets: &[f64]) -> Self {
		let minimum = targets.iter().copied().fold(f64::INFINITY, f64::min);
		let maximum = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
		// A constant target spans nothing; encode it as the minimum of a unit span so scaling stays finite.
		Self { minimum, span: if maximum == minimum { 1.0 } else { maximum - minimum } }
	}
	fn encode(self, value: f64) -> f64 {
		(value - self.minimum) / self.span
	}
	fn decode(self, value: f64) -> f64 {
		self.minimum + self.span * logistic(value)
	}
	fn logit(self, value: f64) -> f64 {
		let value = value.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
		(value / (1.0 - value)).ln()
	}
}
fn model_loss(predictions: &[f64], targets: &[f64], loss: LossFunction, threshold: f64) -> f64 {
	let values = predictions.iter().zip(targets);
	let mut result = values.map(|(prediction, target)| loss.value(*prediction, *target, threshold)).sum::<f64>() / targets.len() as f64;
	if loss.0 == 1 {
		result = result.sqrt();
	}
	result
}
fn predicted_char(prediction: f64) -> Result<char> {
	require(prediction.is_finite(), "autoregressive forward produced a nonfinite char ID")?;
	let id = prediction.round().clamp(0.0, (CHAR_IDS.len() - 1) as f64) as usize;
	Ok(CHAR_IDS[id])
}
fn coefficient(targets: &[f64], predictions: &[f64]) -> f64 {
	let mean = targets.iter().sum::<f64>() / targets.len() as f64;
	let residual = targets.iter().zip(predictions).map(|(target, value)| (target - value).powi(2)).sum::<f64>();
	let total = targets.iter().map(|target| (target - mean).powi(2)).sum::<f64>();
	if total == 0.0 { 0.0 } else { 1.0 - residual / total }
}
