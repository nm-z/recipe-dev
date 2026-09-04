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
		pub mode: NormalizeMode,
		pub prefix: &'a str,
	}

	pub struct NormalizeFragment {
		pub code: String,
		pub value: String,
	}

	/// Emit one normalized element from the fixed statistics arena. The arena is
	/// laid out as `mean[group]` followed by `scale[group]`, with `groups` fixed by
	/// the normalization mode. A training caller must run a separate fixed stats
	/// pass before this fragment; evaluation and inference reuse the stored arena.
	pub fn emit_normalize(context: NormalizeContext<'_>, element: &str) -> NormalizeFragment {
		let mut output = String::new();
		let elements = context.channels * context.length;
		let length = context.length;
		let local = format!("%{}.normalize.local", context.prefix);
		let row = format!("%{}.normalize.row", context.prefix);
		let position = format!("%{}.normalize.position", context.prefix);
		let group = format!("%{}.normalize.group", context.prefix);
		let groups = format!("%{}.normalize.groups", context.prefix);
		let scale_index = format!("%{}.normalize.scale.index", context.prefix);
		let mean_pointer = format!("%{}.normalize.mean.ptr", context.prefix);
		let scale_pointer = format!("%{}.normalize.scale.ptr", context.prefix);
		let mean = format!("%{}.normalize.mean", context.prefix);
		let scale = format!("%{}.normalize.scale", context.prefix);
		let centered = format!("%{}.normalize.centered", context.prefix);
		let value = format!("%{}.normalize.value", context.prefix);
		let _ = writeln!(output, "{row} = udiv i32 {element}, {elements}");
		let _ = writeln!(output, "{local} = urem i32 {element}, {elements}");
		let _ = writeln!(output, "{position} = urem i32 {local}, {length}");
		match context.mode {
			NormalizeMode::Batch | NormalizeMode::Evaluation => {
				let channel = format!("%{}.normalize.channel", context.prefix);
				let _ = writeln!(output, "{channel} = udiv i32 {local}, {length}");
				let _ = writeln!(output, "{group} = add i32 {channel}, 0");
				let _ = writeln!(output, "{groups} = add i32 0, {channels}", channels = context.channels);
			}
			NormalizeMode::Layer | NormalizeMode::Rms => {
				let row_base = format!("%{}.normalize.layer.row.base", context.prefix);
				let _ = writeln!(output, "{row_base} = mul i32 {row}, {length}");
				let _ = writeln!(output, "{group} = add i32 {row_base}, {position}");
				let _ = writeln!(output, "{groups} = mul i32 {rows}, {length}", rows = context.rows);
			}
		}
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
		NormalizeFragment { code: output, value }
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
		pub mode: NormalizeMode,
		pub prefix: &'a str,
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
		match context.mode {
			NormalizeMode::Batch => {
				let _ = writeln!(code, "{groups} = add i32 0, {}", context.channels);
				let _ = writeln!(code, "{items} = mul i32 {}, {}", context.rows, context.length);
			}
			NormalizeMode::Layer | NormalizeMode::Rms => {
				let _ = writeln!(code, "{groups} = mul i32 {}, {}", context.rows, context.length);
				let _ = writeln!(code, "{items} = add i32 0, {}", context.channels);
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
			NormalizeMode::Layer | NormalizeMode::Rms => {
				let _ = writeln!(code, "%{prefix}.row = udiv i32 {group}, {}", context.length);
				let _ = writeln!(code, "%{prefix}.position = urem i32 {group}, {}", context.length);
				let _ = writeln!(code, "%{prefix}.row.base = mul i32 %{prefix}.row, {elements}");
				let _ = writeln!(code, "%{prefix}.channel.base = mul i32 %{prefix}.p, {}", context.length);
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
		let _ = writeln!(code, "%{prefix}.delta = call {state} @recipe.state.from.model({ty} %{prefix}.delta.model)", state = context.state_type, ty = context.value_type);
		let _ = writeln!(code, "%{prefix}.output = call {state} @recipe.state.from.model({ty} %{prefix}.output.model)", state = context.state_type, ty = context.value_type);
		if context.mode == NormalizeMode::Rms {
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
		let _ = writeln!(code, "%{prefix}.projected.mean = call {ty} @recipe.state.div({ty} %{prefix}.projected, {ty} %{prefix}.items.value)", ty = context.state_type);
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
		let elements = context.channels * context.length;
		let length = context.length;
		let row = format!("%{}.normalize.reverse.row", context.prefix);
		let local = format!("%{}.normalize.reverse.local", context.prefix);
		let position = format!("%{}.normalize.reverse.position", context.prefix);
		let group = format!("%{}.normalize.reverse.group", context.prefix);
		let groups = format!("%{}.normalize.reverse.groups", context.prefix);
		let scale_index = format!("%{}.normalize.reverse.scale.index", context.prefix);
		let sum_base = format!("%{}.normalize.reverse.sum.base", context.prefix);
		let projected_base = format!("%{}.normalize.reverse.projected.base", context.prefix);
		let sum_index = format!("%{}.normalize.reverse.sum.index", context.prefix);
		let projected_index = format!("%{}.normalize.reverse.projected.index", context.prefix);
		let scale_pointer = format!("%{}.normalize.reverse.scale.ptr", context.prefix);
		let sum_pointer = format!("%{}.normalize.reverse.sum.ptr", context.prefix);
		let projected_pointer = format!("%{}.normalize.reverse.projected.ptr", context.prefix);
		let scale = format!("%{}.normalize.reverse.scale", context.prefix);
		let sum = format!("%{}.normalize.reverse.sum", context.prefix);
		let projected = format!("%{}.normalize.reverse.projected", context.prefix);
		let _ = writeln!(code, "{row} = udiv i32 {element}, {elements}");
		let _ = writeln!(code, "{local} = urem i32 {element}, {elements}");
		let _ = writeln!(code, "{position} = urem i32 {local}, {length}");
		match context.mode {
			NormalizeMode::Batch | NormalizeMode::Evaluation => {
				let channel = format!("%{}.normalize.reverse.channel", context.prefix);
				let _ = writeln!(code, "{channel} = udiv i32 {local}, {length}");
				let _ = writeln!(code, "{group} = add i32 {channel}, 0");
				let _ = writeln!(code, "{groups} = add i32 0, {channels}", channels = context.channels);
			}
			NormalizeMode::Layer | NormalizeMode::Rms => {
				let row_base = format!("%{}.normalize.reverse.row.base", context.prefix);
				let _ = writeln!(code, "{row_base} = mul i32 {row}, {length}");
				let _ = writeln!(code, "{group} = add i32 {row_base}, {position}");
				let _ = writeln!(code, "{groups} = mul i32 {rows}, {length}", rows = context.rows);
			}
		}
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
		let output_projection = format!("%{}.normalize.reverse.output.projection", context.prefix);
		let centered = format!("%{}.normalize.reverse.centered", context.prefix);
		let numerator = format!("%{}.normalize.reverse.numerator", context.prefix);
		let contribution = format!("%{}.normalize.reverse.contribution", context.prefix);
		let _ = writeln!(code, "{output_projection} = call {ty} @recipe.mul({ty} {output_value}, {ty} {projected})", ty = context.value_type);
		let _ = writeln!(code, "{centered} = call {ty} @recipe.sub({ty} {delta}, {ty} {sum})", ty = context.value_type);
		let _ = writeln!(code, "{numerator} = call {ty} @recipe.sub({ty} {centered}, {ty} {output_projection})", ty = context.value_type);
		let _ = writeln!(code, "{contribution} = call {ty} @recipe.mul({ty} {scale}, {ty} {numerator})", ty = context.value_type);
		NormalizeReverseFragment { code, contribution }
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
const NATIVE_FORWARD_LAYOUT: &[u8] = b"888844";
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
	pub(crate) fn from_graph(graph: &Graph, rows: usize, precision: Compute, schedule: NativeSchedule) -> Result<Self> {
		require(rows != 0, "native model rows must be positive")?;
		let layout = NativeLayout::for_graph(graph, rows, precision)?;
		let precision = NativePrecision::new(precision)?;
		let mut plans = Vec::with_capacity(graph.nodes.len());
		let mut storage_bytes = 0usize;
		for (index, node) in graph.nodes.iter().cloned().enumerate() {
			let id = || node.identity(index);
			require(node.source >= -1 && node.source < index as i32, format!("{} has invalid source node {}", id(), node.source))?;
			require(node.second >= -2 && node.second < index as i32, format!("{} has invalid second source node {}", id(), node.second))?;
			require(node.offset.checked_add(node.parameters).is_some_and(|end| end <= graph.parameters.len()), format!("{} parameter range exceeds {} values", id(), graph.parameters.len()))?;
			let width = if node.op == Primitive::Predictor { 2 } else { 3 };
			let program_width = node.program_count.checked_mul(width).ok_or_else(|| RecipeError::new(format!("{} program length overflows", id())))?;
			require(node.program_offset.checked_add(program_width).is_some_and(|end| end <= graph.programs.len()), format!("{} program range exceeds {} values", id(), graph.programs.len()))?;
			let stored = graph.stored.get(index).cloned().unwrap_or(None);
			if let Some(weight) = &stored {
				require(weight.count == node.parameters, format!("{} stored weight count {} does not match parameter count {}", id(), weight.count, node.parameters))?;
			}
			let storage_offset = align(storage_bytes, alignment("float"))?;
			if let Some(weight) = &stored {
				storage_bytes = checked_add(storage_offset, weight.bytes.len(), "native storage arena")?;
			}
			plans.push(NodePlan { node, value: layout.values[index], context: layout.contexts[index], adjoint: layout.adjoints[index], stored, storage_offset });
		}
		Ok(Self { graph: graph.clone(), layout, precision, rows, schedule, plans, storage_bytes })
	}
	fn storage(&self) -> Vec<u8> {
		let mut storage = Vec::with_capacity(self.storage_bytes);
		for plan in &self.plans {
			if let Some(weight) = &plan.stored {
				storage.resize(plan.storage_offset, 0);
				storage.extend_from_slice(&weight.bytes);
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
	format!("%{name} = getelementptr i8, {pointer} %{base}, i32 {offset}\n")
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
			match (reverse, node.op) {
				(false, Primitive::Contraction) => {
					let extent = self.schedule.contractions[index].ok_or_else(|| RecipeError::new("native contraction schedule is absent"))?.forward;
					require(node.argument[1] == 0.0 || node.argument[1] == 1.0, "contraction ReLU flag is invalid")?;
					let call = format!(
						"call void @contraction_forward_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {source}, i32 %rows, i32 {in_channels}, i32 {in_length}, i32 {out_channels}, i32 {out_length}, i32 {kernel}, i1 true, i1 {relu}, i1 false, i1 false, i1 false, i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads )\n",
						pointer = pointer_type(backend),
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
				(false, Primitive::Expand) => {
					let count = checked_mul(self.rows, node.output.elements(), "expand count")?;
					emit_fixed_loop(&mut ir, index, "expand", count, |ir, p| {
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
				(false, Primitive::Read) => {
					let count = checked_mul(self.rows, node.output.elements(), "read count")?;
					emit_fixed_loop(&mut ir, index, "read", count, |ir, p| {
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
				(false, Primitive::Outer) => {
					let count = checked_mul(self.rows, node.output.elements(), "outer count")?;
					emit_fixed_loop(&mut ir, index, "outer", count, |ir, p| {
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
				(false, Primitive::Pool) => {
					let size = integer_argument(node.argument[0], "pool size")?;
					let count = checked_mul(self.rows, node.output.elements(), "pool output count")?;
					emit_fixed_loop(&mut ir, index, "pool", count, |ir, p| {
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
					ir.push_str(&format!("call void @{attention}( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {context}, i32 %rows, i32 {from}, i32 {heads}, i32 {channels}, i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads )\n", pointer = pointer_type(backend), source = pointers.source, weights = pointers.weights, value = pointers.value, context = pointers.context, from = node.output.elements(), heads = integer_argument(node.argument[0], "attention heads")?, channels = node.output.channels, tile_m = extent.m, tile_n = extent.n, tile_k = extent.k));
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Scan) => {
					let extent = self.schedule.contractions[index].ok_or_else(|| RecipeError::new("native scan schedule is absent"))?.forward;
					ir.push_str(&format!("call void @scan_forward_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {context}, i32 %rows, i32 {in_channels}, i32 {in_length}, i32 {out_channels}, i32 {gates}, i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads )\n", pointer = pointer_type(backend), source = pointers.source, weights = pointers.weights, value = pointers.value, context = pointers.context, in_channels = node.input.channels, in_length = node.input.length, out_channels = node.output.channels, gates = integer_argument(node.argument[0], "scan gates")?, tile_m = extent.m, tile_n = extent.n, tile_k = extent.k));
					ir.push_str(barrier(backend));
				}
				(false, Primitive::Elementwise) => {
					let count = checked_mul(self.rows, node.output.elements(), "scalar output count")?;
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
							prefix: &prefix,
							literal: &literal,
						},
					)
					.map_err(|error| RecipeError::new(error.to_string()))?;
					emit_fixed_loop(&mut ir, index, "scalar", count, |ir, p| {
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
					let count = checked_mul(self.rows, node.output.elements(), "predictor output count")?;
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
					emit_fixed_loop(&mut ir, index, "predictor", count, |ir, p| {
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
					ir.push_str(&format!("call void @contraction_reverse_body( {pointer} {source}, {pointer} {weights}, {pointer} {value}, {pointer} {delta}, {pointer} {source_adjoint}, {pointer} %gradient, i1 {write_input}, i1 true, i1 {relu}, i1 {matrix_gradient}, i32 %rows, i32 {in_channels}, i32 {in_length}, i32 {out_channels}, i32 {out_length}, i32 {kernel}, i32 {offset}, i32 {gradient_m}, i32 {gradient_n}, i32 {gradient_k}, i32 {previous_m}, i32 {previous_n}, i32 {previous_k}, i32 %threads )\n", pointer = pointer_type(backend), source = pointers.source, weights = pointers.weights, value = pointers.value, delta = pointers.delta, source_adjoint = pointers.source_adjoint, write_input = !composed_previous, matrix_gradient = matrix_gradient, in_channels = node.input.channels, in_length = node.input.length, out_channels = node.output.channels, out_length = node.output.length, kernel = kernel, offset = plan.node.offset, relu = node.argument[1] == 1.0, gradient_m = tiles.gradient.m, gradient_n = tiles.gradient.n, gradient_k = tiles.gradient.k, previous_m = tiles.previous.m, previous_n = tiles.previous.n, previous_k = tiles.previous.k));
					if composed_previous {
						ir.push_str(&format!("call void @contraction_forward_body( {pointer} {delta}, {pointer} {weights}, {pointer} {source_adjoint}, {pointer} {value}, i32 %rows, i32 {out_channels}, i32 {out_length}, i32 {in_channels}, i32 {in_length}, i32 0, i1 false, i1 {relu}, i1 true, i1 true, i1 {accumulate}, i32 {previous_m}, i32 {previous_n}, i32 {previous_k}, i32 %threads )\n", pointer = pointer_type(backend), delta = pointers.delta, weights = pointers.weights, source_adjoint = pointers.source_adjoint, value = pointers.value, out_channels = node.output.channels, out_length = node.output.length, in_channels = node.input.channels, in_length = node.input.length, relu = node.argument[1] == 1.0, accumulate = accumulate_previous, previous_m = tiles.previous.m, previous_n = tiles.previous.n, previous_k = tiles.previous.k));
					}
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Expand) => {
					let count = checked_mul(self.rows, node.input.elements(), "expand reverse count")?;
					emit_fixed_loop(&mut ir, index, "expand.reverse", count, |ir, p| {
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
				(true, Primitive::Read) => {
					let count = checked_mul(self.rows, node.input.elements(), "read reverse count")?;
					emit_fixed_loop(&mut ir, index, "read.reverse", count, |ir, p| {
						ir.push_str(&format!("call void @read_reverse_body( {pointer} {source}, {pointer} {gate}, {pointer} {delta}, {pointer} {adjoint}, {pointer} {gate_adjoint}, i32 {p}, i32 {channels}, i32 {length}, i32 {lanes}, i1 {gated} )\n", pointer = pointer_type(backend), source = pointers.source, gate = pointers.second, delta = pointers.delta, adjoint = pointers.source_adjoint, gate_adjoint = pointers.second_adjoint, channels = node.output.channels, length = node.output.length, lanes = node.argument[0], gated = node.second >= 0));
					})?;
					ir.push_str(barrier(backend));
				}
				(true, Primitive::Outer) => {
					let count = checked_mul(self.rows, node.input.elements(), "outer reverse count")?;
					emit_fixed_loop(&mut ir, index, "outer.reverse", count, |ir, p| {
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
						let gates = checked_mul(checked_mul(self.rows, node.argument[0] as usize, "outer gate rows")?, node.input.length, "outer gate count")?;
						emit_fixed_loop(&mut ir, index, "outer.gate.reverse", gates, |ir, p| {
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
				(true, Primitive::Pool) => {
					let count = checked_mul(self.rows, node.output.elements(), "pool reverse count")?;
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let prefix = format!("n{index}.pool.reverse");
					emit_fixed_loop(&mut ir, index, "pool.reverse", count, |ir, p| {
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
					ir.push_str(&format!("call void @{attention}( {pointer} {source}, {pointer} {value}, {pointer} {context}, {pointer} {delta}, {pointer} {source_adjoint}, i32 %rows, i32 {from}, i32 {heads}, i32 {channels}, i32 {tile_m}, i32 {tile_n}, i32 {tile_k}, i32 %threads )\n", pointer = pointer_type(backend), source = pointers.source, value = pointers.value, context = pointers.context, delta = pointers.delta, source_adjoint = pointers.source_adjoint, from = node.output.elements(), heads = integer_argument(node.argument[0], "attention heads")?, channels = node.output.channels, tile_m = extent.m, tile_n = extent.n, tile_k = extent.k));
					ir.push_str(barrier(backend));
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
					let count = checked_mul(self.rows, node.output.elements(), "scalar reverse count")?;
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
						emit_fixed_loop(&mut ir, index, "scalar.reverse", count, scalar_body)?;
						ir.push_str(barrier(backend));
					} else {
						// A trainable scalar is one destination shared by every element, so
						// the summation order has to belong to the program rather than to
						// the schedule. Each partition sums its own contiguous run of
						// elements in ascending order into its own scratch row, and one
						// owner then folds the rows in ascending partition order.
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
					let count = checked_mul(self.rows, node.output.elements(), "normalize output count")?;
					let mode = normalize_mode(node.argument[0])?;
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let prefix = format!("n{index}.normalize");
					if mode != program_ir::NormalizeMode::Evaluation && (training || matches!(mode, program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms)) {
						ir.push_str(&self.emit_normalize_stats(backend, index, node, &pointers, mode)?);
						ir.push_str(barrier(backend));
					}
					emit_fixed_loop(&mut ir, index, "normalize", count, |ir, p| {
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
					let count = checked_mul(self.rows, node.output.elements(), "normalize reverse count")?;
					let mode = normalize_mode(node.argument[0])?;
					let pointer = pointer_type(backend);
					let ty = self.precision.model_type;
					let prefix = format!("n{index}.normalize.reverse");
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
								mode,
								prefix: &stats_prefix,
							},
							&pointers.delta,
							&pointers.value,
						));
						ir.push_str(barrier(backend));
					}
					emit_fixed_loop(&mut ir, index, "normalize.reverse", count, |ir, p| {
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
		match mode {
			program_ir::NormalizeMode::Batch => {
				ir.push_str(&format!("{items} = mul i32 %rows, {length}\n", length = length));
			}
			program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms => {
				ir.push_str(&format!("{groups} = mul i32 %rows, {length}\n{items} = add i32 0, {channels}\n", length = length, channels = channels));
			}
			program_ir::NormalizeMode::Evaluation => return Ok(ir),
		}
		let group_limit = match mode {
			program_ir::NormalizeMode::Batch => channels.to_string(),
			program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms => groups.clone(),
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
				program_ir::NormalizeMode::Layer | program_ir::NormalizeMode::Rms => {
					code.push_str(&format!("{row} = udiv i32 {group}, {length}\n{position} = urem i32 {group}, {length}\n{row_base} = mul i32 {row}, {elements}\n{channel_base} = mul i32 {p}, {length}\n{local} = add i32 {channel_base}, {position}\n{value_index} = add i32 {row_base}, {local}\n", p = p, length = length, elements = elements, group = group));
				}
				program_ir::NormalizeMode::Evaluation => unreachable!(),
			}
		};
		ir.push_str(&format!("br label %{prefix}.entry\n{prefix}.entry:\nbr label %{prefix}.group.loop\n{prefix}.group.loop:\n{group} = phi i32 [ %tid, %{prefix}.entry ], [ %{prefix}.group.next, %{prefix}.store ]\n%{prefix}.group.more = icmp ult i32 {group}, {group_limit}\nbr i1 %{prefix}.group.more, label %{prefix}.mean.loop, label %{prefix}.done\n{prefix}.mean.loop:\n%{prefix}.mean.p = phi i32 [ 0, %{prefix}.group.loop ], [ %{prefix}.mean.next, %{prefix}.mean.step ]\n%{prefix}.mean.sum = phi {ty} [ {zero}, %{prefix}.group.loop ], [ %{prefix}.mean.sum.next, %{prefix}.mean.step ]\n%{prefix}.mean.more = icmp ult i32 %{prefix}.mean.p, {items}\nbr i1 %{prefix}.mean.more, label %{prefix}.mean.step, label %{prefix}.variance.loop\n{prefix}.mean.step:\n", group = group, group_limit = group_limit, ty = state_ty, zero = zero, items = items));
		emit_index(&mut ir, "mean", &format!("%{prefix}.mean.p"));
		ir.push_str(&format!("%{prefix}.mean.ptr = getelementptr inbounds {ty}, {pointer} {source}, i32 %{prefix}.mean.index\n%{prefix}.mean.model = load {ty}, {pointer} %{prefix}.mean.ptr, align {align}\n%{prefix}.mean.value = call {state_ty} @recipe.state.from.model({ty} %{prefix}.mean.model)\n%{prefix}.mean.sum.next = call {state_ty} @recipe.state.add({state_ty} %{prefix}.mean.sum, {state_ty} %{prefix}.mean.value)\n%{prefix}.mean.next = add i32 %{prefix}.mean.p, 1\nbr label %{prefix}.mean.loop\n{prefix}.variance.loop:\n%{prefix}.variance.p = phi i32 [ 0, %{prefix}.mean.loop ], [ %{prefix}.variance.next, %{prefix}.variance.step ]\n%{prefix}.variance.sum = phi {state_ty} [ {zero}, %{prefix}.mean.loop ], [ %{prefix}.variance.sum.next, %{prefix}.variance.step ]\n%{prefix}.items.value = call {state_ty} @recipe.state.from.u32(i32 {items})\n%{prefix}.mean = call {state_ty} @recipe.state.div({state_ty} %{prefix}.mean.sum, {state_ty} %{prefix}.items.value)\n%{prefix}.variance.more = icmp ult i32 %{prefix}.variance.p, {items}\nbr i1 %{prefix}.variance.more, label %{prefix}.variance.step, label %{prefix}.store\n{prefix}.variance.step:\n", pointer = pointer, source = pointers.source, ty = ty, state_ty = state_ty, zero = zero, items = items, align = alignment(ty)));
		emit_index(&mut ir, "variance", &format!("%{prefix}.variance.p"));
		ir.push_str(&format!("%{prefix}.variance.ptr = getelementptr inbounds {ty}, {pointer} {source}, i32 %{prefix}.variance.index\n%{prefix}.variance.model = load {ty}, {pointer} %{prefix}.variance.ptr, align {align}\n%{prefix}.variance.value = call {state_ty} @recipe.state.from.model({ty} %{prefix}.variance.model)\n%{prefix}.variance.centered = call {state_ty} @recipe.state.sub({state_ty} %{prefix}.variance.value, {state_ty} %{prefix}.mean)\n", pointer = pointer, source = pointers.source, ty = ty, state_ty = state_ty, align = alignment(ty)));
		let difference = if mode == program_ir::NormalizeMode::Rms { format!("%{prefix}.variance.value") } else { format!("%{prefix}.variance.centered") };
		ir.push_str(&format!("%{prefix}.variance.square = call {state_ty} @recipe.state.mul({state_ty} {difference}, {state_ty} {difference})\n%{prefix}.variance.sum.next = call {state_ty} @recipe.state.add({state_ty} %{prefix}.variance.sum, {state_ty} %{prefix}.variance.square)\n%{prefix}.variance.next = add i32 %{prefix}.variance.p, 1\nbr label %{prefix}.variance.loop\n{prefix}.store:\n%{prefix}.variance = call {state_ty} @recipe.state.div({state_ty} %{prefix}.variance.sum, {state_ty} %{prefix}.items.value)\n%{prefix}.adjusted = call {state_ty} @recipe.state.add({state_ty} %{prefix}.variance, {state_ty} {epsilon})\n%{prefix}.deviation = call {state_ty} @recipe.state.sqrt({state_ty} %{prefix}.adjusted)\n%{prefix}.scale.state = call {state_ty} @recipe.state.div({state_ty} {one}, {state_ty} %{prefix}.deviation)\n%{prefix}.mean.stored = call {ty} @recipe.model.from.state({state_ty} %{prefix}.mean)\n%{prefix}.scale = call {ty} @recipe.model.from.state({state_ty} %{prefix}.scale.state)\n%{prefix}.mean.context.ptr = getelementptr inbounds {ty}, {pointer} {context}, i32 {group}\n%{prefix}.scale.index = add i32 {group_limit}, {group}\n%{prefix}.scale.ptr = getelementptr inbounds {ty}, {pointer} {context}, i32 %{prefix}.scale.index\n", pointer = pointer, context = pointers.context, ty = ty, state_ty = state_ty, epsilon = epsilon, one = one, group = group, group_limit = group_limit));
		let stored_mean = if mode == program_ir::NormalizeMode::Rms { model_zero.clone() } else { format!("%{prefix}.mean.stored") };
		ir.push_str(&format!("store {ty} {stored_mean}, {pointer} %{prefix}.mean.context.ptr, align {align}\nstore {ty} %{prefix}.scale, {pointer} %{prefix}.scale.ptr, align {align}\n%{prefix}.group.next = add i32 {group}, %threads\nbr label %{prefix}.group.loop\n{prefix}.done:\n", pointer = pointer, ty = ty, stored_mean = stored_mean, align = alignment(ty), group = group));
		Ok(ir)
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
		let weight_bytes = checked_mul(plan.node.offset, self.precision.model.bytes(), "native parameter offset")?;
		ir.push_str(&ptr_gep(backend, "weights", weight_bytes, &format!("{prefix}.weights")));
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
			let Some(stored) = &plan.stored else { continue };
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
			let Some(stored) = &plan.stored else { continue };
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
			ir.push_str(&format!("br label %{prefix}.loop\n{prefix}.loop:\n%{prefix}.p = phi i32 [ %tid, %entry ], [ %{prefix}.next, %{prefix}.step ]\n%{prefix}.more = icmp ult i32 %{prefix}.p, {count}\nbr i1 %{prefix}.more, label %{prefix}.step, label %{prefix}.done\n{prefix}.step:\n%{prefix}.storage = getelementptr i8, {pointer} %storage, i32 {storage}\n%{prefix}.index = add i32 %{prefix}.p, {weight}\n%{prefix}.weights = getelementptr {ty}, {pointer} %weights, i32 %{prefix}.index\n%{prefix}.value = call {ty} @recipe_model_quantized_{name}({pointer} %{prefix}.storage, i32 0, i32 %{prefix}.p, i32 {columns})\nstore {ty} %{prefix}.value, {pointer} %{prefix}.weights, align {align}\n%{prefix}.next = add i32 %{prefix}.p, %threads\nbr label %{prefix}.loop\n{prefix}.done:\n", pointer = pointer, ty = ty, count = count, storage = plan.storage_offset, weight = plan.node.offset, name = name, columns = columns, align = alignment(ty)).replace("%entry", &format!("%{predecessor}")));
			ir.push_str(barrier(backend));
			predecessor = format!("{prefix}.done");
		}
		ir.push_str("ret void\n}\n");
		Ok(ir)
	}

	pub(crate) fn emit(&self, backend: Backend, matrix: Option<NativeMatrix>, loss: Option<LossFunction>) -> Result<String> {
		let register_count = self.schedule.register_count;
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
			.replace("RECIPE_GRADIENT_SCRATCH_BASE", &self.schedule.scratch_base.to_string());
		let quantized_definitions = self.emit_quantized_decoders(backend)?;
		let model_load = self.emit_model_load(backend)?;
		ir.push_str(&quantized_definitions);
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
		let forward_args = format!("{pointer} %samples, {pointer} %weights, {pointer} %values, {pointer} %contexts, i32 %rows, i32 %threads");
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
			body.push_str(&format!("\ncall void @recipe_model_training_forward_body({pointer} %samples, {pointer} %weights, {pointer} %values, {pointer} %contexts, i32 %rows, i32 %threads)\n"));
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
		ir.push_str(&format!("%prediction.base = getelementptr i8, {pointer} %values, i32 {prediction_offset}\n%prediction = bitcast {pointer} %prediction.base to {pointer}\n%metric.ptr = getelementptr {state_ty}, {pointer} %metrics, i32 0\n%loss.leader = icmp eq i32 %tid, 0\nbr i1 %loss.leader, label %loss.entry, label %loss.wait\nloss.entry:\n"));
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
		ir.push_str(&format!("%adjoint.base = getelementptr i8, {pointer} %adjoints, i32 {adjoint_offset}\n%adjoint = bitcast {pointer} %adjoint.base to {pointer}\nbr label %seed.loop\nseed.loop:\n%seed.p = phi i32 [ %tid, %loss.wait ], [ %seed.next, %seed.step ]\n%seed.more = icmp ult i32 %seed.p, {items}\nbr i1 %seed.more, label %seed.step, label %seed.done\nseed.step:\n%seed.pred.ptr = getelementptr {model_ty}, {pointer} %prediction, i32 %seed.p\n%seed.pred.model = load {model_ty}, {pointer} %seed.pred.ptr, align {model_align}\n%seed.pred = call {state_ty} @recipe.state.from.model({model_ty} %seed.pred.model)\n%seed.target.ptr = getelementptr {model_ty}, {pointer} %targets, i32 %seed.p\n%seed.target.model = load {model_ty}, {pointer} %seed.target.ptr, align {model_align}\n%seed.target = call {state_ty} @recipe.state.from.model({model_ty} %seed.target.model)\n",));
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
		_ => Err(RecipeError::new("normalization mode is unsupported")),
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
	ir.push_str(&format!(
		"%{prefix}.row = mul i32 %{prefix}.t, {columns}\nbr label %{prefix}.inner\n{prefix}.inner:\n%{prefix}.p = phi i32 [ %{prefix}.first, %{prefix}.body ], [ %{prefix}.p.next, %{prefix}.fold ]\n"
	));
	for (parameter, _) in gradients {
		ir.push_str(&format!("%{prefix}.sum.{parameter} = phi {ty} [ {zero}, %{prefix}.body ], [ %{prefix}.sum.{parameter}.next, %{prefix}.fold ]\n"));
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
	for column in 0..columns {
		let stored = gradients.iter().find(|(parameter, _)| *parameter as i32 == column).map_or_else(|| zero.to_owned(), |(parameter, _)| format!("%{prefix}.sum.{parameter}"));
		ir.push_str(&format!("%{prefix}.index.{column} = add i32 %{prefix}.row, {column}\n%{prefix}.column.{column} = getelementptr inbounds {ty}, {pointer} {scratch}, i32 %{prefix}.index.{column}\nstore {ty} {stored}, {pointer} %{prefix}.column.{column}, align {align}\n"));
	}
	ir.push_str(&format!("br label %{prefix}.step\n{prefix}.step:\n%{prefix}.advance = add i32 %{prefix}.t, %threads\nbr label %{prefix}.loop\n{prefix}.done:\n"));
	Ok(())
}

fn emit_fixed_loop(ir: &mut String, index: usize, name: &str, count: usize, mut body: impl FnMut(&mut String, &str)) -> Result<()> {
	let prefix = format!("n{index}.{name}");
	let count = i32::try_from(count).map_err(|_| RecipeError::new(format!("native {name} loop count exceeds i32")))?;
	ir.push_str(&format!("br label %{prefix}.entry\n{prefix}.entry:\nbr label %{prefix}.loop\n{prefix}.loop:\n%{prefix}.p = phi i32 [ %tid, %{prefix}.entry ], [ %{prefix}.next, %{prefix}.step ]\n%{prefix}.more = icmp ult i32 %{prefix}.p, {count}\nbr i1 %{prefix}.more, label %{prefix}.body, label %{prefix}.done\n{prefix}.body:\n"));
	body(ir, &format!("%{prefix}.p"));
	ir.push_str(&format!("br label %{prefix}.step\n{prefix}.step:\n%{prefix}.next = add i32 %{prefix}.p, %threads\nbr label %{prefix}.loop\n{prefix}.done:\n"));
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
	let model = NativeModelIr::from_graph(graph, rows, precision, schedule)?;
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
			Operation::Attention(heads) => format!("attn,{heads}"),
			Operation::Rnn(width) => format!("rnn,{width}"),
			Operation::Gru(width) => format!("gru,{width}"),
			Operation::Lstm(width) => format!("lstm,{width}"),
			Operation::Residual(parts) => format!("residual,{}", parts.iter().map(residual_text).collect::<Vec<_>>().join(";")),
			Operation::Moe(top_k, experts) => format!("moe,{top_k},{}", experts.iter().map(residual_text).collect::<Vec<_>>().join(";")),
			Operation::Hyper(lanes, rank, parts) => format!("hyper,{lanes},{rank},{}", parts.iter().map(residual_text).collect::<Vec<_>>().join(";")),
			Operation::Perceptron(width) => format!("perc,{width}"),
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
			"attn" => Ok(Operation::Attention(value_at(Some(rest), "attention heads")?)),
			"rnn" => Ok(Operation::Rnn(value_at(Some(rest), "RNN width")?)),
			"gru" => Ok(Operation::Gru(value_at(Some(rest), "GRU width")?)),
			"lstm" => Ok(Operation::Lstm(value_at(Some(rest), "LSTM width")?)),
			"residual" => Ok(Operation::Residual(if rest.is_empty() { Vec::new() } else { rest.split(';').map(residual).collect::<Result<Vec<_>>>()? })),
			"moe" => {
				let (top_k, experts) = rest.split_once(',').unwrap_or((rest, ""));
				Ok(Operation::Moe(value_at(Some(top_k), "MoE top-k")?, experts.split(';').filter(|part| !part.is_empty()).map(residual).collect::<Result<Vec<_>>>()?))
			}
			"perc" => Ok(Operation::Perceptron(value_at(Some(rest), "perceptron width")?)),
			"hyper" => {
				let (lanes, rest) = rest.split_once(',').unwrap_or((rest, ""));
				let (rank, parts) = rest.split_once(',').unwrap_or((rest, ""));
				Ok(Operation::Hyper(
					value_at(Some(lanes), "hyper-connection lanes")?,
					value_at(Some(rank), "hyper-connection rank")?,
					parts.split(';').filter(|part| !part.is_empty()).map(residual).collect::<Result<Vec<_>>>()?,
				))
			}
			_ => Err(RecipeError::new(format!("invalid model operation {name:?}"))),
		}
	}
	fn block_text(block: &Block) -> String {
		format!("{}|{}|{}|{}|{}", operation_text(&block.operation), block.activation as u8, block.normalization.map_or(0, |value| value as u8 + 1), block.quantization, u8::from(block.profile))
	}
	fn block(value: &str) -> Result<Block> {
		let fields = value.split('|').collect::<Vec<_>>();
		require(fields.len() == 5, "semantic model block has the wrong width")?;
		let normalization = match value_at::<u8>(Some(fields[2]), "block normalization")? {
			0 => None,
			1 => Some(BlockNormalization::Batch),
			2 => Some(BlockNormalization::Layer),
			_ => return Err(RecipeError::new("invalid block normalization")),
		};
		Ok(Block {
			operation: operation(fields[0])?,
			activation: activation(value_at(Some(fields[1]), "block activation")?)?,
			normalization,
			quantization: value_at(Some(fields[3]), "block quantization")?,
			profile: bool_value(fields[4], "block quantization profile")?,
		})
	}
	fn model_text(model: &Model) -> Vec<String> {
		model.blocks.iter().map(block_text).collect()
	}
	fn model(blocks: Vec<Block>, loss: u8, quantization: u16) -> Result<Model> {
		require(!blocks.is_empty(), "semantic model has no blocks")?;
		require(matches!(loss, 0..=4 | 6), format!("saved model loss {loss} is unavailable"))?;
		Ok(Model { blocks, loss: LossFunction(loss), quantization })
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

	fn raw_weight(values: &[f64]) -> StoredWeight {
		let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect();
		StoredWeight { format: StorageFormat(0), count: values.len(), bytes, codebook: Vec::new(), arithmetic: values.to_vec() }
	}
	fn semantic_graph(stored: &StoredGraph) -> Result<SemanticGraph> {
		let graph = &stored.graph;
		let mut tensors = Vec::new();
		for (index, node) in graph.nodes.iter().enumerate() {
			if node.parameters == 0 {
				continue;
			}
			let values = graph.parameters.get(node.offset..node.offset + node.parameters).ok_or_else(|| RecipeError::new("model parameter span is invalid"))?;
			let encoded = graph.stored.get(index).and_then(Clone::clone).unwrap_or_else(|| raw_weight(values));
			require(encoded.count == node.parameters && encoded.arithmetic.len() == node.parameters, format!("model tensor {index} has the wrong shape"))?;
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
		a.loss.0 == b.loss.0 && a.quantization == b.quantization && model_text(a) == model_text(b)
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
			)?;
			require(self.inputs.len() == input.elements(), "semantic model input schema has the wrong width")?;
			require(self.outputs.len() == output.elements(), "semantic model output schema has the wrong width")?;
			require(
				self.norm_mean.len() == self.norm_scale.len() && (self.norm_mean.is_empty() || self.norm_mean.len() == self.inputs.len()),
				"semantic model normalization stats have the wrong width",
			)?;
			require(!self.artifact.is_empty(), "native artifact identity is absent")?;
			require(self.frozen.len() == self.tensors.iter().map(|tensor| tensor.count).sum::<usize>(), "semantic model frozen weights are incomplete")?;
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
		Ok(StoredWeight { format: StorageFormat(format), count, bytes, codebook, arithmetic })
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
					require(fields.len() == 2, "semantic model header has the wrong width")?;
					require(builder.model.is_none(), "semantic graph has more than one model")?;
					builder.model = Some(ModelParts {
						loss: Some(value_at(fields.first().copied(), "semantic model loss")?),
						quantization: Some(value_at(fields.get(1).copied(), "semantic model quantization")?),
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
			field(&mut document, "model", &format!("{} {}", semantic.model.loss.0, semantic.model.quantization));
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
				field(&mut document, "tensor", &format!("{} {} {metadata} {}", tensor.format.0, tensor.count, hex(&tensor.bytes)));
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
		feed(&format!("loss:{};quant:{};blocks:{};", model.loss.0, model.quantization, model_text(model).join("/")));
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
					if node.parameters == 0 {
						continue;
					}
					let encoded = saved.tensors.get(tensor).ok_or_else(|| RecipeError::new("saved semantic tensor is absent"))?;
					require(encoded.count == node.parameters, "saved semantic tensor has the wrong shape")?;
					current.graph.parameters[node.offset..node.offset + node.parameters].copy_from_slice(&encoded.arithmetic);
					if let Some(slot) = current.graph.stored.get_mut(index) {
						*slot = (encoded.format.0 != 0).then_some(encoded.clone())
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
	pub(super) fn run_infer(path: &Path, input: &[f64], forward: impl Fn(&SemanticGraph, &[f64]) -> Result<Vec<f64>>) -> Result<Vec<f64>> {
		let (_, graphs) = load_semantic(path)?;
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
			result = forward(&stored, &samples)?;
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
		Mutex, OnceLock,
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
#[derive(Clone, Debug, PartialEq, Eq)]
enum Operation {
	Layer(usize),
	Conv(usize, usize),
	Pool(usize),
	Estimator(Estimator),
	Attention(usize),
	Rnn(usize),
	Gru(usize),
	Lstm(usize),
	Residual(Vec<Residual>),
	Moe(usize, Vec<Residual>),
	Perceptron(usize),
	Hyper(usize, usize, Vec<Residual>),
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockNormalization {
	Batch,
	Layer,
}
/// The normalization selectors with a declared identity: the batch marker and the layer
/// residual constructor. Any other selector is rejected instead of guessing a mode.
pub trait NormalizationSelector {
	fn normalization(self) -> BlockNormalization;
}
impl NormalizationSelector for Batch {
	fn normalization(self) -> BlockNormalization {
		BlockNormalization::Batch
	}
}
impl<F: Fn(usize) -> Residual> NormalizationSelector for F {
	fn normalization(self) -> BlockNormalization {
		match self(0) {
			Residual::Layer(_) => BlockNormalization::Layer,
			_ => panic!("normalization selector must be batch or layer"),
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
	quantization: u16,
	profile: bool,
}
#[derive(Clone)]
pub struct Model {
	blocks: Vec<Block>,
	loss: LossFunction,
	quantization: u16,
}
macro_rules! operation_methods { ($(fn $method:ident($($argument:ident: $kind:ty),*) = $operation:expr;)+) => {
$(pub fn $method(&self, $($argument: $kind),*) -> Self { self.push($operation) })+ }; }
impl Model {
	fn push(&self, operation: Operation) -> Self {
		let mut model = self.clone();
		model.blocks.push(Block {
			operation,
			activation: Activation::Linear,
			normalization: None,
			quantization: model.quantization,
			profile: StorageFormat(model.quantization).selection().is_some(),
		});
		model
	}
	pub fn activate(&self, activation: Activation) -> Self {
		let mut model = self.clone();
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
	fn attn(heads: usize) = Operation::Attention(heads);
	fn rnn(width: usize) = Operation::Rnn(width);
	fn gru(width: usize) = Operation::Gru(width);
	fn lstm(width: usize) = Operation::Lstm(width);
	fn perc(width: usize) = Operation::Perceptron(width); }
	pub fn res<const N: usize>(&self, parts: [Residual; N]) -> Self {
		self.push(Operation::Residual(parts.into()))
	}
	pub fn moe<const N: usize>(&self, top_k: usize, experts: [Residual; N]) -> Self {
		self.push(Operation::Moe(top_k, experts.into()))
	}
	/// Hyper-connections: a stream of `lanes` copies of the width feeds `parts`
	/// through a gated read and takes their output back through gated writes.
	/// `rank` sizes the gate bottleneck; zero fixes every gate at one.
	pub fn hyper<const N: usize>(&self, lanes: usize, rank: usize, parts: [Residual; N]) -> Self {
		self.push(Operation::Hyper(lanes, rank, parts.into()))
	}
	pub fn norm(&self, normalization: impl NormalizationSelector) -> Self {
		let mut model = self.clone();
		let block = model.blocks.last_mut().unwrap_or_else(|| panic!("normalization requires a preceding block"));
		block.normalization = Some(normalization.normalization());
		model
	}
	pub fn loss(&self, loss: LossFunction) -> Self {
		let mut model = self.clone();
		model.loss = loss;
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
				if selected.2
					&& let Some(name) = block.normalization.map(BlockNormalization::name)
				{
					names.push(name.to_owned())
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
#[derive(Clone, Copy)]
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

#[derive(Clone)]
pub(crate) struct StoredWeight {
	pub(crate) format: StorageFormat,
	pub(crate) count: usize,
	pub(crate) bytes: Vec<u8>,
	pub(crate) codebook: Vec<f64>,
	pub(crate) arithmetic: Vec<f64>,
}

impl StorageFormat {
	fn valid(self) -> bool {
		self.spec().is_some() || self.selection().is_some()
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
		Ok(StoredWeight { format: self, count: arithmetic.len(), bytes, codebook, arithmetic: arithmetic.to_vec() })
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
					let rms = (slice.iter().map(|value| value * value).sum::<f32>() / 32.0).sqrt();
					let weights = slice.iter().map(|value| rms + value.abs()).collect::<Vec<_>>();
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
			Self::Attention(_) => "attn",
			Self::Rnn(_) => "rnn",
			Self::Gru(_) => "gru",
			Self::Lstm(_) => "lstm",
			Self::Residual(_) => "residual",
			Self::Moe(..) => "moe",
			Self::Perceptron(_) => "perc",
			Self::Hyper(..) => "hyper",
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
		Model { blocks: Vec::new(), loss: mse, quantization: 0 }
	}
	pub const fn train(&self) -> Train {
		Train { epochs: 1, learning_rate: 0.001, log_metrics: Vec::new(), stop: Some(1.0), resume: None, save: None, seed: None, precision: Compute::FP64 }
	}
}
impl Recipe {
	pub fn infer(&self, path: impl AsRef<Path>, input: &[f64]) -> Vec<f64> {
		let path = resolve_path(path).unwrap_or_else(|error| panic!("{error}"));
		let device = selected_gpu().unwrap_or_else(|error| panic!("{error}"));
		let result = bundle::run_infer(&path, input, |stored, samples| {
			let config = Config::load()?;
			let graph = materialize_saved_graph(stored, samples, device, config)?;
			let mut tape = NativeTape::new(&graph, samples, &[], device, stored.precision, None)?;
			tape.inject_bn_stats(&stored.bn_stats)?;
			tape.forward()?;
			tape.predictions()
		});
		result.unwrap_or_else(|error| panic!("{error}"))
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
	Expand = 12,
	Read = 13,
	Outer = 14,
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
	fn identity(&self, index: usize) -> String {
		let prim = match self.op {
			Primitive::Contraction => "Contraction",
			Primitive::Pool => "Pool",
			Primitive::Attention => "Attention",
			Primitive::Scan => "Scan",
			Primitive::Elementwise => "Elementwise",
			Primitive::Normalize => "Normalize",
			Primitive::Predictor => "Predictor",
			Primitive::Expand => "Expand",
			Primitive::Read => "Read",
			Primitive::Outer => "Outer",
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
}
impl Graph {
	fn new(shape: Shape) -> Self {
		Self {
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
		}
	}
	fn refresh_storage(&mut self, config: Config) -> Result<()> {
		encode_graph_storage(self, config)
	}
}
fn encode_graph_storage(graph: &mut Graph, config: Config) -> Result<()> {
	require(graph.stored.len() == graph.nodes.len(), "model graph storage spans are incomplete")?;
	for (index, node) in graph.nodes.iter().enumerate() {
		if node.parameters == 0 || node.argument[8] == 0.0 {
			graph.stored[index] = None;
			continue;
		}
		let format = StorageFormat(node.argument[8] as u16);
		require(format.spec().is_some(), format.unavailable().to_string())?;
		let weights = &graph.parameters[node.offset..node.offset + node.parameters];
		// A node that never received gradient, like an unrouted expert, has an all-zero
		// variance slice: it carries no importance signal, so weight it uniformly.
		let importance = graph
			.state
			.variances
			.get(node.offset..node.offset + node.parameters)
			.filter(|values| values.len() == node.parameters && values.iter().any(|value| *value > 0.0))
			.map_or_else(|| vec![1.0; node.parameters], |values| values.to_vec());
		graph.stored[index] = Some(format.encode(weights, &importance, config)?);
	}
	Ok(())
}
fn compile(model: &Model, data: &Prepared, targets: &[f64], rows: usize, gpu: &'static Gpu, config: Config, initialize: bool) -> Result<Graph> {
	require(!model.blocks.is_empty(), "model must contain a block")?;
	if let Some(format) = model.blocks.iter().map(|block| StorageFormat(block.quantization)).find(|format| format.0 != 0 && !format.valid()) {
		return Err(format.unavailable());
	}
	let sequence = data.sequence.map(|(sequence, attention)| if matches!(model.blocks[0].operation, Operation::Attention(_)) { attention } else { sequence });
	let sequential = matches!(model.blocks[0].operation, Operation::Conv(..) | Operation::Pool(..)) || sequence.is_some() && matches!(model.blocks[0].operation, Operation::Attention(_));
	let shape = if sequential { sequence.unwrap_or(Shape { channels: 1, length: data.features }) } else { Shape { channels: data.features, length: 1 } };
	let mut graph = Graph::new(shape);
	for (index, block) in model.blocks.iter().enumerate() {
		graph.block_index = index;
		graph.block_kind = block.operation.name();
		lower_block(&mut graph, block, model.blocks.len(), data, targets, rows, gpu, config)?;
	}
	if graph.lanes != 0 {
		lower_collapse(&mut graph)?;
	}
	let mut output_profile = model.blocks.last().filter(|block| block.profile).map(|block| StorageFormat(block.quantization));
	// A model whose last block already emits one value per target needs no projection; the
	// channel and length are checked separately because a matching element count can still
	// be the wrong shape for the projection's bias.
	if graph.output.channels != data.target_width || graph.output.length != 1 {
		let length = graph.output.length;
		lower_conv(&mut graph, data.target_width, length)?;
		if model.quantization != 0 {
			graph.nodes.last_mut().unwrap().argument[8] = f64::from(model.quantization)
		}
		output_profile = StorageFormat(model.quantization).selection().map(|_| StorageFormat(model.quantization));
	}
	if let Some(format) = output_profile
		&& let Some(node) = graph.nodes.iter_mut().rev().find(|node| node.op != Primitive::Predictor && node.parameters != 0 && node.block_index + 1 == model.blocks.len())
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
	};
	let mut graph = compile(&saved.model, &prepared, &prepared.targets, 1, gpu, config, false)?;
	require(graph.input == saved.input, "saved semantic input shape does not match the compiled model")?;
	require(graph.output == saved.output, "saved semantic output shape does not match the compiled model")?;
	require(graph.parameters.len() == saved.tensors.iter().map(|tensor| tensor.count).sum::<usize>(), "saved semantic weights do not match the compiled model")?;
	let mut tensor = 0;
	for (index, node) in graph.nodes.iter().enumerate() {
		if node.parameters == 0 {
			continue;
		}
		let encoded = saved.tensors.get(tensor).ok_or_else(|| RecipeError::new("saved semantic tensor is absent"))?;
		require(encoded.count == node.parameters, "saved semantic tensor has the wrong shape")?;
		graph.parameters[node.offset..node.offset + node.parameters].copy_from_slice(&encoded.arithmetic);
		if let Some(slot) = graph.stored.get_mut(index) {
			*slot = (encoded.format.0 != 0).then_some(encoded.clone())
		}
		tensor += 1;
	}
	require(tensor == saved.tensors.len(), "saved semantic tensors are incomplete")?;
	graph.frozen = saved.frozen.clone();
	graph.state = saved.state.clone();
	Ok(graph)
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
	if graph.lanes != 0 && !matches!(block.operation, Operation::Hyper(..)) {
		lower_collapse(graph)?;
	}
	let skip = graph.source;
	let first = graph.nodes.len();
	match &block.operation {
		Operation::Layer(width) | Operation::Perceptron(width) => lower_project(graph, *width)?,
		Operation::Conv(f, k) => lower_conv(graph, *f, *k)?,
		Operation::Pool(size) => lower_pool(graph, *size)?,
		Operation::Attention(heads) => lower_attention(graph, *heads)?,
		Operation::Rnn(width) => lower_scan(graph, *width, 1)?,
		Operation::Gru(width) => lower_scan(graph, *width, 3)?,
		Operation::Lstm(width) => lower_scan(graph, *width, 4)?,
		Operation::Residual(parts) => lower_residual(graph, parts, skip, config)?,
		Operation::Moe(top_k, experts) => lower_moe(graph, *top_k, experts, config)?,
		Operation::Hyper(lanes, rank, parts) => lower_hyper(graph, *lanes, *rank, parts, config)?,
		Operation::Estimator(estimator) => {
			initialize_graph(graph, config);
			lower_estimator(graph, estimator, data, targets, rows, gpu, config)?
		}
	}
	if block.activation != Activation::Linear {
		lower_activation(graph, block.activation, config)?;
	}
	if let Some(normalization) = block.normalization {
		let epsilon = number("normalization epsilon", env!("RECIPE_NORMALIZATION_EPSILON"))?;
		push_node(graph, Primitive::Normalize, graph.output, 0, arguments(normalization as u8 as f64, epsilon), -2)?;
	}
	if block.quantization != 0 {
		let more = graph.block_index < total / 8 || graph.block_index >= 7 * total / 8 || (graph.block_index - total / 8) % 3 == 2;
		let mut parameter = 0;
		for node in &mut graph.nodes[first..] {
			if node.op != Primitive::Predictor && node.parameters != 0 {
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
	let (source, offset) = (graph.source, graph.parameters.len());
	graph.parameters.resize(checked_add(offset, parameters, "model parameters")?, 0.0);
	graph.frozen.resize(graph.parameters.len(), 0);
	graph.nodes.push(Node {
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
	});
	graph.stored.push(None);
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
fn lower_project(graph: &mut Graph, channels: usize) -> Result<()> {
	require(channels != 0, "layer width must be positive")?;
	let (parameters, output) = (checked_add(checked_mul(graph.output.channels, channels, "projection matrix")?, channels, "projection bias")?, Shape { channels, length: graph.output.length });
	push_node(graph, Primitive::Contraction, output, parameters, [0.0; 9], -2)
}
fn lower_conv(graph: &mut Graph, filters: usize, kernel: usize) -> Result<()> {
	require(filters != 0 && kernel != 0, "convolution dimensions must be positive")?;
	require(kernel <= graph.output.length, "convolution kernel exceeds sequence length")?;
	let parameters = checked_add(checked_mul(filters, checked_mul(graph.output.channels, kernel, "convolution window")?, "conv matrix")?, filters, "conv bias")?;
	let output = Shape { channels: filters, length: graph.output.length - kernel + 1 };
	push_node(graph, Primitive::Contraction, output, parameters, arguments(kernel as f64, 0.0), -2)
}
fn output_bias_offset(graph: &Graph) -> Option<usize> {
	graph.nodes.iter().rev().find(|node| node.op == Primitive::Contraction).map(|node| node.offset + node.parameters - node.output.channels)
}
fn lower_pool(graph: &mut Graph, size: usize) -> Result<()> {
	require(size != 0, "pool window must be positive")?;
	let output = Shape { channels: graph.output.channels, length: graph.output.length.div_ceil(size) };
	push_node(graph, Primitive::Pool, output, 0, arguments(size as f64, 0.0), -2)
}
fn lower_attention(graph: &mut Graph, heads: usize) -> Result<()> {
	require(heads != 0 && graph.output.channels % heads == 0, "attention head partition is invalid")?;
	let input = graph.output;
	lower_project(graph, checked_mul(input.channels, 3, "attention QKV projection width")?)?;
	let width = input.channels / heads;
	push_node(graph, Primitive::Attention, input, 0, [heads as f64, heads as f64, width as f64, 0.0, 0.0, (width as f64).sqrt(), 0.0, 0.0, 0.0], -2)?;
	lower_project(graph, input.channels)
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
fn constant(graph: &mut Graph, source: i32, shape: Shape, value: f64) -> Result<i32> {
	let mut scalar = ScalarProgram(Vec::new());
	scalar.constant(value);
	program(graph, source, -2, shape, &[], scalar)
}
fn activation(graph: &mut Graph, source: i32, shape: Shape, value: Activation, config: Config) -> Result<(i32, Shape)> {
	reset(graph, source, shape);
	if value != Activation::Linear {
		lower_activation(graph, value, config)?;
	}
	Ok((graph.source, graph.output))
}
fn expert(graph: &mut Graph, source: i32, shape: Shape, value: &Residual, config: Config) -> Result<(i32, Shape)> {
	reset(graph, source, shape);
	match value {
		Residual::Layer(width) => lower_project(graph, *width)?,
		Residual::Conv(filters, kernel) => lower_conv(graph, *filters, *kernel)?,
		Residual::Activation(value) if *value != Activation::Linear => lower_activation(graph, *value, config)?,
		Residual::Activation(_) => {}
	}
	Ok((graph.source, graph.output))
}
fn maximum(graph: &mut Graph, first: i32, second: i32, shape: Shape) -> Result<i32> {
	let mut scalar = ScalarProgram(Vec::new());
	let condition = scalar.op(ScalarOpcode::Greater, -1.0, -2.0);
	scalar.choose(condition, -1.0, -2.0);
	program(graph, first, second, shape, &[], scalar)
}
fn one_minus(graph: &mut Graph, source: i32, shape: Shape) -> Result<i32> {
	let mut scalar = ScalarProgram(Vec::new());
	let one = scalar.constant(1.0);
	scalar.op(ScalarOpcode::Subtract, one, -1.0);
	program(graph, source, -2, shape, &[], scalar)
}
fn greater_than(graph: &mut Graph, value: f64, source: i32, shape: Shape) -> Result<i32> {
	let mut scalar = ScalarProgram(Vec::new());
	let limit = scalar.constant(value);
	scalar.op(ScalarOpcode::Greater, limit, -1.0);
	program(graph, source, -2, shape, &[], scalar)
}
fn rank_mask(graph: &mut Graph, scores: &[i32], selected: usize, shape: Shape, top_k: usize) -> Result<i32> {
	let mut rank = constant(graph, scores[selected], shape, 0.0)?;
	for candidate in 0..scores.len() {
		if candidate == selected {
			continue;
		}
		let higher = binary(graph, scores[candidate], scores[selected], shape, ScalarOpcode::Greater)?;
		let order = if candidate < selected {
			let lower = binary(graph, scores[selected], scores[candidate], shape, ScalarOpcode::Greater)?;
			let unequal = binary(graph, higher, lower, shape, ScalarOpcode::Add)?;
			let tied = one_minus(graph, unequal, shape)?;
			binary(graph, higher, tied, shape, ScalarOpcode::Add)?
		} else {
			higher
		};
		rank = binary(graph, rank, order, shape, ScalarOpcode::Add)?;
	}
	greater_than(graph, top_k as f64, rank, shape)
}
fn select(graph: &mut Graph, branches: &[i32], scores: &[i32], shape: Shape, top_k: usize, config: Config) -> Result<()> {
	let mut maximum_score = scores[0];
	for &score in &scores[1..] {
		maximum_score = maximum(graph, maximum_score, score, shape)?;
	}
	let mut weighted = Vec::with_capacity(scores.len());
	for (index, &score) in scores.iter().enumerate() {
		let centered = binary(graph, score, maximum_score, shape, ScalarOpcode::Subtract)?;
		let exponential = activation(graph, centered, shape, Activation::Exp, config)?.0;
		let mask = rank_mask(graph, scores, index, shape, top_k)?;
		weighted.push(binary(graph, mask, exponential, shape, ScalarOpcode::Multiply)?);
	}
	let mut denominator = weighted[0];
	for &value in &weighted[1..] {
		denominator = binary(graph, denominator, value, shape, ScalarOpcode::Add)?;
	}
	let mut output = None;
	for (index, &branch) in branches.iter().enumerate() {
		let probability = binary(graph, weighted[index], denominator, shape, ScalarOpcode::Divide)?;
		let routed = binary(graph, probability, branch, shape, ScalarOpcode::Multiply)?;
		output = Some(match output {
			Some(previous) => binary(graph, previous, routed, shape, ScalarOpcode::Add)?,
			None => routed,
		});
	}
	reset(graph, output.ok_or_else(|| RecipeError::new("selection has no output"))?, shape);
	Ok(())
}
fn lower_moe(graph: &mut Graph, top_k: usize, experts: &[Residual], config: Config) -> Result<()> {
	require(!experts.is_empty(), "moe requires an expert")?;
	require(top_k != 0 && top_k <= experts.len(), "moe top-k is invalid")?;
	let (source, input, mut branches) = (graph.source, graph.output, Vec::with_capacity(experts.len()));
	let mut output = None;
	for value in experts {
		let (branch, shape) = expert(graph, source, input, value, config)?;
		if let Some(expected) = output {
			require(shape == expected, "moe experts must have one output shape")?;
		}
		output = Some(shape);
		branches.push(branch);
	}
	let output = output.ok_or_else(|| RecipeError::new("moe has no output shape"))?;
	let mut scores = Vec::with_capacity(experts.len());
	for _ in experts {
		reset(graph, source, input);
		lower_project(graph, output.channels)?;
		require(graph.output == output, "moe router shape does not match its experts")?;
		scores.push(graph.source);
	}
	select(graph, &branches, &scores, output, top_k, config)
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
fn lower_hyper(graph: &mut Graph, lanes: usize, rank: usize, parts: &[Residual], config: Config) -> Result<()> {
	require(lanes != 0 && !parts.is_empty(), "hyper-connections need at least one lane and one operation")?;
	if graph.lanes == 0 {
		let shape = graph.output;
		push_node(graph, Primitive::Expand, Shape { channels: checked_mul(shape.channels, lanes, "hyper-connection stream")?, length: shape.length }, 0, arguments(lanes as f64, 0.0), -2)?;
		graph.lanes = lanes;
	}
	require(graph.lanes == lanes, format!("hyper-connections with {lanes} lanes follow a stream of {}", graph.lanes))?;
	graph.rank = rank;
	let (stream, shape) = (graph.source, graph.output);
	let width = shape.channels / lanes;
	let (read, write) = lower_gates(graph, lanes, rank, true)?;
	reset(graph, stream, shape);
	push_node(graph, Primitive::Read, Shape { channels: width, length: shape.length }, 0, arguments(lanes as f64, 0.0), read)?;
	for part in parts {
		match part {
			Residual::Layer(width) => lower_project(graph, *width)?,
			Residual::Conv(filters, kernel) => lower_conv(graph, *filters, *kernel)?,
			Residual::Activation(activation) => lower_activation(graph, *activation, config)?,
		}
	}
	require(graph.output.channels == width && graph.output.length == shape.length, "hyper-connection branch shape mismatch")?;
	push_node(graph, Primitive::Outer, shape, 0, arguments(lanes as f64, 0.0), write)?;
	let mut program = ScalarProgram(Vec::new());
	program.op(ScalarOpcode::Add, -1.0, -2.0);
	push_program(graph, stream, &[], program)
}
/// Data-dependent gates from the layer-normalized stream: a read gate the width
/// of the stream through a bottleneck of `rank`, and one write gate per lane,
/// both offset by one so a fresh model starts at the plain residual. With
/// `rank` zero every gate is one and no node is added.
fn lower_gates(graph: &mut Graph, lanes: usize, rank: usize, write: bool) -> Result<(i32, i32)> {
	if rank == 0 {
		return Ok((-2, -2));
	}
	let (stream, shape) = (graph.source, graph.output);
	let epsilon = number("normalization epsilon", env!("RECIPE_NORMALIZATION_EPSILON"))?;
	push_node(graph, Primitive::Normalize, shape, 0, arguments(1.0, epsilon), -2)?;
	let normalized = graph.source;
	lower_project(graph, rank)?;
	lower_project(graph, shape.channels)?;
	lower_offset_one(graph)?;
	let read = graph.source;
	if !write {
		reset(graph, stream, shape);
		return Ok((read, -2));
	}
	reset(graph, normalized, shape);
	lower_project(graph, lanes)?;
	lower_offset_one(graph)?;
	let write = graph.source;
	reset(graph, stream, shape);
	Ok((read, write))
}
fn lower_offset_one(graph: &mut Graph) -> Result<()> {
	let mut program = ScalarProgram(Vec::new());
	let one = program.constant(1.0);
	program.op(ScalarOpcode::Add, -1.0, one);
	push_program(graph, -2, &[], program)
}
/// The head read: the stream collapses to the width through its own read gate.
fn lower_collapse(graph: &mut Graph) -> Result<()> {
	let (lanes, rank, stream, shape) = (graph.lanes, graph.rank, graph.source, graph.output);
	let (read, _) = lower_gates(graph, lanes, rank, false)?;
	reset(graph, stream, shape);
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
fn initialize_graph(graph: &mut Graph, config: Config) {
	let mut state = config.random_seed as u64;
	for node in &graph.nodes {
		if node.op == Primitive::Elementwise {
			continue;
		}
		let fan_in = (node.parameters / node.output.channels.max(1)).max(1) as f64;
		let scale = config.initial / fan_in.sqrt();
		for index in node.offset..node.offset + node.parameters {
			if graph.frozen[index] == 0 {
				state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
				graph.parameters[index] = ((state >> 11) as f64 / ((1_u64 << 53) as f64) * 2.0 - 1.0) * scale;
			}
		}
		if node.op == Primitive::Contraction {
			graph.parameters[node.offset + node.parameters - node.output.channels..node.offset + node.parameters].fill(0.0);
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
struct NativeTape {
	program: NativeProgram,
	precision: NativePrecision,
	values: Buffer,
	contexts: Buffer,
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
	output: usize,
	capacity: usize,
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
	fn new(graph: &Graph, samples: &[f64], targets: &[f64], gpu: &'static Gpu, precision: Compute, loss: Option<LossFunction>) -> Result<Self> {
		let input = graph.input.elements();
		require(input != 0 && !samples.is_empty() && samples.len() % input == 0, format!("model input batch expected a nonempty multiple of {input} values, received {}", samples.len()))?;
		let rows = samples.len() / input;
		let output = graph.output.elements();
		require(targets.is_empty() || targets.len() == rows * output, format!("target batch expected 0 or {} values, received {}", rows * output, targets.len()))?;
		let program = gpu.native_program(graph, rows, precision, loss)?;
		let (precision, layout, parameters) = (program.artifact.precision, program.artifact.layout.clone(), graph.parameters.len());
		let zeros = vec![0.0; parameters.max(1)];
		let gradient_bytes = checked_mul(program.gradient_values.max(1), precision.model.bytes(), "native gradient allocation")?;
		require(graph.state.moments.is_empty() || graph.state.moments.len() == parameters, "saved optimizer moments have the wrong shape")?;
		require(graph.state.variances.is_empty() || graph.state.variances.len() == parameters, "saved optimizer variances have the wrong shape")?;
		require(graph.frozen.is_empty() || graph.frozen.len() == parameters, "frozen parameters have the wrong shape")?;
		let moments = if graph.state.moments.is_empty() { zeros.clone() } else { graph.state.moments.clone() };
		let variances = if graph.state.variances.is_empty() { zeros.clone() } else { graph.state.variances.clone() };
		let frozen = if graph.frozen.is_empty() { vec![0_u8; parameters.max(1)] } else { graph.frozen.clone() };
		let batch_normalizations =
			graph.nodes.iter().enumerate().filter_map(|(index, node)| (node.op == Primitive::Normalize && node.argument[0] == 0.0).then_some((index, node.output.channels))).collect();
		let best_loss = if graph.state.best_loss.is_empty() {
			[f64::INFINITY, f64::NAN, f64::NAN, f64::INFINITY]
		} else {
			graph.state.best_loss.as_slice().try_into().map_err(|_| RecipeError::new("saved loss state is invalid"))?
		};
		let step = narrow(graph.state.epoch, "optimizer epoch")? as u32;
		let target_buffer = if targets.is_empty() { vec![0.0] } else { targets.to_vec() };
		let parameter_values = if graph.parameters.is_empty() { vec![0.0] } else { graph.parameters.clone() };
		let adjoints_bytes = layout.adjoints_bytes.max(1);
		let input_adjoint_bytes = checked_mul(samples.len(), precision.model.bytes(), "native input adjoint allocation")?.max(1);
		let weights = Buffer::upload_float(gpu, &parameter_values, precision.model)?;
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
		Ok(Self {
			program,
			precision,
			values: Buffer::upload(gpu, &vec![0_u8; layout.values_bytes.max(1)])?,
			contexts: Buffer::upload(gpu, &vec![0_u8; layout.contexts_bytes.max(1)])?,
			adjoints: Buffer { runtime: gpu, pointer: gpu.allocate(adjoints_bytes)?, bytes: adjoints_bytes },
			batch_normalizations,
			samples: Buffer::upload_float(gpu, samples, precision.model)?,
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
			output,
			capacity: rows,
		})
	}
	fn forward(&mut self) -> Result<()> {
		let threads = self.program.forward.geometry.threads()?;
		let rows = self.rows;
		let thread_count = threads;
		let mut call = ptrs![self.samples.pointer, self.weights.pointer, self.values.pointer, self.contexts.pointer, rows, thread_count];
		self.program.launch_forward(&mut call).map_err(|error| RecipeError::new(format!("forward: {error}")))?;
		Ok(())
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
		let offset = *self.program.artifact.layout.values.last().ok_or_else(|| RecipeError::new("native model has no output arena"))?;
		let values = self.values.download_float_bytes(offset, self.capacity * self.output, self.precision.model)?;
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
	};
	let graph = compile(&surrogate_model(config.surrogate_width), &prepared, &targets, rows, gpu, config, true)?;
	let mut tape = NativeTape::new(&graph, &samples, &targets, gpu, config.precision, Some(mse))?;
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
			shards.push(NativeTape::new(graph, &samples[start * input..end * input], &targets[start * output..end * output], gpus[*device], precision, Some(loss))?);
			start = end;
		}
		Ok(Self { shards, placement })
	}
	fn forward(&mut self) -> Result<()> {
		self.shards.iter_mut().try_for_each(NativeTape::forward)
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
	fn print_devices(&self) -> Result<()> {
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
fn node_context(node: &Node, rows: usize, element: usize) -> Result<usize> {
	let elements = match node.op {
		// One scratch row per reduction partition, holding this node's trainable
		// scalars. Programs without trainable scalars reduce nothing and take the
		// minimum allocation below.
		Primitive::Elementwise => checked_mul(checked_mul(rows, node.output.elements(), "program batch")?.min(NATIVE_SCALAR_PARTITIONS), node.parameters, "scalar gradient partials")?,
		Primitive::Predictor => checked_mul(checked_add(node.argument[0] as usize, node.argument[1] as usize, "predictor workspace")?, rows, "predictor batch")?,
		Primitive::Attention => {
			checked_mul(checked_mul(checked_mul(rows, node.output.length, "attention statistics rows")?, node.argument[0] as usize, "attention statistics heads")?, 2, "attention statistics")?
		}
		Primitive::Scan => {
			let (state_count, gates) = (checked_mul(rows, node.output.elements(), "scan batch")?, node.argument[0] as usize);
			let states = checked_mul(2 * gates + 1, state_count, "scan states")?;
			let gradients = checked_mul(rows, node.parameters, "scan gradients")?;
			checked_add(states, checked_add(gradients, 2 * rows * node.output.channels, "scan scratch")?, "scan")?
		}
		Primitive::Pool => return checked_mul(checked_mul(rows, node.output.elements(), "pool context")?, size_of::<u64>(), "pool context bytes"),
		Primitive::Normalize => {
			let groups = node.output.channels.max(checked_mul(rows, node.output.length, "layer groups")?);
			checked_mul(4, groups, "normalization context")?
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
		let bytes = precision.bytes();
		let encoded = values.iter().flat_map(|value| precision.pack(*value).to_le_bytes().into_iter().take(bytes)).collect::<Vec<_>>();
		Self::upload(runtime, &encoded)
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
type NativeForward = unsafe extern "C" fn(Ptr, Ptr, Ptr, Ptr, i32, i32);
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
	allocate: unsafe extern "C" fn(u64, usize, u32, *mut Ptr) -> i32,
	free: unsafe extern "C" fn(Ptr) -> i32,
	allow: unsafe extern "C" fn(u32, *const u64, *const u32, *const c_void) -> i32,
	copy: unsafe extern "C" fn(Ptr, *const c_void, usize) -> i32,
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
			Ok::<_, RecipeError>(aligned && node.output.channels / heads as usize % fragment_k as usize == 0)
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
fn devices() -> Result<&'static [Gpu]> {
	DEVICES
		.get_or_init(|| {
			if std::env::var_os("RECIPE_FORCE_CPU").is_some() {
				return cpu_device().map(|gpu| vec![gpu]);
			}
			let mut found = Vec::new();
			let mut errors = Vec::new();
			for load in [load_amd as fn() -> Result<Vec<Gpu>>, load_nvidia] {
				match load() {
					Ok(mut devices) => found.append(&mut devices),
					Err(error) => errors.push(error.to_string()),
				}
			}
			if found.is_empty() {
				if cfg!(any(amd, nvidia)) {
					return Err(RecipeError::new(errors.join("; ")));
				}
				found.push(cpu_device()?);
			}
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
	require(found.len() == 1, "multiple GPUs require named selection")?;
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

fn load_amd() -> Result<Vec<Gpu>> {
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
		gpu.found.into_iter().enumerate().map(|(index, agent)| load_amd_gpu(&runtime, info, cpu.found, agent, index)).collect()
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
			allocate,
			allow,
			queue,
			cpu_agent,
			free: runtime.function(b"hsa_amd_memory_pool_free\0")?,
			copy: runtime.function(b"hsa_memory_copy\0")?,
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
fn load_nvidia() -> Result<Vec<Gpu>> {
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
		let mut found = Vec::new();
		for ordinal in 0..count {
			let (mut gpu, mut integrated) = (0, 0);
			check(get_device(&mut gpu, ordinal), "device enumeration")?;
			check(attribute(&mut integrated, INTEGRATED, gpu), "device probe")?;
			if integrated == 0 {
				found.push(load_device(gpu, found.len())?)
			}
		}
		require(!found.is_empty(), "Nvidia has no discrete GPU")?;
		Ok(found)
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
	let mut tape = NativeTape::new(graph, &samples[..input_count], &[], gpu, precision, None)?;
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
	};
	let mut graph = compile(&model, &prepared, targets, prepared.rows, gpu, config, true)?;
	let mut tape = NativeTape::new(&graph, samples, targets, gpu, config.precision, Some(mse))?;
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
	Ok(Prepared { samples, targets, target_width, rows, source_rows, features, schema, sequence, target_categorical, norm_mean, norm_scale, identities, fitted: Vec::new() })
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
		tape.print_devices()?;
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
				let mut validation = NativeTape::new(&graph, sample, &[], gpu, config.precision, None)?;
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
			let mut validation = NativeTape::new(&graph, &prepared.samples[start..], validation_targets, gpu, config.precision, None)?;
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
