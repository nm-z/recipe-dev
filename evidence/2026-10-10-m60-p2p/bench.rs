use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::time::Instant;
type Handle = *mut c_void;
type Ptr = u64;
#[link(name = "cuda")]
unsafe extern "C" {
	fn cuInit(flags: c_uint) -> c_int;
	fn cuDeviceGetCount(n: *mut c_int) -> c_int;
	fn cuDeviceGet(d: *mut c_int, index: c_int) -> c_int;
	fn cuDeviceGetUuid(uuid: *mut [u8; 16], dev: c_int) -> c_int;
	fn cuDeviceCanAccessPeer(can: *mut c_int, a: c_int, b: c_int) -> c_int;
	fn cuDevicePrimaryCtxRetain(ctx: *mut Handle, dev: c_int) -> c_int;
	fn cuCtxSetCurrent(ctx: Handle) -> c_int;
	fn cuCtxEnablePeerAccess(ctx: Handle, flags: c_uint) -> c_int;
	fn cuCtxSynchronize() -> c_int;
	fn cuDevicePrimaryCtxRelease_v2(dev: c_int) -> c_int;
	fn cuMemAlloc_v2(ptr: *mut Ptr, bytes: usize) -> c_int;
	fn cuMemsetD32_v2(ptr: Ptr, value: c_uint, n: usize) -> c_int;
	fn cuMemcpyDtoH_v2(dst: *mut c_void, src: Ptr, bytes: usize) -> c_int;
	fn cuMemcpyHtoD_v2(dst: Ptr, src: *const c_void, bytes: usize) -> c_int;
	fn cudaMemcpyPeer(dst: *mut c_void, dst_dev: c_int, src: *const c_void, src_dev: c_int, bytes: usize) -> c_int;
	fn cudaGetErrorString(err: c_int) -> *const c_char;
	fn cuModuleLoad(module: *mut Handle, name: *const c_char) -> c_int;
	fn cuModuleGetFunction(fun: *mut Handle, module: Handle, name: *const c_char) -> c_int;
	fn cuLaunchKernel(fun: Handle, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
		shared: c_uint, stream: Handle, args: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
	fn cuOccupancyMaxActiveBlocksPerMultiprocessor(n: *mut c_int, fun: Handle, block: c_int, shared: usize) -> c_int;
	fn cuGetErrorString(err: c_int, text: *mut *const c_char) -> c_int;
}
fn check(code: c_int) {
	if code != 0 {
		let mut msg = std::ptr::null();
		unsafe { cuGetErrorString(code, &mut msg); }
		panic!("CUDA {code}: {}", unsafe { CStr::from_ptr(msg) }.to_string_lossy());
	}
}
fn arg<T>(v: &mut T) -> *mut c_void { (v as *mut T).cast() }
#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Report { start_ns: u64, end_ns: u64, errors: u32, timeout: u32, sent: u32, received: u32, max_abs: f32, max_rel: f32 }
struct Die { dev: c_int, ctx: Handle, packet: Ptr, report: Ptr, relay: Handle }
impl Die {
	fn select(&self) { unsafe { check(cuCtxSetCurrent(self.ctx)); } }
	fn alloc(&self, n: usize) -> Ptr {
		self.select();
		let mut ptr = 0;
		unsafe { check(cuMemAlloc_v2(&mut ptr, n)); }
		ptr
	}
	fn read<T: Default>(&self, ptr: Ptr) -> T {
		self.select();
		let mut out = T::default();
		unsafe { check(cuMemcpyDtoH_v2(arg(&mut out), ptr, size_of::<T>())); }
		out
	}
	fn launch(&self, peer: &Die, mut die: u32, mut hops: u32, mut base: u32, mut seed: u32, mut delay: u32, mut skip: u32) {
		self.select();
		let (mut local, mut remote, mut report) = (self.packet, peer.packet, self.report);
		let mut args = [arg(&mut local), arg(&mut remote), arg(&mut report), arg(&mut die), arg(&mut hops), arg(&mut base),
			arg(&mut seed), arg(&mut delay), arg(&mut skip)];
		unsafe { check(cuLaunchKernel(self.relay, 1, 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut())); }
	}
	fn sync(&self) { self.select(); unsafe { check(cuCtxSynchronize()); } }
}
fn setup() -> [Die; 2] {
	let expected = ["ccb8b3a3f45d8962215bc2b140e4bb28", "c767d2504454d181d6488706b5f2fb64"];
	unsafe {
		check(cuInit(0));
		let mut n = 0;
		check(cuDeviceGetCount(&mut n));
		assert_eq!(n, 2, "Set CUDA_VISIBLE_DEVICES to the two specified UUIDs");
		let devices: Vec<Die> = (0..2).map(|i| {
			let (mut dev, mut uuid) = (0, [0_u8; 16]);
			check(cuDeviceGet(&mut dev, i));
			check(cuDeviceGetUuid(&mut uuid, dev));
			let actual: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
			assert_eq!(actual, expected[i as usize], "Unexpected GPU UUID; no context created");
			println!("die={i} uuid_hex={actual}");
			let mut ctx = std::ptr::null_mut();
			check(cuDevicePrimaryCtxRetain(&mut ctx, dev));
			check(cuCtxSetCurrent(ctx));
			let (mut module, mut relay) = (std::ptr::null_mut(), std::ptr::null_mut());
			check(cuModuleLoad(&mut module, c"../../barrier.cubin".as_ptr()));
			check(cuModuleGetFunction(&mut relay, module, c"relay".as_ptr()));
			let mut d = Die { dev, ctx, packet: 0, report: 0, relay };
			d.packet = d.alloc(16452);
			d.report = d.alloc(size_of::<Report>());
			check(cuMemsetD32_v2(d.packet, 0, 16452 / 4));
			d
		}).collect();
		let d: [Die; 2] = devices.try_into().ok().unwrap();
		for i in 0..2 {
			let mut can = 0;
			check(cuDeviceCanAccessPeer(&mut can, i as i32, (1 - i) as i32));
			assert_eq!(can, 1, "Peer mapping is required");
			d[i].select();
			check(cuCtxEnablePeerAccess(d[1-i].ctx, 0));
		}
		d
	}
}
fn relay(d: &[Die; 2], hops: u32, token: u32, delay: u32, skip: u32) -> (f64, [Report; 2]) {
	let start = Instant::now();
	// Queue the receiving die first so its local spin is ready before the first publication.
	d[1].launch(&d[0], 1, hops, token * 128, token * 1024, delay, skip);
	d[0].launch(&d[1], 0, hops, token * 128, token * 1024, delay, skip);
	d[0].sync();
	d[1].sync();
	let us = start.elapsed().as_secs_f64() * 1e6;
	(us, [d[0].read(d[0].report), d[1].read(d[1].report)])
}
fn baseline(d: &[Die; 2], bytes: usize, hops: u32, trials: u32) {
	let input: Vec<u32> = (0..4096).map(|i| 17 + i).collect();
	d[0].select();
	unsafe { check(cuMemcpyHtoD_v2(d[0].packet, input.as_ptr().cast(), bytes)); }
	let mut sum = 0.0;
	for t in 0..(trials + 20) {
		let start = Instant::now();
		for h in 0..hops {
			let s = h as usize & 1;
			d[s].select();
			unsafe { let code = cudaMemcpyPeer(d[1-s].packet as *mut c_void, d[1-s].dev, d[s].packet as *const c_void, d[s].dev, bytes); if code != 0 { panic!("cudaMemcpyPeer {code}: {}", CStr::from_ptr(cudaGetErrorString(code)).to_string_lossy()); } }
			d[1-s].sync();
		}
		if t >= 20 { sum += start.elapsed().as_secs_f64() * 1e6; }
	}
	let dst = (hops & 1) as usize;
	let mut out = vec![0_u32; bytes / 4];
	d[dst].select();
	unsafe { check(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), d[dst].packet, bytes)); }
	assert_eq!(out, input[..bytes / 4]);
	println!("cudaMemcpyPeer bytes={bytes} hops={hops} trials={trials} total_us={sum:.3} token_us={:.3} hop_us={:.3} errors=0", sum / trials as f64, sum / (trials * hops) as f64);
}
fn main() {
	let d = setup();
	if std::env::args().any(|arg| arg == "--multi") { multi_demo(); return; }
	let trials = 200;
	let mut token = 1;
	let mut gate = true;
	for hops in [1, 7, 100] {
		let (mut machine, mut device, mut errors) = (0.0, 0.0, 0_u64);
		let mut samples = Vec::new();
		for t in 0..(trials + 20) {
			let (us, r) = relay(&d, hops, token, 0, 0);
			token += 1;
			for i in 0..2 {
				assert_eq!(r[i].timeout, 0, "die {i} timed out");
				assert_eq!(r[i].sent, (hops + (1-i as u32)) / 2);
				assert_eq!(r[i].received, (hops + i as u32) / 2);
				errors += r[i].errors as u64;
			}
			if t >= 20 { machine += us; device += (r[0].end_ns - r[0].start_ns) as f64 / 1000.0; samples.push(us); }
		}
		assert_eq!(errors, 0, "payload corruption");
		samples.sort_by(f64::total_cmp);
		let hop_us = machine / (trials * hops) as f64;
		if hops > 1 { gate &= hop_us < 10.0; }
		println!("relay bytes=16384 hops={hops} trials={trials} machine_total_us={machine:.3} token_us={:.3} hop_us={hop_us:.3} device_token_us={:.3} device_hop_us={:.3} p50_token_us={:.3} p99_token_us={:.3} errors={errors}",
			machine / trials as f64, device / trials as f64, device / (trials * hops) as f64, samples[trials as usize/2], samples[(trials as usize*99/100)-1]);
	}
	let (_, delayed) = relay(&d, 7, token, 5000, 0);
	token += 1;
	assert!(delayed.iter().all(|r| r.errors == 0 && r.timeout == 0));
	println!("delay_publication_ns=5000 hops=7 errors=0");
	let (_, negative) = relay(&d, 1, token, 0, 1);
	assert_ne!(negative[0].timeout, 0);
	assert_ne!(negative[1].timeout, 0);
	println!("missing_sequence die0_timeout={} die1_timeout={}", negative[0].timeout, negative[1].timeout);
	for bytes in [4096, 16384] { for hops in [1, 7, 100] { baseline(&d, bytes, hops, trials); } }
	println!("gate_lt_10us={gate}");
	layer_demo(&d);
	multi_demo();
	for die in &d { die.select(); unsafe { check(cuDevicePrimaryCtxRelease_v2(die.dev)); } }
	if !gate { std::process::exit(2); }
}

fn packed_weights_rows(seed: u32, rows: usize, scale_bits: u16) -> Vec<u8> {
	let mut w = vec![0_u8; rows * 16 * 144];
	let mut state = seed;
	for block in w.chunks_exact_mut(144) {
		block[..2].copy_from_slice(&scale_bits.to_le_bytes());
		block[2..4].copy_from_slice(&scale_bits.to_le_bytes());
		for i in 0..4 { block[4+i] = 1 + ((state >> (i*2)) % 3) as u8; block[8+i] = 1 + ((state >> (i*3)) % 3) as u8; }
		for i in 0..4 { block[12+i] = (1 + ((state >> (i*2)) % 3) as u8) | ((1 + ((state >> (i*3)) % 3) as u8) << 4); }
		for q in &mut block[16..] {
			state ^= state << 13; state ^= state >> 17; state ^= state << 5;
			*q = state as u8;
		}
	}
	w
}
// Independent scalar Q4_K/Q8_1 algebra. Only block-scale application uses floating point.
fn reference(w: &[u8], x: &[f32]) -> Vec<f32> {
	let mut q = vec![0_i32; 4096];
	let mut d = vec![0_f64; 128];
	for b in 0..128 {
		let max = x[b*32..b*32+32].iter().fold(0_f32, |a, v| a.max(v.abs()));
		let raw_d = max / 127.;
		if max == 0. { continue; }
		let exp = ((raw_d.to_bits() >> 23) & 255) as i32 - 127;
		let step = 2_f32.powi((exp - 10).max(-24));
		d[b] = ((raw_d / step).round_ties_even() * step) as f64;
		for l in 0..32 { q[b*32+l] = (x[b*32+l] / raw_d).round() as i32; }
	}
	(0..w.len() / (16 * 144)).map(|row| {
		let mut acc = 0_f64;
		for sb in 0..16 {
			let block = &w[(row*16+sb)*144..(row*16+sb+1)*144];
			let wd = half_value(u16::from_le_bytes([block[0],block[1]]));
			let wm = half_value(u16::from_le_bytes([block[2],block[3]]));
			for sub in 0..8 {
				let (scale, min) = if sub < 4 { (block[4+sub] & 63, block[8+sub] & 63) }
					else { ((block[12+sub-4] & 15) | ((block[4+sub-4] >> 6) << 4), (block[12+sub-4] >> 4) | ((block[8+sub-4] >> 6) << 4)) };
				let (mut dot, mut sum) = (0_i32, 0_i32);
				for lane in 0..32 {
					let packed = block[16 + (sub/2)*32+lane];
					let weight = if sub & 1 == 0 { packed & 15 } else { packed >> 4 };
					let activation = q[sb*256+sub*32+lane];
					dot += weight as i32 * activation; sum += activation;
				}
				acc += ((scale as i32 * dot) as f64 * wd - (min as i32 * sum) as f64 * wm) * d[sb*8+sub];
			}
		}
		acc as f32
	}).collect()
}
fn upload<T>(die: &Die, v: &[T]) -> Ptr {
	let ptr = die.alloc(size_of_val(v));
	unsafe { check(cuMemcpyHtoD_v2(ptr, v.as_ptr().cast(), size_of_val(v))); }
	ptr
}
fn layer_demo(d: &[Die; 2]) {
	let input: Vec<f32> = (0..4096).map(|i| ((i * 37 + 13) % 255) as f32 - 127.).collect();
	let w = [packed_weights_rows(0x12345678, 4096, 0x0400), packed_weights_rows(0xfedcba98, 4096, 0x0400)];
	let reference0 = reference(&w[0], &input);
	let reference1 = reference(&w[1], &reference0);
	assert!(reference1.iter().all(|v| v.is_finite()));
	let expected = [reference0, reference1];
	let mut params = Vec::new();
	let mut functions = Vec::new();
	for i in 0..2 {
		d[i].select();
		let (mut module, mut fun) = (std::ptr::null_mut(), std::ptr::null_mut());
		unsafe {
			check(cuModuleLoad(&mut module, c"../../barrier.cubin".as_ptr()));
			check(cuModuleGetFunction(&mut fun, module, c"layer".as_ptr()));
		}
		functions.push(fun);
		params.push([upload(&d[i], &w[i]), d[i].alloc(8192), d[i].alloc(1024), d[i].alloc(16384), upload(&d[i], &expected[i])]);
	}
	d[0].select();
	unsafe { check(cuMemcpyHtoD_v2(d[0].packet, input.as_ptr().cast(), 16384)); }
	let trials = 8;
	let mut machine_total_us = 0.0;
	let mut device_total_us = [0.0; 2];
	let mut max_abs = [0_f32; 2];
	let mut max_rel = [0_f32; 2];
	for token in 0..(trials + 2) {
	let start = Instant::now();
	for i in [1, 0] {
		d[i].select();
		let (mut local, mut remote, mut report, mut die, mut base) = (d[i].packet, d[1-i].packet, d[i].report, i as u32, (1000000 + token * 128) as u32);
		let [mut weights, mut y16, mut dsf, mut output, mut expected] = params[i];
		let mut args = [arg(&mut local), arg(&mut remote), arg(&mut report), arg(&mut die), arg(&mut base), arg(&mut weights),
			arg(&mut y16), arg(&mut dsf), arg(&mut output), arg(&mut expected)];
		unsafe { check(cuLaunchKernel(functions[i], 1, 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut())); }
	}
	d[0].sync(); d[1].sync();
	let machine_us = start.elapsed().as_secs_f64() * 1e6;
	if token >= 2 { machine_total_us += machine_us; }
	for i in 0..2 {
		let r: Report = d[i].read(d[i].report);
		assert_eq!(r.timeout, 0); assert_eq!(r.errors, 0);
		assert_eq!(r.sent, if i == 0 { 1 } else { 0 });
		assert_eq!(r.received, if i == 1 { 1 } else { 0 });
		if token >= 2 { device_total_us[i] += (r.end_ns-r.start_ns) as f64/1000.; }
		max_abs[i] = max_abs[i].max(r.max_abs); max_rel[i] = max_rel[i].max(r.max_rel);
	}
	}
	for i in 0..2 {
		println!("layer die={i} shape=4096x4096 storage=Q4_K activation=Q8_1 accumulator=i32_then_f32 weights_bytes={} device_us={:.3} errors=0 timeout=0 max_abs={:.8} max_rel={:.8}",
			w[i].len(), device_total_us[i]/trials as f64, max_abs[i], max_rel[i]);
	}
	println!("layer_demo trials={trials} weights_uploads=2 machine_total_us={machine_total_us:.3} token_us={:.3} launches_per_token=2 peer_payload_bytes=16384 cpu_between_layers=0", machine_total_us/trials as f64);
}

fn report_vector(die: &Die, n: usize) -> Vec<Report> {
	let mut out = vec![Report::default(); n];
	die.select();
	unsafe { check(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), die.report, size_of_val(out.as_slice()))); }
	out
}
fn load_function(die: &Die, name: &CStr) -> Handle {
	die.select();
	let (mut module, mut fun, mut active) = (std::ptr::null_mut(), std::ptr::null_mut(), 0);
	unsafe {
		check(cuModuleLoad(&mut module, c"../../barrier.cubin".as_ptr()));
		check(cuModuleGetFunction(&mut fun, module, name.as_ptr()));
		check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut active, fun, 256, 0));
	}
	assert!(active >= 1, "Every CTA must be resident before it waits for other CTAs");
	println!("symbol={} active_ctas_per_sm={active}", name.to_string_lossy());
	fun
}
fn validate_grid(reports: &[Report], negative: bool) -> (f64, f32) {
	let sms: std::collections::BTreeSet<u32> = reports.iter().map(|r| r.sent).collect();
	let ctas: std::collections::BTreeSet<u32> = reports.iter().map(|r| r.received).collect();
	assert_eq!(sms.len(), 16, "All 16 SMs must participate");
	assert_eq!(ctas.len(), 16);
	assert!(reports.iter().all(|r| r.errors == 0), "Payload or packed arithmetic error: count={}, max_abs={}", reports.iter().map(|r| r.errors as u64).sum::<u64>(), reports.iter().map(|r| r.max_abs).fold(0., f32::max));
	if negative { assert!(reports.iter().any(|r| r.timeout != 0)); }
	else { assert!(reports.iter().all(|r| r.timeout == 0)); }
	let start = reports.iter().map(|r| r.start_ns).min().unwrap();
	let end = reports.iter().map(|r| r.end_ns).max().unwrap();
	((end-start) as f64 / 1000., reports.iter().map(|r| r.max_abs).fold(0., f32::max))
}
fn grid_relay(d: &[Die; 2], p: &mut [recipe::PeerPacket; 2], functions: &[Handle; 2], delay: u32, skip: u32, trials: usize) {
	let mut total = 0.;
	let mut samples = Vec::new();
	for token in 0..(trials+5) {
		let bases = [p[0].reserve_sequences(100).unwrap(), p[1].reserve_sequences(100).unwrap()];
		assert_eq!(bases[0], bases[1]);
		let start = Instant::now();
		for i in [1, 0] {
			d[i].select();
			let (mut local, mut remote, mut report, mut slots) = (p[i].address(), p[1-i].address(), d[i].report, p[i].completion_address());
			let (mut die, mut base, mut hops, mut seed, mut delay, mut skip) = (i as u32, bases[i], 100_u32, token as u32 * 123, delay, skip);
			let mut args = [arg(&mut local), arg(&mut remote), arg(&mut report), arg(&mut slots), arg(&mut die), arg(&mut base), arg(&mut hops), arg(&mut seed), arg(&mut delay), arg(&mut skip)];
			unsafe { check(cuLaunchKernel(functions[i], p[i].resident_ctas(), 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut())); }
		}
		d[0].sync(); d[1].sync();
		let us = start.elapsed().as_secs_f64()*1e6;
		for i in 0..2 { validate_grid(&report_vector(&d[i], 4096), skip != 0); }
		if token >= 5 { total += us; samples.push(us); }
		if skip != 0 {
			assert_ne!(d[1].read::<u32>(p[1].sequence_address()), bases[0]+1, "Missing CTA must prevent peer publication");
			break;
		}
	}
	if skip != 0 { println!("grid_missing_cta_slot=8 timeout_observed_on_both_dies=true"); return; }
	samples.sort_by(f64::total_cmp);
	println!("grid_relay ctas_per_die=16 active_sms=16 bytes=16384 hops=100 trials={trials} delay_cta7_ns={delay} machine_total_us={total:.3} token_us={:.3} hop_us={:.3} p99_token_us={:.3} errors=0 timeouts=0 launches_per_token=2",
		total / trials as f64, total / trials as f64 / 100., samples[(samples.len()*99/100).min(samples.len()-1)]);
	if delay == 0 { assert!(total / trials as f64 / 100. < 10., "All-SM transport hop_us={} exceeds 10", total / trials as f64 / 100.); }
}
fn multi_demo() {
	let owned = std::fs::read_to_string("../../barrier.ptx").unwrap();
	let start = owned.find("// Hand-owned sm_52 peer publication.").unwrap();
	let end = start + owned[start..].find("// Persistent all-SM packed layer chain.").unwrap();
	assert_eq!(recipe::p2p_functions(), &owned[start..end], "Runtime function source must match the measured PTX");
	let mut p = [recipe::PeerPacket::new("nv0").unwrap(), recipe::PeerPacket::new("nv1").unwrap()];
	assert!(recipe::PeerPacket::new("cpu").is_err());
	assert!(p[0].reserve_sequences(0).is_err());
	assert!(p[0].enable_sender(&p[0]).is_err());
	p[0].enable_sender(&p[1]).unwrap(); p[1].enable_sender(&p[0]).unwrap();
	assert_eq!(p[0].resident_ctas(), 16); assert_eq!(p[1].resident_ctas(), 16);
	let d: [Die; 2] = std::array::from_fn(|i| {
		let mut die = Die { dev: i as c_int, ctx: p[i].context_address() as Handle, packet: p[i].address(), report: 0, relay: std::ptr::null_mut() };
		die.report = die.alloc(4096 * size_of::<Report>());
		die
	});
	let relay_f = [load_function(&d[0], c"p2p_grid_relay"), load_function(&d[1], c"p2p_grid_relay")];
	grid_relay(&d, &mut p, &relay_f, 0, 0, 200);
	grid_relay(&d, &mut p, &relay_f, 5000, 0, 20);
	grid_relay(&d, &mut p, &relay_f, 0, 8, 1);
	let functions = [load_function(&d[0], c"p2p_layers"), load_function(&d[1], c"p2p_layers")];
	for rows in [2560_usize, 10240] {
		let input: Vec<f32> = (0..4096).map(|i| ((i * 37 + 13) % 255) as f32 - 127.).collect();
		let w = [packed_weights_rows(0x12345678, rows, 0x0100), packed_weights_rows(0xfedcba98, rows, 0x0100)];
		let mut x = input.clone();
		let mut expected = Vec::with_capacity(rows*100);
		let mut nonzero_hops = 0;
		for hop in 0..100 {
			let y = reference(&w[hop&1], &x);
			if y.iter().any(|v| v.abs() > 1e-6) { nonzero_hops += 1; }
			x.fill(0.);
			let n = rows.min(4096);
			x[..n].copy_from_slice(&y[..n]);
			expected.extend_from_slice(&y);
		}
		let params: Vec<[Ptr; 5]> = (0..2).map(|i| [upload(&d[i], &w[i]), d[i].alloc(8192*16), d[i].alloc(1024*16), d[i].alloc(rows*4), upload(&d[i], &expected)]).collect();
		let trials = 20;
		let (mut total, mut device, mut max_abs) = (0., [0.; 2], [0_f32; 2]);
		let mut samples = Vec::new();
		for token in 0..(trials+5) {
			d[0].select();
			unsafe { check(cuMemcpyHtoD_v2(p[0].address(), input.as_ptr().cast(), 16384)); }
			let bases = [p[0].reserve_sequences(100).unwrap(), p[1].reserve_sequences(100).unwrap()];
			assert_eq!(bases[0], bases[1]);
			let start = Instant::now();
			for i in [1, 0] {
				d[i].select();
				let (mut local, mut remote, mut report, mut slots) = (p[i].address(), p[1-i].address(), d[i].report, p[i].completion_address());
				let (mut die, mut base) = (i as u32, bases[i]);
				let [mut weights, mut packed_x, mut scales, mut output, mut expected] = params[i];
				let (mut rows, mut hops, mut delay, mut skip) = (rows as u32, 100_u32, 0_u32, 0_u32);
				let mut args = [arg(&mut local), arg(&mut remote), arg(&mut report), arg(&mut slots), arg(&mut die), arg(&mut base), arg(&mut weights), arg(&mut packed_x), arg(&mut scales), arg(&mut output), arg(&mut expected), arg(&mut rows), arg(&mut hops), arg(&mut delay), arg(&mut skip)];
				unsafe { check(cuLaunchKernel(functions[i], p[i].resident_ctas(), 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut())); }
			}
			d[0].sync(); d[1].sync();
			let us = start.elapsed().as_secs_f64()*1e6;
			if token >= 5 { total += us; samples.push(us); }
			for i in 0..2 {
				let (us, abs) = validate_grid(&report_vector(&d[i], 4096), false);
				if token >= 5 { device[i] += us; }
				max_abs[i] = max_abs[i].max(abs);
				let mut slots = [0_u32; 16*32];
				d[i].select();
				unsafe { check(cuMemcpyDtoH_v2(slots.as_mut_ptr().cast(), p[i].completion_address(), size_of_val(&slots))); }
				for cta in 0..16 { assert_eq!(slots[cta*32], bases[i]+if i == 0 {99} else {100}); }
			}
		}
		samples.sort_by(f64::total_cmp);
		println!("multi_layer rows={rows} k=4096 storage=Q4_K activation=Q8_1 accumulator=i32_then_f32 ctas_per_die=16 active_sms=16 hops=100 trials={trials} weights_bytes_per_die={} weights_uploads=2 launches_per_token=2 cpu_between_layers=0 peer_bytes=16384 machine_total_us={total:.3} token_us={:.3} hop_us={:.3} device_hop_us_a={:.3} device_hop_us_b={:.3} p99_token_us={:.3} max_abs_a={:.9} max_abs_b={:.9} scalar_nonzero_hops={nonzero_hops} errors=0 timeouts=0", w[0].len(), total/trials as f64, total/trials as f64/100., device[0]/trials as f64/100., device[1]/trials as f64/100., samples[samples.len()*99/100], max_abs[0], max_abs[1]);
	}
}

fn half_value(bits: u16) -> f64 {
	let exp = (bits >> 10) & 31;
	let frac = bits & 1023;
	let value = if exp == 0 { frac as f64 * 2_f64.powi(-24) } else { (1. + frac as f64 / 1024.) * 2_f64.powi(exp as i32 - 15) };
	if bits & 0x8000 == 0 { value } else { -value }
}
