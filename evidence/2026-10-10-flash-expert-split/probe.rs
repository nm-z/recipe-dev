use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::time::Instant;
type Handle = *mut c_void;
#[link(name = "cuda")]
unsafe extern "C" {
	fn cuInit(flags: c_uint) -> c_int;
	fn cuDeviceGetCount(n: *mut c_int) -> c_int;
	fn cuDeviceGet(d: *mut c_int, index: c_int) -> c_int;
	fn cuDeviceGetUuid(uuid: *mut [u8; 16], dev: c_int) -> c_int;
	fn cuCtxSetCurrent(ctx: Handle) -> c_int;
	fn cuCtxSynchronize() -> c_int;
	fn cuMemAlloc_v2(ptr: *mut u64, bytes: usize) -> c_int;
	fn cuMemFree_v2(ptr: u64) -> c_int;
	fn cuMemcpyDtoH_v2(dst: *mut c_void, src: u64, bytes: usize) -> c_int;
	fn cuMemcpyHtoD_v2(dst: u64, src: *const c_void, bytes: usize) -> c_int;
	fn cuModuleLoad(module: *mut Handle, name: *const c_char) -> c_int;
	fn cuModuleGetFunction(fun: *mut Handle, module: Handle, name: *const c_char) -> c_int;
	fn cuLaunchKernel(fun: Handle, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint,
		shared: c_uint, stream: Handle, args: *mut *mut c_void, extra: *mut *mut c_void) -> c_int;
	fn cuOccupancyMaxActiveBlocksPerMultiprocessor(n: *mut c_int, fun: Handle, block: c_int, shared: usize) -> c_int;
	fn cuGetErrorString(err: c_int, text: *mut *const c_char) -> c_int;
}
fn check(code: c_int) {
	if code != 0 {
		let mut message = std::ptr::null();
		unsafe { cuGetErrorString(code, &mut message); }
		panic!("CUDA {code}: {}", unsafe { CStr::from_ptr(message) }.to_string_lossy());
	}
}
fn arg<T>(value: &mut T) -> *mut c_void { (value as *mut T).cast() }
struct Die { ctx: Handle, fun: Handle, buffers: Vec<u64> }
impl Die {
	fn new(packet: &recipe::PeerPacket, entry: &CStr) -> Self {
		packet.activate().unwrap();
		let (mut module, mut fun, mut active) = (std::ptr::null_mut(), std::ptr::null_mut(), 0);
		unsafe {
			check(cuModuleLoad(&mut module, c"/home/nate/codex/cx-fl-split-build/probe.cubin".as_ptr()));
			check(cuModuleGetFunction(&mut fun, module, entry.as_ptr()));
			check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut active, fun, 256, 0));
		}
		assert!(active >= 1, "one fully resident CTA per SM required");
		assert_eq!(packet.resident_ctas(), 16);
		Self { ctx: packet.context_address() as Handle, fun, buffers: Vec::new() }
	}
	fn select(&self) { unsafe { check(cuCtxSetCurrent(self.ctx)); } }
	fn upload<T>(&mut self, values: &[T]) -> u64 {
		self.select();
		let mut pointer = 0;
		unsafe {
			check(cuMemAlloc_v2(&mut pointer, std::mem::size_of_val(values)));
			check(cuMemcpyHtoD_v2(pointer, values.as_ptr().cast(), std::mem::size_of_val(values)));
		}
		self.buffers.push(pointer);
		pointer
	}
	fn read<T: Copy + Default>(&self, pointer: u64, count: usize) -> Vec<T> {
		self.select();
		let mut values = vec![T::default(); count];
		unsafe { check(cuMemcpyDtoH_v2(values.as_mut_ptr().cast(), pointer, count * size_of::<T>())); }
		values
	}
	fn launch(&self, args: &mut [*mut c_void]) {
		self.select();
		unsafe { check(cuLaunchKernel(self.fun, 16, 1, 1, 256, 1, 1, 0, std::ptr::null_mut(), args.as_mut_ptr(), std::ptr::null_mut())); }
	}
	fn sync(&self) { self.select(); unsafe { check(cuCtxSynchronize()); } }
}
impl Drop for Die {
	fn drop(&mut self) {
		self.select();
		for pointer in &self.buffers { unsafe { check(cuMemFree_v2(*pointer)); } }
	}
}
fn main() {
	let expected = ["ccb8b3a3f45d8962215bc2b140e4bb28", "c767d2504454d181d6488706b5f2fb64"];
	unsafe {
		check(cuInit(0));
		let mut count = 0;
		check(cuDeviceGetCount(&mut count));
		assert_eq!(count, 2, "only the two assigned UUIDs may be visible");
		for ordinal in 0..2 {
			let (mut dev, mut uuid) = (0, [0u8; 16]);
			check(cuDeviceGet(&mut dev, ordinal));
			check(cuDeviceGetUuid(&mut uuid, dev));
			let actual: String = uuid.iter().map(|byte| format!("{byte:02x}")).collect();
			assert_eq!(actual, expected[ordinal as usize], "unexpected UUID before context creation");
		}
	}
	println!("fixture only: two dies, owned callable transport, 16 CTAs/die, 256 threads/CTA, 48 routed layers/token; no expert matvec or real router capture");
	println!("rows,positions,layers,measured_tokens,machine_us_per_token,machine_us_per_layer,device_us_per_token,machine_p99_us,errors");
	for rows in [2560u32, 10240] {
		for positions in 1..=5u32 { run(rows, positions); }
	}
}
fn run(mut rows: u32, mut positions: u32) {
	let mut request = recipe::PeerPacket::with_payload("nv0", 5 * 10240 * 4).unwrap();
	let mut response = recipe::PeerPacket::with_payload("nv1", 5 * 10240 * 4).unwrap();
	assert_eq!(request.sequence_address() - request.address(), 5 * 10240 * 4);
	request.enable_sender(&response).unwrap();
	response.enable_sender(&request).unwrap();
	let mut worker = Die::new(&request, c"split_probe_worker");
	let mut main = Die::new(&response, c"split_probe_main");
	let values: Vec<f32> = (0..rows * positions).map(|index| (index % 101) as f32 / 101.0 - 0.5).collect();
	let mut selected = vec![0i32; positions as usize * 11];
	let mut coefficients = vec![0f32; positions as usize * 10];
	for position in 0..positions as usize {
		let count = [10, 0, 3, 10, 1][position];
		selected[position * 11] = count;
		for slot in 0..count as usize {
			selected[position * 11 + slot + 1] = ((position * 73 + slot * 37) % 512) as i32;
			coefficients[position * 10 + slot] = (slot + 1) as f32 / 64.0;
		}
	}
	let mut hidden = main.upload(&values);
	let mut selections = main.upload(&selected);
	let mut routing = main.upload(&coefficients);
	let mut owners = main.upload(&[1u32; 512]);
	let mut table = main.upload(&[response.address(), response.sequence_address()]);
	let mut output = main.upload(&vec![0f32; (rows * positions) as usize]);
	let mut report = main.upload(&vec![0u64; (16 + 16 * 256 * 4) / 8]);
	let mut partial = worker.upload(&vec![0f32; (rows * positions) as usize]);
	let mut route = worker.upload(&[0u8; 1024]);
	let mut req = request.address();
	let mut req_flag = request.sequence_address();
	let mut resp = response.address();
	let mut resp_flag = response.sequence_address();
	let mut main_slots = response.completion_address();
	let mut worker_slots = request.completion_address();
	let mut layers = 48u32;
	let mut packet_rows = rows.max(4096);
	let mut machine = Vec::new();
	let mut device_ns = 0u64;
	for iteration in 0..220 {
		let mut base = request.reserve_sequences(layers).unwrap();
		assert_eq!(base, response.reserve_sequences(layers).unwrap());
		let started = Instant::now();
		worker.launch(&mut [arg(&mut req), arg(&mut req_flag), arg(&mut route), arg(&mut partial), arg(&mut resp), arg(&mut resp_flag),
			arg(&mut worker_slots), arg(&mut base), arg(&mut layers), arg(&mut rows), arg(&mut positions), arg(&mut packet_rows)]);
		main.launch(&mut [arg(&mut hidden), arg(&mut selections), arg(&mut routing), arg(&mut owners), arg(&mut req), arg(&mut route),
			arg(&mut main_slots), arg(&mut req_flag), arg(&mut table), arg(&mut output), arg(&mut report), arg(&mut base),
			arg(&mut layers), arg(&mut rows), arg(&mut positions), arg(&mut packet_rows)]);
		main.sync();
		worker.sync();
		let elapsed = started.elapsed().as_secs_f64() * 1e6;
		let times = main.read::<u64>(report, 2);
		let failures = main.read::<u32>(report + 16, 16 * 256);
		assert!(failures.iter().all(|word| *word == 0), "per-layer row validation or timeout failed: {:?}", failures.iter().filter(|word| **word != 0).take(16).collect::<Vec<_>>());
		if iteration >= 20 { machine.push(elapsed); device_ns += times[1] - times[0]; }
	}
	let sum: f64 = machine.iter().sum();
	machine.sort_by(f64::total_cmp);
	let mean = sum / machine.len() as f64;
	println!("{rows},{positions},{layers},{},{mean:.3},{:.3},{:.3},{:.3},0", machine.len(), mean / layers as f64,
		device_ns as f64 / machine.len() as f64 / 1e3, machine[machine.len() * 99 / 100]);
}
