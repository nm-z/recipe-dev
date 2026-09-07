#!/usr/bin/env bash
# Probes the Metal device this macOS runner exposes and executes one tiny
# native command buffer against it, so the cell can record a real device
# identity and confirm the device can actually run work.
#
# This is a prerequisite diagnostic. It is never a substitute for Recipe's
# runtime suite: a passing probe with no Recipe execution is still a failed
# cell.
set -euo pipefail

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/probe.swift" <<'SWIFT'
import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else {
	FileHandle.standardError.write("no Metal device is available\n".data(using: .utf8)!)
	exit(1)
}
print("metal device name=\(device.name)")
print("metal device registry_id=\(device.registryID)")
print("metal device low_power=\(device.isLowPower)")
print("metal device removable=\(device.isRemovable)")
print("metal device unified_memory=\(device.hasUnifiedMemory)")
print("metal device max_threads_per_threadgroup=\(device.maxThreadsPerThreadgroup.width)")

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void add(device const float* a [[buffer(0)]],
                device const float* b [[buffer(1)]],
                device float* out [[buffer(2)]],
                uint index [[thread_position_in_grid]]) {
	out[index] = a[index] + b[index];
}
"""

let library = try device.makeLibrary(source: source, options: nil)
guard let function = library.makeFunction(name: "add") else {
	FileHandle.standardError.write("the kernel did not compile\n".data(using: .utf8)!)
	exit(1)
}
let pipeline = try device.makeComputePipelineState(function: function)
guard let queue = device.makeCommandQueue() else {
	FileHandle.standardError.write("no command queue\n".data(using: .utf8)!)
	exit(1)
}

let count = 256
let left = (0..<count).map { Float($0) }
let right = (0..<count).map { Float($0) * 2.0 }
let size = count * MemoryLayout<Float>.stride
guard let a = device.makeBuffer(bytes: left, length: size, options: .storageModeShared),
      let b = device.makeBuffer(bytes: right, length: size, options: .storageModeShared),
      let out = device.makeBuffer(length: size, options: .storageModeShared),
      let buffer = queue.makeCommandBuffer(),
      let encoder = buffer.makeComputeCommandEncoder() else {
	FileHandle.standardError.write("could not build the command buffer\n".data(using: .utf8)!)
	exit(1)
}

encoder.setComputePipelineState(pipeline)
encoder.setBuffer(a, offset: 0, index: 0)
encoder.setBuffer(b, offset: 0, index: 1)
encoder.setBuffer(out, offset: 0, index: 2)
encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
encoder.endEncoding()
buffer.commit()
buffer.waitUntilCompleted()

if let error = buffer.error {
	FileHandle.standardError.write("command buffer failed: \(error)\n".data(using: .utf8)!)
	exit(1)
}

let produced = out.contents().bindMemory(to: Float.self, capacity: count)
for index in 0..<count {
	let expected = Float(index) * 3.0
	if produced[index] != expected {
		FileHandle.standardError.write("readback mismatch at \(index): \(produced[index]) != \(expected)\n".data(using: .utf8)!)
		exit(1)
	}
}
print("metal command_buffer=completed elements=\(count) readback=correct")
SWIFT

swiftc -O "$work/probe.swift" -o "$work/probe"
"$work/probe"
