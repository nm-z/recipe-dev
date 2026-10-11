target triple = "nvptx64-nvidia-cuda"
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
define internal i32 @global_id() #1 { entry:
%lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() %group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %base = mul i32 %group, %width %id = add i32 %base, %lane ret i32 %id }
@grid.count = internal addrspace(1) global i32 0, align 4
@grid.phase = internal addrspace(1) global i32 0, align 4
define internal void @grid_barrier(i32 %threads) #1 { entry:
call void @llvm.nvvm.barrier0() %lane = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%leader = icmp eq i32 %lane, 0 br i1 %leader, label %arrive, label %joined arrive:
%width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %groups = udiv i32 %threads, %width
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
fence acquire br label %joined joined: call void @llvm.nvvm.barrier0() ret void }
; NUMERIC BEGIN
declare float @llvm.sqrt.f32(float) declare float @llvm.fabs.f32(float) declare float @llvm.floor.f32(float)
define internal float @recipe.decode(float %value) #1 { entry: ret float %value  }
define internal float @recipe.encode(float %value) #1 { entry: ret float %value  }


declare float @llvm.fma.f32(float, float, float)
declare double @llvm.floor.f64(double) declare double @llvm.fabs.f64(double) declare double @llvm.fma.f64(double, double, double)



define internal float @recipe.math.exp.narrow.pow2(i32 %k) #1 {
entry:
%biased = add i32 %k, 127
%low = icmp slt i32 %biased, 0
%floor = select i1 %low, i32 0, i32 %biased
%high = icmp sgt i32 %floor, 254
%clamped = select i1 %high, i32 254, i32 %floor
%stored = add i32 %clamped, 0
%bits = shl i32 %stored, 23
%result = bitcast i32 %bits to float
ret float %result
}
define internal float @recipe.math.exp.narrow.scale(float %value, i32 %k) #1 {
entry:
%half = ashr i32 %k, 1
%rest = sub i32 %k, %half
%first = call float @recipe.math.exp.narrow.pow2(i32 %half)
%second = call float @recipe.math.exp.narrow.pow2(i32 %rest)
%scaled = fmul float %value, %first
%result = fmul float %scaled, %second
ret float %result
}
define internal float @recipe.math.exp.narrow(float %x) #1 {
entry:
%unordered = fcmp uno float %x, %x
br i1 %unordered, label %quiet, label %high.check
quiet:
ret float %x
high.check:
%high = fcmp ogt float %x, 0x40562E4300000000
br i1 %high, label %overflow, label %low.check
overflow:
ret float 0x7FF0000000000000
low.check:
%low = fcmp olt float %x, 0xC059FE36A0000000
br i1 %low, label %underflow, label %reduce
underflow:
ret float 0.0
reduce:
%scaled = fmul float %x, 0x3FF7154760000000
%shifted = fadd float %scaled, 0x3FE0000000000000
%kf = call float @llvm.floor.f32(float %shifted)
%k = fptosi float %kf to i32
%upper = fmul float %kf, 0x3FE62E4300000000
%partial = fsub float %x, %upper
%lower = fmul float %kf, 0x3DEA39EF40000000
%r = fsub float %partial, %lower
%exp.poly.0 = call float @llvm.fma.f32(float 0x3EFA01A020000000, float %r, float 0x3F2A01A020000000)
%exp.poly.1 = call float @llvm.fma.f32(float %exp.poly.0, float %r, float 0x3F56C16C20000000)
%exp.poly.2 = call float @llvm.fma.f32(float %exp.poly.1, float %r, float 0x3F81111120000000)
%exp.poly.3 = call float @llvm.fma.f32(float %exp.poly.2, float %r, float 0x3FA5555560000000)
%exp.poly.4 = call float @llvm.fma.f32(float %exp.poly.3, float %r, float 0x3FC5555560000000)
%exp.poly.5 = call float @llvm.fma.f32(float %exp.poly.4, float %r, float 0x3FE0000000000000)
%exp.poly.6 = call float @llvm.fma.f32(float %exp.poly.5, float %r, float 0x3FF0000000000000)
%body = fmul float %r, %exp.poly.6
%expanded = fadd float 0x3FF0000000000000, %body
%result = call float @recipe.math.exp.narrow.scale(float %expanded, i32 %k)
ret float %result
}






define internal float @recipe.math.exp(float %value) #1 { entry: %result = call float @recipe.math.exp.narrow(float %value) ret float %result }




define internal float @recipe.add(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %wide = fadd float %left.wide, %right.wide %result = call float @recipe.encode(float %wide) ret float %result }
define internal float @recipe.sub(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %wide = fsub float %left.wide, %right.wide %result = call float @recipe.encode(float %wide) ret float %result }
define internal float @recipe.mul(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %wide = fmul float %left.wide, %right.wide %result = call float @recipe.encode(float %wide) ret float %result }
define internal float @recipe.div(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %wide = fdiv float %left.wide, %right.wide %result = call float @recipe.encode(float %wide) ret float %result }

declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)




define internal i1 @recipe.ogt(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %result = fcmp ogt float %left.wide, %right.wide ret i1 %result }




define internal float @recipe.from.u1(i1 %value) #1 { entry: %wide = uitofp i1 %value to float %result = call float @recipe.encode(float %wide) ret float %result }










define internal float @recipe.exp(float %value) #1 { entry: %wide = call float @recipe.decode(float %value) %computed = call float @recipe.math.exp(float %wide) %result = call float @recipe.encode(float %computed) ret float %result }






define internal float @recipe.state.add(float %left, float %right) #1 { entry: %result = fadd float %left, %right ret float %result }



define internal float @recipe.state.madd(float %sum, float %left, float %right) #1 { entry: %result = call float @llvm.fma.f32(float %left, float %right, float %sum) ret float %result }
define internal <8 x float> @recipe.state.madd.vector(<8 x float> %sum, <8 x float> %left, <8 x float> %right) #1 {
entry:
%result = call <8 x float> @llvm.fma.v8f32(<8 x float> %left, <8 x float> %right, <8 x float> %sum)
ret <8 x float> %result
}








define internal float @recipe.state.from.u1(i1 %value) #1 { entry: %result = uitofp i1 %value to float ret float %result }


















; NUMERIC END
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare void @llvm.nvvm.barrier0() declare i64 @__ockl_steadyctr_u64()
declare void @llvm.trap() @contraction_tile_f32 = external addrspace(3) global [0 x float], align 16
define internal float @contraction_input(
ptr addrspace(1) %input, i32 %row.base, i32 %position, i32 %term, i32 %span, i32 %length, i1 %conv ) #1 { entry:
%channel = udiv i32 %term, %span %window = urem i32 %term, %span
%offset = select i1 %conv, i32 %window, i32 0 %channel.base = mul i32 %channel, %length
%local.0 = add i32 %channel.base, %position %local = add i32 %local.0, %offset
%index = add i32 %row.base, %local %ptr = getelementptr inbounds float, ptr addrspace(1) %input, i32 %index
%value = load float, ptr addrspace(1) %ptr, align 4 ret float %value }



; The staged B tile is addressed as k-major rows of %tile.n terms. One layout is
; the whole contract: every producer and consumer of the tile routes through this
; function so a vector-loaded K fragment can never assume contiguous slots.
define internal i32 @contraction_vector_a_index(i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
%row = mul i32 %k, %tile.m
%index = add i32 %row, %m
ret i32 %index
}

define internal i32 @contraction_vector_b_index(i32 %k, i32 %n, i32 %tile.n, i32 %tile.k) #1 {
entry:
%row = mul i32 %k, %tile.n
%index = add i32 %row, %n
ret i32 %index
}


define internal void @contraction_zero_edges(i32 %m.count, i32 %n.count, i32 %k.count, i32 %lid, i32 %block, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%a.missing = sub i32 %tile.m, %m.count
%b.missing = sub i32 %tile.n, %n.count
%k.missing = sub i32 %tile.k, %k.count
%a.count = mul i32 %a.missing, %k.count
%b.count = mul i32 %b.missing, %k.count
%output.count = add i32 %a.count, %b.count
%a.k.count = mul i32 %k.missing, %m.count
%a.k.limit = add i32 %output.count, %a.k.count
%b.k.count = mul i32 %k.missing, %n.count
%count = add i32 %a.k.limit, %b.k.count
br label %loop
loop:
%p = phi i32 [ %lid, %entry ], [ %next, %store ]
%more = icmp ult i32 %p, %count
br i1 %more, label %classify, label %exit
classify:
%a = icmp ult i32 %p, %a.count
br i1 %a, label %a.step, label %classify.b
classify.b:
%b = icmp ult i32 %p, %output.count
br i1 %b, label %b.step, label %classify.a.k
classify.a.k:
%is.a.k = icmp ult i32 %p, %a.k.limit
br i1 %is.a.k, label %a.k.step, label %b.k.step
a.step:
%a.k = udiv i32 %p, %a.missing
%a.local = urem i32 %p, %a.missing
%a.m = add i32 %m.count, %a.local
%a.index = call i32 @contraction_vector_a_index(i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br label %store
b.step:
%b.p = sub i32 %p, %a.count
%b.k = udiv i32 %b.p, %b.missing
%b.local = urem i32 %b.p, %b.missing
%b.n = add i32 %n.count, %b.local
%b.base = mul i32 %tile.m, %tile.k
%b.local.index = call i32 @contraction_vector_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k)
%b.index = add i32 %b.base, %b.local.index
br label %store
a.k.step:
%a.k.p = sub i32 %p, %output.count
%a.k.local = udiv i32 %a.k.p, %m.count
%a.k.value = add i32 %k.count, %a.k.local
%a.k.m = urem i32 %a.k.p, %m.count
%a.k.index = call i32 @contraction_vector_a_index(i32 %a.k.value, i32 %a.k.m, i32 %tile.m, i32 %tile.k)
br label %store
b.k.step:
%b.k.p = sub i32 %p, %a.k.limit
%b.k.local = udiv i32 %b.k.p, %n.count
%b.k.value = add i32 %k.count, %b.k.local
%b.k.n = urem i32 %b.k.p, %n.count
%b.k.base = mul i32 %tile.m, %tile.k
%b.k.local.index = call i32 @contraction_vector_b_index(i32 %b.k.value, i32 %b.k.n, i32 %tile.n, i32 %tile.k)
%b.k.index = add i32 %b.k.base, %b.k.local.index
br label %store
store:
%index = phi i32 [ %a.index, %a.step ], [ %b.index, %b.step ], [ %a.k.index, %a.k.step ], [ %b.k.index, %b.k.step ]
%ptr = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %index
store float 0.0, ptr addrspace(3) %ptr, align 4
%next = add i32 %p, %block
br label %loop
exit:
ret void
}
define internal <8 x float> @contraction_a_fragment(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k) #1 {
entry:
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%ptr = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %index
%fragment = load <8 x float>, ptr addrspace(3) %ptr, align 4
ret <8 x float> %fragment
}
define internal <8 x float> @contraction_b_fragment(i32 %k, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %output.n.base, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%ptr = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %index
%fragment = load <8 x float>, ptr addrspace(3) %ptr, align 4
ret <8 x float> %fragment
}
define internal void @contraction_stage_a_fragment(<16 x float> %fragment, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, 16
br i1 %more, label %step, label %exit
step:
%local.k = add i32 %k, %element
%index = call i32 @contraction_vector_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
%value = extractelement <16 x float> %fragment, i32 %element
%target = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %index
store float %value, ptr addrspace(3) %target, align 4
%next = add i32 %element, 1
br label %loop
exit:
ret void
}

; Stage a fragment of consecutive K for one channel. The B tile is k-major, so
; consecutive K lands one %tile.n row apart and the elements are placed through
; @contraction_b_index rather than stored as one contiguous vector.
define internal void @contraction_stage_b_terms(<16 x float> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, 16
br i1 %more, label %step, label %exit
step:
%local.k = add i32 %k, %element
%local = call i32 @contraction_vector_b_index(i32 %local.k, i32 %n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <16 x float> %fragment, i32 %element
%target = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %index
store float %value, ptr addrspace(3) %target, align 4
%next = add i32 %element, 1
br label %loop
exit:
ret void
}


define internal i32 @contraction_output_lanes(i32 %m.lanes, i32 %n.lanes, i32 %block) #1 {
entry:
%lanes = mul i32 %m.lanes, %n.lanes
ret i32 %lanes
}
define internal i32 @contraction_vector_output_m(i32 %lid, i32 %register, i32 %m.lanes) #1 {
entry:
%lane = urem i32 %lid, %m.lanes
%base = mul i32 %lane, 8
%local = urem i32 %register, 8
%m = add i32 %base, %local
ret i32 %m
}
define internal i32 @contraction_vector_output_n(i32 %lid, i32 %register, i32 %m.lanes) #1 {
entry:
%lane = udiv i32 %lid, %m.lanes
%base = mul i32 %lane, 8
%local = udiv i32 %register, 8
%n = add i32 %base, %local
ret i32 %n
}


define internal i1 @contraction_vector_store_lane(i1 %store, i32 %lid) #1 {
entry:
ret i1 %store
}

define internal i1 @contraction_output_register_valid(i32 %register) #1 {
entry:
ret i1 true
}
; The K walk is cut into fixed chunks of 64 elements. Each chunk
; is summed in ascending order into a private partial and the partials are
; folded into the running sums in ascending chunk order, so the partial values
; and the final parenthesisation follow the K extent and one program constant
; and nothing else. A job whose output fills the workgroup has a single k lane
; that owns every chunk: it folds each finished chunk locally, its accumulator
; indices are compile-time constants, and no barrier or local memory is
; involved. A one-chunk job also stays local because only its first k lane owns
; work. A job with spare lanes and several chunks gives each k lane a stride of chunks and
; exchanges the partials through the staged-tile region of local memory once
; every lane has consumed the staged operands; one owner lane then folds them.
; The k lane count is uniform across the workgroup, so the two paths never
; split a barrier, and both walk the same chunks in the same order, so the
; bytes agree between them and between backends.
define internal void @contraction_vector_accumulate(
ptr %sums, i1 %lane.active, i1 %lane.store, i32 %lid,
i32 %lane.k, i32 %k.lanes, i32 %output.lane, i32 %output.lanes,
i32 %output.m.base, i32 %output.n.base, i32 %m.count, i32 %n.count,
i32 %k.count, i32 %tile.m, i32 %tile.n, i32 %tile.k ) #1 {
entry:
%chunk.sums = alloca [64 x float], align 4
%state.zero = call float @recipe.state.from.u1(i1 false)
%chunks.adjusted = add i32 %k.count, 64
%chunks.numerator = sub i32 %chunks.adjusted, 1
%chunks = udiv i32 %chunks.numerator, 64
%single.sum.slot = icmp eq i32 64, 64
%publish.lane.width = mul i32 %output.lanes, 64
%chunk.first = select i1 %lane.active, i32 %lane.k, i32 %chunks
%local.lane = icmp eq i32 %k.lanes, 1
%few.chunks = icmp ule i32 %chunks, 3
%local = or i1 %local.lane, %few.chunks
%local.owner = icmp eq i32 %lane.k, 0
%local.owner.active = and i1 %lane.active, %local.owner
br i1 %local, label %local.owner.check, label %shared.chunk.loop
local.owner.check:
br i1 %local.owner.active, label %local.k.begin, label %exit
local.k.begin:
%local.sums.initial = load <64 x float>, ptr %sums, align 4
%local.a.initial = call <8 x float> @contraction_a_fragment(i32 0, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.initial = call <8 x float> @contraction_b_fragment(i32 0, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %local.k.loop
local.k.loop:
%local.k = phi i32 [ 0, %local.k.begin ], [ %local.k.next, %local.product.done ]
%local.sums = phi <64 x float> [ %local.sums.initial, %local.k.begin ], [ %local.sums.current, %local.product.done ]
%local.a.fragment = phi <8 x float> [ %local.a.initial, %local.k.begin ], [ %local.a.next, %local.product.done ]
%local.b.fragment = phi <8 x float> [ %local.b.initial, %local.k.begin ], [ %local.b.next, %local.product.done ]
%local.k.next = add i32 %local.k, 1
%local.k.more = icmp ult i32 %local.k.next, %k.count
%local.k.prefetch = select i1 %local.k.more, i32 %local.k.next, i32 %local.k
%local.a.next = call <8 x float> @contraction_a_fragment(i32 %local.k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%local.b.next = call <8 x float> @contraction_b_fragment(i32 %local.k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%local.a.wide = call <8 x float> @contraction_widen_m(<8 x float> %local.a.fragment)
br label %local.product.loop
local.product.loop:
%local.product = phi i32 [ 0, %local.k.loop ], [ %local.product.next, %local.product.step ]
%local.sums.current = phi <64 x float> [ %local.sums, %local.k.loop ], [ %local.candidate, %local.product.step ]
%local.product.more = icmp ult i32 %local.product, 64
br i1 %local.product.more, label %local.product.step, label %local.product.done
local.product.step:
%local.a.index = urem i32 %local.product, 8
%local.b.index = udiv i32 %local.product, 8
%local.a = extractelement <8 x float> %local.a.wide, i32 %local.a.index
%local.b = extractelement <8 x float> %local.b.fragment, i32 %local.b.index
%local.b.wide = call float @recipe.decode(float %local.b)
%local.sum = extractelement <64 x float> %local.sums.current, i32 %local.product
%local.value = call float @recipe.state.madd(float %local.sum, float %local.a, float %local.b.wide)
%local.candidate = insertelement <64 x float> %local.sums.current, float %local.value, i32 %local.product
%local.product.next = add i32 %local.product, 1
br label %local.product.loop, !llvm.loop !0
local.product.done:
br i1 %local.k.more, label %local.k.loop, label %local.store
local.store:
store <64 x float> %local.sums.current, ptr %sums, align 4
br label %exit
shared.chunk.loop:
%chunk = phi i32 [ %chunk.first, %entry ], [ %chunk.next, %chunk.finish ]
%slot = phi i32 [ 0, %entry ], [ %slot.next, %chunk.finish ]
%sum.slot = select i1 %single.sum.slot, i32 0, i32 %slot
%chunk.more = icmp ult i32 %chunk, %chunks
br i1 %chunk.more, label %chunk.zero.loop, label %chunk.done
chunk.zero.loop:
%zero.r = phi i32 [ 0, %shared.chunk.loop ], [ %zero.next, %chunk.zero.step ]
%zero.more = icmp ult i32 %zero.r, 64
br i1 %zero.more, label %chunk.zero.step, label %k.begin
chunk.zero.step:
%zero.base = mul i32 %sum.slot, 64
%zero.index = add i32 %zero.base, %zero.r
%zero.ptr = getelementptr float, ptr %chunk.sums, i32 %zero.index
store float %state.zero, ptr %zero.ptr, align 4
%zero.next = add i32 %zero.r, 1
br label %chunk.zero.loop
k.begin:
%k.first = mul i32 %chunk, 64
%k.limit.raw = add i32 %k.first, 64
%k.over = icmp ugt i32 %k.limit.raw, %k.count
%k.limit = select i1 %k.over, i32 %k.count, i32 %k.limit.raw
%slot.sum.base = mul i32 %sum.slot, 64
%a.initial = call <8 x float> @contraction_a_fragment(i32 %k.first, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.initial = call <8 x float> @contraction_b_fragment(i32 %k.first, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %k.loop
k.loop:
%k = phi i32 [ %k.first, %k.begin ], [ %k.next, %register.done ]
%a.fragment = phi <8 x float> [ %a.initial, %k.begin ], [ %a.next, %register.done ]
%b.fragment = phi <8 x float> [ %b.initial, %k.begin ], [ %b.next, %register.done ]
%k.next = add i32 %k, 1
%k.more = icmp ult i32 %k.next, %k.limit
%k.prefetch = select i1 %k.more, i32 %k.next, i32 %k
%a.next = call <8 x float> @contraction_a_fragment(i32 %k.prefetch, i32 %output.m.base, i32 %tile.m, i32 %tile.k)
%b.next = call <8 x float> @contraction_b_fragment(i32 %k.prefetch, i32 %output.n.base, i32 %tile.m, i32 %tile.n, i32 %tile.k)
%a.wide = call <8 x float> @contraction_widen_m(<8 x float> %a.fragment)
br label %register.loop
register.loop:
%register.n = phi i32 [ 0, %k.loop ], [ %register.n.next, %register.next ]
%register.more = icmp ult i32 %register.n, 8
br i1 %register.more, label %register.step, label %register.done
register.step:
%output.n.raw = add i32 %output.n.base, %register.n
%output.n.valid = icmp ult i32 %output.n.raw, %n.count
%b = extractelement <8 x float> %b.fragment, i32 %register.n
%b.wide = call float @recipe.decode(float %b)
%b.seed = insertelement <8 x float> poison, float %b.wide, i32 0
%b.vector = shufflevector <8 x float> %b.seed, <8 x float> poison, <8 x i32> zeroinitializer
%register.base = mul i32 %register.n, 8
%sum.index = add i32 %slot.sum.base, %register.base
%sum.ptr = getelementptr float, ptr %chunk.sums, i32 %sum.index
%sum = load <8 x float>, ptr %sum.ptr, align 4
%candidate = call <8 x float> @recipe.state.madd.vector(<8 x float> %sum, <8 x float> %a.wide, <8 x float> %b.vector)
store <8 x float> %candidate, ptr %sum.ptr, align 4
br label %register.next
register.next:
%register.n.next = add i32 %register.n, 1
br label %register.loop
register.done:
br i1 %k.more, label %k.loop, label %chunk.finish
chunk.finish:
%chunk.next = add i32 %chunk, %k.lanes
%slot.next = add i32 %slot, 1
br label %shared.chunk.loop
chunk.done:
call void @llvm.nvvm.barrier0()
br label %publish.loop
publish.loop:
%publish.chunk = phi i32 [ %chunk.first, %chunk.done ], [ %publish.chunk.next, %publish.finish ]
%publish.slot = phi i32 [ 0, %chunk.done ], [ %publish.slot.next, %publish.finish ]
%publish.sum.slot = select i1 %single.sum.slot, i32 0, i32 %publish.slot
%publish.more = icmp ult i32 %publish.chunk, %chunks
br i1 %publish.more, label %publish.sum.loop, label %publish.done
publish.sum.loop:
%publish.r = phi i32 [ 0, %publish.loop ], [ %publish.r.next, %publish.sum.step ]
%publish.r.more = icmp ult i32 %publish.r, 64
br i1 %publish.r.more, label %publish.sum.step, label %publish.finish
publish.sum.step:
%publish.source.base = mul i32 %publish.sum.slot, 64
%publish.source.index = add i32 %publish.source.base, %publish.r
%publish.source = getelementptr float, ptr %chunk.sums, i32 %publish.source.index
%publish.value = load float, ptr %publish.source, align 4
%publish.row = mul i32 %publish.chunk, %output.lanes
%publish.column = add i32 %publish.row, %output.lane
%publish.target.base = mul i32 %publish.column, 64
%publish.target.index = add i32 %publish.target.base, %publish.r
%publish.target = getelementptr float, ptr addrspace(3) @contraction_tile_f32, i32 %publish.target.index
store float %publish.value, ptr addrspace(3) %publish.target, align 4
%publish.r.next = add i32 %publish.r, 1
br label %publish.sum.loop
publish.finish:
%publish.chunk.next = add i32 %publish.chunk, %k.lanes
%publish.slot.next = add i32 %publish.slot, 1
br label %publish.loop
publish.done:
call void @llvm.nvvm.barrier0()
%owner = icmp eq i32 %lane.k, 0
%fold.active = and i1 %lane.active, %owner
br i1 %fold.active, label %fold.loop, label %exit
fold.loop:
%fold.chunk = phi i32 [ 0, %publish.done ], [ %fold.chunk.next, %fold.finish ]
%fold.more = icmp ult i32 %fold.chunk, %chunks
br i1 %fold.more, label %fold.sum.loop, label %exit
fold.sum.loop:
%fold.r = phi i32 [ 0, %fold.loop ], [ %fold.r.next, %fold.sum.step ]
%fold.r.more = icmp ult i32 %fold.r, 64
br i1 %fold.r.more, label %fold.sum.step, label %fold.finish
fold.sum.step:
%fold.row = mul i32 %fold.chunk, %output.lanes
%fold.column = add i32 %fold.row, %output.lane
%fold.source.base = mul i32 %fold.column, 64
%fold.source.index = add i32 %fold.source.base, %fold.r
%fold.source = getelementptr float, ptr addrspace(3) @contraction_tile_f32, i32 %fold.source.index
%fold.value = load float, ptr addrspace(3) %fold.source, align 4
%fold.target = getelementptr float, ptr %sums, i32 %fold.r
%fold.current = load float, ptr %fold.target, align 4
%fold.next = call float @recipe.state.add(float %fold.current, float %fold.value)
store float %fold.next, ptr %fold.target, align 4
%fold.r.next = add i32 %fold.r, 1
br label %fold.sum.loop
fold.finish:
%fold.chunk.next = add i32 %fold.chunk, 1
br label %fold.loop
exit:
ret void
}
; Bias gradients consume the same staged delta tile as the weight product.
; Each lane carries one sum for every output channel in its workgroup stride.

; Widen a staged A fragment to the arithmetic type once per K step, so the inner
; register loop never converts.
define internal <8 x float> @contraction_widen_m(<8 x float> %source) #1 {
entry:
br label %loop
loop:
%p = phi i32 [ 0, %entry ], [ %p.next, %step ]
%result = phi <8 x float> [ poison, %entry ], [ %next, %step ]
%more = icmp ult i32 %p, 8
br i1 %more, label %step, label %done
step:
%value = extractelement <8 x float> %source, i32 %p
%wide = call float @recipe.decode(float %value)
%next = insertelement <8 x float> %result, float %wide, i32 %p
%p.next = add i32 %p, 1
br label %loop
done:
ret <8 x float> %result
}


declare <8 x float> @recipe.wmma(<16 x float>, <16 x float>, <8 x float>)
; Matrix arithmetic consumes the operands staged by the common contraction
; composer, writes one state-width partial tile beside them, then maps that tile
; back into the composer's register ownership. Scheduling, tails, K tiling, and
; the epilogue therefore remain identical to the vector method.

define internal void @contraction_forward_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %activation, i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %kernel,
i1 %has.bias, i1 %relu, i1 %transpose, i1 %reverse, i1 %accumulate, i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %threads ) #1 { entry:
; The running sums live in the arithmetic type for the whole K extent and are
; rounded to the model type once, at the store. Staging the operands in tiles
; therefore cannot move a rounding point.
%sums = alloca [64 x float], align 4 %state.zero = call float @recipe.state.from.u1(i1 false) %lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() %group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() %block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %groups = udiv i32 %threads, %block
%in.elements = mul i32 %in.channels, %in.length %out.elements = mul i32 %out.channels, %out.length %is.conv = icmp ne i32 %kernel, 0 %span = select i1 %is.conv, i32 %kernel, i32 1 %terms = mul i32 %in.channels, %span %m.total = mul i32 %rows, %out.length
%m.short = icmp ult i32 %tile.m, %m.total %m.tile = select i1 %m.short, i32 %tile.m, i32 %m.total %n.short = icmp ult i32 %tile.n, %out.channels %n.tile = select i1 %n.short, i32 %tile.n, i32 %out.channels %k.short = icmp ult i32 %tile.k, %terms %k.tile = select i1 %k.short, i32 %tile.k, i32 %terms
%m.adjusted = add i32 %m.total, %m.tile %m.numerator = sub i32 %m.adjusted, 1 %m.tiles = udiv i32 %m.numerator, %m.tile %n.adjusted = add i32 %out.channels, %n.tile %n.numerator = sub i32 %n.adjusted, 1 %n.tiles = udiv i32 %n.numerator, %n.tile %jobs = mul i32 %m.tiles, %n.tiles br label %job.loop job.loop:
%job = phi i32 [ %group, %entry ], [ %job.next, %job.done ] %job.more = icmp ult i32 %job, %jobs br i1 %job.more, label %job.step, label %exit job.step:
%m.group.short = icmp ult i32 %m.tiles, 4 %m.group.limit = select i1 %m.group.short, i32 %m.tiles, i32 4 %group.width = mul i32 %m.group.limit, %n.tiles %group.index = udiv i32 %job, %group.width %m.group.base = mul i32 %group.index, %m.group.limit %m.group.remaining = sub i32 %m.tiles, %m.group.base %m.group.tail = icmp ult i32 %m.group.remaining, %m.group.limit %m.group.count = select i1 %m.group.tail, i32 %m.group.remaining, i32 %m.group.limit %group.local = urem i32 %job, %group.width %m.group.local = urem i32 %group.local, %m.group.count %m.tile.index = add i32 %m.group.base, %m.group.local %n.tile.index = udiv i32 %group.local, %m.group.count %m.base = mul i32 %m.tile.index, %m.tile %n.base = mul i32 %n.tile.index, %n.tile
%m.remaining = sub i32 %m.total, %m.base %m.partial = icmp ult i32 %m.remaining, %m.tile %m.count = select i1 %m.partial, i32 %m.remaining, i32 %m.tile %n.remaining = sub i32 %out.channels, %n.base %n.partial = icmp ult i32 %n.remaining, %n.tile %n.count = select i1 %n.partial, i32 %n.remaining, i32 %n.tile
%m.lanes.adjusted = add i32 %m.count, 8 %m.lanes.numerator = sub i32 %m.lanes.adjusted, 1 %m.lanes = udiv i32 %m.lanes.numerator, 8 %n.lanes.adjusted = add i32 %n.count, 8 %n.lanes.numerator = sub i32 %n.lanes.adjusted, 1 %n.lanes = udiv i32 %n.lanes.numerator, 8
; A lane owns one output position; the lanes left over at the same output
; position each own a share of the K chunks, so a skinny output tile still
; drives the whole workgroup.
%lanes = call i32 @contraction_output_lanes(i32 %m.lanes, i32 %n.lanes, i32 %block)
%k.lanes.raw = udiv i32 %block, %lanes
%k.lanes.some = icmp ugt i32 %k.lanes.raw, 0
%k.lanes = select i1 %k.lanes.some, i32 %k.lanes.raw, i32 1
%active.lanes = mul i32 %lanes, %k.lanes
%lane.active = icmp ult i32 %lid, %active.lanes
%output.lane.raw = urem i32 %lid, %lanes
%output.lane = select i1 %lane.active, i32 %output.lane.raw, i32 0
%lane.k.raw = udiv i32 %lid, %lanes
%lane.k = select i1 %lane.active, i32 %lane.k.raw, i32 0
%lane.owner = icmp eq i32 %lane.k, 0
%lane.store = and i1 %lane.active, %lane.owner
%method.store = call i1 @contraction_vector_store_lane(i1 %lane.store, i32 %lid)
%lane.n = udiv i32 %output.lane, %m.lanes %lane.m = urem i32 %output.lane, %m.lanes
%output.m.base = mul i32 %lane.m, 8 %output.n.base = mul i32 %lane.n, 8 br label %sum.init.loop sum.init.loop:
%sum.init = phi i32 [ 0, %job.step ], [ %sum.init.next, %sum.init.step ] %sum.init.more = icmp ult i32 %sum.init, 64 br i1 %sum.init.more, label %sum.init.step, label %sum.init.done
sum.init.step: %sum.init.ptr = getelementptr [64 x float], ptr %sums, i32 0, i32 %sum.init store float %state.zero, ptr %sum.init.ptr, align 4 %sum.init.next = add i32 %sum.init, 1 br label %sum.init.loop sum.init.done: br label %tile.loop tile.loop:
%term.base = phi i32 [ 0, %sum.init.done ], [ %term.next, %tile.done ] %k.remaining = sub i32 %terms, %term.base %k.partial = icmp ult i32 %k.remaining, %k.tile %k.count = select i1 %k.partial, i32 %k.remaining, i32 %k.tile
%a.project = icmp eq i32 %span, 1
%a.unit = icmp eq i32 %in.length, 1
%a.contiguous = and i1 %a.project, %a.unit
%a.fragment.remainder = urem i32 %k.count, 16
%a.fragment.full = icmp eq i32 %a.fragment.remainder, 0
%a.gate = and i1 %reverse, %relu %a.ungated = xor i1 %a.gate, true %a.vector.shape = and i1 %a.contiguous, %a.fragment.full %a.vector = and i1 %a.vector.shape, %a.ungated
%a.width = select i1 %a.vector, i32 16, i32 1
%a.columns = udiv i32 %k.count, %a.width
%b.fragment.remainder = urem i32 %k.count, 16
%b.fragment.full = icmp eq i32 %b.fragment.remainder, 0 %b.direct = xor i1 %transpose, true %b.vector = and i1 %b.fragment.full, %b.direct
%b.width = select i1 %b.vector, i32 16, i32 1
%b.rows = udiv i32 %k.count, %b.width
%a.count = mul i32 %m.count, %a.columns %b.count = mul i32 %n.count, %b.rows %load.count = add i32 %a.count, %b.count br label %load.loop load.loop:
%load = phi i32 [ %lid, %tile.loop ], [ %load.next, %load.advance ] %load.more = icmp ult i32 %load, %load.count br i1 %load.more, label %load.classify, label %load.done load.classify: %load.a = icmp ult i32 %load, %a.count br i1 %load.a, label %load.a.step, label %load.b.step
load.a.step: %a.m = udiv i32 %load, %a.columns %a.column = urem i32 %load, %a.columns %a.k = mul i32 %a.column, %a.width %a.global = add i32 %m.base, %a.m %a.row = udiv i32 %a.global, %out.length %a.position = urem i32 %a.global, %out.length %a.row.base = mul i32 %a.row, %in.elements %a.term = add i32 %term.base, %a.k
%a.tile.index = call i32 @contraction_vector_a_index(i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br i1 %a.vector, label %load.a.vector, label %load.a.scalar
load.a.vector:
%a.vector.index = add i32 %a.row.base, %a.term
%a.vector.source = getelementptr inbounds float, ptr addrspace(1) %input, i32 %a.vector.index
%a.vector.value = load <16 x float>, ptr addrspace(1) %a.vector.source, align 4
call void @contraction_stage_a_fragment(<16 x float> %a.vector.value, i32 %a.k, i32 %a.m, i32 %tile.m, i32 %tile.k)
br label %load.advance
load.a.scalar:
%a.loaded = call float @contraction_input( ptr addrspace(1) %input, i32 %a.row.base, i32 %a.position, i32 %a.term, i32 %span, i32 %in.length, i1 %is.conv )
br i1 %a.gate, label %load.a.activation, label %load.a.ready
load.a.activation:
%a.activation.channel = mul i32 %a.term, %in.length %a.activation.local = add i32 %a.activation.channel, %a.position %a.activation.index = add i32 %a.row.base, %a.activation.local %a.activation.ptr = getelementptr inbounds float, ptr addrspace(1) %activation, i32 %a.activation.index %a.activation.value = load float, ptr addrspace(1) %a.activation.ptr, align 2 %a.activation.positive = call i1 @recipe.ogt(float %a.activation.value, float 0.0) %a.gated = select i1 %a.activation.positive, float %a.loaded, float 0.0
br label %load.a.ready
load.a.ready:
%a.value = phi float [ %a.loaded, %load.a.scalar ], [ %a.gated, %load.a.activation ]
br label %load.store
load.b.step: %b.local = sub i32 %load, %a.count %b.n = udiv i32 %b.local, %b.rows %b.row = urem i32 %b.local, %b.rows %b.k = mul i32 %b.row, %b.width %b.channel = add i32 %n.base, %b.n %b.channel.base = mul i32 %b.channel, %terms %b.term = add i32 %term.base, %b.k
%b.direct.index = add i32 %b.channel.base, %b.term %b.transpose.base = mul i32 %b.term, %out.channels %b.transpose.index = add i32 %b.transpose.base, %b.channel %b.index = select i1 %transpose, i32 %b.transpose.index, i32 %b.direct.index %b.tile.base = mul i32 %tile.m, %tile.k %b.tile.local = call i32 @contraction_vector_b_index(i32 %b.k, i32 %b.n, i32 %tile.n, i32 %tile.k) %b.tile.index = add i32 %b.tile.base, %b.tile.local
br i1 %b.vector, label %load.b.vector, label %load.b.scalar
load.b.vector:
%b.vector.source = getelementptr inbounds float, ptr addrspace(1) %weights, i32 %b.index
%b.vector.value = load <16 x float>, ptr addrspace(1) %b.vector.source, align 4
call void @contraction_stage_b_terms(<16 x float> %b.vector.value, i32 %b.k, i32 %b.n, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %load.advance
load.b.scalar:
%b.ptr = getelementptr inbounds float, ptr addrspace(1) %weights, i32 %b.index
%b.value = load float, ptr addrspace(1) %b.ptr, align 4
br label %load.store
load.store: %load.value = phi float [ %a.value, %load.a.ready ], [ %b.value, %load.b.scalar ] %load.tile.index = phi i32 [ %a.tile.index, %load.a.ready ], [ %b.tile.index, %load.b.scalar ] %load.tile.ptr = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %load.tile.index store float %load.value, ptr addrspace(3) %load.tile.ptr, align 4
br label %load.advance
load.advance:
%load.next = add i32 %load, %block br label %load.loop load.done:
%load.logical.output.edge = or i1 %m.partial, %n.partial
%load.logical.edge = or i1 %load.logical.output.edge, %k.partial
%load.m.edge = icmp ult i32 %m.count, %tile.m
%load.n.edge = icmp ult i32 %n.count, %tile.n
%load.k.edge = icmp ult i32 %k.count, %tile.k
%load.schedule.output.edge = or i1 %load.m.edge, %load.n.edge
%load.schedule.edge = or i1 %load.schedule.output.edge, %load.k.edge
; Zero whenever the staged tile is not completely filled. The logical counts are
; clamped to the shape, so they miss the case where the tile is wider than the
; whole operand and the unwritten lanes would read uninitialised local memory.
%load.vector.edge = or i1 %load.schedule.edge, %load.logical.edge
br i1 %load.vector.edge, label %load.zero, label %load.ready
load.zero:
call void @contraction_zero_edges(i32 %m.count, i32 %n.count, i32 %k.count, i32 %lid, i32 %block, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %load.ready
load.ready:
call void @llvm.nvvm.barrier0()
call void @contraction_vector_accumulate(ptr %sums, i1 %lane.active, i1 %method.store, i32 %lid, i32 %lane.k, i32 %k.lanes, i32 %output.lane, i32 %lanes, i32 %output.m.base, i32 %output.n.base, i32 %m.count, i32 %n.count, i32 %k.count, i32 %tile.m, i32 %tile.n, i32 %tile.k)
br label %accumulate.done
accumulate.done:
call void @llvm.nvvm.barrier0()
%term.next = add i32 %term.base, %k.count %term.more = icmp ult i32 %term.next, %terms br i1 %term.more, label %tile.done, label %store.loop tile.done: br label %tile.loop store.loop:
%store.register = phi i32 [ 0, %accumulate.done ], [ %store.register.next, %store.next ] %store.more = icmp ult i32 %store.register, 64 br i1 %store.more, label %store.check, label %job.done
store.check: %store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %store.register, i32 %m.lanes) %store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %store.register, i32 %m.lanes) %store.register.valid = call i1 @contraction_output_register_valid(i32 %store.register)
%store.output.m.valid = icmp ult i32 %store.output.m.raw, %m.count %store.output.n.valid = icmp ult i32 %store.output.n.raw, %n.count %store.output.valid = and i1 %store.output.m.valid, %store.output.n.valid %store.lane.active = and i1 %method.store, %store.output.valid %store.active = and i1 %store.lane.active, %store.register.valid br i1 %store.active, label %store, label %store.next
store: %store.channel = add i32 %n.base, %store.output.n.raw %store.m.global = add i32 %m.base, %store.output.m.raw %store.position = urem i32 %store.m.global, %out.length %store.row = udiv i32 %store.m.global, %out.length %store.output.row.base = mul i32 %store.row, %out.elements
%store.output.channel.base = mul i32 %store.channel, %out.length %store.output.local = add i32 %store.output.channel.base, %store.position %store.output.index = add i32 %store.output.row.base, %store.output.local %store.output.ptr = getelementptr inbounds float, ptr addrspace(1) %output, i32 %store.output.index
%store.bias.base = mul i32 %out.channels, %terms %store.bias.index = add i32 %store.bias.base, %store.channel %store.bias.ptr = getelementptr inbounds float, ptr addrspace(1) %weights, i32 %store.bias.index %store.bias = load float, ptr addrspace(1) %store.bias.ptr, align 4 %store.sum.ptr = getelementptr [64 x float], ptr %sums, i32 0, i32 %store.register %store.sum.wide = load float, ptr %store.sum.ptr, align 4 %store.sum = call float @recipe.encode(float %store.sum.wide)
%store.biased = call float @recipe.add(float %store.sum, float %store.bias) %store.raw = select i1 %has.bias, float %store.biased, float %store.sum %store.forward = xor i1 %reverse, true %store.activate = and i1 %relu, %store.forward %store.positive = call i1 @recipe.ogt(float %store.raw, float 0.0) %store.activated = select i1 %store.positive, float %store.raw, float 0.0 %store.result = select i1 %store.activate, float %store.activated, float %store.raw %store.prior = load float, ptr addrspace(1) %store.output.ptr, align 2 %store.accumulated = call float @recipe.add(float %store.prior, float %store.result) %store.value = select i1 %accumulate, float %store.accumulated, float %store.result store float %store.value, ptr addrspace(1) %store.output.ptr, align 4 br label %store.next
store.next: %store.register.next = add i32 %store.register, 1 br label %store.loop job.done: %job.next = add i32 %job, %groups br label %job.loop exit: ret void }

; Rotary embedding over the first %rotated channels of a fused QKV row: inside
; each head the channel pairs (i, i + dims/2) below %dims rotate by
; position * base^(-2i/dims). With %reverse the transpose rotation is added
; into %output, which makes the same body the adjoint pass.




; Inverse Euclidean norm of a strided indexer vector, floored by the epsilon.

; True when the query keeps the block that holds this key. Each query owns one
; row of block scores followed by one admission flag per block.

; One key-block representative of the indexer: the sum of the unit norm indexer
; keys inside the block.

; Block scores and the selection threshold of one query. The indexer scores
; every causal block and the threshold is the score of the keep-th best, so a
; query keeps every block whose score reaches it.

; Loss gradient of one block score: the softmax derivative of every key the
; block admits, summed over the attention heads that share the selection.

; One channel of the unit indexer query gradient: every causal block
; representative weighted by the gradient of the block score it scored.

; One channel of the unit indexer key gradient. The representative sums the
; unit keys of the block, so every query that scores the block contributes its
; block score gradient times the sum of its unit indexer queries.

; Indexer gradient of one sequence position. A block score biases the logits of
; the keys the block admits, so its gradient is the softmax derivative summed
; over those keys, and the chain rule carries it through the unit norms into
; the indexer query planes and the shared indexer key plane.










 attributes #0 = { nounwind } attributes #1 = { alwaysinline nounwind } attributes #3 = { noinline nounwind }
; Fully unroll the product loop so each insertelement uses a constant lane index
; and the accumulator vector remains in registers.
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}define internal void @recipe_model_inference_forward_body(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads) #3 {
entry:
%tid = call i32 @global_id()
%n0.value = getelementptr i8, ptr addrspace(1) %values, i64 0
%n0.context = getelementptr i8, ptr addrspace(1) %contexts, i64 0
%n0.weights = getelementptr i8, ptr addrspace(1) %weights, i64 0
%n0.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3128
%n0.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 0
%n0.schedule.0 = load i32, ptr addrspace(1) %n0.schedule.0.ptr, align 4
%n0.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 1
%n0.schedule.1 = load i32, ptr addrspace(1) %n0.schedule.1.ptr, align 4
%n0.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 2
%n0.schedule.2 = load i32, ptr addrspace(1) %n0.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %samples, ptr addrspace(1) %n0.weights, ptr addrspace(1) %n0.value, ptr addrspace(1) %samples, i32 %rows, i32 477, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n0.schedule.0, i32 %n0.schedule.1, i32 %n0.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n1.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n1.value = getelementptr i8, ptr addrspace(1) %values, i64 64
%n1.context = getelementptr i8, ptr addrspace(1) %contexts, i64 40
%n1.weights = getelementptr i8, ptr addrspace(1) %weights, i64 30592
%n1.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3168
%n1.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n1.schedule.base, i32 0
%n1.schedule.0 = load i32, ptr addrspace(1) %n1.schedule.0.ptr, align 4
%n1.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n1.schedule.base, i32 1
%n1.schedule.1 = load i32, ptr addrspace(1) %n1.schedule.1.ptr, align 4
%n1.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n1.schedule.base, i32 2
%n1.schedule.2 = load i32, ptr addrspace(1) %n1.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n1.source, ptr addrspace(1) %n1.weights, ptr addrspace(1) %n1.value, ptr addrspace(1) %n1.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n1.schedule.0, i32 %n1.schedule.1, i32 %n1.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n2.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n2.value = getelementptr i8, ptr addrspace(1) %values, i64 128
%n2.context = getelementptr i8, ptr addrspace(1) %contexts, i64 80
%n2.weights = getelementptr i8, ptr addrspace(1) %weights, i64 31680
%n2.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3208
%n2.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 0
%n2.schedule.0 = load i32, ptr addrspace(1) %n2.schedule.0.ptr, align 4
%n2.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 1
%n2.schedule.1 = load i32, ptr addrspace(1) %n2.schedule.1.ptr, align 4
%n2.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 2
%n2.schedule.2 = load i32, ptr addrspace(1) %n2.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n2.source, ptr addrspace(1) %n2.weights, ptr addrspace(1) %n2.value, ptr addrspace(1) %n2.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n2.schedule.0, i32 %n2.schedule.1, i32 %n2.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n3.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n3.value = getelementptr i8, ptr addrspace(1) %values, i64 192
%n3.context = getelementptr i8, ptr addrspace(1) %contexts, i64 120
%n3.weights = getelementptr i8, ptr addrspace(1) %weights, i64 32768
%n3.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3248
%n3.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n3.schedule.base, i32 0
%n3.schedule.0 = load i32, ptr addrspace(1) %n3.schedule.0.ptr, align 4
%n3.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n3.schedule.base, i32 1
%n3.schedule.1 = load i32, ptr addrspace(1) %n3.schedule.1.ptr, align 4
%n3.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n3.schedule.base, i32 2
%n3.schedule.2 = load i32, ptr addrspace(1) %n3.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n3.source, ptr addrspace(1) %n3.weights, ptr addrspace(1) %n3.value, ptr addrspace(1) %n3.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n3.schedule.0, i32 %n3.schedule.1, i32 %n3.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n4.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n4.value = getelementptr i8, ptr addrspace(1) %values, i64 256
%n4.context = getelementptr i8, ptr addrspace(1) %contexts, i64 160
%n4.weights = getelementptr i8, ptr addrspace(1) %weights, i64 33856
%n4.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3288
%n4.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n4.schedule.base, i32 0
%n4.schedule.0 = load i32, ptr addrspace(1) %n4.schedule.0.ptr, align 4
%n4.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n4.schedule.base, i32 1
%n4.schedule.1 = load i32, ptr addrspace(1) %n4.schedule.1.ptr, align 4
%n4.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n4.schedule.base, i32 2
%n4.schedule.2 = load i32, ptr addrspace(1) %n4.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n4.source, ptr addrspace(1) %n4.weights, ptr addrspace(1) %n4.value, ptr addrspace(1) %n4.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n4.schedule.0, i32 %n4.schedule.1, i32 %n4.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n5.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n5.value = getelementptr i8, ptr addrspace(1) %values, i64 320
%n5.context = getelementptr i8, ptr addrspace(1) %contexts, i64 200
%n5.weights = getelementptr i8, ptr addrspace(1) %weights, i64 34944
%n5.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3328
%n5.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n5.schedule.base, i32 0
%n5.schedule.0 = load i32, ptr addrspace(1) %n5.schedule.0.ptr, align 4
%n5.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n5.schedule.base, i32 1
%n5.schedule.1 = load i32, ptr addrspace(1) %n5.schedule.1.ptr, align 4
%n5.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n5.schedule.base, i32 2
%n5.schedule.2 = load i32, ptr addrspace(1) %n5.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n5.source, ptr addrspace(1) %n5.weights, ptr addrspace(1) %n5.value, ptr addrspace(1) %n5.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n5.schedule.0, i32 %n5.schedule.1, i32 %n5.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n6.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n6.value = getelementptr i8, ptr addrspace(1) %values, i64 384
%n6.context = getelementptr i8, ptr addrspace(1) %contexts, i64 240
%n6.weights = getelementptr i8, ptr addrspace(1) %weights, i64 36032
%n6.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3368
%n6.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n6.schedule.base, i32 0
%n6.schedule.0 = load i32, ptr addrspace(1) %n6.schedule.0.ptr, align 4
%n6.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n6.schedule.base, i32 1
%n6.schedule.1 = load i32, ptr addrspace(1) %n6.schedule.1.ptr, align 4
%n6.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n6.schedule.base, i32 2
%n6.schedule.2 = load i32, ptr addrspace(1) %n6.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n6.source, ptr addrspace(1) %n6.weights, ptr addrspace(1) %n6.value, ptr addrspace(1) %n6.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n6.schedule.0, i32 %n6.schedule.1, i32 %n6.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n7.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n7.value = getelementptr i8, ptr addrspace(1) %values, i64 448
%n7.context = getelementptr i8, ptr addrspace(1) %contexts, i64 280
%n7.weights = getelementptr i8, ptr addrspace(1) %weights, i64 37120
%n7.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3408
%n7.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n7.schedule.base, i32 0
%n7.schedule.0 = load i32, ptr addrspace(1) %n7.schedule.0.ptr, align 4
%n7.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n7.schedule.base, i32 1
%n7.schedule.1 = load i32, ptr addrspace(1) %n7.schedule.1.ptr, align 4
%n7.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n7.schedule.base, i32 2
%n7.schedule.2 = load i32, ptr addrspace(1) %n7.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n7.source, ptr addrspace(1) %n7.weights, ptr addrspace(1) %n7.value, ptr addrspace(1) %n7.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n7.schedule.0, i32 %n7.schedule.1, i32 %n7.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n8.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n8.value = getelementptr i8, ptr addrspace(1) %values, i64 512
%n8.context = getelementptr i8, ptr addrspace(1) %contexts, i64 320
%n8.weights = getelementptr i8, ptr addrspace(1) %weights, i64 38208
%n8.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3448
%n8.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n8.schedule.base, i32 0
%n8.schedule.0 = load i32, ptr addrspace(1) %n8.schedule.0.ptr, align 4
%n8.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n8.schedule.base, i32 1
%n8.schedule.1 = load i32, ptr addrspace(1) %n8.schedule.1.ptr, align 4
%n8.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n8.schedule.base, i32 2
%n8.schedule.2 = load i32, ptr addrspace(1) %n8.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n8.source, ptr addrspace(1) %n8.weights, ptr addrspace(1) %n8.value, ptr addrspace(1) %n8.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n8.schedule.0, i32 %n8.schedule.1, i32 %n8.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n9.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n9.value = getelementptr i8, ptr addrspace(1) %values, i64 576
%n9.context = getelementptr i8, ptr addrspace(1) %contexts, i64 360
%n9.weights = getelementptr i8, ptr addrspace(1) %weights, i64 39296
%n9.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3488
%n9.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n9.schedule.base, i32 0
%n9.schedule.0 = load i32, ptr addrspace(1) %n9.schedule.0.ptr, align 4
%n9.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n9.schedule.base, i32 1
%n9.schedule.1 = load i32, ptr addrspace(1) %n9.schedule.1.ptr, align 4
%n9.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n9.schedule.base, i32 2
%n9.schedule.2 = load i32, ptr addrspace(1) %n9.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n9.source, ptr addrspace(1) %n9.weights, ptr addrspace(1) %n9.value, ptr addrspace(1) %n9.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n9.schedule.0, i32 %n9.schedule.1, i32 %n9.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n10.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n10.value = getelementptr i8, ptr addrspace(1) %values, i64 640
%n10.context = getelementptr i8, ptr addrspace(1) %contexts, i64 400
%n10.weights = getelementptr i8, ptr addrspace(1) %weights, i64 40384
%n10.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3528
%n10.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n10.schedule.base, i32 0
%n10.schedule.0 = load i32, ptr addrspace(1) %n10.schedule.0.ptr, align 4
%n10.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n10.schedule.base, i32 1
%n10.schedule.1 = load i32, ptr addrspace(1) %n10.schedule.1.ptr, align 4
%n10.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n10.schedule.base, i32 2
%n10.schedule.2 = load i32, ptr addrspace(1) %n10.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n10.source, ptr addrspace(1) %n10.weights, ptr addrspace(1) %n10.value, ptr addrspace(1) %n10.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n10.schedule.0, i32 %n10.schedule.1, i32 %n10.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n11.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n11.value = getelementptr i8, ptr addrspace(1) %values, i64 704
%n11.context = getelementptr i8, ptr addrspace(1) %contexts, i64 440
%n11.weights = getelementptr i8, ptr addrspace(1) %weights, i64 41472
%n11.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3568
%n11.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n11.schedule.base, i32 0
%n11.schedule.0 = load i32, ptr addrspace(1) %n11.schedule.0.ptr, align 4
%n11.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n11.schedule.base, i32 1
%n11.schedule.1 = load i32, ptr addrspace(1) %n11.schedule.1.ptr, align 4
%n11.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n11.schedule.base, i32 2
%n11.schedule.2 = load i32, ptr addrspace(1) %n11.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n11.source, ptr addrspace(1) %n11.weights, ptr addrspace(1) %n11.value, ptr addrspace(1) %n11.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n11.schedule.0, i32 %n11.schedule.1, i32 %n11.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n12.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n12.value = getelementptr i8, ptr addrspace(1) %values, i64 768
%n12.context = getelementptr i8, ptr addrspace(1) %contexts, i64 480
%n12.weights = getelementptr i8, ptr addrspace(1) %weights, i64 42560
%n12.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3608
%n12.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n12.schedule.base, i32 0
%n12.schedule.0 = load i32, ptr addrspace(1) %n12.schedule.0.ptr, align 4
%n12.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n12.schedule.base, i32 1
%n12.schedule.1 = load i32, ptr addrspace(1) %n12.schedule.1.ptr, align 4
%n12.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n12.schedule.base, i32 2
%n12.schedule.2 = load i32, ptr addrspace(1) %n12.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n12.source, ptr addrspace(1) %n12.weights, ptr addrspace(1) %n12.value, ptr addrspace(1) %n12.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n12.schedule.0, i32 %n12.schedule.1, i32 %n12.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n13.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n13.value = getelementptr i8, ptr addrspace(1) %values, i64 832
%n13.context = getelementptr i8, ptr addrspace(1) %contexts, i64 520
%n13.weights = getelementptr i8, ptr addrspace(1) %weights, i64 43648
%n13.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3648
%n13.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n13.schedule.base, i32 0
%n13.schedule.0 = load i32, ptr addrspace(1) %n13.schedule.0.ptr, align 4
%n13.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n13.schedule.base, i32 1
%n13.schedule.1 = load i32, ptr addrspace(1) %n13.schedule.1.ptr, align 4
%n13.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n13.schedule.base, i32 2
%n13.schedule.2 = load i32, ptr addrspace(1) %n13.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n13.source, ptr addrspace(1) %n13.weights, ptr addrspace(1) %n13.value, ptr addrspace(1) %n13.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n13.schedule.0, i32 %n13.schedule.1, i32 %n13.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n14.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n14.value = getelementptr i8, ptr addrspace(1) %values, i64 896
%n14.context = getelementptr i8, ptr addrspace(1) %contexts, i64 560
%n14.weights = getelementptr i8, ptr addrspace(1) %weights, i64 44736
%n14.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3688
%n14.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n14.schedule.base, i32 0
%n14.schedule.0 = load i32, ptr addrspace(1) %n14.schedule.0.ptr, align 4
%n14.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n14.schedule.base, i32 1
%n14.schedule.1 = load i32, ptr addrspace(1) %n14.schedule.1.ptr, align 4
%n14.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n14.schedule.base, i32 2
%n14.schedule.2 = load i32, ptr addrspace(1) %n14.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n14.source, ptr addrspace(1) %n14.weights, ptr addrspace(1) %n14.value, ptr addrspace(1) %n14.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n14.schedule.0, i32 %n14.schedule.1, i32 %n14.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n15.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n15.value = getelementptr i8, ptr addrspace(1) %values, i64 960
%n15.context = getelementptr i8, ptr addrspace(1) %contexts, i64 600
%n15.weights = getelementptr i8, ptr addrspace(1) %weights, i64 45824
%n15.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3728
%n15.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n15.schedule.base, i32 0
%n15.schedule.0 = load i32, ptr addrspace(1) %n15.schedule.0.ptr, align 4
%n15.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n15.schedule.base, i32 1
%n15.schedule.1 = load i32, ptr addrspace(1) %n15.schedule.1.ptr, align 4
%n15.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n15.schedule.base, i32 2
%n15.schedule.2 = load i32, ptr addrspace(1) %n15.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n15.source, ptr addrspace(1) %n15.weights, ptr addrspace(1) %n15.value, ptr addrspace(1) %n15.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n15.schedule.0, i32 %n15.schedule.1, i32 %n15.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n16.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n16.value = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n16.context = getelementptr i8, ptr addrspace(1) %contexts, i64 640
%n16.weights = getelementptr i8, ptr addrspace(1) %weights, i64 46912
%n16.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3768
%n16.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n16.schedule.base, i32 0
%n16.schedule.0 = load i32, ptr addrspace(1) %n16.schedule.0.ptr, align 4
%n16.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n16.schedule.base, i32 1
%n16.schedule.1 = load i32, ptr addrspace(1) %n16.schedule.1.ptr, align 4
%n16.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n16.schedule.base, i32 2
%n16.schedule.2 = load i32, ptr addrspace(1) %n16.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n16.source, ptr addrspace(1) %n16.weights, ptr addrspace(1) %n16.value, ptr addrspace(1) %n16.source, i32 %rows, i32 16, i32 1, i32 16, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n16.schedule.0, i32 %n16.schedule.1, i32 %n16.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n17.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n17.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n17.value = getelementptr i8, ptr addrspace(1) %values, i64 1088
%n17.context = getelementptr i8, ptr addrspace(1) %contexts, i64 680
%n17.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n17.scalar.entry
n17.scalar.entry:
%n17.scalar.count = mul i32 %rows, 16
br label %n17.scalar.loop
n17.scalar.loop:
%n17.scalar.p = phi i32 [ %tid, %n17.scalar.entry ], [ %n17.scalar.next, %n17.scalar.step ]
%n17.scalar.more = icmp ult i32 %n17.scalar.p, %n17.scalar.count
br i1 %n17.scalar.more, label %n17.scalar.body, label %n17.scalar.done
n17.scalar.body:
%n17.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n17.source, i32 %n17.scalar.p
%n17.scalar.first = load float, ptr addrspace(1) %n17.scalar.first.ptr, align 4
%n17.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n17.second, i32 %n17.scalar.p
%n17.scalar.second = load float, ptr addrspace(1) %n17.scalar.second.ptr, align 4
%n17.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n17.scalar.first, float %n17.scalar.second)
%n17.scalar.scalar.0 = call float @recipe.from.u1(i1 %n17.scalar.scalar.0.condition)
%n17.scalar.scalar.2 = call float @recipe.sub(float 0x3FF0000000000000, float %n17.scalar.scalar.0)
%n17.scalar.scalar.3.condition = call i1 @recipe.ogt(float %n17.scalar.scalar.0, float 0x0000000000000000)
%n17.scalar.scalar.3 = select i1 %n17.scalar.scalar.3.condition, float %n17.scalar.first, float 0x0000000000000000
%n17.scalar.scalar.4.condition = call i1 @recipe.ogt(float %n17.scalar.scalar.2, float 0x0000000000000000)
%n17.scalar.scalar.4 = select i1 %n17.scalar.scalar.4.condition, float %n17.scalar.second, float 0x0000000000000000
%n17.scalar.scalar.5 = call float @recipe.add(float %n17.scalar.scalar.3, float %n17.scalar.scalar.4)
%n17.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n17.value, i32 %n17.scalar.p
store float %n17.scalar.scalar.5, ptr addrspace(1) %n17.scalar.output.ptr, align 4
br label %n17.scalar.step
n17.scalar.step:
%n17.scalar.next = add i32 %n17.scalar.p, %threads
br label %n17.scalar.loop
n17.scalar.done:
call void @grid_barrier(i32 %threads)%n18.source = getelementptr i8, ptr addrspace(1) %values, i64 1088
%n18.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n18.value = getelementptr i8, ptr addrspace(1) %values, i64 1152
%n18.context = getelementptr i8, ptr addrspace(1) %contexts, i64 688
%n18.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n18.scalar.entry
n18.scalar.entry:
%n18.scalar.count = mul i32 %rows, 16
br label %n18.scalar.loop
n18.scalar.loop:
%n18.scalar.p = phi i32 [ %tid, %n18.scalar.entry ], [ %n18.scalar.next, %n18.scalar.step ]
%n18.scalar.more = icmp ult i32 %n18.scalar.p, %n18.scalar.count
br i1 %n18.scalar.more, label %n18.scalar.body, label %n18.scalar.done
n18.scalar.body:
%n18.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n18.source, i32 %n18.scalar.p
%n18.scalar.first = load float, ptr addrspace(1) %n18.scalar.first.ptr, align 4
%n18.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n18.second, i32 %n18.scalar.p
%n18.scalar.second = load float, ptr addrspace(1) %n18.scalar.second.ptr, align 4
%n18.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n18.scalar.first, float %n18.scalar.second)
%n18.scalar.scalar.0 = call float @recipe.from.u1(i1 %n18.scalar.scalar.0.condition)
%n18.scalar.scalar.2 = call float @recipe.sub(float 0x3FF0000000000000, float %n18.scalar.scalar.0)
%n18.scalar.scalar.3.condition = call i1 @recipe.ogt(float %n18.scalar.scalar.0, float 0x0000000000000000)
%n18.scalar.scalar.3 = select i1 %n18.scalar.scalar.3.condition, float %n18.scalar.first, float 0x0000000000000000
%n18.scalar.scalar.4.condition = call i1 @recipe.ogt(float %n18.scalar.scalar.2, float 0x0000000000000000)
%n18.scalar.scalar.4 = select i1 %n18.scalar.scalar.4.condition, float %n18.scalar.second, float 0x0000000000000000
%n18.scalar.scalar.5 = call float @recipe.add(float %n18.scalar.scalar.3, float %n18.scalar.scalar.4)
%n18.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n18.value, i32 %n18.scalar.p
store float %n18.scalar.scalar.5, ptr addrspace(1) %n18.scalar.output.ptr, align 4
br label %n18.scalar.step
n18.scalar.step:
%n18.scalar.next = add i32 %n18.scalar.p, %threads
br label %n18.scalar.loop
n18.scalar.done:
call void @grid_barrier(i32 %threads)%n19.source = getelementptr i8, ptr addrspace(1) %values, i64 1152
%n19.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n19.value = getelementptr i8, ptr addrspace(1) %values, i64 1216
%n19.context = getelementptr i8, ptr addrspace(1) %contexts, i64 696
%n19.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n19.scalar.entry
n19.scalar.entry:
%n19.scalar.count = mul i32 %rows, 16
br label %n19.scalar.loop
n19.scalar.loop:
%n19.scalar.p = phi i32 [ %tid, %n19.scalar.entry ], [ %n19.scalar.next, %n19.scalar.step ]
%n19.scalar.more = icmp ult i32 %n19.scalar.p, %n19.scalar.count
br i1 %n19.scalar.more, label %n19.scalar.body, label %n19.scalar.done
n19.scalar.body:
%n19.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n19.source, i32 %n19.scalar.p
%n19.scalar.first = load float, ptr addrspace(1) %n19.scalar.first.ptr, align 4
%n19.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n19.second, i32 %n19.scalar.p
%n19.scalar.second = load float, ptr addrspace(1) %n19.scalar.second.ptr, align 4
%n19.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n19.scalar.first, float %n19.scalar.second)
%n19.scalar.scalar.0 = call float @recipe.from.u1(i1 %n19.scalar.scalar.0.condition)
%n19.scalar.scalar.2 = call float @recipe.sub(float 0x3FF0000000000000, float %n19.scalar.scalar.0)
%n19.scalar.scalar.3.condition = call i1 @recipe.ogt(float %n19.scalar.scalar.0, float 0x0000000000000000)
%n19.scalar.scalar.3 = select i1 %n19.scalar.scalar.3.condition, float %n19.scalar.first, float 0x0000000000000000
%n19.scalar.scalar.4.condition = call i1 @recipe.ogt(float %n19.scalar.scalar.2, float 0x0000000000000000)
%n19.scalar.scalar.4 = select i1 %n19.scalar.scalar.4.condition, float %n19.scalar.second, float 0x0000000000000000
%n19.scalar.scalar.5 = call float @recipe.add(float %n19.scalar.scalar.3, float %n19.scalar.scalar.4)
%n19.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n19.value, i32 %n19.scalar.p
store float %n19.scalar.scalar.5, ptr addrspace(1) %n19.scalar.output.ptr, align 4
br label %n19.scalar.step
n19.scalar.step:
%n19.scalar.next = add i32 %n19.scalar.p, %threads
br label %n19.scalar.loop
n19.scalar.done:
call void @grid_barrier(i32 %threads)%n20.source = getelementptr i8, ptr addrspace(1) %values, i64 1216
%n20.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n20.value = getelementptr i8, ptr addrspace(1) %values, i64 1280
%n20.context = getelementptr i8, ptr addrspace(1) %contexts, i64 704
%n20.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n20.scalar.entry
n20.scalar.entry:
%n20.scalar.count = mul i32 %rows, 16
br label %n20.scalar.loop
n20.scalar.loop:
%n20.scalar.p = phi i32 [ %tid, %n20.scalar.entry ], [ %n20.scalar.next, %n20.scalar.step ]
%n20.scalar.more = icmp ult i32 %n20.scalar.p, %n20.scalar.count
br i1 %n20.scalar.more, label %n20.scalar.body, label %n20.scalar.done
n20.scalar.body:
%n20.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n20.source, i32 %n20.scalar.p
%n20.scalar.first = load float, ptr addrspace(1) %n20.scalar.first.ptr, align 4
%n20.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n20.second, i32 %n20.scalar.p
%n20.scalar.second = load float, ptr addrspace(1) %n20.scalar.second.ptr, align 4
%n20.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n20.scalar.first, float %n20.scalar.second)
%n20.scalar.scalar.0 = call float @recipe.from.u1(i1 %n20.scalar.scalar.0.condition)
%n20.scalar.scalar.2 = call float @recipe.sub(float 0x3FF0000000000000, float %n20.scalar.scalar.0)
%n20.scalar.scalar.3.condition = call i1 @recipe.ogt(float %n20.scalar.scalar.0, float 0x0000000000000000)
%n20.scalar.scalar.3 = select i1 %n20.scalar.scalar.3.condition, float %n20.scalar.first, float 0x0000000000000000
%n20.scalar.scalar.4.condition = call i1 @recipe.ogt(float %n20.scalar.scalar.2, float 0x0000000000000000)
%n20.scalar.scalar.4 = select i1 %n20.scalar.scalar.4.condition, float %n20.scalar.second, float 0x0000000000000000
%n20.scalar.scalar.5 = call float @recipe.add(float %n20.scalar.scalar.3, float %n20.scalar.scalar.4)
%n20.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n20.value, i32 %n20.scalar.p
store float %n20.scalar.scalar.5, ptr addrspace(1) %n20.scalar.output.ptr, align 4
br label %n20.scalar.step
n20.scalar.step:
%n20.scalar.next = add i32 %n20.scalar.p, %threads
br label %n20.scalar.loop
n20.scalar.done:
call void @grid_barrier(i32 %threads)%n21.source = getelementptr i8, ptr addrspace(1) %values, i64 1280
%n21.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n21.value = getelementptr i8, ptr addrspace(1) %values, i64 1344
%n21.context = getelementptr i8, ptr addrspace(1) %contexts, i64 712
%n21.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n21.scalar.entry
n21.scalar.entry:
%n21.scalar.count = mul i32 %rows, 16
br label %n21.scalar.loop
n21.scalar.loop:
%n21.scalar.p = phi i32 [ %tid, %n21.scalar.entry ], [ %n21.scalar.next, %n21.scalar.step ]
%n21.scalar.more = icmp ult i32 %n21.scalar.p, %n21.scalar.count
br i1 %n21.scalar.more, label %n21.scalar.body, label %n21.scalar.done
n21.scalar.body:
%n21.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n21.source, i32 %n21.scalar.p
%n21.scalar.first = load float, ptr addrspace(1) %n21.scalar.first.ptr, align 4
%n21.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n21.second, i32 %n21.scalar.p
%n21.scalar.second = load float, ptr addrspace(1) %n21.scalar.second.ptr, align 4
%n21.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n21.scalar.first, float %n21.scalar.second)
%n21.scalar.scalar.0 = call float @recipe.from.u1(i1 %n21.scalar.scalar.0.condition)
%n21.scalar.scalar.2 = call float @recipe.sub(float 0x3FF0000000000000, float %n21.scalar.scalar.0)
%n21.scalar.scalar.3.condition = call i1 @recipe.ogt(float %n21.scalar.scalar.0, float 0x0000000000000000)
%n21.scalar.scalar.3 = select i1 %n21.scalar.scalar.3.condition, float %n21.scalar.first, float 0x0000000000000000
%n21.scalar.scalar.4.condition = call i1 @recipe.ogt(float %n21.scalar.scalar.2, float 0x0000000000000000)
%n21.scalar.scalar.4 = select i1 %n21.scalar.scalar.4.condition, float %n21.scalar.second, float 0x0000000000000000
%n21.scalar.scalar.5 = call float @recipe.add(float %n21.scalar.scalar.3, float %n21.scalar.scalar.4)
%n21.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n21.value, i32 %n21.scalar.p
store float %n21.scalar.scalar.5, ptr addrspace(1) %n21.scalar.output.ptr, align 4
br label %n21.scalar.step
n21.scalar.step:
%n21.scalar.next = add i32 %n21.scalar.p, %threads
br label %n21.scalar.loop
n21.scalar.done:
call void @grid_barrier(i32 %threads)%n22.source = getelementptr i8, ptr addrspace(1) %values, i64 1344
%n22.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n22.value = getelementptr i8, ptr addrspace(1) %values, i64 1408
%n22.context = getelementptr i8, ptr addrspace(1) %contexts, i64 720
%n22.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n22.scalar.entry
n22.scalar.entry:
%n22.scalar.count = mul i32 %rows, 16
br label %n22.scalar.loop
n22.scalar.loop:
%n22.scalar.p = phi i32 [ %tid, %n22.scalar.entry ], [ %n22.scalar.next, %n22.scalar.step ]
%n22.scalar.more = icmp ult i32 %n22.scalar.p, %n22.scalar.count
br i1 %n22.scalar.more, label %n22.scalar.body, label %n22.scalar.done
n22.scalar.body:
%n22.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n22.source, i32 %n22.scalar.p
%n22.scalar.first = load float, ptr addrspace(1) %n22.scalar.first.ptr, align 4
%n22.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n22.second, i32 %n22.scalar.p
%n22.scalar.second = load float, ptr addrspace(1) %n22.scalar.second.ptr, align 4
%n22.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n22.scalar.first, float %n22.scalar.second)
%n22.scalar.scalar.0 = call float @recipe.from.u1(i1 %n22.scalar.scalar.0.condition)
%n22.scalar.scalar.2 = call float @recipe.sub(float 0x3FF0000000000000, float %n22.scalar.scalar.0)
%n22.scalar.scalar.3.condition = call i1 @recipe.ogt(float %n22.scalar.scalar.0, float 0x0000000000000000)
%n22.scalar.scalar.3 = select i1 %n22.scalar.scalar.3.condition, float %n22.scalar.first, float 0x0000000000000000
%n22.scalar.scalar.4.condition = call i1 @recipe.ogt(float %n22.scalar.scalar.2, float 0x0000000000000000)
%n22.scalar.scalar.4 = select i1 %n22.scalar.scalar.4.condition, float %n22.scalar.second, float 0x0000000000000000
%n22.scalar.scalar.5 = call float @recipe.add(float %n22.scalar.scalar.3, float %n22.scalar.scalar.4)
%n22.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n22.value, i32 %n22.scalar.p
store float %n22.scalar.scalar.5, ptr addrspace(1) %n22.scalar.output.ptr, align 4
br label %n22.scalar.step
n22.scalar.step:
%n22.scalar.next = add i32 %n22.scalar.p, %threads
br label %n22.scalar.loop
n22.scalar.done:
call void @grid_barrier(i32 %threads)%n23.source = getelementptr i8, ptr addrspace(1) %values, i64 1408
%n23.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n23.value = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n23.context = getelementptr i8, ptr addrspace(1) %contexts, i64 728
%n23.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n23.scalar.entry
n23.scalar.entry:
%n23.scalar.count = mul i32 %rows, 16
br label %n23.scalar.loop
n23.scalar.loop:
%n23.scalar.p = phi i32 [ %tid, %n23.scalar.entry ], [ %n23.scalar.next, %n23.scalar.step ]
%n23.scalar.more = icmp ult i32 %n23.scalar.p, %n23.scalar.count
br i1 %n23.scalar.more, label %n23.scalar.body, label %n23.scalar.done
n23.scalar.body:
%n23.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n23.source, i32 %n23.scalar.p
%n23.scalar.first = load float, ptr addrspace(1) %n23.scalar.first.ptr, align 4
%n23.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n23.second, i32 %n23.scalar.p
%n23.scalar.second = load float, ptr addrspace(1) %n23.scalar.second.ptr, align 4
%n23.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n23.scalar.first, float %n23.scalar.second)
%n23.scalar.scalar.0 = call float @recipe.from.u1(i1 %n23.scalar.scalar.0.condition)
%n23.scalar.scalar.2 = call float @recipe.sub(float 0x3FF0000000000000, float %n23.scalar.scalar.0)
%n23.scalar.scalar.3.condition = call i1 @recipe.ogt(float %n23.scalar.scalar.0, float 0x0000000000000000)
%n23.scalar.scalar.3 = select i1 %n23.scalar.scalar.3.condition, float %n23.scalar.first, float 0x0000000000000000
%n23.scalar.scalar.4.condition = call i1 @recipe.ogt(float %n23.scalar.scalar.2, float 0x0000000000000000)
%n23.scalar.scalar.4 = select i1 %n23.scalar.scalar.4.condition, float %n23.scalar.second, float 0x0000000000000000
%n23.scalar.scalar.5 = call float @recipe.add(float %n23.scalar.scalar.3, float %n23.scalar.scalar.4)
%n23.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n23.value, i32 %n23.scalar.p
store float %n23.scalar.scalar.5, ptr addrspace(1) %n23.scalar.output.ptr, align 4
br label %n23.scalar.step
n23.scalar.step:
%n23.scalar.next = add i32 %n23.scalar.p, %threads
br label %n23.scalar.loop
n23.scalar.done:
call void @grid_barrier(i32 %threads)%n24.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n24.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n24.value = getelementptr i8, ptr addrspace(1) %values, i64 1536
%n24.context = getelementptr i8, ptr addrspace(1) %contexts, i64 736
%n24.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n24.scalar.entry
n24.scalar.entry:
%n24.scalar.count = mul i32 %rows, 16
br label %n24.scalar.loop
n24.scalar.loop:
%n24.scalar.p = phi i32 [ %tid, %n24.scalar.entry ], [ %n24.scalar.next, %n24.scalar.step ]
%n24.scalar.more = icmp ult i32 %n24.scalar.p, %n24.scalar.count
br i1 %n24.scalar.more, label %n24.scalar.body, label %n24.scalar.done
n24.scalar.body:
%n24.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n24.source, i32 %n24.scalar.p
%n24.scalar.first = load float, ptr addrspace(1) %n24.scalar.first.ptr, align 4
%n24.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n24.second, i32 %n24.scalar.p
%n24.scalar.second = load float, ptr addrspace(1) %n24.scalar.second.ptr, align 4
%n24.scalar.scalar.0 = call float @recipe.sub(float %n24.scalar.first, float %n24.scalar.second)
%n24.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n24.value, i32 %n24.scalar.p
store float %n24.scalar.scalar.0, ptr addrspace(1) %n24.scalar.output.ptr, align 4
br label %n24.scalar.step
n24.scalar.step:
%n24.scalar.next = add i32 %n24.scalar.p, %threads
br label %n24.scalar.loop
n24.scalar.done:
call void @grid_barrier(i32 %threads)%n25.source = getelementptr i8, ptr addrspace(1) %values, i64 1536
%n25.value = getelementptr i8, ptr addrspace(1) %values, i64 1600
%n25.context = getelementptr i8, ptr addrspace(1) %contexts, i64 744
%n25.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n25.scalar.entry
n25.scalar.entry:
%n25.scalar.count = mul i32 %rows, 16
br label %n25.scalar.loop
n25.scalar.loop:
%n25.scalar.p = phi i32 [ %tid, %n25.scalar.entry ], [ %n25.scalar.next, %n25.scalar.step ]
%n25.scalar.more = icmp ult i32 %n25.scalar.p, %n25.scalar.count
br i1 %n25.scalar.more, label %n25.scalar.body, label %n25.scalar.done
n25.scalar.body:
%n25.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n25.source, i32 %n25.scalar.p
%n25.scalar.first = load float, ptr addrspace(1) %n25.scalar.first.ptr, align 4
%n25.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n25.scalar.first, float 0x0000000000000000)
%n25.scalar.scalar.2 = call float @recipe.from.u1(i1 %n25.scalar.scalar.2.condition)
%n25.scalar.scalar.3 = call float @recipe.exp(float %n25.scalar.first)
%n25.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n25.value, i32 %n25.scalar.p
store float %n25.scalar.scalar.3, ptr addrspace(1) %n25.scalar.output.ptr, align 4
br label %n25.scalar.step
n25.scalar.step:
%n25.scalar.next = add i32 %n25.scalar.p, %threads
br label %n25.scalar.loop
n25.scalar.done:
call void @grid_barrier(i32 %threads)%n26.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n26.value = getelementptr i8, ptr addrspace(1) %values, i64 1664
%n26.context = getelementptr i8, ptr addrspace(1) %contexts, i64 752
%n26.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n26.scalar.entry
n26.scalar.entry:
%n26.scalar.count = mul i32 %rows, 16
br label %n26.scalar.loop
n26.scalar.loop:
%n26.scalar.p = phi i32 [ %tid, %n26.scalar.entry ], [ %n26.scalar.next, %n26.scalar.step ]
%n26.scalar.more = icmp ult i32 %n26.scalar.p, %n26.scalar.count
br i1 %n26.scalar.more, label %n26.scalar.body, label %n26.scalar.done
n26.scalar.body:
%n26.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n26.source, i32 %n26.scalar.p
%n26.scalar.first = load float, ptr addrspace(1) %n26.scalar.first.ptr, align 4
%n26.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n26.value, i32 %n26.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n26.scalar.output.ptr, align 4
br label %n26.scalar.step
n26.scalar.step:
%n26.scalar.next = add i32 %n26.scalar.p, %threads
br label %n26.scalar.loop
n26.scalar.done:
call void @grid_barrier(i32 %threads)%n27.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n27.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n27.value = getelementptr i8, ptr addrspace(1) %values, i64 1728
%n27.context = getelementptr i8, ptr addrspace(1) %contexts, i64 760
%n27.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n27.scalar.entry
n27.scalar.entry:
%n27.scalar.count = mul i32 %rows, 16
br label %n27.scalar.loop
n27.scalar.loop:
%n27.scalar.p = phi i32 [ %tid, %n27.scalar.entry ], [ %n27.scalar.next, %n27.scalar.step ]
%n27.scalar.more = icmp ult i32 %n27.scalar.p, %n27.scalar.count
br i1 %n27.scalar.more, label %n27.scalar.body, label %n27.scalar.done
n27.scalar.body:
%n27.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n27.source, i32 %n27.scalar.p
%n27.scalar.first = load float, ptr addrspace(1) %n27.scalar.first.ptr, align 4
%n27.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n27.second, i32 %n27.scalar.p
%n27.scalar.second = load float, ptr addrspace(1) %n27.scalar.second.ptr, align 4
%n27.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n27.scalar.first, float %n27.scalar.second)
%n27.scalar.scalar.0 = call float @recipe.from.u1(i1 %n27.scalar.scalar.0.condition)
%n27.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n27.value, i32 %n27.scalar.p
store float %n27.scalar.scalar.0, ptr addrspace(1) %n27.scalar.output.ptr, align 4
br label %n27.scalar.step
n27.scalar.step:
%n27.scalar.next = add i32 %n27.scalar.p, %threads
br label %n27.scalar.loop
n27.scalar.done:
call void @grid_barrier(i32 %threads)%n28.source = getelementptr i8, ptr addrspace(1) %values, i64 1664
%n28.second = getelementptr i8, ptr addrspace(1) %values, i64 1728
%n28.value = getelementptr i8, ptr addrspace(1) %values, i64 1792
%n28.context = getelementptr i8, ptr addrspace(1) %contexts, i64 768
%n28.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n28.scalar.entry
n28.scalar.entry:
%n28.scalar.count = mul i32 %rows, 16
br label %n28.scalar.loop
n28.scalar.loop:
%n28.scalar.p = phi i32 [ %tid, %n28.scalar.entry ], [ %n28.scalar.next, %n28.scalar.step ]
%n28.scalar.more = icmp ult i32 %n28.scalar.p, %n28.scalar.count
br i1 %n28.scalar.more, label %n28.scalar.body, label %n28.scalar.done
n28.scalar.body:
%n28.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n28.source, i32 %n28.scalar.p
%n28.scalar.first = load float, ptr addrspace(1) %n28.scalar.first.ptr, align 4
%n28.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n28.second, i32 %n28.scalar.p
%n28.scalar.second = load float, ptr addrspace(1) %n28.scalar.second.ptr, align 4
%n28.scalar.scalar.0 = call float @recipe.add(float %n28.scalar.first, float %n28.scalar.second)
%n28.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n28.value, i32 %n28.scalar.p
store float %n28.scalar.scalar.0, ptr addrspace(1) %n28.scalar.output.ptr, align 4
br label %n28.scalar.step
n28.scalar.step:
%n28.scalar.next = add i32 %n28.scalar.p, %threads
br label %n28.scalar.loop
n28.scalar.done:
call void @grid_barrier(i32 %threads)%n29.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n29.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n29.value = getelementptr i8, ptr addrspace(1) %values, i64 1856
%n29.context = getelementptr i8, ptr addrspace(1) %contexts, i64 776
%n29.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n29.scalar.entry
n29.scalar.entry:
%n29.scalar.count = mul i32 %rows, 16
br label %n29.scalar.loop
n29.scalar.loop:
%n29.scalar.p = phi i32 [ %tid, %n29.scalar.entry ], [ %n29.scalar.next, %n29.scalar.step ]
%n29.scalar.more = icmp ult i32 %n29.scalar.p, %n29.scalar.count
br i1 %n29.scalar.more, label %n29.scalar.body, label %n29.scalar.done
n29.scalar.body:
%n29.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n29.source, i32 %n29.scalar.p
%n29.scalar.first = load float, ptr addrspace(1) %n29.scalar.first.ptr, align 4
%n29.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n29.second, i32 %n29.scalar.p
%n29.scalar.second = load float, ptr addrspace(1) %n29.scalar.second.ptr, align 4
%n29.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n29.scalar.first, float %n29.scalar.second)
%n29.scalar.scalar.0 = call float @recipe.from.u1(i1 %n29.scalar.scalar.0.condition)
%n29.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n29.value, i32 %n29.scalar.p
store float %n29.scalar.scalar.0, ptr addrspace(1) %n29.scalar.output.ptr, align 4
br label %n29.scalar.step
n29.scalar.step:
%n29.scalar.next = add i32 %n29.scalar.p, %threads
br label %n29.scalar.loop
n29.scalar.done:
call void @grid_barrier(i32 %threads)%n30.source = getelementptr i8, ptr addrspace(1) %values, i64 1792
%n30.second = getelementptr i8, ptr addrspace(1) %values, i64 1856
%n30.value = getelementptr i8, ptr addrspace(1) %values, i64 1920
%n30.context = getelementptr i8, ptr addrspace(1) %contexts, i64 784
%n30.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n30.scalar.entry
n30.scalar.entry:
%n30.scalar.count = mul i32 %rows, 16
br label %n30.scalar.loop
n30.scalar.loop:
%n30.scalar.p = phi i32 [ %tid, %n30.scalar.entry ], [ %n30.scalar.next, %n30.scalar.step ]
%n30.scalar.more = icmp ult i32 %n30.scalar.p, %n30.scalar.count
br i1 %n30.scalar.more, label %n30.scalar.body, label %n30.scalar.done
n30.scalar.body:
%n30.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n30.source, i32 %n30.scalar.p
%n30.scalar.first = load float, ptr addrspace(1) %n30.scalar.first.ptr, align 4
%n30.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n30.second, i32 %n30.scalar.p
%n30.scalar.second = load float, ptr addrspace(1) %n30.scalar.second.ptr, align 4
%n30.scalar.scalar.0 = call float @recipe.add(float %n30.scalar.first, float %n30.scalar.second)
%n30.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n30.value, i32 %n30.scalar.p
store float %n30.scalar.scalar.0, ptr addrspace(1) %n30.scalar.output.ptr, align 4
br label %n30.scalar.step
n30.scalar.step:
%n30.scalar.next = add i32 %n30.scalar.p, %threads
br label %n30.scalar.loop
n30.scalar.done:
call void @grid_barrier(i32 %threads)%n31.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n31.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n31.value = getelementptr i8, ptr addrspace(1) %values, i64 1984
%n31.context = getelementptr i8, ptr addrspace(1) %contexts, i64 792
%n31.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n31.scalar.entry
n31.scalar.entry:
%n31.scalar.count = mul i32 %rows, 16
br label %n31.scalar.loop
n31.scalar.loop:
%n31.scalar.p = phi i32 [ %tid, %n31.scalar.entry ], [ %n31.scalar.next, %n31.scalar.step ]
%n31.scalar.more = icmp ult i32 %n31.scalar.p, %n31.scalar.count
br i1 %n31.scalar.more, label %n31.scalar.body, label %n31.scalar.done
n31.scalar.body:
%n31.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n31.source, i32 %n31.scalar.p
%n31.scalar.first = load float, ptr addrspace(1) %n31.scalar.first.ptr, align 4
%n31.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n31.second, i32 %n31.scalar.p
%n31.scalar.second = load float, ptr addrspace(1) %n31.scalar.second.ptr, align 4
%n31.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n31.scalar.first, float %n31.scalar.second)
%n31.scalar.scalar.0 = call float @recipe.from.u1(i1 %n31.scalar.scalar.0.condition)
%n31.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n31.value, i32 %n31.scalar.p
store float %n31.scalar.scalar.0, ptr addrspace(1) %n31.scalar.output.ptr, align 4
br label %n31.scalar.step
n31.scalar.step:
%n31.scalar.next = add i32 %n31.scalar.p, %threads
br label %n31.scalar.loop
n31.scalar.done:
call void @grid_barrier(i32 %threads)%n32.source = getelementptr i8, ptr addrspace(1) %values, i64 1920
%n32.second = getelementptr i8, ptr addrspace(1) %values, i64 1984
%n32.value = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n32.context = getelementptr i8, ptr addrspace(1) %contexts, i64 800
%n32.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n32.scalar.entry
n32.scalar.entry:
%n32.scalar.count = mul i32 %rows, 16
br label %n32.scalar.loop
n32.scalar.loop:
%n32.scalar.p = phi i32 [ %tid, %n32.scalar.entry ], [ %n32.scalar.next, %n32.scalar.step ]
%n32.scalar.more = icmp ult i32 %n32.scalar.p, %n32.scalar.count
br i1 %n32.scalar.more, label %n32.scalar.body, label %n32.scalar.done
n32.scalar.body:
%n32.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n32.source, i32 %n32.scalar.p
%n32.scalar.first = load float, ptr addrspace(1) %n32.scalar.first.ptr, align 4
%n32.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n32.second, i32 %n32.scalar.p
%n32.scalar.second = load float, ptr addrspace(1) %n32.scalar.second.ptr, align 4
%n32.scalar.scalar.0 = call float @recipe.add(float %n32.scalar.first, float %n32.scalar.second)
%n32.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n32.value, i32 %n32.scalar.p
store float %n32.scalar.scalar.0, ptr addrspace(1) %n32.scalar.output.ptr, align 4
br label %n32.scalar.step
n32.scalar.step:
%n32.scalar.next = add i32 %n32.scalar.p, %threads
br label %n32.scalar.loop
n32.scalar.done:
call void @grid_barrier(i32 %threads)%n33.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n33.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n33.value = getelementptr i8, ptr addrspace(1) %values, i64 2112
%n33.context = getelementptr i8, ptr addrspace(1) %contexts, i64 808
%n33.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n33.scalar.entry
n33.scalar.entry:
%n33.scalar.count = mul i32 %rows, 16
br label %n33.scalar.loop
n33.scalar.loop:
%n33.scalar.p = phi i32 [ %tid, %n33.scalar.entry ], [ %n33.scalar.next, %n33.scalar.step ]
%n33.scalar.more = icmp ult i32 %n33.scalar.p, %n33.scalar.count
br i1 %n33.scalar.more, label %n33.scalar.body, label %n33.scalar.done
n33.scalar.body:
%n33.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n33.source, i32 %n33.scalar.p
%n33.scalar.first = load float, ptr addrspace(1) %n33.scalar.first.ptr, align 4
%n33.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n33.second, i32 %n33.scalar.p
%n33.scalar.second = load float, ptr addrspace(1) %n33.scalar.second.ptr, align 4
%n33.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n33.scalar.first, float %n33.scalar.second)
%n33.scalar.scalar.0 = call float @recipe.from.u1(i1 %n33.scalar.scalar.0.condition)
%n33.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n33.value, i32 %n33.scalar.p
store float %n33.scalar.scalar.0, ptr addrspace(1) %n33.scalar.output.ptr, align 4
br label %n33.scalar.step
n33.scalar.step:
%n33.scalar.next = add i32 %n33.scalar.p, %threads
br label %n33.scalar.loop
n33.scalar.done:
call void @grid_barrier(i32 %threads)%n34.source = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n34.second = getelementptr i8, ptr addrspace(1) %values, i64 2112
%n34.value = getelementptr i8, ptr addrspace(1) %values, i64 2176
%n34.context = getelementptr i8, ptr addrspace(1) %contexts, i64 816
%n34.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n34.scalar.entry
n34.scalar.entry:
%n34.scalar.count = mul i32 %rows, 16
br label %n34.scalar.loop
n34.scalar.loop:
%n34.scalar.p = phi i32 [ %tid, %n34.scalar.entry ], [ %n34.scalar.next, %n34.scalar.step ]
%n34.scalar.more = icmp ult i32 %n34.scalar.p, %n34.scalar.count
br i1 %n34.scalar.more, label %n34.scalar.body, label %n34.scalar.done
n34.scalar.body:
%n34.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n34.source, i32 %n34.scalar.p
%n34.scalar.first = load float, ptr addrspace(1) %n34.scalar.first.ptr, align 4
%n34.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n34.second, i32 %n34.scalar.p
%n34.scalar.second = load float, ptr addrspace(1) %n34.scalar.second.ptr, align 4
%n34.scalar.scalar.0 = call float @recipe.add(float %n34.scalar.first, float %n34.scalar.second)
%n34.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n34.value, i32 %n34.scalar.p
store float %n34.scalar.scalar.0, ptr addrspace(1) %n34.scalar.output.ptr, align 4
br label %n34.scalar.step
n34.scalar.step:
%n34.scalar.next = add i32 %n34.scalar.p, %threads
br label %n34.scalar.loop
n34.scalar.done:
call void @grid_barrier(i32 %threads)%n35.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n35.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n35.value = getelementptr i8, ptr addrspace(1) %values, i64 2240
%n35.context = getelementptr i8, ptr addrspace(1) %contexts, i64 824
%n35.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n35.scalar.entry
n35.scalar.entry:
%n35.scalar.count = mul i32 %rows, 16
br label %n35.scalar.loop
n35.scalar.loop:
%n35.scalar.p = phi i32 [ %tid, %n35.scalar.entry ], [ %n35.scalar.next, %n35.scalar.step ]
%n35.scalar.more = icmp ult i32 %n35.scalar.p, %n35.scalar.count
br i1 %n35.scalar.more, label %n35.scalar.body, label %n35.scalar.done
n35.scalar.body:
%n35.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n35.source, i32 %n35.scalar.p
%n35.scalar.first = load float, ptr addrspace(1) %n35.scalar.first.ptr, align 4
%n35.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n35.second, i32 %n35.scalar.p
%n35.scalar.second = load float, ptr addrspace(1) %n35.scalar.second.ptr, align 4
%n35.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n35.scalar.first, float %n35.scalar.second)
%n35.scalar.scalar.0 = call float @recipe.from.u1(i1 %n35.scalar.scalar.0.condition)
%n35.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n35.value, i32 %n35.scalar.p
store float %n35.scalar.scalar.0, ptr addrspace(1) %n35.scalar.output.ptr, align 4
br label %n35.scalar.step
n35.scalar.step:
%n35.scalar.next = add i32 %n35.scalar.p, %threads
br label %n35.scalar.loop
n35.scalar.done:
call void @grid_barrier(i32 %threads)%n36.source = getelementptr i8, ptr addrspace(1) %values, i64 2176
%n36.second = getelementptr i8, ptr addrspace(1) %values, i64 2240
%n36.value = getelementptr i8, ptr addrspace(1) %values, i64 2304
%n36.context = getelementptr i8, ptr addrspace(1) %contexts, i64 832
%n36.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n36.scalar.entry
n36.scalar.entry:
%n36.scalar.count = mul i32 %rows, 16
br label %n36.scalar.loop
n36.scalar.loop:
%n36.scalar.p = phi i32 [ %tid, %n36.scalar.entry ], [ %n36.scalar.next, %n36.scalar.step ]
%n36.scalar.more = icmp ult i32 %n36.scalar.p, %n36.scalar.count
br i1 %n36.scalar.more, label %n36.scalar.body, label %n36.scalar.done
n36.scalar.body:
%n36.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n36.source, i32 %n36.scalar.p
%n36.scalar.first = load float, ptr addrspace(1) %n36.scalar.first.ptr, align 4
%n36.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n36.second, i32 %n36.scalar.p
%n36.scalar.second = load float, ptr addrspace(1) %n36.scalar.second.ptr, align 4
%n36.scalar.scalar.0 = call float @recipe.add(float %n36.scalar.first, float %n36.scalar.second)
%n36.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n36.value, i32 %n36.scalar.p
store float %n36.scalar.scalar.0, ptr addrspace(1) %n36.scalar.output.ptr, align 4
br label %n36.scalar.step
n36.scalar.step:
%n36.scalar.next = add i32 %n36.scalar.p, %threads
br label %n36.scalar.loop
n36.scalar.done:
call void @grid_barrier(i32 %threads)%n37.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n37.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n37.value = getelementptr i8, ptr addrspace(1) %values, i64 2368
%n37.context = getelementptr i8, ptr addrspace(1) %contexts, i64 840
%n37.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n37.scalar.entry
n37.scalar.entry:
%n37.scalar.count = mul i32 %rows, 16
br label %n37.scalar.loop
n37.scalar.loop:
%n37.scalar.p = phi i32 [ %tid, %n37.scalar.entry ], [ %n37.scalar.next, %n37.scalar.step ]
%n37.scalar.more = icmp ult i32 %n37.scalar.p, %n37.scalar.count
br i1 %n37.scalar.more, label %n37.scalar.body, label %n37.scalar.done
n37.scalar.body:
%n37.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n37.source, i32 %n37.scalar.p
%n37.scalar.first = load float, ptr addrspace(1) %n37.scalar.first.ptr, align 4
%n37.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n37.second, i32 %n37.scalar.p
%n37.scalar.second = load float, ptr addrspace(1) %n37.scalar.second.ptr, align 4
%n37.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n37.scalar.first, float %n37.scalar.second)
%n37.scalar.scalar.0 = call float @recipe.from.u1(i1 %n37.scalar.scalar.0.condition)
%n37.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n37.value, i32 %n37.scalar.p
store float %n37.scalar.scalar.0, ptr addrspace(1) %n37.scalar.output.ptr, align 4
br label %n37.scalar.step
n37.scalar.step:
%n37.scalar.next = add i32 %n37.scalar.p, %threads
br label %n37.scalar.loop
n37.scalar.done:
call void @grid_barrier(i32 %threads)%n38.source = getelementptr i8, ptr addrspace(1) %values, i64 2304
%n38.second = getelementptr i8, ptr addrspace(1) %values, i64 2368
%n38.value = getelementptr i8, ptr addrspace(1) %values, i64 2432
%n38.context = getelementptr i8, ptr addrspace(1) %contexts, i64 848
%n38.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n38.scalar.entry
n38.scalar.entry:
%n38.scalar.count = mul i32 %rows, 16
br label %n38.scalar.loop
n38.scalar.loop:
%n38.scalar.p = phi i32 [ %tid, %n38.scalar.entry ], [ %n38.scalar.next, %n38.scalar.step ]
%n38.scalar.more = icmp ult i32 %n38.scalar.p, %n38.scalar.count
br i1 %n38.scalar.more, label %n38.scalar.body, label %n38.scalar.done
n38.scalar.body:
%n38.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n38.source, i32 %n38.scalar.p
%n38.scalar.first = load float, ptr addrspace(1) %n38.scalar.first.ptr, align 4
%n38.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n38.second, i32 %n38.scalar.p
%n38.scalar.second = load float, ptr addrspace(1) %n38.scalar.second.ptr, align 4
%n38.scalar.scalar.0 = call float @recipe.add(float %n38.scalar.first, float %n38.scalar.second)
%n38.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n38.value, i32 %n38.scalar.p
store float %n38.scalar.scalar.0, ptr addrspace(1) %n38.scalar.output.ptr, align 4
br label %n38.scalar.step
n38.scalar.step:
%n38.scalar.next = add i32 %n38.scalar.p, %threads
br label %n38.scalar.loop
n38.scalar.done:
call void @grid_barrier(i32 %threads)%n39.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n39.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n39.value = getelementptr i8, ptr addrspace(1) %values, i64 2496
%n39.context = getelementptr i8, ptr addrspace(1) %contexts, i64 856
%n39.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n39.scalar.entry
n39.scalar.entry:
%n39.scalar.count = mul i32 %rows, 16
br label %n39.scalar.loop
n39.scalar.loop:
%n39.scalar.p = phi i32 [ %tid, %n39.scalar.entry ], [ %n39.scalar.next, %n39.scalar.step ]
%n39.scalar.more = icmp ult i32 %n39.scalar.p, %n39.scalar.count
br i1 %n39.scalar.more, label %n39.scalar.body, label %n39.scalar.done
n39.scalar.body:
%n39.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n39.source, i32 %n39.scalar.p
%n39.scalar.first = load float, ptr addrspace(1) %n39.scalar.first.ptr, align 4
%n39.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n39.second, i32 %n39.scalar.p
%n39.scalar.second = load float, ptr addrspace(1) %n39.scalar.second.ptr, align 4
%n39.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n39.scalar.first, float %n39.scalar.second)
%n39.scalar.scalar.0 = call float @recipe.from.u1(i1 %n39.scalar.scalar.0.condition)
%n39.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n39.value, i32 %n39.scalar.p
store float %n39.scalar.scalar.0, ptr addrspace(1) %n39.scalar.output.ptr, align 4
br label %n39.scalar.step
n39.scalar.step:
%n39.scalar.next = add i32 %n39.scalar.p, %threads
br label %n39.scalar.loop
n39.scalar.done:
call void @grid_barrier(i32 %threads)%n40.source = getelementptr i8, ptr addrspace(1) %values, i64 2432
%n40.second = getelementptr i8, ptr addrspace(1) %values, i64 2496
%n40.value = getelementptr i8, ptr addrspace(1) %values, i64 2560
%n40.context = getelementptr i8, ptr addrspace(1) %contexts, i64 864
%n40.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n40.scalar.entry
n40.scalar.entry:
%n40.scalar.count = mul i32 %rows, 16
br label %n40.scalar.loop
n40.scalar.loop:
%n40.scalar.p = phi i32 [ %tid, %n40.scalar.entry ], [ %n40.scalar.next, %n40.scalar.step ]
%n40.scalar.more = icmp ult i32 %n40.scalar.p, %n40.scalar.count
br i1 %n40.scalar.more, label %n40.scalar.body, label %n40.scalar.done
n40.scalar.body:
%n40.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n40.source, i32 %n40.scalar.p
%n40.scalar.first = load float, ptr addrspace(1) %n40.scalar.first.ptr, align 4
%n40.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n40.second, i32 %n40.scalar.p
%n40.scalar.second = load float, ptr addrspace(1) %n40.scalar.second.ptr, align 4
%n40.scalar.scalar.0 = call float @recipe.add(float %n40.scalar.first, float %n40.scalar.second)
%n40.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n40.value, i32 %n40.scalar.p
store float %n40.scalar.scalar.0, ptr addrspace(1) %n40.scalar.output.ptr, align 4
br label %n40.scalar.step
n40.scalar.step:
%n40.scalar.next = add i32 %n40.scalar.p, %threads
br label %n40.scalar.loop
n40.scalar.done:
call void @grid_barrier(i32 %threads)%n41.source = getelementptr i8, ptr addrspace(1) %values, i64 2560
%n41.value = getelementptr i8, ptr addrspace(1) %values, i64 2624
%n41.context = getelementptr i8, ptr addrspace(1) %contexts, i64 872
%n41.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n41.scalar.entry
n41.scalar.entry:
%n41.scalar.count = mul i32 %rows, 16
br label %n41.scalar.loop
n41.scalar.loop:
%n41.scalar.p = phi i32 [ %tid, %n41.scalar.entry ], [ %n41.scalar.next, %n41.scalar.step ]
%n41.scalar.more = icmp ult i32 %n41.scalar.p, %n41.scalar.count
br i1 %n41.scalar.more, label %n41.scalar.body, label %n41.scalar.done
n41.scalar.body:
%n41.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n41.source, i32 %n41.scalar.p
%n41.scalar.first = load float, ptr addrspace(1) %n41.scalar.first.ptr, align 4
%n41.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n41.scalar.first)
%n41.scalar.scalar.1 = call float @recipe.from.u1(i1 %n41.scalar.scalar.1.condition)
%n41.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n41.value, i32 %n41.scalar.p
store float %n41.scalar.scalar.1, ptr addrspace(1) %n41.scalar.output.ptr, align 4
br label %n41.scalar.step
n41.scalar.step:
%n41.scalar.next = add i32 %n41.scalar.p, %threads
br label %n41.scalar.loop
n41.scalar.done:
call void @grid_barrier(i32 %threads)%n42.source = getelementptr i8, ptr addrspace(1) %values, i64 2624
%n42.second = getelementptr i8, ptr addrspace(1) %values, i64 1600
%n42.value = getelementptr i8, ptr addrspace(1) %values, i64 2688
%n42.context = getelementptr i8, ptr addrspace(1) %contexts, i64 880
%n42.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n42.scalar.entry
n42.scalar.entry:
%n42.scalar.count = mul i32 %rows, 16
br label %n42.scalar.loop
n42.scalar.loop:
%n42.scalar.p = phi i32 [ %tid, %n42.scalar.entry ], [ %n42.scalar.next, %n42.scalar.step ]
%n42.scalar.more = icmp ult i32 %n42.scalar.p, %n42.scalar.count
br i1 %n42.scalar.more, label %n42.scalar.body, label %n42.scalar.done
n42.scalar.body:
%n42.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n42.source, i32 %n42.scalar.p
%n42.scalar.first = load float, ptr addrspace(1) %n42.scalar.first.ptr, align 4
%n42.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n42.second, i32 %n42.scalar.p
%n42.scalar.second = load float, ptr addrspace(1) %n42.scalar.second.ptr, align 4
%n42.scalar.scalar.0 = call float @recipe.mul(float %n42.scalar.first, float %n42.scalar.second)
%n42.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n42.value, i32 %n42.scalar.p
store float %n42.scalar.scalar.0, ptr addrspace(1) %n42.scalar.output.ptr, align 4
br label %n42.scalar.step
n42.scalar.step:
%n42.scalar.next = add i32 %n42.scalar.p, %threads
br label %n42.scalar.loop
n42.scalar.done:
call void @grid_barrier(i32 %threads)%n43.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n43.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n43.value = getelementptr i8, ptr addrspace(1) %values, i64 2752
%n43.context = getelementptr i8, ptr addrspace(1) %contexts, i64 888
%n43.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n43.scalar.entry
n43.scalar.entry:
%n43.scalar.count = mul i32 %rows, 16
br label %n43.scalar.loop
n43.scalar.loop:
%n43.scalar.p = phi i32 [ %tid, %n43.scalar.entry ], [ %n43.scalar.next, %n43.scalar.step ]
%n43.scalar.more = icmp ult i32 %n43.scalar.p, %n43.scalar.count
br i1 %n43.scalar.more, label %n43.scalar.body, label %n43.scalar.done
n43.scalar.body:
%n43.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n43.source, i32 %n43.scalar.p
%n43.scalar.first = load float, ptr addrspace(1) %n43.scalar.first.ptr, align 4
%n43.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n43.second, i32 %n43.scalar.p
%n43.scalar.second = load float, ptr addrspace(1) %n43.scalar.second.ptr, align 4
%n43.scalar.scalar.0 = call float @recipe.sub(float %n43.scalar.first, float %n43.scalar.second)
%n43.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n43.value, i32 %n43.scalar.p
store float %n43.scalar.scalar.0, ptr addrspace(1) %n43.scalar.output.ptr, align 4
br label %n43.scalar.step
n43.scalar.step:
%n43.scalar.next = add i32 %n43.scalar.p, %threads
br label %n43.scalar.loop
n43.scalar.done:
call void @grid_barrier(i32 %threads)%n44.source = getelementptr i8, ptr addrspace(1) %values, i64 2752
%n44.value = getelementptr i8, ptr addrspace(1) %values, i64 2816
%n44.context = getelementptr i8, ptr addrspace(1) %contexts, i64 896
%n44.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n44.scalar.entry
n44.scalar.entry:
%n44.scalar.count = mul i32 %rows, 16
br label %n44.scalar.loop
n44.scalar.loop:
%n44.scalar.p = phi i32 [ %tid, %n44.scalar.entry ], [ %n44.scalar.next, %n44.scalar.step ]
%n44.scalar.more = icmp ult i32 %n44.scalar.p, %n44.scalar.count
br i1 %n44.scalar.more, label %n44.scalar.body, label %n44.scalar.done
n44.scalar.body:
%n44.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n44.source, i32 %n44.scalar.p
%n44.scalar.first = load float, ptr addrspace(1) %n44.scalar.first.ptr, align 4
%n44.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n44.scalar.first, float 0x0000000000000000)
%n44.scalar.scalar.2 = call float @recipe.from.u1(i1 %n44.scalar.scalar.2.condition)
%n44.scalar.scalar.3 = call float @recipe.exp(float %n44.scalar.first)
%n44.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n44.value, i32 %n44.scalar.p
store float %n44.scalar.scalar.3, ptr addrspace(1) %n44.scalar.output.ptr, align 4
br label %n44.scalar.step
n44.scalar.step:
%n44.scalar.next = add i32 %n44.scalar.p, %threads
br label %n44.scalar.loop
n44.scalar.done:
call void @grid_barrier(i32 %threads)%n45.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n45.value = getelementptr i8, ptr addrspace(1) %values, i64 2880
%n45.context = getelementptr i8, ptr addrspace(1) %contexts, i64 904
%n45.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n45.scalar.entry
n45.scalar.entry:
%n45.scalar.count = mul i32 %rows, 16
br label %n45.scalar.loop
n45.scalar.loop:
%n45.scalar.p = phi i32 [ %tid, %n45.scalar.entry ], [ %n45.scalar.next, %n45.scalar.step ]
%n45.scalar.more = icmp ult i32 %n45.scalar.p, %n45.scalar.count
br i1 %n45.scalar.more, label %n45.scalar.body, label %n45.scalar.done
n45.scalar.body:
%n45.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n45.source, i32 %n45.scalar.p
%n45.scalar.first = load float, ptr addrspace(1) %n45.scalar.first.ptr, align 4
%n45.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n45.value, i32 %n45.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n45.scalar.output.ptr, align 4
br label %n45.scalar.step
n45.scalar.step:
%n45.scalar.next = add i32 %n45.scalar.p, %threads
br label %n45.scalar.loop
n45.scalar.done:
call void @grid_barrier(i32 %threads)%n46.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n46.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n46.value = getelementptr i8, ptr addrspace(1) %values, i64 2944
%n46.context = getelementptr i8, ptr addrspace(1) %contexts, i64 912
%n46.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n46.scalar.entry
n46.scalar.entry:
%n46.scalar.count = mul i32 %rows, 16
br label %n46.scalar.loop
n46.scalar.loop:
%n46.scalar.p = phi i32 [ %tid, %n46.scalar.entry ], [ %n46.scalar.next, %n46.scalar.step ]
%n46.scalar.more = icmp ult i32 %n46.scalar.p, %n46.scalar.count
br i1 %n46.scalar.more, label %n46.scalar.body, label %n46.scalar.done
n46.scalar.body:
%n46.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n46.source, i32 %n46.scalar.p
%n46.scalar.first = load float, ptr addrspace(1) %n46.scalar.first.ptr, align 4
%n46.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n46.second, i32 %n46.scalar.p
%n46.scalar.second = load float, ptr addrspace(1) %n46.scalar.second.ptr, align 4
%n46.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n46.scalar.first, float %n46.scalar.second)
%n46.scalar.scalar.0 = call float @recipe.from.u1(i1 %n46.scalar.scalar.0.condition)
%n46.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n46.value, i32 %n46.scalar.p
store float %n46.scalar.scalar.0, ptr addrspace(1) %n46.scalar.output.ptr, align 4
br label %n46.scalar.step
n46.scalar.step:
%n46.scalar.next = add i32 %n46.scalar.p, %threads
br label %n46.scalar.loop
n46.scalar.done:
call void @grid_barrier(i32 %threads)%n47.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n47.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n47.value = getelementptr i8, ptr addrspace(1) %values, i64 3008
%n47.context = getelementptr i8, ptr addrspace(1) %contexts, i64 920
%n47.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n47.scalar.entry
n47.scalar.entry:
%n47.scalar.count = mul i32 %rows, 16
br label %n47.scalar.loop
n47.scalar.loop:
%n47.scalar.p = phi i32 [ %tid, %n47.scalar.entry ], [ %n47.scalar.next, %n47.scalar.step ]
%n47.scalar.more = icmp ult i32 %n47.scalar.p, %n47.scalar.count
br i1 %n47.scalar.more, label %n47.scalar.body, label %n47.scalar.done
n47.scalar.body:
%n47.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n47.source, i32 %n47.scalar.p
%n47.scalar.first = load float, ptr addrspace(1) %n47.scalar.first.ptr, align 4
%n47.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n47.second, i32 %n47.scalar.p
%n47.scalar.second = load float, ptr addrspace(1) %n47.scalar.second.ptr, align 4
%n47.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n47.scalar.first, float %n47.scalar.second)
%n47.scalar.scalar.0 = call float @recipe.from.u1(i1 %n47.scalar.scalar.0.condition)
%n47.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n47.value, i32 %n47.scalar.p
store float %n47.scalar.scalar.0, ptr addrspace(1) %n47.scalar.output.ptr, align 4
br label %n47.scalar.step
n47.scalar.step:
%n47.scalar.next = add i32 %n47.scalar.p, %threads
br label %n47.scalar.loop
n47.scalar.done:
call void @grid_barrier(i32 %threads)%n48.source = getelementptr i8, ptr addrspace(1) %values, i64 2944
%n48.second = getelementptr i8, ptr addrspace(1) %values, i64 3008
%n48.value = getelementptr i8, ptr addrspace(1) %values, i64 3072
%n48.context = getelementptr i8, ptr addrspace(1) %contexts, i64 928
%n48.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n48.scalar.entry
n48.scalar.entry:
%n48.scalar.count = mul i32 %rows, 16
br label %n48.scalar.loop
n48.scalar.loop:
%n48.scalar.p = phi i32 [ %tid, %n48.scalar.entry ], [ %n48.scalar.next, %n48.scalar.step ]
%n48.scalar.more = icmp ult i32 %n48.scalar.p, %n48.scalar.count
br i1 %n48.scalar.more, label %n48.scalar.body, label %n48.scalar.done
n48.scalar.body:
%n48.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n48.source, i32 %n48.scalar.p
%n48.scalar.first = load float, ptr addrspace(1) %n48.scalar.first.ptr, align 4
%n48.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n48.second, i32 %n48.scalar.p
%n48.scalar.second = load float, ptr addrspace(1) %n48.scalar.second.ptr, align 4
%n48.scalar.scalar.0 = call float @recipe.add(float %n48.scalar.first, float %n48.scalar.second)
%n48.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n48.value, i32 %n48.scalar.p
store float %n48.scalar.scalar.0, ptr addrspace(1) %n48.scalar.output.ptr, align 4
br label %n48.scalar.step
n48.scalar.step:
%n48.scalar.next = add i32 %n48.scalar.p, %threads
br label %n48.scalar.loop
n48.scalar.done:
call void @grid_barrier(i32 %threads)%n49.source = getelementptr i8, ptr addrspace(1) %values, i64 3072
%n49.value = getelementptr i8, ptr addrspace(1) %values, i64 3136
%n49.context = getelementptr i8, ptr addrspace(1) %contexts, i64 936
%n49.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n49.scalar.entry
n49.scalar.entry:
%n49.scalar.count = mul i32 %rows, 16
br label %n49.scalar.loop
n49.scalar.loop:
%n49.scalar.p = phi i32 [ %tid, %n49.scalar.entry ], [ %n49.scalar.next, %n49.scalar.step ]
%n49.scalar.more = icmp ult i32 %n49.scalar.p, %n49.scalar.count
br i1 %n49.scalar.more, label %n49.scalar.body, label %n49.scalar.done
n49.scalar.body:
%n49.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n49.source, i32 %n49.scalar.p
%n49.scalar.first = load float, ptr addrspace(1) %n49.scalar.first.ptr, align 4
%n49.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n49.scalar.first)
%n49.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n49.value, i32 %n49.scalar.p
store float %n49.scalar.scalar.1, ptr addrspace(1) %n49.scalar.output.ptr, align 4
br label %n49.scalar.step
n49.scalar.step:
%n49.scalar.next = add i32 %n49.scalar.p, %threads
br label %n49.scalar.loop
n49.scalar.done:
call void @grid_barrier(i32 %threads)%n50.source = getelementptr i8, ptr addrspace(1) %values, i64 2944
%n50.second = getelementptr i8, ptr addrspace(1) %values, i64 3136
%n50.value = getelementptr i8, ptr addrspace(1) %values, i64 3200
%n50.context = getelementptr i8, ptr addrspace(1) %contexts, i64 944
%n50.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n50.scalar.entry
n50.scalar.entry:
%n50.scalar.count = mul i32 %rows, 16
br label %n50.scalar.loop
n50.scalar.loop:
%n50.scalar.p = phi i32 [ %tid, %n50.scalar.entry ], [ %n50.scalar.next, %n50.scalar.step ]
%n50.scalar.more = icmp ult i32 %n50.scalar.p, %n50.scalar.count
br i1 %n50.scalar.more, label %n50.scalar.body, label %n50.scalar.done
n50.scalar.body:
%n50.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n50.source, i32 %n50.scalar.p
%n50.scalar.first = load float, ptr addrspace(1) %n50.scalar.first.ptr, align 4
%n50.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n50.second, i32 %n50.scalar.p
%n50.scalar.second = load float, ptr addrspace(1) %n50.scalar.second.ptr, align 4
%n50.scalar.scalar.0 = call float @recipe.add(float %n50.scalar.first, float %n50.scalar.second)
%n50.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n50.value, i32 %n50.scalar.p
store float %n50.scalar.scalar.0, ptr addrspace(1) %n50.scalar.output.ptr, align 4
br label %n50.scalar.step
n50.scalar.step:
%n50.scalar.next = add i32 %n50.scalar.p, %threads
br label %n50.scalar.loop
n50.scalar.done:
call void @grid_barrier(i32 %threads)%n51.source = getelementptr i8, ptr addrspace(1) %values, i64 2880
%n51.second = getelementptr i8, ptr addrspace(1) %values, i64 3200
%n51.value = getelementptr i8, ptr addrspace(1) %values, i64 3264
%n51.context = getelementptr i8, ptr addrspace(1) %contexts, i64 952
%n51.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n51.scalar.entry
n51.scalar.entry:
%n51.scalar.count = mul i32 %rows, 16
br label %n51.scalar.loop
n51.scalar.loop:
%n51.scalar.p = phi i32 [ %tid, %n51.scalar.entry ], [ %n51.scalar.next, %n51.scalar.step ]
%n51.scalar.more = icmp ult i32 %n51.scalar.p, %n51.scalar.count
br i1 %n51.scalar.more, label %n51.scalar.body, label %n51.scalar.done
n51.scalar.body:
%n51.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n51.source, i32 %n51.scalar.p
%n51.scalar.first = load float, ptr addrspace(1) %n51.scalar.first.ptr, align 4
%n51.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n51.second, i32 %n51.scalar.p
%n51.scalar.second = load float, ptr addrspace(1) %n51.scalar.second.ptr, align 4
%n51.scalar.scalar.0 = call float @recipe.add(float %n51.scalar.first, float %n51.scalar.second)
%n51.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n51.value, i32 %n51.scalar.p
store float %n51.scalar.scalar.0, ptr addrspace(1) %n51.scalar.output.ptr, align 4
br label %n51.scalar.step
n51.scalar.step:
%n51.scalar.next = add i32 %n51.scalar.p, %threads
br label %n51.scalar.loop
n51.scalar.done:
call void @grid_barrier(i32 %threads)%n52.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n52.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n52.value = getelementptr i8, ptr addrspace(1) %values, i64 3328
%n52.context = getelementptr i8, ptr addrspace(1) %contexts, i64 960
%n52.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n52.scalar.entry
n52.scalar.entry:
%n52.scalar.count = mul i32 %rows, 16
br label %n52.scalar.loop
n52.scalar.loop:
%n52.scalar.p = phi i32 [ %tid, %n52.scalar.entry ], [ %n52.scalar.next, %n52.scalar.step ]
%n52.scalar.more = icmp ult i32 %n52.scalar.p, %n52.scalar.count
br i1 %n52.scalar.more, label %n52.scalar.body, label %n52.scalar.done
n52.scalar.body:
%n52.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n52.source, i32 %n52.scalar.p
%n52.scalar.first = load float, ptr addrspace(1) %n52.scalar.first.ptr, align 4
%n52.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n52.second, i32 %n52.scalar.p
%n52.scalar.second = load float, ptr addrspace(1) %n52.scalar.second.ptr, align 4
%n52.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n52.scalar.first, float %n52.scalar.second)
%n52.scalar.scalar.0 = call float @recipe.from.u1(i1 %n52.scalar.scalar.0.condition)
%n52.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n52.value, i32 %n52.scalar.p
store float %n52.scalar.scalar.0, ptr addrspace(1) %n52.scalar.output.ptr, align 4
br label %n52.scalar.step
n52.scalar.step:
%n52.scalar.next = add i32 %n52.scalar.p, %threads
br label %n52.scalar.loop
n52.scalar.done:
call void @grid_barrier(i32 %threads)%n53.source = getelementptr i8, ptr addrspace(1) %values, i64 3264
%n53.second = getelementptr i8, ptr addrspace(1) %values, i64 3328
%n53.value = getelementptr i8, ptr addrspace(1) %values, i64 3392
%n53.context = getelementptr i8, ptr addrspace(1) %contexts, i64 968
%n53.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n53.scalar.entry
n53.scalar.entry:
%n53.scalar.count = mul i32 %rows, 16
br label %n53.scalar.loop
n53.scalar.loop:
%n53.scalar.p = phi i32 [ %tid, %n53.scalar.entry ], [ %n53.scalar.next, %n53.scalar.step ]
%n53.scalar.more = icmp ult i32 %n53.scalar.p, %n53.scalar.count
br i1 %n53.scalar.more, label %n53.scalar.body, label %n53.scalar.done
n53.scalar.body:
%n53.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n53.source, i32 %n53.scalar.p
%n53.scalar.first = load float, ptr addrspace(1) %n53.scalar.first.ptr, align 4
%n53.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n53.second, i32 %n53.scalar.p
%n53.scalar.second = load float, ptr addrspace(1) %n53.scalar.second.ptr, align 4
%n53.scalar.scalar.0 = call float @recipe.add(float %n53.scalar.first, float %n53.scalar.second)
%n53.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n53.value, i32 %n53.scalar.p
store float %n53.scalar.scalar.0, ptr addrspace(1) %n53.scalar.output.ptr, align 4
br label %n53.scalar.step
n53.scalar.step:
%n53.scalar.next = add i32 %n53.scalar.p, %threads
br label %n53.scalar.loop
n53.scalar.done:
call void @grid_barrier(i32 %threads)%n54.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n54.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n54.value = getelementptr i8, ptr addrspace(1) %values, i64 3456
%n54.context = getelementptr i8, ptr addrspace(1) %contexts, i64 976
%n54.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n54.scalar.entry
n54.scalar.entry:
%n54.scalar.count = mul i32 %rows, 16
br label %n54.scalar.loop
n54.scalar.loop:
%n54.scalar.p = phi i32 [ %tid, %n54.scalar.entry ], [ %n54.scalar.next, %n54.scalar.step ]
%n54.scalar.more = icmp ult i32 %n54.scalar.p, %n54.scalar.count
br i1 %n54.scalar.more, label %n54.scalar.body, label %n54.scalar.done
n54.scalar.body:
%n54.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n54.source, i32 %n54.scalar.p
%n54.scalar.first = load float, ptr addrspace(1) %n54.scalar.first.ptr, align 4
%n54.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n54.second, i32 %n54.scalar.p
%n54.scalar.second = load float, ptr addrspace(1) %n54.scalar.second.ptr, align 4
%n54.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n54.scalar.first, float %n54.scalar.second)
%n54.scalar.scalar.0 = call float @recipe.from.u1(i1 %n54.scalar.scalar.0.condition)
%n54.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n54.value, i32 %n54.scalar.p
store float %n54.scalar.scalar.0, ptr addrspace(1) %n54.scalar.output.ptr, align 4
br label %n54.scalar.step
n54.scalar.step:
%n54.scalar.next = add i32 %n54.scalar.p, %threads
br label %n54.scalar.loop
n54.scalar.done:
call void @grid_barrier(i32 %threads)%n55.source = getelementptr i8, ptr addrspace(1) %values, i64 3392
%n55.second = getelementptr i8, ptr addrspace(1) %values, i64 3456
%n55.value = getelementptr i8, ptr addrspace(1) %values, i64 3520
%n55.context = getelementptr i8, ptr addrspace(1) %contexts, i64 984
%n55.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n55.scalar.entry
n55.scalar.entry:
%n55.scalar.count = mul i32 %rows, 16
br label %n55.scalar.loop
n55.scalar.loop:
%n55.scalar.p = phi i32 [ %tid, %n55.scalar.entry ], [ %n55.scalar.next, %n55.scalar.step ]
%n55.scalar.more = icmp ult i32 %n55.scalar.p, %n55.scalar.count
br i1 %n55.scalar.more, label %n55.scalar.body, label %n55.scalar.done
n55.scalar.body:
%n55.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n55.source, i32 %n55.scalar.p
%n55.scalar.first = load float, ptr addrspace(1) %n55.scalar.first.ptr, align 4
%n55.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n55.second, i32 %n55.scalar.p
%n55.scalar.second = load float, ptr addrspace(1) %n55.scalar.second.ptr, align 4
%n55.scalar.scalar.0 = call float @recipe.add(float %n55.scalar.first, float %n55.scalar.second)
%n55.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n55.value, i32 %n55.scalar.p
store float %n55.scalar.scalar.0, ptr addrspace(1) %n55.scalar.output.ptr, align 4
br label %n55.scalar.step
n55.scalar.step:
%n55.scalar.next = add i32 %n55.scalar.p, %threads
br label %n55.scalar.loop
n55.scalar.done:
call void @grid_barrier(i32 %threads)%n56.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n56.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n56.value = getelementptr i8, ptr addrspace(1) %values, i64 3584
%n56.context = getelementptr i8, ptr addrspace(1) %contexts, i64 992
%n56.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n56.scalar.entry
n56.scalar.entry:
%n56.scalar.count = mul i32 %rows, 16
br label %n56.scalar.loop
n56.scalar.loop:
%n56.scalar.p = phi i32 [ %tid, %n56.scalar.entry ], [ %n56.scalar.next, %n56.scalar.step ]
%n56.scalar.more = icmp ult i32 %n56.scalar.p, %n56.scalar.count
br i1 %n56.scalar.more, label %n56.scalar.body, label %n56.scalar.done
n56.scalar.body:
%n56.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n56.source, i32 %n56.scalar.p
%n56.scalar.first = load float, ptr addrspace(1) %n56.scalar.first.ptr, align 4
%n56.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n56.second, i32 %n56.scalar.p
%n56.scalar.second = load float, ptr addrspace(1) %n56.scalar.second.ptr, align 4
%n56.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n56.scalar.first, float %n56.scalar.second)
%n56.scalar.scalar.0 = call float @recipe.from.u1(i1 %n56.scalar.scalar.0.condition)
%n56.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n56.value, i32 %n56.scalar.p
store float %n56.scalar.scalar.0, ptr addrspace(1) %n56.scalar.output.ptr, align 4
br label %n56.scalar.step
n56.scalar.step:
%n56.scalar.next = add i32 %n56.scalar.p, %threads
br label %n56.scalar.loop
n56.scalar.done:
call void @grid_barrier(i32 %threads)%n57.source = getelementptr i8, ptr addrspace(1) %values, i64 3520
%n57.second = getelementptr i8, ptr addrspace(1) %values, i64 3584
%n57.value = getelementptr i8, ptr addrspace(1) %values, i64 3648
%n57.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1000
%n57.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n57.scalar.entry
n57.scalar.entry:
%n57.scalar.count = mul i32 %rows, 16
br label %n57.scalar.loop
n57.scalar.loop:
%n57.scalar.p = phi i32 [ %tid, %n57.scalar.entry ], [ %n57.scalar.next, %n57.scalar.step ]
%n57.scalar.more = icmp ult i32 %n57.scalar.p, %n57.scalar.count
br i1 %n57.scalar.more, label %n57.scalar.body, label %n57.scalar.done
n57.scalar.body:
%n57.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n57.source, i32 %n57.scalar.p
%n57.scalar.first = load float, ptr addrspace(1) %n57.scalar.first.ptr, align 4
%n57.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n57.second, i32 %n57.scalar.p
%n57.scalar.second = load float, ptr addrspace(1) %n57.scalar.second.ptr, align 4
%n57.scalar.scalar.0 = call float @recipe.add(float %n57.scalar.first, float %n57.scalar.second)
%n57.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n57.value, i32 %n57.scalar.p
store float %n57.scalar.scalar.0, ptr addrspace(1) %n57.scalar.output.ptr, align 4
br label %n57.scalar.step
n57.scalar.step:
%n57.scalar.next = add i32 %n57.scalar.p, %threads
br label %n57.scalar.loop
n57.scalar.done:
call void @grid_barrier(i32 %threads)%n58.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n58.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n58.value = getelementptr i8, ptr addrspace(1) %values, i64 3712
%n58.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1008
%n58.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n58.scalar.entry
n58.scalar.entry:
%n58.scalar.count = mul i32 %rows, 16
br label %n58.scalar.loop
n58.scalar.loop:
%n58.scalar.p = phi i32 [ %tid, %n58.scalar.entry ], [ %n58.scalar.next, %n58.scalar.step ]
%n58.scalar.more = icmp ult i32 %n58.scalar.p, %n58.scalar.count
br i1 %n58.scalar.more, label %n58.scalar.body, label %n58.scalar.done
n58.scalar.body:
%n58.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n58.source, i32 %n58.scalar.p
%n58.scalar.first = load float, ptr addrspace(1) %n58.scalar.first.ptr, align 4
%n58.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n58.second, i32 %n58.scalar.p
%n58.scalar.second = load float, ptr addrspace(1) %n58.scalar.second.ptr, align 4
%n58.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n58.scalar.first, float %n58.scalar.second)
%n58.scalar.scalar.0 = call float @recipe.from.u1(i1 %n58.scalar.scalar.0.condition)
%n58.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n58.value, i32 %n58.scalar.p
store float %n58.scalar.scalar.0, ptr addrspace(1) %n58.scalar.output.ptr, align 4
br label %n58.scalar.step
n58.scalar.step:
%n58.scalar.next = add i32 %n58.scalar.p, %threads
br label %n58.scalar.loop
n58.scalar.done:
call void @grid_barrier(i32 %threads)%n59.source = getelementptr i8, ptr addrspace(1) %values, i64 3648
%n59.second = getelementptr i8, ptr addrspace(1) %values, i64 3712
%n59.value = getelementptr i8, ptr addrspace(1) %values, i64 3776
%n59.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1016
%n59.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n59.scalar.entry
n59.scalar.entry:
%n59.scalar.count = mul i32 %rows, 16
br label %n59.scalar.loop
n59.scalar.loop:
%n59.scalar.p = phi i32 [ %tid, %n59.scalar.entry ], [ %n59.scalar.next, %n59.scalar.step ]
%n59.scalar.more = icmp ult i32 %n59.scalar.p, %n59.scalar.count
br i1 %n59.scalar.more, label %n59.scalar.body, label %n59.scalar.done
n59.scalar.body:
%n59.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n59.source, i32 %n59.scalar.p
%n59.scalar.first = load float, ptr addrspace(1) %n59.scalar.first.ptr, align 4
%n59.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n59.second, i32 %n59.scalar.p
%n59.scalar.second = load float, ptr addrspace(1) %n59.scalar.second.ptr, align 4
%n59.scalar.scalar.0 = call float @recipe.add(float %n59.scalar.first, float %n59.scalar.second)
%n59.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n59.value, i32 %n59.scalar.p
store float %n59.scalar.scalar.0, ptr addrspace(1) %n59.scalar.output.ptr, align 4
br label %n59.scalar.step
n59.scalar.step:
%n59.scalar.next = add i32 %n59.scalar.p, %threads
br label %n59.scalar.loop
n59.scalar.done:
call void @grid_barrier(i32 %threads)%n60.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n60.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n60.value = getelementptr i8, ptr addrspace(1) %values, i64 3840
%n60.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1024
%n60.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n60.scalar.entry
n60.scalar.entry:
%n60.scalar.count = mul i32 %rows, 16
br label %n60.scalar.loop
n60.scalar.loop:
%n60.scalar.p = phi i32 [ %tid, %n60.scalar.entry ], [ %n60.scalar.next, %n60.scalar.step ]
%n60.scalar.more = icmp ult i32 %n60.scalar.p, %n60.scalar.count
br i1 %n60.scalar.more, label %n60.scalar.body, label %n60.scalar.done
n60.scalar.body:
%n60.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n60.source, i32 %n60.scalar.p
%n60.scalar.first = load float, ptr addrspace(1) %n60.scalar.first.ptr, align 4
%n60.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n60.second, i32 %n60.scalar.p
%n60.scalar.second = load float, ptr addrspace(1) %n60.scalar.second.ptr, align 4
%n60.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n60.scalar.first, float %n60.scalar.second)
%n60.scalar.scalar.0 = call float @recipe.from.u1(i1 %n60.scalar.scalar.0.condition)
%n60.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n60.value, i32 %n60.scalar.p
store float %n60.scalar.scalar.0, ptr addrspace(1) %n60.scalar.output.ptr, align 4
br label %n60.scalar.step
n60.scalar.step:
%n60.scalar.next = add i32 %n60.scalar.p, %threads
br label %n60.scalar.loop
n60.scalar.done:
call void @grid_barrier(i32 %threads)%n61.source = getelementptr i8, ptr addrspace(1) %values, i64 3776
%n61.second = getelementptr i8, ptr addrspace(1) %values, i64 3840
%n61.value = getelementptr i8, ptr addrspace(1) %values, i64 3904
%n61.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1032
%n61.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n61.scalar.entry
n61.scalar.entry:
%n61.scalar.count = mul i32 %rows, 16
br label %n61.scalar.loop
n61.scalar.loop:
%n61.scalar.p = phi i32 [ %tid, %n61.scalar.entry ], [ %n61.scalar.next, %n61.scalar.step ]
%n61.scalar.more = icmp ult i32 %n61.scalar.p, %n61.scalar.count
br i1 %n61.scalar.more, label %n61.scalar.body, label %n61.scalar.done
n61.scalar.body:
%n61.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n61.source, i32 %n61.scalar.p
%n61.scalar.first = load float, ptr addrspace(1) %n61.scalar.first.ptr, align 4
%n61.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n61.second, i32 %n61.scalar.p
%n61.scalar.second = load float, ptr addrspace(1) %n61.scalar.second.ptr, align 4
%n61.scalar.scalar.0 = call float @recipe.add(float %n61.scalar.first, float %n61.scalar.second)
%n61.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n61.value, i32 %n61.scalar.p
store float %n61.scalar.scalar.0, ptr addrspace(1) %n61.scalar.output.ptr, align 4
br label %n61.scalar.step
n61.scalar.step:
%n61.scalar.next = add i32 %n61.scalar.p, %threads
br label %n61.scalar.loop
n61.scalar.done:
call void @grid_barrier(i32 %threads)%n62.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n62.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n62.value = getelementptr i8, ptr addrspace(1) %values, i64 3968
%n62.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1040
%n62.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n62.scalar.entry
n62.scalar.entry:
%n62.scalar.count = mul i32 %rows, 16
br label %n62.scalar.loop
n62.scalar.loop:
%n62.scalar.p = phi i32 [ %tid, %n62.scalar.entry ], [ %n62.scalar.next, %n62.scalar.step ]
%n62.scalar.more = icmp ult i32 %n62.scalar.p, %n62.scalar.count
br i1 %n62.scalar.more, label %n62.scalar.body, label %n62.scalar.done
n62.scalar.body:
%n62.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n62.source, i32 %n62.scalar.p
%n62.scalar.first = load float, ptr addrspace(1) %n62.scalar.first.ptr, align 4
%n62.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n62.second, i32 %n62.scalar.p
%n62.scalar.second = load float, ptr addrspace(1) %n62.scalar.second.ptr, align 4
%n62.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n62.scalar.first, float %n62.scalar.second)
%n62.scalar.scalar.0 = call float @recipe.from.u1(i1 %n62.scalar.scalar.0.condition)
%n62.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n62.value, i32 %n62.scalar.p
store float %n62.scalar.scalar.0, ptr addrspace(1) %n62.scalar.output.ptr, align 4
br label %n62.scalar.step
n62.scalar.step:
%n62.scalar.next = add i32 %n62.scalar.p, %threads
br label %n62.scalar.loop
n62.scalar.done:
call void @grid_barrier(i32 %threads)%n63.source = getelementptr i8, ptr addrspace(1) %values, i64 3904
%n63.second = getelementptr i8, ptr addrspace(1) %values, i64 3968
%n63.value = getelementptr i8, ptr addrspace(1) %values, i64 4032
%n63.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1048
%n63.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n63.scalar.entry
n63.scalar.entry:
%n63.scalar.count = mul i32 %rows, 16
br label %n63.scalar.loop
n63.scalar.loop:
%n63.scalar.p = phi i32 [ %tid, %n63.scalar.entry ], [ %n63.scalar.next, %n63.scalar.step ]
%n63.scalar.more = icmp ult i32 %n63.scalar.p, %n63.scalar.count
br i1 %n63.scalar.more, label %n63.scalar.body, label %n63.scalar.done
n63.scalar.body:
%n63.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n63.source, i32 %n63.scalar.p
%n63.scalar.first = load float, ptr addrspace(1) %n63.scalar.first.ptr, align 4
%n63.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n63.second, i32 %n63.scalar.p
%n63.scalar.second = load float, ptr addrspace(1) %n63.scalar.second.ptr, align 4
%n63.scalar.scalar.0 = call float @recipe.add(float %n63.scalar.first, float %n63.scalar.second)
%n63.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n63.value, i32 %n63.scalar.p
store float %n63.scalar.scalar.0, ptr addrspace(1) %n63.scalar.output.ptr, align 4
br label %n63.scalar.step
n63.scalar.step:
%n63.scalar.next = add i32 %n63.scalar.p, %threads
br label %n63.scalar.loop
n63.scalar.done:
call void @grid_barrier(i32 %threads)%n64.source = getelementptr i8, ptr addrspace(1) %values, i64 4032
%n64.value = getelementptr i8, ptr addrspace(1) %values, i64 4096
%n64.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1056
%n64.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n64.scalar.entry
n64.scalar.entry:
%n64.scalar.count = mul i32 %rows, 16
br label %n64.scalar.loop
n64.scalar.loop:
%n64.scalar.p = phi i32 [ %tid, %n64.scalar.entry ], [ %n64.scalar.next, %n64.scalar.step ]
%n64.scalar.more = icmp ult i32 %n64.scalar.p, %n64.scalar.count
br i1 %n64.scalar.more, label %n64.scalar.body, label %n64.scalar.done
n64.scalar.body:
%n64.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n64.source, i32 %n64.scalar.p
%n64.scalar.first = load float, ptr addrspace(1) %n64.scalar.first.ptr, align 4
%n64.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n64.scalar.first)
%n64.scalar.scalar.1 = call float @recipe.from.u1(i1 %n64.scalar.scalar.1.condition)
%n64.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n64.value, i32 %n64.scalar.p
store float %n64.scalar.scalar.1, ptr addrspace(1) %n64.scalar.output.ptr, align 4
br label %n64.scalar.step
n64.scalar.step:
%n64.scalar.next = add i32 %n64.scalar.p, %threads
br label %n64.scalar.loop
n64.scalar.done:
call void @grid_barrier(i32 %threads)%n65.source = getelementptr i8, ptr addrspace(1) %values, i64 4096
%n65.second = getelementptr i8, ptr addrspace(1) %values, i64 2816
%n65.value = getelementptr i8, ptr addrspace(1) %values, i64 4160
%n65.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1064
%n65.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n65.scalar.entry
n65.scalar.entry:
%n65.scalar.count = mul i32 %rows, 16
br label %n65.scalar.loop
n65.scalar.loop:
%n65.scalar.p = phi i32 [ %tid, %n65.scalar.entry ], [ %n65.scalar.next, %n65.scalar.step ]
%n65.scalar.more = icmp ult i32 %n65.scalar.p, %n65.scalar.count
br i1 %n65.scalar.more, label %n65.scalar.body, label %n65.scalar.done
n65.scalar.body:
%n65.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n65.source, i32 %n65.scalar.p
%n65.scalar.first = load float, ptr addrspace(1) %n65.scalar.first.ptr, align 4
%n65.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n65.second, i32 %n65.scalar.p
%n65.scalar.second = load float, ptr addrspace(1) %n65.scalar.second.ptr, align 4
%n65.scalar.scalar.0 = call float @recipe.mul(float %n65.scalar.first, float %n65.scalar.second)
%n65.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n65.value, i32 %n65.scalar.p
store float %n65.scalar.scalar.0, ptr addrspace(1) %n65.scalar.output.ptr, align 4
br label %n65.scalar.step
n65.scalar.step:
%n65.scalar.next = add i32 %n65.scalar.p, %threads
br label %n65.scalar.loop
n65.scalar.done:
call void @grid_barrier(i32 %threads)%n66.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n66.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n66.value = getelementptr i8, ptr addrspace(1) %values, i64 4224
%n66.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1072
%n66.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n66.scalar.entry
n66.scalar.entry:
%n66.scalar.count = mul i32 %rows, 16
br label %n66.scalar.loop
n66.scalar.loop:
%n66.scalar.p = phi i32 [ %tid, %n66.scalar.entry ], [ %n66.scalar.next, %n66.scalar.step ]
%n66.scalar.more = icmp ult i32 %n66.scalar.p, %n66.scalar.count
br i1 %n66.scalar.more, label %n66.scalar.body, label %n66.scalar.done
n66.scalar.body:
%n66.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n66.source, i32 %n66.scalar.p
%n66.scalar.first = load float, ptr addrspace(1) %n66.scalar.first.ptr, align 4
%n66.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n66.second, i32 %n66.scalar.p
%n66.scalar.second = load float, ptr addrspace(1) %n66.scalar.second.ptr, align 4
%n66.scalar.scalar.0 = call float @recipe.sub(float %n66.scalar.first, float %n66.scalar.second)
%n66.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n66.value, i32 %n66.scalar.p
store float %n66.scalar.scalar.0, ptr addrspace(1) %n66.scalar.output.ptr, align 4
br label %n66.scalar.step
n66.scalar.step:
%n66.scalar.next = add i32 %n66.scalar.p, %threads
br label %n66.scalar.loop
n66.scalar.done:
call void @grid_barrier(i32 %threads)%n67.source = getelementptr i8, ptr addrspace(1) %values, i64 4224
%n67.value = getelementptr i8, ptr addrspace(1) %values, i64 4288
%n67.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1080
%n67.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n67.scalar.entry
n67.scalar.entry:
%n67.scalar.count = mul i32 %rows, 16
br label %n67.scalar.loop
n67.scalar.loop:
%n67.scalar.p = phi i32 [ %tid, %n67.scalar.entry ], [ %n67.scalar.next, %n67.scalar.step ]
%n67.scalar.more = icmp ult i32 %n67.scalar.p, %n67.scalar.count
br i1 %n67.scalar.more, label %n67.scalar.body, label %n67.scalar.done
n67.scalar.body:
%n67.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n67.source, i32 %n67.scalar.p
%n67.scalar.first = load float, ptr addrspace(1) %n67.scalar.first.ptr, align 4
%n67.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n67.scalar.first, float 0x0000000000000000)
%n67.scalar.scalar.2 = call float @recipe.from.u1(i1 %n67.scalar.scalar.2.condition)
%n67.scalar.scalar.3 = call float @recipe.exp(float %n67.scalar.first)
%n67.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n67.value, i32 %n67.scalar.p
store float %n67.scalar.scalar.3, ptr addrspace(1) %n67.scalar.output.ptr, align 4
br label %n67.scalar.step
n67.scalar.step:
%n67.scalar.next = add i32 %n67.scalar.p, %threads
br label %n67.scalar.loop
n67.scalar.done:
call void @grid_barrier(i32 %threads)%n68.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n68.value = getelementptr i8, ptr addrspace(1) %values, i64 4352
%n68.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1088
%n68.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n68.scalar.entry
n68.scalar.entry:
%n68.scalar.count = mul i32 %rows, 16
br label %n68.scalar.loop
n68.scalar.loop:
%n68.scalar.p = phi i32 [ %tid, %n68.scalar.entry ], [ %n68.scalar.next, %n68.scalar.step ]
%n68.scalar.more = icmp ult i32 %n68.scalar.p, %n68.scalar.count
br i1 %n68.scalar.more, label %n68.scalar.body, label %n68.scalar.done
n68.scalar.body:
%n68.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n68.source, i32 %n68.scalar.p
%n68.scalar.first = load float, ptr addrspace(1) %n68.scalar.first.ptr, align 4
%n68.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n68.value, i32 %n68.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n68.scalar.output.ptr, align 4
br label %n68.scalar.step
n68.scalar.step:
%n68.scalar.next = add i32 %n68.scalar.p, %threads
br label %n68.scalar.loop
n68.scalar.done:
call void @grid_barrier(i32 %threads)%n69.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n69.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n69.value = getelementptr i8, ptr addrspace(1) %values, i64 4416
%n69.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1096
%n69.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n69.scalar.entry
n69.scalar.entry:
%n69.scalar.count = mul i32 %rows, 16
br label %n69.scalar.loop
n69.scalar.loop:
%n69.scalar.p = phi i32 [ %tid, %n69.scalar.entry ], [ %n69.scalar.next, %n69.scalar.step ]
%n69.scalar.more = icmp ult i32 %n69.scalar.p, %n69.scalar.count
br i1 %n69.scalar.more, label %n69.scalar.body, label %n69.scalar.done
n69.scalar.body:
%n69.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n69.source, i32 %n69.scalar.p
%n69.scalar.first = load float, ptr addrspace(1) %n69.scalar.first.ptr, align 4
%n69.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n69.second, i32 %n69.scalar.p
%n69.scalar.second = load float, ptr addrspace(1) %n69.scalar.second.ptr, align 4
%n69.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n69.scalar.first, float %n69.scalar.second)
%n69.scalar.scalar.0 = call float @recipe.from.u1(i1 %n69.scalar.scalar.0.condition)
%n69.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n69.value, i32 %n69.scalar.p
store float %n69.scalar.scalar.0, ptr addrspace(1) %n69.scalar.output.ptr, align 4
br label %n69.scalar.step
n69.scalar.step:
%n69.scalar.next = add i32 %n69.scalar.p, %threads
br label %n69.scalar.loop
n69.scalar.done:
call void @grid_barrier(i32 %threads)%n70.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n70.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n70.value = getelementptr i8, ptr addrspace(1) %values, i64 4480
%n70.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1104
%n70.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n70.scalar.entry
n70.scalar.entry:
%n70.scalar.count = mul i32 %rows, 16
br label %n70.scalar.loop
n70.scalar.loop:
%n70.scalar.p = phi i32 [ %tid, %n70.scalar.entry ], [ %n70.scalar.next, %n70.scalar.step ]
%n70.scalar.more = icmp ult i32 %n70.scalar.p, %n70.scalar.count
br i1 %n70.scalar.more, label %n70.scalar.body, label %n70.scalar.done
n70.scalar.body:
%n70.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n70.source, i32 %n70.scalar.p
%n70.scalar.first = load float, ptr addrspace(1) %n70.scalar.first.ptr, align 4
%n70.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n70.second, i32 %n70.scalar.p
%n70.scalar.second = load float, ptr addrspace(1) %n70.scalar.second.ptr, align 4
%n70.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n70.scalar.first, float %n70.scalar.second)
%n70.scalar.scalar.0 = call float @recipe.from.u1(i1 %n70.scalar.scalar.0.condition)
%n70.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n70.value, i32 %n70.scalar.p
store float %n70.scalar.scalar.0, ptr addrspace(1) %n70.scalar.output.ptr, align 4
br label %n70.scalar.step
n70.scalar.step:
%n70.scalar.next = add i32 %n70.scalar.p, %threads
br label %n70.scalar.loop
n70.scalar.done:
call void @grid_barrier(i32 %threads)%n71.source = getelementptr i8, ptr addrspace(1) %values, i64 4416
%n71.second = getelementptr i8, ptr addrspace(1) %values, i64 4480
%n71.value = getelementptr i8, ptr addrspace(1) %values, i64 4544
%n71.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1112
%n71.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n71.scalar.entry
n71.scalar.entry:
%n71.scalar.count = mul i32 %rows, 16
br label %n71.scalar.loop
n71.scalar.loop:
%n71.scalar.p = phi i32 [ %tid, %n71.scalar.entry ], [ %n71.scalar.next, %n71.scalar.step ]
%n71.scalar.more = icmp ult i32 %n71.scalar.p, %n71.scalar.count
br i1 %n71.scalar.more, label %n71.scalar.body, label %n71.scalar.done
n71.scalar.body:
%n71.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n71.source, i32 %n71.scalar.p
%n71.scalar.first = load float, ptr addrspace(1) %n71.scalar.first.ptr, align 4
%n71.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n71.second, i32 %n71.scalar.p
%n71.scalar.second = load float, ptr addrspace(1) %n71.scalar.second.ptr, align 4
%n71.scalar.scalar.0 = call float @recipe.add(float %n71.scalar.first, float %n71.scalar.second)
%n71.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n71.value, i32 %n71.scalar.p
store float %n71.scalar.scalar.0, ptr addrspace(1) %n71.scalar.output.ptr, align 4
br label %n71.scalar.step
n71.scalar.step:
%n71.scalar.next = add i32 %n71.scalar.p, %threads
br label %n71.scalar.loop
n71.scalar.done:
call void @grid_barrier(i32 %threads)%n72.source = getelementptr i8, ptr addrspace(1) %values, i64 4544
%n72.value = getelementptr i8, ptr addrspace(1) %values, i64 4608
%n72.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1120
%n72.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n72.scalar.entry
n72.scalar.entry:
%n72.scalar.count = mul i32 %rows, 16
br label %n72.scalar.loop
n72.scalar.loop:
%n72.scalar.p = phi i32 [ %tid, %n72.scalar.entry ], [ %n72.scalar.next, %n72.scalar.step ]
%n72.scalar.more = icmp ult i32 %n72.scalar.p, %n72.scalar.count
br i1 %n72.scalar.more, label %n72.scalar.body, label %n72.scalar.done
n72.scalar.body:
%n72.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n72.source, i32 %n72.scalar.p
%n72.scalar.first = load float, ptr addrspace(1) %n72.scalar.first.ptr, align 4
%n72.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n72.scalar.first)
%n72.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n72.value, i32 %n72.scalar.p
store float %n72.scalar.scalar.1, ptr addrspace(1) %n72.scalar.output.ptr, align 4
br label %n72.scalar.step
n72.scalar.step:
%n72.scalar.next = add i32 %n72.scalar.p, %threads
br label %n72.scalar.loop
n72.scalar.done:
call void @grid_barrier(i32 %threads)%n73.source = getelementptr i8, ptr addrspace(1) %values, i64 4416
%n73.second = getelementptr i8, ptr addrspace(1) %values, i64 4608
%n73.value = getelementptr i8, ptr addrspace(1) %values, i64 4672
%n73.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1128
%n73.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n73.scalar.entry
n73.scalar.entry:
%n73.scalar.count = mul i32 %rows, 16
br label %n73.scalar.loop
n73.scalar.loop:
%n73.scalar.p = phi i32 [ %tid, %n73.scalar.entry ], [ %n73.scalar.next, %n73.scalar.step ]
%n73.scalar.more = icmp ult i32 %n73.scalar.p, %n73.scalar.count
br i1 %n73.scalar.more, label %n73.scalar.body, label %n73.scalar.done
n73.scalar.body:
%n73.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n73.source, i32 %n73.scalar.p
%n73.scalar.first = load float, ptr addrspace(1) %n73.scalar.first.ptr, align 4
%n73.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n73.second, i32 %n73.scalar.p
%n73.scalar.second = load float, ptr addrspace(1) %n73.scalar.second.ptr, align 4
%n73.scalar.scalar.0 = call float @recipe.add(float %n73.scalar.first, float %n73.scalar.second)
%n73.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n73.value, i32 %n73.scalar.p
store float %n73.scalar.scalar.0, ptr addrspace(1) %n73.scalar.output.ptr, align 4
br label %n73.scalar.step
n73.scalar.step:
%n73.scalar.next = add i32 %n73.scalar.p, %threads
br label %n73.scalar.loop
n73.scalar.done:
call void @grid_barrier(i32 %threads)%n74.source = getelementptr i8, ptr addrspace(1) %values, i64 4352
%n74.second = getelementptr i8, ptr addrspace(1) %values, i64 4672
%n74.value = getelementptr i8, ptr addrspace(1) %values, i64 4736
%n74.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1136
%n74.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n74.scalar.entry
n74.scalar.entry:
%n74.scalar.count = mul i32 %rows, 16
br label %n74.scalar.loop
n74.scalar.loop:
%n74.scalar.p = phi i32 [ %tid, %n74.scalar.entry ], [ %n74.scalar.next, %n74.scalar.step ]
%n74.scalar.more = icmp ult i32 %n74.scalar.p, %n74.scalar.count
br i1 %n74.scalar.more, label %n74.scalar.body, label %n74.scalar.done
n74.scalar.body:
%n74.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n74.source, i32 %n74.scalar.p
%n74.scalar.first = load float, ptr addrspace(1) %n74.scalar.first.ptr, align 4
%n74.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n74.second, i32 %n74.scalar.p
%n74.scalar.second = load float, ptr addrspace(1) %n74.scalar.second.ptr, align 4
%n74.scalar.scalar.0 = call float @recipe.add(float %n74.scalar.first, float %n74.scalar.second)
%n74.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n74.value, i32 %n74.scalar.p
store float %n74.scalar.scalar.0, ptr addrspace(1) %n74.scalar.output.ptr, align 4
br label %n74.scalar.step
n74.scalar.step:
%n74.scalar.next = add i32 %n74.scalar.p, %threads
br label %n74.scalar.loop
n74.scalar.done:
call void @grid_barrier(i32 %threads)%n75.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n75.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n75.value = getelementptr i8, ptr addrspace(1) %values, i64 4800
%n75.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1144
%n75.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n75.scalar.entry
n75.scalar.entry:
%n75.scalar.count = mul i32 %rows, 16
br label %n75.scalar.loop
n75.scalar.loop:
%n75.scalar.p = phi i32 [ %tid, %n75.scalar.entry ], [ %n75.scalar.next, %n75.scalar.step ]
%n75.scalar.more = icmp ult i32 %n75.scalar.p, %n75.scalar.count
br i1 %n75.scalar.more, label %n75.scalar.body, label %n75.scalar.done
n75.scalar.body:
%n75.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n75.source, i32 %n75.scalar.p
%n75.scalar.first = load float, ptr addrspace(1) %n75.scalar.first.ptr, align 4
%n75.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n75.second, i32 %n75.scalar.p
%n75.scalar.second = load float, ptr addrspace(1) %n75.scalar.second.ptr, align 4
%n75.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n75.scalar.first, float %n75.scalar.second)
%n75.scalar.scalar.0 = call float @recipe.from.u1(i1 %n75.scalar.scalar.0.condition)
%n75.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n75.value, i32 %n75.scalar.p
store float %n75.scalar.scalar.0, ptr addrspace(1) %n75.scalar.output.ptr, align 4
br label %n75.scalar.step
n75.scalar.step:
%n75.scalar.next = add i32 %n75.scalar.p, %threads
br label %n75.scalar.loop
n75.scalar.done:
call void @grid_barrier(i32 %threads)%n76.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n76.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n76.value = getelementptr i8, ptr addrspace(1) %values, i64 4864
%n76.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1152
%n76.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n76.scalar.entry
n76.scalar.entry:
%n76.scalar.count = mul i32 %rows, 16
br label %n76.scalar.loop
n76.scalar.loop:
%n76.scalar.p = phi i32 [ %tid, %n76.scalar.entry ], [ %n76.scalar.next, %n76.scalar.step ]
%n76.scalar.more = icmp ult i32 %n76.scalar.p, %n76.scalar.count
br i1 %n76.scalar.more, label %n76.scalar.body, label %n76.scalar.done
n76.scalar.body:
%n76.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n76.source, i32 %n76.scalar.p
%n76.scalar.first = load float, ptr addrspace(1) %n76.scalar.first.ptr, align 4
%n76.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n76.second, i32 %n76.scalar.p
%n76.scalar.second = load float, ptr addrspace(1) %n76.scalar.second.ptr, align 4
%n76.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n76.scalar.first, float %n76.scalar.second)
%n76.scalar.scalar.0 = call float @recipe.from.u1(i1 %n76.scalar.scalar.0.condition)
%n76.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n76.value, i32 %n76.scalar.p
store float %n76.scalar.scalar.0, ptr addrspace(1) %n76.scalar.output.ptr, align 4
br label %n76.scalar.step
n76.scalar.step:
%n76.scalar.next = add i32 %n76.scalar.p, %threads
br label %n76.scalar.loop
n76.scalar.done:
call void @grid_barrier(i32 %threads)%n77.source = getelementptr i8, ptr addrspace(1) %values, i64 4800
%n77.second = getelementptr i8, ptr addrspace(1) %values, i64 4864
%n77.value = getelementptr i8, ptr addrspace(1) %values, i64 4928
%n77.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1160
%n77.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n77.scalar.entry
n77.scalar.entry:
%n77.scalar.count = mul i32 %rows, 16
br label %n77.scalar.loop
n77.scalar.loop:
%n77.scalar.p = phi i32 [ %tid, %n77.scalar.entry ], [ %n77.scalar.next, %n77.scalar.step ]
%n77.scalar.more = icmp ult i32 %n77.scalar.p, %n77.scalar.count
br i1 %n77.scalar.more, label %n77.scalar.body, label %n77.scalar.done
n77.scalar.body:
%n77.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n77.source, i32 %n77.scalar.p
%n77.scalar.first = load float, ptr addrspace(1) %n77.scalar.first.ptr, align 4
%n77.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n77.second, i32 %n77.scalar.p
%n77.scalar.second = load float, ptr addrspace(1) %n77.scalar.second.ptr, align 4
%n77.scalar.scalar.0 = call float @recipe.add(float %n77.scalar.first, float %n77.scalar.second)
%n77.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n77.value, i32 %n77.scalar.p
store float %n77.scalar.scalar.0, ptr addrspace(1) %n77.scalar.output.ptr, align 4
br label %n77.scalar.step
n77.scalar.step:
%n77.scalar.next = add i32 %n77.scalar.p, %threads
br label %n77.scalar.loop
n77.scalar.done:
call void @grid_barrier(i32 %threads)%n78.source = getelementptr i8, ptr addrspace(1) %values, i64 4928
%n78.value = getelementptr i8, ptr addrspace(1) %values, i64 4992
%n78.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1168
%n78.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n78.scalar.entry
n78.scalar.entry:
%n78.scalar.count = mul i32 %rows, 16
br label %n78.scalar.loop
n78.scalar.loop:
%n78.scalar.p = phi i32 [ %tid, %n78.scalar.entry ], [ %n78.scalar.next, %n78.scalar.step ]
%n78.scalar.more = icmp ult i32 %n78.scalar.p, %n78.scalar.count
br i1 %n78.scalar.more, label %n78.scalar.body, label %n78.scalar.done
n78.scalar.body:
%n78.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n78.source, i32 %n78.scalar.p
%n78.scalar.first = load float, ptr addrspace(1) %n78.scalar.first.ptr, align 4
%n78.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n78.scalar.first)
%n78.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n78.value, i32 %n78.scalar.p
store float %n78.scalar.scalar.1, ptr addrspace(1) %n78.scalar.output.ptr, align 4
br label %n78.scalar.step
n78.scalar.step:
%n78.scalar.next = add i32 %n78.scalar.p, %threads
br label %n78.scalar.loop
n78.scalar.done:
call void @grid_barrier(i32 %threads)%n79.source = getelementptr i8, ptr addrspace(1) %values, i64 4800
%n79.second = getelementptr i8, ptr addrspace(1) %values, i64 4992
%n79.value = getelementptr i8, ptr addrspace(1) %values, i64 5056
%n79.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1176
%n79.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n79.scalar.entry
n79.scalar.entry:
%n79.scalar.count = mul i32 %rows, 16
br label %n79.scalar.loop
n79.scalar.loop:
%n79.scalar.p = phi i32 [ %tid, %n79.scalar.entry ], [ %n79.scalar.next, %n79.scalar.step ]
%n79.scalar.more = icmp ult i32 %n79.scalar.p, %n79.scalar.count
br i1 %n79.scalar.more, label %n79.scalar.body, label %n79.scalar.done
n79.scalar.body:
%n79.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n79.source, i32 %n79.scalar.p
%n79.scalar.first = load float, ptr addrspace(1) %n79.scalar.first.ptr, align 4
%n79.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n79.second, i32 %n79.scalar.p
%n79.scalar.second = load float, ptr addrspace(1) %n79.scalar.second.ptr, align 4
%n79.scalar.scalar.0 = call float @recipe.add(float %n79.scalar.first, float %n79.scalar.second)
%n79.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n79.value, i32 %n79.scalar.p
store float %n79.scalar.scalar.0, ptr addrspace(1) %n79.scalar.output.ptr, align 4
br label %n79.scalar.step
n79.scalar.step:
%n79.scalar.next = add i32 %n79.scalar.p, %threads
br label %n79.scalar.loop
n79.scalar.done:
call void @grid_barrier(i32 %threads)%n80.source = getelementptr i8, ptr addrspace(1) %values, i64 4736
%n80.second = getelementptr i8, ptr addrspace(1) %values, i64 5056
%n80.value = getelementptr i8, ptr addrspace(1) %values, i64 5120
%n80.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1184
%n80.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n80.scalar.entry
n80.scalar.entry:
%n80.scalar.count = mul i32 %rows, 16
br label %n80.scalar.loop
n80.scalar.loop:
%n80.scalar.p = phi i32 [ %tid, %n80.scalar.entry ], [ %n80.scalar.next, %n80.scalar.step ]
%n80.scalar.more = icmp ult i32 %n80.scalar.p, %n80.scalar.count
br i1 %n80.scalar.more, label %n80.scalar.body, label %n80.scalar.done
n80.scalar.body:
%n80.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n80.source, i32 %n80.scalar.p
%n80.scalar.first = load float, ptr addrspace(1) %n80.scalar.first.ptr, align 4
%n80.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n80.second, i32 %n80.scalar.p
%n80.scalar.second = load float, ptr addrspace(1) %n80.scalar.second.ptr, align 4
%n80.scalar.scalar.0 = call float @recipe.add(float %n80.scalar.first, float %n80.scalar.second)
%n80.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n80.value, i32 %n80.scalar.p
store float %n80.scalar.scalar.0, ptr addrspace(1) %n80.scalar.output.ptr, align 4
br label %n80.scalar.step
n80.scalar.step:
%n80.scalar.next = add i32 %n80.scalar.p, %threads
br label %n80.scalar.loop
n80.scalar.done:
call void @grid_barrier(i32 %threads)%n81.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n81.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n81.value = getelementptr i8, ptr addrspace(1) %values, i64 5184
%n81.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1192
%n81.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n81.scalar.entry
n81.scalar.entry:
%n81.scalar.count = mul i32 %rows, 16
br label %n81.scalar.loop
n81.scalar.loop:
%n81.scalar.p = phi i32 [ %tid, %n81.scalar.entry ], [ %n81.scalar.next, %n81.scalar.step ]
%n81.scalar.more = icmp ult i32 %n81.scalar.p, %n81.scalar.count
br i1 %n81.scalar.more, label %n81.scalar.body, label %n81.scalar.done
n81.scalar.body:
%n81.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n81.source, i32 %n81.scalar.p
%n81.scalar.first = load float, ptr addrspace(1) %n81.scalar.first.ptr, align 4
%n81.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n81.second, i32 %n81.scalar.p
%n81.scalar.second = load float, ptr addrspace(1) %n81.scalar.second.ptr, align 4
%n81.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n81.scalar.first, float %n81.scalar.second)
%n81.scalar.scalar.0 = call float @recipe.from.u1(i1 %n81.scalar.scalar.0.condition)
%n81.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n81.value, i32 %n81.scalar.p
store float %n81.scalar.scalar.0, ptr addrspace(1) %n81.scalar.output.ptr, align 4
br label %n81.scalar.step
n81.scalar.step:
%n81.scalar.next = add i32 %n81.scalar.p, %threads
br label %n81.scalar.loop
n81.scalar.done:
call void @grid_barrier(i32 %threads)%n82.source = getelementptr i8, ptr addrspace(1) %values, i64 5120
%n82.second = getelementptr i8, ptr addrspace(1) %values, i64 5184
%n82.value = getelementptr i8, ptr addrspace(1) %values, i64 5248
%n82.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1200
%n82.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n82.scalar.entry
n82.scalar.entry:
%n82.scalar.count = mul i32 %rows, 16
br label %n82.scalar.loop
n82.scalar.loop:
%n82.scalar.p = phi i32 [ %tid, %n82.scalar.entry ], [ %n82.scalar.next, %n82.scalar.step ]
%n82.scalar.more = icmp ult i32 %n82.scalar.p, %n82.scalar.count
br i1 %n82.scalar.more, label %n82.scalar.body, label %n82.scalar.done
n82.scalar.body:
%n82.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n82.source, i32 %n82.scalar.p
%n82.scalar.first = load float, ptr addrspace(1) %n82.scalar.first.ptr, align 4
%n82.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n82.second, i32 %n82.scalar.p
%n82.scalar.second = load float, ptr addrspace(1) %n82.scalar.second.ptr, align 4
%n82.scalar.scalar.0 = call float @recipe.add(float %n82.scalar.first, float %n82.scalar.second)
%n82.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n82.value, i32 %n82.scalar.p
store float %n82.scalar.scalar.0, ptr addrspace(1) %n82.scalar.output.ptr, align 4
br label %n82.scalar.step
n82.scalar.step:
%n82.scalar.next = add i32 %n82.scalar.p, %threads
br label %n82.scalar.loop
n82.scalar.done:
call void @grid_barrier(i32 %threads)%n83.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n83.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n83.value = getelementptr i8, ptr addrspace(1) %values, i64 5312
%n83.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1208
%n83.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n83.scalar.entry
n83.scalar.entry:
%n83.scalar.count = mul i32 %rows, 16
br label %n83.scalar.loop
n83.scalar.loop:
%n83.scalar.p = phi i32 [ %tid, %n83.scalar.entry ], [ %n83.scalar.next, %n83.scalar.step ]
%n83.scalar.more = icmp ult i32 %n83.scalar.p, %n83.scalar.count
br i1 %n83.scalar.more, label %n83.scalar.body, label %n83.scalar.done
n83.scalar.body:
%n83.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n83.source, i32 %n83.scalar.p
%n83.scalar.first = load float, ptr addrspace(1) %n83.scalar.first.ptr, align 4
%n83.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n83.second, i32 %n83.scalar.p
%n83.scalar.second = load float, ptr addrspace(1) %n83.scalar.second.ptr, align 4
%n83.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n83.scalar.first, float %n83.scalar.second)
%n83.scalar.scalar.0 = call float @recipe.from.u1(i1 %n83.scalar.scalar.0.condition)
%n83.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n83.value, i32 %n83.scalar.p
store float %n83.scalar.scalar.0, ptr addrspace(1) %n83.scalar.output.ptr, align 4
br label %n83.scalar.step
n83.scalar.step:
%n83.scalar.next = add i32 %n83.scalar.p, %threads
br label %n83.scalar.loop
n83.scalar.done:
call void @grid_barrier(i32 %threads)%n84.source = getelementptr i8, ptr addrspace(1) %values, i64 5248
%n84.second = getelementptr i8, ptr addrspace(1) %values, i64 5312
%n84.value = getelementptr i8, ptr addrspace(1) %values, i64 5376
%n84.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1216
%n84.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n84.scalar.entry
n84.scalar.entry:
%n84.scalar.count = mul i32 %rows, 16
br label %n84.scalar.loop
n84.scalar.loop:
%n84.scalar.p = phi i32 [ %tid, %n84.scalar.entry ], [ %n84.scalar.next, %n84.scalar.step ]
%n84.scalar.more = icmp ult i32 %n84.scalar.p, %n84.scalar.count
br i1 %n84.scalar.more, label %n84.scalar.body, label %n84.scalar.done
n84.scalar.body:
%n84.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n84.source, i32 %n84.scalar.p
%n84.scalar.first = load float, ptr addrspace(1) %n84.scalar.first.ptr, align 4
%n84.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n84.second, i32 %n84.scalar.p
%n84.scalar.second = load float, ptr addrspace(1) %n84.scalar.second.ptr, align 4
%n84.scalar.scalar.0 = call float @recipe.add(float %n84.scalar.first, float %n84.scalar.second)
%n84.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n84.value, i32 %n84.scalar.p
store float %n84.scalar.scalar.0, ptr addrspace(1) %n84.scalar.output.ptr, align 4
br label %n84.scalar.step
n84.scalar.step:
%n84.scalar.next = add i32 %n84.scalar.p, %threads
br label %n84.scalar.loop
n84.scalar.done:
call void @grid_barrier(i32 %threads)%n85.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n85.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n85.value = getelementptr i8, ptr addrspace(1) %values, i64 5440
%n85.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1224
%n85.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n85.scalar.entry
n85.scalar.entry:
%n85.scalar.count = mul i32 %rows, 16
br label %n85.scalar.loop
n85.scalar.loop:
%n85.scalar.p = phi i32 [ %tid, %n85.scalar.entry ], [ %n85.scalar.next, %n85.scalar.step ]
%n85.scalar.more = icmp ult i32 %n85.scalar.p, %n85.scalar.count
br i1 %n85.scalar.more, label %n85.scalar.body, label %n85.scalar.done
n85.scalar.body:
%n85.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n85.source, i32 %n85.scalar.p
%n85.scalar.first = load float, ptr addrspace(1) %n85.scalar.first.ptr, align 4
%n85.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n85.second, i32 %n85.scalar.p
%n85.scalar.second = load float, ptr addrspace(1) %n85.scalar.second.ptr, align 4
%n85.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n85.scalar.first, float %n85.scalar.second)
%n85.scalar.scalar.0 = call float @recipe.from.u1(i1 %n85.scalar.scalar.0.condition)
%n85.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n85.value, i32 %n85.scalar.p
store float %n85.scalar.scalar.0, ptr addrspace(1) %n85.scalar.output.ptr, align 4
br label %n85.scalar.step
n85.scalar.step:
%n85.scalar.next = add i32 %n85.scalar.p, %threads
br label %n85.scalar.loop
n85.scalar.done:
call void @grid_barrier(i32 %threads)%n86.source = getelementptr i8, ptr addrspace(1) %values, i64 5376
%n86.second = getelementptr i8, ptr addrspace(1) %values, i64 5440
%n86.value = getelementptr i8, ptr addrspace(1) %values, i64 5504
%n86.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1232
%n86.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n86.scalar.entry
n86.scalar.entry:
%n86.scalar.count = mul i32 %rows, 16
br label %n86.scalar.loop
n86.scalar.loop:
%n86.scalar.p = phi i32 [ %tid, %n86.scalar.entry ], [ %n86.scalar.next, %n86.scalar.step ]
%n86.scalar.more = icmp ult i32 %n86.scalar.p, %n86.scalar.count
br i1 %n86.scalar.more, label %n86.scalar.body, label %n86.scalar.done
n86.scalar.body:
%n86.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n86.source, i32 %n86.scalar.p
%n86.scalar.first = load float, ptr addrspace(1) %n86.scalar.first.ptr, align 4
%n86.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n86.second, i32 %n86.scalar.p
%n86.scalar.second = load float, ptr addrspace(1) %n86.scalar.second.ptr, align 4
%n86.scalar.scalar.0 = call float @recipe.add(float %n86.scalar.first, float %n86.scalar.second)
%n86.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n86.value, i32 %n86.scalar.p
store float %n86.scalar.scalar.0, ptr addrspace(1) %n86.scalar.output.ptr, align 4
br label %n86.scalar.step
n86.scalar.step:
%n86.scalar.next = add i32 %n86.scalar.p, %threads
br label %n86.scalar.loop
n86.scalar.done:
call void @grid_barrier(i32 %threads)%n87.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n87.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n87.value = getelementptr i8, ptr addrspace(1) %values, i64 5568
%n87.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1240
%n87.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n87.scalar.entry
n87.scalar.entry:
%n87.scalar.count = mul i32 %rows, 16
br label %n87.scalar.loop
n87.scalar.loop:
%n87.scalar.p = phi i32 [ %tid, %n87.scalar.entry ], [ %n87.scalar.next, %n87.scalar.step ]
%n87.scalar.more = icmp ult i32 %n87.scalar.p, %n87.scalar.count
br i1 %n87.scalar.more, label %n87.scalar.body, label %n87.scalar.done
n87.scalar.body:
%n87.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n87.source, i32 %n87.scalar.p
%n87.scalar.first = load float, ptr addrspace(1) %n87.scalar.first.ptr, align 4
%n87.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n87.second, i32 %n87.scalar.p
%n87.scalar.second = load float, ptr addrspace(1) %n87.scalar.second.ptr, align 4
%n87.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n87.scalar.first, float %n87.scalar.second)
%n87.scalar.scalar.0 = call float @recipe.from.u1(i1 %n87.scalar.scalar.0.condition)
%n87.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n87.value, i32 %n87.scalar.p
store float %n87.scalar.scalar.0, ptr addrspace(1) %n87.scalar.output.ptr, align 4
br label %n87.scalar.step
n87.scalar.step:
%n87.scalar.next = add i32 %n87.scalar.p, %threads
br label %n87.scalar.loop
n87.scalar.done:
call void @grid_barrier(i32 %threads)%n88.source = getelementptr i8, ptr addrspace(1) %values, i64 5504
%n88.second = getelementptr i8, ptr addrspace(1) %values, i64 5568
%n88.value = getelementptr i8, ptr addrspace(1) %values, i64 5632
%n88.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1248
%n88.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n88.scalar.entry
n88.scalar.entry:
%n88.scalar.count = mul i32 %rows, 16
br label %n88.scalar.loop
n88.scalar.loop:
%n88.scalar.p = phi i32 [ %tid, %n88.scalar.entry ], [ %n88.scalar.next, %n88.scalar.step ]
%n88.scalar.more = icmp ult i32 %n88.scalar.p, %n88.scalar.count
br i1 %n88.scalar.more, label %n88.scalar.body, label %n88.scalar.done
n88.scalar.body:
%n88.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n88.source, i32 %n88.scalar.p
%n88.scalar.first = load float, ptr addrspace(1) %n88.scalar.first.ptr, align 4
%n88.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n88.second, i32 %n88.scalar.p
%n88.scalar.second = load float, ptr addrspace(1) %n88.scalar.second.ptr, align 4
%n88.scalar.scalar.0 = call float @recipe.add(float %n88.scalar.first, float %n88.scalar.second)
%n88.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n88.value, i32 %n88.scalar.p
store float %n88.scalar.scalar.0, ptr addrspace(1) %n88.scalar.output.ptr, align 4
br label %n88.scalar.step
n88.scalar.step:
%n88.scalar.next = add i32 %n88.scalar.p, %threads
br label %n88.scalar.loop
n88.scalar.done:
call void @grid_barrier(i32 %threads)%n89.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n89.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n89.value = getelementptr i8, ptr addrspace(1) %values, i64 5696
%n89.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1256
%n89.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n89.scalar.entry
n89.scalar.entry:
%n89.scalar.count = mul i32 %rows, 16
br label %n89.scalar.loop
n89.scalar.loop:
%n89.scalar.p = phi i32 [ %tid, %n89.scalar.entry ], [ %n89.scalar.next, %n89.scalar.step ]
%n89.scalar.more = icmp ult i32 %n89.scalar.p, %n89.scalar.count
br i1 %n89.scalar.more, label %n89.scalar.body, label %n89.scalar.done
n89.scalar.body:
%n89.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n89.source, i32 %n89.scalar.p
%n89.scalar.first = load float, ptr addrspace(1) %n89.scalar.first.ptr, align 4
%n89.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n89.second, i32 %n89.scalar.p
%n89.scalar.second = load float, ptr addrspace(1) %n89.scalar.second.ptr, align 4
%n89.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n89.scalar.first, float %n89.scalar.second)
%n89.scalar.scalar.0 = call float @recipe.from.u1(i1 %n89.scalar.scalar.0.condition)
%n89.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n89.value, i32 %n89.scalar.p
store float %n89.scalar.scalar.0, ptr addrspace(1) %n89.scalar.output.ptr, align 4
br label %n89.scalar.step
n89.scalar.step:
%n89.scalar.next = add i32 %n89.scalar.p, %threads
br label %n89.scalar.loop
n89.scalar.done:
call void @grid_barrier(i32 %threads)%n90.source = getelementptr i8, ptr addrspace(1) %values, i64 5632
%n90.second = getelementptr i8, ptr addrspace(1) %values, i64 5696
%n90.value = getelementptr i8, ptr addrspace(1) %values, i64 5760
%n90.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1264
%n90.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n90.scalar.entry
n90.scalar.entry:
%n90.scalar.count = mul i32 %rows, 16
br label %n90.scalar.loop
n90.scalar.loop:
%n90.scalar.p = phi i32 [ %tid, %n90.scalar.entry ], [ %n90.scalar.next, %n90.scalar.step ]
%n90.scalar.more = icmp ult i32 %n90.scalar.p, %n90.scalar.count
br i1 %n90.scalar.more, label %n90.scalar.body, label %n90.scalar.done
n90.scalar.body:
%n90.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n90.source, i32 %n90.scalar.p
%n90.scalar.first = load float, ptr addrspace(1) %n90.scalar.first.ptr, align 4
%n90.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n90.second, i32 %n90.scalar.p
%n90.scalar.second = load float, ptr addrspace(1) %n90.scalar.second.ptr, align 4
%n90.scalar.scalar.0 = call float @recipe.add(float %n90.scalar.first, float %n90.scalar.second)
%n90.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n90.value, i32 %n90.scalar.p
store float %n90.scalar.scalar.0, ptr addrspace(1) %n90.scalar.output.ptr, align 4
br label %n90.scalar.step
n90.scalar.step:
%n90.scalar.next = add i32 %n90.scalar.p, %threads
br label %n90.scalar.loop
n90.scalar.done:
call void @grid_barrier(i32 %threads)%n91.source = getelementptr i8, ptr addrspace(1) %values, i64 5760
%n91.value = getelementptr i8, ptr addrspace(1) %values, i64 5824
%n91.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1272
%n91.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n91.scalar.entry
n91.scalar.entry:
%n91.scalar.count = mul i32 %rows, 16
br label %n91.scalar.loop
n91.scalar.loop:
%n91.scalar.p = phi i32 [ %tid, %n91.scalar.entry ], [ %n91.scalar.next, %n91.scalar.step ]
%n91.scalar.more = icmp ult i32 %n91.scalar.p, %n91.scalar.count
br i1 %n91.scalar.more, label %n91.scalar.body, label %n91.scalar.done
n91.scalar.body:
%n91.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n91.source, i32 %n91.scalar.p
%n91.scalar.first = load float, ptr addrspace(1) %n91.scalar.first.ptr, align 4
%n91.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n91.scalar.first)
%n91.scalar.scalar.1 = call float @recipe.from.u1(i1 %n91.scalar.scalar.1.condition)
%n91.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n91.value, i32 %n91.scalar.p
store float %n91.scalar.scalar.1, ptr addrspace(1) %n91.scalar.output.ptr, align 4
br label %n91.scalar.step
n91.scalar.step:
%n91.scalar.next = add i32 %n91.scalar.p, %threads
br label %n91.scalar.loop
n91.scalar.done:
call void @grid_barrier(i32 %threads)%n92.source = getelementptr i8, ptr addrspace(1) %values, i64 5824
%n92.second = getelementptr i8, ptr addrspace(1) %values, i64 4288
%n92.value = getelementptr i8, ptr addrspace(1) %values, i64 5888
%n92.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1280
%n92.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n92.scalar.entry
n92.scalar.entry:
%n92.scalar.count = mul i32 %rows, 16
br label %n92.scalar.loop
n92.scalar.loop:
%n92.scalar.p = phi i32 [ %tid, %n92.scalar.entry ], [ %n92.scalar.next, %n92.scalar.step ]
%n92.scalar.more = icmp ult i32 %n92.scalar.p, %n92.scalar.count
br i1 %n92.scalar.more, label %n92.scalar.body, label %n92.scalar.done
n92.scalar.body:
%n92.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n92.source, i32 %n92.scalar.p
%n92.scalar.first = load float, ptr addrspace(1) %n92.scalar.first.ptr, align 4
%n92.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n92.second, i32 %n92.scalar.p
%n92.scalar.second = load float, ptr addrspace(1) %n92.scalar.second.ptr, align 4
%n92.scalar.scalar.0 = call float @recipe.mul(float %n92.scalar.first, float %n92.scalar.second)
%n92.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n92.value, i32 %n92.scalar.p
store float %n92.scalar.scalar.0, ptr addrspace(1) %n92.scalar.output.ptr, align 4
br label %n92.scalar.step
n92.scalar.step:
%n92.scalar.next = add i32 %n92.scalar.p, %threads
br label %n92.scalar.loop
n92.scalar.done:
call void @grid_barrier(i32 %threads)%n93.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n93.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n93.value = getelementptr i8, ptr addrspace(1) %values, i64 5952
%n93.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1288
%n93.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n93.scalar.entry
n93.scalar.entry:
%n93.scalar.count = mul i32 %rows, 16
br label %n93.scalar.loop
n93.scalar.loop:
%n93.scalar.p = phi i32 [ %tid, %n93.scalar.entry ], [ %n93.scalar.next, %n93.scalar.step ]
%n93.scalar.more = icmp ult i32 %n93.scalar.p, %n93.scalar.count
br i1 %n93.scalar.more, label %n93.scalar.body, label %n93.scalar.done
n93.scalar.body:
%n93.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n93.source, i32 %n93.scalar.p
%n93.scalar.first = load float, ptr addrspace(1) %n93.scalar.first.ptr, align 4
%n93.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n93.second, i32 %n93.scalar.p
%n93.scalar.second = load float, ptr addrspace(1) %n93.scalar.second.ptr, align 4
%n93.scalar.scalar.0 = call float @recipe.sub(float %n93.scalar.first, float %n93.scalar.second)
%n93.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n93.value, i32 %n93.scalar.p
store float %n93.scalar.scalar.0, ptr addrspace(1) %n93.scalar.output.ptr, align 4
br label %n93.scalar.step
n93.scalar.step:
%n93.scalar.next = add i32 %n93.scalar.p, %threads
br label %n93.scalar.loop
n93.scalar.done:
call void @grid_barrier(i32 %threads)%n94.source = getelementptr i8, ptr addrspace(1) %values, i64 5952
%n94.value = getelementptr i8, ptr addrspace(1) %values, i64 6016
%n94.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1296
%n94.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n94.scalar.entry
n94.scalar.entry:
%n94.scalar.count = mul i32 %rows, 16
br label %n94.scalar.loop
n94.scalar.loop:
%n94.scalar.p = phi i32 [ %tid, %n94.scalar.entry ], [ %n94.scalar.next, %n94.scalar.step ]
%n94.scalar.more = icmp ult i32 %n94.scalar.p, %n94.scalar.count
br i1 %n94.scalar.more, label %n94.scalar.body, label %n94.scalar.done
n94.scalar.body:
%n94.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n94.source, i32 %n94.scalar.p
%n94.scalar.first = load float, ptr addrspace(1) %n94.scalar.first.ptr, align 4
%n94.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n94.scalar.first, float 0x0000000000000000)
%n94.scalar.scalar.2 = call float @recipe.from.u1(i1 %n94.scalar.scalar.2.condition)
%n94.scalar.scalar.3 = call float @recipe.exp(float %n94.scalar.first)
%n94.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n94.value, i32 %n94.scalar.p
store float %n94.scalar.scalar.3, ptr addrspace(1) %n94.scalar.output.ptr, align 4
br label %n94.scalar.step
n94.scalar.step:
%n94.scalar.next = add i32 %n94.scalar.p, %threads
br label %n94.scalar.loop
n94.scalar.done:
call void @grid_barrier(i32 %threads)%n95.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n95.value = getelementptr i8, ptr addrspace(1) %values, i64 6080
%n95.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1304
%n95.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n95.scalar.entry
n95.scalar.entry:
%n95.scalar.count = mul i32 %rows, 16
br label %n95.scalar.loop
n95.scalar.loop:
%n95.scalar.p = phi i32 [ %tid, %n95.scalar.entry ], [ %n95.scalar.next, %n95.scalar.step ]
%n95.scalar.more = icmp ult i32 %n95.scalar.p, %n95.scalar.count
br i1 %n95.scalar.more, label %n95.scalar.body, label %n95.scalar.done
n95.scalar.body:
%n95.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n95.source, i32 %n95.scalar.p
%n95.scalar.first = load float, ptr addrspace(1) %n95.scalar.first.ptr, align 4
%n95.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n95.value, i32 %n95.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n95.scalar.output.ptr, align 4
br label %n95.scalar.step
n95.scalar.step:
%n95.scalar.next = add i32 %n95.scalar.p, %threads
br label %n95.scalar.loop
n95.scalar.done:
call void @grid_barrier(i32 %threads)%n96.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n96.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n96.value = getelementptr i8, ptr addrspace(1) %values, i64 6144
%n96.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1312
%n96.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n96.scalar.entry
n96.scalar.entry:
%n96.scalar.count = mul i32 %rows, 16
br label %n96.scalar.loop
n96.scalar.loop:
%n96.scalar.p = phi i32 [ %tid, %n96.scalar.entry ], [ %n96.scalar.next, %n96.scalar.step ]
%n96.scalar.more = icmp ult i32 %n96.scalar.p, %n96.scalar.count
br i1 %n96.scalar.more, label %n96.scalar.body, label %n96.scalar.done
n96.scalar.body:
%n96.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n96.source, i32 %n96.scalar.p
%n96.scalar.first = load float, ptr addrspace(1) %n96.scalar.first.ptr, align 4
%n96.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n96.second, i32 %n96.scalar.p
%n96.scalar.second = load float, ptr addrspace(1) %n96.scalar.second.ptr, align 4
%n96.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n96.scalar.first, float %n96.scalar.second)
%n96.scalar.scalar.0 = call float @recipe.from.u1(i1 %n96.scalar.scalar.0.condition)
%n96.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n96.value, i32 %n96.scalar.p
store float %n96.scalar.scalar.0, ptr addrspace(1) %n96.scalar.output.ptr, align 4
br label %n96.scalar.step
n96.scalar.step:
%n96.scalar.next = add i32 %n96.scalar.p, %threads
br label %n96.scalar.loop
n96.scalar.done:
call void @grid_barrier(i32 %threads)%n97.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n97.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n97.value = getelementptr i8, ptr addrspace(1) %values, i64 6208
%n97.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1320
%n97.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n97.scalar.entry
n97.scalar.entry:
%n97.scalar.count = mul i32 %rows, 16
br label %n97.scalar.loop
n97.scalar.loop:
%n97.scalar.p = phi i32 [ %tid, %n97.scalar.entry ], [ %n97.scalar.next, %n97.scalar.step ]
%n97.scalar.more = icmp ult i32 %n97.scalar.p, %n97.scalar.count
br i1 %n97.scalar.more, label %n97.scalar.body, label %n97.scalar.done
n97.scalar.body:
%n97.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n97.source, i32 %n97.scalar.p
%n97.scalar.first = load float, ptr addrspace(1) %n97.scalar.first.ptr, align 4
%n97.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n97.second, i32 %n97.scalar.p
%n97.scalar.second = load float, ptr addrspace(1) %n97.scalar.second.ptr, align 4
%n97.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n97.scalar.first, float %n97.scalar.second)
%n97.scalar.scalar.0 = call float @recipe.from.u1(i1 %n97.scalar.scalar.0.condition)
%n97.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n97.value, i32 %n97.scalar.p
store float %n97.scalar.scalar.0, ptr addrspace(1) %n97.scalar.output.ptr, align 4
br label %n97.scalar.step
n97.scalar.step:
%n97.scalar.next = add i32 %n97.scalar.p, %threads
br label %n97.scalar.loop
n97.scalar.done:
call void @grid_barrier(i32 %threads)%n98.source = getelementptr i8, ptr addrspace(1) %values, i64 6144
%n98.second = getelementptr i8, ptr addrspace(1) %values, i64 6208
%n98.value = getelementptr i8, ptr addrspace(1) %values, i64 6272
%n98.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1328
%n98.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n98.scalar.entry
n98.scalar.entry:
%n98.scalar.count = mul i32 %rows, 16
br label %n98.scalar.loop
n98.scalar.loop:
%n98.scalar.p = phi i32 [ %tid, %n98.scalar.entry ], [ %n98.scalar.next, %n98.scalar.step ]
%n98.scalar.more = icmp ult i32 %n98.scalar.p, %n98.scalar.count
br i1 %n98.scalar.more, label %n98.scalar.body, label %n98.scalar.done
n98.scalar.body:
%n98.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n98.source, i32 %n98.scalar.p
%n98.scalar.first = load float, ptr addrspace(1) %n98.scalar.first.ptr, align 4
%n98.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n98.second, i32 %n98.scalar.p
%n98.scalar.second = load float, ptr addrspace(1) %n98.scalar.second.ptr, align 4
%n98.scalar.scalar.0 = call float @recipe.add(float %n98.scalar.first, float %n98.scalar.second)
%n98.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n98.value, i32 %n98.scalar.p
store float %n98.scalar.scalar.0, ptr addrspace(1) %n98.scalar.output.ptr, align 4
br label %n98.scalar.step
n98.scalar.step:
%n98.scalar.next = add i32 %n98.scalar.p, %threads
br label %n98.scalar.loop
n98.scalar.done:
call void @grid_barrier(i32 %threads)%n99.source = getelementptr i8, ptr addrspace(1) %values, i64 6272
%n99.value = getelementptr i8, ptr addrspace(1) %values, i64 6336
%n99.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1336
%n99.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n99.scalar.entry
n99.scalar.entry:
%n99.scalar.count = mul i32 %rows, 16
br label %n99.scalar.loop
n99.scalar.loop:
%n99.scalar.p = phi i32 [ %tid, %n99.scalar.entry ], [ %n99.scalar.next, %n99.scalar.step ]
%n99.scalar.more = icmp ult i32 %n99.scalar.p, %n99.scalar.count
br i1 %n99.scalar.more, label %n99.scalar.body, label %n99.scalar.done
n99.scalar.body:
%n99.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n99.source, i32 %n99.scalar.p
%n99.scalar.first = load float, ptr addrspace(1) %n99.scalar.first.ptr, align 4
%n99.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n99.scalar.first)
%n99.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n99.value, i32 %n99.scalar.p
store float %n99.scalar.scalar.1, ptr addrspace(1) %n99.scalar.output.ptr, align 4
br label %n99.scalar.step
n99.scalar.step:
%n99.scalar.next = add i32 %n99.scalar.p, %threads
br label %n99.scalar.loop
n99.scalar.done:
call void @grid_barrier(i32 %threads)%n100.source = getelementptr i8, ptr addrspace(1) %values, i64 6144
%n100.second = getelementptr i8, ptr addrspace(1) %values, i64 6336
%n100.value = getelementptr i8, ptr addrspace(1) %values, i64 6400
%n100.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1344
%n100.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n100.scalar.entry
n100.scalar.entry:
%n100.scalar.count = mul i32 %rows, 16
br label %n100.scalar.loop
n100.scalar.loop:
%n100.scalar.p = phi i32 [ %tid, %n100.scalar.entry ], [ %n100.scalar.next, %n100.scalar.step ]
%n100.scalar.more = icmp ult i32 %n100.scalar.p, %n100.scalar.count
br i1 %n100.scalar.more, label %n100.scalar.body, label %n100.scalar.done
n100.scalar.body:
%n100.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n100.source, i32 %n100.scalar.p
%n100.scalar.first = load float, ptr addrspace(1) %n100.scalar.first.ptr, align 4
%n100.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n100.second, i32 %n100.scalar.p
%n100.scalar.second = load float, ptr addrspace(1) %n100.scalar.second.ptr, align 4
%n100.scalar.scalar.0 = call float @recipe.add(float %n100.scalar.first, float %n100.scalar.second)
%n100.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n100.value, i32 %n100.scalar.p
store float %n100.scalar.scalar.0, ptr addrspace(1) %n100.scalar.output.ptr, align 4
br label %n100.scalar.step
n100.scalar.step:
%n100.scalar.next = add i32 %n100.scalar.p, %threads
br label %n100.scalar.loop
n100.scalar.done:
call void @grid_barrier(i32 %threads)%n101.source = getelementptr i8, ptr addrspace(1) %values, i64 6080
%n101.second = getelementptr i8, ptr addrspace(1) %values, i64 6400
%n101.value = getelementptr i8, ptr addrspace(1) %values, i64 6464
%n101.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1352
%n101.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n101.scalar.entry
n101.scalar.entry:
%n101.scalar.count = mul i32 %rows, 16
br label %n101.scalar.loop
n101.scalar.loop:
%n101.scalar.p = phi i32 [ %tid, %n101.scalar.entry ], [ %n101.scalar.next, %n101.scalar.step ]
%n101.scalar.more = icmp ult i32 %n101.scalar.p, %n101.scalar.count
br i1 %n101.scalar.more, label %n101.scalar.body, label %n101.scalar.done
n101.scalar.body:
%n101.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n101.source, i32 %n101.scalar.p
%n101.scalar.first = load float, ptr addrspace(1) %n101.scalar.first.ptr, align 4
%n101.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n101.second, i32 %n101.scalar.p
%n101.scalar.second = load float, ptr addrspace(1) %n101.scalar.second.ptr, align 4
%n101.scalar.scalar.0 = call float @recipe.add(float %n101.scalar.first, float %n101.scalar.second)
%n101.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n101.value, i32 %n101.scalar.p
store float %n101.scalar.scalar.0, ptr addrspace(1) %n101.scalar.output.ptr, align 4
br label %n101.scalar.step
n101.scalar.step:
%n101.scalar.next = add i32 %n101.scalar.p, %threads
br label %n101.scalar.loop
n101.scalar.done:
call void @grid_barrier(i32 %threads)%n102.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n102.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n102.value = getelementptr i8, ptr addrspace(1) %values, i64 6528
%n102.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1360
%n102.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n102.scalar.entry
n102.scalar.entry:
%n102.scalar.count = mul i32 %rows, 16
br label %n102.scalar.loop
n102.scalar.loop:
%n102.scalar.p = phi i32 [ %tid, %n102.scalar.entry ], [ %n102.scalar.next, %n102.scalar.step ]
%n102.scalar.more = icmp ult i32 %n102.scalar.p, %n102.scalar.count
br i1 %n102.scalar.more, label %n102.scalar.body, label %n102.scalar.done
n102.scalar.body:
%n102.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n102.source, i32 %n102.scalar.p
%n102.scalar.first = load float, ptr addrspace(1) %n102.scalar.first.ptr, align 4
%n102.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n102.second, i32 %n102.scalar.p
%n102.scalar.second = load float, ptr addrspace(1) %n102.scalar.second.ptr, align 4
%n102.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n102.scalar.first, float %n102.scalar.second)
%n102.scalar.scalar.0 = call float @recipe.from.u1(i1 %n102.scalar.scalar.0.condition)
%n102.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n102.value, i32 %n102.scalar.p
store float %n102.scalar.scalar.0, ptr addrspace(1) %n102.scalar.output.ptr, align 4
br label %n102.scalar.step
n102.scalar.step:
%n102.scalar.next = add i32 %n102.scalar.p, %threads
br label %n102.scalar.loop
n102.scalar.done:
call void @grid_barrier(i32 %threads)%n103.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n103.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n103.value = getelementptr i8, ptr addrspace(1) %values, i64 6592
%n103.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1368
%n103.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n103.scalar.entry
n103.scalar.entry:
%n103.scalar.count = mul i32 %rows, 16
br label %n103.scalar.loop
n103.scalar.loop:
%n103.scalar.p = phi i32 [ %tid, %n103.scalar.entry ], [ %n103.scalar.next, %n103.scalar.step ]
%n103.scalar.more = icmp ult i32 %n103.scalar.p, %n103.scalar.count
br i1 %n103.scalar.more, label %n103.scalar.body, label %n103.scalar.done
n103.scalar.body:
%n103.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n103.source, i32 %n103.scalar.p
%n103.scalar.first = load float, ptr addrspace(1) %n103.scalar.first.ptr, align 4
%n103.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n103.second, i32 %n103.scalar.p
%n103.scalar.second = load float, ptr addrspace(1) %n103.scalar.second.ptr, align 4
%n103.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n103.scalar.first, float %n103.scalar.second)
%n103.scalar.scalar.0 = call float @recipe.from.u1(i1 %n103.scalar.scalar.0.condition)
%n103.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n103.value, i32 %n103.scalar.p
store float %n103.scalar.scalar.0, ptr addrspace(1) %n103.scalar.output.ptr, align 4
br label %n103.scalar.step
n103.scalar.step:
%n103.scalar.next = add i32 %n103.scalar.p, %threads
br label %n103.scalar.loop
n103.scalar.done:
call void @grid_barrier(i32 %threads)%n104.source = getelementptr i8, ptr addrspace(1) %values, i64 6528
%n104.second = getelementptr i8, ptr addrspace(1) %values, i64 6592
%n104.value = getelementptr i8, ptr addrspace(1) %values, i64 6656
%n104.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1376
%n104.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n104.scalar.entry
n104.scalar.entry:
%n104.scalar.count = mul i32 %rows, 16
br label %n104.scalar.loop
n104.scalar.loop:
%n104.scalar.p = phi i32 [ %tid, %n104.scalar.entry ], [ %n104.scalar.next, %n104.scalar.step ]
%n104.scalar.more = icmp ult i32 %n104.scalar.p, %n104.scalar.count
br i1 %n104.scalar.more, label %n104.scalar.body, label %n104.scalar.done
n104.scalar.body:
%n104.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n104.source, i32 %n104.scalar.p
%n104.scalar.first = load float, ptr addrspace(1) %n104.scalar.first.ptr, align 4
%n104.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n104.second, i32 %n104.scalar.p
%n104.scalar.second = load float, ptr addrspace(1) %n104.scalar.second.ptr, align 4
%n104.scalar.scalar.0 = call float @recipe.add(float %n104.scalar.first, float %n104.scalar.second)
%n104.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n104.value, i32 %n104.scalar.p
store float %n104.scalar.scalar.0, ptr addrspace(1) %n104.scalar.output.ptr, align 4
br label %n104.scalar.step
n104.scalar.step:
%n104.scalar.next = add i32 %n104.scalar.p, %threads
br label %n104.scalar.loop
n104.scalar.done:
call void @grid_barrier(i32 %threads)%n105.source = getelementptr i8, ptr addrspace(1) %values, i64 6656
%n105.value = getelementptr i8, ptr addrspace(1) %values, i64 6720
%n105.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1384
%n105.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n105.scalar.entry
n105.scalar.entry:
%n105.scalar.count = mul i32 %rows, 16
br label %n105.scalar.loop
n105.scalar.loop:
%n105.scalar.p = phi i32 [ %tid, %n105.scalar.entry ], [ %n105.scalar.next, %n105.scalar.step ]
%n105.scalar.more = icmp ult i32 %n105.scalar.p, %n105.scalar.count
br i1 %n105.scalar.more, label %n105.scalar.body, label %n105.scalar.done
n105.scalar.body:
%n105.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n105.source, i32 %n105.scalar.p
%n105.scalar.first = load float, ptr addrspace(1) %n105.scalar.first.ptr, align 4
%n105.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n105.scalar.first)
%n105.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n105.value, i32 %n105.scalar.p
store float %n105.scalar.scalar.1, ptr addrspace(1) %n105.scalar.output.ptr, align 4
br label %n105.scalar.step
n105.scalar.step:
%n105.scalar.next = add i32 %n105.scalar.p, %threads
br label %n105.scalar.loop
n105.scalar.done:
call void @grid_barrier(i32 %threads)%n106.source = getelementptr i8, ptr addrspace(1) %values, i64 6528
%n106.second = getelementptr i8, ptr addrspace(1) %values, i64 6720
%n106.value = getelementptr i8, ptr addrspace(1) %values, i64 6784
%n106.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1392
%n106.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n106.scalar.entry
n106.scalar.entry:
%n106.scalar.count = mul i32 %rows, 16
br label %n106.scalar.loop
n106.scalar.loop:
%n106.scalar.p = phi i32 [ %tid, %n106.scalar.entry ], [ %n106.scalar.next, %n106.scalar.step ]
%n106.scalar.more = icmp ult i32 %n106.scalar.p, %n106.scalar.count
br i1 %n106.scalar.more, label %n106.scalar.body, label %n106.scalar.done
n106.scalar.body:
%n106.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n106.source, i32 %n106.scalar.p
%n106.scalar.first = load float, ptr addrspace(1) %n106.scalar.first.ptr, align 4
%n106.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n106.second, i32 %n106.scalar.p
%n106.scalar.second = load float, ptr addrspace(1) %n106.scalar.second.ptr, align 4
%n106.scalar.scalar.0 = call float @recipe.add(float %n106.scalar.first, float %n106.scalar.second)
%n106.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n106.value, i32 %n106.scalar.p
store float %n106.scalar.scalar.0, ptr addrspace(1) %n106.scalar.output.ptr, align 4
br label %n106.scalar.step
n106.scalar.step:
%n106.scalar.next = add i32 %n106.scalar.p, %threads
br label %n106.scalar.loop
n106.scalar.done:
call void @grid_barrier(i32 %threads)%n107.source = getelementptr i8, ptr addrspace(1) %values, i64 6464
%n107.second = getelementptr i8, ptr addrspace(1) %values, i64 6784
%n107.value = getelementptr i8, ptr addrspace(1) %values, i64 6848
%n107.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1400
%n107.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n107.scalar.entry
n107.scalar.entry:
%n107.scalar.count = mul i32 %rows, 16
br label %n107.scalar.loop
n107.scalar.loop:
%n107.scalar.p = phi i32 [ %tid, %n107.scalar.entry ], [ %n107.scalar.next, %n107.scalar.step ]
%n107.scalar.more = icmp ult i32 %n107.scalar.p, %n107.scalar.count
br i1 %n107.scalar.more, label %n107.scalar.body, label %n107.scalar.done
n107.scalar.body:
%n107.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n107.source, i32 %n107.scalar.p
%n107.scalar.first = load float, ptr addrspace(1) %n107.scalar.first.ptr, align 4
%n107.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n107.second, i32 %n107.scalar.p
%n107.scalar.second = load float, ptr addrspace(1) %n107.scalar.second.ptr, align 4
%n107.scalar.scalar.0 = call float @recipe.add(float %n107.scalar.first, float %n107.scalar.second)
%n107.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n107.value, i32 %n107.scalar.p
store float %n107.scalar.scalar.0, ptr addrspace(1) %n107.scalar.output.ptr, align 4
br label %n107.scalar.step
n107.scalar.step:
%n107.scalar.next = add i32 %n107.scalar.p, %threads
br label %n107.scalar.loop
n107.scalar.done:
call void @grid_barrier(i32 %threads)%n108.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n108.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n108.value = getelementptr i8, ptr addrspace(1) %values, i64 6912
%n108.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1408
%n108.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n108.scalar.entry
n108.scalar.entry:
%n108.scalar.count = mul i32 %rows, 16
br label %n108.scalar.loop
n108.scalar.loop:
%n108.scalar.p = phi i32 [ %tid, %n108.scalar.entry ], [ %n108.scalar.next, %n108.scalar.step ]
%n108.scalar.more = icmp ult i32 %n108.scalar.p, %n108.scalar.count
br i1 %n108.scalar.more, label %n108.scalar.body, label %n108.scalar.done
n108.scalar.body:
%n108.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n108.source, i32 %n108.scalar.p
%n108.scalar.first = load float, ptr addrspace(1) %n108.scalar.first.ptr, align 4
%n108.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n108.second, i32 %n108.scalar.p
%n108.scalar.second = load float, ptr addrspace(1) %n108.scalar.second.ptr, align 4
%n108.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n108.scalar.first, float %n108.scalar.second)
%n108.scalar.scalar.0 = call float @recipe.from.u1(i1 %n108.scalar.scalar.0.condition)
%n108.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n108.value, i32 %n108.scalar.p
store float %n108.scalar.scalar.0, ptr addrspace(1) %n108.scalar.output.ptr, align 4
br label %n108.scalar.step
n108.scalar.step:
%n108.scalar.next = add i32 %n108.scalar.p, %threads
br label %n108.scalar.loop
n108.scalar.done:
call void @grid_barrier(i32 %threads)%n109.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n109.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n109.value = getelementptr i8, ptr addrspace(1) %values, i64 6976
%n109.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1416
%n109.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n109.scalar.entry
n109.scalar.entry:
%n109.scalar.count = mul i32 %rows, 16
br label %n109.scalar.loop
n109.scalar.loop:
%n109.scalar.p = phi i32 [ %tid, %n109.scalar.entry ], [ %n109.scalar.next, %n109.scalar.step ]
%n109.scalar.more = icmp ult i32 %n109.scalar.p, %n109.scalar.count
br i1 %n109.scalar.more, label %n109.scalar.body, label %n109.scalar.done
n109.scalar.body:
%n109.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n109.source, i32 %n109.scalar.p
%n109.scalar.first = load float, ptr addrspace(1) %n109.scalar.first.ptr, align 4
%n109.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n109.second, i32 %n109.scalar.p
%n109.scalar.second = load float, ptr addrspace(1) %n109.scalar.second.ptr, align 4
%n109.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n109.scalar.first, float %n109.scalar.second)
%n109.scalar.scalar.0 = call float @recipe.from.u1(i1 %n109.scalar.scalar.0.condition)
%n109.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n109.value, i32 %n109.scalar.p
store float %n109.scalar.scalar.0, ptr addrspace(1) %n109.scalar.output.ptr, align 4
br label %n109.scalar.step
n109.scalar.step:
%n109.scalar.next = add i32 %n109.scalar.p, %threads
br label %n109.scalar.loop
n109.scalar.done:
call void @grid_barrier(i32 %threads)%n110.source = getelementptr i8, ptr addrspace(1) %values, i64 6912
%n110.second = getelementptr i8, ptr addrspace(1) %values, i64 6976
%n110.value = getelementptr i8, ptr addrspace(1) %values, i64 7040
%n110.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1424
%n110.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n110.scalar.entry
n110.scalar.entry:
%n110.scalar.count = mul i32 %rows, 16
br label %n110.scalar.loop
n110.scalar.loop:
%n110.scalar.p = phi i32 [ %tid, %n110.scalar.entry ], [ %n110.scalar.next, %n110.scalar.step ]
%n110.scalar.more = icmp ult i32 %n110.scalar.p, %n110.scalar.count
br i1 %n110.scalar.more, label %n110.scalar.body, label %n110.scalar.done
n110.scalar.body:
%n110.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n110.source, i32 %n110.scalar.p
%n110.scalar.first = load float, ptr addrspace(1) %n110.scalar.first.ptr, align 4
%n110.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n110.second, i32 %n110.scalar.p
%n110.scalar.second = load float, ptr addrspace(1) %n110.scalar.second.ptr, align 4
%n110.scalar.scalar.0 = call float @recipe.add(float %n110.scalar.first, float %n110.scalar.second)
%n110.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n110.value, i32 %n110.scalar.p
store float %n110.scalar.scalar.0, ptr addrspace(1) %n110.scalar.output.ptr, align 4
br label %n110.scalar.step
n110.scalar.step:
%n110.scalar.next = add i32 %n110.scalar.p, %threads
br label %n110.scalar.loop
n110.scalar.done:
call void @grid_barrier(i32 %threads)%n111.source = getelementptr i8, ptr addrspace(1) %values, i64 7040
%n111.value = getelementptr i8, ptr addrspace(1) %values, i64 7104
%n111.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1432
%n111.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n111.scalar.entry
n111.scalar.entry:
%n111.scalar.count = mul i32 %rows, 16
br label %n111.scalar.loop
n111.scalar.loop:
%n111.scalar.p = phi i32 [ %tid, %n111.scalar.entry ], [ %n111.scalar.next, %n111.scalar.step ]
%n111.scalar.more = icmp ult i32 %n111.scalar.p, %n111.scalar.count
br i1 %n111.scalar.more, label %n111.scalar.body, label %n111.scalar.done
n111.scalar.body:
%n111.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n111.source, i32 %n111.scalar.p
%n111.scalar.first = load float, ptr addrspace(1) %n111.scalar.first.ptr, align 4
%n111.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n111.scalar.first)
%n111.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n111.value, i32 %n111.scalar.p
store float %n111.scalar.scalar.1, ptr addrspace(1) %n111.scalar.output.ptr, align 4
br label %n111.scalar.step
n111.scalar.step:
%n111.scalar.next = add i32 %n111.scalar.p, %threads
br label %n111.scalar.loop
n111.scalar.done:
call void @grid_barrier(i32 %threads)%n112.source = getelementptr i8, ptr addrspace(1) %values, i64 6912
%n112.second = getelementptr i8, ptr addrspace(1) %values, i64 7104
%n112.value = getelementptr i8, ptr addrspace(1) %values, i64 7168
%n112.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1440
%n112.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n112.scalar.entry
n112.scalar.entry:
%n112.scalar.count = mul i32 %rows, 16
br label %n112.scalar.loop
n112.scalar.loop:
%n112.scalar.p = phi i32 [ %tid, %n112.scalar.entry ], [ %n112.scalar.next, %n112.scalar.step ]
%n112.scalar.more = icmp ult i32 %n112.scalar.p, %n112.scalar.count
br i1 %n112.scalar.more, label %n112.scalar.body, label %n112.scalar.done
n112.scalar.body:
%n112.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n112.source, i32 %n112.scalar.p
%n112.scalar.first = load float, ptr addrspace(1) %n112.scalar.first.ptr, align 4
%n112.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n112.second, i32 %n112.scalar.p
%n112.scalar.second = load float, ptr addrspace(1) %n112.scalar.second.ptr, align 4
%n112.scalar.scalar.0 = call float @recipe.add(float %n112.scalar.first, float %n112.scalar.second)
%n112.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n112.value, i32 %n112.scalar.p
store float %n112.scalar.scalar.0, ptr addrspace(1) %n112.scalar.output.ptr, align 4
br label %n112.scalar.step
n112.scalar.step:
%n112.scalar.next = add i32 %n112.scalar.p, %threads
br label %n112.scalar.loop
n112.scalar.done:
call void @grid_barrier(i32 %threads)%n113.source = getelementptr i8, ptr addrspace(1) %values, i64 6848
%n113.second = getelementptr i8, ptr addrspace(1) %values, i64 7168
%n113.value = getelementptr i8, ptr addrspace(1) %values, i64 7232
%n113.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1448
%n113.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n113.scalar.entry
n113.scalar.entry:
%n113.scalar.count = mul i32 %rows, 16
br label %n113.scalar.loop
n113.scalar.loop:
%n113.scalar.p = phi i32 [ %tid, %n113.scalar.entry ], [ %n113.scalar.next, %n113.scalar.step ]
%n113.scalar.more = icmp ult i32 %n113.scalar.p, %n113.scalar.count
br i1 %n113.scalar.more, label %n113.scalar.body, label %n113.scalar.done
n113.scalar.body:
%n113.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n113.source, i32 %n113.scalar.p
%n113.scalar.first = load float, ptr addrspace(1) %n113.scalar.first.ptr, align 4
%n113.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n113.second, i32 %n113.scalar.p
%n113.scalar.second = load float, ptr addrspace(1) %n113.scalar.second.ptr, align 4
%n113.scalar.scalar.0 = call float @recipe.add(float %n113.scalar.first, float %n113.scalar.second)
%n113.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n113.value, i32 %n113.scalar.p
store float %n113.scalar.scalar.0, ptr addrspace(1) %n113.scalar.output.ptr, align 4
br label %n113.scalar.step
n113.scalar.step:
%n113.scalar.next = add i32 %n113.scalar.p, %threads
br label %n113.scalar.loop
n113.scalar.done:
call void @grid_barrier(i32 %threads)%n114.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n114.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n114.value = getelementptr i8, ptr addrspace(1) %values, i64 7296
%n114.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1456
%n114.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n114.scalar.entry
n114.scalar.entry:
%n114.scalar.count = mul i32 %rows, 16
br label %n114.scalar.loop
n114.scalar.loop:
%n114.scalar.p = phi i32 [ %tid, %n114.scalar.entry ], [ %n114.scalar.next, %n114.scalar.step ]
%n114.scalar.more = icmp ult i32 %n114.scalar.p, %n114.scalar.count
br i1 %n114.scalar.more, label %n114.scalar.body, label %n114.scalar.done
n114.scalar.body:
%n114.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n114.source, i32 %n114.scalar.p
%n114.scalar.first = load float, ptr addrspace(1) %n114.scalar.first.ptr, align 4
%n114.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n114.second, i32 %n114.scalar.p
%n114.scalar.second = load float, ptr addrspace(1) %n114.scalar.second.ptr, align 4
%n114.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n114.scalar.first, float %n114.scalar.second)
%n114.scalar.scalar.0 = call float @recipe.from.u1(i1 %n114.scalar.scalar.0.condition)
%n114.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n114.value, i32 %n114.scalar.p
store float %n114.scalar.scalar.0, ptr addrspace(1) %n114.scalar.output.ptr, align 4
br label %n114.scalar.step
n114.scalar.step:
%n114.scalar.next = add i32 %n114.scalar.p, %threads
br label %n114.scalar.loop
n114.scalar.done:
call void @grid_barrier(i32 %threads)%n115.source = getelementptr i8, ptr addrspace(1) %values, i64 7232
%n115.second = getelementptr i8, ptr addrspace(1) %values, i64 7296
%n115.value = getelementptr i8, ptr addrspace(1) %values, i64 7360
%n115.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1464
%n115.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n115.scalar.entry
n115.scalar.entry:
%n115.scalar.count = mul i32 %rows, 16
br label %n115.scalar.loop
n115.scalar.loop:
%n115.scalar.p = phi i32 [ %tid, %n115.scalar.entry ], [ %n115.scalar.next, %n115.scalar.step ]
%n115.scalar.more = icmp ult i32 %n115.scalar.p, %n115.scalar.count
br i1 %n115.scalar.more, label %n115.scalar.body, label %n115.scalar.done
n115.scalar.body:
%n115.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n115.source, i32 %n115.scalar.p
%n115.scalar.first = load float, ptr addrspace(1) %n115.scalar.first.ptr, align 4
%n115.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n115.second, i32 %n115.scalar.p
%n115.scalar.second = load float, ptr addrspace(1) %n115.scalar.second.ptr, align 4
%n115.scalar.scalar.0 = call float @recipe.add(float %n115.scalar.first, float %n115.scalar.second)
%n115.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n115.value, i32 %n115.scalar.p
store float %n115.scalar.scalar.0, ptr addrspace(1) %n115.scalar.output.ptr, align 4
br label %n115.scalar.step
n115.scalar.step:
%n115.scalar.next = add i32 %n115.scalar.p, %threads
br label %n115.scalar.loop
n115.scalar.done:
call void @grid_barrier(i32 %threads)%n116.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n116.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n116.value = getelementptr i8, ptr addrspace(1) %values, i64 7424
%n116.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1472
%n116.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n116.scalar.entry
n116.scalar.entry:
%n116.scalar.count = mul i32 %rows, 16
br label %n116.scalar.loop
n116.scalar.loop:
%n116.scalar.p = phi i32 [ %tid, %n116.scalar.entry ], [ %n116.scalar.next, %n116.scalar.step ]
%n116.scalar.more = icmp ult i32 %n116.scalar.p, %n116.scalar.count
br i1 %n116.scalar.more, label %n116.scalar.body, label %n116.scalar.done
n116.scalar.body:
%n116.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n116.source, i32 %n116.scalar.p
%n116.scalar.first = load float, ptr addrspace(1) %n116.scalar.first.ptr, align 4
%n116.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n116.second, i32 %n116.scalar.p
%n116.scalar.second = load float, ptr addrspace(1) %n116.scalar.second.ptr, align 4
%n116.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n116.scalar.first, float %n116.scalar.second)
%n116.scalar.scalar.0 = call float @recipe.from.u1(i1 %n116.scalar.scalar.0.condition)
%n116.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n116.value, i32 %n116.scalar.p
store float %n116.scalar.scalar.0, ptr addrspace(1) %n116.scalar.output.ptr, align 4
br label %n116.scalar.step
n116.scalar.step:
%n116.scalar.next = add i32 %n116.scalar.p, %threads
br label %n116.scalar.loop
n116.scalar.done:
call void @grid_barrier(i32 %threads)%n117.source = getelementptr i8, ptr addrspace(1) %values, i64 7360
%n117.second = getelementptr i8, ptr addrspace(1) %values, i64 7424
%n117.value = getelementptr i8, ptr addrspace(1) %values, i64 7488
%n117.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1480
%n117.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n117.scalar.entry
n117.scalar.entry:
%n117.scalar.count = mul i32 %rows, 16
br label %n117.scalar.loop
n117.scalar.loop:
%n117.scalar.p = phi i32 [ %tid, %n117.scalar.entry ], [ %n117.scalar.next, %n117.scalar.step ]
%n117.scalar.more = icmp ult i32 %n117.scalar.p, %n117.scalar.count
br i1 %n117.scalar.more, label %n117.scalar.body, label %n117.scalar.done
n117.scalar.body:
%n117.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n117.source, i32 %n117.scalar.p
%n117.scalar.first = load float, ptr addrspace(1) %n117.scalar.first.ptr, align 4
%n117.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n117.second, i32 %n117.scalar.p
%n117.scalar.second = load float, ptr addrspace(1) %n117.scalar.second.ptr, align 4
%n117.scalar.scalar.0 = call float @recipe.add(float %n117.scalar.first, float %n117.scalar.second)
%n117.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n117.value, i32 %n117.scalar.p
store float %n117.scalar.scalar.0, ptr addrspace(1) %n117.scalar.output.ptr, align 4
br label %n117.scalar.step
n117.scalar.step:
%n117.scalar.next = add i32 %n117.scalar.p, %threads
br label %n117.scalar.loop
n117.scalar.done:
call void @grid_barrier(i32 %threads)%n118.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n118.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n118.value = getelementptr i8, ptr addrspace(1) %values, i64 7552
%n118.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1488
%n118.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n118.scalar.entry
n118.scalar.entry:
%n118.scalar.count = mul i32 %rows, 16
br label %n118.scalar.loop
n118.scalar.loop:
%n118.scalar.p = phi i32 [ %tid, %n118.scalar.entry ], [ %n118.scalar.next, %n118.scalar.step ]
%n118.scalar.more = icmp ult i32 %n118.scalar.p, %n118.scalar.count
br i1 %n118.scalar.more, label %n118.scalar.body, label %n118.scalar.done
n118.scalar.body:
%n118.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n118.source, i32 %n118.scalar.p
%n118.scalar.first = load float, ptr addrspace(1) %n118.scalar.first.ptr, align 4
%n118.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n118.second, i32 %n118.scalar.p
%n118.scalar.second = load float, ptr addrspace(1) %n118.scalar.second.ptr, align 4
%n118.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n118.scalar.first, float %n118.scalar.second)
%n118.scalar.scalar.0 = call float @recipe.from.u1(i1 %n118.scalar.scalar.0.condition)
%n118.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n118.value, i32 %n118.scalar.p
store float %n118.scalar.scalar.0, ptr addrspace(1) %n118.scalar.output.ptr, align 4
br label %n118.scalar.step
n118.scalar.step:
%n118.scalar.next = add i32 %n118.scalar.p, %threads
br label %n118.scalar.loop
n118.scalar.done:
call void @grid_barrier(i32 %threads)%n119.source = getelementptr i8, ptr addrspace(1) %values, i64 7488
%n119.second = getelementptr i8, ptr addrspace(1) %values, i64 7552
%n119.value = getelementptr i8, ptr addrspace(1) %values, i64 7616
%n119.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1496
%n119.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n119.scalar.entry
n119.scalar.entry:
%n119.scalar.count = mul i32 %rows, 16
br label %n119.scalar.loop
n119.scalar.loop:
%n119.scalar.p = phi i32 [ %tid, %n119.scalar.entry ], [ %n119.scalar.next, %n119.scalar.step ]
%n119.scalar.more = icmp ult i32 %n119.scalar.p, %n119.scalar.count
br i1 %n119.scalar.more, label %n119.scalar.body, label %n119.scalar.done
n119.scalar.body:
%n119.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n119.source, i32 %n119.scalar.p
%n119.scalar.first = load float, ptr addrspace(1) %n119.scalar.first.ptr, align 4
%n119.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n119.second, i32 %n119.scalar.p
%n119.scalar.second = load float, ptr addrspace(1) %n119.scalar.second.ptr, align 4
%n119.scalar.scalar.0 = call float @recipe.add(float %n119.scalar.first, float %n119.scalar.second)
%n119.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n119.value, i32 %n119.scalar.p
store float %n119.scalar.scalar.0, ptr addrspace(1) %n119.scalar.output.ptr, align 4
br label %n119.scalar.step
n119.scalar.step:
%n119.scalar.next = add i32 %n119.scalar.p, %threads
br label %n119.scalar.loop
n119.scalar.done:
call void @grid_barrier(i32 %threads)%n120.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n120.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n120.value = getelementptr i8, ptr addrspace(1) %values, i64 7680
%n120.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1504
%n120.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n120.scalar.entry
n120.scalar.entry:
%n120.scalar.count = mul i32 %rows, 16
br label %n120.scalar.loop
n120.scalar.loop:
%n120.scalar.p = phi i32 [ %tid, %n120.scalar.entry ], [ %n120.scalar.next, %n120.scalar.step ]
%n120.scalar.more = icmp ult i32 %n120.scalar.p, %n120.scalar.count
br i1 %n120.scalar.more, label %n120.scalar.body, label %n120.scalar.done
n120.scalar.body:
%n120.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n120.source, i32 %n120.scalar.p
%n120.scalar.first = load float, ptr addrspace(1) %n120.scalar.first.ptr, align 4
%n120.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n120.second, i32 %n120.scalar.p
%n120.scalar.second = load float, ptr addrspace(1) %n120.scalar.second.ptr, align 4
%n120.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n120.scalar.first, float %n120.scalar.second)
%n120.scalar.scalar.0 = call float @recipe.from.u1(i1 %n120.scalar.scalar.0.condition)
%n120.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n120.value, i32 %n120.scalar.p
store float %n120.scalar.scalar.0, ptr addrspace(1) %n120.scalar.output.ptr, align 4
br label %n120.scalar.step
n120.scalar.step:
%n120.scalar.next = add i32 %n120.scalar.p, %threads
br label %n120.scalar.loop
n120.scalar.done:
call void @grid_barrier(i32 %threads)%n121.source = getelementptr i8, ptr addrspace(1) %values, i64 7616
%n121.second = getelementptr i8, ptr addrspace(1) %values, i64 7680
%n121.value = getelementptr i8, ptr addrspace(1) %values, i64 7744
%n121.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1512
%n121.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n121.scalar.entry
n121.scalar.entry:
%n121.scalar.count = mul i32 %rows, 16
br label %n121.scalar.loop
n121.scalar.loop:
%n121.scalar.p = phi i32 [ %tid, %n121.scalar.entry ], [ %n121.scalar.next, %n121.scalar.step ]
%n121.scalar.more = icmp ult i32 %n121.scalar.p, %n121.scalar.count
br i1 %n121.scalar.more, label %n121.scalar.body, label %n121.scalar.done
n121.scalar.body:
%n121.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n121.source, i32 %n121.scalar.p
%n121.scalar.first = load float, ptr addrspace(1) %n121.scalar.first.ptr, align 4
%n121.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n121.second, i32 %n121.scalar.p
%n121.scalar.second = load float, ptr addrspace(1) %n121.scalar.second.ptr, align 4
%n121.scalar.scalar.0 = call float @recipe.add(float %n121.scalar.first, float %n121.scalar.second)
%n121.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n121.value, i32 %n121.scalar.p
store float %n121.scalar.scalar.0, ptr addrspace(1) %n121.scalar.output.ptr, align 4
br label %n121.scalar.step
n121.scalar.step:
%n121.scalar.next = add i32 %n121.scalar.p, %threads
br label %n121.scalar.loop
n121.scalar.done:
call void @grid_barrier(i32 %threads)%n122.source = getelementptr i8, ptr addrspace(1) %values, i64 7744
%n122.value = getelementptr i8, ptr addrspace(1) %values, i64 7808
%n122.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1520
%n122.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n122.scalar.entry
n122.scalar.entry:
%n122.scalar.count = mul i32 %rows, 16
br label %n122.scalar.loop
n122.scalar.loop:
%n122.scalar.p = phi i32 [ %tid, %n122.scalar.entry ], [ %n122.scalar.next, %n122.scalar.step ]
%n122.scalar.more = icmp ult i32 %n122.scalar.p, %n122.scalar.count
br i1 %n122.scalar.more, label %n122.scalar.body, label %n122.scalar.done
n122.scalar.body:
%n122.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n122.source, i32 %n122.scalar.p
%n122.scalar.first = load float, ptr addrspace(1) %n122.scalar.first.ptr, align 4
%n122.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n122.scalar.first)
%n122.scalar.scalar.1 = call float @recipe.from.u1(i1 %n122.scalar.scalar.1.condition)
%n122.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n122.value, i32 %n122.scalar.p
store float %n122.scalar.scalar.1, ptr addrspace(1) %n122.scalar.output.ptr, align 4
br label %n122.scalar.step
n122.scalar.step:
%n122.scalar.next = add i32 %n122.scalar.p, %threads
br label %n122.scalar.loop
n122.scalar.done:
call void @grid_barrier(i32 %threads)%n123.source = getelementptr i8, ptr addrspace(1) %values, i64 7808
%n123.second = getelementptr i8, ptr addrspace(1) %values, i64 6016
%n123.value = getelementptr i8, ptr addrspace(1) %values, i64 7872
%n123.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1528
%n123.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n123.scalar.entry
n123.scalar.entry:
%n123.scalar.count = mul i32 %rows, 16
br label %n123.scalar.loop
n123.scalar.loop:
%n123.scalar.p = phi i32 [ %tid, %n123.scalar.entry ], [ %n123.scalar.next, %n123.scalar.step ]
%n123.scalar.more = icmp ult i32 %n123.scalar.p, %n123.scalar.count
br i1 %n123.scalar.more, label %n123.scalar.body, label %n123.scalar.done
n123.scalar.body:
%n123.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n123.source, i32 %n123.scalar.p
%n123.scalar.first = load float, ptr addrspace(1) %n123.scalar.first.ptr, align 4
%n123.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n123.second, i32 %n123.scalar.p
%n123.scalar.second = load float, ptr addrspace(1) %n123.scalar.second.ptr, align 4
%n123.scalar.scalar.0 = call float @recipe.mul(float %n123.scalar.first, float %n123.scalar.second)
%n123.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n123.value, i32 %n123.scalar.p
store float %n123.scalar.scalar.0, ptr addrspace(1) %n123.scalar.output.ptr, align 4
br label %n123.scalar.step
n123.scalar.step:
%n123.scalar.next = add i32 %n123.scalar.p, %threads
br label %n123.scalar.loop
n123.scalar.done:
call void @grid_barrier(i32 %threads)%n124.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n124.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n124.value = getelementptr i8, ptr addrspace(1) %values, i64 7936
%n124.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1536
%n124.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n124.scalar.entry
n124.scalar.entry:
%n124.scalar.count = mul i32 %rows, 16
br label %n124.scalar.loop
n124.scalar.loop:
%n124.scalar.p = phi i32 [ %tid, %n124.scalar.entry ], [ %n124.scalar.next, %n124.scalar.step ]
%n124.scalar.more = icmp ult i32 %n124.scalar.p, %n124.scalar.count
br i1 %n124.scalar.more, label %n124.scalar.body, label %n124.scalar.done
n124.scalar.body:
%n124.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n124.source, i32 %n124.scalar.p
%n124.scalar.first = load float, ptr addrspace(1) %n124.scalar.first.ptr, align 4
%n124.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n124.second, i32 %n124.scalar.p
%n124.scalar.second = load float, ptr addrspace(1) %n124.scalar.second.ptr, align 4
%n124.scalar.scalar.0 = call float @recipe.sub(float %n124.scalar.first, float %n124.scalar.second)
%n124.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n124.value, i32 %n124.scalar.p
store float %n124.scalar.scalar.0, ptr addrspace(1) %n124.scalar.output.ptr, align 4
br label %n124.scalar.step
n124.scalar.step:
%n124.scalar.next = add i32 %n124.scalar.p, %threads
br label %n124.scalar.loop
n124.scalar.done:
call void @grid_barrier(i32 %threads)%n125.source = getelementptr i8, ptr addrspace(1) %values, i64 7936
%n125.value = getelementptr i8, ptr addrspace(1) %values, i64 8000
%n125.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1544
%n125.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n125.scalar.entry
n125.scalar.entry:
%n125.scalar.count = mul i32 %rows, 16
br label %n125.scalar.loop
n125.scalar.loop:
%n125.scalar.p = phi i32 [ %tid, %n125.scalar.entry ], [ %n125.scalar.next, %n125.scalar.step ]
%n125.scalar.more = icmp ult i32 %n125.scalar.p, %n125.scalar.count
br i1 %n125.scalar.more, label %n125.scalar.body, label %n125.scalar.done
n125.scalar.body:
%n125.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n125.source, i32 %n125.scalar.p
%n125.scalar.first = load float, ptr addrspace(1) %n125.scalar.first.ptr, align 4
%n125.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n125.scalar.first, float 0x0000000000000000)
%n125.scalar.scalar.2 = call float @recipe.from.u1(i1 %n125.scalar.scalar.2.condition)
%n125.scalar.scalar.3 = call float @recipe.exp(float %n125.scalar.first)
%n125.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n125.value, i32 %n125.scalar.p
store float %n125.scalar.scalar.3, ptr addrspace(1) %n125.scalar.output.ptr, align 4
br label %n125.scalar.step
n125.scalar.step:
%n125.scalar.next = add i32 %n125.scalar.p, %threads
br label %n125.scalar.loop
n125.scalar.done:
call void @grid_barrier(i32 %threads)%n126.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n126.value = getelementptr i8, ptr addrspace(1) %values, i64 8064
%n126.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1552
%n126.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n126.scalar.entry
n126.scalar.entry:
%n126.scalar.count = mul i32 %rows, 16
br label %n126.scalar.loop
n126.scalar.loop:
%n126.scalar.p = phi i32 [ %tid, %n126.scalar.entry ], [ %n126.scalar.next, %n126.scalar.step ]
%n126.scalar.more = icmp ult i32 %n126.scalar.p, %n126.scalar.count
br i1 %n126.scalar.more, label %n126.scalar.body, label %n126.scalar.done
n126.scalar.body:
%n126.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n126.source, i32 %n126.scalar.p
%n126.scalar.first = load float, ptr addrspace(1) %n126.scalar.first.ptr, align 4
%n126.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n126.value, i32 %n126.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n126.scalar.output.ptr, align 4
br label %n126.scalar.step
n126.scalar.step:
%n126.scalar.next = add i32 %n126.scalar.p, %threads
br label %n126.scalar.loop
n126.scalar.done:
call void @grid_barrier(i32 %threads)%n127.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n127.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n127.value = getelementptr i8, ptr addrspace(1) %values, i64 8128
%n127.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1560
%n127.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n127.scalar.entry
n127.scalar.entry:
%n127.scalar.count = mul i32 %rows, 16
br label %n127.scalar.loop
n127.scalar.loop:
%n127.scalar.p = phi i32 [ %tid, %n127.scalar.entry ], [ %n127.scalar.next, %n127.scalar.step ]
%n127.scalar.more = icmp ult i32 %n127.scalar.p, %n127.scalar.count
br i1 %n127.scalar.more, label %n127.scalar.body, label %n127.scalar.done
n127.scalar.body:
%n127.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n127.source, i32 %n127.scalar.p
%n127.scalar.first = load float, ptr addrspace(1) %n127.scalar.first.ptr, align 4
%n127.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n127.second, i32 %n127.scalar.p
%n127.scalar.second = load float, ptr addrspace(1) %n127.scalar.second.ptr, align 4
%n127.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n127.scalar.first, float %n127.scalar.second)
%n127.scalar.scalar.0 = call float @recipe.from.u1(i1 %n127.scalar.scalar.0.condition)
%n127.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n127.value, i32 %n127.scalar.p
store float %n127.scalar.scalar.0, ptr addrspace(1) %n127.scalar.output.ptr, align 4
br label %n127.scalar.step
n127.scalar.step:
%n127.scalar.next = add i32 %n127.scalar.p, %threads
br label %n127.scalar.loop
n127.scalar.done:
call void @grid_barrier(i32 %threads)%n128.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n128.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n128.value = getelementptr i8, ptr addrspace(1) %values, i64 8192
%n128.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1568
%n128.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n128.scalar.entry
n128.scalar.entry:
%n128.scalar.count = mul i32 %rows, 16
br label %n128.scalar.loop
n128.scalar.loop:
%n128.scalar.p = phi i32 [ %tid, %n128.scalar.entry ], [ %n128.scalar.next, %n128.scalar.step ]
%n128.scalar.more = icmp ult i32 %n128.scalar.p, %n128.scalar.count
br i1 %n128.scalar.more, label %n128.scalar.body, label %n128.scalar.done
n128.scalar.body:
%n128.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n128.source, i32 %n128.scalar.p
%n128.scalar.first = load float, ptr addrspace(1) %n128.scalar.first.ptr, align 4
%n128.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n128.second, i32 %n128.scalar.p
%n128.scalar.second = load float, ptr addrspace(1) %n128.scalar.second.ptr, align 4
%n128.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n128.scalar.first, float %n128.scalar.second)
%n128.scalar.scalar.0 = call float @recipe.from.u1(i1 %n128.scalar.scalar.0.condition)
%n128.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n128.value, i32 %n128.scalar.p
store float %n128.scalar.scalar.0, ptr addrspace(1) %n128.scalar.output.ptr, align 4
br label %n128.scalar.step
n128.scalar.step:
%n128.scalar.next = add i32 %n128.scalar.p, %threads
br label %n128.scalar.loop
n128.scalar.done:
call void @grid_barrier(i32 %threads)%n129.source = getelementptr i8, ptr addrspace(1) %values, i64 8128
%n129.second = getelementptr i8, ptr addrspace(1) %values, i64 8192
%n129.value = getelementptr i8, ptr addrspace(1) %values, i64 8256
%n129.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1576
%n129.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n129.scalar.entry
n129.scalar.entry:
%n129.scalar.count = mul i32 %rows, 16
br label %n129.scalar.loop
n129.scalar.loop:
%n129.scalar.p = phi i32 [ %tid, %n129.scalar.entry ], [ %n129.scalar.next, %n129.scalar.step ]
%n129.scalar.more = icmp ult i32 %n129.scalar.p, %n129.scalar.count
br i1 %n129.scalar.more, label %n129.scalar.body, label %n129.scalar.done
n129.scalar.body:
%n129.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n129.source, i32 %n129.scalar.p
%n129.scalar.first = load float, ptr addrspace(1) %n129.scalar.first.ptr, align 4
%n129.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n129.second, i32 %n129.scalar.p
%n129.scalar.second = load float, ptr addrspace(1) %n129.scalar.second.ptr, align 4
%n129.scalar.scalar.0 = call float @recipe.add(float %n129.scalar.first, float %n129.scalar.second)
%n129.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n129.value, i32 %n129.scalar.p
store float %n129.scalar.scalar.0, ptr addrspace(1) %n129.scalar.output.ptr, align 4
br label %n129.scalar.step
n129.scalar.step:
%n129.scalar.next = add i32 %n129.scalar.p, %threads
br label %n129.scalar.loop
n129.scalar.done:
call void @grid_barrier(i32 %threads)%n130.source = getelementptr i8, ptr addrspace(1) %values, i64 8256
%n130.value = getelementptr i8, ptr addrspace(1) %values, i64 8320
%n130.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1584
%n130.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n130.scalar.entry
n130.scalar.entry:
%n130.scalar.count = mul i32 %rows, 16
br label %n130.scalar.loop
n130.scalar.loop:
%n130.scalar.p = phi i32 [ %tid, %n130.scalar.entry ], [ %n130.scalar.next, %n130.scalar.step ]
%n130.scalar.more = icmp ult i32 %n130.scalar.p, %n130.scalar.count
br i1 %n130.scalar.more, label %n130.scalar.body, label %n130.scalar.done
n130.scalar.body:
%n130.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n130.source, i32 %n130.scalar.p
%n130.scalar.first = load float, ptr addrspace(1) %n130.scalar.first.ptr, align 4
%n130.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n130.scalar.first)
%n130.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n130.value, i32 %n130.scalar.p
store float %n130.scalar.scalar.1, ptr addrspace(1) %n130.scalar.output.ptr, align 4
br label %n130.scalar.step
n130.scalar.step:
%n130.scalar.next = add i32 %n130.scalar.p, %threads
br label %n130.scalar.loop
n130.scalar.done:
call void @grid_barrier(i32 %threads)%n131.source = getelementptr i8, ptr addrspace(1) %values, i64 8128
%n131.second = getelementptr i8, ptr addrspace(1) %values, i64 8320
%n131.value = getelementptr i8, ptr addrspace(1) %values, i64 8384
%n131.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1592
%n131.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n131.scalar.entry
n131.scalar.entry:
%n131.scalar.count = mul i32 %rows, 16
br label %n131.scalar.loop
n131.scalar.loop:
%n131.scalar.p = phi i32 [ %tid, %n131.scalar.entry ], [ %n131.scalar.next, %n131.scalar.step ]
%n131.scalar.more = icmp ult i32 %n131.scalar.p, %n131.scalar.count
br i1 %n131.scalar.more, label %n131.scalar.body, label %n131.scalar.done
n131.scalar.body:
%n131.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n131.source, i32 %n131.scalar.p
%n131.scalar.first = load float, ptr addrspace(1) %n131.scalar.first.ptr, align 4
%n131.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n131.second, i32 %n131.scalar.p
%n131.scalar.second = load float, ptr addrspace(1) %n131.scalar.second.ptr, align 4
%n131.scalar.scalar.0 = call float @recipe.add(float %n131.scalar.first, float %n131.scalar.second)
%n131.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n131.value, i32 %n131.scalar.p
store float %n131.scalar.scalar.0, ptr addrspace(1) %n131.scalar.output.ptr, align 4
br label %n131.scalar.step
n131.scalar.step:
%n131.scalar.next = add i32 %n131.scalar.p, %threads
br label %n131.scalar.loop
n131.scalar.done:
call void @grid_barrier(i32 %threads)%n132.source = getelementptr i8, ptr addrspace(1) %values, i64 8064
%n132.second = getelementptr i8, ptr addrspace(1) %values, i64 8384
%n132.value = getelementptr i8, ptr addrspace(1) %values, i64 8448
%n132.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1600
%n132.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n132.scalar.entry
n132.scalar.entry:
%n132.scalar.count = mul i32 %rows, 16
br label %n132.scalar.loop
n132.scalar.loop:
%n132.scalar.p = phi i32 [ %tid, %n132.scalar.entry ], [ %n132.scalar.next, %n132.scalar.step ]
%n132.scalar.more = icmp ult i32 %n132.scalar.p, %n132.scalar.count
br i1 %n132.scalar.more, label %n132.scalar.body, label %n132.scalar.done
n132.scalar.body:
%n132.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n132.source, i32 %n132.scalar.p
%n132.scalar.first = load float, ptr addrspace(1) %n132.scalar.first.ptr, align 4
%n132.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n132.second, i32 %n132.scalar.p
%n132.scalar.second = load float, ptr addrspace(1) %n132.scalar.second.ptr, align 4
%n132.scalar.scalar.0 = call float @recipe.add(float %n132.scalar.first, float %n132.scalar.second)
%n132.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n132.value, i32 %n132.scalar.p
store float %n132.scalar.scalar.0, ptr addrspace(1) %n132.scalar.output.ptr, align 4
br label %n132.scalar.step
n132.scalar.step:
%n132.scalar.next = add i32 %n132.scalar.p, %threads
br label %n132.scalar.loop
n132.scalar.done:
call void @grid_barrier(i32 %threads)%n133.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n133.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n133.value = getelementptr i8, ptr addrspace(1) %values, i64 8512
%n133.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1608
%n133.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n133.scalar.entry
n133.scalar.entry:
%n133.scalar.count = mul i32 %rows, 16
br label %n133.scalar.loop
n133.scalar.loop:
%n133.scalar.p = phi i32 [ %tid, %n133.scalar.entry ], [ %n133.scalar.next, %n133.scalar.step ]
%n133.scalar.more = icmp ult i32 %n133.scalar.p, %n133.scalar.count
br i1 %n133.scalar.more, label %n133.scalar.body, label %n133.scalar.done
n133.scalar.body:
%n133.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n133.source, i32 %n133.scalar.p
%n133.scalar.first = load float, ptr addrspace(1) %n133.scalar.first.ptr, align 4
%n133.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n133.second, i32 %n133.scalar.p
%n133.scalar.second = load float, ptr addrspace(1) %n133.scalar.second.ptr, align 4
%n133.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n133.scalar.first, float %n133.scalar.second)
%n133.scalar.scalar.0 = call float @recipe.from.u1(i1 %n133.scalar.scalar.0.condition)
%n133.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n133.value, i32 %n133.scalar.p
store float %n133.scalar.scalar.0, ptr addrspace(1) %n133.scalar.output.ptr, align 4
br label %n133.scalar.step
n133.scalar.step:
%n133.scalar.next = add i32 %n133.scalar.p, %threads
br label %n133.scalar.loop
n133.scalar.done:
call void @grid_barrier(i32 %threads)%n134.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n134.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n134.value = getelementptr i8, ptr addrspace(1) %values, i64 8576
%n134.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1616
%n134.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n134.scalar.entry
n134.scalar.entry:
%n134.scalar.count = mul i32 %rows, 16
br label %n134.scalar.loop
n134.scalar.loop:
%n134.scalar.p = phi i32 [ %tid, %n134.scalar.entry ], [ %n134.scalar.next, %n134.scalar.step ]
%n134.scalar.more = icmp ult i32 %n134.scalar.p, %n134.scalar.count
br i1 %n134.scalar.more, label %n134.scalar.body, label %n134.scalar.done
n134.scalar.body:
%n134.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n134.source, i32 %n134.scalar.p
%n134.scalar.first = load float, ptr addrspace(1) %n134.scalar.first.ptr, align 4
%n134.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n134.second, i32 %n134.scalar.p
%n134.scalar.second = load float, ptr addrspace(1) %n134.scalar.second.ptr, align 4
%n134.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n134.scalar.first, float %n134.scalar.second)
%n134.scalar.scalar.0 = call float @recipe.from.u1(i1 %n134.scalar.scalar.0.condition)
%n134.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n134.value, i32 %n134.scalar.p
store float %n134.scalar.scalar.0, ptr addrspace(1) %n134.scalar.output.ptr, align 4
br label %n134.scalar.step
n134.scalar.step:
%n134.scalar.next = add i32 %n134.scalar.p, %threads
br label %n134.scalar.loop
n134.scalar.done:
call void @grid_barrier(i32 %threads)%n135.source = getelementptr i8, ptr addrspace(1) %values, i64 8512
%n135.second = getelementptr i8, ptr addrspace(1) %values, i64 8576
%n135.value = getelementptr i8, ptr addrspace(1) %values, i64 8640
%n135.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1624
%n135.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n135.scalar.entry
n135.scalar.entry:
%n135.scalar.count = mul i32 %rows, 16
br label %n135.scalar.loop
n135.scalar.loop:
%n135.scalar.p = phi i32 [ %tid, %n135.scalar.entry ], [ %n135.scalar.next, %n135.scalar.step ]
%n135.scalar.more = icmp ult i32 %n135.scalar.p, %n135.scalar.count
br i1 %n135.scalar.more, label %n135.scalar.body, label %n135.scalar.done
n135.scalar.body:
%n135.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n135.source, i32 %n135.scalar.p
%n135.scalar.first = load float, ptr addrspace(1) %n135.scalar.first.ptr, align 4
%n135.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n135.second, i32 %n135.scalar.p
%n135.scalar.second = load float, ptr addrspace(1) %n135.scalar.second.ptr, align 4
%n135.scalar.scalar.0 = call float @recipe.add(float %n135.scalar.first, float %n135.scalar.second)
%n135.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n135.value, i32 %n135.scalar.p
store float %n135.scalar.scalar.0, ptr addrspace(1) %n135.scalar.output.ptr, align 4
br label %n135.scalar.step
n135.scalar.step:
%n135.scalar.next = add i32 %n135.scalar.p, %threads
br label %n135.scalar.loop
n135.scalar.done:
call void @grid_barrier(i32 %threads)%n136.source = getelementptr i8, ptr addrspace(1) %values, i64 8640
%n136.value = getelementptr i8, ptr addrspace(1) %values, i64 8704
%n136.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1632
%n136.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n136.scalar.entry
n136.scalar.entry:
%n136.scalar.count = mul i32 %rows, 16
br label %n136.scalar.loop
n136.scalar.loop:
%n136.scalar.p = phi i32 [ %tid, %n136.scalar.entry ], [ %n136.scalar.next, %n136.scalar.step ]
%n136.scalar.more = icmp ult i32 %n136.scalar.p, %n136.scalar.count
br i1 %n136.scalar.more, label %n136.scalar.body, label %n136.scalar.done
n136.scalar.body:
%n136.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n136.source, i32 %n136.scalar.p
%n136.scalar.first = load float, ptr addrspace(1) %n136.scalar.first.ptr, align 4
%n136.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n136.scalar.first)
%n136.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n136.value, i32 %n136.scalar.p
store float %n136.scalar.scalar.1, ptr addrspace(1) %n136.scalar.output.ptr, align 4
br label %n136.scalar.step
n136.scalar.step:
%n136.scalar.next = add i32 %n136.scalar.p, %threads
br label %n136.scalar.loop
n136.scalar.done:
call void @grid_barrier(i32 %threads)%n137.source = getelementptr i8, ptr addrspace(1) %values, i64 8512
%n137.second = getelementptr i8, ptr addrspace(1) %values, i64 8704
%n137.value = getelementptr i8, ptr addrspace(1) %values, i64 8768
%n137.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1640
%n137.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n137.scalar.entry
n137.scalar.entry:
%n137.scalar.count = mul i32 %rows, 16
br label %n137.scalar.loop
n137.scalar.loop:
%n137.scalar.p = phi i32 [ %tid, %n137.scalar.entry ], [ %n137.scalar.next, %n137.scalar.step ]
%n137.scalar.more = icmp ult i32 %n137.scalar.p, %n137.scalar.count
br i1 %n137.scalar.more, label %n137.scalar.body, label %n137.scalar.done
n137.scalar.body:
%n137.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n137.source, i32 %n137.scalar.p
%n137.scalar.first = load float, ptr addrspace(1) %n137.scalar.first.ptr, align 4
%n137.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n137.second, i32 %n137.scalar.p
%n137.scalar.second = load float, ptr addrspace(1) %n137.scalar.second.ptr, align 4
%n137.scalar.scalar.0 = call float @recipe.add(float %n137.scalar.first, float %n137.scalar.second)
%n137.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n137.value, i32 %n137.scalar.p
store float %n137.scalar.scalar.0, ptr addrspace(1) %n137.scalar.output.ptr, align 4
br label %n137.scalar.step
n137.scalar.step:
%n137.scalar.next = add i32 %n137.scalar.p, %threads
br label %n137.scalar.loop
n137.scalar.done:
call void @grid_barrier(i32 %threads)%n138.source = getelementptr i8, ptr addrspace(1) %values, i64 8448
%n138.second = getelementptr i8, ptr addrspace(1) %values, i64 8768
%n138.value = getelementptr i8, ptr addrspace(1) %values, i64 8832
%n138.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1648
%n138.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n138.scalar.entry
n138.scalar.entry:
%n138.scalar.count = mul i32 %rows, 16
br label %n138.scalar.loop
n138.scalar.loop:
%n138.scalar.p = phi i32 [ %tid, %n138.scalar.entry ], [ %n138.scalar.next, %n138.scalar.step ]
%n138.scalar.more = icmp ult i32 %n138.scalar.p, %n138.scalar.count
br i1 %n138.scalar.more, label %n138.scalar.body, label %n138.scalar.done
n138.scalar.body:
%n138.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n138.source, i32 %n138.scalar.p
%n138.scalar.first = load float, ptr addrspace(1) %n138.scalar.first.ptr, align 4
%n138.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n138.second, i32 %n138.scalar.p
%n138.scalar.second = load float, ptr addrspace(1) %n138.scalar.second.ptr, align 4
%n138.scalar.scalar.0 = call float @recipe.add(float %n138.scalar.first, float %n138.scalar.second)
%n138.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n138.value, i32 %n138.scalar.p
store float %n138.scalar.scalar.0, ptr addrspace(1) %n138.scalar.output.ptr, align 4
br label %n138.scalar.step
n138.scalar.step:
%n138.scalar.next = add i32 %n138.scalar.p, %threads
br label %n138.scalar.loop
n138.scalar.done:
call void @grid_barrier(i32 %threads)%n139.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n139.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n139.value = getelementptr i8, ptr addrspace(1) %values, i64 8896
%n139.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1656
%n139.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n139.scalar.entry
n139.scalar.entry:
%n139.scalar.count = mul i32 %rows, 16
br label %n139.scalar.loop
n139.scalar.loop:
%n139.scalar.p = phi i32 [ %tid, %n139.scalar.entry ], [ %n139.scalar.next, %n139.scalar.step ]
%n139.scalar.more = icmp ult i32 %n139.scalar.p, %n139.scalar.count
br i1 %n139.scalar.more, label %n139.scalar.body, label %n139.scalar.done
n139.scalar.body:
%n139.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n139.source, i32 %n139.scalar.p
%n139.scalar.first = load float, ptr addrspace(1) %n139.scalar.first.ptr, align 4
%n139.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n139.second, i32 %n139.scalar.p
%n139.scalar.second = load float, ptr addrspace(1) %n139.scalar.second.ptr, align 4
%n139.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n139.scalar.first, float %n139.scalar.second)
%n139.scalar.scalar.0 = call float @recipe.from.u1(i1 %n139.scalar.scalar.0.condition)
%n139.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n139.value, i32 %n139.scalar.p
store float %n139.scalar.scalar.0, ptr addrspace(1) %n139.scalar.output.ptr, align 4
br label %n139.scalar.step
n139.scalar.step:
%n139.scalar.next = add i32 %n139.scalar.p, %threads
br label %n139.scalar.loop
n139.scalar.done:
call void @grid_barrier(i32 %threads)%n140.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n140.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n140.value = getelementptr i8, ptr addrspace(1) %values, i64 8960
%n140.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1664
%n140.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n140.scalar.entry
n140.scalar.entry:
%n140.scalar.count = mul i32 %rows, 16
br label %n140.scalar.loop
n140.scalar.loop:
%n140.scalar.p = phi i32 [ %tid, %n140.scalar.entry ], [ %n140.scalar.next, %n140.scalar.step ]
%n140.scalar.more = icmp ult i32 %n140.scalar.p, %n140.scalar.count
br i1 %n140.scalar.more, label %n140.scalar.body, label %n140.scalar.done
n140.scalar.body:
%n140.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n140.source, i32 %n140.scalar.p
%n140.scalar.first = load float, ptr addrspace(1) %n140.scalar.first.ptr, align 4
%n140.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n140.second, i32 %n140.scalar.p
%n140.scalar.second = load float, ptr addrspace(1) %n140.scalar.second.ptr, align 4
%n140.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n140.scalar.first, float %n140.scalar.second)
%n140.scalar.scalar.0 = call float @recipe.from.u1(i1 %n140.scalar.scalar.0.condition)
%n140.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n140.value, i32 %n140.scalar.p
store float %n140.scalar.scalar.0, ptr addrspace(1) %n140.scalar.output.ptr, align 4
br label %n140.scalar.step
n140.scalar.step:
%n140.scalar.next = add i32 %n140.scalar.p, %threads
br label %n140.scalar.loop
n140.scalar.done:
call void @grid_barrier(i32 %threads)%n141.source = getelementptr i8, ptr addrspace(1) %values, i64 8896
%n141.second = getelementptr i8, ptr addrspace(1) %values, i64 8960
%n141.value = getelementptr i8, ptr addrspace(1) %values, i64 9024
%n141.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1672
%n141.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n141.scalar.entry
n141.scalar.entry:
%n141.scalar.count = mul i32 %rows, 16
br label %n141.scalar.loop
n141.scalar.loop:
%n141.scalar.p = phi i32 [ %tid, %n141.scalar.entry ], [ %n141.scalar.next, %n141.scalar.step ]
%n141.scalar.more = icmp ult i32 %n141.scalar.p, %n141.scalar.count
br i1 %n141.scalar.more, label %n141.scalar.body, label %n141.scalar.done
n141.scalar.body:
%n141.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n141.source, i32 %n141.scalar.p
%n141.scalar.first = load float, ptr addrspace(1) %n141.scalar.first.ptr, align 4
%n141.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n141.second, i32 %n141.scalar.p
%n141.scalar.second = load float, ptr addrspace(1) %n141.scalar.second.ptr, align 4
%n141.scalar.scalar.0 = call float @recipe.add(float %n141.scalar.first, float %n141.scalar.second)
%n141.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n141.value, i32 %n141.scalar.p
store float %n141.scalar.scalar.0, ptr addrspace(1) %n141.scalar.output.ptr, align 4
br label %n141.scalar.step
n141.scalar.step:
%n141.scalar.next = add i32 %n141.scalar.p, %threads
br label %n141.scalar.loop
n141.scalar.done:
call void @grid_barrier(i32 %threads)%n142.source = getelementptr i8, ptr addrspace(1) %values, i64 9024
%n142.value = getelementptr i8, ptr addrspace(1) %values, i64 9088
%n142.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1680
%n142.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n142.scalar.entry
n142.scalar.entry:
%n142.scalar.count = mul i32 %rows, 16
br label %n142.scalar.loop
n142.scalar.loop:
%n142.scalar.p = phi i32 [ %tid, %n142.scalar.entry ], [ %n142.scalar.next, %n142.scalar.step ]
%n142.scalar.more = icmp ult i32 %n142.scalar.p, %n142.scalar.count
br i1 %n142.scalar.more, label %n142.scalar.body, label %n142.scalar.done
n142.scalar.body:
%n142.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n142.source, i32 %n142.scalar.p
%n142.scalar.first = load float, ptr addrspace(1) %n142.scalar.first.ptr, align 4
%n142.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n142.scalar.first)
%n142.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n142.value, i32 %n142.scalar.p
store float %n142.scalar.scalar.1, ptr addrspace(1) %n142.scalar.output.ptr, align 4
br label %n142.scalar.step
n142.scalar.step:
%n142.scalar.next = add i32 %n142.scalar.p, %threads
br label %n142.scalar.loop
n142.scalar.done:
call void @grid_barrier(i32 %threads)%n143.source = getelementptr i8, ptr addrspace(1) %values, i64 8896
%n143.second = getelementptr i8, ptr addrspace(1) %values, i64 9088
%n143.value = getelementptr i8, ptr addrspace(1) %values, i64 9152
%n143.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1688
%n143.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n143.scalar.entry
n143.scalar.entry:
%n143.scalar.count = mul i32 %rows, 16
br label %n143.scalar.loop
n143.scalar.loop:
%n143.scalar.p = phi i32 [ %tid, %n143.scalar.entry ], [ %n143.scalar.next, %n143.scalar.step ]
%n143.scalar.more = icmp ult i32 %n143.scalar.p, %n143.scalar.count
br i1 %n143.scalar.more, label %n143.scalar.body, label %n143.scalar.done
n143.scalar.body:
%n143.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n143.source, i32 %n143.scalar.p
%n143.scalar.first = load float, ptr addrspace(1) %n143.scalar.first.ptr, align 4
%n143.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n143.second, i32 %n143.scalar.p
%n143.scalar.second = load float, ptr addrspace(1) %n143.scalar.second.ptr, align 4
%n143.scalar.scalar.0 = call float @recipe.add(float %n143.scalar.first, float %n143.scalar.second)
%n143.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n143.value, i32 %n143.scalar.p
store float %n143.scalar.scalar.0, ptr addrspace(1) %n143.scalar.output.ptr, align 4
br label %n143.scalar.step
n143.scalar.step:
%n143.scalar.next = add i32 %n143.scalar.p, %threads
br label %n143.scalar.loop
n143.scalar.done:
call void @grid_barrier(i32 %threads)%n144.source = getelementptr i8, ptr addrspace(1) %values, i64 8832
%n144.second = getelementptr i8, ptr addrspace(1) %values, i64 9152
%n144.value = getelementptr i8, ptr addrspace(1) %values, i64 9216
%n144.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1696
%n144.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n144.scalar.entry
n144.scalar.entry:
%n144.scalar.count = mul i32 %rows, 16
br label %n144.scalar.loop
n144.scalar.loop:
%n144.scalar.p = phi i32 [ %tid, %n144.scalar.entry ], [ %n144.scalar.next, %n144.scalar.step ]
%n144.scalar.more = icmp ult i32 %n144.scalar.p, %n144.scalar.count
br i1 %n144.scalar.more, label %n144.scalar.body, label %n144.scalar.done
n144.scalar.body:
%n144.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n144.source, i32 %n144.scalar.p
%n144.scalar.first = load float, ptr addrspace(1) %n144.scalar.first.ptr, align 4
%n144.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n144.second, i32 %n144.scalar.p
%n144.scalar.second = load float, ptr addrspace(1) %n144.scalar.second.ptr, align 4
%n144.scalar.scalar.0 = call float @recipe.add(float %n144.scalar.first, float %n144.scalar.second)
%n144.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n144.value, i32 %n144.scalar.p
store float %n144.scalar.scalar.0, ptr addrspace(1) %n144.scalar.output.ptr, align 4
br label %n144.scalar.step
n144.scalar.step:
%n144.scalar.next = add i32 %n144.scalar.p, %threads
br label %n144.scalar.loop
n144.scalar.done:
call void @grid_barrier(i32 %threads)%n145.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n145.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n145.value = getelementptr i8, ptr addrspace(1) %values, i64 9280
%n145.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1704
%n145.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n145.scalar.entry
n145.scalar.entry:
%n145.scalar.count = mul i32 %rows, 16
br label %n145.scalar.loop
n145.scalar.loop:
%n145.scalar.p = phi i32 [ %tid, %n145.scalar.entry ], [ %n145.scalar.next, %n145.scalar.step ]
%n145.scalar.more = icmp ult i32 %n145.scalar.p, %n145.scalar.count
br i1 %n145.scalar.more, label %n145.scalar.body, label %n145.scalar.done
n145.scalar.body:
%n145.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n145.source, i32 %n145.scalar.p
%n145.scalar.first = load float, ptr addrspace(1) %n145.scalar.first.ptr, align 4
%n145.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n145.second, i32 %n145.scalar.p
%n145.scalar.second = load float, ptr addrspace(1) %n145.scalar.second.ptr, align 4
%n145.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n145.scalar.first, float %n145.scalar.second)
%n145.scalar.scalar.0 = call float @recipe.from.u1(i1 %n145.scalar.scalar.0.condition)
%n145.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n145.value, i32 %n145.scalar.p
store float %n145.scalar.scalar.0, ptr addrspace(1) %n145.scalar.output.ptr, align 4
br label %n145.scalar.step
n145.scalar.step:
%n145.scalar.next = add i32 %n145.scalar.p, %threads
br label %n145.scalar.loop
n145.scalar.done:
call void @grid_barrier(i32 %threads)%n146.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n146.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n146.value = getelementptr i8, ptr addrspace(1) %values, i64 9344
%n146.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1712
%n146.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n146.scalar.entry
n146.scalar.entry:
%n146.scalar.count = mul i32 %rows, 16
br label %n146.scalar.loop
n146.scalar.loop:
%n146.scalar.p = phi i32 [ %tid, %n146.scalar.entry ], [ %n146.scalar.next, %n146.scalar.step ]
%n146.scalar.more = icmp ult i32 %n146.scalar.p, %n146.scalar.count
br i1 %n146.scalar.more, label %n146.scalar.body, label %n146.scalar.done
n146.scalar.body:
%n146.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n146.source, i32 %n146.scalar.p
%n146.scalar.first = load float, ptr addrspace(1) %n146.scalar.first.ptr, align 4
%n146.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n146.second, i32 %n146.scalar.p
%n146.scalar.second = load float, ptr addrspace(1) %n146.scalar.second.ptr, align 4
%n146.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n146.scalar.first, float %n146.scalar.second)
%n146.scalar.scalar.0 = call float @recipe.from.u1(i1 %n146.scalar.scalar.0.condition)
%n146.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n146.value, i32 %n146.scalar.p
store float %n146.scalar.scalar.0, ptr addrspace(1) %n146.scalar.output.ptr, align 4
br label %n146.scalar.step
n146.scalar.step:
%n146.scalar.next = add i32 %n146.scalar.p, %threads
br label %n146.scalar.loop
n146.scalar.done:
call void @grid_barrier(i32 %threads)%n147.source = getelementptr i8, ptr addrspace(1) %values, i64 9280
%n147.second = getelementptr i8, ptr addrspace(1) %values, i64 9344
%n147.value = getelementptr i8, ptr addrspace(1) %values, i64 9408
%n147.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1720
%n147.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n147.scalar.entry
n147.scalar.entry:
%n147.scalar.count = mul i32 %rows, 16
br label %n147.scalar.loop
n147.scalar.loop:
%n147.scalar.p = phi i32 [ %tid, %n147.scalar.entry ], [ %n147.scalar.next, %n147.scalar.step ]
%n147.scalar.more = icmp ult i32 %n147.scalar.p, %n147.scalar.count
br i1 %n147.scalar.more, label %n147.scalar.body, label %n147.scalar.done
n147.scalar.body:
%n147.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n147.source, i32 %n147.scalar.p
%n147.scalar.first = load float, ptr addrspace(1) %n147.scalar.first.ptr, align 4
%n147.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n147.second, i32 %n147.scalar.p
%n147.scalar.second = load float, ptr addrspace(1) %n147.scalar.second.ptr, align 4
%n147.scalar.scalar.0 = call float @recipe.add(float %n147.scalar.first, float %n147.scalar.second)
%n147.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n147.value, i32 %n147.scalar.p
store float %n147.scalar.scalar.0, ptr addrspace(1) %n147.scalar.output.ptr, align 4
br label %n147.scalar.step
n147.scalar.step:
%n147.scalar.next = add i32 %n147.scalar.p, %threads
br label %n147.scalar.loop
n147.scalar.done:
call void @grid_barrier(i32 %threads)%n148.source = getelementptr i8, ptr addrspace(1) %values, i64 9408
%n148.value = getelementptr i8, ptr addrspace(1) %values, i64 9472
%n148.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1728
%n148.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n148.scalar.entry
n148.scalar.entry:
%n148.scalar.count = mul i32 %rows, 16
br label %n148.scalar.loop
n148.scalar.loop:
%n148.scalar.p = phi i32 [ %tid, %n148.scalar.entry ], [ %n148.scalar.next, %n148.scalar.step ]
%n148.scalar.more = icmp ult i32 %n148.scalar.p, %n148.scalar.count
br i1 %n148.scalar.more, label %n148.scalar.body, label %n148.scalar.done
n148.scalar.body:
%n148.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n148.source, i32 %n148.scalar.p
%n148.scalar.first = load float, ptr addrspace(1) %n148.scalar.first.ptr, align 4
%n148.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n148.scalar.first)
%n148.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n148.value, i32 %n148.scalar.p
store float %n148.scalar.scalar.1, ptr addrspace(1) %n148.scalar.output.ptr, align 4
br label %n148.scalar.step
n148.scalar.step:
%n148.scalar.next = add i32 %n148.scalar.p, %threads
br label %n148.scalar.loop
n148.scalar.done:
call void @grid_barrier(i32 %threads)%n149.source = getelementptr i8, ptr addrspace(1) %values, i64 9280
%n149.second = getelementptr i8, ptr addrspace(1) %values, i64 9472
%n149.value = getelementptr i8, ptr addrspace(1) %values, i64 9536
%n149.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1736
%n149.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n149.scalar.entry
n149.scalar.entry:
%n149.scalar.count = mul i32 %rows, 16
br label %n149.scalar.loop
n149.scalar.loop:
%n149.scalar.p = phi i32 [ %tid, %n149.scalar.entry ], [ %n149.scalar.next, %n149.scalar.step ]
%n149.scalar.more = icmp ult i32 %n149.scalar.p, %n149.scalar.count
br i1 %n149.scalar.more, label %n149.scalar.body, label %n149.scalar.done
n149.scalar.body:
%n149.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n149.source, i32 %n149.scalar.p
%n149.scalar.first = load float, ptr addrspace(1) %n149.scalar.first.ptr, align 4
%n149.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n149.second, i32 %n149.scalar.p
%n149.scalar.second = load float, ptr addrspace(1) %n149.scalar.second.ptr, align 4
%n149.scalar.scalar.0 = call float @recipe.add(float %n149.scalar.first, float %n149.scalar.second)
%n149.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n149.value, i32 %n149.scalar.p
store float %n149.scalar.scalar.0, ptr addrspace(1) %n149.scalar.output.ptr, align 4
br label %n149.scalar.step
n149.scalar.step:
%n149.scalar.next = add i32 %n149.scalar.p, %threads
br label %n149.scalar.loop
n149.scalar.done:
call void @grid_barrier(i32 %threads)%n150.source = getelementptr i8, ptr addrspace(1) %values, i64 9216
%n150.second = getelementptr i8, ptr addrspace(1) %values, i64 9536
%n150.value = getelementptr i8, ptr addrspace(1) %values, i64 9600
%n150.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1744
%n150.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n150.scalar.entry
n150.scalar.entry:
%n150.scalar.count = mul i32 %rows, 16
br label %n150.scalar.loop
n150.scalar.loop:
%n150.scalar.p = phi i32 [ %tid, %n150.scalar.entry ], [ %n150.scalar.next, %n150.scalar.step ]
%n150.scalar.more = icmp ult i32 %n150.scalar.p, %n150.scalar.count
br i1 %n150.scalar.more, label %n150.scalar.body, label %n150.scalar.done
n150.scalar.body:
%n150.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n150.source, i32 %n150.scalar.p
%n150.scalar.first = load float, ptr addrspace(1) %n150.scalar.first.ptr, align 4
%n150.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n150.second, i32 %n150.scalar.p
%n150.scalar.second = load float, ptr addrspace(1) %n150.scalar.second.ptr, align 4
%n150.scalar.scalar.0 = call float @recipe.add(float %n150.scalar.first, float %n150.scalar.second)
%n150.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n150.value, i32 %n150.scalar.p
store float %n150.scalar.scalar.0, ptr addrspace(1) %n150.scalar.output.ptr, align 4
br label %n150.scalar.step
n150.scalar.step:
%n150.scalar.next = add i32 %n150.scalar.p, %threads
br label %n150.scalar.loop
n150.scalar.done:
call void @grid_barrier(i32 %threads)%n151.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n151.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n151.value = getelementptr i8, ptr addrspace(1) %values, i64 9664
%n151.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1752
%n151.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n151.scalar.entry
n151.scalar.entry:
%n151.scalar.count = mul i32 %rows, 16
br label %n151.scalar.loop
n151.scalar.loop:
%n151.scalar.p = phi i32 [ %tid, %n151.scalar.entry ], [ %n151.scalar.next, %n151.scalar.step ]
%n151.scalar.more = icmp ult i32 %n151.scalar.p, %n151.scalar.count
br i1 %n151.scalar.more, label %n151.scalar.body, label %n151.scalar.done
n151.scalar.body:
%n151.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n151.source, i32 %n151.scalar.p
%n151.scalar.first = load float, ptr addrspace(1) %n151.scalar.first.ptr, align 4
%n151.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n151.second, i32 %n151.scalar.p
%n151.scalar.second = load float, ptr addrspace(1) %n151.scalar.second.ptr, align 4
%n151.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n151.scalar.first, float %n151.scalar.second)
%n151.scalar.scalar.0 = call float @recipe.from.u1(i1 %n151.scalar.scalar.0.condition)
%n151.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n151.value, i32 %n151.scalar.p
store float %n151.scalar.scalar.0, ptr addrspace(1) %n151.scalar.output.ptr, align 4
br label %n151.scalar.step
n151.scalar.step:
%n151.scalar.next = add i32 %n151.scalar.p, %threads
br label %n151.scalar.loop
n151.scalar.done:
call void @grid_barrier(i32 %threads)%n152.source = getelementptr i8, ptr addrspace(1) %values, i64 9600
%n152.second = getelementptr i8, ptr addrspace(1) %values, i64 9664
%n152.value = getelementptr i8, ptr addrspace(1) %values, i64 9728
%n152.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1760
%n152.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n152.scalar.entry
n152.scalar.entry:
%n152.scalar.count = mul i32 %rows, 16
br label %n152.scalar.loop
n152.scalar.loop:
%n152.scalar.p = phi i32 [ %tid, %n152.scalar.entry ], [ %n152.scalar.next, %n152.scalar.step ]
%n152.scalar.more = icmp ult i32 %n152.scalar.p, %n152.scalar.count
br i1 %n152.scalar.more, label %n152.scalar.body, label %n152.scalar.done
n152.scalar.body:
%n152.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n152.source, i32 %n152.scalar.p
%n152.scalar.first = load float, ptr addrspace(1) %n152.scalar.first.ptr, align 4
%n152.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n152.second, i32 %n152.scalar.p
%n152.scalar.second = load float, ptr addrspace(1) %n152.scalar.second.ptr, align 4
%n152.scalar.scalar.0 = call float @recipe.add(float %n152.scalar.first, float %n152.scalar.second)
%n152.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n152.value, i32 %n152.scalar.p
store float %n152.scalar.scalar.0, ptr addrspace(1) %n152.scalar.output.ptr, align 4
br label %n152.scalar.step
n152.scalar.step:
%n152.scalar.next = add i32 %n152.scalar.p, %threads
br label %n152.scalar.loop
n152.scalar.done:
call void @grid_barrier(i32 %threads)%n153.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n153.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n153.value = getelementptr i8, ptr addrspace(1) %values, i64 9792
%n153.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1768
%n153.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n153.scalar.entry
n153.scalar.entry:
%n153.scalar.count = mul i32 %rows, 16
br label %n153.scalar.loop
n153.scalar.loop:
%n153.scalar.p = phi i32 [ %tid, %n153.scalar.entry ], [ %n153.scalar.next, %n153.scalar.step ]
%n153.scalar.more = icmp ult i32 %n153.scalar.p, %n153.scalar.count
br i1 %n153.scalar.more, label %n153.scalar.body, label %n153.scalar.done
n153.scalar.body:
%n153.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n153.source, i32 %n153.scalar.p
%n153.scalar.first = load float, ptr addrspace(1) %n153.scalar.first.ptr, align 4
%n153.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n153.second, i32 %n153.scalar.p
%n153.scalar.second = load float, ptr addrspace(1) %n153.scalar.second.ptr, align 4
%n153.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n153.scalar.first, float %n153.scalar.second)
%n153.scalar.scalar.0 = call float @recipe.from.u1(i1 %n153.scalar.scalar.0.condition)
%n153.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n153.value, i32 %n153.scalar.p
store float %n153.scalar.scalar.0, ptr addrspace(1) %n153.scalar.output.ptr, align 4
br label %n153.scalar.step
n153.scalar.step:
%n153.scalar.next = add i32 %n153.scalar.p, %threads
br label %n153.scalar.loop
n153.scalar.done:
call void @grid_barrier(i32 %threads)%n154.source = getelementptr i8, ptr addrspace(1) %values, i64 9728
%n154.second = getelementptr i8, ptr addrspace(1) %values, i64 9792
%n154.value = getelementptr i8, ptr addrspace(1) %values, i64 9856
%n154.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1776
%n154.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n154.scalar.entry
n154.scalar.entry:
%n154.scalar.count = mul i32 %rows, 16
br label %n154.scalar.loop
n154.scalar.loop:
%n154.scalar.p = phi i32 [ %tid, %n154.scalar.entry ], [ %n154.scalar.next, %n154.scalar.step ]
%n154.scalar.more = icmp ult i32 %n154.scalar.p, %n154.scalar.count
br i1 %n154.scalar.more, label %n154.scalar.body, label %n154.scalar.done
n154.scalar.body:
%n154.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n154.source, i32 %n154.scalar.p
%n154.scalar.first = load float, ptr addrspace(1) %n154.scalar.first.ptr, align 4
%n154.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n154.second, i32 %n154.scalar.p
%n154.scalar.second = load float, ptr addrspace(1) %n154.scalar.second.ptr, align 4
%n154.scalar.scalar.0 = call float @recipe.add(float %n154.scalar.first, float %n154.scalar.second)
%n154.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n154.value, i32 %n154.scalar.p
store float %n154.scalar.scalar.0, ptr addrspace(1) %n154.scalar.output.ptr, align 4
br label %n154.scalar.step
n154.scalar.step:
%n154.scalar.next = add i32 %n154.scalar.p, %threads
br label %n154.scalar.loop
n154.scalar.done:
call void @grid_barrier(i32 %threads)%n155.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n155.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n155.value = getelementptr i8, ptr addrspace(1) %values, i64 9920
%n155.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1784
%n155.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n155.scalar.entry
n155.scalar.entry:
%n155.scalar.count = mul i32 %rows, 16
br label %n155.scalar.loop
n155.scalar.loop:
%n155.scalar.p = phi i32 [ %tid, %n155.scalar.entry ], [ %n155.scalar.next, %n155.scalar.step ]
%n155.scalar.more = icmp ult i32 %n155.scalar.p, %n155.scalar.count
br i1 %n155.scalar.more, label %n155.scalar.body, label %n155.scalar.done
n155.scalar.body:
%n155.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n155.source, i32 %n155.scalar.p
%n155.scalar.first = load float, ptr addrspace(1) %n155.scalar.first.ptr, align 4
%n155.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n155.second, i32 %n155.scalar.p
%n155.scalar.second = load float, ptr addrspace(1) %n155.scalar.second.ptr, align 4
%n155.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n155.scalar.first, float %n155.scalar.second)
%n155.scalar.scalar.0 = call float @recipe.from.u1(i1 %n155.scalar.scalar.0.condition)
%n155.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n155.value, i32 %n155.scalar.p
store float %n155.scalar.scalar.0, ptr addrspace(1) %n155.scalar.output.ptr, align 4
br label %n155.scalar.step
n155.scalar.step:
%n155.scalar.next = add i32 %n155.scalar.p, %threads
br label %n155.scalar.loop
n155.scalar.done:
call void @grid_barrier(i32 %threads)%n156.source = getelementptr i8, ptr addrspace(1) %values, i64 9856
%n156.second = getelementptr i8, ptr addrspace(1) %values, i64 9920
%n156.value = getelementptr i8, ptr addrspace(1) %values, i64 9984
%n156.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1792
%n156.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n156.scalar.entry
n156.scalar.entry:
%n156.scalar.count = mul i32 %rows, 16
br label %n156.scalar.loop
n156.scalar.loop:
%n156.scalar.p = phi i32 [ %tid, %n156.scalar.entry ], [ %n156.scalar.next, %n156.scalar.step ]
%n156.scalar.more = icmp ult i32 %n156.scalar.p, %n156.scalar.count
br i1 %n156.scalar.more, label %n156.scalar.body, label %n156.scalar.done
n156.scalar.body:
%n156.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n156.source, i32 %n156.scalar.p
%n156.scalar.first = load float, ptr addrspace(1) %n156.scalar.first.ptr, align 4
%n156.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n156.second, i32 %n156.scalar.p
%n156.scalar.second = load float, ptr addrspace(1) %n156.scalar.second.ptr, align 4
%n156.scalar.scalar.0 = call float @recipe.add(float %n156.scalar.first, float %n156.scalar.second)
%n156.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n156.value, i32 %n156.scalar.p
store float %n156.scalar.scalar.0, ptr addrspace(1) %n156.scalar.output.ptr, align 4
br label %n156.scalar.step
n156.scalar.step:
%n156.scalar.next = add i32 %n156.scalar.p, %threads
br label %n156.scalar.loop
n156.scalar.done:
call void @grid_barrier(i32 %threads)%n157.source = getelementptr i8, ptr addrspace(1) %values, i64 9984
%n157.value = getelementptr i8, ptr addrspace(1) %values, i64 10048
%n157.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1800
%n157.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n157.scalar.entry
n157.scalar.entry:
%n157.scalar.count = mul i32 %rows, 16
br label %n157.scalar.loop
n157.scalar.loop:
%n157.scalar.p = phi i32 [ %tid, %n157.scalar.entry ], [ %n157.scalar.next, %n157.scalar.step ]
%n157.scalar.more = icmp ult i32 %n157.scalar.p, %n157.scalar.count
br i1 %n157.scalar.more, label %n157.scalar.body, label %n157.scalar.done
n157.scalar.body:
%n157.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n157.source, i32 %n157.scalar.p
%n157.scalar.first = load float, ptr addrspace(1) %n157.scalar.first.ptr, align 4
%n157.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n157.scalar.first)
%n157.scalar.scalar.1 = call float @recipe.from.u1(i1 %n157.scalar.scalar.1.condition)
%n157.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n157.value, i32 %n157.scalar.p
store float %n157.scalar.scalar.1, ptr addrspace(1) %n157.scalar.output.ptr, align 4
br label %n157.scalar.step
n157.scalar.step:
%n157.scalar.next = add i32 %n157.scalar.p, %threads
br label %n157.scalar.loop
n157.scalar.done:
call void @grid_barrier(i32 %threads)%n158.source = getelementptr i8, ptr addrspace(1) %values, i64 10048
%n158.second = getelementptr i8, ptr addrspace(1) %values, i64 8000
%n158.value = getelementptr i8, ptr addrspace(1) %values, i64 10112
%n158.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1808
%n158.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n158.scalar.entry
n158.scalar.entry:
%n158.scalar.count = mul i32 %rows, 16
br label %n158.scalar.loop
n158.scalar.loop:
%n158.scalar.p = phi i32 [ %tid, %n158.scalar.entry ], [ %n158.scalar.next, %n158.scalar.step ]
%n158.scalar.more = icmp ult i32 %n158.scalar.p, %n158.scalar.count
br i1 %n158.scalar.more, label %n158.scalar.body, label %n158.scalar.done
n158.scalar.body:
%n158.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n158.source, i32 %n158.scalar.p
%n158.scalar.first = load float, ptr addrspace(1) %n158.scalar.first.ptr, align 4
%n158.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n158.second, i32 %n158.scalar.p
%n158.scalar.second = load float, ptr addrspace(1) %n158.scalar.second.ptr, align 4
%n158.scalar.scalar.0 = call float @recipe.mul(float %n158.scalar.first, float %n158.scalar.second)
%n158.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n158.value, i32 %n158.scalar.p
store float %n158.scalar.scalar.0, ptr addrspace(1) %n158.scalar.output.ptr, align 4
br label %n158.scalar.step
n158.scalar.step:
%n158.scalar.next = add i32 %n158.scalar.p, %threads
br label %n158.scalar.loop
n158.scalar.done:
call void @grid_barrier(i32 %threads)%n159.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n159.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n159.value = getelementptr i8, ptr addrspace(1) %values, i64 10176
%n159.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1816
%n159.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n159.scalar.entry
n159.scalar.entry:
%n159.scalar.count = mul i32 %rows, 16
br label %n159.scalar.loop
n159.scalar.loop:
%n159.scalar.p = phi i32 [ %tid, %n159.scalar.entry ], [ %n159.scalar.next, %n159.scalar.step ]
%n159.scalar.more = icmp ult i32 %n159.scalar.p, %n159.scalar.count
br i1 %n159.scalar.more, label %n159.scalar.body, label %n159.scalar.done
n159.scalar.body:
%n159.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n159.source, i32 %n159.scalar.p
%n159.scalar.first = load float, ptr addrspace(1) %n159.scalar.first.ptr, align 4
%n159.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n159.second, i32 %n159.scalar.p
%n159.scalar.second = load float, ptr addrspace(1) %n159.scalar.second.ptr, align 4
%n159.scalar.scalar.0 = call float @recipe.sub(float %n159.scalar.first, float %n159.scalar.second)
%n159.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n159.value, i32 %n159.scalar.p
store float %n159.scalar.scalar.0, ptr addrspace(1) %n159.scalar.output.ptr, align 4
br label %n159.scalar.step
n159.scalar.step:
%n159.scalar.next = add i32 %n159.scalar.p, %threads
br label %n159.scalar.loop
n159.scalar.done:
call void @grid_barrier(i32 %threads)%n160.source = getelementptr i8, ptr addrspace(1) %values, i64 10176
%n160.value = getelementptr i8, ptr addrspace(1) %values, i64 10240
%n160.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1824
%n160.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n160.scalar.entry
n160.scalar.entry:
%n160.scalar.count = mul i32 %rows, 16
br label %n160.scalar.loop
n160.scalar.loop:
%n160.scalar.p = phi i32 [ %tid, %n160.scalar.entry ], [ %n160.scalar.next, %n160.scalar.step ]
%n160.scalar.more = icmp ult i32 %n160.scalar.p, %n160.scalar.count
br i1 %n160.scalar.more, label %n160.scalar.body, label %n160.scalar.done
n160.scalar.body:
%n160.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n160.source, i32 %n160.scalar.p
%n160.scalar.first = load float, ptr addrspace(1) %n160.scalar.first.ptr, align 4
%n160.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n160.scalar.first, float 0x0000000000000000)
%n160.scalar.scalar.2 = call float @recipe.from.u1(i1 %n160.scalar.scalar.2.condition)
%n160.scalar.scalar.3 = call float @recipe.exp(float %n160.scalar.first)
%n160.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n160.value, i32 %n160.scalar.p
store float %n160.scalar.scalar.3, ptr addrspace(1) %n160.scalar.output.ptr, align 4
br label %n160.scalar.step
n160.scalar.step:
%n160.scalar.next = add i32 %n160.scalar.p, %threads
br label %n160.scalar.loop
n160.scalar.done:
call void @grid_barrier(i32 %threads)%n161.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n161.value = getelementptr i8, ptr addrspace(1) %values, i64 10304
%n161.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1832
%n161.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n161.scalar.entry
n161.scalar.entry:
%n161.scalar.count = mul i32 %rows, 16
br label %n161.scalar.loop
n161.scalar.loop:
%n161.scalar.p = phi i32 [ %tid, %n161.scalar.entry ], [ %n161.scalar.next, %n161.scalar.step ]
%n161.scalar.more = icmp ult i32 %n161.scalar.p, %n161.scalar.count
br i1 %n161.scalar.more, label %n161.scalar.body, label %n161.scalar.done
n161.scalar.body:
%n161.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n161.source, i32 %n161.scalar.p
%n161.scalar.first = load float, ptr addrspace(1) %n161.scalar.first.ptr, align 4
%n161.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n161.value, i32 %n161.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n161.scalar.output.ptr, align 4
br label %n161.scalar.step
n161.scalar.step:
%n161.scalar.next = add i32 %n161.scalar.p, %threads
br label %n161.scalar.loop
n161.scalar.done:
call void @grid_barrier(i32 %threads)%n162.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n162.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n162.value = getelementptr i8, ptr addrspace(1) %values, i64 10368
%n162.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1840
%n162.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n162.scalar.entry
n162.scalar.entry:
%n162.scalar.count = mul i32 %rows, 16
br label %n162.scalar.loop
n162.scalar.loop:
%n162.scalar.p = phi i32 [ %tid, %n162.scalar.entry ], [ %n162.scalar.next, %n162.scalar.step ]
%n162.scalar.more = icmp ult i32 %n162.scalar.p, %n162.scalar.count
br i1 %n162.scalar.more, label %n162.scalar.body, label %n162.scalar.done
n162.scalar.body:
%n162.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n162.source, i32 %n162.scalar.p
%n162.scalar.first = load float, ptr addrspace(1) %n162.scalar.first.ptr, align 4
%n162.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n162.second, i32 %n162.scalar.p
%n162.scalar.second = load float, ptr addrspace(1) %n162.scalar.second.ptr, align 4
%n162.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n162.scalar.first, float %n162.scalar.second)
%n162.scalar.scalar.0 = call float @recipe.from.u1(i1 %n162.scalar.scalar.0.condition)
%n162.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n162.value, i32 %n162.scalar.p
store float %n162.scalar.scalar.0, ptr addrspace(1) %n162.scalar.output.ptr, align 4
br label %n162.scalar.step
n162.scalar.step:
%n162.scalar.next = add i32 %n162.scalar.p, %threads
br label %n162.scalar.loop
n162.scalar.done:
call void @grid_barrier(i32 %threads)%n163.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n163.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n163.value = getelementptr i8, ptr addrspace(1) %values, i64 10432
%n163.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1848
%n163.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n163.scalar.entry
n163.scalar.entry:
%n163.scalar.count = mul i32 %rows, 16
br label %n163.scalar.loop
n163.scalar.loop:
%n163.scalar.p = phi i32 [ %tid, %n163.scalar.entry ], [ %n163.scalar.next, %n163.scalar.step ]
%n163.scalar.more = icmp ult i32 %n163.scalar.p, %n163.scalar.count
br i1 %n163.scalar.more, label %n163.scalar.body, label %n163.scalar.done
n163.scalar.body:
%n163.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n163.source, i32 %n163.scalar.p
%n163.scalar.first = load float, ptr addrspace(1) %n163.scalar.first.ptr, align 4
%n163.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n163.second, i32 %n163.scalar.p
%n163.scalar.second = load float, ptr addrspace(1) %n163.scalar.second.ptr, align 4
%n163.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n163.scalar.first, float %n163.scalar.second)
%n163.scalar.scalar.0 = call float @recipe.from.u1(i1 %n163.scalar.scalar.0.condition)
%n163.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n163.value, i32 %n163.scalar.p
store float %n163.scalar.scalar.0, ptr addrspace(1) %n163.scalar.output.ptr, align 4
br label %n163.scalar.step
n163.scalar.step:
%n163.scalar.next = add i32 %n163.scalar.p, %threads
br label %n163.scalar.loop
n163.scalar.done:
call void @grid_barrier(i32 %threads)%n164.source = getelementptr i8, ptr addrspace(1) %values, i64 10368
%n164.second = getelementptr i8, ptr addrspace(1) %values, i64 10432
%n164.value = getelementptr i8, ptr addrspace(1) %values, i64 10496
%n164.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1856
%n164.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n164.scalar.entry
n164.scalar.entry:
%n164.scalar.count = mul i32 %rows, 16
br label %n164.scalar.loop
n164.scalar.loop:
%n164.scalar.p = phi i32 [ %tid, %n164.scalar.entry ], [ %n164.scalar.next, %n164.scalar.step ]
%n164.scalar.more = icmp ult i32 %n164.scalar.p, %n164.scalar.count
br i1 %n164.scalar.more, label %n164.scalar.body, label %n164.scalar.done
n164.scalar.body:
%n164.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n164.source, i32 %n164.scalar.p
%n164.scalar.first = load float, ptr addrspace(1) %n164.scalar.first.ptr, align 4
%n164.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n164.second, i32 %n164.scalar.p
%n164.scalar.second = load float, ptr addrspace(1) %n164.scalar.second.ptr, align 4
%n164.scalar.scalar.0 = call float @recipe.add(float %n164.scalar.first, float %n164.scalar.second)
%n164.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n164.value, i32 %n164.scalar.p
store float %n164.scalar.scalar.0, ptr addrspace(1) %n164.scalar.output.ptr, align 4
br label %n164.scalar.step
n164.scalar.step:
%n164.scalar.next = add i32 %n164.scalar.p, %threads
br label %n164.scalar.loop
n164.scalar.done:
call void @grid_barrier(i32 %threads)%n165.source = getelementptr i8, ptr addrspace(1) %values, i64 10496
%n165.value = getelementptr i8, ptr addrspace(1) %values, i64 10560
%n165.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1864
%n165.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n165.scalar.entry
n165.scalar.entry:
%n165.scalar.count = mul i32 %rows, 16
br label %n165.scalar.loop
n165.scalar.loop:
%n165.scalar.p = phi i32 [ %tid, %n165.scalar.entry ], [ %n165.scalar.next, %n165.scalar.step ]
%n165.scalar.more = icmp ult i32 %n165.scalar.p, %n165.scalar.count
br i1 %n165.scalar.more, label %n165.scalar.body, label %n165.scalar.done
n165.scalar.body:
%n165.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n165.source, i32 %n165.scalar.p
%n165.scalar.first = load float, ptr addrspace(1) %n165.scalar.first.ptr, align 4
%n165.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n165.scalar.first)
%n165.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n165.value, i32 %n165.scalar.p
store float %n165.scalar.scalar.1, ptr addrspace(1) %n165.scalar.output.ptr, align 4
br label %n165.scalar.step
n165.scalar.step:
%n165.scalar.next = add i32 %n165.scalar.p, %threads
br label %n165.scalar.loop
n165.scalar.done:
call void @grid_barrier(i32 %threads)%n166.source = getelementptr i8, ptr addrspace(1) %values, i64 10368
%n166.second = getelementptr i8, ptr addrspace(1) %values, i64 10560
%n166.value = getelementptr i8, ptr addrspace(1) %values, i64 10624
%n166.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1872
%n166.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n166.scalar.entry
n166.scalar.entry:
%n166.scalar.count = mul i32 %rows, 16
br label %n166.scalar.loop
n166.scalar.loop:
%n166.scalar.p = phi i32 [ %tid, %n166.scalar.entry ], [ %n166.scalar.next, %n166.scalar.step ]
%n166.scalar.more = icmp ult i32 %n166.scalar.p, %n166.scalar.count
br i1 %n166.scalar.more, label %n166.scalar.body, label %n166.scalar.done
n166.scalar.body:
%n166.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n166.source, i32 %n166.scalar.p
%n166.scalar.first = load float, ptr addrspace(1) %n166.scalar.first.ptr, align 4
%n166.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n166.second, i32 %n166.scalar.p
%n166.scalar.second = load float, ptr addrspace(1) %n166.scalar.second.ptr, align 4
%n166.scalar.scalar.0 = call float @recipe.add(float %n166.scalar.first, float %n166.scalar.second)
%n166.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n166.value, i32 %n166.scalar.p
store float %n166.scalar.scalar.0, ptr addrspace(1) %n166.scalar.output.ptr, align 4
br label %n166.scalar.step
n166.scalar.step:
%n166.scalar.next = add i32 %n166.scalar.p, %threads
br label %n166.scalar.loop
n166.scalar.done:
call void @grid_barrier(i32 %threads)%n167.source = getelementptr i8, ptr addrspace(1) %values, i64 10304
%n167.second = getelementptr i8, ptr addrspace(1) %values, i64 10624
%n167.value = getelementptr i8, ptr addrspace(1) %values, i64 10688
%n167.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1880
%n167.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n167.scalar.entry
n167.scalar.entry:
%n167.scalar.count = mul i32 %rows, 16
br label %n167.scalar.loop
n167.scalar.loop:
%n167.scalar.p = phi i32 [ %tid, %n167.scalar.entry ], [ %n167.scalar.next, %n167.scalar.step ]
%n167.scalar.more = icmp ult i32 %n167.scalar.p, %n167.scalar.count
br i1 %n167.scalar.more, label %n167.scalar.body, label %n167.scalar.done
n167.scalar.body:
%n167.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n167.source, i32 %n167.scalar.p
%n167.scalar.first = load float, ptr addrspace(1) %n167.scalar.first.ptr, align 4
%n167.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n167.second, i32 %n167.scalar.p
%n167.scalar.second = load float, ptr addrspace(1) %n167.scalar.second.ptr, align 4
%n167.scalar.scalar.0 = call float @recipe.add(float %n167.scalar.first, float %n167.scalar.second)
%n167.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n167.value, i32 %n167.scalar.p
store float %n167.scalar.scalar.0, ptr addrspace(1) %n167.scalar.output.ptr, align 4
br label %n167.scalar.step
n167.scalar.step:
%n167.scalar.next = add i32 %n167.scalar.p, %threads
br label %n167.scalar.loop
n167.scalar.done:
call void @grid_barrier(i32 %threads)%n168.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n168.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n168.value = getelementptr i8, ptr addrspace(1) %values, i64 10752
%n168.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1888
%n168.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n168.scalar.entry
n168.scalar.entry:
%n168.scalar.count = mul i32 %rows, 16
br label %n168.scalar.loop
n168.scalar.loop:
%n168.scalar.p = phi i32 [ %tid, %n168.scalar.entry ], [ %n168.scalar.next, %n168.scalar.step ]
%n168.scalar.more = icmp ult i32 %n168.scalar.p, %n168.scalar.count
br i1 %n168.scalar.more, label %n168.scalar.body, label %n168.scalar.done
n168.scalar.body:
%n168.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n168.source, i32 %n168.scalar.p
%n168.scalar.first = load float, ptr addrspace(1) %n168.scalar.first.ptr, align 4
%n168.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n168.second, i32 %n168.scalar.p
%n168.scalar.second = load float, ptr addrspace(1) %n168.scalar.second.ptr, align 4
%n168.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n168.scalar.first, float %n168.scalar.second)
%n168.scalar.scalar.0 = call float @recipe.from.u1(i1 %n168.scalar.scalar.0.condition)
%n168.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n168.value, i32 %n168.scalar.p
store float %n168.scalar.scalar.0, ptr addrspace(1) %n168.scalar.output.ptr, align 4
br label %n168.scalar.step
n168.scalar.step:
%n168.scalar.next = add i32 %n168.scalar.p, %threads
br label %n168.scalar.loop
n168.scalar.done:
call void @grid_barrier(i32 %threads)%n169.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n169.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n169.value = getelementptr i8, ptr addrspace(1) %values, i64 10816
%n169.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1896
%n169.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n169.scalar.entry
n169.scalar.entry:
%n169.scalar.count = mul i32 %rows, 16
br label %n169.scalar.loop
n169.scalar.loop:
%n169.scalar.p = phi i32 [ %tid, %n169.scalar.entry ], [ %n169.scalar.next, %n169.scalar.step ]
%n169.scalar.more = icmp ult i32 %n169.scalar.p, %n169.scalar.count
br i1 %n169.scalar.more, label %n169.scalar.body, label %n169.scalar.done
n169.scalar.body:
%n169.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n169.source, i32 %n169.scalar.p
%n169.scalar.first = load float, ptr addrspace(1) %n169.scalar.first.ptr, align 4
%n169.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n169.second, i32 %n169.scalar.p
%n169.scalar.second = load float, ptr addrspace(1) %n169.scalar.second.ptr, align 4
%n169.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n169.scalar.first, float %n169.scalar.second)
%n169.scalar.scalar.0 = call float @recipe.from.u1(i1 %n169.scalar.scalar.0.condition)
%n169.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n169.value, i32 %n169.scalar.p
store float %n169.scalar.scalar.0, ptr addrspace(1) %n169.scalar.output.ptr, align 4
br label %n169.scalar.step
n169.scalar.step:
%n169.scalar.next = add i32 %n169.scalar.p, %threads
br label %n169.scalar.loop
n169.scalar.done:
call void @grid_barrier(i32 %threads)%n170.source = getelementptr i8, ptr addrspace(1) %values, i64 10752
%n170.second = getelementptr i8, ptr addrspace(1) %values, i64 10816
%n170.value = getelementptr i8, ptr addrspace(1) %values, i64 10880
%n170.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1904
%n170.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n170.scalar.entry
n170.scalar.entry:
%n170.scalar.count = mul i32 %rows, 16
br label %n170.scalar.loop
n170.scalar.loop:
%n170.scalar.p = phi i32 [ %tid, %n170.scalar.entry ], [ %n170.scalar.next, %n170.scalar.step ]
%n170.scalar.more = icmp ult i32 %n170.scalar.p, %n170.scalar.count
br i1 %n170.scalar.more, label %n170.scalar.body, label %n170.scalar.done
n170.scalar.body:
%n170.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n170.source, i32 %n170.scalar.p
%n170.scalar.first = load float, ptr addrspace(1) %n170.scalar.first.ptr, align 4
%n170.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n170.second, i32 %n170.scalar.p
%n170.scalar.second = load float, ptr addrspace(1) %n170.scalar.second.ptr, align 4
%n170.scalar.scalar.0 = call float @recipe.add(float %n170.scalar.first, float %n170.scalar.second)
%n170.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n170.value, i32 %n170.scalar.p
store float %n170.scalar.scalar.0, ptr addrspace(1) %n170.scalar.output.ptr, align 4
br label %n170.scalar.step
n170.scalar.step:
%n170.scalar.next = add i32 %n170.scalar.p, %threads
br label %n170.scalar.loop
n170.scalar.done:
call void @grid_barrier(i32 %threads)%n171.source = getelementptr i8, ptr addrspace(1) %values, i64 10880
%n171.value = getelementptr i8, ptr addrspace(1) %values, i64 10944
%n171.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1912
%n171.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n171.scalar.entry
n171.scalar.entry:
%n171.scalar.count = mul i32 %rows, 16
br label %n171.scalar.loop
n171.scalar.loop:
%n171.scalar.p = phi i32 [ %tid, %n171.scalar.entry ], [ %n171.scalar.next, %n171.scalar.step ]
%n171.scalar.more = icmp ult i32 %n171.scalar.p, %n171.scalar.count
br i1 %n171.scalar.more, label %n171.scalar.body, label %n171.scalar.done
n171.scalar.body:
%n171.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n171.source, i32 %n171.scalar.p
%n171.scalar.first = load float, ptr addrspace(1) %n171.scalar.first.ptr, align 4
%n171.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n171.scalar.first)
%n171.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n171.value, i32 %n171.scalar.p
store float %n171.scalar.scalar.1, ptr addrspace(1) %n171.scalar.output.ptr, align 4
br label %n171.scalar.step
n171.scalar.step:
%n171.scalar.next = add i32 %n171.scalar.p, %threads
br label %n171.scalar.loop
n171.scalar.done:
call void @grid_barrier(i32 %threads)%n172.source = getelementptr i8, ptr addrspace(1) %values, i64 10752
%n172.second = getelementptr i8, ptr addrspace(1) %values, i64 10944
%n172.value = getelementptr i8, ptr addrspace(1) %values, i64 11008
%n172.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1920
%n172.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n172.scalar.entry
n172.scalar.entry:
%n172.scalar.count = mul i32 %rows, 16
br label %n172.scalar.loop
n172.scalar.loop:
%n172.scalar.p = phi i32 [ %tid, %n172.scalar.entry ], [ %n172.scalar.next, %n172.scalar.step ]
%n172.scalar.more = icmp ult i32 %n172.scalar.p, %n172.scalar.count
br i1 %n172.scalar.more, label %n172.scalar.body, label %n172.scalar.done
n172.scalar.body:
%n172.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n172.source, i32 %n172.scalar.p
%n172.scalar.first = load float, ptr addrspace(1) %n172.scalar.first.ptr, align 4
%n172.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n172.second, i32 %n172.scalar.p
%n172.scalar.second = load float, ptr addrspace(1) %n172.scalar.second.ptr, align 4
%n172.scalar.scalar.0 = call float @recipe.add(float %n172.scalar.first, float %n172.scalar.second)
%n172.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n172.value, i32 %n172.scalar.p
store float %n172.scalar.scalar.0, ptr addrspace(1) %n172.scalar.output.ptr, align 4
br label %n172.scalar.step
n172.scalar.step:
%n172.scalar.next = add i32 %n172.scalar.p, %threads
br label %n172.scalar.loop
n172.scalar.done:
call void @grid_barrier(i32 %threads)%n173.source = getelementptr i8, ptr addrspace(1) %values, i64 10688
%n173.second = getelementptr i8, ptr addrspace(1) %values, i64 11008
%n173.value = getelementptr i8, ptr addrspace(1) %values, i64 11072
%n173.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1928
%n173.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n173.scalar.entry
n173.scalar.entry:
%n173.scalar.count = mul i32 %rows, 16
br label %n173.scalar.loop
n173.scalar.loop:
%n173.scalar.p = phi i32 [ %tid, %n173.scalar.entry ], [ %n173.scalar.next, %n173.scalar.step ]
%n173.scalar.more = icmp ult i32 %n173.scalar.p, %n173.scalar.count
br i1 %n173.scalar.more, label %n173.scalar.body, label %n173.scalar.done
n173.scalar.body:
%n173.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n173.source, i32 %n173.scalar.p
%n173.scalar.first = load float, ptr addrspace(1) %n173.scalar.first.ptr, align 4
%n173.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n173.second, i32 %n173.scalar.p
%n173.scalar.second = load float, ptr addrspace(1) %n173.scalar.second.ptr, align 4
%n173.scalar.scalar.0 = call float @recipe.add(float %n173.scalar.first, float %n173.scalar.second)
%n173.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n173.value, i32 %n173.scalar.p
store float %n173.scalar.scalar.0, ptr addrspace(1) %n173.scalar.output.ptr, align 4
br label %n173.scalar.step
n173.scalar.step:
%n173.scalar.next = add i32 %n173.scalar.p, %threads
br label %n173.scalar.loop
n173.scalar.done:
call void @grid_barrier(i32 %threads)%n174.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n174.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n174.value = getelementptr i8, ptr addrspace(1) %values, i64 11136
%n174.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1936
%n174.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n174.scalar.entry
n174.scalar.entry:
%n174.scalar.count = mul i32 %rows, 16
br label %n174.scalar.loop
n174.scalar.loop:
%n174.scalar.p = phi i32 [ %tid, %n174.scalar.entry ], [ %n174.scalar.next, %n174.scalar.step ]
%n174.scalar.more = icmp ult i32 %n174.scalar.p, %n174.scalar.count
br i1 %n174.scalar.more, label %n174.scalar.body, label %n174.scalar.done
n174.scalar.body:
%n174.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n174.source, i32 %n174.scalar.p
%n174.scalar.first = load float, ptr addrspace(1) %n174.scalar.first.ptr, align 4
%n174.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n174.second, i32 %n174.scalar.p
%n174.scalar.second = load float, ptr addrspace(1) %n174.scalar.second.ptr, align 4
%n174.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n174.scalar.first, float %n174.scalar.second)
%n174.scalar.scalar.0 = call float @recipe.from.u1(i1 %n174.scalar.scalar.0.condition)
%n174.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n174.value, i32 %n174.scalar.p
store float %n174.scalar.scalar.0, ptr addrspace(1) %n174.scalar.output.ptr, align 4
br label %n174.scalar.step
n174.scalar.step:
%n174.scalar.next = add i32 %n174.scalar.p, %threads
br label %n174.scalar.loop
n174.scalar.done:
call void @grid_barrier(i32 %threads)%n175.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n175.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n175.value = getelementptr i8, ptr addrspace(1) %values, i64 11200
%n175.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1944
%n175.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n175.scalar.entry
n175.scalar.entry:
%n175.scalar.count = mul i32 %rows, 16
br label %n175.scalar.loop
n175.scalar.loop:
%n175.scalar.p = phi i32 [ %tid, %n175.scalar.entry ], [ %n175.scalar.next, %n175.scalar.step ]
%n175.scalar.more = icmp ult i32 %n175.scalar.p, %n175.scalar.count
br i1 %n175.scalar.more, label %n175.scalar.body, label %n175.scalar.done
n175.scalar.body:
%n175.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n175.source, i32 %n175.scalar.p
%n175.scalar.first = load float, ptr addrspace(1) %n175.scalar.first.ptr, align 4
%n175.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n175.second, i32 %n175.scalar.p
%n175.scalar.second = load float, ptr addrspace(1) %n175.scalar.second.ptr, align 4
%n175.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n175.scalar.first, float %n175.scalar.second)
%n175.scalar.scalar.0 = call float @recipe.from.u1(i1 %n175.scalar.scalar.0.condition)
%n175.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n175.value, i32 %n175.scalar.p
store float %n175.scalar.scalar.0, ptr addrspace(1) %n175.scalar.output.ptr, align 4
br label %n175.scalar.step
n175.scalar.step:
%n175.scalar.next = add i32 %n175.scalar.p, %threads
br label %n175.scalar.loop
n175.scalar.done:
call void @grid_barrier(i32 %threads)%n176.source = getelementptr i8, ptr addrspace(1) %values, i64 11136
%n176.second = getelementptr i8, ptr addrspace(1) %values, i64 11200
%n176.value = getelementptr i8, ptr addrspace(1) %values, i64 11264
%n176.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1952
%n176.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n176.scalar.entry
n176.scalar.entry:
%n176.scalar.count = mul i32 %rows, 16
br label %n176.scalar.loop
n176.scalar.loop:
%n176.scalar.p = phi i32 [ %tid, %n176.scalar.entry ], [ %n176.scalar.next, %n176.scalar.step ]
%n176.scalar.more = icmp ult i32 %n176.scalar.p, %n176.scalar.count
br i1 %n176.scalar.more, label %n176.scalar.body, label %n176.scalar.done
n176.scalar.body:
%n176.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n176.source, i32 %n176.scalar.p
%n176.scalar.first = load float, ptr addrspace(1) %n176.scalar.first.ptr, align 4
%n176.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n176.second, i32 %n176.scalar.p
%n176.scalar.second = load float, ptr addrspace(1) %n176.scalar.second.ptr, align 4
%n176.scalar.scalar.0 = call float @recipe.add(float %n176.scalar.first, float %n176.scalar.second)
%n176.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n176.value, i32 %n176.scalar.p
store float %n176.scalar.scalar.0, ptr addrspace(1) %n176.scalar.output.ptr, align 4
br label %n176.scalar.step
n176.scalar.step:
%n176.scalar.next = add i32 %n176.scalar.p, %threads
br label %n176.scalar.loop
n176.scalar.done:
call void @grid_barrier(i32 %threads)%n177.source = getelementptr i8, ptr addrspace(1) %values, i64 11264
%n177.value = getelementptr i8, ptr addrspace(1) %values, i64 11328
%n177.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1960
%n177.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n177.scalar.entry
n177.scalar.entry:
%n177.scalar.count = mul i32 %rows, 16
br label %n177.scalar.loop
n177.scalar.loop:
%n177.scalar.p = phi i32 [ %tid, %n177.scalar.entry ], [ %n177.scalar.next, %n177.scalar.step ]
%n177.scalar.more = icmp ult i32 %n177.scalar.p, %n177.scalar.count
br i1 %n177.scalar.more, label %n177.scalar.body, label %n177.scalar.done
n177.scalar.body:
%n177.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n177.source, i32 %n177.scalar.p
%n177.scalar.first = load float, ptr addrspace(1) %n177.scalar.first.ptr, align 4
%n177.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n177.scalar.first)
%n177.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n177.value, i32 %n177.scalar.p
store float %n177.scalar.scalar.1, ptr addrspace(1) %n177.scalar.output.ptr, align 4
br label %n177.scalar.step
n177.scalar.step:
%n177.scalar.next = add i32 %n177.scalar.p, %threads
br label %n177.scalar.loop
n177.scalar.done:
call void @grid_barrier(i32 %threads)%n178.source = getelementptr i8, ptr addrspace(1) %values, i64 11136
%n178.second = getelementptr i8, ptr addrspace(1) %values, i64 11328
%n178.value = getelementptr i8, ptr addrspace(1) %values, i64 11392
%n178.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1968
%n178.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n178.scalar.entry
n178.scalar.entry:
%n178.scalar.count = mul i32 %rows, 16
br label %n178.scalar.loop
n178.scalar.loop:
%n178.scalar.p = phi i32 [ %tid, %n178.scalar.entry ], [ %n178.scalar.next, %n178.scalar.step ]
%n178.scalar.more = icmp ult i32 %n178.scalar.p, %n178.scalar.count
br i1 %n178.scalar.more, label %n178.scalar.body, label %n178.scalar.done
n178.scalar.body:
%n178.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n178.source, i32 %n178.scalar.p
%n178.scalar.first = load float, ptr addrspace(1) %n178.scalar.first.ptr, align 4
%n178.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n178.second, i32 %n178.scalar.p
%n178.scalar.second = load float, ptr addrspace(1) %n178.scalar.second.ptr, align 4
%n178.scalar.scalar.0 = call float @recipe.add(float %n178.scalar.first, float %n178.scalar.second)
%n178.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n178.value, i32 %n178.scalar.p
store float %n178.scalar.scalar.0, ptr addrspace(1) %n178.scalar.output.ptr, align 4
br label %n178.scalar.step
n178.scalar.step:
%n178.scalar.next = add i32 %n178.scalar.p, %threads
br label %n178.scalar.loop
n178.scalar.done:
call void @grid_barrier(i32 %threads)%n179.source = getelementptr i8, ptr addrspace(1) %values, i64 11072
%n179.second = getelementptr i8, ptr addrspace(1) %values, i64 11392
%n179.value = getelementptr i8, ptr addrspace(1) %values, i64 11456
%n179.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1976
%n179.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n179.scalar.entry
n179.scalar.entry:
%n179.scalar.count = mul i32 %rows, 16
br label %n179.scalar.loop
n179.scalar.loop:
%n179.scalar.p = phi i32 [ %tid, %n179.scalar.entry ], [ %n179.scalar.next, %n179.scalar.step ]
%n179.scalar.more = icmp ult i32 %n179.scalar.p, %n179.scalar.count
br i1 %n179.scalar.more, label %n179.scalar.body, label %n179.scalar.done
n179.scalar.body:
%n179.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n179.source, i32 %n179.scalar.p
%n179.scalar.first = load float, ptr addrspace(1) %n179.scalar.first.ptr, align 4
%n179.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n179.second, i32 %n179.scalar.p
%n179.scalar.second = load float, ptr addrspace(1) %n179.scalar.second.ptr, align 4
%n179.scalar.scalar.0 = call float @recipe.add(float %n179.scalar.first, float %n179.scalar.second)
%n179.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n179.value, i32 %n179.scalar.p
store float %n179.scalar.scalar.0, ptr addrspace(1) %n179.scalar.output.ptr, align 4
br label %n179.scalar.step
n179.scalar.step:
%n179.scalar.next = add i32 %n179.scalar.p, %threads
br label %n179.scalar.loop
n179.scalar.done:
call void @grid_barrier(i32 %threads)%n180.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n180.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n180.value = getelementptr i8, ptr addrspace(1) %values, i64 11520
%n180.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1984
%n180.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n180.scalar.entry
n180.scalar.entry:
%n180.scalar.count = mul i32 %rows, 16
br label %n180.scalar.loop
n180.scalar.loop:
%n180.scalar.p = phi i32 [ %tid, %n180.scalar.entry ], [ %n180.scalar.next, %n180.scalar.step ]
%n180.scalar.more = icmp ult i32 %n180.scalar.p, %n180.scalar.count
br i1 %n180.scalar.more, label %n180.scalar.body, label %n180.scalar.done
n180.scalar.body:
%n180.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n180.source, i32 %n180.scalar.p
%n180.scalar.first = load float, ptr addrspace(1) %n180.scalar.first.ptr, align 4
%n180.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n180.second, i32 %n180.scalar.p
%n180.scalar.second = load float, ptr addrspace(1) %n180.scalar.second.ptr, align 4
%n180.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n180.scalar.first, float %n180.scalar.second)
%n180.scalar.scalar.0 = call float @recipe.from.u1(i1 %n180.scalar.scalar.0.condition)
%n180.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n180.value, i32 %n180.scalar.p
store float %n180.scalar.scalar.0, ptr addrspace(1) %n180.scalar.output.ptr, align 4
br label %n180.scalar.step
n180.scalar.step:
%n180.scalar.next = add i32 %n180.scalar.p, %threads
br label %n180.scalar.loop
n180.scalar.done:
call void @grid_barrier(i32 %threads)%n181.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n181.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n181.value = getelementptr i8, ptr addrspace(1) %values, i64 11584
%n181.context = getelementptr i8, ptr addrspace(1) %contexts, i64 1992
%n181.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n181.scalar.entry
n181.scalar.entry:
%n181.scalar.count = mul i32 %rows, 16
br label %n181.scalar.loop
n181.scalar.loop:
%n181.scalar.p = phi i32 [ %tid, %n181.scalar.entry ], [ %n181.scalar.next, %n181.scalar.step ]
%n181.scalar.more = icmp ult i32 %n181.scalar.p, %n181.scalar.count
br i1 %n181.scalar.more, label %n181.scalar.body, label %n181.scalar.done
n181.scalar.body:
%n181.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n181.source, i32 %n181.scalar.p
%n181.scalar.first = load float, ptr addrspace(1) %n181.scalar.first.ptr, align 4
%n181.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n181.second, i32 %n181.scalar.p
%n181.scalar.second = load float, ptr addrspace(1) %n181.scalar.second.ptr, align 4
%n181.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n181.scalar.first, float %n181.scalar.second)
%n181.scalar.scalar.0 = call float @recipe.from.u1(i1 %n181.scalar.scalar.0.condition)
%n181.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n181.value, i32 %n181.scalar.p
store float %n181.scalar.scalar.0, ptr addrspace(1) %n181.scalar.output.ptr, align 4
br label %n181.scalar.step
n181.scalar.step:
%n181.scalar.next = add i32 %n181.scalar.p, %threads
br label %n181.scalar.loop
n181.scalar.done:
call void @grid_barrier(i32 %threads)%n182.source = getelementptr i8, ptr addrspace(1) %values, i64 11520
%n182.second = getelementptr i8, ptr addrspace(1) %values, i64 11584
%n182.value = getelementptr i8, ptr addrspace(1) %values, i64 11648
%n182.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2000
%n182.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n182.scalar.entry
n182.scalar.entry:
%n182.scalar.count = mul i32 %rows, 16
br label %n182.scalar.loop
n182.scalar.loop:
%n182.scalar.p = phi i32 [ %tid, %n182.scalar.entry ], [ %n182.scalar.next, %n182.scalar.step ]
%n182.scalar.more = icmp ult i32 %n182.scalar.p, %n182.scalar.count
br i1 %n182.scalar.more, label %n182.scalar.body, label %n182.scalar.done
n182.scalar.body:
%n182.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n182.source, i32 %n182.scalar.p
%n182.scalar.first = load float, ptr addrspace(1) %n182.scalar.first.ptr, align 4
%n182.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n182.second, i32 %n182.scalar.p
%n182.scalar.second = load float, ptr addrspace(1) %n182.scalar.second.ptr, align 4
%n182.scalar.scalar.0 = call float @recipe.add(float %n182.scalar.first, float %n182.scalar.second)
%n182.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n182.value, i32 %n182.scalar.p
store float %n182.scalar.scalar.0, ptr addrspace(1) %n182.scalar.output.ptr, align 4
br label %n182.scalar.step
n182.scalar.step:
%n182.scalar.next = add i32 %n182.scalar.p, %threads
br label %n182.scalar.loop
n182.scalar.done:
call void @grid_barrier(i32 %threads)%n183.source = getelementptr i8, ptr addrspace(1) %values, i64 11648
%n183.value = getelementptr i8, ptr addrspace(1) %values, i64 11712
%n183.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2008
%n183.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n183.scalar.entry
n183.scalar.entry:
%n183.scalar.count = mul i32 %rows, 16
br label %n183.scalar.loop
n183.scalar.loop:
%n183.scalar.p = phi i32 [ %tid, %n183.scalar.entry ], [ %n183.scalar.next, %n183.scalar.step ]
%n183.scalar.more = icmp ult i32 %n183.scalar.p, %n183.scalar.count
br i1 %n183.scalar.more, label %n183.scalar.body, label %n183.scalar.done
n183.scalar.body:
%n183.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n183.source, i32 %n183.scalar.p
%n183.scalar.first = load float, ptr addrspace(1) %n183.scalar.first.ptr, align 4
%n183.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n183.scalar.first)
%n183.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n183.value, i32 %n183.scalar.p
store float %n183.scalar.scalar.1, ptr addrspace(1) %n183.scalar.output.ptr, align 4
br label %n183.scalar.step
n183.scalar.step:
%n183.scalar.next = add i32 %n183.scalar.p, %threads
br label %n183.scalar.loop
n183.scalar.done:
call void @grid_barrier(i32 %threads)%n184.source = getelementptr i8, ptr addrspace(1) %values, i64 11520
%n184.second = getelementptr i8, ptr addrspace(1) %values, i64 11712
%n184.value = getelementptr i8, ptr addrspace(1) %values, i64 11776
%n184.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2016
%n184.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n184.scalar.entry
n184.scalar.entry:
%n184.scalar.count = mul i32 %rows, 16
br label %n184.scalar.loop
n184.scalar.loop:
%n184.scalar.p = phi i32 [ %tid, %n184.scalar.entry ], [ %n184.scalar.next, %n184.scalar.step ]
%n184.scalar.more = icmp ult i32 %n184.scalar.p, %n184.scalar.count
br i1 %n184.scalar.more, label %n184.scalar.body, label %n184.scalar.done
n184.scalar.body:
%n184.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n184.source, i32 %n184.scalar.p
%n184.scalar.first = load float, ptr addrspace(1) %n184.scalar.first.ptr, align 4
%n184.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n184.second, i32 %n184.scalar.p
%n184.scalar.second = load float, ptr addrspace(1) %n184.scalar.second.ptr, align 4
%n184.scalar.scalar.0 = call float @recipe.add(float %n184.scalar.first, float %n184.scalar.second)
%n184.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n184.value, i32 %n184.scalar.p
store float %n184.scalar.scalar.0, ptr addrspace(1) %n184.scalar.output.ptr, align 4
br label %n184.scalar.step
n184.scalar.step:
%n184.scalar.next = add i32 %n184.scalar.p, %threads
br label %n184.scalar.loop
n184.scalar.done:
call void @grid_barrier(i32 %threads)%n185.source = getelementptr i8, ptr addrspace(1) %values, i64 11456
%n185.second = getelementptr i8, ptr addrspace(1) %values, i64 11776
%n185.value = getelementptr i8, ptr addrspace(1) %values, i64 11840
%n185.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2024
%n185.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n185.scalar.entry
n185.scalar.entry:
%n185.scalar.count = mul i32 %rows, 16
br label %n185.scalar.loop
n185.scalar.loop:
%n185.scalar.p = phi i32 [ %tid, %n185.scalar.entry ], [ %n185.scalar.next, %n185.scalar.step ]
%n185.scalar.more = icmp ult i32 %n185.scalar.p, %n185.scalar.count
br i1 %n185.scalar.more, label %n185.scalar.body, label %n185.scalar.done
n185.scalar.body:
%n185.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n185.source, i32 %n185.scalar.p
%n185.scalar.first = load float, ptr addrspace(1) %n185.scalar.first.ptr, align 4
%n185.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n185.second, i32 %n185.scalar.p
%n185.scalar.second = load float, ptr addrspace(1) %n185.scalar.second.ptr, align 4
%n185.scalar.scalar.0 = call float @recipe.add(float %n185.scalar.first, float %n185.scalar.second)
%n185.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n185.value, i32 %n185.scalar.p
store float %n185.scalar.scalar.0, ptr addrspace(1) %n185.scalar.output.ptr, align 4
br label %n185.scalar.step
n185.scalar.step:
%n185.scalar.next = add i32 %n185.scalar.p, %threads
br label %n185.scalar.loop
n185.scalar.done:
call void @grid_barrier(i32 %threads)%n186.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n186.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n186.value = getelementptr i8, ptr addrspace(1) %values, i64 11904
%n186.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2032
%n186.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n186.scalar.entry
n186.scalar.entry:
%n186.scalar.count = mul i32 %rows, 16
br label %n186.scalar.loop
n186.scalar.loop:
%n186.scalar.p = phi i32 [ %tid, %n186.scalar.entry ], [ %n186.scalar.next, %n186.scalar.step ]
%n186.scalar.more = icmp ult i32 %n186.scalar.p, %n186.scalar.count
br i1 %n186.scalar.more, label %n186.scalar.body, label %n186.scalar.done
n186.scalar.body:
%n186.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n186.source, i32 %n186.scalar.p
%n186.scalar.first = load float, ptr addrspace(1) %n186.scalar.first.ptr, align 4
%n186.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n186.second, i32 %n186.scalar.p
%n186.scalar.second = load float, ptr addrspace(1) %n186.scalar.second.ptr, align 4
%n186.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n186.scalar.first, float %n186.scalar.second)
%n186.scalar.scalar.0 = call float @recipe.from.u1(i1 %n186.scalar.scalar.0.condition)
%n186.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n186.value, i32 %n186.scalar.p
store float %n186.scalar.scalar.0, ptr addrspace(1) %n186.scalar.output.ptr, align 4
br label %n186.scalar.step
n186.scalar.step:
%n186.scalar.next = add i32 %n186.scalar.p, %threads
br label %n186.scalar.loop
n186.scalar.done:
call void @grid_barrier(i32 %threads)%n187.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n187.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n187.value = getelementptr i8, ptr addrspace(1) %values, i64 11968
%n187.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2040
%n187.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n187.scalar.entry
n187.scalar.entry:
%n187.scalar.count = mul i32 %rows, 16
br label %n187.scalar.loop
n187.scalar.loop:
%n187.scalar.p = phi i32 [ %tid, %n187.scalar.entry ], [ %n187.scalar.next, %n187.scalar.step ]
%n187.scalar.more = icmp ult i32 %n187.scalar.p, %n187.scalar.count
br i1 %n187.scalar.more, label %n187.scalar.body, label %n187.scalar.done
n187.scalar.body:
%n187.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n187.source, i32 %n187.scalar.p
%n187.scalar.first = load float, ptr addrspace(1) %n187.scalar.first.ptr, align 4
%n187.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n187.second, i32 %n187.scalar.p
%n187.scalar.second = load float, ptr addrspace(1) %n187.scalar.second.ptr, align 4
%n187.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n187.scalar.first, float %n187.scalar.second)
%n187.scalar.scalar.0 = call float @recipe.from.u1(i1 %n187.scalar.scalar.0.condition)
%n187.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n187.value, i32 %n187.scalar.p
store float %n187.scalar.scalar.0, ptr addrspace(1) %n187.scalar.output.ptr, align 4
br label %n187.scalar.step
n187.scalar.step:
%n187.scalar.next = add i32 %n187.scalar.p, %threads
br label %n187.scalar.loop
n187.scalar.done:
call void @grid_barrier(i32 %threads)%n188.source = getelementptr i8, ptr addrspace(1) %values, i64 11904
%n188.second = getelementptr i8, ptr addrspace(1) %values, i64 11968
%n188.value = getelementptr i8, ptr addrspace(1) %values, i64 12032
%n188.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2048
%n188.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n188.scalar.entry
n188.scalar.entry:
%n188.scalar.count = mul i32 %rows, 16
br label %n188.scalar.loop
n188.scalar.loop:
%n188.scalar.p = phi i32 [ %tid, %n188.scalar.entry ], [ %n188.scalar.next, %n188.scalar.step ]
%n188.scalar.more = icmp ult i32 %n188.scalar.p, %n188.scalar.count
br i1 %n188.scalar.more, label %n188.scalar.body, label %n188.scalar.done
n188.scalar.body:
%n188.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n188.source, i32 %n188.scalar.p
%n188.scalar.first = load float, ptr addrspace(1) %n188.scalar.first.ptr, align 4
%n188.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n188.second, i32 %n188.scalar.p
%n188.scalar.second = load float, ptr addrspace(1) %n188.scalar.second.ptr, align 4
%n188.scalar.scalar.0 = call float @recipe.add(float %n188.scalar.first, float %n188.scalar.second)
%n188.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n188.value, i32 %n188.scalar.p
store float %n188.scalar.scalar.0, ptr addrspace(1) %n188.scalar.output.ptr, align 4
br label %n188.scalar.step
n188.scalar.step:
%n188.scalar.next = add i32 %n188.scalar.p, %threads
br label %n188.scalar.loop
n188.scalar.done:
call void @grid_barrier(i32 %threads)%n189.source = getelementptr i8, ptr addrspace(1) %values, i64 12032
%n189.value = getelementptr i8, ptr addrspace(1) %values, i64 12096
%n189.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2056
%n189.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n189.scalar.entry
n189.scalar.entry:
%n189.scalar.count = mul i32 %rows, 16
br label %n189.scalar.loop
n189.scalar.loop:
%n189.scalar.p = phi i32 [ %tid, %n189.scalar.entry ], [ %n189.scalar.next, %n189.scalar.step ]
%n189.scalar.more = icmp ult i32 %n189.scalar.p, %n189.scalar.count
br i1 %n189.scalar.more, label %n189.scalar.body, label %n189.scalar.done
n189.scalar.body:
%n189.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n189.source, i32 %n189.scalar.p
%n189.scalar.first = load float, ptr addrspace(1) %n189.scalar.first.ptr, align 4
%n189.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n189.scalar.first)
%n189.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n189.value, i32 %n189.scalar.p
store float %n189.scalar.scalar.1, ptr addrspace(1) %n189.scalar.output.ptr, align 4
br label %n189.scalar.step
n189.scalar.step:
%n189.scalar.next = add i32 %n189.scalar.p, %threads
br label %n189.scalar.loop
n189.scalar.done:
call void @grid_barrier(i32 %threads)%n190.source = getelementptr i8, ptr addrspace(1) %values, i64 11904
%n190.second = getelementptr i8, ptr addrspace(1) %values, i64 12096
%n190.value = getelementptr i8, ptr addrspace(1) %values, i64 12160
%n190.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2064
%n190.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n190.scalar.entry
n190.scalar.entry:
%n190.scalar.count = mul i32 %rows, 16
br label %n190.scalar.loop
n190.scalar.loop:
%n190.scalar.p = phi i32 [ %tid, %n190.scalar.entry ], [ %n190.scalar.next, %n190.scalar.step ]
%n190.scalar.more = icmp ult i32 %n190.scalar.p, %n190.scalar.count
br i1 %n190.scalar.more, label %n190.scalar.body, label %n190.scalar.done
n190.scalar.body:
%n190.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n190.source, i32 %n190.scalar.p
%n190.scalar.first = load float, ptr addrspace(1) %n190.scalar.first.ptr, align 4
%n190.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n190.second, i32 %n190.scalar.p
%n190.scalar.second = load float, ptr addrspace(1) %n190.scalar.second.ptr, align 4
%n190.scalar.scalar.0 = call float @recipe.add(float %n190.scalar.first, float %n190.scalar.second)
%n190.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n190.value, i32 %n190.scalar.p
store float %n190.scalar.scalar.0, ptr addrspace(1) %n190.scalar.output.ptr, align 4
br label %n190.scalar.step
n190.scalar.step:
%n190.scalar.next = add i32 %n190.scalar.p, %threads
br label %n190.scalar.loop
n190.scalar.done:
call void @grid_barrier(i32 %threads)%n191.source = getelementptr i8, ptr addrspace(1) %values, i64 11840
%n191.second = getelementptr i8, ptr addrspace(1) %values, i64 12160
%n191.value = getelementptr i8, ptr addrspace(1) %values, i64 12224
%n191.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2072
%n191.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n191.scalar.entry
n191.scalar.entry:
%n191.scalar.count = mul i32 %rows, 16
br label %n191.scalar.loop
n191.scalar.loop:
%n191.scalar.p = phi i32 [ %tid, %n191.scalar.entry ], [ %n191.scalar.next, %n191.scalar.step ]
%n191.scalar.more = icmp ult i32 %n191.scalar.p, %n191.scalar.count
br i1 %n191.scalar.more, label %n191.scalar.body, label %n191.scalar.done
n191.scalar.body:
%n191.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n191.source, i32 %n191.scalar.p
%n191.scalar.first = load float, ptr addrspace(1) %n191.scalar.first.ptr, align 4
%n191.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n191.second, i32 %n191.scalar.p
%n191.scalar.second = load float, ptr addrspace(1) %n191.scalar.second.ptr, align 4
%n191.scalar.scalar.0 = call float @recipe.add(float %n191.scalar.first, float %n191.scalar.second)
%n191.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n191.value, i32 %n191.scalar.p
store float %n191.scalar.scalar.0, ptr addrspace(1) %n191.scalar.output.ptr, align 4
br label %n191.scalar.step
n191.scalar.step:
%n191.scalar.next = add i32 %n191.scalar.p, %threads
br label %n191.scalar.loop
n191.scalar.done:
call void @grid_barrier(i32 %threads)%n192.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n192.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n192.value = getelementptr i8, ptr addrspace(1) %values, i64 12288
%n192.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2080
%n192.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n192.scalar.entry
n192.scalar.entry:
%n192.scalar.count = mul i32 %rows, 16
br label %n192.scalar.loop
n192.scalar.loop:
%n192.scalar.p = phi i32 [ %tid, %n192.scalar.entry ], [ %n192.scalar.next, %n192.scalar.step ]
%n192.scalar.more = icmp ult i32 %n192.scalar.p, %n192.scalar.count
br i1 %n192.scalar.more, label %n192.scalar.body, label %n192.scalar.done
n192.scalar.body:
%n192.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n192.source, i32 %n192.scalar.p
%n192.scalar.first = load float, ptr addrspace(1) %n192.scalar.first.ptr, align 4
%n192.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n192.second, i32 %n192.scalar.p
%n192.scalar.second = load float, ptr addrspace(1) %n192.scalar.second.ptr, align 4
%n192.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n192.scalar.first, float %n192.scalar.second)
%n192.scalar.scalar.0 = call float @recipe.from.u1(i1 %n192.scalar.scalar.0.condition)
%n192.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n192.value, i32 %n192.scalar.p
store float %n192.scalar.scalar.0, ptr addrspace(1) %n192.scalar.output.ptr, align 4
br label %n192.scalar.step
n192.scalar.step:
%n192.scalar.next = add i32 %n192.scalar.p, %threads
br label %n192.scalar.loop
n192.scalar.done:
call void @grid_barrier(i32 %threads)%n193.source = getelementptr i8, ptr addrspace(1) %values, i64 12224
%n193.second = getelementptr i8, ptr addrspace(1) %values, i64 12288
%n193.value = getelementptr i8, ptr addrspace(1) %values, i64 12352
%n193.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2088
%n193.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n193.scalar.entry
n193.scalar.entry:
%n193.scalar.count = mul i32 %rows, 16
br label %n193.scalar.loop
n193.scalar.loop:
%n193.scalar.p = phi i32 [ %tid, %n193.scalar.entry ], [ %n193.scalar.next, %n193.scalar.step ]
%n193.scalar.more = icmp ult i32 %n193.scalar.p, %n193.scalar.count
br i1 %n193.scalar.more, label %n193.scalar.body, label %n193.scalar.done
n193.scalar.body:
%n193.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n193.source, i32 %n193.scalar.p
%n193.scalar.first = load float, ptr addrspace(1) %n193.scalar.first.ptr, align 4
%n193.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n193.second, i32 %n193.scalar.p
%n193.scalar.second = load float, ptr addrspace(1) %n193.scalar.second.ptr, align 4
%n193.scalar.scalar.0 = call float @recipe.add(float %n193.scalar.first, float %n193.scalar.second)
%n193.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n193.value, i32 %n193.scalar.p
store float %n193.scalar.scalar.0, ptr addrspace(1) %n193.scalar.output.ptr, align 4
br label %n193.scalar.step
n193.scalar.step:
%n193.scalar.next = add i32 %n193.scalar.p, %threads
br label %n193.scalar.loop
n193.scalar.done:
call void @grid_barrier(i32 %threads)%n194.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n194.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n194.value = getelementptr i8, ptr addrspace(1) %values, i64 12416
%n194.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2096
%n194.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n194.scalar.entry
n194.scalar.entry:
%n194.scalar.count = mul i32 %rows, 16
br label %n194.scalar.loop
n194.scalar.loop:
%n194.scalar.p = phi i32 [ %tid, %n194.scalar.entry ], [ %n194.scalar.next, %n194.scalar.step ]
%n194.scalar.more = icmp ult i32 %n194.scalar.p, %n194.scalar.count
br i1 %n194.scalar.more, label %n194.scalar.body, label %n194.scalar.done
n194.scalar.body:
%n194.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n194.source, i32 %n194.scalar.p
%n194.scalar.first = load float, ptr addrspace(1) %n194.scalar.first.ptr, align 4
%n194.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n194.second, i32 %n194.scalar.p
%n194.scalar.second = load float, ptr addrspace(1) %n194.scalar.second.ptr, align 4
%n194.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n194.scalar.first, float %n194.scalar.second)
%n194.scalar.scalar.0 = call float @recipe.from.u1(i1 %n194.scalar.scalar.0.condition)
%n194.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n194.value, i32 %n194.scalar.p
store float %n194.scalar.scalar.0, ptr addrspace(1) %n194.scalar.output.ptr, align 4
br label %n194.scalar.step
n194.scalar.step:
%n194.scalar.next = add i32 %n194.scalar.p, %threads
br label %n194.scalar.loop
n194.scalar.done:
call void @grid_barrier(i32 %threads)%n195.source = getelementptr i8, ptr addrspace(1) %values, i64 12352
%n195.second = getelementptr i8, ptr addrspace(1) %values, i64 12416
%n195.value = getelementptr i8, ptr addrspace(1) %values, i64 12480
%n195.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2104
%n195.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n195.scalar.entry
n195.scalar.entry:
%n195.scalar.count = mul i32 %rows, 16
br label %n195.scalar.loop
n195.scalar.loop:
%n195.scalar.p = phi i32 [ %tid, %n195.scalar.entry ], [ %n195.scalar.next, %n195.scalar.step ]
%n195.scalar.more = icmp ult i32 %n195.scalar.p, %n195.scalar.count
br i1 %n195.scalar.more, label %n195.scalar.body, label %n195.scalar.done
n195.scalar.body:
%n195.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n195.source, i32 %n195.scalar.p
%n195.scalar.first = load float, ptr addrspace(1) %n195.scalar.first.ptr, align 4
%n195.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n195.second, i32 %n195.scalar.p
%n195.scalar.second = load float, ptr addrspace(1) %n195.scalar.second.ptr, align 4
%n195.scalar.scalar.0 = call float @recipe.add(float %n195.scalar.first, float %n195.scalar.second)
%n195.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n195.value, i32 %n195.scalar.p
store float %n195.scalar.scalar.0, ptr addrspace(1) %n195.scalar.output.ptr, align 4
br label %n195.scalar.step
n195.scalar.step:
%n195.scalar.next = add i32 %n195.scalar.p, %threads
br label %n195.scalar.loop
n195.scalar.done:
call void @grid_barrier(i32 %threads)%n196.source = getelementptr i8, ptr addrspace(1) %values, i64 12480
%n196.value = getelementptr i8, ptr addrspace(1) %values, i64 12544
%n196.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2112
%n196.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n196.scalar.entry
n196.scalar.entry:
%n196.scalar.count = mul i32 %rows, 16
br label %n196.scalar.loop
n196.scalar.loop:
%n196.scalar.p = phi i32 [ %tid, %n196.scalar.entry ], [ %n196.scalar.next, %n196.scalar.step ]
%n196.scalar.more = icmp ult i32 %n196.scalar.p, %n196.scalar.count
br i1 %n196.scalar.more, label %n196.scalar.body, label %n196.scalar.done
n196.scalar.body:
%n196.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n196.source, i32 %n196.scalar.p
%n196.scalar.first = load float, ptr addrspace(1) %n196.scalar.first.ptr, align 4
%n196.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n196.scalar.first)
%n196.scalar.scalar.1 = call float @recipe.from.u1(i1 %n196.scalar.scalar.1.condition)
%n196.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n196.value, i32 %n196.scalar.p
store float %n196.scalar.scalar.1, ptr addrspace(1) %n196.scalar.output.ptr, align 4
br label %n196.scalar.step
n196.scalar.step:
%n196.scalar.next = add i32 %n196.scalar.p, %threads
br label %n196.scalar.loop
n196.scalar.done:
call void @grid_barrier(i32 %threads)%n197.source = getelementptr i8, ptr addrspace(1) %values, i64 12544
%n197.second = getelementptr i8, ptr addrspace(1) %values, i64 10240
%n197.value = getelementptr i8, ptr addrspace(1) %values, i64 12608
%n197.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2120
%n197.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n197.scalar.entry
n197.scalar.entry:
%n197.scalar.count = mul i32 %rows, 16
br label %n197.scalar.loop
n197.scalar.loop:
%n197.scalar.p = phi i32 [ %tid, %n197.scalar.entry ], [ %n197.scalar.next, %n197.scalar.step ]
%n197.scalar.more = icmp ult i32 %n197.scalar.p, %n197.scalar.count
br i1 %n197.scalar.more, label %n197.scalar.body, label %n197.scalar.done
n197.scalar.body:
%n197.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n197.source, i32 %n197.scalar.p
%n197.scalar.first = load float, ptr addrspace(1) %n197.scalar.first.ptr, align 4
%n197.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n197.second, i32 %n197.scalar.p
%n197.scalar.second = load float, ptr addrspace(1) %n197.scalar.second.ptr, align 4
%n197.scalar.scalar.0 = call float @recipe.mul(float %n197.scalar.first, float %n197.scalar.second)
%n197.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n197.value, i32 %n197.scalar.p
store float %n197.scalar.scalar.0, ptr addrspace(1) %n197.scalar.output.ptr, align 4
br label %n197.scalar.step
n197.scalar.step:
%n197.scalar.next = add i32 %n197.scalar.p, %threads
br label %n197.scalar.loop
n197.scalar.done:
call void @grid_barrier(i32 %threads)%n198.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n198.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n198.value = getelementptr i8, ptr addrspace(1) %values, i64 12672
%n198.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2128
%n198.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n198.scalar.entry
n198.scalar.entry:
%n198.scalar.count = mul i32 %rows, 16
br label %n198.scalar.loop
n198.scalar.loop:
%n198.scalar.p = phi i32 [ %tid, %n198.scalar.entry ], [ %n198.scalar.next, %n198.scalar.step ]
%n198.scalar.more = icmp ult i32 %n198.scalar.p, %n198.scalar.count
br i1 %n198.scalar.more, label %n198.scalar.body, label %n198.scalar.done
n198.scalar.body:
%n198.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n198.source, i32 %n198.scalar.p
%n198.scalar.first = load float, ptr addrspace(1) %n198.scalar.first.ptr, align 4
%n198.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n198.second, i32 %n198.scalar.p
%n198.scalar.second = load float, ptr addrspace(1) %n198.scalar.second.ptr, align 4
%n198.scalar.scalar.0 = call float @recipe.sub(float %n198.scalar.first, float %n198.scalar.second)
%n198.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n198.value, i32 %n198.scalar.p
store float %n198.scalar.scalar.0, ptr addrspace(1) %n198.scalar.output.ptr, align 4
br label %n198.scalar.step
n198.scalar.step:
%n198.scalar.next = add i32 %n198.scalar.p, %threads
br label %n198.scalar.loop
n198.scalar.done:
call void @grid_barrier(i32 %threads)%n199.source = getelementptr i8, ptr addrspace(1) %values, i64 12672
%n199.value = getelementptr i8, ptr addrspace(1) %values, i64 12736
%n199.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2136
%n199.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n199.scalar.entry
n199.scalar.entry:
%n199.scalar.count = mul i32 %rows, 16
br label %n199.scalar.loop
n199.scalar.loop:
%n199.scalar.p = phi i32 [ %tid, %n199.scalar.entry ], [ %n199.scalar.next, %n199.scalar.step ]
%n199.scalar.more = icmp ult i32 %n199.scalar.p, %n199.scalar.count
br i1 %n199.scalar.more, label %n199.scalar.body, label %n199.scalar.done
n199.scalar.body:
%n199.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n199.source, i32 %n199.scalar.p
%n199.scalar.first = load float, ptr addrspace(1) %n199.scalar.first.ptr, align 4
%n199.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n199.scalar.first, float 0x0000000000000000)
%n199.scalar.scalar.2 = call float @recipe.from.u1(i1 %n199.scalar.scalar.2.condition)
%n199.scalar.scalar.3 = call float @recipe.exp(float %n199.scalar.first)
%n199.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n199.value, i32 %n199.scalar.p
store float %n199.scalar.scalar.3, ptr addrspace(1) %n199.scalar.output.ptr, align 4
br label %n199.scalar.step
n199.scalar.step:
%n199.scalar.next = add i32 %n199.scalar.p, %threads
br label %n199.scalar.loop
n199.scalar.done:
call void @grid_barrier(i32 %threads)%n200.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n200.value = getelementptr i8, ptr addrspace(1) %values, i64 12800
%n200.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2144
%n200.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n200.scalar.entry
n200.scalar.entry:
%n200.scalar.count = mul i32 %rows, 16
br label %n200.scalar.loop
n200.scalar.loop:
%n200.scalar.p = phi i32 [ %tid, %n200.scalar.entry ], [ %n200.scalar.next, %n200.scalar.step ]
%n200.scalar.more = icmp ult i32 %n200.scalar.p, %n200.scalar.count
br i1 %n200.scalar.more, label %n200.scalar.body, label %n200.scalar.done
n200.scalar.body:
%n200.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n200.source, i32 %n200.scalar.p
%n200.scalar.first = load float, ptr addrspace(1) %n200.scalar.first.ptr, align 4
%n200.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n200.value, i32 %n200.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n200.scalar.output.ptr, align 4
br label %n200.scalar.step
n200.scalar.step:
%n200.scalar.next = add i32 %n200.scalar.p, %threads
br label %n200.scalar.loop
n200.scalar.done:
call void @grid_barrier(i32 %threads)%n201.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n201.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n201.value = getelementptr i8, ptr addrspace(1) %values, i64 12864
%n201.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2152
%n201.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n201.scalar.entry
n201.scalar.entry:
%n201.scalar.count = mul i32 %rows, 16
br label %n201.scalar.loop
n201.scalar.loop:
%n201.scalar.p = phi i32 [ %tid, %n201.scalar.entry ], [ %n201.scalar.next, %n201.scalar.step ]
%n201.scalar.more = icmp ult i32 %n201.scalar.p, %n201.scalar.count
br i1 %n201.scalar.more, label %n201.scalar.body, label %n201.scalar.done
n201.scalar.body:
%n201.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n201.source, i32 %n201.scalar.p
%n201.scalar.first = load float, ptr addrspace(1) %n201.scalar.first.ptr, align 4
%n201.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n201.second, i32 %n201.scalar.p
%n201.scalar.second = load float, ptr addrspace(1) %n201.scalar.second.ptr, align 4
%n201.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n201.scalar.first, float %n201.scalar.second)
%n201.scalar.scalar.0 = call float @recipe.from.u1(i1 %n201.scalar.scalar.0.condition)
%n201.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n201.value, i32 %n201.scalar.p
store float %n201.scalar.scalar.0, ptr addrspace(1) %n201.scalar.output.ptr, align 4
br label %n201.scalar.step
n201.scalar.step:
%n201.scalar.next = add i32 %n201.scalar.p, %threads
br label %n201.scalar.loop
n201.scalar.done:
call void @grid_barrier(i32 %threads)%n202.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n202.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n202.value = getelementptr i8, ptr addrspace(1) %values, i64 12928
%n202.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2160
%n202.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n202.scalar.entry
n202.scalar.entry:
%n202.scalar.count = mul i32 %rows, 16
br label %n202.scalar.loop
n202.scalar.loop:
%n202.scalar.p = phi i32 [ %tid, %n202.scalar.entry ], [ %n202.scalar.next, %n202.scalar.step ]
%n202.scalar.more = icmp ult i32 %n202.scalar.p, %n202.scalar.count
br i1 %n202.scalar.more, label %n202.scalar.body, label %n202.scalar.done
n202.scalar.body:
%n202.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n202.source, i32 %n202.scalar.p
%n202.scalar.first = load float, ptr addrspace(1) %n202.scalar.first.ptr, align 4
%n202.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n202.second, i32 %n202.scalar.p
%n202.scalar.second = load float, ptr addrspace(1) %n202.scalar.second.ptr, align 4
%n202.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n202.scalar.first, float %n202.scalar.second)
%n202.scalar.scalar.0 = call float @recipe.from.u1(i1 %n202.scalar.scalar.0.condition)
%n202.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n202.value, i32 %n202.scalar.p
store float %n202.scalar.scalar.0, ptr addrspace(1) %n202.scalar.output.ptr, align 4
br label %n202.scalar.step
n202.scalar.step:
%n202.scalar.next = add i32 %n202.scalar.p, %threads
br label %n202.scalar.loop
n202.scalar.done:
call void @grid_barrier(i32 %threads)%n203.source = getelementptr i8, ptr addrspace(1) %values, i64 12864
%n203.second = getelementptr i8, ptr addrspace(1) %values, i64 12928
%n203.value = getelementptr i8, ptr addrspace(1) %values, i64 12992
%n203.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2168
%n203.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n203.scalar.entry
n203.scalar.entry:
%n203.scalar.count = mul i32 %rows, 16
br label %n203.scalar.loop
n203.scalar.loop:
%n203.scalar.p = phi i32 [ %tid, %n203.scalar.entry ], [ %n203.scalar.next, %n203.scalar.step ]
%n203.scalar.more = icmp ult i32 %n203.scalar.p, %n203.scalar.count
br i1 %n203.scalar.more, label %n203.scalar.body, label %n203.scalar.done
n203.scalar.body:
%n203.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n203.source, i32 %n203.scalar.p
%n203.scalar.first = load float, ptr addrspace(1) %n203.scalar.first.ptr, align 4
%n203.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n203.second, i32 %n203.scalar.p
%n203.scalar.second = load float, ptr addrspace(1) %n203.scalar.second.ptr, align 4
%n203.scalar.scalar.0 = call float @recipe.add(float %n203.scalar.first, float %n203.scalar.second)
%n203.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n203.value, i32 %n203.scalar.p
store float %n203.scalar.scalar.0, ptr addrspace(1) %n203.scalar.output.ptr, align 4
br label %n203.scalar.step
n203.scalar.step:
%n203.scalar.next = add i32 %n203.scalar.p, %threads
br label %n203.scalar.loop
n203.scalar.done:
call void @grid_barrier(i32 %threads)%n204.source = getelementptr i8, ptr addrspace(1) %values, i64 12992
%n204.value = getelementptr i8, ptr addrspace(1) %values, i64 13056
%n204.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2176
%n204.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n204.scalar.entry
n204.scalar.entry:
%n204.scalar.count = mul i32 %rows, 16
br label %n204.scalar.loop
n204.scalar.loop:
%n204.scalar.p = phi i32 [ %tid, %n204.scalar.entry ], [ %n204.scalar.next, %n204.scalar.step ]
%n204.scalar.more = icmp ult i32 %n204.scalar.p, %n204.scalar.count
br i1 %n204.scalar.more, label %n204.scalar.body, label %n204.scalar.done
n204.scalar.body:
%n204.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n204.source, i32 %n204.scalar.p
%n204.scalar.first = load float, ptr addrspace(1) %n204.scalar.first.ptr, align 4
%n204.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n204.scalar.first)
%n204.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n204.value, i32 %n204.scalar.p
store float %n204.scalar.scalar.1, ptr addrspace(1) %n204.scalar.output.ptr, align 4
br label %n204.scalar.step
n204.scalar.step:
%n204.scalar.next = add i32 %n204.scalar.p, %threads
br label %n204.scalar.loop
n204.scalar.done:
call void @grid_barrier(i32 %threads)%n205.source = getelementptr i8, ptr addrspace(1) %values, i64 12864
%n205.second = getelementptr i8, ptr addrspace(1) %values, i64 13056
%n205.value = getelementptr i8, ptr addrspace(1) %values, i64 13120
%n205.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2184
%n205.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n205.scalar.entry
n205.scalar.entry:
%n205.scalar.count = mul i32 %rows, 16
br label %n205.scalar.loop
n205.scalar.loop:
%n205.scalar.p = phi i32 [ %tid, %n205.scalar.entry ], [ %n205.scalar.next, %n205.scalar.step ]
%n205.scalar.more = icmp ult i32 %n205.scalar.p, %n205.scalar.count
br i1 %n205.scalar.more, label %n205.scalar.body, label %n205.scalar.done
n205.scalar.body:
%n205.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n205.source, i32 %n205.scalar.p
%n205.scalar.first = load float, ptr addrspace(1) %n205.scalar.first.ptr, align 4
%n205.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n205.second, i32 %n205.scalar.p
%n205.scalar.second = load float, ptr addrspace(1) %n205.scalar.second.ptr, align 4
%n205.scalar.scalar.0 = call float @recipe.add(float %n205.scalar.first, float %n205.scalar.second)
%n205.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n205.value, i32 %n205.scalar.p
store float %n205.scalar.scalar.0, ptr addrspace(1) %n205.scalar.output.ptr, align 4
br label %n205.scalar.step
n205.scalar.step:
%n205.scalar.next = add i32 %n205.scalar.p, %threads
br label %n205.scalar.loop
n205.scalar.done:
call void @grid_barrier(i32 %threads)%n206.source = getelementptr i8, ptr addrspace(1) %values, i64 12800
%n206.second = getelementptr i8, ptr addrspace(1) %values, i64 13120
%n206.value = getelementptr i8, ptr addrspace(1) %values, i64 13184
%n206.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2192
%n206.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n206.scalar.entry
n206.scalar.entry:
%n206.scalar.count = mul i32 %rows, 16
br label %n206.scalar.loop
n206.scalar.loop:
%n206.scalar.p = phi i32 [ %tid, %n206.scalar.entry ], [ %n206.scalar.next, %n206.scalar.step ]
%n206.scalar.more = icmp ult i32 %n206.scalar.p, %n206.scalar.count
br i1 %n206.scalar.more, label %n206.scalar.body, label %n206.scalar.done
n206.scalar.body:
%n206.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n206.source, i32 %n206.scalar.p
%n206.scalar.first = load float, ptr addrspace(1) %n206.scalar.first.ptr, align 4
%n206.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n206.second, i32 %n206.scalar.p
%n206.scalar.second = load float, ptr addrspace(1) %n206.scalar.second.ptr, align 4
%n206.scalar.scalar.0 = call float @recipe.add(float %n206.scalar.first, float %n206.scalar.second)
%n206.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n206.value, i32 %n206.scalar.p
store float %n206.scalar.scalar.0, ptr addrspace(1) %n206.scalar.output.ptr, align 4
br label %n206.scalar.step
n206.scalar.step:
%n206.scalar.next = add i32 %n206.scalar.p, %threads
br label %n206.scalar.loop
n206.scalar.done:
call void @grid_barrier(i32 %threads)%n207.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n207.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n207.value = getelementptr i8, ptr addrspace(1) %values, i64 13248
%n207.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2200
%n207.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n207.scalar.entry
n207.scalar.entry:
%n207.scalar.count = mul i32 %rows, 16
br label %n207.scalar.loop
n207.scalar.loop:
%n207.scalar.p = phi i32 [ %tid, %n207.scalar.entry ], [ %n207.scalar.next, %n207.scalar.step ]
%n207.scalar.more = icmp ult i32 %n207.scalar.p, %n207.scalar.count
br i1 %n207.scalar.more, label %n207.scalar.body, label %n207.scalar.done
n207.scalar.body:
%n207.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n207.source, i32 %n207.scalar.p
%n207.scalar.first = load float, ptr addrspace(1) %n207.scalar.first.ptr, align 4
%n207.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n207.second, i32 %n207.scalar.p
%n207.scalar.second = load float, ptr addrspace(1) %n207.scalar.second.ptr, align 4
%n207.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n207.scalar.first, float %n207.scalar.second)
%n207.scalar.scalar.0 = call float @recipe.from.u1(i1 %n207.scalar.scalar.0.condition)
%n207.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n207.value, i32 %n207.scalar.p
store float %n207.scalar.scalar.0, ptr addrspace(1) %n207.scalar.output.ptr, align 4
br label %n207.scalar.step
n207.scalar.step:
%n207.scalar.next = add i32 %n207.scalar.p, %threads
br label %n207.scalar.loop
n207.scalar.done:
call void @grid_barrier(i32 %threads)%n208.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n208.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n208.value = getelementptr i8, ptr addrspace(1) %values, i64 13312
%n208.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2208
%n208.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n208.scalar.entry
n208.scalar.entry:
%n208.scalar.count = mul i32 %rows, 16
br label %n208.scalar.loop
n208.scalar.loop:
%n208.scalar.p = phi i32 [ %tid, %n208.scalar.entry ], [ %n208.scalar.next, %n208.scalar.step ]
%n208.scalar.more = icmp ult i32 %n208.scalar.p, %n208.scalar.count
br i1 %n208.scalar.more, label %n208.scalar.body, label %n208.scalar.done
n208.scalar.body:
%n208.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n208.source, i32 %n208.scalar.p
%n208.scalar.first = load float, ptr addrspace(1) %n208.scalar.first.ptr, align 4
%n208.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n208.second, i32 %n208.scalar.p
%n208.scalar.second = load float, ptr addrspace(1) %n208.scalar.second.ptr, align 4
%n208.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n208.scalar.first, float %n208.scalar.second)
%n208.scalar.scalar.0 = call float @recipe.from.u1(i1 %n208.scalar.scalar.0.condition)
%n208.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n208.value, i32 %n208.scalar.p
store float %n208.scalar.scalar.0, ptr addrspace(1) %n208.scalar.output.ptr, align 4
br label %n208.scalar.step
n208.scalar.step:
%n208.scalar.next = add i32 %n208.scalar.p, %threads
br label %n208.scalar.loop
n208.scalar.done:
call void @grid_barrier(i32 %threads)%n209.source = getelementptr i8, ptr addrspace(1) %values, i64 13248
%n209.second = getelementptr i8, ptr addrspace(1) %values, i64 13312
%n209.value = getelementptr i8, ptr addrspace(1) %values, i64 13376
%n209.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2216
%n209.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n209.scalar.entry
n209.scalar.entry:
%n209.scalar.count = mul i32 %rows, 16
br label %n209.scalar.loop
n209.scalar.loop:
%n209.scalar.p = phi i32 [ %tid, %n209.scalar.entry ], [ %n209.scalar.next, %n209.scalar.step ]
%n209.scalar.more = icmp ult i32 %n209.scalar.p, %n209.scalar.count
br i1 %n209.scalar.more, label %n209.scalar.body, label %n209.scalar.done
n209.scalar.body:
%n209.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n209.source, i32 %n209.scalar.p
%n209.scalar.first = load float, ptr addrspace(1) %n209.scalar.first.ptr, align 4
%n209.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n209.second, i32 %n209.scalar.p
%n209.scalar.second = load float, ptr addrspace(1) %n209.scalar.second.ptr, align 4
%n209.scalar.scalar.0 = call float @recipe.add(float %n209.scalar.first, float %n209.scalar.second)
%n209.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n209.value, i32 %n209.scalar.p
store float %n209.scalar.scalar.0, ptr addrspace(1) %n209.scalar.output.ptr, align 4
br label %n209.scalar.step
n209.scalar.step:
%n209.scalar.next = add i32 %n209.scalar.p, %threads
br label %n209.scalar.loop
n209.scalar.done:
call void @grid_barrier(i32 %threads)%n210.source = getelementptr i8, ptr addrspace(1) %values, i64 13376
%n210.value = getelementptr i8, ptr addrspace(1) %values, i64 13440
%n210.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2224
%n210.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n210.scalar.entry
n210.scalar.entry:
%n210.scalar.count = mul i32 %rows, 16
br label %n210.scalar.loop
n210.scalar.loop:
%n210.scalar.p = phi i32 [ %tid, %n210.scalar.entry ], [ %n210.scalar.next, %n210.scalar.step ]
%n210.scalar.more = icmp ult i32 %n210.scalar.p, %n210.scalar.count
br i1 %n210.scalar.more, label %n210.scalar.body, label %n210.scalar.done
n210.scalar.body:
%n210.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n210.source, i32 %n210.scalar.p
%n210.scalar.first = load float, ptr addrspace(1) %n210.scalar.first.ptr, align 4
%n210.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n210.scalar.first)
%n210.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n210.value, i32 %n210.scalar.p
store float %n210.scalar.scalar.1, ptr addrspace(1) %n210.scalar.output.ptr, align 4
br label %n210.scalar.step
n210.scalar.step:
%n210.scalar.next = add i32 %n210.scalar.p, %threads
br label %n210.scalar.loop
n210.scalar.done:
call void @grid_barrier(i32 %threads)%n211.source = getelementptr i8, ptr addrspace(1) %values, i64 13248
%n211.second = getelementptr i8, ptr addrspace(1) %values, i64 13440
%n211.value = getelementptr i8, ptr addrspace(1) %values, i64 13504
%n211.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2232
%n211.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n211.scalar.entry
n211.scalar.entry:
%n211.scalar.count = mul i32 %rows, 16
br label %n211.scalar.loop
n211.scalar.loop:
%n211.scalar.p = phi i32 [ %tid, %n211.scalar.entry ], [ %n211.scalar.next, %n211.scalar.step ]
%n211.scalar.more = icmp ult i32 %n211.scalar.p, %n211.scalar.count
br i1 %n211.scalar.more, label %n211.scalar.body, label %n211.scalar.done
n211.scalar.body:
%n211.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n211.source, i32 %n211.scalar.p
%n211.scalar.first = load float, ptr addrspace(1) %n211.scalar.first.ptr, align 4
%n211.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n211.second, i32 %n211.scalar.p
%n211.scalar.second = load float, ptr addrspace(1) %n211.scalar.second.ptr, align 4
%n211.scalar.scalar.0 = call float @recipe.add(float %n211.scalar.first, float %n211.scalar.second)
%n211.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n211.value, i32 %n211.scalar.p
store float %n211.scalar.scalar.0, ptr addrspace(1) %n211.scalar.output.ptr, align 4
br label %n211.scalar.step
n211.scalar.step:
%n211.scalar.next = add i32 %n211.scalar.p, %threads
br label %n211.scalar.loop
n211.scalar.done:
call void @grid_barrier(i32 %threads)%n212.source = getelementptr i8, ptr addrspace(1) %values, i64 13184
%n212.second = getelementptr i8, ptr addrspace(1) %values, i64 13504
%n212.value = getelementptr i8, ptr addrspace(1) %values, i64 13568
%n212.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2240
%n212.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n212.scalar.entry
n212.scalar.entry:
%n212.scalar.count = mul i32 %rows, 16
br label %n212.scalar.loop
n212.scalar.loop:
%n212.scalar.p = phi i32 [ %tid, %n212.scalar.entry ], [ %n212.scalar.next, %n212.scalar.step ]
%n212.scalar.more = icmp ult i32 %n212.scalar.p, %n212.scalar.count
br i1 %n212.scalar.more, label %n212.scalar.body, label %n212.scalar.done
n212.scalar.body:
%n212.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n212.source, i32 %n212.scalar.p
%n212.scalar.first = load float, ptr addrspace(1) %n212.scalar.first.ptr, align 4
%n212.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n212.second, i32 %n212.scalar.p
%n212.scalar.second = load float, ptr addrspace(1) %n212.scalar.second.ptr, align 4
%n212.scalar.scalar.0 = call float @recipe.add(float %n212.scalar.first, float %n212.scalar.second)
%n212.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n212.value, i32 %n212.scalar.p
store float %n212.scalar.scalar.0, ptr addrspace(1) %n212.scalar.output.ptr, align 4
br label %n212.scalar.step
n212.scalar.step:
%n212.scalar.next = add i32 %n212.scalar.p, %threads
br label %n212.scalar.loop
n212.scalar.done:
call void @grid_barrier(i32 %threads)%n213.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n213.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n213.value = getelementptr i8, ptr addrspace(1) %values, i64 13632
%n213.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2248
%n213.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n213.scalar.entry
n213.scalar.entry:
%n213.scalar.count = mul i32 %rows, 16
br label %n213.scalar.loop
n213.scalar.loop:
%n213.scalar.p = phi i32 [ %tid, %n213.scalar.entry ], [ %n213.scalar.next, %n213.scalar.step ]
%n213.scalar.more = icmp ult i32 %n213.scalar.p, %n213.scalar.count
br i1 %n213.scalar.more, label %n213.scalar.body, label %n213.scalar.done
n213.scalar.body:
%n213.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n213.source, i32 %n213.scalar.p
%n213.scalar.first = load float, ptr addrspace(1) %n213.scalar.first.ptr, align 4
%n213.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n213.second, i32 %n213.scalar.p
%n213.scalar.second = load float, ptr addrspace(1) %n213.scalar.second.ptr, align 4
%n213.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n213.scalar.first, float %n213.scalar.second)
%n213.scalar.scalar.0 = call float @recipe.from.u1(i1 %n213.scalar.scalar.0.condition)
%n213.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n213.value, i32 %n213.scalar.p
store float %n213.scalar.scalar.0, ptr addrspace(1) %n213.scalar.output.ptr, align 4
br label %n213.scalar.step
n213.scalar.step:
%n213.scalar.next = add i32 %n213.scalar.p, %threads
br label %n213.scalar.loop
n213.scalar.done:
call void @grid_barrier(i32 %threads)%n214.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n214.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n214.value = getelementptr i8, ptr addrspace(1) %values, i64 13696
%n214.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2256
%n214.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n214.scalar.entry
n214.scalar.entry:
%n214.scalar.count = mul i32 %rows, 16
br label %n214.scalar.loop
n214.scalar.loop:
%n214.scalar.p = phi i32 [ %tid, %n214.scalar.entry ], [ %n214.scalar.next, %n214.scalar.step ]
%n214.scalar.more = icmp ult i32 %n214.scalar.p, %n214.scalar.count
br i1 %n214.scalar.more, label %n214.scalar.body, label %n214.scalar.done
n214.scalar.body:
%n214.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n214.source, i32 %n214.scalar.p
%n214.scalar.first = load float, ptr addrspace(1) %n214.scalar.first.ptr, align 4
%n214.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n214.second, i32 %n214.scalar.p
%n214.scalar.second = load float, ptr addrspace(1) %n214.scalar.second.ptr, align 4
%n214.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n214.scalar.first, float %n214.scalar.second)
%n214.scalar.scalar.0 = call float @recipe.from.u1(i1 %n214.scalar.scalar.0.condition)
%n214.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n214.value, i32 %n214.scalar.p
store float %n214.scalar.scalar.0, ptr addrspace(1) %n214.scalar.output.ptr, align 4
br label %n214.scalar.step
n214.scalar.step:
%n214.scalar.next = add i32 %n214.scalar.p, %threads
br label %n214.scalar.loop
n214.scalar.done:
call void @grid_barrier(i32 %threads)%n215.source = getelementptr i8, ptr addrspace(1) %values, i64 13632
%n215.second = getelementptr i8, ptr addrspace(1) %values, i64 13696
%n215.value = getelementptr i8, ptr addrspace(1) %values, i64 13760
%n215.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2264
%n215.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n215.scalar.entry
n215.scalar.entry:
%n215.scalar.count = mul i32 %rows, 16
br label %n215.scalar.loop
n215.scalar.loop:
%n215.scalar.p = phi i32 [ %tid, %n215.scalar.entry ], [ %n215.scalar.next, %n215.scalar.step ]
%n215.scalar.more = icmp ult i32 %n215.scalar.p, %n215.scalar.count
br i1 %n215.scalar.more, label %n215.scalar.body, label %n215.scalar.done
n215.scalar.body:
%n215.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n215.source, i32 %n215.scalar.p
%n215.scalar.first = load float, ptr addrspace(1) %n215.scalar.first.ptr, align 4
%n215.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n215.second, i32 %n215.scalar.p
%n215.scalar.second = load float, ptr addrspace(1) %n215.scalar.second.ptr, align 4
%n215.scalar.scalar.0 = call float @recipe.add(float %n215.scalar.first, float %n215.scalar.second)
%n215.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n215.value, i32 %n215.scalar.p
store float %n215.scalar.scalar.0, ptr addrspace(1) %n215.scalar.output.ptr, align 4
br label %n215.scalar.step
n215.scalar.step:
%n215.scalar.next = add i32 %n215.scalar.p, %threads
br label %n215.scalar.loop
n215.scalar.done:
call void @grid_barrier(i32 %threads)%n216.source = getelementptr i8, ptr addrspace(1) %values, i64 13760
%n216.value = getelementptr i8, ptr addrspace(1) %values, i64 13824
%n216.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2272
%n216.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n216.scalar.entry
n216.scalar.entry:
%n216.scalar.count = mul i32 %rows, 16
br label %n216.scalar.loop
n216.scalar.loop:
%n216.scalar.p = phi i32 [ %tid, %n216.scalar.entry ], [ %n216.scalar.next, %n216.scalar.step ]
%n216.scalar.more = icmp ult i32 %n216.scalar.p, %n216.scalar.count
br i1 %n216.scalar.more, label %n216.scalar.body, label %n216.scalar.done
n216.scalar.body:
%n216.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n216.source, i32 %n216.scalar.p
%n216.scalar.first = load float, ptr addrspace(1) %n216.scalar.first.ptr, align 4
%n216.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n216.scalar.first)
%n216.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n216.value, i32 %n216.scalar.p
store float %n216.scalar.scalar.1, ptr addrspace(1) %n216.scalar.output.ptr, align 4
br label %n216.scalar.step
n216.scalar.step:
%n216.scalar.next = add i32 %n216.scalar.p, %threads
br label %n216.scalar.loop
n216.scalar.done:
call void @grid_barrier(i32 %threads)%n217.source = getelementptr i8, ptr addrspace(1) %values, i64 13632
%n217.second = getelementptr i8, ptr addrspace(1) %values, i64 13824
%n217.value = getelementptr i8, ptr addrspace(1) %values, i64 13888
%n217.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2280
%n217.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n217.scalar.entry
n217.scalar.entry:
%n217.scalar.count = mul i32 %rows, 16
br label %n217.scalar.loop
n217.scalar.loop:
%n217.scalar.p = phi i32 [ %tid, %n217.scalar.entry ], [ %n217.scalar.next, %n217.scalar.step ]
%n217.scalar.more = icmp ult i32 %n217.scalar.p, %n217.scalar.count
br i1 %n217.scalar.more, label %n217.scalar.body, label %n217.scalar.done
n217.scalar.body:
%n217.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n217.source, i32 %n217.scalar.p
%n217.scalar.first = load float, ptr addrspace(1) %n217.scalar.first.ptr, align 4
%n217.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n217.second, i32 %n217.scalar.p
%n217.scalar.second = load float, ptr addrspace(1) %n217.scalar.second.ptr, align 4
%n217.scalar.scalar.0 = call float @recipe.add(float %n217.scalar.first, float %n217.scalar.second)
%n217.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n217.value, i32 %n217.scalar.p
store float %n217.scalar.scalar.0, ptr addrspace(1) %n217.scalar.output.ptr, align 4
br label %n217.scalar.step
n217.scalar.step:
%n217.scalar.next = add i32 %n217.scalar.p, %threads
br label %n217.scalar.loop
n217.scalar.done:
call void @grid_barrier(i32 %threads)%n218.source = getelementptr i8, ptr addrspace(1) %values, i64 13568
%n218.second = getelementptr i8, ptr addrspace(1) %values, i64 13888
%n218.value = getelementptr i8, ptr addrspace(1) %values, i64 13952
%n218.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2288
%n218.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n218.scalar.entry
n218.scalar.entry:
%n218.scalar.count = mul i32 %rows, 16
br label %n218.scalar.loop
n218.scalar.loop:
%n218.scalar.p = phi i32 [ %tid, %n218.scalar.entry ], [ %n218.scalar.next, %n218.scalar.step ]
%n218.scalar.more = icmp ult i32 %n218.scalar.p, %n218.scalar.count
br i1 %n218.scalar.more, label %n218.scalar.body, label %n218.scalar.done
n218.scalar.body:
%n218.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n218.source, i32 %n218.scalar.p
%n218.scalar.first = load float, ptr addrspace(1) %n218.scalar.first.ptr, align 4
%n218.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n218.second, i32 %n218.scalar.p
%n218.scalar.second = load float, ptr addrspace(1) %n218.scalar.second.ptr, align 4
%n218.scalar.scalar.0 = call float @recipe.add(float %n218.scalar.first, float %n218.scalar.second)
%n218.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n218.value, i32 %n218.scalar.p
store float %n218.scalar.scalar.0, ptr addrspace(1) %n218.scalar.output.ptr, align 4
br label %n218.scalar.step
n218.scalar.step:
%n218.scalar.next = add i32 %n218.scalar.p, %threads
br label %n218.scalar.loop
n218.scalar.done:
call void @grid_barrier(i32 %threads)%n219.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n219.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n219.value = getelementptr i8, ptr addrspace(1) %values, i64 14016
%n219.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2296
%n219.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n219.scalar.entry
n219.scalar.entry:
%n219.scalar.count = mul i32 %rows, 16
br label %n219.scalar.loop
n219.scalar.loop:
%n219.scalar.p = phi i32 [ %tid, %n219.scalar.entry ], [ %n219.scalar.next, %n219.scalar.step ]
%n219.scalar.more = icmp ult i32 %n219.scalar.p, %n219.scalar.count
br i1 %n219.scalar.more, label %n219.scalar.body, label %n219.scalar.done
n219.scalar.body:
%n219.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n219.source, i32 %n219.scalar.p
%n219.scalar.first = load float, ptr addrspace(1) %n219.scalar.first.ptr, align 4
%n219.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n219.second, i32 %n219.scalar.p
%n219.scalar.second = load float, ptr addrspace(1) %n219.scalar.second.ptr, align 4
%n219.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n219.scalar.first, float %n219.scalar.second)
%n219.scalar.scalar.0 = call float @recipe.from.u1(i1 %n219.scalar.scalar.0.condition)
%n219.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n219.value, i32 %n219.scalar.p
store float %n219.scalar.scalar.0, ptr addrspace(1) %n219.scalar.output.ptr, align 4
br label %n219.scalar.step
n219.scalar.step:
%n219.scalar.next = add i32 %n219.scalar.p, %threads
br label %n219.scalar.loop
n219.scalar.done:
call void @grid_barrier(i32 %threads)%n220.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n220.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n220.value = getelementptr i8, ptr addrspace(1) %values, i64 14080
%n220.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2304
%n220.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n220.scalar.entry
n220.scalar.entry:
%n220.scalar.count = mul i32 %rows, 16
br label %n220.scalar.loop
n220.scalar.loop:
%n220.scalar.p = phi i32 [ %tid, %n220.scalar.entry ], [ %n220.scalar.next, %n220.scalar.step ]
%n220.scalar.more = icmp ult i32 %n220.scalar.p, %n220.scalar.count
br i1 %n220.scalar.more, label %n220.scalar.body, label %n220.scalar.done
n220.scalar.body:
%n220.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n220.source, i32 %n220.scalar.p
%n220.scalar.first = load float, ptr addrspace(1) %n220.scalar.first.ptr, align 4
%n220.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n220.second, i32 %n220.scalar.p
%n220.scalar.second = load float, ptr addrspace(1) %n220.scalar.second.ptr, align 4
%n220.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n220.scalar.first, float %n220.scalar.second)
%n220.scalar.scalar.0 = call float @recipe.from.u1(i1 %n220.scalar.scalar.0.condition)
%n220.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n220.value, i32 %n220.scalar.p
store float %n220.scalar.scalar.0, ptr addrspace(1) %n220.scalar.output.ptr, align 4
br label %n220.scalar.step
n220.scalar.step:
%n220.scalar.next = add i32 %n220.scalar.p, %threads
br label %n220.scalar.loop
n220.scalar.done:
call void @grid_barrier(i32 %threads)%n221.source = getelementptr i8, ptr addrspace(1) %values, i64 14016
%n221.second = getelementptr i8, ptr addrspace(1) %values, i64 14080
%n221.value = getelementptr i8, ptr addrspace(1) %values, i64 14144
%n221.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2312
%n221.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n221.scalar.entry
n221.scalar.entry:
%n221.scalar.count = mul i32 %rows, 16
br label %n221.scalar.loop
n221.scalar.loop:
%n221.scalar.p = phi i32 [ %tid, %n221.scalar.entry ], [ %n221.scalar.next, %n221.scalar.step ]
%n221.scalar.more = icmp ult i32 %n221.scalar.p, %n221.scalar.count
br i1 %n221.scalar.more, label %n221.scalar.body, label %n221.scalar.done
n221.scalar.body:
%n221.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n221.source, i32 %n221.scalar.p
%n221.scalar.first = load float, ptr addrspace(1) %n221.scalar.first.ptr, align 4
%n221.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n221.second, i32 %n221.scalar.p
%n221.scalar.second = load float, ptr addrspace(1) %n221.scalar.second.ptr, align 4
%n221.scalar.scalar.0 = call float @recipe.add(float %n221.scalar.first, float %n221.scalar.second)
%n221.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n221.value, i32 %n221.scalar.p
store float %n221.scalar.scalar.0, ptr addrspace(1) %n221.scalar.output.ptr, align 4
br label %n221.scalar.step
n221.scalar.step:
%n221.scalar.next = add i32 %n221.scalar.p, %threads
br label %n221.scalar.loop
n221.scalar.done:
call void @grid_barrier(i32 %threads)%n222.source = getelementptr i8, ptr addrspace(1) %values, i64 14144
%n222.value = getelementptr i8, ptr addrspace(1) %values, i64 14208
%n222.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2320
%n222.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n222.scalar.entry
n222.scalar.entry:
%n222.scalar.count = mul i32 %rows, 16
br label %n222.scalar.loop
n222.scalar.loop:
%n222.scalar.p = phi i32 [ %tid, %n222.scalar.entry ], [ %n222.scalar.next, %n222.scalar.step ]
%n222.scalar.more = icmp ult i32 %n222.scalar.p, %n222.scalar.count
br i1 %n222.scalar.more, label %n222.scalar.body, label %n222.scalar.done
n222.scalar.body:
%n222.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n222.source, i32 %n222.scalar.p
%n222.scalar.first = load float, ptr addrspace(1) %n222.scalar.first.ptr, align 4
%n222.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n222.scalar.first)
%n222.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n222.value, i32 %n222.scalar.p
store float %n222.scalar.scalar.1, ptr addrspace(1) %n222.scalar.output.ptr, align 4
br label %n222.scalar.step
n222.scalar.step:
%n222.scalar.next = add i32 %n222.scalar.p, %threads
br label %n222.scalar.loop
n222.scalar.done:
call void @grid_barrier(i32 %threads)%n223.source = getelementptr i8, ptr addrspace(1) %values, i64 14016
%n223.second = getelementptr i8, ptr addrspace(1) %values, i64 14208
%n223.value = getelementptr i8, ptr addrspace(1) %values, i64 14272
%n223.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2328
%n223.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n223.scalar.entry
n223.scalar.entry:
%n223.scalar.count = mul i32 %rows, 16
br label %n223.scalar.loop
n223.scalar.loop:
%n223.scalar.p = phi i32 [ %tid, %n223.scalar.entry ], [ %n223.scalar.next, %n223.scalar.step ]
%n223.scalar.more = icmp ult i32 %n223.scalar.p, %n223.scalar.count
br i1 %n223.scalar.more, label %n223.scalar.body, label %n223.scalar.done
n223.scalar.body:
%n223.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n223.source, i32 %n223.scalar.p
%n223.scalar.first = load float, ptr addrspace(1) %n223.scalar.first.ptr, align 4
%n223.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n223.second, i32 %n223.scalar.p
%n223.scalar.second = load float, ptr addrspace(1) %n223.scalar.second.ptr, align 4
%n223.scalar.scalar.0 = call float @recipe.add(float %n223.scalar.first, float %n223.scalar.second)
%n223.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n223.value, i32 %n223.scalar.p
store float %n223.scalar.scalar.0, ptr addrspace(1) %n223.scalar.output.ptr, align 4
br label %n223.scalar.step
n223.scalar.step:
%n223.scalar.next = add i32 %n223.scalar.p, %threads
br label %n223.scalar.loop
n223.scalar.done:
call void @grid_barrier(i32 %threads)%n224.source = getelementptr i8, ptr addrspace(1) %values, i64 13952
%n224.second = getelementptr i8, ptr addrspace(1) %values, i64 14272
%n224.value = getelementptr i8, ptr addrspace(1) %values, i64 14336
%n224.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2336
%n224.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n224.scalar.entry
n224.scalar.entry:
%n224.scalar.count = mul i32 %rows, 16
br label %n224.scalar.loop
n224.scalar.loop:
%n224.scalar.p = phi i32 [ %tid, %n224.scalar.entry ], [ %n224.scalar.next, %n224.scalar.step ]
%n224.scalar.more = icmp ult i32 %n224.scalar.p, %n224.scalar.count
br i1 %n224.scalar.more, label %n224.scalar.body, label %n224.scalar.done
n224.scalar.body:
%n224.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n224.source, i32 %n224.scalar.p
%n224.scalar.first = load float, ptr addrspace(1) %n224.scalar.first.ptr, align 4
%n224.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n224.second, i32 %n224.scalar.p
%n224.scalar.second = load float, ptr addrspace(1) %n224.scalar.second.ptr, align 4
%n224.scalar.scalar.0 = call float @recipe.add(float %n224.scalar.first, float %n224.scalar.second)
%n224.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n224.value, i32 %n224.scalar.p
store float %n224.scalar.scalar.0, ptr addrspace(1) %n224.scalar.output.ptr, align 4
br label %n224.scalar.step
n224.scalar.step:
%n224.scalar.next = add i32 %n224.scalar.p, %threads
br label %n224.scalar.loop
n224.scalar.done:
call void @grid_barrier(i32 %threads)%n225.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n225.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n225.value = getelementptr i8, ptr addrspace(1) %values, i64 14400
%n225.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2344
%n225.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n225.scalar.entry
n225.scalar.entry:
%n225.scalar.count = mul i32 %rows, 16
br label %n225.scalar.loop
n225.scalar.loop:
%n225.scalar.p = phi i32 [ %tid, %n225.scalar.entry ], [ %n225.scalar.next, %n225.scalar.step ]
%n225.scalar.more = icmp ult i32 %n225.scalar.p, %n225.scalar.count
br i1 %n225.scalar.more, label %n225.scalar.body, label %n225.scalar.done
n225.scalar.body:
%n225.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n225.source, i32 %n225.scalar.p
%n225.scalar.first = load float, ptr addrspace(1) %n225.scalar.first.ptr, align 4
%n225.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n225.second, i32 %n225.scalar.p
%n225.scalar.second = load float, ptr addrspace(1) %n225.scalar.second.ptr, align 4
%n225.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n225.scalar.first, float %n225.scalar.second)
%n225.scalar.scalar.0 = call float @recipe.from.u1(i1 %n225.scalar.scalar.0.condition)
%n225.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n225.value, i32 %n225.scalar.p
store float %n225.scalar.scalar.0, ptr addrspace(1) %n225.scalar.output.ptr, align 4
br label %n225.scalar.step
n225.scalar.step:
%n225.scalar.next = add i32 %n225.scalar.p, %threads
br label %n225.scalar.loop
n225.scalar.done:
call void @grid_barrier(i32 %threads)%n226.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n226.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n226.value = getelementptr i8, ptr addrspace(1) %values, i64 14464
%n226.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2352
%n226.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n226.scalar.entry
n226.scalar.entry:
%n226.scalar.count = mul i32 %rows, 16
br label %n226.scalar.loop
n226.scalar.loop:
%n226.scalar.p = phi i32 [ %tid, %n226.scalar.entry ], [ %n226.scalar.next, %n226.scalar.step ]
%n226.scalar.more = icmp ult i32 %n226.scalar.p, %n226.scalar.count
br i1 %n226.scalar.more, label %n226.scalar.body, label %n226.scalar.done
n226.scalar.body:
%n226.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n226.source, i32 %n226.scalar.p
%n226.scalar.first = load float, ptr addrspace(1) %n226.scalar.first.ptr, align 4
%n226.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n226.second, i32 %n226.scalar.p
%n226.scalar.second = load float, ptr addrspace(1) %n226.scalar.second.ptr, align 4
%n226.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n226.scalar.first, float %n226.scalar.second)
%n226.scalar.scalar.0 = call float @recipe.from.u1(i1 %n226.scalar.scalar.0.condition)
%n226.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n226.value, i32 %n226.scalar.p
store float %n226.scalar.scalar.0, ptr addrspace(1) %n226.scalar.output.ptr, align 4
br label %n226.scalar.step
n226.scalar.step:
%n226.scalar.next = add i32 %n226.scalar.p, %threads
br label %n226.scalar.loop
n226.scalar.done:
call void @grid_barrier(i32 %threads)%n227.source = getelementptr i8, ptr addrspace(1) %values, i64 14400
%n227.second = getelementptr i8, ptr addrspace(1) %values, i64 14464
%n227.value = getelementptr i8, ptr addrspace(1) %values, i64 14528
%n227.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2360
%n227.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n227.scalar.entry
n227.scalar.entry:
%n227.scalar.count = mul i32 %rows, 16
br label %n227.scalar.loop
n227.scalar.loop:
%n227.scalar.p = phi i32 [ %tid, %n227.scalar.entry ], [ %n227.scalar.next, %n227.scalar.step ]
%n227.scalar.more = icmp ult i32 %n227.scalar.p, %n227.scalar.count
br i1 %n227.scalar.more, label %n227.scalar.body, label %n227.scalar.done
n227.scalar.body:
%n227.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n227.source, i32 %n227.scalar.p
%n227.scalar.first = load float, ptr addrspace(1) %n227.scalar.first.ptr, align 4
%n227.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n227.second, i32 %n227.scalar.p
%n227.scalar.second = load float, ptr addrspace(1) %n227.scalar.second.ptr, align 4
%n227.scalar.scalar.0 = call float @recipe.add(float %n227.scalar.first, float %n227.scalar.second)
%n227.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n227.value, i32 %n227.scalar.p
store float %n227.scalar.scalar.0, ptr addrspace(1) %n227.scalar.output.ptr, align 4
br label %n227.scalar.step
n227.scalar.step:
%n227.scalar.next = add i32 %n227.scalar.p, %threads
br label %n227.scalar.loop
n227.scalar.done:
call void @grid_barrier(i32 %threads)%n228.source = getelementptr i8, ptr addrspace(1) %values, i64 14528
%n228.value = getelementptr i8, ptr addrspace(1) %values, i64 14592
%n228.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2368
%n228.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n228.scalar.entry
n228.scalar.entry:
%n228.scalar.count = mul i32 %rows, 16
br label %n228.scalar.loop
n228.scalar.loop:
%n228.scalar.p = phi i32 [ %tid, %n228.scalar.entry ], [ %n228.scalar.next, %n228.scalar.step ]
%n228.scalar.more = icmp ult i32 %n228.scalar.p, %n228.scalar.count
br i1 %n228.scalar.more, label %n228.scalar.body, label %n228.scalar.done
n228.scalar.body:
%n228.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n228.source, i32 %n228.scalar.p
%n228.scalar.first = load float, ptr addrspace(1) %n228.scalar.first.ptr, align 4
%n228.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n228.scalar.first)
%n228.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n228.value, i32 %n228.scalar.p
store float %n228.scalar.scalar.1, ptr addrspace(1) %n228.scalar.output.ptr, align 4
br label %n228.scalar.step
n228.scalar.step:
%n228.scalar.next = add i32 %n228.scalar.p, %threads
br label %n228.scalar.loop
n228.scalar.done:
call void @grid_barrier(i32 %threads)%n229.source = getelementptr i8, ptr addrspace(1) %values, i64 14400
%n229.second = getelementptr i8, ptr addrspace(1) %values, i64 14592
%n229.value = getelementptr i8, ptr addrspace(1) %values, i64 14656
%n229.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2376
%n229.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n229.scalar.entry
n229.scalar.entry:
%n229.scalar.count = mul i32 %rows, 16
br label %n229.scalar.loop
n229.scalar.loop:
%n229.scalar.p = phi i32 [ %tid, %n229.scalar.entry ], [ %n229.scalar.next, %n229.scalar.step ]
%n229.scalar.more = icmp ult i32 %n229.scalar.p, %n229.scalar.count
br i1 %n229.scalar.more, label %n229.scalar.body, label %n229.scalar.done
n229.scalar.body:
%n229.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n229.source, i32 %n229.scalar.p
%n229.scalar.first = load float, ptr addrspace(1) %n229.scalar.first.ptr, align 4
%n229.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n229.second, i32 %n229.scalar.p
%n229.scalar.second = load float, ptr addrspace(1) %n229.scalar.second.ptr, align 4
%n229.scalar.scalar.0 = call float @recipe.add(float %n229.scalar.first, float %n229.scalar.second)
%n229.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n229.value, i32 %n229.scalar.p
store float %n229.scalar.scalar.0, ptr addrspace(1) %n229.scalar.output.ptr, align 4
br label %n229.scalar.step
n229.scalar.step:
%n229.scalar.next = add i32 %n229.scalar.p, %threads
br label %n229.scalar.loop
n229.scalar.done:
call void @grid_barrier(i32 %threads)%n230.source = getelementptr i8, ptr addrspace(1) %values, i64 14336
%n230.second = getelementptr i8, ptr addrspace(1) %values, i64 14656
%n230.value = getelementptr i8, ptr addrspace(1) %values, i64 14720
%n230.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2384
%n230.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n230.scalar.entry
n230.scalar.entry:
%n230.scalar.count = mul i32 %rows, 16
br label %n230.scalar.loop
n230.scalar.loop:
%n230.scalar.p = phi i32 [ %tid, %n230.scalar.entry ], [ %n230.scalar.next, %n230.scalar.step ]
%n230.scalar.more = icmp ult i32 %n230.scalar.p, %n230.scalar.count
br i1 %n230.scalar.more, label %n230.scalar.body, label %n230.scalar.done
n230.scalar.body:
%n230.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n230.source, i32 %n230.scalar.p
%n230.scalar.first = load float, ptr addrspace(1) %n230.scalar.first.ptr, align 4
%n230.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n230.second, i32 %n230.scalar.p
%n230.scalar.second = load float, ptr addrspace(1) %n230.scalar.second.ptr, align 4
%n230.scalar.scalar.0 = call float @recipe.add(float %n230.scalar.first, float %n230.scalar.second)
%n230.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n230.value, i32 %n230.scalar.p
store float %n230.scalar.scalar.0, ptr addrspace(1) %n230.scalar.output.ptr, align 4
br label %n230.scalar.step
n230.scalar.step:
%n230.scalar.next = add i32 %n230.scalar.p, %threads
br label %n230.scalar.loop
n230.scalar.done:
call void @grid_barrier(i32 %threads)%n231.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n231.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n231.value = getelementptr i8, ptr addrspace(1) %values, i64 14784
%n231.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2392
%n231.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n231.scalar.entry
n231.scalar.entry:
%n231.scalar.count = mul i32 %rows, 16
br label %n231.scalar.loop
n231.scalar.loop:
%n231.scalar.p = phi i32 [ %tid, %n231.scalar.entry ], [ %n231.scalar.next, %n231.scalar.step ]
%n231.scalar.more = icmp ult i32 %n231.scalar.p, %n231.scalar.count
br i1 %n231.scalar.more, label %n231.scalar.body, label %n231.scalar.done
n231.scalar.body:
%n231.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n231.source, i32 %n231.scalar.p
%n231.scalar.first = load float, ptr addrspace(1) %n231.scalar.first.ptr, align 4
%n231.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n231.second, i32 %n231.scalar.p
%n231.scalar.second = load float, ptr addrspace(1) %n231.scalar.second.ptr, align 4
%n231.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n231.scalar.first, float %n231.scalar.second)
%n231.scalar.scalar.0 = call float @recipe.from.u1(i1 %n231.scalar.scalar.0.condition)
%n231.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n231.value, i32 %n231.scalar.p
store float %n231.scalar.scalar.0, ptr addrspace(1) %n231.scalar.output.ptr, align 4
br label %n231.scalar.step
n231.scalar.step:
%n231.scalar.next = add i32 %n231.scalar.p, %threads
br label %n231.scalar.loop
n231.scalar.done:
call void @grid_barrier(i32 %threads)%n232.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n232.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n232.value = getelementptr i8, ptr addrspace(1) %values, i64 14848
%n232.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2400
%n232.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n232.scalar.entry
n232.scalar.entry:
%n232.scalar.count = mul i32 %rows, 16
br label %n232.scalar.loop
n232.scalar.loop:
%n232.scalar.p = phi i32 [ %tid, %n232.scalar.entry ], [ %n232.scalar.next, %n232.scalar.step ]
%n232.scalar.more = icmp ult i32 %n232.scalar.p, %n232.scalar.count
br i1 %n232.scalar.more, label %n232.scalar.body, label %n232.scalar.done
n232.scalar.body:
%n232.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n232.source, i32 %n232.scalar.p
%n232.scalar.first = load float, ptr addrspace(1) %n232.scalar.first.ptr, align 4
%n232.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n232.second, i32 %n232.scalar.p
%n232.scalar.second = load float, ptr addrspace(1) %n232.scalar.second.ptr, align 4
%n232.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n232.scalar.first, float %n232.scalar.second)
%n232.scalar.scalar.0 = call float @recipe.from.u1(i1 %n232.scalar.scalar.0.condition)
%n232.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n232.value, i32 %n232.scalar.p
store float %n232.scalar.scalar.0, ptr addrspace(1) %n232.scalar.output.ptr, align 4
br label %n232.scalar.step
n232.scalar.step:
%n232.scalar.next = add i32 %n232.scalar.p, %threads
br label %n232.scalar.loop
n232.scalar.done:
call void @grid_barrier(i32 %threads)%n233.source = getelementptr i8, ptr addrspace(1) %values, i64 14784
%n233.second = getelementptr i8, ptr addrspace(1) %values, i64 14848
%n233.value = getelementptr i8, ptr addrspace(1) %values, i64 14912
%n233.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2408
%n233.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n233.scalar.entry
n233.scalar.entry:
%n233.scalar.count = mul i32 %rows, 16
br label %n233.scalar.loop
n233.scalar.loop:
%n233.scalar.p = phi i32 [ %tid, %n233.scalar.entry ], [ %n233.scalar.next, %n233.scalar.step ]
%n233.scalar.more = icmp ult i32 %n233.scalar.p, %n233.scalar.count
br i1 %n233.scalar.more, label %n233.scalar.body, label %n233.scalar.done
n233.scalar.body:
%n233.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n233.source, i32 %n233.scalar.p
%n233.scalar.first = load float, ptr addrspace(1) %n233.scalar.first.ptr, align 4
%n233.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n233.second, i32 %n233.scalar.p
%n233.scalar.second = load float, ptr addrspace(1) %n233.scalar.second.ptr, align 4
%n233.scalar.scalar.0 = call float @recipe.add(float %n233.scalar.first, float %n233.scalar.second)
%n233.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n233.value, i32 %n233.scalar.p
store float %n233.scalar.scalar.0, ptr addrspace(1) %n233.scalar.output.ptr, align 4
br label %n233.scalar.step
n233.scalar.step:
%n233.scalar.next = add i32 %n233.scalar.p, %threads
br label %n233.scalar.loop
n233.scalar.done:
call void @grid_barrier(i32 %threads)%n234.source = getelementptr i8, ptr addrspace(1) %values, i64 14912
%n234.value = getelementptr i8, ptr addrspace(1) %values, i64 14976
%n234.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2416
%n234.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n234.scalar.entry
n234.scalar.entry:
%n234.scalar.count = mul i32 %rows, 16
br label %n234.scalar.loop
n234.scalar.loop:
%n234.scalar.p = phi i32 [ %tid, %n234.scalar.entry ], [ %n234.scalar.next, %n234.scalar.step ]
%n234.scalar.more = icmp ult i32 %n234.scalar.p, %n234.scalar.count
br i1 %n234.scalar.more, label %n234.scalar.body, label %n234.scalar.done
n234.scalar.body:
%n234.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n234.source, i32 %n234.scalar.p
%n234.scalar.first = load float, ptr addrspace(1) %n234.scalar.first.ptr, align 4
%n234.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n234.scalar.first)
%n234.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n234.value, i32 %n234.scalar.p
store float %n234.scalar.scalar.1, ptr addrspace(1) %n234.scalar.output.ptr, align 4
br label %n234.scalar.step
n234.scalar.step:
%n234.scalar.next = add i32 %n234.scalar.p, %threads
br label %n234.scalar.loop
n234.scalar.done:
call void @grid_barrier(i32 %threads)%n235.source = getelementptr i8, ptr addrspace(1) %values, i64 14784
%n235.second = getelementptr i8, ptr addrspace(1) %values, i64 14976
%n235.value = getelementptr i8, ptr addrspace(1) %values, i64 15040
%n235.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2424
%n235.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n235.scalar.entry
n235.scalar.entry:
%n235.scalar.count = mul i32 %rows, 16
br label %n235.scalar.loop
n235.scalar.loop:
%n235.scalar.p = phi i32 [ %tid, %n235.scalar.entry ], [ %n235.scalar.next, %n235.scalar.step ]
%n235.scalar.more = icmp ult i32 %n235.scalar.p, %n235.scalar.count
br i1 %n235.scalar.more, label %n235.scalar.body, label %n235.scalar.done
n235.scalar.body:
%n235.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n235.source, i32 %n235.scalar.p
%n235.scalar.first = load float, ptr addrspace(1) %n235.scalar.first.ptr, align 4
%n235.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n235.second, i32 %n235.scalar.p
%n235.scalar.second = load float, ptr addrspace(1) %n235.scalar.second.ptr, align 4
%n235.scalar.scalar.0 = call float @recipe.add(float %n235.scalar.first, float %n235.scalar.second)
%n235.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n235.value, i32 %n235.scalar.p
store float %n235.scalar.scalar.0, ptr addrspace(1) %n235.scalar.output.ptr, align 4
br label %n235.scalar.step
n235.scalar.step:
%n235.scalar.next = add i32 %n235.scalar.p, %threads
br label %n235.scalar.loop
n235.scalar.done:
call void @grid_barrier(i32 %threads)%n236.source = getelementptr i8, ptr addrspace(1) %values, i64 14720
%n236.second = getelementptr i8, ptr addrspace(1) %values, i64 15040
%n236.value = getelementptr i8, ptr addrspace(1) %values, i64 15104
%n236.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2432
%n236.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n236.scalar.entry
n236.scalar.entry:
%n236.scalar.count = mul i32 %rows, 16
br label %n236.scalar.loop
n236.scalar.loop:
%n236.scalar.p = phi i32 [ %tid, %n236.scalar.entry ], [ %n236.scalar.next, %n236.scalar.step ]
%n236.scalar.more = icmp ult i32 %n236.scalar.p, %n236.scalar.count
br i1 %n236.scalar.more, label %n236.scalar.body, label %n236.scalar.done
n236.scalar.body:
%n236.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n236.source, i32 %n236.scalar.p
%n236.scalar.first = load float, ptr addrspace(1) %n236.scalar.first.ptr, align 4
%n236.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n236.second, i32 %n236.scalar.p
%n236.scalar.second = load float, ptr addrspace(1) %n236.scalar.second.ptr, align 4
%n236.scalar.scalar.0 = call float @recipe.add(float %n236.scalar.first, float %n236.scalar.second)
%n236.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n236.value, i32 %n236.scalar.p
store float %n236.scalar.scalar.0, ptr addrspace(1) %n236.scalar.output.ptr, align 4
br label %n236.scalar.step
n236.scalar.step:
%n236.scalar.next = add i32 %n236.scalar.p, %threads
br label %n236.scalar.loop
n236.scalar.done:
call void @grid_barrier(i32 %threads)%n237.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n237.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n237.value = getelementptr i8, ptr addrspace(1) %values, i64 15168
%n237.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2440
%n237.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n237.scalar.entry
n237.scalar.entry:
%n237.scalar.count = mul i32 %rows, 16
br label %n237.scalar.loop
n237.scalar.loop:
%n237.scalar.p = phi i32 [ %tid, %n237.scalar.entry ], [ %n237.scalar.next, %n237.scalar.step ]
%n237.scalar.more = icmp ult i32 %n237.scalar.p, %n237.scalar.count
br i1 %n237.scalar.more, label %n237.scalar.body, label %n237.scalar.done
n237.scalar.body:
%n237.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n237.source, i32 %n237.scalar.p
%n237.scalar.first = load float, ptr addrspace(1) %n237.scalar.first.ptr, align 4
%n237.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n237.second, i32 %n237.scalar.p
%n237.scalar.second = load float, ptr addrspace(1) %n237.scalar.second.ptr, align 4
%n237.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n237.scalar.first, float %n237.scalar.second)
%n237.scalar.scalar.0 = call float @recipe.from.u1(i1 %n237.scalar.scalar.0.condition)
%n237.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n237.value, i32 %n237.scalar.p
store float %n237.scalar.scalar.0, ptr addrspace(1) %n237.scalar.output.ptr, align 4
br label %n237.scalar.step
n237.scalar.step:
%n237.scalar.next = add i32 %n237.scalar.p, %threads
br label %n237.scalar.loop
n237.scalar.done:
call void @grid_barrier(i32 %threads)%n238.source = getelementptr i8, ptr addrspace(1) %values, i64 15104
%n238.second = getelementptr i8, ptr addrspace(1) %values, i64 15168
%n238.value = getelementptr i8, ptr addrspace(1) %values, i64 15232
%n238.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2448
%n238.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n238.scalar.entry
n238.scalar.entry:
%n238.scalar.count = mul i32 %rows, 16
br label %n238.scalar.loop
n238.scalar.loop:
%n238.scalar.p = phi i32 [ %tid, %n238.scalar.entry ], [ %n238.scalar.next, %n238.scalar.step ]
%n238.scalar.more = icmp ult i32 %n238.scalar.p, %n238.scalar.count
br i1 %n238.scalar.more, label %n238.scalar.body, label %n238.scalar.done
n238.scalar.body:
%n238.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n238.source, i32 %n238.scalar.p
%n238.scalar.first = load float, ptr addrspace(1) %n238.scalar.first.ptr, align 4
%n238.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n238.second, i32 %n238.scalar.p
%n238.scalar.second = load float, ptr addrspace(1) %n238.scalar.second.ptr, align 4
%n238.scalar.scalar.0 = call float @recipe.add(float %n238.scalar.first, float %n238.scalar.second)
%n238.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n238.value, i32 %n238.scalar.p
store float %n238.scalar.scalar.0, ptr addrspace(1) %n238.scalar.output.ptr, align 4
br label %n238.scalar.step
n238.scalar.step:
%n238.scalar.next = add i32 %n238.scalar.p, %threads
br label %n238.scalar.loop
n238.scalar.done:
call void @grid_barrier(i32 %threads)%n239.source = getelementptr i8, ptr addrspace(1) %values, i64 15232
%n239.value = getelementptr i8, ptr addrspace(1) %values, i64 15296
%n239.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2456
%n239.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n239.scalar.entry
n239.scalar.entry:
%n239.scalar.count = mul i32 %rows, 16
br label %n239.scalar.loop
n239.scalar.loop:
%n239.scalar.p = phi i32 [ %tid, %n239.scalar.entry ], [ %n239.scalar.next, %n239.scalar.step ]
%n239.scalar.more = icmp ult i32 %n239.scalar.p, %n239.scalar.count
br i1 %n239.scalar.more, label %n239.scalar.body, label %n239.scalar.done
n239.scalar.body:
%n239.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n239.source, i32 %n239.scalar.p
%n239.scalar.first = load float, ptr addrspace(1) %n239.scalar.first.ptr, align 4
%n239.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n239.scalar.first)
%n239.scalar.scalar.1 = call float @recipe.from.u1(i1 %n239.scalar.scalar.1.condition)
%n239.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n239.value, i32 %n239.scalar.p
store float %n239.scalar.scalar.1, ptr addrspace(1) %n239.scalar.output.ptr, align 4
br label %n239.scalar.step
n239.scalar.step:
%n239.scalar.next = add i32 %n239.scalar.p, %threads
br label %n239.scalar.loop
n239.scalar.done:
call void @grid_barrier(i32 %threads)%n240.source = getelementptr i8, ptr addrspace(1) %values, i64 15296
%n240.second = getelementptr i8, ptr addrspace(1) %values, i64 12736
%n240.value = getelementptr i8, ptr addrspace(1) %values, i64 15360
%n240.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2464
%n240.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n240.scalar.entry
n240.scalar.entry:
%n240.scalar.count = mul i32 %rows, 16
br label %n240.scalar.loop
n240.scalar.loop:
%n240.scalar.p = phi i32 [ %tid, %n240.scalar.entry ], [ %n240.scalar.next, %n240.scalar.step ]
%n240.scalar.more = icmp ult i32 %n240.scalar.p, %n240.scalar.count
br i1 %n240.scalar.more, label %n240.scalar.body, label %n240.scalar.done
n240.scalar.body:
%n240.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n240.source, i32 %n240.scalar.p
%n240.scalar.first = load float, ptr addrspace(1) %n240.scalar.first.ptr, align 4
%n240.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n240.second, i32 %n240.scalar.p
%n240.scalar.second = load float, ptr addrspace(1) %n240.scalar.second.ptr, align 4
%n240.scalar.scalar.0 = call float @recipe.mul(float %n240.scalar.first, float %n240.scalar.second)
%n240.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n240.value, i32 %n240.scalar.p
store float %n240.scalar.scalar.0, ptr addrspace(1) %n240.scalar.output.ptr, align 4
br label %n240.scalar.step
n240.scalar.step:
%n240.scalar.next = add i32 %n240.scalar.p, %threads
br label %n240.scalar.loop
n240.scalar.done:
call void @grid_barrier(i32 %threads)%n241.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n241.second = getelementptr i8, ptr addrspace(1) %values, i64 1472
%n241.value = getelementptr i8, ptr addrspace(1) %values, i64 15424
%n241.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2472
%n241.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n241.scalar.entry
n241.scalar.entry:
%n241.scalar.count = mul i32 %rows, 16
br label %n241.scalar.loop
n241.scalar.loop:
%n241.scalar.p = phi i32 [ %tid, %n241.scalar.entry ], [ %n241.scalar.next, %n241.scalar.step ]
%n241.scalar.more = icmp ult i32 %n241.scalar.p, %n241.scalar.count
br i1 %n241.scalar.more, label %n241.scalar.body, label %n241.scalar.done
n241.scalar.body:
%n241.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n241.source, i32 %n241.scalar.p
%n241.scalar.first = load float, ptr addrspace(1) %n241.scalar.first.ptr, align 4
%n241.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n241.second, i32 %n241.scalar.p
%n241.scalar.second = load float, ptr addrspace(1) %n241.scalar.second.ptr, align 4
%n241.scalar.scalar.0 = call float @recipe.sub(float %n241.scalar.first, float %n241.scalar.second)
%n241.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n241.value, i32 %n241.scalar.p
store float %n241.scalar.scalar.0, ptr addrspace(1) %n241.scalar.output.ptr, align 4
br label %n241.scalar.step
n241.scalar.step:
%n241.scalar.next = add i32 %n241.scalar.p, %threads
br label %n241.scalar.loop
n241.scalar.done:
call void @grid_barrier(i32 %threads)%n242.source = getelementptr i8, ptr addrspace(1) %values, i64 15424
%n242.value = getelementptr i8, ptr addrspace(1) %values, i64 15488
%n242.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2480
%n242.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n242.scalar.entry
n242.scalar.entry:
%n242.scalar.count = mul i32 %rows, 16
br label %n242.scalar.loop
n242.scalar.loop:
%n242.scalar.p = phi i32 [ %tid, %n242.scalar.entry ], [ %n242.scalar.next, %n242.scalar.step ]
%n242.scalar.more = icmp ult i32 %n242.scalar.p, %n242.scalar.count
br i1 %n242.scalar.more, label %n242.scalar.body, label %n242.scalar.done
n242.scalar.body:
%n242.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n242.source, i32 %n242.scalar.p
%n242.scalar.first = load float, ptr addrspace(1) %n242.scalar.first.ptr, align 4
%n242.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n242.scalar.first, float 0x0000000000000000)
%n242.scalar.scalar.2 = call float @recipe.from.u1(i1 %n242.scalar.scalar.2.condition)
%n242.scalar.scalar.3 = call float @recipe.exp(float %n242.scalar.first)
%n242.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n242.value, i32 %n242.scalar.p
store float %n242.scalar.scalar.3, ptr addrspace(1) %n242.scalar.output.ptr, align 4
br label %n242.scalar.step
n242.scalar.step:
%n242.scalar.next = add i32 %n242.scalar.p, %threads
br label %n242.scalar.loop
n242.scalar.done:
call void @grid_barrier(i32 %threads)%n243.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n243.value = getelementptr i8, ptr addrspace(1) %values, i64 15552
%n243.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2488
%n243.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n243.scalar.entry
n243.scalar.entry:
%n243.scalar.count = mul i32 %rows, 16
br label %n243.scalar.loop
n243.scalar.loop:
%n243.scalar.p = phi i32 [ %tid, %n243.scalar.entry ], [ %n243.scalar.next, %n243.scalar.step ]
%n243.scalar.more = icmp ult i32 %n243.scalar.p, %n243.scalar.count
br i1 %n243.scalar.more, label %n243.scalar.body, label %n243.scalar.done
n243.scalar.body:
%n243.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n243.source, i32 %n243.scalar.p
%n243.scalar.first = load float, ptr addrspace(1) %n243.scalar.first.ptr, align 4
%n243.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n243.value, i32 %n243.scalar.p
store float 0x0000000000000000, ptr addrspace(1) %n243.scalar.output.ptr, align 4
br label %n243.scalar.step
n243.scalar.step:
%n243.scalar.next = add i32 %n243.scalar.p, %threads
br label %n243.scalar.loop
n243.scalar.done:
call void @grid_barrier(i32 %threads)%n244.source = getelementptr i8, ptr addrspace(1) %values, i64 576
%n244.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n244.value = getelementptr i8, ptr addrspace(1) %values, i64 15616
%n244.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2496
%n244.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n244.scalar.entry
n244.scalar.entry:
%n244.scalar.count = mul i32 %rows, 16
br label %n244.scalar.loop
n244.scalar.loop:
%n244.scalar.p = phi i32 [ %tid, %n244.scalar.entry ], [ %n244.scalar.next, %n244.scalar.step ]
%n244.scalar.more = icmp ult i32 %n244.scalar.p, %n244.scalar.count
br i1 %n244.scalar.more, label %n244.scalar.body, label %n244.scalar.done
n244.scalar.body:
%n244.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n244.source, i32 %n244.scalar.p
%n244.scalar.first = load float, ptr addrspace(1) %n244.scalar.first.ptr, align 4
%n244.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n244.second, i32 %n244.scalar.p
%n244.scalar.second = load float, ptr addrspace(1) %n244.scalar.second.ptr, align 4
%n244.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n244.scalar.first, float %n244.scalar.second)
%n244.scalar.scalar.0 = call float @recipe.from.u1(i1 %n244.scalar.scalar.0.condition)
%n244.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n244.value, i32 %n244.scalar.p
store float %n244.scalar.scalar.0, ptr addrspace(1) %n244.scalar.output.ptr, align 4
br label %n244.scalar.step
n244.scalar.step:
%n244.scalar.next = add i32 %n244.scalar.p, %threads
br label %n244.scalar.loop
n244.scalar.done:
call void @grid_barrier(i32 %threads)%n245.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n245.second = getelementptr i8, ptr addrspace(1) %values, i64 576
%n245.value = getelementptr i8, ptr addrspace(1) %values, i64 15680
%n245.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2504
%n245.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n245.scalar.entry
n245.scalar.entry:
%n245.scalar.count = mul i32 %rows, 16
br label %n245.scalar.loop
n245.scalar.loop:
%n245.scalar.p = phi i32 [ %tid, %n245.scalar.entry ], [ %n245.scalar.next, %n245.scalar.step ]
%n245.scalar.more = icmp ult i32 %n245.scalar.p, %n245.scalar.count
br i1 %n245.scalar.more, label %n245.scalar.body, label %n245.scalar.done
n245.scalar.body:
%n245.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n245.source, i32 %n245.scalar.p
%n245.scalar.first = load float, ptr addrspace(1) %n245.scalar.first.ptr, align 4
%n245.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n245.second, i32 %n245.scalar.p
%n245.scalar.second = load float, ptr addrspace(1) %n245.scalar.second.ptr, align 4
%n245.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n245.scalar.first, float %n245.scalar.second)
%n245.scalar.scalar.0 = call float @recipe.from.u1(i1 %n245.scalar.scalar.0.condition)
%n245.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n245.value, i32 %n245.scalar.p
store float %n245.scalar.scalar.0, ptr addrspace(1) %n245.scalar.output.ptr, align 4
br label %n245.scalar.step
n245.scalar.step:
%n245.scalar.next = add i32 %n245.scalar.p, %threads
br label %n245.scalar.loop
n245.scalar.done:
call void @grid_barrier(i32 %threads)%n246.source = getelementptr i8, ptr addrspace(1) %values, i64 15616
%n246.second = getelementptr i8, ptr addrspace(1) %values, i64 15680
%n246.value = getelementptr i8, ptr addrspace(1) %values, i64 15744
%n246.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2512
%n246.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n246.scalar.entry
n246.scalar.entry:
%n246.scalar.count = mul i32 %rows, 16
br label %n246.scalar.loop
n246.scalar.loop:
%n246.scalar.p = phi i32 [ %tid, %n246.scalar.entry ], [ %n246.scalar.next, %n246.scalar.step ]
%n246.scalar.more = icmp ult i32 %n246.scalar.p, %n246.scalar.count
br i1 %n246.scalar.more, label %n246.scalar.body, label %n246.scalar.done
n246.scalar.body:
%n246.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n246.source, i32 %n246.scalar.p
%n246.scalar.first = load float, ptr addrspace(1) %n246.scalar.first.ptr, align 4
%n246.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n246.second, i32 %n246.scalar.p
%n246.scalar.second = load float, ptr addrspace(1) %n246.scalar.second.ptr, align 4
%n246.scalar.scalar.0 = call float @recipe.add(float %n246.scalar.first, float %n246.scalar.second)
%n246.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n246.value, i32 %n246.scalar.p
store float %n246.scalar.scalar.0, ptr addrspace(1) %n246.scalar.output.ptr, align 4
br label %n246.scalar.step
n246.scalar.step:
%n246.scalar.next = add i32 %n246.scalar.p, %threads
br label %n246.scalar.loop
n246.scalar.done:
call void @grid_barrier(i32 %threads)%n247.source = getelementptr i8, ptr addrspace(1) %values, i64 15744
%n247.value = getelementptr i8, ptr addrspace(1) %values, i64 15808
%n247.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2520
%n247.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n247.scalar.entry
n247.scalar.entry:
%n247.scalar.count = mul i32 %rows, 16
br label %n247.scalar.loop
n247.scalar.loop:
%n247.scalar.p = phi i32 [ %tid, %n247.scalar.entry ], [ %n247.scalar.next, %n247.scalar.step ]
%n247.scalar.more = icmp ult i32 %n247.scalar.p, %n247.scalar.count
br i1 %n247.scalar.more, label %n247.scalar.body, label %n247.scalar.done
n247.scalar.body:
%n247.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n247.source, i32 %n247.scalar.p
%n247.scalar.first = load float, ptr addrspace(1) %n247.scalar.first.ptr, align 4
%n247.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n247.scalar.first)
%n247.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n247.value, i32 %n247.scalar.p
store float %n247.scalar.scalar.1, ptr addrspace(1) %n247.scalar.output.ptr, align 4
br label %n247.scalar.step
n247.scalar.step:
%n247.scalar.next = add i32 %n247.scalar.p, %threads
br label %n247.scalar.loop
n247.scalar.done:
call void @grid_barrier(i32 %threads)%n248.source = getelementptr i8, ptr addrspace(1) %values, i64 15616
%n248.second = getelementptr i8, ptr addrspace(1) %values, i64 15808
%n248.value = getelementptr i8, ptr addrspace(1) %values, i64 15872
%n248.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2528
%n248.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n248.scalar.entry
n248.scalar.entry:
%n248.scalar.count = mul i32 %rows, 16
br label %n248.scalar.loop
n248.scalar.loop:
%n248.scalar.p = phi i32 [ %tid, %n248.scalar.entry ], [ %n248.scalar.next, %n248.scalar.step ]
%n248.scalar.more = icmp ult i32 %n248.scalar.p, %n248.scalar.count
br i1 %n248.scalar.more, label %n248.scalar.body, label %n248.scalar.done
n248.scalar.body:
%n248.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n248.source, i32 %n248.scalar.p
%n248.scalar.first = load float, ptr addrspace(1) %n248.scalar.first.ptr, align 4
%n248.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n248.second, i32 %n248.scalar.p
%n248.scalar.second = load float, ptr addrspace(1) %n248.scalar.second.ptr, align 4
%n248.scalar.scalar.0 = call float @recipe.add(float %n248.scalar.first, float %n248.scalar.second)
%n248.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n248.value, i32 %n248.scalar.p
store float %n248.scalar.scalar.0, ptr addrspace(1) %n248.scalar.output.ptr, align 4
br label %n248.scalar.step
n248.scalar.step:
%n248.scalar.next = add i32 %n248.scalar.p, %threads
br label %n248.scalar.loop
n248.scalar.done:
call void @grid_barrier(i32 %threads)%n249.source = getelementptr i8, ptr addrspace(1) %values, i64 15552
%n249.second = getelementptr i8, ptr addrspace(1) %values, i64 15872
%n249.value = getelementptr i8, ptr addrspace(1) %values, i64 15936
%n249.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2536
%n249.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n249.scalar.entry
n249.scalar.entry:
%n249.scalar.count = mul i32 %rows, 16
br label %n249.scalar.loop
n249.scalar.loop:
%n249.scalar.p = phi i32 [ %tid, %n249.scalar.entry ], [ %n249.scalar.next, %n249.scalar.step ]
%n249.scalar.more = icmp ult i32 %n249.scalar.p, %n249.scalar.count
br i1 %n249.scalar.more, label %n249.scalar.body, label %n249.scalar.done
n249.scalar.body:
%n249.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n249.source, i32 %n249.scalar.p
%n249.scalar.first = load float, ptr addrspace(1) %n249.scalar.first.ptr, align 4
%n249.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n249.second, i32 %n249.scalar.p
%n249.scalar.second = load float, ptr addrspace(1) %n249.scalar.second.ptr, align 4
%n249.scalar.scalar.0 = call float @recipe.add(float %n249.scalar.first, float %n249.scalar.second)
%n249.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n249.value, i32 %n249.scalar.p
store float %n249.scalar.scalar.0, ptr addrspace(1) %n249.scalar.output.ptr, align 4
br label %n249.scalar.step
n249.scalar.step:
%n249.scalar.next = add i32 %n249.scalar.p, %threads
br label %n249.scalar.loop
n249.scalar.done:
call void @grid_barrier(i32 %threads)%n250.source = getelementptr i8, ptr addrspace(1) %values, i64 640
%n250.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n250.value = getelementptr i8, ptr addrspace(1) %values, i64 16000
%n250.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2544
%n250.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n250.scalar.entry
n250.scalar.entry:
%n250.scalar.count = mul i32 %rows, 16
br label %n250.scalar.loop
n250.scalar.loop:
%n250.scalar.p = phi i32 [ %tid, %n250.scalar.entry ], [ %n250.scalar.next, %n250.scalar.step ]
%n250.scalar.more = icmp ult i32 %n250.scalar.p, %n250.scalar.count
br i1 %n250.scalar.more, label %n250.scalar.body, label %n250.scalar.done
n250.scalar.body:
%n250.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n250.source, i32 %n250.scalar.p
%n250.scalar.first = load float, ptr addrspace(1) %n250.scalar.first.ptr, align 4
%n250.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n250.second, i32 %n250.scalar.p
%n250.scalar.second = load float, ptr addrspace(1) %n250.scalar.second.ptr, align 4
%n250.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n250.scalar.first, float %n250.scalar.second)
%n250.scalar.scalar.0 = call float @recipe.from.u1(i1 %n250.scalar.scalar.0.condition)
%n250.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n250.value, i32 %n250.scalar.p
store float %n250.scalar.scalar.0, ptr addrspace(1) %n250.scalar.output.ptr, align 4
br label %n250.scalar.step
n250.scalar.step:
%n250.scalar.next = add i32 %n250.scalar.p, %threads
br label %n250.scalar.loop
n250.scalar.done:
call void @grid_barrier(i32 %threads)%n251.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n251.second = getelementptr i8, ptr addrspace(1) %values, i64 640
%n251.value = getelementptr i8, ptr addrspace(1) %values, i64 16064
%n251.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2552
%n251.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n251.scalar.entry
n251.scalar.entry:
%n251.scalar.count = mul i32 %rows, 16
br label %n251.scalar.loop
n251.scalar.loop:
%n251.scalar.p = phi i32 [ %tid, %n251.scalar.entry ], [ %n251.scalar.next, %n251.scalar.step ]
%n251.scalar.more = icmp ult i32 %n251.scalar.p, %n251.scalar.count
br i1 %n251.scalar.more, label %n251.scalar.body, label %n251.scalar.done
n251.scalar.body:
%n251.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n251.source, i32 %n251.scalar.p
%n251.scalar.first = load float, ptr addrspace(1) %n251.scalar.first.ptr, align 4
%n251.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n251.second, i32 %n251.scalar.p
%n251.scalar.second = load float, ptr addrspace(1) %n251.scalar.second.ptr, align 4
%n251.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n251.scalar.first, float %n251.scalar.second)
%n251.scalar.scalar.0 = call float @recipe.from.u1(i1 %n251.scalar.scalar.0.condition)
%n251.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n251.value, i32 %n251.scalar.p
store float %n251.scalar.scalar.0, ptr addrspace(1) %n251.scalar.output.ptr, align 4
br label %n251.scalar.step
n251.scalar.step:
%n251.scalar.next = add i32 %n251.scalar.p, %threads
br label %n251.scalar.loop
n251.scalar.done:
call void @grid_barrier(i32 %threads)%n252.source = getelementptr i8, ptr addrspace(1) %values, i64 16000
%n252.second = getelementptr i8, ptr addrspace(1) %values, i64 16064
%n252.value = getelementptr i8, ptr addrspace(1) %values, i64 16128
%n252.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2560
%n252.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n252.scalar.entry
n252.scalar.entry:
%n252.scalar.count = mul i32 %rows, 16
br label %n252.scalar.loop
n252.scalar.loop:
%n252.scalar.p = phi i32 [ %tid, %n252.scalar.entry ], [ %n252.scalar.next, %n252.scalar.step ]
%n252.scalar.more = icmp ult i32 %n252.scalar.p, %n252.scalar.count
br i1 %n252.scalar.more, label %n252.scalar.body, label %n252.scalar.done
n252.scalar.body:
%n252.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n252.source, i32 %n252.scalar.p
%n252.scalar.first = load float, ptr addrspace(1) %n252.scalar.first.ptr, align 4
%n252.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n252.second, i32 %n252.scalar.p
%n252.scalar.second = load float, ptr addrspace(1) %n252.scalar.second.ptr, align 4
%n252.scalar.scalar.0 = call float @recipe.add(float %n252.scalar.first, float %n252.scalar.second)
%n252.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n252.value, i32 %n252.scalar.p
store float %n252.scalar.scalar.0, ptr addrspace(1) %n252.scalar.output.ptr, align 4
br label %n252.scalar.step
n252.scalar.step:
%n252.scalar.next = add i32 %n252.scalar.p, %threads
br label %n252.scalar.loop
n252.scalar.done:
call void @grid_barrier(i32 %threads)%n253.source = getelementptr i8, ptr addrspace(1) %values, i64 16128
%n253.value = getelementptr i8, ptr addrspace(1) %values, i64 16192
%n253.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2568
%n253.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n253.scalar.entry
n253.scalar.entry:
%n253.scalar.count = mul i32 %rows, 16
br label %n253.scalar.loop
n253.scalar.loop:
%n253.scalar.p = phi i32 [ %tid, %n253.scalar.entry ], [ %n253.scalar.next, %n253.scalar.step ]
%n253.scalar.more = icmp ult i32 %n253.scalar.p, %n253.scalar.count
br i1 %n253.scalar.more, label %n253.scalar.body, label %n253.scalar.done
n253.scalar.body:
%n253.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n253.source, i32 %n253.scalar.p
%n253.scalar.first = load float, ptr addrspace(1) %n253.scalar.first.ptr, align 4
%n253.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n253.scalar.first)
%n253.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n253.value, i32 %n253.scalar.p
store float %n253.scalar.scalar.1, ptr addrspace(1) %n253.scalar.output.ptr, align 4
br label %n253.scalar.step
n253.scalar.step:
%n253.scalar.next = add i32 %n253.scalar.p, %threads
br label %n253.scalar.loop
n253.scalar.done:
call void @grid_barrier(i32 %threads)%n254.source = getelementptr i8, ptr addrspace(1) %values, i64 16000
%n254.second = getelementptr i8, ptr addrspace(1) %values, i64 16192
%n254.value = getelementptr i8, ptr addrspace(1) %values, i64 16256
%n254.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2576
%n254.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n254.scalar.entry
n254.scalar.entry:
%n254.scalar.count = mul i32 %rows, 16
br label %n254.scalar.loop
n254.scalar.loop:
%n254.scalar.p = phi i32 [ %tid, %n254.scalar.entry ], [ %n254.scalar.next, %n254.scalar.step ]
%n254.scalar.more = icmp ult i32 %n254.scalar.p, %n254.scalar.count
br i1 %n254.scalar.more, label %n254.scalar.body, label %n254.scalar.done
n254.scalar.body:
%n254.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n254.source, i32 %n254.scalar.p
%n254.scalar.first = load float, ptr addrspace(1) %n254.scalar.first.ptr, align 4
%n254.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n254.second, i32 %n254.scalar.p
%n254.scalar.second = load float, ptr addrspace(1) %n254.scalar.second.ptr, align 4
%n254.scalar.scalar.0 = call float @recipe.add(float %n254.scalar.first, float %n254.scalar.second)
%n254.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n254.value, i32 %n254.scalar.p
store float %n254.scalar.scalar.0, ptr addrspace(1) %n254.scalar.output.ptr, align 4
br label %n254.scalar.step
n254.scalar.step:
%n254.scalar.next = add i32 %n254.scalar.p, %threads
br label %n254.scalar.loop
n254.scalar.done:
call void @grid_barrier(i32 %threads)%n255.source = getelementptr i8, ptr addrspace(1) %values, i64 15936
%n255.second = getelementptr i8, ptr addrspace(1) %values, i64 16256
%n255.value = getelementptr i8, ptr addrspace(1) %values, i64 16320
%n255.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2584
%n255.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n255.scalar.entry
n255.scalar.entry:
%n255.scalar.count = mul i32 %rows, 16
br label %n255.scalar.loop
n255.scalar.loop:
%n255.scalar.p = phi i32 [ %tid, %n255.scalar.entry ], [ %n255.scalar.next, %n255.scalar.step ]
%n255.scalar.more = icmp ult i32 %n255.scalar.p, %n255.scalar.count
br i1 %n255.scalar.more, label %n255.scalar.body, label %n255.scalar.done
n255.scalar.body:
%n255.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n255.source, i32 %n255.scalar.p
%n255.scalar.first = load float, ptr addrspace(1) %n255.scalar.first.ptr, align 4
%n255.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n255.second, i32 %n255.scalar.p
%n255.scalar.second = load float, ptr addrspace(1) %n255.scalar.second.ptr, align 4
%n255.scalar.scalar.0 = call float @recipe.add(float %n255.scalar.first, float %n255.scalar.second)
%n255.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n255.value, i32 %n255.scalar.p
store float %n255.scalar.scalar.0, ptr addrspace(1) %n255.scalar.output.ptr, align 4
br label %n255.scalar.step
n255.scalar.step:
%n255.scalar.next = add i32 %n255.scalar.p, %threads
br label %n255.scalar.loop
n255.scalar.done:
call void @grid_barrier(i32 %threads)%n256.source = getelementptr i8, ptr addrspace(1) %values, i64 704
%n256.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n256.value = getelementptr i8, ptr addrspace(1) %values, i64 16384
%n256.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2592
%n256.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n256.scalar.entry
n256.scalar.entry:
%n256.scalar.count = mul i32 %rows, 16
br label %n256.scalar.loop
n256.scalar.loop:
%n256.scalar.p = phi i32 [ %tid, %n256.scalar.entry ], [ %n256.scalar.next, %n256.scalar.step ]
%n256.scalar.more = icmp ult i32 %n256.scalar.p, %n256.scalar.count
br i1 %n256.scalar.more, label %n256.scalar.body, label %n256.scalar.done
n256.scalar.body:
%n256.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n256.source, i32 %n256.scalar.p
%n256.scalar.first = load float, ptr addrspace(1) %n256.scalar.first.ptr, align 4
%n256.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n256.second, i32 %n256.scalar.p
%n256.scalar.second = load float, ptr addrspace(1) %n256.scalar.second.ptr, align 4
%n256.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n256.scalar.first, float %n256.scalar.second)
%n256.scalar.scalar.0 = call float @recipe.from.u1(i1 %n256.scalar.scalar.0.condition)
%n256.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n256.value, i32 %n256.scalar.p
store float %n256.scalar.scalar.0, ptr addrspace(1) %n256.scalar.output.ptr, align 4
br label %n256.scalar.step
n256.scalar.step:
%n256.scalar.next = add i32 %n256.scalar.p, %threads
br label %n256.scalar.loop
n256.scalar.done:
call void @grid_barrier(i32 %threads)%n257.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n257.second = getelementptr i8, ptr addrspace(1) %values, i64 704
%n257.value = getelementptr i8, ptr addrspace(1) %values, i64 16448
%n257.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2600
%n257.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n257.scalar.entry
n257.scalar.entry:
%n257.scalar.count = mul i32 %rows, 16
br label %n257.scalar.loop
n257.scalar.loop:
%n257.scalar.p = phi i32 [ %tid, %n257.scalar.entry ], [ %n257.scalar.next, %n257.scalar.step ]
%n257.scalar.more = icmp ult i32 %n257.scalar.p, %n257.scalar.count
br i1 %n257.scalar.more, label %n257.scalar.body, label %n257.scalar.done
n257.scalar.body:
%n257.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n257.source, i32 %n257.scalar.p
%n257.scalar.first = load float, ptr addrspace(1) %n257.scalar.first.ptr, align 4
%n257.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n257.second, i32 %n257.scalar.p
%n257.scalar.second = load float, ptr addrspace(1) %n257.scalar.second.ptr, align 4
%n257.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n257.scalar.first, float %n257.scalar.second)
%n257.scalar.scalar.0 = call float @recipe.from.u1(i1 %n257.scalar.scalar.0.condition)
%n257.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n257.value, i32 %n257.scalar.p
store float %n257.scalar.scalar.0, ptr addrspace(1) %n257.scalar.output.ptr, align 4
br label %n257.scalar.step
n257.scalar.step:
%n257.scalar.next = add i32 %n257.scalar.p, %threads
br label %n257.scalar.loop
n257.scalar.done:
call void @grid_barrier(i32 %threads)%n258.source = getelementptr i8, ptr addrspace(1) %values, i64 16384
%n258.second = getelementptr i8, ptr addrspace(1) %values, i64 16448
%n258.value = getelementptr i8, ptr addrspace(1) %values, i64 16512
%n258.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2608
%n258.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n258.scalar.entry
n258.scalar.entry:
%n258.scalar.count = mul i32 %rows, 16
br label %n258.scalar.loop
n258.scalar.loop:
%n258.scalar.p = phi i32 [ %tid, %n258.scalar.entry ], [ %n258.scalar.next, %n258.scalar.step ]
%n258.scalar.more = icmp ult i32 %n258.scalar.p, %n258.scalar.count
br i1 %n258.scalar.more, label %n258.scalar.body, label %n258.scalar.done
n258.scalar.body:
%n258.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n258.source, i32 %n258.scalar.p
%n258.scalar.first = load float, ptr addrspace(1) %n258.scalar.first.ptr, align 4
%n258.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n258.second, i32 %n258.scalar.p
%n258.scalar.second = load float, ptr addrspace(1) %n258.scalar.second.ptr, align 4
%n258.scalar.scalar.0 = call float @recipe.add(float %n258.scalar.first, float %n258.scalar.second)
%n258.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n258.value, i32 %n258.scalar.p
store float %n258.scalar.scalar.0, ptr addrspace(1) %n258.scalar.output.ptr, align 4
br label %n258.scalar.step
n258.scalar.step:
%n258.scalar.next = add i32 %n258.scalar.p, %threads
br label %n258.scalar.loop
n258.scalar.done:
call void @grid_barrier(i32 %threads)%n259.source = getelementptr i8, ptr addrspace(1) %values, i64 16512
%n259.value = getelementptr i8, ptr addrspace(1) %values, i64 16576
%n259.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2616
%n259.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n259.scalar.entry
n259.scalar.entry:
%n259.scalar.count = mul i32 %rows, 16
br label %n259.scalar.loop
n259.scalar.loop:
%n259.scalar.p = phi i32 [ %tid, %n259.scalar.entry ], [ %n259.scalar.next, %n259.scalar.step ]
%n259.scalar.more = icmp ult i32 %n259.scalar.p, %n259.scalar.count
br i1 %n259.scalar.more, label %n259.scalar.body, label %n259.scalar.done
n259.scalar.body:
%n259.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n259.source, i32 %n259.scalar.p
%n259.scalar.first = load float, ptr addrspace(1) %n259.scalar.first.ptr, align 4
%n259.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n259.scalar.first)
%n259.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n259.value, i32 %n259.scalar.p
store float %n259.scalar.scalar.1, ptr addrspace(1) %n259.scalar.output.ptr, align 4
br label %n259.scalar.step
n259.scalar.step:
%n259.scalar.next = add i32 %n259.scalar.p, %threads
br label %n259.scalar.loop
n259.scalar.done:
call void @grid_barrier(i32 %threads)%n260.source = getelementptr i8, ptr addrspace(1) %values, i64 16384
%n260.second = getelementptr i8, ptr addrspace(1) %values, i64 16576
%n260.value = getelementptr i8, ptr addrspace(1) %values, i64 16640
%n260.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2624
%n260.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n260.scalar.entry
n260.scalar.entry:
%n260.scalar.count = mul i32 %rows, 16
br label %n260.scalar.loop
n260.scalar.loop:
%n260.scalar.p = phi i32 [ %tid, %n260.scalar.entry ], [ %n260.scalar.next, %n260.scalar.step ]
%n260.scalar.more = icmp ult i32 %n260.scalar.p, %n260.scalar.count
br i1 %n260.scalar.more, label %n260.scalar.body, label %n260.scalar.done
n260.scalar.body:
%n260.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n260.source, i32 %n260.scalar.p
%n260.scalar.first = load float, ptr addrspace(1) %n260.scalar.first.ptr, align 4
%n260.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n260.second, i32 %n260.scalar.p
%n260.scalar.second = load float, ptr addrspace(1) %n260.scalar.second.ptr, align 4
%n260.scalar.scalar.0 = call float @recipe.add(float %n260.scalar.first, float %n260.scalar.second)
%n260.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n260.value, i32 %n260.scalar.p
store float %n260.scalar.scalar.0, ptr addrspace(1) %n260.scalar.output.ptr, align 4
br label %n260.scalar.step
n260.scalar.step:
%n260.scalar.next = add i32 %n260.scalar.p, %threads
br label %n260.scalar.loop
n260.scalar.done:
call void @grid_barrier(i32 %threads)%n261.source = getelementptr i8, ptr addrspace(1) %values, i64 16320
%n261.second = getelementptr i8, ptr addrspace(1) %values, i64 16640
%n261.value = getelementptr i8, ptr addrspace(1) %values, i64 16704
%n261.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2632
%n261.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n261.scalar.entry
n261.scalar.entry:
%n261.scalar.count = mul i32 %rows, 16
br label %n261.scalar.loop
n261.scalar.loop:
%n261.scalar.p = phi i32 [ %tid, %n261.scalar.entry ], [ %n261.scalar.next, %n261.scalar.step ]
%n261.scalar.more = icmp ult i32 %n261.scalar.p, %n261.scalar.count
br i1 %n261.scalar.more, label %n261.scalar.body, label %n261.scalar.done
n261.scalar.body:
%n261.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n261.source, i32 %n261.scalar.p
%n261.scalar.first = load float, ptr addrspace(1) %n261.scalar.first.ptr, align 4
%n261.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n261.second, i32 %n261.scalar.p
%n261.scalar.second = load float, ptr addrspace(1) %n261.scalar.second.ptr, align 4
%n261.scalar.scalar.0 = call float @recipe.add(float %n261.scalar.first, float %n261.scalar.second)
%n261.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n261.value, i32 %n261.scalar.p
store float %n261.scalar.scalar.0, ptr addrspace(1) %n261.scalar.output.ptr, align 4
br label %n261.scalar.step
n261.scalar.step:
%n261.scalar.next = add i32 %n261.scalar.p, %threads
br label %n261.scalar.loop
n261.scalar.done:
call void @grid_barrier(i32 %threads)%n262.source = getelementptr i8, ptr addrspace(1) %values, i64 768
%n262.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n262.value = getelementptr i8, ptr addrspace(1) %values, i64 16768
%n262.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2640
%n262.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n262.scalar.entry
n262.scalar.entry:
%n262.scalar.count = mul i32 %rows, 16
br label %n262.scalar.loop
n262.scalar.loop:
%n262.scalar.p = phi i32 [ %tid, %n262.scalar.entry ], [ %n262.scalar.next, %n262.scalar.step ]
%n262.scalar.more = icmp ult i32 %n262.scalar.p, %n262.scalar.count
br i1 %n262.scalar.more, label %n262.scalar.body, label %n262.scalar.done
n262.scalar.body:
%n262.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n262.source, i32 %n262.scalar.p
%n262.scalar.first = load float, ptr addrspace(1) %n262.scalar.first.ptr, align 4
%n262.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n262.second, i32 %n262.scalar.p
%n262.scalar.second = load float, ptr addrspace(1) %n262.scalar.second.ptr, align 4
%n262.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n262.scalar.first, float %n262.scalar.second)
%n262.scalar.scalar.0 = call float @recipe.from.u1(i1 %n262.scalar.scalar.0.condition)
%n262.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n262.value, i32 %n262.scalar.p
store float %n262.scalar.scalar.0, ptr addrspace(1) %n262.scalar.output.ptr, align 4
br label %n262.scalar.step
n262.scalar.step:
%n262.scalar.next = add i32 %n262.scalar.p, %threads
br label %n262.scalar.loop
n262.scalar.done:
call void @grid_barrier(i32 %threads)%n263.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n263.second = getelementptr i8, ptr addrspace(1) %values, i64 768
%n263.value = getelementptr i8, ptr addrspace(1) %values, i64 16832
%n263.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2648
%n263.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n263.scalar.entry
n263.scalar.entry:
%n263.scalar.count = mul i32 %rows, 16
br label %n263.scalar.loop
n263.scalar.loop:
%n263.scalar.p = phi i32 [ %tid, %n263.scalar.entry ], [ %n263.scalar.next, %n263.scalar.step ]
%n263.scalar.more = icmp ult i32 %n263.scalar.p, %n263.scalar.count
br i1 %n263.scalar.more, label %n263.scalar.body, label %n263.scalar.done
n263.scalar.body:
%n263.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n263.source, i32 %n263.scalar.p
%n263.scalar.first = load float, ptr addrspace(1) %n263.scalar.first.ptr, align 4
%n263.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n263.second, i32 %n263.scalar.p
%n263.scalar.second = load float, ptr addrspace(1) %n263.scalar.second.ptr, align 4
%n263.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n263.scalar.first, float %n263.scalar.second)
%n263.scalar.scalar.0 = call float @recipe.from.u1(i1 %n263.scalar.scalar.0.condition)
%n263.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n263.value, i32 %n263.scalar.p
store float %n263.scalar.scalar.0, ptr addrspace(1) %n263.scalar.output.ptr, align 4
br label %n263.scalar.step
n263.scalar.step:
%n263.scalar.next = add i32 %n263.scalar.p, %threads
br label %n263.scalar.loop
n263.scalar.done:
call void @grid_barrier(i32 %threads)%n264.source = getelementptr i8, ptr addrspace(1) %values, i64 16768
%n264.second = getelementptr i8, ptr addrspace(1) %values, i64 16832
%n264.value = getelementptr i8, ptr addrspace(1) %values, i64 16896
%n264.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2656
%n264.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n264.scalar.entry
n264.scalar.entry:
%n264.scalar.count = mul i32 %rows, 16
br label %n264.scalar.loop
n264.scalar.loop:
%n264.scalar.p = phi i32 [ %tid, %n264.scalar.entry ], [ %n264.scalar.next, %n264.scalar.step ]
%n264.scalar.more = icmp ult i32 %n264.scalar.p, %n264.scalar.count
br i1 %n264.scalar.more, label %n264.scalar.body, label %n264.scalar.done
n264.scalar.body:
%n264.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n264.source, i32 %n264.scalar.p
%n264.scalar.first = load float, ptr addrspace(1) %n264.scalar.first.ptr, align 4
%n264.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n264.second, i32 %n264.scalar.p
%n264.scalar.second = load float, ptr addrspace(1) %n264.scalar.second.ptr, align 4
%n264.scalar.scalar.0 = call float @recipe.add(float %n264.scalar.first, float %n264.scalar.second)
%n264.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n264.value, i32 %n264.scalar.p
store float %n264.scalar.scalar.0, ptr addrspace(1) %n264.scalar.output.ptr, align 4
br label %n264.scalar.step
n264.scalar.step:
%n264.scalar.next = add i32 %n264.scalar.p, %threads
br label %n264.scalar.loop
n264.scalar.done:
call void @grid_barrier(i32 %threads)%n265.source = getelementptr i8, ptr addrspace(1) %values, i64 16896
%n265.value = getelementptr i8, ptr addrspace(1) %values, i64 16960
%n265.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2664
%n265.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n265.scalar.entry
n265.scalar.entry:
%n265.scalar.count = mul i32 %rows, 16
br label %n265.scalar.loop
n265.scalar.loop:
%n265.scalar.p = phi i32 [ %tid, %n265.scalar.entry ], [ %n265.scalar.next, %n265.scalar.step ]
%n265.scalar.more = icmp ult i32 %n265.scalar.p, %n265.scalar.count
br i1 %n265.scalar.more, label %n265.scalar.body, label %n265.scalar.done
n265.scalar.body:
%n265.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n265.source, i32 %n265.scalar.p
%n265.scalar.first = load float, ptr addrspace(1) %n265.scalar.first.ptr, align 4
%n265.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n265.scalar.first)
%n265.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n265.value, i32 %n265.scalar.p
store float %n265.scalar.scalar.1, ptr addrspace(1) %n265.scalar.output.ptr, align 4
br label %n265.scalar.step
n265.scalar.step:
%n265.scalar.next = add i32 %n265.scalar.p, %threads
br label %n265.scalar.loop
n265.scalar.done:
call void @grid_barrier(i32 %threads)%n266.source = getelementptr i8, ptr addrspace(1) %values, i64 16768
%n266.second = getelementptr i8, ptr addrspace(1) %values, i64 16960
%n266.value = getelementptr i8, ptr addrspace(1) %values, i64 17024
%n266.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2672
%n266.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n266.scalar.entry
n266.scalar.entry:
%n266.scalar.count = mul i32 %rows, 16
br label %n266.scalar.loop
n266.scalar.loop:
%n266.scalar.p = phi i32 [ %tid, %n266.scalar.entry ], [ %n266.scalar.next, %n266.scalar.step ]
%n266.scalar.more = icmp ult i32 %n266.scalar.p, %n266.scalar.count
br i1 %n266.scalar.more, label %n266.scalar.body, label %n266.scalar.done
n266.scalar.body:
%n266.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n266.source, i32 %n266.scalar.p
%n266.scalar.first = load float, ptr addrspace(1) %n266.scalar.first.ptr, align 4
%n266.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n266.second, i32 %n266.scalar.p
%n266.scalar.second = load float, ptr addrspace(1) %n266.scalar.second.ptr, align 4
%n266.scalar.scalar.0 = call float @recipe.add(float %n266.scalar.first, float %n266.scalar.second)
%n266.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n266.value, i32 %n266.scalar.p
store float %n266.scalar.scalar.0, ptr addrspace(1) %n266.scalar.output.ptr, align 4
br label %n266.scalar.step
n266.scalar.step:
%n266.scalar.next = add i32 %n266.scalar.p, %threads
br label %n266.scalar.loop
n266.scalar.done:
call void @grid_barrier(i32 %threads)%n267.source = getelementptr i8, ptr addrspace(1) %values, i64 16704
%n267.second = getelementptr i8, ptr addrspace(1) %values, i64 17024
%n267.value = getelementptr i8, ptr addrspace(1) %values, i64 17088
%n267.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2680
%n267.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n267.scalar.entry
n267.scalar.entry:
%n267.scalar.count = mul i32 %rows, 16
br label %n267.scalar.loop
n267.scalar.loop:
%n267.scalar.p = phi i32 [ %tid, %n267.scalar.entry ], [ %n267.scalar.next, %n267.scalar.step ]
%n267.scalar.more = icmp ult i32 %n267.scalar.p, %n267.scalar.count
br i1 %n267.scalar.more, label %n267.scalar.body, label %n267.scalar.done
n267.scalar.body:
%n267.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n267.source, i32 %n267.scalar.p
%n267.scalar.first = load float, ptr addrspace(1) %n267.scalar.first.ptr, align 4
%n267.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n267.second, i32 %n267.scalar.p
%n267.scalar.second = load float, ptr addrspace(1) %n267.scalar.second.ptr, align 4
%n267.scalar.scalar.0 = call float @recipe.add(float %n267.scalar.first, float %n267.scalar.second)
%n267.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n267.value, i32 %n267.scalar.p
store float %n267.scalar.scalar.0, ptr addrspace(1) %n267.scalar.output.ptr, align 4
br label %n267.scalar.step
n267.scalar.step:
%n267.scalar.next = add i32 %n267.scalar.p, %threads
br label %n267.scalar.loop
n267.scalar.done:
call void @grid_barrier(i32 %threads)%n268.source = getelementptr i8, ptr addrspace(1) %values, i64 832
%n268.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n268.value = getelementptr i8, ptr addrspace(1) %values, i64 17152
%n268.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2688
%n268.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n268.scalar.entry
n268.scalar.entry:
%n268.scalar.count = mul i32 %rows, 16
br label %n268.scalar.loop
n268.scalar.loop:
%n268.scalar.p = phi i32 [ %tid, %n268.scalar.entry ], [ %n268.scalar.next, %n268.scalar.step ]
%n268.scalar.more = icmp ult i32 %n268.scalar.p, %n268.scalar.count
br i1 %n268.scalar.more, label %n268.scalar.body, label %n268.scalar.done
n268.scalar.body:
%n268.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n268.source, i32 %n268.scalar.p
%n268.scalar.first = load float, ptr addrspace(1) %n268.scalar.first.ptr, align 4
%n268.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n268.second, i32 %n268.scalar.p
%n268.scalar.second = load float, ptr addrspace(1) %n268.scalar.second.ptr, align 4
%n268.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n268.scalar.first, float %n268.scalar.second)
%n268.scalar.scalar.0 = call float @recipe.from.u1(i1 %n268.scalar.scalar.0.condition)
%n268.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n268.value, i32 %n268.scalar.p
store float %n268.scalar.scalar.0, ptr addrspace(1) %n268.scalar.output.ptr, align 4
br label %n268.scalar.step
n268.scalar.step:
%n268.scalar.next = add i32 %n268.scalar.p, %threads
br label %n268.scalar.loop
n268.scalar.done:
call void @grid_barrier(i32 %threads)%n269.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n269.second = getelementptr i8, ptr addrspace(1) %values, i64 832
%n269.value = getelementptr i8, ptr addrspace(1) %values, i64 17216
%n269.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2696
%n269.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n269.scalar.entry
n269.scalar.entry:
%n269.scalar.count = mul i32 %rows, 16
br label %n269.scalar.loop
n269.scalar.loop:
%n269.scalar.p = phi i32 [ %tid, %n269.scalar.entry ], [ %n269.scalar.next, %n269.scalar.step ]
%n269.scalar.more = icmp ult i32 %n269.scalar.p, %n269.scalar.count
br i1 %n269.scalar.more, label %n269.scalar.body, label %n269.scalar.done
n269.scalar.body:
%n269.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n269.source, i32 %n269.scalar.p
%n269.scalar.first = load float, ptr addrspace(1) %n269.scalar.first.ptr, align 4
%n269.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n269.second, i32 %n269.scalar.p
%n269.scalar.second = load float, ptr addrspace(1) %n269.scalar.second.ptr, align 4
%n269.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n269.scalar.first, float %n269.scalar.second)
%n269.scalar.scalar.0 = call float @recipe.from.u1(i1 %n269.scalar.scalar.0.condition)
%n269.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n269.value, i32 %n269.scalar.p
store float %n269.scalar.scalar.0, ptr addrspace(1) %n269.scalar.output.ptr, align 4
br label %n269.scalar.step
n269.scalar.step:
%n269.scalar.next = add i32 %n269.scalar.p, %threads
br label %n269.scalar.loop
n269.scalar.done:
call void @grid_barrier(i32 %threads)%n270.source = getelementptr i8, ptr addrspace(1) %values, i64 17152
%n270.second = getelementptr i8, ptr addrspace(1) %values, i64 17216
%n270.value = getelementptr i8, ptr addrspace(1) %values, i64 17280
%n270.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2704
%n270.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n270.scalar.entry
n270.scalar.entry:
%n270.scalar.count = mul i32 %rows, 16
br label %n270.scalar.loop
n270.scalar.loop:
%n270.scalar.p = phi i32 [ %tid, %n270.scalar.entry ], [ %n270.scalar.next, %n270.scalar.step ]
%n270.scalar.more = icmp ult i32 %n270.scalar.p, %n270.scalar.count
br i1 %n270.scalar.more, label %n270.scalar.body, label %n270.scalar.done
n270.scalar.body:
%n270.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n270.source, i32 %n270.scalar.p
%n270.scalar.first = load float, ptr addrspace(1) %n270.scalar.first.ptr, align 4
%n270.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n270.second, i32 %n270.scalar.p
%n270.scalar.second = load float, ptr addrspace(1) %n270.scalar.second.ptr, align 4
%n270.scalar.scalar.0 = call float @recipe.add(float %n270.scalar.first, float %n270.scalar.second)
%n270.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n270.value, i32 %n270.scalar.p
store float %n270.scalar.scalar.0, ptr addrspace(1) %n270.scalar.output.ptr, align 4
br label %n270.scalar.step
n270.scalar.step:
%n270.scalar.next = add i32 %n270.scalar.p, %threads
br label %n270.scalar.loop
n270.scalar.done:
call void @grid_barrier(i32 %threads)%n271.source = getelementptr i8, ptr addrspace(1) %values, i64 17280
%n271.value = getelementptr i8, ptr addrspace(1) %values, i64 17344
%n271.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2712
%n271.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n271.scalar.entry
n271.scalar.entry:
%n271.scalar.count = mul i32 %rows, 16
br label %n271.scalar.loop
n271.scalar.loop:
%n271.scalar.p = phi i32 [ %tid, %n271.scalar.entry ], [ %n271.scalar.next, %n271.scalar.step ]
%n271.scalar.more = icmp ult i32 %n271.scalar.p, %n271.scalar.count
br i1 %n271.scalar.more, label %n271.scalar.body, label %n271.scalar.done
n271.scalar.body:
%n271.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n271.source, i32 %n271.scalar.p
%n271.scalar.first = load float, ptr addrspace(1) %n271.scalar.first.ptr, align 4
%n271.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n271.scalar.first)
%n271.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n271.value, i32 %n271.scalar.p
store float %n271.scalar.scalar.1, ptr addrspace(1) %n271.scalar.output.ptr, align 4
br label %n271.scalar.step
n271.scalar.step:
%n271.scalar.next = add i32 %n271.scalar.p, %threads
br label %n271.scalar.loop
n271.scalar.done:
call void @grid_barrier(i32 %threads)%n272.source = getelementptr i8, ptr addrspace(1) %values, i64 17152
%n272.second = getelementptr i8, ptr addrspace(1) %values, i64 17344
%n272.value = getelementptr i8, ptr addrspace(1) %values, i64 17408
%n272.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2720
%n272.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n272.scalar.entry
n272.scalar.entry:
%n272.scalar.count = mul i32 %rows, 16
br label %n272.scalar.loop
n272.scalar.loop:
%n272.scalar.p = phi i32 [ %tid, %n272.scalar.entry ], [ %n272.scalar.next, %n272.scalar.step ]
%n272.scalar.more = icmp ult i32 %n272.scalar.p, %n272.scalar.count
br i1 %n272.scalar.more, label %n272.scalar.body, label %n272.scalar.done
n272.scalar.body:
%n272.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n272.source, i32 %n272.scalar.p
%n272.scalar.first = load float, ptr addrspace(1) %n272.scalar.first.ptr, align 4
%n272.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n272.second, i32 %n272.scalar.p
%n272.scalar.second = load float, ptr addrspace(1) %n272.scalar.second.ptr, align 4
%n272.scalar.scalar.0 = call float @recipe.add(float %n272.scalar.first, float %n272.scalar.second)
%n272.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n272.value, i32 %n272.scalar.p
store float %n272.scalar.scalar.0, ptr addrspace(1) %n272.scalar.output.ptr, align 4
br label %n272.scalar.step
n272.scalar.step:
%n272.scalar.next = add i32 %n272.scalar.p, %threads
br label %n272.scalar.loop
n272.scalar.done:
call void @grid_barrier(i32 %threads)%n273.source = getelementptr i8, ptr addrspace(1) %values, i64 17088
%n273.second = getelementptr i8, ptr addrspace(1) %values, i64 17408
%n273.value = getelementptr i8, ptr addrspace(1) %values, i64 17472
%n273.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2728
%n273.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n273.scalar.entry
n273.scalar.entry:
%n273.scalar.count = mul i32 %rows, 16
br label %n273.scalar.loop
n273.scalar.loop:
%n273.scalar.p = phi i32 [ %tid, %n273.scalar.entry ], [ %n273.scalar.next, %n273.scalar.step ]
%n273.scalar.more = icmp ult i32 %n273.scalar.p, %n273.scalar.count
br i1 %n273.scalar.more, label %n273.scalar.body, label %n273.scalar.done
n273.scalar.body:
%n273.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n273.source, i32 %n273.scalar.p
%n273.scalar.first = load float, ptr addrspace(1) %n273.scalar.first.ptr, align 4
%n273.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n273.second, i32 %n273.scalar.p
%n273.scalar.second = load float, ptr addrspace(1) %n273.scalar.second.ptr, align 4
%n273.scalar.scalar.0 = call float @recipe.add(float %n273.scalar.first, float %n273.scalar.second)
%n273.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n273.value, i32 %n273.scalar.p
store float %n273.scalar.scalar.0, ptr addrspace(1) %n273.scalar.output.ptr, align 4
br label %n273.scalar.step
n273.scalar.step:
%n273.scalar.next = add i32 %n273.scalar.p, %threads
br label %n273.scalar.loop
n273.scalar.done:
call void @grid_barrier(i32 %threads)%n274.source = getelementptr i8, ptr addrspace(1) %values, i64 896
%n274.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n274.value = getelementptr i8, ptr addrspace(1) %values, i64 17536
%n274.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2736
%n274.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n274.scalar.entry
n274.scalar.entry:
%n274.scalar.count = mul i32 %rows, 16
br label %n274.scalar.loop
n274.scalar.loop:
%n274.scalar.p = phi i32 [ %tid, %n274.scalar.entry ], [ %n274.scalar.next, %n274.scalar.step ]
%n274.scalar.more = icmp ult i32 %n274.scalar.p, %n274.scalar.count
br i1 %n274.scalar.more, label %n274.scalar.body, label %n274.scalar.done
n274.scalar.body:
%n274.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n274.source, i32 %n274.scalar.p
%n274.scalar.first = load float, ptr addrspace(1) %n274.scalar.first.ptr, align 4
%n274.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n274.second, i32 %n274.scalar.p
%n274.scalar.second = load float, ptr addrspace(1) %n274.scalar.second.ptr, align 4
%n274.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n274.scalar.first, float %n274.scalar.second)
%n274.scalar.scalar.0 = call float @recipe.from.u1(i1 %n274.scalar.scalar.0.condition)
%n274.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n274.value, i32 %n274.scalar.p
store float %n274.scalar.scalar.0, ptr addrspace(1) %n274.scalar.output.ptr, align 4
br label %n274.scalar.step
n274.scalar.step:
%n274.scalar.next = add i32 %n274.scalar.p, %threads
br label %n274.scalar.loop
n274.scalar.done:
call void @grid_barrier(i32 %threads)%n275.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n275.second = getelementptr i8, ptr addrspace(1) %values, i64 896
%n275.value = getelementptr i8, ptr addrspace(1) %values, i64 17600
%n275.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2744
%n275.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n275.scalar.entry
n275.scalar.entry:
%n275.scalar.count = mul i32 %rows, 16
br label %n275.scalar.loop
n275.scalar.loop:
%n275.scalar.p = phi i32 [ %tid, %n275.scalar.entry ], [ %n275.scalar.next, %n275.scalar.step ]
%n275.scalar.more = icmp ult i32 %n275.scalar.p, %n275.scalar.count
br i1 %n275.scalar.more, label %n275.scalar.body, label %n275.scalar.done
n275.scalar.body:
%n275.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n275.source, i32 %n275.scalar.p
%n275.scalar.first = load float, ptr addrspace(1) %n275.scalar.first.ptr, align 4
%n275.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n275.second, i32 %n275.scalar.p
%n275.scalar.second = load float, ptr addrspace(1) %n275.scalar.second.ptr, align 4
%n275.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n275.scalar.first, float %n275.scalar.second)
%n275.scalar.scalar.0 = call float @recipe.from.u1(i1 %n275.scalar.scalar.0.condition)
%n275.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n275.value, i32 %n275.scalar.p
store float %n275.scalar.scalar.0, ptr addrspace(1) %n275.scalar.output.ptr, align 4
br label %n275.scalar.step
n275.scalar.step:
%n275.scalar.next = add i32 %n275.scalar.p, %threads
br label %n275.scalar.loop
n275.scalar.done:
call void @grid_barrier(i32 %threads)%n276.source = getelementptr i8, ptr addrspace(1) %values, i64 17536
%n276.second = getelementptr i8, ptr addrspace(1) %values, i64 17600
%n276.value = getelementptr i8, ptr addrspace(1) %values, i64 17664
%n276.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2752
%n276.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n276.scalar.entry
n276.scalar.entry:
%n276.scalar.count = mul i32 %rows, 16
br label %n276.scalar.loop
n276.scalar.loop:
%n276.scalar.p = phi i32 [ %tid, %n276.scalar.entry ], [ %n276.scalar.next, %n276.scalar.step ]
%n276.scalar.more = icmp ult i32 %n276.scalar.p, %n276.scalar.count
br i1 %n276.scalar.more, label %n276.scalar.body, label %n276.scalar.done
n276.scalar.body:
%n276.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n276.source, i32 %n276.scalar.p
%n276.scalar.first = load float, ptr addrspace(1) %n276.scalar.first.ptr, align 4
%n276.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n276.second, i32 %n276.scalar.p
%n276.scalar.second = load float, ptr addrspace(1) %n276.scalar.second.ptr, align 4
%n276.scalar.scalar.0 = call float @recipe.add(float %n276.scalar.first, float %n276.scalar.second)
%n276.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n276.value, i32 %n276.scalar.p
store float %n276.scalar.scalar.0, ptr addrspace(1) %n276.scalar.output.ptr, align 4
br label %n276.scalar.step
n276.scalar.step:
%n276.scalar.next = add i32 %n276.scalar.p, %threads
br label %n276.scalar.loop
n276.scalar.done:
call void @grid_barrier(i32 %threads)%n277.source = getelementptr i8, ptr addrspace(1) %values, i64 17664
%n277.value = getelementptr i8, ptr addrspace(1) %values, i64 17728
%n277.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2760
%n277.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n277.scalar.entry
n277.scalar.entry:
%n277.scalar.count = mul i32 %rows, 16
br label %n277.scalar.loop
n277.scalar.loop:
%n277.scalar.p = phi i32 [ %tid, %n277.scalar.entry ], [ %n277.scalar.next, %n277.scalar.step ]
%n277.scalar.more = icmp ult i32 %n277.scalar.p, %n277.scalar.count
br i1 %n277.scalar.more, label %n277.scalar.body, label %n277.scalar.done
n277.scalar.body:
%n277.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n277.source, i32 %n277.scalar.p
%n277.scalar.first = load float, ptr addrspace(1) %n277.scalar.first.ptr, align 4
%n277.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n277.scalar.first)
%n277.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n277.value, i32 %n277.scalar.p
store float %n277.scalar.scalar.1, ptr addrspace(1) %n277.scalar.output.ptr, align 4
br label %n277.scalar.step
n277.scalar.step:
%n277.scalar.next = add i32 %n277.scalar.p, %threads
br label %n277.scalar.loop
n277.scalar.done:
call void @grid_barrier(i32 %threads)%n278.source = getelementptr i8, ptr addrspace(1) %values, i64 17536
%n278.second = getelementptr i8, ptr addrspace(1) %values, i64 17728
%n278.value = getelementptr i8, ptr addrspace(1) %values, i64 17792
%n278.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2768
%n278.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n278.scalar.entry
n278.scalar.entry:
%n278.scalar.count = mul i32 %rows, 16
br label %n278.scalar.loop
n278.scalar.loop:
%n278.scalar.p = phi i32 [ %tid, %n278.scalar.entry ], [ %n278.scalar.next, %n278.scalar.step ]
%n278.scalar.more = icmp ult i32 %n278.scalar.p, %n278.scalar.count
br i1 %n278.scalar.more, label %n278.scalar.body, label %n278.scalar.done
n278.scalar.body:
%n278.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n278.source, i32 %n278.scalar.p
%n278.scalar.first = load float, ptr addrspace(1) %n278.scalar.first.ptr, align 4
%n278.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n278.second, i32 %n278.scalar.p
%n278.scalar.second = load float, ptr addrspace(1) %n278.scalar.second.ptr, align 4
%n278.scalar.scalar.0 = call float @recipe.add(float %n278.scalar.first, float %n278.scalar.second)
%n278.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n278.value, i32 %n278.scalar.p
store float %n278.scalar.scalar.0, ptr addrspace(1) %n278.scalar.output.ptr, align 4
br label %n278.scalar.step
n278.scalar.step:
%n278.scalar.next = add i32 %n278.scalar.p, %threads
br label %n278.scalar.loop
n278.scalar.done:
call void @grid_barrier(i32 %threads)%n279.source = getelementptr i8, ptr addrspace(1) %values, i64 17472
%n279.second = getelementptr i8, ptr addrspace(1) %values, i64 17792
%n279.value = getelementptr i8, ptr addrspace(1) %values, i64 17856
%n279.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2776
%n279.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n279.scalar.entry
n279.scalar.entry:
%n279.scalar.count = mul i32 %rows, 16
br label %n279.scalar.loop
n279.scalar.loop:
%n279.scalar.p = phi i32 [ %tid, %n279.scalar.entry ], [ %n279.scalar.next, %n279.scalar.step ]
%n279.scalar.more = icmp ult i32 %n279.scalar.p, %n279.scalar.count
br i1 %n279.scalar.more, label %n279.scalar.body, label %n279.scalar.done
n279.scalar.body:
%n279.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n279.source, i32 %n279.scalar.p
%n279.scalar.first = load float, ptr addrspace(1) %n279.scalar.first.ptr, align 4
%n279.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n279.second, i32 %n279.scalar.p
%n279.scalar.second = load float, ptr addrspace(1) %n279.scalar.second.ptr, align 4
%n279.scalar.scalar.0 = call float @recipe.add(float %n279.scalar.first, float %n279.scalar.second)
%n279.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n279.value, i32 %n279.scalar.p
store float %n279.scalar.scalar.0, ptr addrspace(1) %n279.scalar.output.ptr, align 4
br label %n279.scalar.step
n279.scalar.step:
%n279.scalar.next = add i32 %n279.scalar.p, %threads
br label %n279.scalar.loop
n279.scalar.done:
call void @grid_barrier(i32 %threads)%n280.source = getelementptr i8, ptr addrspace(1) %values, i64 960
%n280.second = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n280.value = getelementptr i8, ptr addrspace(1) %values, i64 17920
%n280.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2784
%n280.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n280.scalar.entry
n280.scalar.entry:
%n280.scalar.count = mul i32 %rows, 16
br label %n280.scalar.loop
n280.scalar.loop:
%n280.scalar.p = phi i32 [ %tid, %n280.scalar.entry ], [ %n280.scalar.next, %n280.scalar.step ]
%n280.scalar.more = icmp ult i32 %n280.scalar.p, %n280.scalar.count
br i1 %n280.scalar.more, label %n280.scalar.body, label %n280.scalar.done
n280.scalar.body:
%n280.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n280.source, i32 %n280.scalar.p
%n280.scalar.first = load float, ptr addrspace(1) %n280.scalar.first.ptr, align 4
%n280.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n280.second, i32 %n280.scalar.p
%n280.scalar.second = load float, ptr addrspace(1) %n280.scalar.second.ptr, align 4
%n280.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n280.scalar.first, float %n280.scalar.second)
%n280.scalar.scalar.0 = call float @recipe.from.u1(i1 %n280.scalar.scalar.0.condition)
%n280.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n280.value, i32 %n280.scalar.p
store float %n280.scalar.scalar.0, ptr addrspace(1) %n280.scalar.output.ptr, align 4
br label %n280.scalar.step
n280.scalar.step:
%n280.scalar.next = add i32 %n280.scalar.p, %threads
br label %n280.scalar.loop
n280.scalar.done:
call void @grid_barrier(i32 %threads)%n281.source = getelementptr i8, ptr addrspace(1) %values, i64 1024
%n281.second = getelementptr i8, ptr addrspace(1) %values, i64 960
%n281.value = getelementptr i8, ptr addrspace(1) %values, i64 17984
%n281.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2792
%n281.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n281.scalar.entry
n281.scalar.entry:
%n281.scalar.count = mul i32 %rows, 16
br label %n281.scalar.loop
n281.scalar.loop:
%n281.scalar.p = phi i32 [ %tid, %n281.scalar.entry ], [ %n281.scalar.next, %n281.scalar.step ]
%n281.scalar.more = icmp ult i32 %n281.scalar.p, %n281.scalar.count
br i1 %n281.scalar.more, label %n281.scalar.body, label %n281.scalar.done
n281.scalar.body:
%n281.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n281.source, i32 %n281.scalar.p
%n281.scalar.first = load float, ptr addrspace(1) %n281.scalar.first.ptr, align 4
%n281.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n281.second, i32 %n281.scalar.p
%n281.scalar.second = load float, ptr addrspace(1) %n281.scalar.second.ptr, align 4
%n281.scalar.scalar.0.condition = call i1 @recipe.ogt(float %n281.scalar.first, float %n281.scalar.second)
%n281.scalar.scalar.0 = call float @recipe.from.u1(i1 %n281.scalar.scalar.0.condition)
%n281.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n281.value, i32 %n281.scalar.p
store float %n281.scalar.scalar.0, ptr addrspace(1) %n281.scalar.output.ptr, align 4
br label %n281.scalar.step
n281.scalar.step:
%n281.scalar.next = add i32 %n281.scalar.p, %threads
br label %n281.scalar.loop
n281.scalar.done:
call void @grid_barrier(i32 %threads)%n282.source = getelementptr i8, ptr addrspace(1) %values, i64 17920
%n282.second = getelementptr i8, ptr addrspace(1) %values, i64 17984
%n282.value = getelementptr i8, ptr addrspace(1) %values, i64 18048
%n282.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2800
%n282.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n282.scalar.entry
n282.scalar.entry:
%n282.scalar.count = mul i32 %rows, 16
br label %n282.scalar.loop
n282.scalar.loop:
%n282.scalar.p = phi i32 [ %tid, %n282.scalar.entry ], [ %n282.scalar.next, %n282.scalar.step ]
%n282.scalar.more = icmp ult i32 %n282.scalar.p, %n282.scalar.count
br i1 %n282.scalar.more, label %n282.scalar.body, label %n282.scalar.done
n282.scalar.body:
%n282.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n282.source, i32 %n282.scalar.p
%n282.scalar.first = load float, ptr addrspace(1) %n282.scalar.first.ptr, align 4
%n282.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n282.second, i32 %n282.scalar.p
%n282.scalar.second = load float, ptr addrspace(1) %n282.scalar.second.ptr, align 4
%n282.scalar.scalar.0 = call float @recipe.add(float %n282.scalar.first, float %n282.scalar.second)
%n282.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n282.value, i32 %n282.scalar.p
store float %n282.scalar.scalar.0, ptr addrspace(1) %n282.scalar.output.ptr, align 4
br label %n282.scalar.step
n282.scalar.step:
%n282.scalar.next = add i32 %n282.scalar.p, %threads
br label %n282.scalar.loop
n282.scalar.done:
call void @grid_barrier(i32 %threads)%n283.source = getelementptr i8, ptr addrspace(1) %values, i64 18048
%n283.value = getelementptr i8, ptr addrspace(1) %values, i64 18112
%n283.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2808
%n283.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n283.scalar.entry
n283.scalar.entry:
%n283.scalar.count = mul i32 %rows, 16
br label %n283.scalar.loop
n283.scalar.loop:
%n283.scalar.p = phi i32 [ %tid, %n283.scalar.entry ], [ %n283.scalar.next, %n283.scalar.step ]
%n283.scalar.more = icmp ult i32 %n283.scalar.p, %n283.scalar.count
br i1 %n283.scalar.more, label %n283.scalar.body, label %n283.scalar.done
n283.scalar.body:
%n283.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n283.source, i32 %n283.scalar.p
%n283.scalar.first = load float, ptr addrspace(1) %n283.scalar.first.ptr, align 4
%n283.scalar.scalar.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n283.scalar.first)
%n283.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n283.value, i32 %n283.scalar.p
store float %n283.scalar.scalar.1, ptr addrspace(1) %n283.scalar.output.ptr, align 4
br label %n283.scalar.step
n283.scalar.step:
%n283.scalar.next = add i32 %n283.scalar.p, %threads
br label %n283.scalar.loop
n283.scalar.done:
call void @grid_barrier(i32 %threads)%n284.source = getelementptr i8, ptr addrspace(1) %values, i64 17920
%n284.second = getelementptr i8, ptr addrspace(1) %values, i64 18112
%n284.value = getelementptr i8, ptr addrspace(1) %values, i64 18176
%n284.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2816
%n284.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n284.scalar.entry
n284.scalar.entry:
%n284.scalar.count = mul i32 %rows, 16
br label %n284.scalar.loop
n284.scalar.loop:
%n284.scalar.p = phi i32 [ %tid, %n284.scalar.entry ], [ %n284.scalar.next, %n284.scalar.step ]
%n284.scalar.more = icmp ult i32 %n284.scalar.p, %n284.scalar.count
br i1 %n284.scalar.more, label %n284.scalar.body, label %n284.scalar.done
n284.scalar.body:
%n284.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n284.source, i32 %n284.scalar.p
%n284.scalar.first = load float, ptr addrspace(1) %n284.scalar.first.ptr, align 4
%n284.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n284.second, i32 %n284.scalar.p
%n284.scalar.second = load float, ptr addrspace(1) %n284.scalar.second.ptr, align 4
%n284.scalar.scalar.0 = call float @recipe.add(float %n284.scalar.first, float %n284.scalar.second)
%n284.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n284.value, i32 %n284.scalar.p
store float %n284.scalar.scalar.0, ptr addrspace(1) %n284.scalar.output.ptr, align 4
br label %n284.scalar.step
n284.scalar.step:
%n284.scalar.next = add i32 %n284.scalar.p, %threads
br label %n284.scalar.loop
n284.scalar.done:
call void @grid_barrier(i32 %threads)%n285.source = getelementptr i8, ptr addrspace(1) %values, i64 17856
%n285.second = getelementptr i8, ptr addrspace(1) %values, i64 18176
%n285.value = getelementptr i8, ptr addrspace(1) %values, i64 18240
%n285.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2824
%n285.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n285.scalar.entry
n285.scalar.entry:
%n285.scalar.count = mul i32 %rows, 16
br label %n285.scalar.loop
n285.scalar.loop:
%n285.scalar.p = phi i32 [ %tid, %n285.scalar.entry ], [ %n285.scalar.next, %n285.scalar.step ]
%n285.scalar.more = icmp ult i32 %n285.scalar.p, %n285.scalar.count
br i1 %n285.scalar.more, label %n285.scalar.body, label %n285.scalar.done
n285.scalar.body:
%n285.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n285.source, i32 %n285.scalar.p
%n285.scalar.first = load float, ptr addrspace(1) %n285.scalar.first.ptr, align 4
%n285.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n285.second, i32 %n285.scalar.p
%n285.scalar.second = load float, ptr addrspace(1) %n285.scalar.second.ptr, align 4
%n285.scalar.scalar.0 = call float @recipe.add(float %n285.scalar.first, float %n285.scalar.second)
%n285.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n285.value, i32 %n285.scalar.p
store float %n285.scalar.scalar.0, ptr addrspace(1) %n285.scalar.output.ptr, align 4
br label %n285.scalar.step
n285.scalar.step:
%n285.scalar.next = add i32 %n285.scalar.p, %threads
br label %n285.scalar.loop
n285.scalar.done:
call void @grid_barrier(i32 %threads)%n286.source = getelementptr i8, ptr addrspace(1) %values, i64 18240
%n286.value = getelementptr i8, ptr addrspace(1) %values, i64 18304
%n286.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2832
%n286.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n286.scalar.entry
n286.scalar.entry:
%n286.scalar.count = mul i32 %rows, 16
br label %n286.scalar.loop
n286.scalar.loop:
%n286.scalar.p = phi i32 [ %tid, %n286.scalar.entry ], [ %n286.scalar.next, %n286.scalar.step ]
%n286.scalar.more = icmp ult i32 %n286.scalar.p, %n286.scalar.count
br i1 %n286.scalar.more, label %n286.scalar.body, label %n286.scalar.done
n286.scalar.body:
%n286.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n286.source, i32 %n286.scalar.p
%n286.scalar.first = load float, ptr addrspace(1) %n286.scalar.first.ptr, align 4
%n286.scalar.scalar.1.condition = call i1 @recipe.ogt(float 0x4010000000000000, float %n286.scalar.first)
%n286.scalar.scalar.1 = call float @recipe.from.u1(i1 %n286.scalar.scalar.1.condition)
%n286.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n286.value, i32 %n286.scalar.p
store float %n286.scalar.scalar.1, ptr addrspace(1) %n286.scalar.output.ptr, align 4
br label %n286.scalar.step
n286.scalar.step:
%n286.scalar.next = add i32 %n286.scalar.p, %threads
br label %n286.scalar.loop
n286.scalar.done:
call void @grid_barrier(i32 %threads)%n287.source = getelementptr i8, ptr addrspace(1) %values, i64 18304
%n287.second = getelementptr i8, ptr addrspace(1) %values, i64 15488
%n287.value = getelementptr i8, ptr addrspace(1) %values, i64 18368
%n287.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2840
%n287.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n287.scalar.entry
n287.scalar.entry:
%n287.scalar.count = mul i32 %rows, 16
br label %n287.scalar.loop
n287.scalar.loop:
%n287.scalar.p = phi i32 [ %tid, %n287.scalar.entry ], [ %n287.scalar.next, %n287.scalar.step ]
%n287.scalar.more = icmp ult i32 %n287.scalar.p, %n287.scalar.count
br i1 %n287.scalar.more, label %n287.scalar.body, label %n287.scalar.done
n287.scalar.body:
%n287.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n287.source, i32 %n287.scalar.p
%n287.scalar.first = load float, ptr addrspace(1) %n287.scalar.first.ptr, align 4
%n287.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n287.second, i32 %n287.scalar.p
%n287.scalar.second = load float, ptr addrspace(1) %n287.scalar.second.ptr, align 4
%n287.scalar.scalar.0 = call float @recipe.mul(float %n287.scalar.first, float %n287.scalar.second)
%n287.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n287.value, i32 %n287.scalar.p
store float %n287.scalar.scalar.0, ptr addrspace(1) %n287.scalar.output.ptr, align 4
br label %n287.scalar.step
n287.scalar.step:
%n287.scalar.next = add i32 %n287.scalar.p, %threads
br label %n287.scalar.loop
n287.scalar.done:
call void @grid_barrier(i32 %threads)%n288.source = getelementptr i8, ptr addrspace(1) %values, i64 2688
%n288.second = getelementptr i8, ptr addrspace(1) %values, i64 4160
%n288.value = getelementptr i8, ptr addrspace(1) %values, i64 18432
%n288.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2848
%n288.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n288.scalar.entry
n288.scalar.entry:
%n288.scalar.count = mul i32 %rows, 16
br label %n288.scalar.loop
n288.scalar.loop:
%n288.scalar.p = phi i32 [ %tid, %n288.scalar.entry ], [ %n288.scalar.next, %n288.scalar.step ]
%n288.scalar.more = icmp ult i32 %n288.scalar.p, %n288.scalar.count
br i1 %n288.scalar.more, label %n288.scalar.body, label %n288.scalar.done
n288.scalar.body:
%n288.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n288.source, i32 %n288.scalar.p
%n288.scalar.first = load float, ptr addrspace(1) %n288.scalar.first.ptr, align 4
%n288.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n288.second, i32 %n288.scalar.p
%n288.scalar.second = load float, ptr addrspace(1) %n288.scalar.second.ptr, align 4
%n288.scalar.scalar.0 = call float @recipe.add(float %n288.scalar.first, float %n288.scalar.second)
%n288.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n288.value, i32 %n288.scalar.p
store float %n288.scalar.scalar.0, ptr addrspace(1) %n288.scalar.output.ptr, align 4
br label %n288.scalar.step
n288.scalar.step:
%n288.scalar.next = add i32 %n288.scalar.p, %threads
br label %n288.scalar.loop
n288.scalar.done:
call void @grid_barrier(i32 %threads)%n289.source = getelementptr i8, ptr addrspace(1) %values, i64 18432
%n289.second = getelementptr i8, ptr addrspace(1) %values, i64 5888
%n289.value = getelementptr i8, ptr addrspace(1) %values, i64 18496
%n289.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2856
%n289.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n289.scalar.entry
n289.scalar.entry:
%n289.scalar.count = mul i32 %rows, 16
br label %n289.scalar.loop
n289.scalar.loop:
%n289.scalar.p = phi i32 [ %tid, %n289.scalar.entry ], [ %n289.scalar.next, %n289.scalar.step ]
%n289.scalar.more = icmp ult i32 %n289.scalar.p, %n289.scalar.count
br i1 %n289.scalar.more, label %n289.scalar.body, label %n289.scalar.done
n289.scalar.body:
%n289.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n289.source, i32 %n289.scalar.p
%n289.scalar.first = load float, ptr addrspace(1) %n289.scalar.first.ptr, align 4
%n289.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n289.second, i32 %n289.scalar.p
%n289.scalar.second = load float, ptr addrspace(1) %n289.scalar.second.ptr, align 4
%n289.scalar.scalar.0 = call float @recipe.add(float %n289.scalar.first, float %n289.scalar.second)
%n289.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n289.value, i32 %n289.scalar.p
store float %n289.scalar.scalar.0, ptr addrspace(1) %n289.scalar.output.ptr, align 4
br label %n289.scalar.step
n289.scalar.step:
%n289.scalar.next = add i32 %n289.scalar.p, %threads
br label %n289.scalar.loop
n289.scalar.done:
call void @grid_barrier(i32 %threads)%n290.source = getelementptr i8, ptr addrspace(1) %values, i64 18496
%n290.second = getelementptr i8, ptr addrspace(1) %values, i64 7872
%n290.value = getelementptr i8, ptr addrspace(1) %values, i64 18560
%n290.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2864
%n290.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n290.scalar.entry
n290.scalar.entry:
%n290.scalar.count = mul i32 %rows, 16
br label %n290.scalar.loop
n290.scalar.loop:
%n290.scalar.p = phi i32 [ %tid, %n290.scalar.entry ], [ %n290.scalar.next, %n290.scalar.step ]
%n290.scalar.more = icmp ult i32 %n290.scalar.p, %n290.scalar.count
br i1 %n290.scalar.more, label %n290.scalar.body, label %n290.scalar.done
n290.scalar.body:
%n290.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n290.source, i32 %n290.scalar.p
%n290.scalar.first = load float, ptr addrspace(1) %n290.scalar.first.ptr, align 4
%n290.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n290.second, i32 %n290.scalar.p
%n290.scalar.second = load float, ptr addrspace(1) %n290.scalar.second.ptr, align 4
%n290.scalar.scalar.0 = call float @recipe.add(float %n290.scalar.first, float %n290.scalar.second)
%n290.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n290.value, i32 %n290.scalar.p
store float %n290.scalar.scalar.0, ptr addrspace(1) %n290.scalar.output.ptr, align 4
br label %n290.scalar.step
n290.scalar.step:
%n290.scalar.next = add i32 %n290.scalar.p, %threads
br label %n290.scalar.loop
n290.scalar.done:
call void @grid_barrier(i32 %threads)%n291.source = getelementptr i8, ptr addrspace(1) %values, i64 18560
%n291.second = getelementptr i8, ptr addrspace(1) %values, i64 10112
%n291.value = getelementptr i8, ptr addrspace(1) %values, i64 18624
%n291.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2872
%n291.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n291.scalar.entry
n291.scalar.entry:
%n291.scalar.count = mul i32 %rows, 16
br label %n291.scalar.loop
n291.scalar.loop:
%n291.scalar.p = phi i32 [ %tid, %n291.scalar.entry ], [ %n291.scalar.next, %n291.scalar.step ]
%n291.scalar.more = icmp ult i32 %n291.scalar.p, %n291.scalar.count
br i1 %n291.scalar.more, label %n291.scalar.body, label %n291.scalar.done
n291.scalar.body:
%n291.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n291.source, i32 %n291.scalar.p
%n291.scalar.first = load float, ptr addrspace(1) %n291.scalar.first.ptr, align 4
%n291.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n291.second, i32 %n291.scalar.p
%n291.scalar.second = load float, ptr addrspace(1) %n291.scalar.second.ptr, align 4
%n291.scalar.scalar.0 = call float @recipe.add(float %n291.scalar.first, float %n291.scalar.second)
%n291.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n291.value, i32 %n291.scalar.p
store float %n291.scalar.scalar.0, ptr addrspace(1) %n291.scalar.output.ptr, align 4
br label %n291.scalar.step
n291.scalar.step:
%n291.scalar.next = add i32 %n291.scalar.p, %threads
br label %n291.scalar.loop
n291.scalar.done:
call void @grid_barrier(i32 %threads)%n292.source = getelementptr i8, ptr addrspace(1) %values, i64 18624
%n292.second = getelementptr i8, ptr addrspace(1) %values, i64 12608
%n292.value = getelementptr i8, ptr addrspace(1) %values, i64 18688
%n292.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2880
%n292.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n292.scalar.entry
n292.scalar.entry:
%n292.scalar.count = mul i32 %rows, 16
br label %n292.scalar.loop
n292.scalar.loop:
%n292.scalar.p = phi i32 [ %tid, %n292.scalar.entry ], [ %n292.scalar.next, %n292.scalar.step ]
%n292.scalar.more = icmp ult i32 %n292.scalar.p, %n292.scalar.count
br i1 %n292.scalar.more, label %n292.scalar.body, label %n292.scalar.done
n292.scalar.body:
%n292.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n292.source, i32 %n292.scalar.p
%n292.scalar.first = load float, ptr addrspace(1) %n292.scalar.first.ptr, align 4
%n292.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n292.second, i32 %n292.scalar.p
%n292.scalar.second = load float, ptr addrspace(1) %n292.scalar.second.ptr, align 4
%n292.scalar.scalar.0 = call float @recipe.add(float %n292.scalar.first, float %n292.scalar.second)
%n292.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n292.value, i32 %n292.scalar.p
store float %n292.scalar.scalar.0, ptr addrspace(1) %n292.scalar.output.ptr, align 4
br label %n292.scalar.step
n292.scalar.step:
%n292.scalar.next = add i32 %n292.scalar.p, %threads
br label %n292.scalar.loop
n292.scalar.done:
call void @grid_barrier(i32 %threads)%n293.source = getelementptr i8, ptr addrspace(1) %values, i64 18688
%n293.second = getelementptr i8, ptr addrspace(1) %values, i64 15360
%n293.value = getelementptr i8, ptr addrspace(1) %values, i64 18752
%n293.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2888
%n293.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n293.scalar.entry
n293.scalar.entry:
%n293.scalar.count = mul i32 %rows, 16
br label %n293.scalar.loop
n293.scalar.loop:
%n293.scalar.p = phi i32 [ %tid, %n293.scalar.entry ], [ %n293.scalar.next, %n293.scalar.step ]
%n293.scalar.more = icmp ult i32 %n293.scalar.p, %n293.scalar.count
br i1 %n293.scalar.more, label %n293.scalar.body, label %n293.scalar.done
n293.scalar.body:
%n293.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n293.source, i32 %n293.scalar.p
%n293.scalar.first = load float, ptr addrspace(1) %n293.scalar.first.ptr, align 4
%n293.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n293.second, i32 %n293.scalar.p
%n293.scalar.second = load float, ptr addrspace(1) %n293.scalar.second.ptr, align 4
%n293.scalar.scalar.0 = call float @recipe.add(float %n293.scalar.first, float %n293.scalar.second)
%n293.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n293.value, i32 %n293.scalar.p
store float %n293.scalar.scalar.0, ptr addrspace(1) %n293.scalar.output.ptr, align 4
br label %n293.scalar.step
n293.scalar.step:
%n293.scalar.next = add i32 %n293.scalar.p, %threads
br label %n293.scalar.loop
n293.scalar.done:
call void @grid_barrier(i32 %threads)%n294.source = getelementptr i8, ptr addrspace(1) %values, i64 18752
%n294.second = getelementptr i8, ptr addrspace(1) %values, i64 18368
%n294.value = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n294.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2896
%n294.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n294.scalar.entry
n294.scalar.entry:
%n294.scalar.count = mul i32 %rows, 16
br label %n294.scalar.loop
n294.scalar.loop:
%n294.scalar.p = phi i32 [ %tid, %n294.scalar.entry ], [ %n294.scalar.next, %n294.scalar.step ]
%n294.scalar.more = icmp ult i32 %n294.scalar.p, %n294.scalar.count
br i1 %n294.scalar.more, label %n294.scalar.body, label %n294.scalar.done
n294.scalar.body:
%n294.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n294.source, i32 %n294.scalar.p
%n294.scalar.first = load float, ptr addrspace(1) %n294.scalar.first.ptr, align 4
%n294.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n294.second, i32 %n294.scalar.p
%n294.scalar.second = load float, ptr addrspace(1) %n294.scalar.second.ptr, align 4
%n294.scalar.scalar.0 = call float @recipe.add(float %n294.scalar.first, float %n294.scalar.second)
%n294.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n294.value, i32 %n294.scalar.p
store float %n294.scalar.scalar.0, ptr addrspace(1) %n294.scalar.output.ptr, align 4
br label %n294.scalar.step
n294.scalar.step:
%n294.scalar.next = add i32 %n294.scalar.p, %threads
br label %n294.scalar.loop
n294.scalar.done:
call void @grid_barrier(i32 %threads)%n295.source = getelementptr i8, ptr addrspace(1) %values, i64 2688
%n295.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n295.value = getelementptr i8, ptr addrspace(1) %values, i64 18880
%n295.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2904
%n295.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n295.scalar.entry
n295.scalar.entry:
%n295.scalar.count = mul i32 %rows, 16
br label %n295.scalar.loop
n295.scalar.loop:
%n295.scalar.p = phi i32 [ %tid, %n295.scalar.entry ], [ %n295.scalar.next, %n295.scalar.step ]
%n295.scalar.more = icmp ult i32 %n295.scalar.p, %n295.scalar.count
br i1 %n295.scalar.more, label %n295.scalar.body, label %n295.scalar.done
n295.scalar.body:
%n295.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n295.source, i32 %n295.scalar.p
%n295.scalar.first = load float, ptr addrspace(1) %n295.scalar.first.ptr, align 4
%n295.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n295.second, i32 %n295.scalar.p
%n295.scalar.second = load float, ptr addrspace(1) %n295.scalar.second.ptr, align 4
%n295.scalar.scalar.0 = call float @recipe.div(float %n295.scalar.first, float %n295.scalar.second)
%n295.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n295.value, i32 %n295.scalar.p
store float %n295.scalar.scalar.0, ptr addrspace(1) %n295.scalar.output.ptr, align 4
br label %n295.scalar.step
n295.scalar.step:
%n295.scalar.next = add i32 %n295.scalar.p, %threads
br label %n295.scalar.loop
n295.scalar.done:
call void @grid_barrier(i32 %threads)%n296.source = getelementptr i8, ptr addrspace(1) %values, i64 18880
%n296.second = getelementptr i8, ptr addrspace(1) %values, i64 64
%n296.value = getelementptr i8, ptr addrspace(1) %values, i64 18944
%n296.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2912
%n296.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n296.scalar.entry
n296.scalar.entry:
%n296.scalar.count = mul i32 %rows, 16
br label %n296.scalar.loop
n296.scalar.loop:
%n296.scalar.p = phi i32 [ %tid, %n296.scalar.entry ], [ %n296.scalar.next, %n296.scalar.step ]
%n296.scalar.more = icmp ult i32 %n296.scalar.p, %n296.scalar.count
br i1 %n296.scalar.more, label %n296.scalar.body, label %n296.scalar.done
n296.scalar.body:
%n296.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n296.source, i32 %n296.scalar.p
%n296.scalar.first = load float, ptr addrspace(1) %n296.scalar.first.ptr, align 4
%n296.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n296.second, i32 %n296.scalar.p
%n296.scalar.second = load float, ptr addrspace(1) %n296.scalar.second.ptr, align 4
%n296.scalar.scalar.0 = call float @recipe.mul(float %n296.scalar.first, float %n296.scalar.second)
%n296.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n296.value, i32 %n296.scalar.p
store float %n296.scalar.scalar.0, ptr addrspace(1) %n296.scalar.output.ptr, align 4
br label %n296.scalar.step
n296.scalar.step:
%n296.scalar.next = add i32 %n296.scalar.p, %threads
br label %n296.scalar.loop
n296.scalar.done:
call void @grid_barrier(i32 %threads)%n297.source = getelementptr i8, ptr addrspace(1) %values, i64 4160
%n297.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n297.value = getelementptr i8, ptr addrspace(1) %values, i64 19008
%n297.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2920
%n297.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n297.scalar.entry
n297.scalar.entry:
%n297.scalar.count = mul i32 %rows, 16
br label %n297.scalar.loop
n297.scalar.loop:
%n297.scalar.p = phi i32 [ %tid, %n297.scalar.entry ], [ %n297.scalar.next, %n297.scalar.step ]
%n297.scalar.more = icmp ult i32 %n297.scalar.p, %n297.scalar.count
br i1 %n297.scalar.more, label %n297.scalar.body, label %n297.scalar.done
n297.scalar.body:
%n297.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n297.source, i32 %n297.scalar.p
%n297.scalar.first = load float, ptr addrspace(1) %n297.scalar.first.ptr, align 4
%n297.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n297.second, i32 %n297.scalar.p
%n297.scalar.second = load float, ptr addrspace(1) %n297.scalar.second.ptr, align 4
%n297.scalar.scalar.0 = call float @recipe.div(float %n297.scalar.first, float %n297.scalar.second)
%n297.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n297.value, i32 %n297.scalar.p
store float %n297.scalar.scalar.0, ptr addrspace(1) %n297.scalar.output.ptr, align 4
br label %n297.scalar.step
n297.scalar.step:
%n297.scalar.next = add i32 %n297.scalar.p, %threads
br label %n297.scalar.loop
n297.scalar.done:
call void @grid_barrier(i32 %threads)%n298.source = getelementptr i8, ptr addrspace(1) %values, i64 19008
%n298.second = getelementptr i8, ptr addrspace(1) %values, i64 128
%n298.value = getelementptr i8, ptr addrspace(1) %values, i64 19072
%n298.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2928
%n298.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n298.scalar.entry
n298.scalar.entry:
%n298.scalar.count = mul i32 %rows, 16
br label %n298.scalar.loop
n298.scalar.loop:
%n298.scalar.p = phi i32 [ %tid, %n298.scalar.entry ], [ %n298.scalar.next, %n298.scalar.step ]
%n298.scalar.more = icmp ult i32 %n298.scalar.p, %n298.scalar.count
br i1 %n298.scalar.more, label %n298.scalar.body, label %n298.scalar.done
n298.scalar.body:
%n298.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n298.source, i32 %n298.scalar.p
%n298.scalar.first = load float, ptr addrspace(1) %n298.scalar.first.ptr, align 4
%n298.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n298.second, i32 %n298.scalar.p
%n298.scalar.second = load float, ptr addrspace(1) %n298.scalar.second.ptr, align 4
%n298.scalar.scalar.0 = call float @recipe.mul(float %n298.scalar.first, float %n298.scalar.second)
%n298.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n298.value, i32 %n298.scalar.p
store float %n298.scalar.scalar.0, ptr addrspace(1) %n298.scalar.output.ptr, align 4
br label %n298.scalar.step
n298.scalar.step:
%n298.scalar.next = add i32 %n298.scalar.p, %threads
br label %n298.scalar.loop
n298.scalar.done:
call void @grid_barrier(i32 %threads)%n299.source = getelementptr i8, ptr addrspace(1) %values, i64 18944
%n299.second = getelementptr i8, ptr addrspace(1) %values, i64 19072
%n299.value = getelementptr i8, ptr addrspace(1) %values, i64 19136
%n299.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2936
%n299.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n299.scalar.entry
n299.scalar.entry:
%n299.scalar.count = mul i32 %rows, 16
br label %n299.scalar.loop
n299.scalar.loop:
%n299.scalar.p = phi i32 [ %tid, %n299.scalar.entry ], [ %n299.scalar.next, %n299.scalar.step ]
%n299.scalar.more = icmp ult i32 %n299.scalar.p, %n299.scalar.count
br i1 %n299.scalar.more, label %n299.scalar.body, label %n299.scalar.done
n299.scalar.body:
%n299.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n299.source, i32 %n299.scalar.p
%n299.scalar.first = load float, ptr addrspace(1) %n299.scalar.first.ptr, align 4
%n299.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n299.second, i32 %n299.scalar.p
%n299.scalar.second = load float, ptr addrspace(1) %n299.scalar.second.ptr, align 4
%n299.scalar.scalar.0 = call float @recipe.add(float %n299.scalar.first, float %n299.scalar.second)
%n299.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n299.value, i32 %n299.scalar.p
store float %n299.scalar.scalar.0, ptr addrspace(1) %n299.scalar.output.ptr, align 4
br label %n299.scalar.step
n299.scalar.step:
%n299.scalar.next = add i32 %n299.scalar.p, %threads
br label %n299.scalar.loop
n299.scalar.done:
call void @grid_barrier(i32 %threads)%n300.source = getelementptr i8, ptr addrspace(1) %values, i64 5888
%n300.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n300.value = getelementptr i8, ptr addrspace(1) %values, i64 19200
%n300.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2944
%n300.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n300.scalar.entry
n300.scalar.entry:
%n300.scalar.count = mul i32 %rows, 16
br label %n300.scalar.loop
n300.scalar.loop:
%n300.scalar.p = phi i32 [ %tid, %n300.scalar.entry ], [ %n300.scalar.next, %n300.scalar.step ]
%n300.scalar.more = icmp ult i32 %n300.scalar.p, %n300.scalar.count
br i1 %n300.scalar.more, label %n300.scalar.body, label %n300.scalar.done
n300.scalar.body:
%n300.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n300.source, i32 %n300.scalar.p
%n300.scalar.first = load float, ptr addrspace(1) %n300.scalar.first.ptr, align 4
%n300.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n300.second, i32 %n300.scalar.p
%n300.scalar.second = load float, ptr addrspace(1) %n300.scalar.second.ptr, align 4
%n300.scalar.scalar.0 = call float @recipe.div(float %n300.scalar.first, float %n300.scalar.second)
%n300.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n300.value, i32 %n300.scalar.p
store float %n300.scalar.scalar.0, ptr addrspace(1) %n300.scalar.output.ptr, align 4
br label %n300.scalar.step
n300.scalar.step:
%n300.scalar.next = add i32 %n300.scalar.p, %threads
br label %n300.scalar.loop
n300.scalar.done:
call void @grid_barrier(i32 %threads)%n301.source = getelementptr i8, ptr addrspace(1) %values, i64 19200
%n301.second = getelementptr i8, ptr addrspace(1) %values, i64 192
%n301.value = getelementptr i8, ptr addrspace(1) %values, i64 19264
%n301.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2952
%n301.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n301.scalar.entry
n301.scalar.entry:
%n301.scalar.count = mul i32 %rows, 16
br label %n301.scalar.loop
n301.scalar.loop:
%n301.scalar.p = phi i32 [ %tid, %n301.scalar.entry ], [ %n301.scalar.next, %n301.scalar.step ]
%n301.scalar.more = icmp ult i32 %n301.scalar.p, %n301.scalar.count
br i1 %n301.scalar.more, label %n301.scalar.body, label %n301.scalar.done
n301.scalar.body:
%n301.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n301.source, i32 %n301.scalar.p
%n301.scalar.first = load float, ptr addrspace(1) %n301.scalar.first.ptr, align 4
%n301.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n301.second, i32 %n301.scalar.p
%n301.scalar.second = load float, ptr addrspace(1) %n301.scalar.second.ptr, align 4
%n301.scalar.scalar.0 = call float @recipe.mul(float %n301.scalar.first, float %n301.scalar.second)
%n301.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n301.value, i32 %n301.scalar.p
store float %n301.scalar.scalar.0, ptr addrspace(1) %n301.scalar.output.ptr, align 4
br label %n301.scalar.step
n301.scalar.step:
%n301.scalar.next = add i32 %n301.scalar.p, %threads
br label %n301.scalar.loop
n301.scalar.done:
call void @grid_barrier(i32 %threads)%n302.source = getelementptr i8, ptr addrspace(1) %values, i64 19136
%n302.second = getelementptr i8, ptr addrspace(1) %values, i64 19264
%n302.value = getelementptr i8, ptr addrspace(1) %values, i64 19328
%n302.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2960
%n302.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n302.scalar.entry
n302.scalar.entry:
%n302.scalar.count = mul i32 %rows, 16
br label %n302.scalar.loop
n302.scalar.loop:
%n302.scalar.p = phi i32 [ %tid, %n302.scalar.entry ], [ %n302.scalar.next, %n302.scalar.step ]
%n302.scalar.more = icmp ult i32 %n302.scalar.p, %n302.scalar.count
br i1 %n302.scalar.more, label %n302.scalar.body, label %n302.scalar.done
n302.scalar.body:
%n302.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n302.source, i32 %n302.scalar.p
%n302.scalar.first = load float, ptr addrspace(1) %n302.scalar.first.ptr, align 4
%n302.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n302.second, i32 %n302.scalar.p
%n302.scalar.second = load float, ptr addrspace(1) %n302.scalar.second.ptr, align 4
%n302.scalar.scalar.0 = call float @recipe.add(float %n302.scalar.first, float %n302.scalar.second)
%n302.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n302.value, i32 %n302.scalar.p
store float %n302.scalar.scalar.0, ptr addrspace(1) %n302.scalar.output.ptr, align 4
br label %n302.scalar.step
n302.scalar.step:
%n302.scalar.next = add i32 %n302.scalar.p, %threads
br label %n302.scalar.loop
n302.scalar.done:
call void @grid_barrier(i32 %threads)%n303.source = getelementptr i8, ptr addrspace(1) %values, i64 7872
%n303.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n303.value = getelementptr i8, ptr addrspace(1) %values, i64 19392
%n303.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2968
%n303.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n303.scalar.entry
n303.scalar.entry:
%n303.scalar.count = mul i32 %rows, 16
br label %n303.scalar.loop
n303.scalar.loop:
%n303.scalar.p = phi i32 [ %tid, %n303.scalar.entry ], [ %n303.scalar.next, %n303.scalar.step ]
%n303.scalar.more = icmp ult i32 %n303.scalar.p, %n303.scalar.count
br i1 %n303.scalar.more, label %n303.scalar.body, label %n303.scalar.done
n303.scalar.body:
%n303.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n303.source, i32 %n303.scalar.p
%n303.scalar.first = load float, ptr addrspace(1) %n303.scalar.first.ptr, align 4
%n303.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n303.second, i32 %n303.scalar.p
%n303.scalar.second = load float, ptr addrspace(1) %n303.scalar.second.ptr, align 4
%n303.scalar.scalar.0 = call float @recipe.div(float %n303.scalar.first, float %n303.scalar.second)
%n303.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n303.value, i32 %n303.scalar.p
store float %n303.scalar.scalar.0, ptr addrspace(1) %n303.scalar.output.ptr, align 4
br label %n303.scalar.step
n303.scalar.step:
%n303.scalar.next = add i32 %n303.scalar.p, %threads
br label %n303.scalar.loop
n303.scalar.done:
call void @grid_barrier(i32 %threads)%n304.source = getelementptr i8, ptr addrspace(1) %values, i64 19392
%n304.second = getelementptr i8, ptr addrspace(1) %values, i64 256
%n304.value = getelementptr i8, ptr addrspace(1) %values, i64 19456
%n304.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2976
%n304.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n304.scalar.entry
n304.scalar.entry:
%n304.scalar.count = mul i32 %rows, 16
br label %n304.scalar.loop
n304.scalar.loop:
%n304.scalar.p = phi i32 [ %tid, %n304.scalar.entry ], [ %n304.scalar.next, %n304.scalar.step ]
%n304.scalar.more = icmp ult i32 %n304.scalar.p, %n304.scalar.count
br i1 %n304.scalar.more, label %n304.scalar.body, label %n304.scalar.done
n304.scalar.body:
%n304.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n304.source, i32 %n304.scalar.p
%n304.scalar.first = load float, ptr addrspace(1) %n304.scalar.first.ptr, align 4
%n304.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n304.second, i32 %n304.scalar.p
%n304.scalar.second = load float, ptr addrspace(1) %n304.scalar.second.ptr, align 4
%n304.scalar.scalar.0 = call float @recipe.mul(float %n304.scalar.first, float %n304.scalar.second)
%n304.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n304.value, i32 %n304.scalar.p
store float %n304.scalar.scalar.0, ptr addrspace(1) %n304.scalar.output.ptr, align 4
br label %n304.scalar.step
n304.scalar.step:
%n304.scalar.next = add i32 %n304.scalar.p, %threads
br label %n304.scalar.loop
n304.scalar.done:
call void @grid_barrier(i32 %threads)%n305.source = getelementptr i8, ptr addrspace(1) %values, i64 19328
%n305.second = getelementptr i8, ptr addrspace(1) %values, i64 19456
%n305.value = getelementptr i8, ptr addrspace(1) %values, i64 19520
%n305.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2984
%n305.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n305.scalar.entry
n305.scalar.entry:
%n305.scalar.count = mul i32 %rows, 16
br label %n305.scalar.loop
n305.scalar.loop:
%n305.scalar.p = phi i32 [ %tid, %n305.scalar.entry ], [ %n305.scalar.next, %n305.scalar.step ]
%n305.scalar.more = icmp ult i32 %n305.scalar.p, %n305.scalar.count
br i1 %n305.scalar.more, label %n305.scalar.body, label %n305.scalar.done
n305.scalar.body:
%n305.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n305.source, i32 %n305.scalar.p
%n305.scalar.first = load float, ptr addrspace(1) %n305.scalar.first.ptr, align 4
%n305.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n305.second, i32 %n305.scalar.p
%n305.scalar.second = load float, ptr addrspace(1) %n305.scalar.second.ptr, align 4
%n305.scalar.scalar.0 = call float @recipe.add(float %n305.scalar.first, float %n305.scalar.second)
%n305.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n305.value, i32 %n305.scalar.p
store float %n305.scalar.scalar.0, ptr addrspace(1) %n305.scalar.output.ptr, align 4
br label %n305.scalar.step
n305.scalar.step:
%n305.scalar.next = add i32 %n305.scalar.p, %threads
br label %n305.scalar.loop
n305.scalar.done:
call void @grid_barrier(i32 %threads)%n306.source = getelementptr i8, ptr addrspace(1) %values, i64 10112
%n306.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n306.value = getelementptr i8, ptr addrspace(1) %values, i64 19584
%n306.context = getelementptr i8, ptr addrspace(1) %contexts, i64 2992
%n306.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n306.scalar.entry
n306.scalar.entry:
%n306.scalar.count = mul i32 %rows, 16
br label %n306.scalar.loop
n306.scalar.loop:
%n306.scalar.p = phi i32 [ %tid, %n306.scalar.entry ], [ %n306.scalar.next, %n306.scalar.step ]
%n306.scalar.more = icmp ult i32 %n306.scalar.p, %n306.scalar.count
br i1 %n306.scalar.more, label %n306.scalar.body, label %n306.scalar.done
n306.scalar.body:
%n306.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n306.source, i32 %n306.scalar.p
%n306.scalar.first = load float, ptr addrspace(1) %n306.scalar.first.ptr, align 4
%n306.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n306.second, i32 %n306.scalar.p
%n306.scalar.second = load float, ptr addrspace(1) %n306.scalar.second.ptr, align 4
%n306.scalar.scalar.0 = call float @recipe.div(float %n306.scalar.first, float %n306.scalar.second)
%n306.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n306.value, i32 %n306.scalar.p
store float %n306.scalar.scalar.0, ptr addrspace(1) %n306.scalar.output.ptr, align 4
br label %n306.scalar.step
n306.scalar.step:
%n306.scalar.next = add i32 %n306.scalar.p, %threads
br label %n306.scalar.loop
n306.scalar.done:
call void @grid_barrier(i32 %threads)%n307.source = getelementptr i8, ptr addrspace(1) %values, i64 19584
%n307.second = getelementptr i8, ptr addrspace(1) %values, i64 320
%n307.value = getelementptr i8, ptr addrspace(1) %values, i64 19648
%n307.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3000
%n307.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n307.scalar.entry
n307.scalar.entry:
%n307.scalar.count = mul i32 %rows, 16
br label %n307.scalar.loop
n307.scalar.loop:
%n307.scalar.p = phi i32 [ %tid, %n307.scalar.entry ], [ %n307.scalar.next, %n307.scalar.step ]
%n307.scalar.more = icmp ult i32 %n307.scalar.p, %n307.scalar.count
br i1 %n307.scalar.more, label %n307.scalar.body, label %n307.scalar.done
n307.scalar.body:
%n307.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n307.source, i32 %n307.scalar.p
%n307.scalar.first = load float, ptr addrspace(1) %n307.scalar.first.ptr, align 4
%n307.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n307.second, i32 %n307.scalar.p
%n307.scalar.second = load float, ptr addrspace(1) %n307.scalar.second.ptr, align 4
%n307.scalar.scalar.0 = call float @recipe.mul(float %n307.scalar.first, float %n307.scalar.second)
%n307.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n307.value, i32 %n307.scalar.p
store float %n307.scalar.scalar.0, ptr addrspace(1) %n307.scalar.output.ptr, align 4
br label %n307.scalar.step
n307.scalar.step:
%n307.scalar.next = add i32 %n307.scalar.p, %threads
br label %n307.scalar.loop
n307.scalar.done:
call void @grid_barrier(i32 %threads)%n308.source = getelementptr i8, ptr addrspace(1) %values, i64 19520
%n308.second = getelementptr i8, ptr addrspace(1) %values, i64 19648
%n308.value = getelementptr i8, ptr addrspace(1) %values, i64 19712
%n308.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3008
%n308.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n308.scalar.entry
n308.scalar.entry:
%n308.scalar.count = mul i32 %rows, 16
br label %n308.scalar.loop
n308.scalar.loop:
%n308.scalar.p = phi i32 [ %tid, %n308.scalar.entry ], [ %n308.scalar.next, %n308.scalar.step ]
%n308.scalar.more = icmp ult i32 %n308.scalar.p, %n308.scalar.count
br i1 %n308.scalar.more, label %n308.scalar.body, label %n308.scalar.done
n308.scalar.body:
%n308.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n308.source, i32 %n308.scalar.p
%n308.scalar.first = load float, ptr addrspace(1) %n308.scalar.first.ptr, align 4
%n308.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n308.second, i32 %n308.scalar.p
%n308.scalar.second = load float, ptr addrspace(1) %n308.scalar.second.ptr, align 4
%n308.scalar.scalar.0 = call float @recipe.add(float %n308.scalar.first, float %n308.scalar.second)
%n308.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n308.value, i32 %n308.scalar.p
store float %n308.scalar.scalar.0, ptr addrspace(1) %n308.scalar.output.ptr, align 4
br label %n308.scalar.step
n308.scalar.step:
%n308.scalar.next = add i32 %n308.scalar.p, %threads
br label %n308.scalar.loop
n308.scalar.done:
call void @grid_barrier(i32 %threads)%n309.source = getelementptr i8, ptr addrspace(1) %values, i64 12608
%n309.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n309.value = getelementptr i8, ptr addrspace(1) %values, i64 19776
%n309.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3016
%n309.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n309.scalar.entry
n309.scalar.entry:
%n309.scalar.count = mul i32 %rows, 16
br label %n309.scalar.loop
n309.scalar.loop:
%n309.scalar.p = phi i32 [ %tid, %n309.scalar.entry ], [ %n309.scalar.next, %n309.scalar.step ]
%n309.scalar.more = icmp ult i32 %n309.scalar.p, %n309.scalar.count
br i1 %n309.scalar.more, label %n309.scalar.body, label %n309.scalar.done
n309.scalar.body:
%n309.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n309.source, i32 %n309.scalar.p
%n309.scalar.first = load float, ptr addrspace(1) %n309.scalar.first.ptr, align 4
%n309.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n309.second, i32 %n309.scalar.p
%n309.scalar.second = load float, ptr addrspace(1) %n309.scalar.second.ptr, align 4
%n309.scalar.scalar.0 = call float @recipe.div(float %n309.scalar.first, float %n309.scalar.second)
%n309.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n309.value, i32 %n309.scalar.p
store float %n309.scalar.scalar.0, ptr addrspace(1) %n309.scalar.output.ptr, align 4
br label %n309.scalar.step
n309.scalar.step:
%n309.scalar.next = add i32 %n309.scalar.p, %threads
br label %n309.scalar.loop
n309.scalar.done:
call void @grid_barrier(i32 %threads)%n310.source = getelementptr i8, ptr addrspace(1) %values, i64 19776
%n310.second = getelementptr i8, ptr addrspace(1) %values, i64 384
%n310.value = getelementptr i8, ptr addrspace(1) %values, i64 19840
%n310.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3024
%n310.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n310.scalar.entry
n310.scalar.entry:
%n310.scalar.count = mul i32 %rows, 16
br label %n310.scalar.loop
n310.scalar.loop:
%n310.scalar.p = phi i32 [ %tid, %n310.scalar.entry ], [ %n310.scalar.next, %n310.scalar.step ]
%n310.scalar.more = icmp ult i32 %n310.scalar.p, %n310.scalar.count
br i1 %n310.scalar.more, label %n310.scalar.body, label %n310.scalar.done
n310.scalar.body:
%n310.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n310.source, i32 %n310.scalar.p
%n310.scalar.first = load float, ptr addrspace(1) %n310.scalar.first.ptr, align 4
%n310.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n310.second, i32 %n310.scalar.p
%n310.scalar.second = load float, ptr addrspace(1) %n310.scalar.second.ptr, align 4
%n310.scalar.scalar.0 = call float @recipe.mul(float %n310.scalar.first, float %n310.scalar.second)
%n310.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n310.value, i32 %n310.scalar.p
store float %n310.scalar.scalar.0, ptr addrspace(1) %n310.scalar.output.ptr, align 4
br label %n310.scalar.step
n310.scalar.step:
%n310.scalar.next = add i32 %n310.scalar.p, %threads
br label %n310.scalar.loop
n310.scalar.done:
call void @grid_barrier(i32 %threads)%n311.source = getelementptr i8, ptr addrspace(1) %values, i64 19712
%n311.second = getelementptr i8, ptr addrspace(1) %values, i64 19840
%n311.value = getelementptr i8, ptr addrspace(1) %values, i64 19904
%n311.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3032
%n311.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n311.scalar.entry
n311.scalar.entry:
%n311.scalar.count = mul i32 %rows, 16
br label %n311.scalar.loop
n311.scalar.loop:
%n311.scalar.p = phi i32 [ %tid, %n311.scalar.entry ], [ %n311.scalar.next, %n311.scalar.step ]
%n311.scalar.more = icmp ult i32 %n311.scalar.p, %n311.scalar.count
br i1 %n311.scalar.more, label %n311.scalar.body, label %n311.scalar.done
n311.scalar.body:
%n311.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n311.source, i32 %n311.scalar.p
%n311.scalar.first = load float, ptr addrspace(1) %n311.scalar.first.ptr, align 4
%n311.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n311.second, i32 %n311.scalar.p
%n311.scalar.second = load float, ptr addrspace(1) %n311.scalar.second.ptr, align 4
%n311.scalar.scalar.0 = call float @recipe.add(float %n311.scalar.first, float %n311.scalar.second)
%n311.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n311.value, i32 %n311.scalar.p
store float %n311.scalar.scalar.0, ptr addrspace(1) %n311.scalar.output.ptr, align 4
br label %n311.scalar.step
n311.scalar.step:
%n311.scalar.next = add i32 %n311.scalar.p, %threads
br label %n311.scalar.loop
n311.scalar.done:
call void @grid_barrier(i32 %threads)%n312.source = getelementptr i8, ptr addrspace(1) %values, i64 15360
%n312.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n312.value = getelementptr i8, ptr addrspace(1) %values, i64 19968
%n312.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3040
%n312.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n312.scalar.entry
n312.scalar.entry:
%n312.scalar.count = mul i32 %rows, 16
br label %n312.scalar.loop
n312.scalar.loop:
%n312.scalar.p = phi i32 [ %tid, %n312.scalar.entry ], [ %n312.scalar.next, %n312.scalar.step ]
%n312.scalar.more = icmp ult i32 %n312.scalar.p, %n312.scalar.count
br i1 %n312.scalar.more, label %n312.scalar.body, label %n312.scalar.done
n312.scalar.body:
%n312.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n312.source, i32 %n312.scalar.p
%n312.scalar.first = load float, ptr addrspace(1) %n312.scalar.first.ptr, align 4
%n312.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n312.second, i32 %n312.scalar.p
%n312.scalar.second = load float, ptr addrspace(1) %n312.scalar.second.ptr, align 4
%n312.scalar.scalar.0 = call float @recipe.div(float %n312.scalar.first, float %n312.scalar.second)
%n312.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n312.value, i32 %n312.scalar.p
store float %n312.scalar.scalar.0, ptr addrspace(1) %n312.scalar.output.ptr, align 4
br label %n312.scalar.step
n312.scalar.step:
%n312.scalar.next = add i32 %n312.scalar.p, %threads
br label %n312.scalar.loop
n312.scalar.done:
call void @grid_barrier(i32 %threads)%n313.source = getelementptr i8, ptr addrspace(1) %values, i64 19968
%n313.second = getelementptr i8, ptr addrspace(1) %values, i64 448
%n313.value = getelementptr i8, ptr addrspace(1) %values, i64 20032
%n313.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3048
%n313.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n313.scalar.entry
n313.scalar.entry:
%n313.scalar.count = mul i32 %rows, 16
br label %n313.scalar.loop
n313.scalar.loop:
%n313.scalar.p = phi i32 [ %tid, %n313.scalar.entry ], [ %n313.scalar.next, %n313.scalar.step ]
%n313.scalar.more = icmp ult i32 %n313.scalar.p, %n313.scalar.count
br i1 %n313.scalar.more, label %n313.scalar.body, label %n313.scalar.done
n313.scalar.body:
%n313.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n313.source, i32 %n313.scalar.p
%n313.scalar.first = load float, ptr addrspace(1) %n313.scalar.first.ptr, align 4
%n313.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n313.second, i32 %n313.scalar.p
%n313.scalar.second = load float, ptr addrspace(1) %n313.scalar.second.ptr, align 4
%n313.scalar.scalar.0 = call float @recipe.mul(float %n313.scalar.first, float %n313.scalar.second)
%n313.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n313.value, i32 %n313.scalar.p
store float %n313.scalar.scalar.0, ptr addrspace(1) %n313.scalar.output.ptr, align 4
br label %n313.scalar.step
n313.scalar.step:
%n313.scalar.next = add i32 %n313.scalar.p, %threads
br label %n313.scalar.loop
n313.scalar.done:
call void @grid_barrier(i32 %threads)%n314.source = getelementptr i8, ptr addrspace(1) %values, i64 19904
%n314.second = getelementptr i8, ptr addrspace(1) %values, i64 20032
%n314.value = getelementptr i8, ptr addrspace(1) %values, i64 20096
%n314.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3056
%n314.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n314.scalar.entry
n314.scalar.entry:
%n314.scalar.count = mul i32 %rows, 16
br label %n314.scalar.loop
n314.scalar.loop:
%n314.scalar.p = phi i32 [ %tid, %n314.scalar.entry ], [ %n314.scalar.next, %n314.scalar.step ]
%n314.scalar.more = icmp ult i32 %n314.scalar.p, %n314.scalar.count
br i1 %n314.scalar.more, label %n314.scalar.body, label %n314.scalar.done
n314.scalar.body:
%n314.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n314.source, i32 %n314.scalar.p
%n314.scalar.first = load float, ptr addrspace(1) %n314.scalar.first.ptr, align 4
%n314.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n314.second, i32 %n314.scalar.p
%n314.scalar.second = load float, ptr addrspace(1) %n314.scalar.second.ptr, align 4
%n314.scalar.scalar.0 = call float @recipe.add(float %n314.scalar.first, float %n314.scalar.second)
%n314.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n314.value, i32 %n314.scalar.p
store float %n314.scalar.scalar.0, ptr addrspace(1) %n314.scalar.output.ptr, align 4
br label %n314.scalar.step
n314.scalar.step:
%n314.scalar.next = add i32 %n314.scalar.p, %threads
br label %n314.scalar.loop
n314.scalar.done:
call void @grid_barrier(i32 %threads)%n315.source = getelementptr i8, ptr addrspace(1) %values, i64 18368
%n315.second = getelementptr i8, ptr addrspace(1) %values, i64 18816
%n315.value = getelementptr i8, ptr addrspace(1) %values, i64 20160
%n315.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3064
%n315.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n315.scalar.entry
n315.scalar.entry:
%n315.scalar.count = mul i32 %rows, 16
br label %n315.scalar.loop
n315.scalar.loop:
%n315.scalar.p = phi i32 [ %tid, %n315.scalar.entry ], [ %n315.scalar.next, %n315.scalar.step ]
%n315.scalar.more = icmp ult i32 %n315.scalar.p, %n315.scalar.count
br i1 %n315.scalar.more, label %n315.scalar.body, label %n315.scalar.done
n315.scalar.body:
%n315.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n315.source, i32 %n315.scalar.p
%n315.scalar.first = load float, ptr addrspace(1) %n315.scalar.first.ptr, align 4
%n315.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n315.second, i32 %n315.scalar.p
%n315.scalar.second = load float, ptr addrspace(1) %n315.scalar.second.ptr, align 4
%n315.scalar.scalar.0 = call float @recipe.div(float %n315.scalar.first, float %n315.scalar.second)
%n315.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n315.value, i32 %n315.scalar.p
store float %n315.scalar.scalar.0, ptr addrspace(1) %n315.scalar.output.ptr, align 4
br label %n315.scalar.step
n315.scalar.step:
%n315.scalar.next = add i32 %n315.scalar.p, %threads
br label %n315.scalar.loop
n315.scalar.done:
call void @grid_barrier(i32 %threads)%n316.source = getelementptr i8, ptr addrspace(1) %values, i64 20160
%n316.second = getelementptr i8, ptr addrspace(1) %values, i64 512
%n316.value = getelementptr i8, ptr addrspace(1) %values, i64 20224
%n316.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3072
%n316.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n316.scalar.entry
n316.scalar.entry:
%n316.scalar.count = mul i32 %rows, 16
br label %n316.scalar.loop
n316.scalar.loop:
%n316.scalar.p = phi i32 [ %tid, %n316.scalar.entry ], [ %n316.scalar.next, %n316.scalar.step ]
%n316.scalar.more = icmp ult i32 %n316.scalar.p, %n316.scalar.count
br i1 %n316.scalar.more, label %n316.scalar.body, label %n316.scalar.done
n316.scalar.body:
%n316.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n316.source, i32 %n316.scalar.p
%n316.scalar.first = load float, ptr addrspace(1) %n316.scalar.first.ptr, align 4
%n316.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n316.second, i32 %n316.scalar.p
%n316.scalar.second = load float, ptr addrspace(1) %n316.scalar.second.ptr, align 4
%n316.scalar.scalar.0 = call float @recipe.mul(float %n316.scalar.first, float %n316.scalar.second)
%n316.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n316.value, i32 %n316.scalar.p
store float %n316.scalar.scalar.0, ptr addrspace(1) %n316.scalar.output.ptr, align 4
br label %n316.scalar.step
n316.scalar.step:
%n316.scalar.next = add i32 %n316.scalar.p, %threads
br label %n316.scalar.loop
n316.scalar.done:
call void @grid_barrier(i32 %threads)%n317.source = getelementptr i8, ptr addrspace(1) %values, i64 20096
%n317.second = getelementptr i8, ptr addrspace(1) %values, i64 20224
%n317.value = getelementptr i8, ptr addrspace(1) %values, i64 20288
%n317.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3080
%n317.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
br label %n317.scalar.entry
n317.scalar.entry:
%n317.scalar.count = mul i32 %rows, 16
br label %n317.scalar.loop
n317.scalar.loop:
%n317.scalar.p = phi i32 [ %tid, %n317.scalar.entry ], [ %n317.scalar.next, %n317.scalar.step ]
%n317.scalar.more = icmp ult i32 %n317.scalar.p, %n317.scalar.count
br i1 %n317.scalar.more, label %n317.scalar.body, label %n317.scalar.done
n317.scalar.body:
%n317.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n317.source, i32 %n317.scalar.p
%n317.scalar.first = load float, ptr addrspace(1) %n317.scalar.first.ptr, align 4
%n317.scalar.second.ptr = getelementptr inbounds float, ptr addrspace(1) %n317.second, i32 %n317.scalar.p
%n317.scalar.second = load float, ptr addrspace(1) %n317.scalar.second.ptr, align 4
%n317.scalar.scalar.0 = call float @recipe.add(float %n317.scalar.first, float %n317.scalar.second)
%n317.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n317.value, i32 %n317.scalar.p
store float %n317.scalar.scalar.0, ptr addrspace(1) %n317.scalar.output.ptr, align 4
br label %n317.scalar.step
n317.scalar.step:
%n317.scalar.next = add i32 %n317.scalar.p, %threads
br label %n317.scalar.loop
n317.scalar.done:
call void @grid_barrier(i32 %threads)%n318.source = getelementptr i8, ptr addrspace(1) %values, i64 20288
%n318.value = getelementptr i8, ptr addrspace(1) %values, i64 20352
%n318.context = getelementptr i8, ptr addrspace(1) %contexts, i64 3088
%n318.weights = getelementptr i8, ptr addrspace(1) %weights, i64 48000
%n318.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 3808
%n318.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n318.schedule.base, i32 0
%n318.schedule.0 = load i32, ptr addrspace(1) %n318.schedule.0.ptr, align 4
%n318.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n318.schedule.base, i32 1
%n318.schedule.1 = load i32, ptr addrspace(1) %n318.schedule.1.ptr, align 4
%n318.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n318.schedule.base, i32 2
%n318.schedule.2 = load i32, ptr addrspace(1) %n318.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n318.source, ptr addrspace(1) %n318.weights, ptr addrspace(1) %n318.value, ptr addrspace(1) %n318.source, i32 %rows, i32 16, i32 1, i32 1, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n318.schedule.0, i32 %n318.schedule.1, i32 %n318.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)ret void
}
define protected ptx_kernel void @recipe_model_forward(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads, i32 %training) #0 {
entry:
call void @recipe_model_inference_forward_body(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads)
ret void
}
