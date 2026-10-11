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
define internal double @recipe.math.exp.wide.pow2(i32 %k) #1 {
entry:
%biased = add i32 %k, 1023
%low = icmp slt i32 %biased, 0
%floor = select i1 %low, i32 0, i32 %biased
%high = icmp sgt i32 %floor, 2046
%clamped = select i1 %high, i32 2046, i32 %floor
%stored = zext i32 %clamped to i64
%bits = shl i64 %stored, 52
%result = bitcast i64 %bits to double
ret double %result
}
define internal double @recipe.math.exp.wide.scale(double %value, i32 %k) #1 {
entry:
%half = ashr i32 %k, 1
%rest = sub i32 %k, %half
%first = call double @recipe.math.exp.wide.pow2(i32 %half)
%second = call double @recipe.math.exp.wide.pow2(i32 %rest)
%scaled = fmul double %value, %first
%result = fmul double %scaled, %second
ret double %result
}
define internal double @recipe.math.exp.wide(double %x) #1 {
entry:
%unordered = fcmp uno double %x, %x
br i1 %unordered, label %quiet, label %high.check
quiet:
ret double %x
high.check:
%high = fcmp ogt double %x, 0x40862E42FEFA39EF
br i1 %high, label %overflow, label %low.check
overflow:
ret double 0x7FF0000000000000
low.check:
%low = fcmp olt double %x, 0xC0874910D52D3051
br i1 %low, label %underflow, label %reduce
underflow:
ret double 0.0
reduce:
%scaled = fmul double %x, 0x3FF71547652B82FE
%shifted = fadd double %scaled, 0x3FE0000000000000
%kf = call double @llvm.floor.f64(double %shifted)
%k = fptosi double %kf to i32
%upper = fmul double %kf, 0x3FE62E42FEE00000
%partial = fsub double %x, %upper
%lower = fmul double %kf, 0x3DEA39EF35793C76
%r = fsub double %partial, %lower
%exp.poly.0 = call double @llvm.fma.f64(double 0x3DE6124613A86D09, double %r, double 0x3E21EED8EFF8D898)
%exp.poly.1 = call double @llvm.fma.f64(double %exp.poly.0, double %r, double 0x3E5AE64567F544E4)
%exp.poly.2 = call double @llvm.fma.f64(double %exp.poly.1, double %r, double 0x3E927E4FB7789F5C)
%exp.poly.3 = call double @llvm.fma.f64(double %exp.poly.2, double %r, double 0x3EC71DE3A556C734)
%exp.poly.4 = call double @llvm.fma.f64(double %exp.poly.3, double %r, double 0x3EFA01A01A01A01A)
%exp.poly.5 = call double @llvm.fma.f64(double %exp.poly.4, double %r, double 0x3F2A01A01A01A01A)
%exp.poly.6 = call double @llvm.fma.f64(double %exp.poly.5, double %r, double 0x3F56C16C16C16C17)
%exp.poly.7 = call double @llvm.fma.f64(double %exp.poly.6, double %r, double 0x3F81111111111111)
%exp.poly.8 = call double @llvm.fma.f64(double %exp.poly.7, double %r, double 0x3FA5555555555555)
%exp.poly.9 = call double @llvm.fma.f64(double %exp.poly.8, double %r, double 0x3FC5555555555555)
%exp.poly.10 = call double @llvm.fma.f64(double %exp.poly.9, double %r, double 0x3FE0000000000000)
%exp.poly.11 = call double @llvm.fma.f64(double %exp.poly.10, double %r, double 0x3FF0000000000000)
%body = fmul double %r, %exp.poly.11
%expanded = fadd double 0x3FF0000000000000, %body
%result = call double @recipe.math.exp.wide.scale(double %expanded, i32 %k)
ret double %result
}



define internal double @recipe.math.expm1(double %x) #1 {
entry:
%absolute = call double @llvm.fabs.f64(double %x)
%small = fcmp olt double %absolute, 0x3FD6666666666666
br i1 %small, label %series, label %general
series:
%expm1.poly.0 = call double @llvm.fma.f64(double 0x3D2AE7F3E733B81F, double %x, double 0x3D6AE7F3E733B81F)
%expm1.poly.1 = call double @llvm.fma.f64(double %expm1.poly.0, double %x, double 0x3DA93974A8C07C9D)
%expm1.poly.2 = call double @llvm.fma.f64(double %expm1.poly.1, double %x, double 0x3DE6124613A86D09)
%expm1.poly.3 = call double @llvm.fma.f64(double %expm1.poly.2, double %x, double 0x3E21EED8EFF8D898)
%expm1.poly.4 = call double @llvm.fma.f64(double %expm1.poly.3, double %x, double 0x3E5AE64567F544E4)
%expm1.poly.5 = call double @llvm.fma.f64(double %expm1.poly.4, double %x, double 0x3E927E4FB7789F5C)
%expm1.poly.6 = call double @llvm.fma.f64(double %expm1.poly.5, double %x, double 0x3EC71DE3A556C734)
%expm1.poly.7 = call double @llvm.fma.f64(double %expm1.poly.6, double %x, double 0x3EFA01A01A01A01A)
%expm1.poly.8 = call double @llvm.fma.f64(double %expm1.poly.7, double %x, double 0x3F2A01A01A01A01A)
%expm1.poly.9 = call double @llvm.fma.f64(double %expm1.poly.8, double %x, double 0x3F56C16C16C16C17)
%expm1.poly.10 = call double @llvm.fma.f64(double %expm1.poly.9, double %x, double 0x3F81111111111111)
%expm1.poly.11 = call double @llvm.fma.f64(double %expm1.poly.10, double %x, double 0x3FA5555555555555)
%expm1.poly.12 = call double @llvm.fma.f64(double %expm1.poly.11, double %x, double 0x3FC5555555555555)
%expm1.poly.13 = call double @llvm.fma.f64(double %expm1.poly.12, double %x, double 0x3FE0000000000000)
%expm1.poly.14 = call double @llvm.fma.f64(double %expm1.poly.13, double %x, double 0x3FF0000000000000)
%result = fmul double %x, %expm1.poly.14
ret double %result
general:
%value = call double @recipe.math.exp.wide(double %x)
%shifted = fsub double %value, 1.0
ret double %shifted
}
define internal double @recipe.math.tanh.wide(double %x) #1 {
entry:
%unordered = fcmp uno double %x, %x
br i1 %unordered, label %quiet, label %range
quiet:
ret double %x
range:
%absolute = call double @llvm.fabs.f64(double %x)
%saturated = fcmp oge double %absolute, 0x4034000000000000
br i1 %saturated, label %unit, label %compute
unit:
%negative = fcmp olt double %x, 0.0
%signed = select i1 %negative, double -1.0, double 1.0
ret double %signed
compute:
%doubled = fmul double %x, 2.0
%u = call double @recipe.math.expm1(double %doubled)
%denominator = fadd double %u, 2.0
%result = fdiv double %u, %denominator
ret double %result
}





define internal float @recipe.math.tanh(float %value) #1 { entry: %wide = fpext float %value to double %computed = call double @recipe.math.tanh.wide(double %wide) %result = fptrunc double %computed to float ret float %result }



define internal float @recipe.add(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %wide = fadd float %left.wide, %right.wide %result = call float @recipe.encode(float %wide) ret float %result }
define internal float @recipe.sub(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %wide = fsub float %left.wide, %right.wide %result = call float @recipe.encode(float %wide) ret float %result }
define internal float @recipe.mul(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %wide = fmul float %left.wide, %right.wide %result = call float @recipe.encode(float %wide) ret float %result }


declare <8 x float> @llvm.fma.v8f32(<8 x float>, <8 x float>, <8 x float>)




define internal i1 @recipe.ogt(float %left, float %right) #1 { entry: %left.wide = call float @recipe.decode(float %left) %right.wide = call float @recipe.decode(float %right) %result = fcmp ogt float %left.wide, %right.wide ret i1 %result }




define internal float @recipe.from.u1(i1 %value) #1 { entry: %wide = uitofp i1 %value to float %result = call float @recipe.encode(float %wide) ret float %result }











define internal float @recipe.tanh(float %value) #1 { entry: %wide = call float @recipe.decode(float %value) %computed = call float @recipe.math.tanh(float %wide) %result = call float @recipe.encode(float %computed) ret float %result }





define internal float @recipe.state.add(float %left, float %right) #1 { entry: %result = fadd float %left, %right ret float %result }
define internal float @recipe.state.sub(float %left, float %right) #1 { entry: %result = fsub float %left, %right ret float %result }
define internal float @recipe.state.mul(float %left, float %right) #1 { entry: %result = fmul float %left, %right ret float %result }
define internal float @recipe.state.div(float %left, float %right) #1 { entry: %result = fdiv float %left, %right ret float %result }
define internal float @recipe.state.madd(float %sum, float %left, float %right) #1 { entry: %result = call float @llvm.fma.f32(float %left, float %right, float %sum) ret float %result }
define internal <8 x float> @recipe.state.madd.vector(<8 x float> %sum, <8 x float> %left, <8 x float> %right) #1 {
entry:
%result = call <8 x float> @llvm.fma.v8f32(<8 x float> %left, <8 x float> %right, <8 x float> %sum)
ret <8 x float> %result
}








define internal float @recipe.state.from.u1(i1 %value) #1 { entry: %result = uitofp i1 %value to float ret float %result }
define internal float @recipe.state.from.u32(i32 %value) #1 { entry: %result = uitofp i32 %value to float ret float %result }








define internal float @recipe.state.sqrt(float %value) #1 { entry: %result = call float @llvm.sqrt.f32(float %value) ret float %result }






define internal float @recipe.state.from.model(float %value) #1 { entry: %result = call float @recipe.decode(float %value) ret float %result }
define internal float @recipe.model.from.state(float %value) #1 { entry: %result = call float @recipe.encode(float %value) ret float %result }
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
define internal float @contraction_delta(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i32 %index
%delta.value = load float, ptr addrspace(1) %delta.ptr, align 4
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds float, ptr addrspace(1) %output, i32 %index
%output.value = load float, ptr addrspace(1) %output.ptr, align 4
%positive = call i1 @recipe.ogt(float %output.value, float 0.0)
%activated = select i1 %positive, float %delta.value, float 0.0
br label %done
done:
%value = phi float [ %delta.value, %entry ], [ %activated, %activation ]
ret float %value
}
define internal <16 x float> @contraction_delta_vector16(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %index, i1 %relu) #1 {
entry:
%delta.ptr = getelementptr inbounds float, ptr addrspace(1) %delta, i32 %index
%delta.value = load <16 x float>, ptr addrspace(1) %delta.ptr, align 4
br i1 %relu, label %activation, label %done
activation:
%output.ptr = getelementptr inbounds float, ptr addrspace(1) %output, i32 %index
%output.value = load <16 x float>, ptr addrspace(1) %output.ptr, align 4
br label %activation.loop
activation.loop:
%activation.index = phi i32 [ 0, %activation ], [ %activation.next, %activation.step ]
%activation.values = phi <16 x float> [ zeroinitializer, %activation ], [ %activation.values.next, %activation.step ]
%activation.more = icmp ult i32 %activation.index, 16
br i1 %activation.more, label %activation.step, label %done
activation.step:
%activation.output = extractelement <16 x float> %output.value, i32 %activation.index
%activation.delta = extractelement <16 x float> %delta.value, i32 %activation.index
%activation.positive = call i1 @recipe.ogt(float %activation.output, float 0.0)
%activation.value = select i1 %activation.positive, float %activation.delta, float 0.0
%activation.values.next = insertelement <16 x float> %activation.values, float %activation.value, i32 %activation.index
%activation.next = add i32 %activation.index, 1
br label %activation.loop
done:
%value = phi <16 x float> [ %delta.value, %entry ], [ %activation.values, %activation.loop ]
ret <16 x float> %value
}
define internal void @reduce_rows(ptr addrspace(1) %source, ptr addrspace(1) %target, i32 %rows, i32 %columns, i32 %stride, i32 %source.offset, i32 %target.offset, i32 %threads) #1 {
entry:
%reduce.lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
%reduce.group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
%reduce.block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
%reduce.group.base = mul i32 %reduce.group, %reduce.block
%tid = add i32 %reduce.group.base, %reduce.lid
br label %parameter.loop
parameter.loop:
%parameter = phi i32 [ %tid, %entry ], [ %parameter.next, %store ]
%parameter.more = icmp ult i32 %parameter, %columns
br i1 %parameter.more, label %seed.load, label %exit
seed.load:
%target.index = add i32 %target.offset, %parameter
%target.ptr = getelementptr inbounds float, ptr addrspace(1) %target, i32 %target.index
%source.first.index = add i32 %source.offset, %parameter
%source.first.ptr = getelementptr inbounds float, ptr addrspace(1) %source, i32 %source.first.index
%source.first = load float, ptr addrspace(1) %source.first.ptr, align 4
br label %row.loop
row.loop:
%row = phi i32 [ 1, %seed.load ], [ %row.next, %row.step ]
%sum = phi float [ %source.first, %seed.load ], [ %sum.next, %row.step ]
%row.more = icmp ult i32 %row, %rows
br i1 %row.more, label %row.step, label %store
row.step:
%row.base = mul i32 %row, %stride
%source.local = add i32 %row.base, %parameter
%source.index = add i32 %source.offset, %source.local
%source.ptr = getelementptr inbounds float, ptr addrspace(1) %source, i32 %source.index
%source.value = load float, ptr addrspace(1) %source.ptr, align 4
%sum.next = call float @recipe.add(float %sum, float %source.value)
%row.next = add i32 %row, 1
br label %row.loop
store:
store float %sum, ptr addrspace(1) %target.ptr, align 4
%parameter.next = add i32 %parameter, %threads
br label %parameter.loop
exit:
ret void
}
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
define internal void @contraction_stage_a_columns(<16 x float> %fragment, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, 16
br i1 %more, label %step, label %exit
step:
%local.m = add i32 %m, %element
%index = call i32 @contraction_vector_a_index(i32 %k, i32 %local.m, i32 %tile.m, i32 %tile.k)
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
define internal void @contraction_stage_b_fragment(<16 x float> %fragment, i32 %k, i32 %n, i32 %tile.m, i32 %tile.n, i32 %tile.k) #1 {
entry:
%base = mul i32 %tile.m, %tile.k
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, 16
br i1 %more, label %step, label %exit
step:
%local.n = add i32 %n, %element
%local = call i32 @contraction_vector_b_index(i32 %k, i32 %local.n, i32 %tile.n, i32 %tile.k)
%index = add i32 %base, %local
%value = extractelement <16 x float> %fragment, i32 %element
%target = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %index
store float %value, ptr addrspace(3) %target, align 4
%next = add i32 %element, 1
br label %loop
exit:
ret void
}
define internal void @contraction_stage_delta_a_fragment(<16 x float> %delta, <16 x float> %output, i1 %relu, i32 %k, i32 %m, i32 %tile.m, i32 %tile.k) #1 {
entry:
br label %loop
loop:
%element = phi i32 [ 0, %entry ], [ %next, %step ]
%more = icmp ult i32 %element, 16
br i1 %more, label %step, label %exit
step:
%delta.value = extractelement <16 x float> %delta, i32 %element
%output.value = extractelement <16 x float> %output, i32 %element
%positive = call i1 @recipe.ogt(float %output.value, float 0.0)
%active = select i1 %positive, float %delta.value, float 0.0
%value = select i1 %relu, float %active, float %delta.value
%local.k = add i32 %k, %element
%index = call i32 @contraction_vector_a_index(i32 %local.k, i32 %m, i32 %tile.m, i32 %tile.k)
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
define internal void @contraction_bias_accumulate(
ptr %sums, ptr addrspace(1) %destination,
i1 %enable, i1 %first, i1 %last, i32 %lid, i32 %block,
i32 %n.base, i32 %n.count, i32 %r.count, i32 %out.channels, i32 %window,
i32 %tile.m, i32 %tile.n, i32 %tile.k, i32 %store.offset ) #1 { entry:
%zero = call float @recipe.state.from.u1(i1 false) br i1 %enable, label %channel.loop, label %exit
channel.loop:
%channel = phi i32 [ %lid, %entry ], [ %channel.next, %channel.done ] %slot = phi i32 [ 0, %entry ], [ %slot.next, %channel.done ]
%channel.more = icmp ult i32 %channel, %n.count br i1 %channel.more, label %channel.begin, label %exit
channel.begin:
%sum.ptr = getelementptr [8 x float], ptr %sums, i32 0, i32 %slot
%previous = load float, ptr %sum.ptr, align 4 %initial = select i1 %first, float %zero, float %previous br label %r.loop
r.loop:
%r = phi i32 [ 0, %channel.begin ], [ %r.next, %r.step ] %sum = phi float [ %initial, %channel.begin ], [ %sum.next, %r.step ]
%r.more = icmp ult i32 %r, %r.count br i1 %r.more, label %r.step, label %sum.store
r.step:
%base = mul i32 %tile.m, %tile.k %local = call i32 @contraction_vector_b_index(i32 %r, i32 %channel, i32 %tile.n, i32 %tile.k) %index = add i32 %base, %local
%ptr = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %index
%raw = load float, ptr addrspace(3) %ptr, align 4 %value = call float @recipe.decode(float %raw) %sum.next = call float @recipe.state.add(float %sum, float %value)
%r.next = add i32 %r, 1 br label %r.loop
sum.store:
store float %sum, ptr %sum.ptr, align 4 br i1 %last, label %destination.store, label %channel.done
destination.store:
%filter = add i32 %n.base, %channel %bias.base = mul i32 %out.channels, %window %bias.local = add i32 %bias.base, %filter %bias.index = add i32 %store.offset, %bias.local
%bias.ptr = getelementptr inbounds float, ptr addrspace(1) %destination, i32 %bias.index
%bias = call float @recipe.encode(float %sum) store float %bias, ptr addrspace(1) %bias.ptr, align 4 br label %channel.done
channel.done:
%channel.next = add i32 %channel, %block %slot.next = add i32 %slot, 1 br label %channel.loop
exit: ret void
}
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









define internal void @contraction_reverse_body(
ptr addrspace(1) %input, ptr addrspace(1) %weights, ptr addrspace(1) %output, ptr addrspace(1) %delta, ptr addrspace(1) %previous, ptr addrspace(1) %gradient, i1 %write.input, i1 %has.bias, i1 %relu, i1 %matrix.gradient,
i32 %rows, i32 %in.channels, i32 %in.length, i32 %out.channels, i32 %out.length, i32 %kernel, i32 %offset,
i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k, i32 %threads ) #1 { entry:
%sums = alloca [64 x float], align 4
%bias.sums = alloca [8 x float], align 4
%state.zero = call float @recipe.state.from.u1(i1 false) %lid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() %group = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() %block = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() %groups = udiv i32 %threads, %block
%in.elements = mul i32 %in.channels, %in.length %out.elements = mul i32 %out.channels, %out.length %is.conv = icmp ne i32 %kernel, 0
%span = select i1 %is.conv, i32 %kernel, i32 1 %window = mul i32 %in.channels, %span
%gradient.r.total = mul i32 %rows, %out.length
%gradient.matrix.values = mul i32 %out.channels, %window
%gradient.bias.values = select i1 %has.bias, i32 %out.channels, i32 0
%gradient.values = add i32 %gradient.matrix.values, %gradient.bias.values
; Split-K scratch rows are written by different workgroups. Pad each row so no
; two rows share a machine word and a partial store cannot lose a neighbour. The
; base is aligned by the host, so row zero starts on the same boundary.
%gradient.stride.raw = add i32 %gradient.values, 3
%gradient.stride = and i32 %gradient.stride.raw, -4
%gradient.scratch = getelementptr inbounds float, ptr addrspace(1) %gradient, i32 84
%gradient.m.short = icmp ult i32 %gradient.tile.m, %window %gradient.m.tile = select i1 %gradient.m.short, i32 %gradient.tile.m, i32 %window %gradient.n.short = icmp ult i32 %gradient.tile.n, %out.channels %gradient.n.tile = select i1 %gradient.n.short, i32 %gradient.tile.n, i32 %out.channels
%gradient.k.short = icmp ult i32 %gradient.tile.k, %gradient.r.total %gradient.k.tile = select i1 %gradient.k.short, i32 %gradient.tile.k, i32 %gradient.r.total
%gradient.m.adjusted = add i32 %window, %gradient.m.tile %gradient.m.numerator = sub i32 %gradient.m.adjusted, 1 %gradient.m.tiles = udiv i32 %gradient.m.numerator, %gradient.m.tile %gradient.n.adjusted = add i32 %out.channels, %gradient.n.tile %gradient.n.numerator = sub i32 %gradient.n.adjusted, 1 %gradient.n.tiles = udiv i32 %gradient.n.numerator, %gradient.n.tile
%gradient.jobs = mul i32 %gradient.m.tiles, %gradient.n.tiles
; The K extent is cut into one contiguous partition per 128
; elements, capped at 32. The count and the
; boundaries are a function of the extent and two program constants. Neither the
; staged tile, the workgroup width, nor the number of compute units appears in
; the formula, so every backend sums the same partials and combines them in the
; same order, while a long K still spreads across enough workgroups to cover the
; device when the output produces few jobs.
%gradient.split.span = select i1 %matrix.gradient, i32 64, i32 128
%gradient.splits.adjusted = add i32 %gradient.r.total, %gradient.split.span
%gradient.splits.numerator = sub i32 %gradient.splits.adjusted, 1
%gradient.splits.raw = udiv i32 %gradient.splits.numerator, %gradient.split.span
%gradient.splits.large = icmp ugt i32 %gradient.splits.raw, 32
%gradient.splits = select i1 %gradient.splits.large, i32 32, i32 %gradient.splits.raw
%gradient.partition = udiv i32 %gradient.r.total, %gradient.splits
%gradient.partition.extra = urem i32 %gradient.r.total, %gradient.splits
%gradient.direct = icmp eq i32 %gradient.splits, 1
%gradient.destination.base = select i1 %gradient.direct, i32 %offset, i32 0
%gradient.destination = select i1 %gradient.direct, ptr addrspace(1) %gradient, ptr addrspace(1) %gradient.scratch
%gradient.tasks = mul i32 %gradient.jobs, %gradient.splits
br label %gradient.job.loop
gradient.job.loop:
%gradient.task = phi i32 [ %group, %entry ], [ %gradient.task.next, %gradient.job.done ]
%gradient.task.more = icmp ult i32 %gradient.task, %gradient.tasks
br i1 %gradient.task.more, label %gradient.job.step, label %gradient.finish
gradient.job.step:
%gradient.job = udiv i32 %gradient.task, %gradient.splits
%gradient.split = urem i32 %gradient.task, %gradient.splits
%gradient.store.row = mul i32 %gradient.split, %gradient.stride
%gradient.store.offset = add i32 %gradient.destination.base, %gradient.store.row
; Partition p spans [p * q + min(p, r), (p + 1) * q + min(p + 1, r)) for the
; quotient q and remainder r of the extent over the partition count. The products
; never exceed the extent, so the boundaries cannot overflow.
%gradient.split.next = add i32 %gradient.split, 1
%gradient.first.short = icmp ult i32 %gradient.split, %gradient.partition.extra
%gradient.first.extra = select i1 %gradient.first.short, i32 %gradient.split, i32 %gradient.partition.extra
%gradient.first.whole = mul i32 %gradient.split, %gradient.partition
%gradient.r.first = add i32 %gradient.first.whole, %gradient.first.extra
%gradient.limit.short = icmp ult i32 %gradient.split.next, %gradient.partition.extra
%gradient.limit.extra = select i1 %gradient.limit.short, i32 %gradient.split.next, i32 %gradient.partition.extra
%gradient.limit.whole = mul i32 %gradient.split.next, %gradient.partition
%gradient.r.limit = add i32 %gradient.limit.whole, %gradient.limit.extra
%gradient.m.group.short = icmp ult i32 %gradient.m.tiles, 4 %gradient.m.group.limit = select i1 %gradient.m.group.short, i32 %gradient.m.tiles, i32 4 %gradient.group.width = mul i32 %gradient.m.group.limit, %gradient.n.tiles %gradient.group.index = udiv i32 %gradient.job, %gradient.group.width %gradient.m.group.base = mul i32 %gradient.group.index, %gradient.m.group.limit %gradient.m.group.remaining = sub i32 %gradient.m.tiles, %gradient.m.group.base %gradient.m.group.tail = icmp ult i32 %gradient.m.group.remaining, %gradient.m.group.limit %gradient.m.group.count = select i1 %gradient.m.group.tail, i32 %gradient.m.group.remaining, i32 %gradient.m.group.limit %gradient.group.local = urem i32 %gradient.job, %gradient.group.width %gradient.m.group.local = urem i32 %gradient.group.local, %gradient.m.group.count %gradient.m.index = add i32 %gradient.m.group.base, %gradient.m.group.local %gradient.n.index = udiv i32 %gradient.group.local, %gradient.m.group.count %gradient.m.base = mul i32 %gradient.m.index, %gradient.m.tile %gradient.n.base = mul i32 %gradient.n.index, %gradient.n.tile
%gradient.m.remaining = sub i32 %window, %gradient.m.base %gradient.m.partial = icmp ult i32 %gradient.m.remaining, %gradient.m.tile %gradient.m.count = select i1 %gradient.m.partial, i32 %gradient.m.remaining, i32 %gradient.m.tile
%gradient.n.remaining = sub i32 %out.channels, %gradient.n.base %gradient.n.partial = icmp ult i32 %gradient.n.remaining, %gradient.n.tile %gradient.n.count = select i1 %gradient.n.partial, i32 %gradient.n.remaining, i32 %gradient.n.tile
%gradient.m.lanes.adjusted = add i32 %gradient.m.count, 8 %gradient.m.lanes.numerator = sub i32 %gradient.m.lanes.adjusted, 1 %gradient.m.lanes = udiv i32 %gradient.m.lanes.numerator, 8 %gradient.n.lanes.adjusted = add i32 %gradient.n.count, 8 %gradient.n.lanes.numerator = sub i32 %gradient.n.lanes.adjusted, 1 %gradient.n.lanes = udiv i32 %gradient.n.lanes.numerator, 8
; A lane owns one output position; the lanes left over at the same output
; position each own a share of the K chunks inside the accumulator, so a skinny
; output tile still drives the whole workgroup.
%gradient.output.lanes = call i32 @contraction_output_lanes(i32 %gradient.m.lanes, i32 %gradient.n.lanes, i32 %block)
%gradient.k.lanes.raw = udiv i32 %block, %gradient.output.lanes
%gradient.k.lanes.some = icmp ugt i32 %gradient.k.lanes.raw, 0
%gradient.k.lanes = select i1 %gradient.k.lanes.some, i32 %gradient.k.lanes.raw, i32 1
%gradient.active.lanes = mul i32 %gradient.output.lanes, %gradient.k.lanes
%gradient.lane.active = icmp ult i32 %lid, %gradient.active.lanes
%gradient.output.lane.raw = urem i32 %lid, %gradient.output.lanes
%gradient.output.lane = select i1 %gradient.lane.active, i32 %gradient.output.lane.raw, i32 0
%gradient.lane.k.raw = udiv i32 %lid, %gradient.output.lanes
%gradient.lane.k = select i1 %gradient.lane.active, i32 %gradient.lane.k.raw, i32 0
%gradient.lane.owner = icmp eq i32 %gradient.lane.k, 0
%gradient.lane.store = and i1 %gradient.lane.active, %gradient.lane.owner
%gradient.method.store = call i1 @contraction_vector_store_lane(i1 %gradient.lane.store, i32 %lid)
%gradient.lane.n = udiv i32 %gradient.output.lane, %gradient.m.lanes
%gradient.lane.m = urem i32 %gradient.output.lane, %gradient.m.lanes
%gradient.output.m.base = mul i32 %gradient.lane.m, 8 %gradient.output.n.base = mul i32 %gradient.lane.n, 8
%gradient.bias.first = icmp eq i32 %gradient.m.base, 0
%gradient.bias.channel = icmp ult i32 %lid, %gradient.n.count
%gradient.bias.owner = and i1 %gradient.bias.first, %gradient.bias.channel
%gradient.bias.enable = and i1 %has.bias, %gradient.bias.owner
br label %gradient.sum.init.loop gradient.sum.init.loop:
%gradient.sum.init = phi i32 [ 0, %gradient.job.step ], [ %gradient.sum.init.next, %gradient.sum.init.step ] %gradient.sum.init.more = icmp ult i32 %gradient.sum.init, 64 br i1 %gradient.sum.init.more, label %gradient.sum.init.step, label %gradient.tile.loop
gradient.sum.init.step: %gradient.sum.init.ptr = getelementptr [64 x float], ptr %sums, i32 0, i32 %gradient.sum.init store float %state.zero, ptr %gradient.sum.init.ptr, align 4 %gradient.sum.init.next = add i32 %gradient.sum.init, 1 br label %gradient.sum.init.loop gradient.tile.loop:
%gradient.r.base = phi i32 [ %gradient.r.first, %gradient.sum.init.loop ], [ %gradient.r.next, %gradient.tile.done ]
%gradient.r.remaining = sub i32 %gradient.r.limit, %gradient.r.base %gradient.r.partial = icmp ult i32 %gradient.r.remaining, %gradient.k.tile %gradient.r.count = select i1 %gradient.r.partial, i32 %gradient.r.remaining, i32 %gradient.k.tile
%gradient.r.next = add i32 %gradient.r.base, %gradient.r.count
%gradient.r.more = icmp ult i32 %gradient.r.next, %gradient.r.limit
%gradient.r.first.tile = icmp eq i32 %gradient.r.base, %gradient.r.first
%gradient.r.last.tile = xor i1 %gradient.r.more, true
br label %gradient.load.generic.entry
gradient.load.generic.entry:
%gradient.a.project = icmp eq i32 %span, 1
%gradient.a.unit = icmp eq i32 %in.length, 1
%gradient.a.contiguous = and i1 %gradient.a.project, %gradient.a.unit
%gradient.a.fragment.remainder = urem i32 %gradient.m.count, 16
%gradient.a.fragment.full = icmp eq i32 %gradient.a.fragment.remainder, 0
%gradient.a.vector = and i1 %gradient.a.contiguous, %gradient.a.fragment.full
%gradient.a.width = select i1 %gradient.a.vector, i32 16, i32 1
%gradient.a.columns = udiv i32 %gradient.m.count, %gradient.a.width
%gradient.b.unit = icmp eq i32 %out.length, 1
%gradient.b.fragment.remainder = urem i32 %gradient.n.count, 16
%gradient.b.fragment.full = icmp eq i32 %gradient.b.fragment.remainder, 0
%gradient.b.vector = and i1 %gradient.b.unit, %gradient.b.fragment.full
%gradient.b.width = select i1 %gradient.b.vector, i32 16, i32 1
%gradient.b.columns = udiv i32 %gradient.n.count, %gradient.b.width
%gradient.a.count = mul i32 %gradient.a.columns, %gradient.r.count %gradient.b.count = mul i32 %gradient.b.columns, %gradient.r.count %gradient.load.count = add i32 %gradient.a.count, %gradient.b.count br label %gradient.load.loop gradient.load.loop:
%gradient.load = phi i32 [ %lid, %gradient.load.generic.entry ], [ %gradient.load.next, %gradient.load.advance ] %gradient.load.more = icmp ult i32 %gradient.load, %gradient.load.count br i1 %gradient.load.more, label %gradient.load.classify, label %gradient.load.done
gradient.load.classify: %gradient.load.a = icmp ult i32 %gradient.load, %gradient.a.count br i1 %gradient.load.a, label %gradient.load.a.step, label %gradient.load.b.step
gradient.load.a.step: %gradient.a.r = udiv i32 %gradient.load, %gradient.a.columns %gradient.a.column = urem i32 %gradient.load, %gradient.a.columns %gradient.a.m = mul i32 %gradient.a.column, %gradient.a.width %gradient.a.global = add i32 %gradient.r.base, %gradient.a.r
%gradient.a.row = udiv i32 %gradient.a.global, %out.length %gradient.a.position = urem i32 %gradient.a.global, %out.length %gradient.a.row.base = mul i32 %gradient.a.row, %in.elements %gradient.a.term = add i32 %gradient.m.base, %gradient.a.m
%gradient.a.tile.index = call i32 @contraction_vector_a_index(i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br i1 %gradient.a.vector, label %gradient.load.a.vector, label %gradient.load.a.scalar
gradient.load.a.vector:
%gradient.a.vector.index = add i32 %gradient.a.row.base, %gradient.a.term
%gradient.a.vector.source = getelementptr inbounds float, ptr addrspace(1) %input, i32 %gradient.a.vector.index
%gradient.a.vector.value = load <16 x float>, ptr addrspace(1) %gradient.a.vector.source, align 4
call void @contraction_stage_a_columns(<16 x float> %gradient.a.vector.value, i32 %gradient.a.r, i32 %gradient.a.m, i32 %gradient.tile.m, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.a.scalar:
%gradient.a.value = call float @contraction_input( ptr addrspace(1) %input, i32 %gradient.a.row.base, i32 %gradient.a.position, i32 %gradient.a.term, i32 %span, i32 %in.length, i1 %is.conv )
br label %gradient.load.store
gradient.load.b.step: %gradient.b.local = sub i32 %gradient.load, %gradient.a.count %gradient.b.r = udiv i32 %gradient.b.local, %gradient.b.columns %gradient.b.column = urem i32 %gradient.b.local, %gradient.b.columns %gradient.b.n = mul i32 %gradient.b.column, %gradient.b.width %gradient.b.global = add i32 %gradient.r.base, %gradient.b.r
%gradient.b.row = udiv i32 %gradient.b.global, %out.length %gradient.b.position = urem i32 %gradient.b.global, %out.length %gradient.b.filter = add i32 %gradient.n.base, %gradient.b.n
%gradient.b.row.base = mul i32 %gradient.b.row, %out.elements %gradient.b.filter.base = mul i32 %gradient.b.filter, %out.length %gradient.b.local.index = add i32 %gradient.b.filter.base, %gradient.b.position %gradient.b.index = add i32 %gradient.b.row.base, %gradient.b.local.index
%gradient.b.tile.base = mul i32 %gradient.tile.m, %gradient.tile.k
%gradient.b.tile.local = call i32 @contraction_vector_b_index(i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.n, i32 %gradient.tile.k) %gradient.b.tile.index = add i32 %gradient.b.tile.base, %gradient.b.tile.local
br i1 %gradient.b.vector, label %gradient.load.b.vector, label %gradient.load.b.scalar
gradient.load.b.vector:
%gradient.b.vector.value = call <16 x float> @contraction_delta_vector16(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %gradient.b.index, i1 %relu)
call void @contraction_stage_b_fragment(<16 x float> %gradient.b.vector.value, i32 %gradient.b.r, i32 %gradient.b.n, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.load.advance
gradient.load.b.scalar:
%gradient.b.value = call float @contraction_delta(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %gradient.b.index, i1 %relu)
br label %gradient.load.store
gradient.load.store: %gradient.load.value = phi float [ %gradient.a.value, %gradient.load.a.scalar ], [ %gradient.b.value, %gradient.load.b.scalar ] %gradient.load.index = phi i32 [ %gradient.a.tile.index, %gradient.load.a.scalar ], [ %gradient.b.tile.index, %gradient.load.b.scalar ]
%gradient.load.ptr = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %gradient.load.index store float %gradient.load.value, ptr addrspace(3) %gradient.load.ptr, align 4
br label %gradient.load.advance
gradient.load.advance:
%gradient.load.next = add i32 %gradient.load, %block br label %gradient.load.loop gradient.load.done:
%gradient.load.logical.output.edge = or i1 %gradient.m.partial, %gradient.n.partial
%gradient.load.logical.edge = or i1 %gradient.load.logical.output.edge, %gradient.r.partial
%gradient.load.m.edge = icmp ult i32 %gradient.m.count, %gradient.tile.m
%gradient.load.n.edge = icmp ult i32 %gradient.n.count, %gradient.tile.n
%gradient.load.k.edge = icmp ult i32 %gradient.r.count, %gradient.tile.k
%gradient.load.schedule.output.edge = or i1 %gradient.load.m.edge, %gradient.load.n.edge
%gradient.load.schedule.edge = or i1 %gradient.load.schedule.output.edge, %gradient.load.k.edge
%gradient.load.vector.edge = or i1 %gradient.load.schedule.edge, %gradient.load.logical.edge
br i1 %gradient.load.vector.edge, label %gradient.load.zero, label %gradient.load.ready
gradient.load.zero:
call void @contraction_zero_edges(i32 %gradient.m.count, i32 %gradient.n.count, i32 %gradient.r.count, i32 %lid, i32 %block, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.load.ready
gradient.load.ready:
call void @llvm.nvvm.barrier0()
br label %gradient.scalar.ready
gradient.scalar.ready:
call void @contraction_bias_accumulate(ptr %bias.sums, ptr addrspace(1) %gradient.destination, i1 %gradient.bias.enable, i1 %gradient.r.first.tile, i1 %gradient.r.last.tile, i32 %lid, i32 %block, i32 %gradient.n.base, i32 %gradient.n.count, i32 %gradient.r.count, i32 %out.channels, i32 %window, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k, i32 %gradient.store.offset)
call void @contraction_vector_accumulate(ptr %sums, i1 %gradient.lane.active, i1 %gradient.method.store, i32 %lid, i32 %gradient.lane.k, i32 %gradient.k.lanes, i32 %gradient.output.lane, i32 %gradient.output.lanes, i32 %gradient.output.m.base, i32 %gradient.output.n.base, i32 %gradient.m.count, i32 %gradient.n.count, i32 %gradient.r.count, i32 %gradient.tile.m, i32 %gradient.tile.n, i32 %gradient.tile.k)
br label %gradient.product.done
gradient.product.done:
call void @llvm.nvvm.barrier0()
br i1 %gradient.r.more, label %gradient.tile.done, label %gradient.store.loop gradient.tile.done: br label %gradient.tile.loop
gradient.store.loop:
%gradient.store.register = phi i32 [ 0, %gradient.product.done ], [ %gradient.store.register.next, %gradient.store.next ] %gradient.store.more = icmp ult i32 %gradient.store.register, 64 br i1 %gradient.store.more, label %gradient.store.check, label %gradient.bias.store.check
gradient.store.check: %gradient.store.register.m = urem i32 %gradient.store.register, 8 %gradient.store.register.n = udiv i32 %gradient.store.register, 8 %gradient.store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %gradient.store.register, i32 %gradient.m.lanes) %gradient.store.register.valid = call i1 @contraction_output_register_valid(i32 %gradient.store.register)
%gradient.store.output.m.valid = icmp ult i32 %gradient.store.output.m.raw, %gradient.m.count %gradient.store.output.n.valid = icmp ult i32 %gradient.store.output.n.raw, %gradient.n.count %gradient.store.output.valid = and i1 %gradient.store.output.m.valid, %gradient.store.output.n.valid %gradient.store.lane.active = and i1 %gradient.method.store, %gradient.store.output.valid %gradient.store.active = and i1 %gradient.store.lane.active, %gradient.store.register.valid br i1 %gradient.store.active, label %gradient.store, label %gradient.store.next
gradient.store: %gradient.store.filter = add i32 %gradient.n.base, %gradient.store.output.n.raw %gradient.store.term = add i32 %gradient.m.base, %gradient.store.output.m.raw %gradient.store.filter.base = mul i32 %gradient.store.filter, %window %gradient.store.local = add i32 %gradient.store.filter.base, %gradient.store.term %gradient.store.index = add i32 %gradient.store.offset, %gradient.store.local
%gradient.store.ptr = getelementptr inbounds float, ptr addrspace(1) %gradient.destination, i32 %gradient.store.index %gradient.store.sum.ptr = getelementptr [64 x float], ptr %sums, i32 0, i32 %gradient.store.register %gradient.store.sum.wide = load float, ptr %gradient.store.sum.ptr, align 4 %gradient.store.sum = call float @recipe.encode(float %gradient.store.sum.wide) store float %gradient.store.sum, ptr addrspace(1) %gradient.store.ptr, align 4
br label %gradient.store.next
gradient.store.next: %gradient.store.register.next = add i32 %gradient.store.register, 1 br label %gradient.store.loop gradient.bias.store.check:
br label %gradient.job.done
gradient.job.done: %gradient.task.next = add i32 %gradient.task, %groups br label %gradient.job.loop
gradient.finish:
br i1 %gradient.direct, label %previous.check, label %gradient.reduce.entry
gradient.reduce.entry:
call void @grid_barrier(i32 %threads)
call void @reduce_rows(ptr addrspace(1) %gradient.scratch, ptr addrspace(1) %gradient, i32 %gradient.splits, i32 %gradient.values, i32 %gradient.stride, i32 0, i32 %offset, i32 %threads)
br label %previous.check
previous.check: br i1 %write.input, label %previous.entry, label %exit previous.entry:
%previous.m.total = mul i32 %rows, %in.length %previous.r.total = mul i32 %out.channels, %span
%previous.m.short = icmp ult i32 %previous.tile.m, %previous.m.total %previous.m.tile = select i1 %previous.m.short, i32 %previous.tile.m, i32 %previous.m.total %previous.n.short = icmp ult i32 %previous.tile.n, %in.channels %previous.n.tile = select i1 %previous.n.short, i32 %previous.tile.n, i32 %in.channels
%previous.k.short = icmp ult i32 %previous.tile.k, %previous.r.total %previous.k.tile = select i1 %previous.k.short, i32 %previous.tile.k, i32 %previous.r.total
%previous.m.adjusted = add i32 %previous.m.total, %previous.m.tile %previous.m.numerator = sub i32 %previous.m.adjusted, 1 %previous.m.tiles = udiv i32 %previous.m.numerator, %previous.m.tile %previous.n.adjusted = add i32 %in.channels, %previous.n.tile %previous.n.numerator = sub i32 %previous.n.adjusted, 1 %previous.n.tiles = udiv i32 %previous.n.numerator, %previous.n.tile
%previous.jobs = mul i32 %previous.m.tiles, %previous.n.tiles br label %previous.job.loop previous.job.loop:
%previous.job = phi i32 [ %group, %previous.entry ], [ %previous.job.next, %previous.job.done ] %previous.job.more = icmp ult i32 %previous.job, %previous.jobs br i1 %previous.job.more, label %previous.job.step, label %exit
previous.job.step: %previous.m.group.short = icmp ult i32 %previous.m.tiles, 4 %previous.m.group.limit = select i1 %previous.m.group.short, i32 %previous.m.tiles, i32 4 %previous.group.width = mul i32 %previous.m.group.limit, %previous.n.tiles %previous.group.index = udiv i32 %previous.job, %previous.group.width %previous.m.group.base = mul i32 %previous.group.index, %previous.m.group.limit %previous.m.group.remaining = sub i32 %previous.m.tiles, %previous.m.group.base %previous.m.group.tail = icmp ult i32 %previous.m.group.remaining, %previous.m.group.limit %previous.m.group.count = select i1 %previous.m.group.tail, i32 %previous.m.group.remaining, i32 %previous.m.group.limit %previous.group.local = urem i32 %previous.job, %previous.group.width %previous.m.group.local = urem i32 %previous.group.local, %previous.m.group.count %previous.m.index = add i32 %previous.m.group.base, %previous.m.group.local %previous.n.index = udiv i32 %previous.group.local, %previous.m.group.count %previous.m.base = mul i32 %previous.m.index, %previous.m.tile %previous.n.base = mul i32 %previous.n.index, %previous.n.tile
%previous.m.remaining = sub i32 %previous.m.total, %previous.m.base %previous.m.partial = icmp ult i32 %previous.m.remaining, %previous.m.tile %previous.m.count = select i1 %previous.m.partial, i32 %previous.m.remaining, i32 %previous.m.tile
%previous.n.remaining = sub i32 %in.channels, %previous.n.base %previous.n.partial = icmp ult i32 %previous.n.remaining, %previous.n.tile %previous.n.count = select i1 %previous.n.partial, i32 %previous.n.remaining, i32 %previous.n.tile
%previous.m.lanes.adjusted = add i32 %previous.m.count, 8 %previous.m.lanes.numerator = sub i32 %previous.m.lanes.adjusted, 1 %previous.m.lanes = udiv i32 %previous.m.lanes.numerator, 8 %previous.n.lanes.adjusted = add i32 %previous.n.count, 8 %previous.n.lanes.numerator = sub i32 %previous.n.lanes.adjusted, 1 %previous.n.lanes = udiv i32 %previous.n.lanes.numerator, 8
%previous.lanes = call i32 @contraction_output_lanes(i32 %previous.m.lanes, i32 %previous.n.lanes, i32 %block)
%previous.k.lanes.raw = udiv i32 %block, %previous.lanes
%previous.k.lanes.some = icmp ugt i32 %previous.k.lanes.raw, 0
%previous.k.lanes = select i1 %previous.k.lanes.some, i32 %previous.k.lanes.raw, i32 1
%previous.active.lanes = mul i32 %previous.lanes, %previous.k.lanes
%previous.lane.active = icmp ult i32 %lid, %previous.active.lanes
%previous.output.lane.raw = urem i32 %lid, %previous.lanes
%previous.output.lane = select i1 %previous.lane.active, i32 %previous.output.lane.raw, i32 0
%previous.lane.k.raw = udiv i32 %lid, %previous.lanes
%previous.lane.k = select i1 %previous.lane.active, i32 %previous.lane.k.raw, i32 0
%previous.lane.owner = icmp eq i32 %previous.lane.k, 0
%previous.lane.store = and i1 %previous.lane.active, %previous.lane.owner
%previous.method.store = call i1 @contraction_vector_store_lane(i1 %previous.lane.store, i32 %lid)
%previous.lane.n = udiv i32 %previous.output.lane, %previous.m.lanes %previous.lane.m = urem i32 %previous.output.lane, %previous.m.lanes
%previous.output.m.base = mul i32 %previous.lane.m, 8 %previous.output.n.base = mul i32 %previous.lane.n, 8 br label %previous.sum.init.loop previous.sum.init.loop:
%previous.sum.init = phi i32 [ 0, %previous.job.step ], [ %previous.sum.init.next, %previous.sum.init.step ] %previous.sum.init.more = icmp ult i32 %previous.sum.init, 64 br i1 %previous.sum.init.more, label %previous.sum.init.step, label %previous.tile.loop
previous.sum.init.step: %previous.sum.init.ptr = getelementptr [64 x float], ptr %sums, i32 0, i32 %previous.sum.init store float %state.zero, ptr %previous.sum.init.ptr, align 4 %previous.sum.init.next = add i32 %previous.sum.init, 1 br label %previous.sum.init.loop previous.tile.loop:
%previous.r.base = phi i32 [ 0, %previous.sum.init.loop ], [ %previous.r.next, %previous.tile.done ]
%previous.r.remaining = sub i32 %previous.r.total, %previous.r.base %previous.r.partial = icmp ult i32 %previous.r.remaining, %previous.k.tile %previous.r.count = select i1 %previous.r.partial, i32 %previous.r.remaining, i32 %previous.k.tile
%previous.a.project = icmp eq i32 %span, 1 %previous.a.unit = icmp eq i32 %out.length, 1 %previous.a.contiguous = and i1 %previous.a.project, %previous.a.unit
%previous.a.fragment.remainder = urem i32 %previous.r.count, 16 %previous.a.fragment.full = icmp eq i32 %previous.a.fragment.remainder, 0 %previous.a.vector = and i1 %previous.a.contiguous, %previous.a.fragment.full
%previous.a.width = select i1 %previous.a.vector, i32 16, i32 1 %previous.a.columns = udiv i32 %previous.r.count, %previous.a.width
%previous.b.fragment.remainder = urem i32 %previous.n.count, 16 %previous.b.fragment.full = icmp eq i32 %previous.b.fragment.remainder, 0 %previous.b.vector = and i1 %previous.a.project, %previous.b.fragment.full
%previous.b.width = select i1 %previous.b.vector, i32 16, i32 1 %previous.b.columns = udiv i32 %previous.n.count, %previous.b.width
%previous.a.count = mul i32 %previous.m.count, %previous.a.columns %previous.b.count = mul i32 %previous.r.count, %previous.b.columns %previous.load.count = add i32 %previous.a.count, %previous.b.count br label %previous.load.loop previous.load.loop:
%previous.load = phi i32 [ %lid, %previous.tile.loop ], [ %previous.load.next, %previous.load.advance ] %previous.load.more = icmp ult i32 %previous.load, %previous.load.count br i1 %previous.load.more, label %previous.load.classify, label %previous.load.done
previous.load.classify: %previous.load.a = icmp ult i32 %previous.load, %previous.a.count br i1 %previous.load.a, label %previous.load.a.step, label %previous.load.b.step
previous.load.a.step: %previous.a.m = udiv i32 %previous.load, %previous.a.columns %previous.a.column = urem i32 %previous.load, %previous.a.columns %previous.a.r = mul i32 %previous.a.column, %previous.a.width %previous.a.term = add i32 %previous.r.base, %previous.a.r
%previous.a.filter = udiv i32 %previous.a.term, %span %previous.a.kernel = urem i32 %previous.a.term, %span %previous.a.global = add i32 %previous.m.base, %previous.a.m %previous.a.row = udiv i32 %previous.a.global, %in.length %previous.a.position = urem i32 %previous.a.global, %in.length
%previous.a.low = icmp uge i32 %previous.a.position, %previous.a.kernel %previous.a.position.raw = sub i32 %previous.a.position, %previous.a.kernel %previous.a.high = icmp ult i32 %previous.a.position.raw, %out.length %previous.a.valid = and i1 %previous.a.low, %previous.a.high
%previous.a.position.safe = select i1 %previous.a.valid, i32 %previous.a.position.raw, i32 0 %previous.a.row.base = mul i32 %previous.a.row, %out.elements %previous.a.filter.base = mul i32 %previous.a.filter, %out.length
%previous.a.local = add i32 %previous.a.filter.base, %previous.a.position.safe %previous.a.index = add i32 %previous.a.row.base, %previous.a.local %previous.a.tile.index = call i32 @contraction_vector_a_index(i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br i1 %previous.a.vector, label %previous.load.a.vector, label %previous.load.a.scalar
previous.load.a.vector:
%previous.a.vector.delta = getelementptr inbounds float, ptr addrspace(1) %delta, i32 %previous.a.index
%previous.a.vector.output = getelementptr inbounds float, ptr addrspace(1) %output, i32 %previous.a.index
%previous.a.vector.delta.value = load <16 x float>, ptr addrspace(1) %previous.a.vector.delta, align 4
%previous.a.vector.output.value = load <16 x float>, ptr addrspace(1) %previous.a.vector.output, align 4
call void @contraction_stage_delta_a_fragment(<16 x float> %previous.a.vector.delta.value, <16 x float> %previous.a.vector.output.value, i1 %relu, i32 %previous.a.r, i32 %previous.a.m, i32 %previous.tile.m, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.a.scalar:
%previous.a.raw = call float @contraction_delta(ptr addrspace(1) %delta, ptr addrspace(1) %output, i32 %previous.a.index, i1 %relu)
%previous.a.value = select i1 %previous.a.valid, float %previous.a.raw, float 0.0
br label %previous.load.store
previous.load.b.step: %previous.b.local = sub i32 %previous.load, %previous.a.count %previous.b.r = udiv i32 %previous.b.local, %previous.b.columns %previous.b.column = urem i32 %previous.b.local, %previous.b.columns %previous.b.n = mul i32 %previous.b.column, %previous.b.width %previous.b.term = add i32 %previous.r.base, %previous.b.r
%previous.b.filter = udiv i32 %previous.b.term, %span %previous.b.kernel = urem i32 %previous.b.term, %span %previous.b.channel = add i32 %previous.n.base, %previous.b.n %previous.b.filter.base = mul i32 %previous.b.filter, %window
%previous.b.channel.base = mul i32 %previous.b.channel, %span %previous.b.channel.local = add i32 %previous.b.channel.base, %previous.b.kernel %previous.b.index = add i32 %previous.b.filter.base, %previous.b.channel.local
%previous.b.tile.base = mul i32 %previous.tile.m, %previous.tile.k
%previous.b.tile.local = call i32 @contraction_vector_b_index(i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.n, i32 %previous.tile.k) %previous.b.tile.index = add i32 %previous.b.tile.base, %previous.b.tile.local
br i1 %previous.b.vector, label %previous.load.b.vector, label %previous.load.b.scalar
previous.load.b.vector:
%previous.b.vector.source = getelementptr inbounds float, ptr addrspace(1) %weights, i32 %previous.b.index
%previous.b.vector.value = load <16 x float>, ptr addrspace(1) %previous.b.vector.source, align 4
call void @contraction_stage_b_fragment(<16 x float> %previous.b.vector.value, i32 %previous.b.r, i32 %previous.b.n, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k)
br label %previous.load.advance
previous.load.b.scalar:
%previous.b.ptr = getelementptr inbounds float, ptr addrspace(1) %weights, i32 %previous.b.index
%previous.b.value = load float, ptr addrspace(1) %previous.b.ptr, align 4
br label %previous.load.store
previous.load.store: %previous.load.value = phi float [ %previous.a.value, %previous.load.a.scalar ], [ %previous.b.value, %previous.load.b.scalar ] %previous.load.index = phi i32 [ %previous.a.tile.index, %previous.load.a.scalar ], [ %previous.b.tile.index, %previous.load.b.scalar ]
%previous.load.ptr = getelementptr [0 x float], ptr addrspace(3) @contraction_tile_f32, i32 0, i32 %previous.load.index store float %previous.load.value, ptr addrspace(3) %previous.load.ptr, align 4
br label %previous.load.advance
previous.load.advance:
%previous.load.next = add i32 %previous.load, %block br label %previous.load.loop previous.load.done:
%previous.load.logical.output.edge = or i1 %previous.m.partial, %previous.n.partial
%previous.load.logical.edge = or i1 %previous.load.logical.output.edge, %previous.r.partial
%previous.load.m.edge = icmp ult i32 %previous.m.count, %previous.tile.m
%previous.load.n.edge = icmp ult i32 %previous.n.count, %previous.tile.n
%previous.load.k.edge = icmp ult i32 %previous.r.count, %previous.tile.k
%previous.load.schedule.output.edge = or i1 %previous.load.m.edge, %previous.load.n.edge
%previous.load.schedule.edge = or i1 %previous.load.schedule.output.edge, %previous.load.k.edge
%previous.load.vector.edge = or i1 %previous.load.schedule.edge, %previous.load.logical.edge
br i1 %previous.load.vector.edge, label %previous.load.zero, label %previous.load.ready
previous.load.zero:
call void @contraction_zero_edges(i32 %previous.m.count, i32 %previous.n.count, i32 %previous.r.count, i32 %lid, i32 %block, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k)
br label %previous.load.ready
previous.load.ready:
call void @llvm.nvvm.barrier0()
call void @contraction_vector_accumulate(ptr %sums, i1 %previous.lane.active, i1 %previous.method.store, i32 %lid, i32 %previous.lane.k, i32 %previous.k.lanes, i32 %previous.output.lane, i32 %previous.lanes, i32 %previous.output.m.base, i32 %previous.output.n.base, i32 %previous.m.count, i32 %previous.n.count, i32 %previous.r.count, i32 %previous.tile.m, i32 %previous.tile.n, i32 %previous.tile.k) call void @llvm.nvvm.barrier0()
%previous.r.next = add i32 %previous.r.base, %previous.r.count %previous.r.more = icmp ult i32 %previous.r.next, %previous.r.total br i1 %previous.r.more, label %previous.tile.done, label %previous.store.loop previous.tile.done: br label %previous.tile.loop previous.store.loop:
%previous.store.register = phi i32 [ 0, %previous.load.ready ], [ %previous.store.register.next, %previous.store.next ] %previous.store.more = icmp ult i32 %previous.store.register, 64 br i1 %previous.store.more, label %previous.store.check, label %previous.job.done
previous.store.check: %previous.store.output.m.raw = call i32 @contraction_vector_output_m(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.output.n.raw = call i32 @contraction_vector_output_n(i32 %lid, i32 %previous.store.register, i32 %previous.m.lanes) %previous.store.register.valid = call i1 @contraction_output_register_valid(i32 %previous.store.register)
%previous.store.output.m.valid = icmp ult i32 %previous.store.output.m.raw, %previous.m.count %previous.store.output.n.valid = icmp ult i32 %previous.store.output.n.raw, %previous.n.count %previous.store.output.valid = and i1 %previous.store.output.m.valid, %previous.store.output.n.valid %previous.lane.output.active = and i1 %previous.method.store, %previous.store.output.valid %previous.store.active = and i1 %previous.lane.output.active, %previous.store.register.valid br i1 %previous.store.active, label %previous.store, label %previous.store.next
previous.store: %previous.store.m.global = add i32 %previous.m.base, %previous.store.output.m.raw %previous.store.channel = add i32 %previous.n.base, %previous.store.output.n.raw %previous.store.row = udiv i32 %previous.store.m.global, %in.length %previous.store.position = urem i32 %previous.store.m.global, %in.length
%previous.store.row.base = mul i32 %previous.store.row, %in.elements %previous.store.channel.base = mul i32 %previous.store.channel, %in.length %previous.store.local = add i32 %previous.store.channel.base, %previous.store.position %previous.store.index = add i32 %previous.store.row.base, %previous.store.local %previous.store.ptr = getelementptr inbounds float, ptr addrspace(1) %previous, i32 %previous.store.index
%previous.store.old = load float, ptr addrspace(1) %previous.store.ptr, align 4 %previous.store.sum.ptr = getelementptr [64 x float], ptr %sums, i32 0, i32 %previous.store.register %previous.store.sum.wide = load float, ptr %previous.store.sum.ptr, align 4 %previous.store.sum = call float @recipe.encode(float %previous.store.sum.wide) %previous.store.value = call float @recipe.add(float %previous.store.old, float %previous.store.sum) store float %previous.store.value, ptr addrspace(1) %previous.store.ptr, align 4 br label %previous.store.next
previous.store.next: %previous.store.register.next = add i32 %previous.store.register, 1 br label %previous.store.loop previous.job.done: %previous.job.next = add i32 %previous.job, %groups br label %previous.job.loop exit: ret void }
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
%n0.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 88
%n0.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 0
%n0.schedule.0 = load i32, ptr addrspace(1) %n0.schedule.0.ptr, align 4
%n0.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 1
%n0.schedule.1 = load i32, ptr addrspace(1) %n0.schedule.1.ptr, align 4
%n0.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 2
%n0.schedule.2 = load i32, ptr addrspace(1) %n0.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %samples, ptr addrspace(1) %n0.weights, ptr addrspace(1) %n0.value, ptr addrspace(1) %samples, i32 %rows, i32 8, i32 1, i32 8, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n0.schedule.0, i32 %n0.schedule.1, i32 %n0.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n1.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n1.value = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n1.context = getelementptr i8, ptr addrspace(1) %contexts, i64 40
%n1.weights = getelementptr i8, ptr addrspace(1) %weights, i64 288
br label %n1.scalar.entry
n1.scalar.entry:
%n1.scalar.count = mul i32 %rows, 8
br label %n1.scalar.loop
n1.scalar.loop:
%n1.scalar.p = phi i32 [ %tid, %n1.scalar.entry ], [ %n1.scalar.next, %n1.scalar.step ]
%n1.scalar.more = icmp ult i32 %n1.scalar.p, %n1.scalar.count
br i1 %n1.scalar.more, label %n1.scalar.body, label %n1.scalar.done
n1.scalar.body:
%n1.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n1.source, i32 %n1.scalar.p
%n1.scalar.first = load float, ptr addrspace(1) %n1.scalar.first.ptr, align 4
%n1.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n1.scalar.first, float 0x0000000000000000)
%n1.scalar.scalar.2 = call float @recipe.from.u1(i1 %n1.scalar.scalar.2.condition)
%n1.scalar.scalar.3 = call float @recipe.tanh(float %n1.scalar.first)
%n1.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n1.value, i32 %n1.scalar.p
store float %n1.scalar.scalar.3, ptr addrspace(1) %n1.scalar.output.ptr, align 4
br label %n1.scalar.step
n1.scalar.step:
%n1.scalar.next = add i32 %n1.scalar.p, %threads
br label %n1.scalar.loop
n1.scalar.done:
call void @grid_barrier(i32 %threads)%n2.source = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n2.value = getelementptr i8, ptr addrspace(1) %values, i64 4096
%n2.context = getelementptr i8, ptr addrspace(1) %contexts, i64 48
%n2.weights = getelementptr i8, ptr addrspace(1) %weights, i64 288
%n2.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 128
%n2.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 0
%n2.schedule.0 = load i32, ptr addrspace(1) %n2.schedule.0.ptr, align 4
%n2.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 1
%n2.schedule.1 = load i32, ptr addrspace(1) %n2.schedule.1.ptr, align 4
%n2.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 2
%n2.schedule.2 = load i32, ptr addrspace(1) %n2.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n2.source, ptr addrspace(1) %n2.weights, ptr addrspace(1) %n2.value, ptr addrspace(1) %n2.source, i32 %rows, i32 8, i32 1, i32 1, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n2.schedule.0, i32 %n2.schedule.1, i32 %n2.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)ret void
}
define internal void @recipe_model_training_forward_body(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads) #3 {
entry:
%tid = call i32 @global_id()
%n0.value = getelementptr i8, ptr addrspace(1) %values, i64 0
%n0.context = getelementptr i8, ptr addrspace(1) %contexts, i64 0
%n0.weights = getelementptr i8, ptr addrspace(1) %weights, i64 0
%n0.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 88
%n0.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 0
%n0.schedule.0 = load i32, ptr addrspace(1) %n0.schedule.0.ptr, align 4
%n0.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 1
%n0.schedule.1 = load i32, ptr addrspace(1) %n0.schedule.1.ptr, align 4
%n0.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n0.schedule.base, i32 2
%n0.schedule.2 = load i32, ptr addrspace(1) %n0.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %samples, ptr addrspace(1) %n0.weights, ptr addrspace(1) %n0.value, ptr addrspace(1) %samples, i32 %rows, i32 8, i32 1, i32 8, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n0.schedule.0, i32 %n0.schedule.1, i32 %n0.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)%n1.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n1.value = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n1.context = getelementptr i8, ptr addrspace(1) %contexts, i64 40
%n1.weights = getelementptr i8, ptr addrspace(1) %weights, i64 288
br label %n1.scalar.entry
n1.scalar.entry:
%n1.scalar.count = mul i32 %rows, 8
br label %n1.scalar.loop
n1.scalar.loop:
%n1.scalar.p = phi i32 [ %tid, %n1.scalar.entry ], [ %n1.scalar.next, %n1.scalar.step ]
%n1.scalar.more = icmp ult i32 %n1.scalar.p, %n1.scalar.count
br i1 %n1.scalar.more, label %n1.scalar.body, label %n1.scalar.done
n1.scalar.body:
%n1.scalar.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n1.source, i32 %n1.scalar.p
%n1.scalar.first = load float, ptr addrspace(1) %n1.scalar.first.ptr, align 4
%n1.scalar.scalar.2.condition = call i1 @recipe.ogt(float %n1.scalar.first, float 0x0000000000000000)
%n1.scalar.scalar.2 = call float @recipe.from.u1(i1 %n1.scalar.scalar.2.condition)
%n1.scalar.scalar.3 = call float @recipe.tanh(float %n1.scalar.first)
%n1.scalar.output.ptr = getelementptr inbounds float, ptr addrspace(1) %n1.value, i32 %n1.scalar.p
store float %n1.scalar.scalar.3, ptr addrspace(1) %n1.scalar.output.ptr, align 4
br label %n1.scalar.step
n1.scalar.step:
%n1.scalar.next = add i32 %n1.scalar.p, %threads
br label %n1.scalar.loop
n1.scalar.done:
call void @grid_barrier(i32 %threads)%n2.source = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n2.value = getelementptr i8, ptr addrspace(1) %values, i64 4096
%n2.context = getelementptr i8, ptr addrspace(1) %contexts, i64 48
%n2.weights = getelementptr i8, ptr addrspace(1) %weights, i64 288
%n2.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 128
%n2.schedule.0.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 0
%n2.schedule.0 = load i32, ptr addrspace(1) %n2.schedule.0.ptr, align 4
%n2.schedule.1.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 1
%n2.schedule.1 = load i32, ptr addrspace(1) %n2.schedule.1.ptr, align 4
%n2.schedule.2.ptr = getelementptr i32, ptr addrspace(1) %n2.schedule.base, i32 2
%n2.schedule.2 = load i32, ptr addrspace(1) %n2.schedule.2.ptr, align 4
call void @contraction_forward_body( ptr addrspace(1) %n2.source, ptr addrspace(1) %n2.weights, ptr addrspace(1) %n2.value, ptr addrspace(1) %n2.source, i32 %rows, i32 8, i32 1, i32 1, i32 1, i32 0, i1 true, i1 false, i1 false, i1 false, i1 false, i32 %n2.schedule.0, i32 %n2.schedule.1, i32 %n2.schedule.2, i32 %threads )
call void @grid_barrier(i32 %threads)ret void
}
define protected ptx_kernel void @recipe_model_forward(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads, i32 %training) #0 {
entry:
%forward.training = icmp ne i32 %training, 0
br i1 %forward.training, label %forward.training.entry, label %forward.inference.entry
forward.inference.entry:
call void @recipe_model_inference_forward_body(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads)
br label %forward.done
forward.training.entry:
call void @recipe_model_training_forward_body(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads)
br label %forward.done
forward.done:
ret void
}
define protected ptx_kernel void @recipe_model_epoch(ptr addrspace(1) %samples, ptr addrspace(1) %targets, ptr addrspace(1) %weights, ptr addrspace(1) %frozen, ptr addrspace(1) %moments, ptr addrspace(1) %variances, ptr addrspace(1) %gradient, ptr addrspace(1) %metrics, ptr addrspace(1) %input_adjoint, ptr addrspace(1) %values, ptr addrspace(1) %contexts, ptr addrspace(1) %adjoints, i32 %rows, i32 %threads, float %rate, float %beta1, float %beta2, float %beta1.power, float %beta2.power, float %epsilon, float %decay, i32 %run.gradient, i32 %run.optimizer) #0 {
entry:
%tid = call i32 @global_id()
%epoch.gradient = icmp ne i32 %run.gradient, 0
%epoch.optimizer = icmp ne i32 %run.optimizer, 0
br i1 %epoch.gradient, label %gradient.entry, label %optimizer.entry
gradient.entry:
%clear.gradient.start = zext i32 %tid to i64
%clear.gradient.stride = zext i32 %threads to i64
br label %clear.gradient.loop
clear.gradient.loop:
%clear.gradient.p = phi i64 [ %clear.gradient.start, %gradient.entry ], [ %clear.gradient.next, %clear.gradient.step ]
%clear.gradient.more = icmp ult i64 %clear.gradient.p, 324
br i1 %clear.gradient.more, label %clear.gradient.step, label %clear.gradient.done
clear.gradient.step:
%clear.gradient.ptr = getelementptr i8, ptr addrspace(1) %gradient, i64 %clear.gradient.p
store i8 0, ptr addrspace(1) %clear.gradient.ptr, align 1
%clear.gradient.next = add i64 %clear.gradient.p, %clear.gradient.stride
br label %clear.gradient.loop
clear.gradient.done:
%clear.adjoints.start = zext i32 %tid to i64
%clear.adjoints.stride = zext i32 %threads to i64
br label %clear.adjoints.loop
clear.adjoints.loop:
%clear.adjoints.p = phi i64 [ %clear.adjoints.start, %clear.gradient.done ], [ %clear.adjoints.next, %clear.adjoints.step ]
%clear.adjoints.more = icmp ult i64 %clear.adjoints.p, 4352
br i1 %clear.adjoints.more, label %clear.adjoints.step, label %clear.adjoints.done
clear.adjoints.step:
%clear.adjoints.ptr = getelementptr i8, ptr addrspace(1) %adjoints, i64 %clear.adjoints.p
store i8 0, ptr addrspace(1) %clear.adjoints.ptr, align 1
%clear.adjoints.next = add i64 %clear.adjoints.p, %clear.adjoints.stride
br label %clear.adjoints.loop
clear.adjoints.done:
%clear.input.start = zext i32 %tid to i64
%clear.input.stride = zext i32 %threads to i64
br label %clear.input.loop
clear.input.loop:
%clear.input.p = phi i64 [ %clear.input.start, %clear.adjoints.done ], [ %clear.input.next, %clear.input.step ]
%clear.input.more = icmp ult i64 %clear.input.p, 2048
br i1 %clear.input.more, label %clear.input.step, label %clear.input.done
clear.input.step:
%clear.input.ptr = getelementptr i8, ptr addrspace(1) %input_adjoint, i64 %clear.input.p
store i8 0, ptr addrspace(1) %clear.input.ptr, align 1
%clear.input.next = add i64 %clear.input.p, %clear.input.stride
br label %clear.input.loop
clear.input.done:
call void @grid_barrier(i32 %threads)
call void @recipe_model_training_forward_body(ptr addrspace(1) %samples, ptr addrspace(1) %weights, ptr addrspace(1) %values, ptr addrspace(1) %contexts, i32 %rows, i32 %threads)

%prediction.base = getelementptr i8, ptr addrspace(1) %values, i64 4096
%prediction = bitcast ptr addrspace(1) %prediction.base to ptr addrspace(1)
%metric.ptr = getelementptr float, ptr addrspace(1) %metrics, i32 0
%loss.count = mul i32 %rows, 1
%loss.leader = icmp eq i32 %tid, 0
br i1 %loss.leader, label %loss.entry, label %loss.wait
loss.entry:
%loss.items = call float @recipe.state.from.u32(i32 %loss.count)
%loss.normalizer = call float @recipe.state.sqrt(float %loss.items)
br label %loss.step
loss.step:
%loss.p = phi i32 [ 0, %loss.entry ], [ %loss.next, %loss.item ]
%loss.mean = phi float [ 0x0000000000000000, %loss.entry ], [ %loss.mean.next, %loss.item ]
%loss.more = icmp ult i32 %loss.p, %loss.count
br i1 %loss.more, label %loss.item, label %loss.store
loss.item:
%loss.prediction.ptr = getelementptr float, ptr addrspace(1) %prediction, i32 %loss.p
%loss.prediction.model = load float, ptr addrspace(1) %loss.prediction.ptr, align 4
%loss.prediction = call float @recipe.state.from.model(float %loss.prediction.model)
%loss.target.ptr = getelementptr float, ptr addrspace(1) %targets, i32 %loss.p
%loss.target.model = load float, ptr addrspace(1) %loss.target.ptr, align 4
%loss.target = call float @recipe.state.from.model(float %loss.target.model)
%loss.difference = call float @recipe.state.sub(float %loss.prediction, float %loss.target)
%loss.scaled = call float @recipe.state.div(float %loss.difference, float %loss.normalizer)
%loss.square = call float @recipe.state.mul(float %loss.scaled, float %loss.scaled)
%loss.mean.next = call float @recipe.state.add(float %loss.mean, float %loss.square)
%loss.next = add i32 %loss.p, 1
br label %loss.step
loss.store:
%loss.value = call float @recipe.state.add(float %loss.mean, float 0x0000000000000000)
store float %loss.value, ptr addrspace(1) %metric.ptr, align 4
br label %loss.wait
loss.wait:
%adjoint.base = getelementptr i8, ptr addrspace(1) %adjoints, i64 4096
%adjoint = bitcast ptr addrspace(1) %adjoint.base to ptr addrspace(1)
br label %seed.loop
seed.loop:
%seed.p = phi i32 [ %tid, %loss.wait ], [ %seed.next, %seed.step ]
%seed.more = icmp ult i32 %seed.p, %loss.count
br i1 %seed.more, label %seed.step, label %seed.done
seed.step:
%seed.pred.ptr = getelementptr float, ptr addrspace(1) %prediction, i32 %seed.p
%seed.pred.model = load float, ptr addrspace(1) %seed.pred.ptr, align 4
%seed.pred = call float @recipe.state.from.model(float %seed.pred.model)
%seed.target.ptr = getelementptr float, ptr addrspace(1) %targets, i32 %seed.p
%seed.target.model = load float, ptr addrspace(1) %seed.target.ptr, align 4
%seed.target = call float @recipe.state.from.model(float %seed.target.model)
%seed.difference = call float @recipe.state.sub(float %seed.pred, float %seed.target)
%seed.rows = call float @recipe.state.from.u32(i32 %loss.count)
%seed.twice = call float @recipe.state.add(float %seed.difference, float %seed.difference)
%seed.mse = call float @recipe.state.div(float %seed.twice, float %seed.rows)
%seed.model = call float @recipe.model.from.state(float %seed.mse)
%seed.ptr = getelementptr float, ptr addrspace(1) %adjoint, i32 %seed.p
store float %seed.model, ptr addrspace(1) %seed.ptr, align 4
%seed.next = add i32 %seed.p, %threads
br label %seed.loop
seed.done:
call void @grid_barrier(i32 %threads)%n2.source = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n2.value = getelementptr i8, ptr addrspace(1) %values, i64 4096
%n2.context = getelementptr i8, ptr addrspace(1) %contexts, i64 48
%n2.delta = getelementptr i8, ptr addrspace(1) %adjoints, i64 4096
%n2.weights = getelementptr i8, ptr addrspace(1) %weights, i64 288
%n2.source.adjoint = getelementptr i8, ptr addrspace(1) %adjoints, i64 2048
%n2.reverse.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 128
%n2.reverse.schedule.3.ptr = getelementptr i32, ptr addrspace(1) %n2.reverse.schedule.base, i32 3
%n2.reverse.schedule.3 = load i32, ptr addrspace(1) %n2.reverse.schedule.3.ptr, align 4
%n2.reverse.schedule.4.ptr = getelementptr i32, ptr addrspace(1) %n2.reverse.schedule.base, i32 4
%n2.reverse.schedule.4 = load i32, ptr addrspace(1) %n2.reverse.schedule.4.ptr, align 4
%n2.reverse.schedule.5.ptr = getelementptr i32, ptr addrspace(1) %n2.reverse.schedule.base, i32 5
%n2.reverse.schedule.5 = load i32, ptr addrspace(1) %n2.reverse.schedule.5.ptr, align 4
%n2.reverse.schedule.6.ptr = getelementptr i32, ptr addrspace(1) %n2.reverse.schedule.base, i32 6
%n2.reverse.schedule.6 = load i32, ptr addrspace(1) %n2.reverse.schedule.6.ptr, align 4
%n2.reverse.schedule.7.ptr = getelementptr i32, ptr addrspace(1) %n2.reverse.schedule.base, i32 7
%n2.reverse.schedule.7 = load i32, ptr addrspace(1) %n2.reverse.schedule.7.ptr, align 4
%n2.reverse.schedule.8.ptr = getelementptr i32, ptr addrspace(1) %n2.reverse.schedule.base, i32 8
%n2.reverse.schedule.8 = load i32, ptr addrspace(1) %n2.reverse.schedule.8.ptr, align 4
call void @contraction_reverse_body( ptr addrspace(1) %n2.source, ptr addrspace(1) %n2.weights, ptr addrspace(1) %n2.value, ptr addrspace(1) %n2.delta, ptr addrspace(1) %n2.source.adjoint, ptr addrspace(1) %gradient, i1 false, i1 true, i1 false, i1 false, i32 %rows, i32 8, i32 1, i32 1, i32 1, i32 0, i32 72, i32 %n2.reverse.schedule.3, i32 %n2.reverse.schedule.4, i32 %n2.reverse.schedule.5, i32 %n2.reverse.schedule.6, i32 %n2.reverse.schedule.7, i32 %n2.reverse.schedule.8, i32 %threads )
call void @contraction_forward_body( ptr addrspace(1) %n2.delta, ptr addrspace(1) %n2.weights, ptr addrspace(1) %n2.source.adjoint, ptr addrspace(1) %n2.value, i32 %rows, i32 1, i32 1, i32 8, i32 1, i32 0, i1 false, i1 false, i1 true, i1 true, i1 false, i32 %n2.reverse.schedule.6, i32 %n2.reverse.schedule.7, i32 %n2.reverse.schedule.8, i32 %threads )
call void @grid_barrier(i32 %threads)%n1.source = getelementptr i8, ptr addrspace(1) %values, i64 0
%n1.value = getelementptr i8, ptr addrspace(1) %values, i64 2048
%n1.context = getelementptr i8, ptr addrspace(1) %contexts, i64 40
%n1.delta = getelementptr i8, ptr addrspace(1) %adjoints, i64 2048
%n1.weights = getelementptr i8, ptr addrspace(1) %weights, i64 288
%n1.source.adjoint = getelementptr i8, ptr addrspace(1) %adjoints, i64 0
br label %n1.scalar.reverse.entry
n1.scalar.reverse.entry:
%n1.scalar.reverse.count = mul i32 %rows, 8
br label %n1.scalar.reverse.loop
n1.scalar.reverse.loop:
%n1.scalar.reverse.p = phi i32 [ %tid, %n1.scalar.reverse.entry ], [ %n1.scalar.reverse.next, %n1.scalar.reverse.step ]
%n1.scalar.reverse.more = icmp ult i32 %n1.scalar.reverse.p, %n1.scalar.reverse.count
br i1 %n1.scalar.reverse.more, label %n1.scalar.reverse.body, label %n1.scalar.reverse.done
n1.scalar.reverse.body:
%n1.scalar.reverse.first.ptr = getelementptr inbounds float, ptr addrspace(1) %n1.source, i32 %n1.scalar.reverse.p
%n1.scalar.reverse.first = load float, ptr addrspace(1) %n1.scalar.reverse.first.ptr, align 4
%n1.scalar.reverse.incoming.ptr = getelementptr inbounds float, ptr addrspace(1) %n1.delta, i32 %n1.scalar.reverse.p
%n1.scalar.reverse.incoming = load float, ptr addrspace(1) %n1.scalar.reverse.incoming.ptr, align 4
%n1.scalar.reverse.scalar.2.condition = call i1 @recipe.ogt(float %n1.scalar.reverse.first, float 0x0000000000000000)
%n1.scalar.reverse.scalar.2 = call float @recipe.from.u1(i1 %n1.scalar.reverse.scalar.2.condition)
%n1.scalar.reverse.scalar.3 = call float @recipe.tanh(float %n1.scalar.reverse.first)
%n1.scalar.reverse.tanh.square.0 = call float @recipe.mul(float %n1.scalar.reverse.scalar.3, float %n1.scalar.reverse.scalar.3)
%n1.scalar.reverse.tanh.base.1 = call float @recipe.sub(float 0x3FF0000000000000, float %n1.scalar.reverse.tanh.square.0)
%n1.scalar.reverse.tanh.2 = call float @recipe.mul(float %n1.scalar.reverse.incoming, float %n1.scalar.reverse.tanh.base.1)
%n1.scalar.reverse.adjoint.3 = call float @recipe.add(float 0x0000000000000000, float %n1.scalar.reverse.tanh.2)
%n1.scalar.reverse.first.adjoint.ptr = getelementptr inbounds float, ptr addrspace(1) %n1.source.adjoint, i32 %n1.scalar.reverse.p
%n1.scalar.reverse.combined = call float @recipe.add(float %n1.scalar.reverse.adjoint.3, float 0x0000000000000000)
%n1.scalar.reverse.combined.owned.prior = load float, ptr addrspace(1) %n1.scalar.reverse.first.adjoint.ptr, align 4
%n1.scalar.reverse.combined.owned.sum = call float @recipe.add(float %n1.scalar.reverse.combined.owned.prior, float %n1.scalar.reverse.combined)
store float %n1.scalar.reverse.combined.owned.sum, ptr addrspace(1) %n1.scalar.reverse.first.adjoint.ptr, align 4
br label %n1.scalar.reverse.step
n1.scalar.reverse.step:
%n1.scalar.reverse.next = add i32 %n1.scalar.reverse.p, %threads
br label %n1.scalar.reverse.loop
n1.scalar.reverse.done:
call void @grid_barrier(i32 %threads)%n0.value = getelementptr i8, ptr addrspace(1) %values, i64 0
%n0.context = getelementptr i8, ptr addrspace(1) %contexts, i64 0
%n0.delta = getelementptr i8, ptr addrspace(1) %adjoints, i64 0
%n0.weights = getelementptr i8, ptr addrspace(1) %weights, i64 0
%n0.reverse.schedule.base = getelementptr i8, ptr addrspace(1) %contexts, i64 88
%n0.reverse.schedule.3.ptr = getelementptr i32, ptr addrspace(1) %n0.reverse.schedule.base, i32 3
%n0.reverse.schedule.3 = load i32, ptr addrspace(1) %n0.reverse.schedule.3.ptr, align 4
%n0.reverse.schedule.4.ptr = getelementptr i32, ptr addrspace(1) %n0.reverse.schedule.base, i32 4
%n0.reverse.schedule.4 = load i32, ptr addrspace(1) %n0.reverse.schedule.4.ptr, align 4
%n0.reverse.schedule.5.ptr = getelementptr i32, ptr addrspace(1) %n0.reverse.schedule.base, i32 5
%n0.reverse.schedule.5 = load i32, ptr addrspace(1) %n0.reverse.schedule.5.ptr, align 4
%n0.reverse.schedule.6.ptr = getelementptr i32, ptr addrspace(1) %n0.reverse.schedule.base, i32 6
%n0.reverse.schedule.6 = load i32, ptr addrspace(1) %n0.reverse.schedule.6.ptr, align 4
%n0.reverse.schedule.7.ptr = getelementptr i32, ptr addrspace(1) %n0.reverse.schedule.base, i32 7
%n0.reverse.schedule.7 = load i32, ptr addrspace(1) %n0.reverse.schedule.7.ptr, align 4
%n0.reverse.schedule.8.ptr = getelementptr i32, ptr addrspace(1) %n0.reverse.schedule.base, i32 8
%n0.reverse.schedule.8 = load i32, ptr addrspace(1) %n0.reverse.schedule.8.ptr, align 4
call void @contraction_reverse_body( ptr addrspace(1) %samples, ptr addrspace(1) %n0.weights, ptr addrspace(1) %n0.value, ptr addrspace(1) %n0.delta, ptr addrspace(1) %input_adjoint, ptr addrspace(1) %gradient, i1 false, i1 true, i1 false, i1 false, i32 %rows, i32 8, i32 1, i32 8, i32 1, i32 0, i32 0, i32 %n0.reverse.schedule.3, i32 %n0.reverse.schedule.4, i32 %n0.reverse.schedule.5, i32 %n0.reverse.schedule.6, i32 %n0.reverse.schedule.7, i32 %n0.reverse.schedule.8, i32 %threads )
call void @contraction_forward_body( ptr addrspace(1) %n0.delta, ptr addrspace(1) %n0.weights, ptr addrspace(1) %input_adjoint, ptr addrspace(1) %n0.value, i32 %rows, i32 8, i32 1, i32 8, i32 1, i32 0, i1 false, i1 false, i1 true, i1 true, i1 false, i32 %n0.reverse.schedule.6, i32 %n0.reverse.schedule.7, i32 %n0.reverse.schedule.8, i32 %threads )
call void @grid_barrier(i32 %threads)br i1 %epoch.optimizer, label %optimizer.entry, label %epoch.done
optimizer.entry:
%optimizer.base = add i32 0, %tid
br label %optimizer.loop
optimizer.loop:
%optimizer.p = phi i32 [ %optimizer.base, %optimizer.entry ], [ %optimizer.next, %optimizer.advance ]
%optimizer.more = icmp ult i32 %optimizer.p, 81
br i1 %optimizer.more, label %optimizer.step, label %optimizer.done
optimizer.step:
%optimizer.frozen.ptr = getelementptr i8, ptr addrspace(1) %frozen, i32 %optimizer.p
%optimizer.gradient.ptr = getelementptr float, ptr addrspace(1) %gradient, i32 %optimizer.p
%optimizer.moment.ptr = getelementptr float, ptr addrspace(1) %moments, i32 %optimizer.p
%optimizer.variance.ptr = getelementptr float, ptr addrspace(1) %variances, i32 %optimizer.p
%optimizer.weight.ptr = getelementptr float, ptr addrspace(1) %weights, i32 %optimizer.p
%optimizer.frozen.value = load i8, ptr addrspace(1) %optimizer.frozen.ptr, align 1
%optimizer.is.frozen = icmp ne i8 %optimizer.frozen.value, 0
br i1 %optimizer.is.frozen, label %optimizer.advance, label %optimizer.update
optimizer.update:
%optimizer.gradient.model = load float, ptr addrspace(1) %optimizer.gradient.ptr, align 4
%optimizer.gradient.value = call float @recipe.state.from.model(float %optimizer.gradient.model)
%optimizer.moment.old = load float, ptr addrspace(1) %optimizer.moment.ptr, align 4
%optimizer.variance.old = load float, ptr addrspace(1) %optimizer.variance.ptr, align 4
%optimizer.weight.model = load float, ptr addrspace(1) %optimizer.weight.ptr, align 4
%optimizer.weight.value = call float @recipe.state.from.model(float %optimizer.weight.model)
%optimizer.one.beta1 = call float @recipe.state.sub(float 0x3FF0000000000000, float %beta1)
%optimizer.one.beta2 = call float @recipe.state.sub(float 0x3FF0000000000000, float %beta2)
%optimizer.moment.part = call float @recipe.state.mul(float %beta1, float %optimizer.moment.old)
%optimizer.gradient.part = call float @recipe.state.mul(float %optimizer.one.beta1, float %optimizer.gradient.value)
%optimizer.moment.new = call float @recipe.state.add(float %optimizer.moment.part, float %optimizer.gradient.part)
%optimizer.gradient.square = call float @recipe.state.mul(float %optimizer.gradient.value, float %optimizer.gradient.value)
%optimizer.variance.part = call float @recipe.state.mul(float %beta2, float %optimizer.variance.old)
%optimizer.gradient.variance = call float @recipe.state.mul(float %optimizer.one.beta2, float %optimizer.gradient.square)
%optimizer.variance.new = call float @recipe.state.add(float %optimizer.variance.part, float %optimizer.gradient.variance)
store float %optimizer.moment.new, ptr addrspace(1) %optimizer.moment.ptr, align 4
store float %optimizer.variance.new, ptr addrspace(1) %optimizer.variance.ptr, align 4
%optimizer.m.correct = call float @recipe.state.sub(float 0x3FF0000000000000, float %beta1.power)
%optimizer.v.correct = call float @recipe.state.sub(float 0x3FF0000000000000, float %beta2.power)
%optimizer.m.hat = call float @recipe.state.div(float %optimizer.moment.new, float %optimizer.m.correct)
%optimizer.v.hat = call float @recipe.state.div(float %optimizer.variance.new, float %optimizer.v.correct)
%optimizer.root = call float @recipe.state.sqrt(float %optimizer.v.hat)
%optimizer.denominator = call float @recipe.state.add(float %optimizer.root, float %epsilon)
%optimizer.direction = call float @recipe.state.div(float %optimizer.m.hat, float %optimizer.denominator)
%optimizer.decay = call float @recipe.state.mul(float %decay, float %optimizer.weight.value)
%optimizer.total = call float @recipe.state.add(float %optimizer.direction, float %optimizer.decay)
%optimizer.change = call float @recipe.state.mul(float %rate, float %optimizer.total)
%optimizer.next.state = call float @recipe.state.sub(float %optimizer.weight.value, float %optimizer.change)
%optimizer.next.weight = call float @recipe.model.from.state(float %optimizer.next.state)
store float %optimizer.next.weight, ptr addrspace(1) %optimizer.weight.ptr, align 4
br label %optimizer.advance
optimizer.advance:
%optimizer.next = add i32 %optimizer.p, %threads
br label %optimizer.loop
optimizer.done:
br label %epoch.done
epoch.done:
ret void
}
